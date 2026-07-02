//! Offline PLDA diarization refinement.
//!
//! This module implements license-clean PLDA-based diarization refinement over
//! OpenASR's own WeSpeaker embeddings. The dense resegmentation default is an
//! honest PLDA mixture update, while a separate HMM VBx variant runs log-domain
//! forward-backward over adjacent dense windows when explicitly selected. Both
//! use the CC-BY-4.0 PLDA/LDA parameters distributed with the public
//! `pyannote-community/speaker-diarization-community-1` bundle, converted to an
//! OpenASR-owned raw f32 asset. No BUT VBx source code is vendored or copied.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::clustering::ClusterContext;
use super::contract::{SpeakerEmbedding, SpeakerId, TimeRange};
use super::embed::SpeakerEmbedder;

const ASSET: &[u8] = include_bytes!("assets/community1_plda_f32.bin");
const ASSET_MAGIC: &[u8; 8] = b"OASRPLD2";
const INPUT_DIM: usize = 256;
const PLDA_DIM: usize = 128;
const DENSE_MIN_EMBEDDINGS: usize = 30;
const DENSE_MIN_INITIAL_SPEAKERS: usize = 9;
const MIN_REFINED_SPEAKERS: usize = 2;
const PLDA_MERGE_THRESHOLD: f32 = 150.0;
const DENSE_WINDOW_S: f64 = 1.5;
const DENSE_SHIFT_S: f64 = 0.25;
const DENSE_MIN_SPEECH_OVERLAP_S: f64 = 0.75;
const DENSE_VBX_INIT_SMOOTHING: f32 = 7.0;
const DENSE_VBX_MAX_ITERS: usize = 20;
const DENSE_VBX_FA: f32 = 0.1;
const DENSE_VBX_FB: f32 = 5.0;
const DENSE_VBX_LOOP_PROB: f32 = 0.95;
const DENSE_VBX_CONVERGENCE: f32 = 1.0e-5;
const DENSE_WINDOW_CACHE_SCHEMA: u32 = 2;
const DENSE_VBX_VARIANT_ENV: &str = "OPENASR_DIAR_DENSE_VBX_VARIANT";
const DENSE_VBX_HMM_ENV: &str = "OPENASR_DIAR_DENSE_VBX_HMM";

pub(crate) fn refine_labels(
    embeddings: &[SpeakerEmbedding],
    context: &[ClusterContext],
    initial: &[SpeakerId],
) -> Option<Vec<SpeakerId>> {
    if embeddings.len() != context.len() || embeddings.len() != initial.len() {
        return None;
    }
    if !vbx_enabled() || !should_refine(embeddings, context, initial) {
        return None;
    }

    let model = CommunityPlda::from_asset()?;
    let transformed: Vec<Vec<f32>> = embeddings
        .iter()
        .map(|embedding| model.transform(&embedding.0))
        .collect::<Option<_>>()?;
    let mut labels = compact_time_order_labels(initial, context);
    labels = merge_close_states(&transformed, context, &labels, &model);
    let labels = compact_time_order_labels(&labels, context);
    (speaker_count(&labels) >= MIN_REFINED_SPEAKERS).then_some(labels)
}

pub(crate) fn refine_dense_labels(
    samples: &[f32],
    sample_rate_hz: u32,
    embedder: &dyn SpeakerEmbedder,
    context: &[ClusterContext],
    initial: &[SpeakerId],
) -> Option<Vec<SpeakerId>> {
    if !vbx_enabled() || !dense_vbx_enabled() || !should_refine_dense(embedder, context, initial) {
        return None;
    }
    let speech_mask = speech_mask_from_context(context)?;
    let states = ordered_states(initial);
    if states.len() < MIN_REFINED_SPEAKERS {
        return None;
    }

    let model = CommunityPlda::from_asset()?;
    let windows = dense_windows(
        samples,
        sample_rate_hz,
        embedder,
        context,
        initial,
        &speech_mask,
        &model,
    )?;
    if windows.len() < DENSE_MIN_EMBEDDINGS {
        return None;
    }
    let state_index: BTreeMap<SpeakerId, usize> = states
        .iter()
        .copied()
        .enumerate()
        .map(|(index, state)| (state, index))
        .collect();
    let window_initial: Vec<usize> = windows
        .iter()
        .map(|window| state_index.get(&window.initial).copied().unwrap_or(0))
        .collect();
    let features: Vec<Vec<f32>> = windows
        .iter()
        .map(|window| window.embedding.clone())
        .collect();
    let params = DenseVbxParams::from_env();
    let responsibilities = match DenseVbxVariant::from_env() {
        DenseVbxVariant::PldaMixture => {
            plda_mixture_responsibilities(&features, &window_initial, states.len(), &model, params)
        }
        DenseVbxVariant::Hmm => {
            hmm_vbx_responsibilities(&features, &window_initial, states.len(), &model, params)
        }
    }?;
    let labels = assign_regions_from_dense(context, initial, &windows, &responsibilities, &states);
    let labels = compact_time_order_labels(&labels, context);
    (speaker_count(&labels) >= MIN_REFINED_SPEAKERS).then_some(labels)
}

fn vbx_enabled() -> bool {
    std::env::var("OPENASR_DIAR_VBX")
        .map(|value| {
            let value = value.trim().to_ascii_lowercase();
            !(value == "0" || value == "false" || value == "off")
        })
        .unwrap_or(true)
}

fn dense_vbx_enabled() -> bool {
    std::env::var("OPENASR_DIAR_DENSE_VBX")
        .map(|value| {
            let value = value.trim().to_ascii_lowercase();
            !(value == "0" || value == "false" || value == "off")
        })
        .unwrap_or(true)
}

fn plda_merge_threshold() -> f32 {
    std::env::var("OPENASR_DIAR_VBX_MERGE_THRESHOLD")
        .ok()
        .and_then(|value| value.trim().parse::<f32>().ok())
        .filter(|value| value.is_finite())
        .unwrap_or(PLDA_MERGE_THRESHOLD)
}

fn should_refine(
    embeddings: &[SpeakerEmbedding],
    context: &[ClusterContext],
    initial: &[SpeakerId],
) -> bool {
    embeddings
        .first()
        .is_some_and(|embedding| embedding.dim() == INPUT_DIM)
        && embeddings.len() >= DENSE_MIN_EMBEDDINGS
        && speaker_count(initial) >= DENSE_MIN_INITIAL_SPEAKERS
        && context
            .iter()
            .any(|item| item.local_speaker.is_some() || item.overlap)
}

fn should_refine_dense(
    embedder: &dyn SpeakerEmbedder,
    context: &[ClusterContext],
    initial: &[SpeakerId],
) -> bool {
    embedder.embedding_dim() == INPUT_DIM
        && context.len() == initial.len()
        && context.len() >= DENSE_MIN_EMBEDDINGS
        && speaker_count(initial) >= DENSE_MIN_INITIAL_SPEAKERS
        && context
            .iter()
            .any(|item| item.local_speaker.is_some() || item.overlap)
}

#[derive(Clone)]
struct CommunityPlda {
    mean1: Vec<f32>,
    mean2: Vec<f32>,
    lda: Vec<f32>,
    mu: Vec<f32>,
    tr: Vec<f32>,
    psi: Vec<f32>,
    psi_weight: Vec<f32>,
}

impl CommunityPlda {
    fn from_asset() -> Option<Self> {
        let mut offset = 0usize;
        let magic = read_exact(ASSET, &mut offset, ASSET_MAGIC.len())?;
        if magic != ASSET_MAGIC {
            return None;
        }
        let input_dim = read_u32(ASSET, &mut offset)? as usize;
        let plda_dim = read_u32(ASSET, &mut offset)? as usize;
        if input_dim != INPUT_DIM || plda_dim != PLDA_DIM {
            return None;
        }
        let mean1 = read_f32_vec(ASSET, &mut offset, INPUT_DIM)?;
        let mean2 = read_f32_vec(ASSET, &mut offset, PLDA_DIM)?;
        let lda = read_f32_vec(ASSET, &mut offset, INPUT_DIM * PLDA_DIM)?;
        let mu = read_f32_vec(ASSET, &mut offset, PLDA_DIM)?;
        let tr = read_f32_vec(ASSET, &mut offset, PLDA_DIM * PLDA_DIM)?;
        let psi = read_f32_vec(ASSET, &mut offset, PLDA_DIM)?;
        if offset != ASSET.len() {
            return None;
        }
        let psi_weight = psi.iter().map(|value| value / (1.0 + value)).collect();
        Some(Self {
            mean1,
            mean2,
            lda,
            mu,
            tr,
            psi,
            psi_weight,
        })
    }

    fn transform(&self, embedding: &[f32]) -> Option<Vec<f32>> {
        if embedding.len() != INPUT_DIM {
            return None;
        }
        let mut centered: Vec<f32> = embedding
            .iter()
            .zip(&self.mean1)
            .map(|(&value, &mean)| value - mean)
            .collect();
        l2_normalize(&mut centered);
        let input_scale = (INPUT_DIM as f32).sqrt();

        let mut projected = vec![0.0f32; PLDA_DIM];
        for (input, &value) in centered.iter().enumerate() {
            let value = value * input_scale;
            for (output, acc) in projected.iter_mut().enumerate() {
                *acc += value * self.lda[input * PLDA_DIM + output];
            }
        }
        for (value, mean) in projected.iter_mut().zip(&self.mean2) {
            *value -= *mean;
        }
        l2_normalize(&mut projected);
        let plda_input_scale = (PLDA_DIM as f32).sqrt();
        for value in &mut projected {
            *value *= plda_input_scale;
        }

        let mut plda = vec![0.0f32; PLDA_DIM];
        for (row, output) in plda.iter_mut().enumerate() {
            let mut value = 0.0f32;
            for (col, (&input, &mean)) in projected.iter().zip(&self.mu).enumerate() {
                value += self.tr[row * PLDA_DIM + col] * (input - mean);
            }
            *output = value;
        }
        Some(plda)
    }

    fn score(&self, left: &[f32], right: &[f32]) -> f32 {
        left.iter()
            .zip(right)
            .zip(&self.psi_weight)
            .map(|((&l, &r), &weight)| l * r * weight)
            .sum()
    }
}

fn merge_close_states(
    embeddings: &[Vec<f32>],
    context: &[ClusterContext],
    initial: &[SpeakerId],
    model: &CommunityPlda,
) -> Vec<SpeakerId> {
    let mut labels = initial.to_vec();
    loop {
        let means = state_means(embeddings, &labels);
        if means.len() <= MIN_REFINED_SPEAKERS {
            break;
        }
        let mut best_pair = None;
        let mut best_score = f32::NEG_INFINITY;
        for left in 0..means.len() {
            for right in (left + 1)..means.len() {
                let left_id = means[left].0;
                let right_id = means[right].0;
                if states_have_cannot_link(context, &labels, left_id, right_id) {
                    continue;
                }
                let score = model.score(&means[left].1, &means[right].1);
                if score > best_score {
                    best_score = score;
                    best_pair = Some((left_id, right_id));
                }
            }
        }
        if best_score < plda_merge_threshold() {
            break;
        }
        let Some((keep, merge)) = best_pair else {
            break;
        };
        for label in &mut labels {
            if *label == merge {
                *label = keep;
            }
        }
        labels = compact_arrival_labels(&labels);
    }
    labels
}

#[derive(Clone)]
struct DenseWindow {
    range: TimeRange,
    initial: SpeakerId,
    embedding: Vec<f32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DenseVbxVariant {
    PldaMixture,
    Hmm,
}

impl DenseVbxVariant {
    fn from_env() -> Self {
        if env_bool_enabled(DENSE_VBX_HMM_ENV).unwrap_or(false) {
            return Self::Hmm;
        }
        let Ok(value) = std::env::var(DENSE_VBX_VARIANT_ENV) else {
            return Self::PldaMixture;
        };
        match value.trim().to_ascii_lowercase().as_str() {
            "hmm" | "hmm-vbx" | "vbx-hmm" => Self::Hmm,
            "mixture"
            | "plda-mixture"
            | "plda_mixture"
            | "plda-mixture-resegmentation"
            | "plda_mixture_resegmentation" => Self::PldaMixture,
            _ => Self::PldaMixture,
        }
    }
}

#[derive(Clone, Copy)]
struct DenseVbxParams {
    fa: f32,
    fb: f32,
    max_iters: usize,
    init_smoothing: f32,
    loop_prob: f32,
}

impl DenseVbxParams {
    fn from_env() -> Self {
        Self {
            fa: env_f32("OPENASR_DIAR_DENSE_VBX_FA", DENSE_VBX_FA).max(f32::EPSILON),
            fb: env_f32("OPENASR_DIAR_DENSE_VBX_FB", DENSE_VBX_FB).max(f32::EPSILON),
            max_iters: env_usize("OPENASR_DIAR_DENSE_VBX_MAX_ITERS", DENSE_VBX_MAX_ITERS).max(1),
            init_smoothing: env_f32(
                "OPENASR_DIAR_DENSE_VBX_INIT_SMOOTHING",
                DENSE_VBX_INIT_SMOOTHING,
            ),
            loop_prob: env_f32("OPENASR_DIAR_DENSE_VBX_LOOP_PROB", DENSE_VBX_LOOP_PROB)
                .clamp(1.0e-6, 1.0 - 1.0e-6),
        }
    }
}

fn env_bool_enabled(name: &str) -> Option<bool> {
    std::env::var(name).ok().map(|value| {
        let value = value.trim().to_ascii_lowercase();
        !(value == "0" || value == "false" || value == "off")
    })
}

fn env_f32(name: &str, default: f32) -> f32 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.trim().parse::<f32>().ok())
        .filter(|value| value.is_finite())
        .unwrap_or(default)
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(default)
}

fn dense_windows(
    samples: &[f32],
    sample_rate_hz: u32,
    embedder: &dyn SpeakerEmbedder,
    context: &[ClusterContext],
    initial: &[SpeakerId],
    speech_mask: &[TimeRange],
    model: &CommunityPlda,
) -> Option<Vec<DenseWindow>> {
    let cache = dense_window_cache_path().and_then(|path| {
        dense_window_cache_metadata(samples, sample_rate_hz, embedder)
            .map(|metadata| DenseWindowCacheTarget { path, metadata })
    });
    if let Some(cache) = cache.as_ref()
        && let Some(mut windows) = read_dense_window_cache(&cache.path, &cache.metadata)
    {
        for window in &mut windows {
            window.initial = dominant_label_for_range(&window.range, context, initial)?;
        }
        return Some(windows);
    }

    let last_end = speech_mask
        .iter()
        .map(|range| range.end_s)
        .max_by(|left, right| left.total_cmp(right))?;
    let duration_s = samples.len() as f64 / sample_rate_hz as f64;
    if last_end <= 0.0 || duration_s < DENSE_WINDOW_S {
        return None;
    }
    let max_start = (duration_s.min(last_end) - DENSE_WINDOW_S).max(0.0);
    let mut windows = Vec::new();
    let mut start_s = 0.0f64;
    while start_s <= max_start + 1.0e-9 {
        let range = TimeRange::new(start_s, start_s + DENSE_WINDOW_S);
        if speech_overlap_s(&range, speech_mask) >= DENSE_MIN_SPEECH_OVERLAP_S
            && let Some(label) = dominant_label_for_range(&range, context, initial)
        {
            let start = (range.start_s * sample_rate_hz as f64).max(0.0) as usize;
            let end = ((range.end_s * sample_rate_hz as f64) as usize).min(samples.len());
            if end > start
                && let Ok(embedding) = embedder.embed(&samples[start..end], sample_rate_hz)
                && embedding.dim() == INPUT_DIM
                && let Some(transformed) = model.transform(&embedding.0)
            {
                windows.push(DenseWindow {
                    range,
                    initial: label,
                    embedding: transformed,
                });
            }
        }
        start_s += DENSE_SHIFT_S;
    }
    if let Some(cache) = cache.as_ref() {
        write_dense_window_cache(&cache.path, &cache.metadata, &windows);
    }
    Some(windows)
}

fn dense_window_cache_path() -> Option<PathBuf> {
    std::env::var_os("OPENASR_DIAR_DENSE_VBX_CACHE").map(PathBuf::from)
}

struct DenseWindowCacheTarget {
    path: PathBuf,
    metadata: DenseWindowCacheMetadata,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct DenseWindowCacheMetadata {
    schema_version: u32,
    audio_sha256: String,
    sample_rate_hz: u32,
    embedder_fingerprint: String,
    embedder_dim: usize,
    window_s: f64,
    shift_s: f64,
    min_speech_overlap_s: f64,
    plda_asset_id: String,
}

#[derive(Serialize, Deserialize)]
struct DenseWindowCacheFile {
    metadata: DenseWindowCacheMetadata,
    windows: Vec<DenseWindowCacheRow>,
}

#[derive(Serialize, Deserialize)]
struct DenseWindowCacheRow {
    start_s: f64,
    end_s: f64,
    initial: u32,
    embedding: Vec<f32>,
}

fn dense_window_cache_metadata(
    samples: &[f32],
    sample_rate_hz: u32,
    embedder: &dyn SpeakerEmbedder,
) -> Option<DenseWindowCacheMetadata> {
    let identity = super::embed::shared_embedder_identity()?;
    if identity.embedding_dim != embedder.embedding_dim() {
        return None;
    }
    Some(DenseWindowCacheMetadata {
        schema_version: DENSE_WINDOW_CACHE_SCHEMA,
        audio_sha256: audio_sha256(samples),
        sample_rate_hz,
        embedder_fingerprint: identity.pack_fingerprint.clone(),
        embedder_dim: identity.embedding_dim,
        window_s: DENSE_WINDOW_S,
        shift_s: DENSE_SHIFT_S,
        min_speech_overlap_s: DENSE_MIN_SPEECH_OVERLAP_S,
        plda_asset_id: asset_sha256(ASSET),
    })
}

fn audio_sha256(samples: &[f32]) -> String {
    let mut hasher = Sha256::new();
    for sample in samples {
        hasher.update(sample.to_le_bytes());
    }
    format!("sha256:{:x}", hasher.finalize())
}

fn asset_sha256(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("sha256:{:x}", hasher.finalize())
}

fn read_dense_window_cache(
    path: &Path,
    expected_metadata: &DenseWindowCacheMetadata,
) -> Option<Vec<DenseWindow>> {
    let bytes = std::fs::read(path).ok()?;
    let root: DenseWindowCacheFile = serde_json::from_slice(&bytes).ok()?;
    if &root.metadata != expected_metadata {
        return None;
    }
    let mut parsed = Vec::with_capacity(root.windows.len());
    for window in root.windows {
        if window.embedding.len() != PLDA_DIM {
            return None;
        }
        parsed.push(DenseWindow {
            range: TimeRange::new(window.start_s, window.end_s),
            initial: SpeakerId(window.initial),
            embedding: window.embedding,
        });
    }
    Some(parsed)
}

fn write_dense_window_cache(
    path: &Path,
    metadata: &DenseWindowCacheMetadata,
    windows: &[DenseWindow],
) {
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let rows: Vec<_> = windows
        .iter()
        .map(|window| DenseWindowCacheRow {
            start_s: window.range.start_s,
            end_s: window.range.end_s,
            initial: window.initial.0,
            embedding: window.embedding.clone(),
        })
        .collect();
    let payload = DenseWindowCacheFile {
        metadata: metadata.clone(),
        windows: rows,
    };
    let _ = std::fs::write(path, serde_json::to_vec(&payload).unwrap_or_default());
}

fn speech_mask_from_context(context: &[ClusterContext]) -> Option<Vec<TimeRange>> {
    let mut ranges: Vec<TimeRange> = context
        .iter()
        .map(|item| item.range)
        .filter(|range| range.duration_s() > 0.0)
        .collect();
    if ranges.is_empty() {
        return None;
    }
    ranges.sort_by(|left, right| left.start_s.total_cmp(&right.start_s));
    let mut merged: Vec<TimeRange> = Vec::new();
    for range in ranges {
        if let Some(last) = merged.last_mut()
            && range.start_s <= last.end_s
        {
            last.end_s = last.end_s.max(range.end_s);
            continue;
        }
        merged.push(range);
    }
    Some(merged)
}

fn speech_overlap_s(range: &TimeRange, mask: &[TimeRange]) -> f64 {
    mask.iter()
        .map(|speech| range.intersection_s(speech))
        .sum::<f64>()
}

fn dominant_label_for_range(
    range: &TimeRange,
    context: &[ClusterContext],
    labels: &[SpeakerId],
) -> Option<SpeakerId> {
    let mut scores: BTreeMap<SpeakerId, f64> = BTreeMap::new();
    for (item, label) in context.iter().zip(labels) {
        let overlap = range.intersection_s(&item.range);
        if overlap > 0.0 {
            *scores.entry(*label).or_default() += overlap;
        }
    }
    scores
        .into_iter()
        .max_by(|left, right| {
            left.1
                .total_cmp(&right.1)
                .then_with(|| right.0.cmp(&left.0))
        })
        .map(|(label, _)| label)
}

fn ordered_states(labels: &[SpeakerId]) -> Vec<SpeakerId> {
    labels
        .iter()
        .copied()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

struct PldaFeatureStats {
    rho: Vec<Vec<f32>>,
    gaussian_terms: Vec<f32>,
}

fn plda_feature_stats(features: &[Vec<f32>], model: &CommunityPlda) -> PldaFeatureStats {
    let v: Vec<f32> = model.psi.iter().map(|value| value.sqrt()).collect();
    let rho: Vec<Vec<f32>> = features
        .iter()
        .map(|feature| feature.iter().zip(&v).map(|(x, scale)| x * scale).collect())
        .collect();
    let gaussian_terms: Vec<f32> = features
        .iter()
        .map(|feature| {
            let norm2 = feature.iter().map(|value| value * value).sum::<f32>();
            -0.5 * (norm2 + PLDA_DIM as f32 * (std::f32::consts::TAU).ln())
        })
        .collect();
    PldaFeatureStats {
        rho,
        gaussian_terms,
    }
}

fn plda_emission_log_probs(
    gamma: &[Vec<f32>],
    stats: &PldaFeatureStats,
    model: &CommunityPlda,
    params: DenseVbxParams,
    state_count: usize,
) -> Vec<Vec<f32>> {
    let phi = &model.psi;
    let ratio = params.fa / params.fb;
    let mut state_mass = vec![0.0f32; state_count];
    for row in gamma {
        for (mass, value) in state_mass.iter_mut().zip(row) {
            *mass += *value;
        }
    }
    let mut inv_l = vec![vec![0.0f32; PLDA_DIM]; state_count];
    let mut alpha = vec![vec![0.0f32; PLDA_DIM]; state_count];
    for state in 0..state_count {
        for (dim, phi_dim) in phi.iter().enumerate() {
            inv_l[state][dim] = 1.0 / (1.0 + ratio * state_mass[state] * phi_dim);
        }
    }
    for (frame, row) in gamma.iter().enumerate() {
        for state in 0..state_count {
            let weight = row[state];
            if weight <= 0.0 {
                continue;
            }
            for (dim, alpha_value) in alpha[state].iter_mut().enumerate() {
                *alpha_value += weight * stats.rho[frame][dim];
            }
        }
    }
    for state in 0..state_count {
        for (dim, alpha_value) in alpha[state].iter_mut().enumerate() {
            *alpha_value *= ratio * inv_l[state][dim];
        }
    }

    let mut emissions = vec![vec![0.0f32; state_count]; gamma.len()];
    for (frame, row) in emissions.iter_mut().enumerate() {
        for state in 0..state_count {
            let mut rho_alpha = 0.0f32;
            let mut precision_term = 0.0f32;
            for (dim, phi_dim) in phi.iter().enumerate() {
                rho_alpha += stats.rho[frame][dim] * alpha[state][dim];
                precision_term +=
                    (inv_l[state][dim] + alpha[state][dim] * alpha[state][dim]) * phi_dim;
            }
            row[state] =
                params.fa * (rho_alpha - 0.5 * precision_term + stats.gaussian_terms[frame]);
        }
    }
    emissions
}

/// Dense PLDA mixture responsibility updates following pyannote.audio's
/// post-2025 VBx utility equations, reimplemented over OpenASR-owned arrays.
fn plda_mixture_responsibilities(
    features: &[Vec<f32>],
    initial: &[usize],
    state_count: usize,
    model: &CommunityPlda,
    params: DenseVbxParams,
) -> Option<Vec<Vec<f32>>> {
    if features.is_empty() || features.len() != initial.len() || state_count == 0 {
        return None;
    }
    if features.iter().any(|feature| feature.len() != PLDA_DIM) {
        return None;
    }

    let mut gamma = initial_responsibilities(initial, state_count, params.init_smoothing);
    let mut pi = vec![1.0f32 / state_count as f32; state_count];
    let stats = plda_feature_stats(features, model);

    for _ in 0..params.max_iters {
        let previous = gamma.clone();
        let emissions = plda_emission_log_probs(&gamma, &stats, model, params, state_count);
        for frame in 0..features.len() {
            let logp: Vec<f32> = emissions[frame]
                .iter()
                .zip(&pi)
                .map(|(&emission, &prior)| emission + prior.max(f32::EPSILON).ln())
                .collect();
            let normalizer = logsumexp(&logp);
            for (slot, value) in gamma[frame].iter_mut().zip(logp) {
                *slot = (value - normalizer).exp();
            }
        }
        pi.fill(0.0);
        for row in &gamma {
            for (mass, value) in pi.iter_mut().zip(row) {
                *mass += *value;
            }
        }
        for mass in &mut pi {
            *mass /= gamma.len() as f32;
        }

        let max_delta = gamma
            .iter()
            .zip(&previous)
            .flat_map(|(current, previous)| current.iter().zip(previous))
            .map(|(current, previous)| (current - previous).abs())
            .fold(0.0f32, f32::max);
        if max_delta < DENSE_VBX_CONVERGENCE {
            break;
        }
    }
    Some(gamma)
}

fn hmm_vbx_responsibilities(
    features: &[Vec<f32>],
    initial: &[usize],
    state_count: usize,
    model: &CommunityPlda,
    params: DenseVbxParams,
) -> Option<Vec<Vec<f32>>> {
    if features.is_empty() || features.len() != initial.len() || state_count == 0 {
        return None;
    }
    if features.iter().any(|feature| feature.len() != PLDA_DIM) {
        return None;
    }

    let mut gamma = initial_responsibilities(initial, state_count, params.init_smoothing);
    let mut pi = vec![1.0f32 / state_count as f32; state_count];
    let stats = plda_feature_stats(features, model);

    for _ in 0..params.max_iters {
        let previous = gamma.clone();
        let emissions = plda_emission_log_probs(&gamma, &stats, model, params, state_count);
        let posterior = hmm_forward_backward(&emissions, &pi, params.loop_prob)?;
        gamma = posterior.gamma;
        pi = hmm_update_priors(
            &posterior.log_forward,
            &posterior.log_backward,
            &emissions,
            posterior.log_likelihood,
            &pi,
            params.loop_prob,
        );

        let max_delta = gamma
            .iter()
            .zip(&previous)
            .flat_map(|(current, previous)| current.iter().zip(previous))
            .map(|(current, previous)| (current - previous).abs())
            .fold(0.0f32, f32::max);
        if max_delta < DENSE_VBX_CONVERGENCE {
            break;
        }
    }
    Some(gamma)
}

struct HmmPosterior {
    gamma: Vec<Vec<f32>>,
    log_forward: Vec<Vec<f32>>,
    log_backward: Vec<Vec<f32>>,
    log_likelihood: f32,
}

fn hmm_forward_backward(
    emissions: &[Vec<f32>],
    pi: &[f32],
    loop_prob: f32,
) -> Option<HmmPosterior> {
    let frame_count = emissions.len();
    let state_count = pi.len();
    if frame_count == 0 || state_count == 0 || emissions.iter().any(|row| row.len() != state_count)
    {
        return None;
    }
    let transitions = hmm_log_transitions(pi, loop_prob);
    let mut log_forward = vec![vec![f32::NEG_INFINITY; state_count]; frame_count];
    for (state, prior) in pi.iter().enumerate() {
        log_forward[0][state] = prior.max(f32::EPSILON).ln() + emissions[0][state];
    }
    for frame in 1..frame_count {
        for state in 0..state_count {
            let incoming: Vec<f32> = (0..state_count)
                .map(|previous| log_forward[frame - 1][previous] + transitions[previous][state])
                .collect();
            log_forward[frame][state] = emissions[frame][state] + logsumexp(&incoming);
        }
    }

    let mut log_backward = vec![vec![0.0f32; state_count]; frame_count];
    for frame in (0..frame_count.saturating_sub(1)).rev() {
        for state in 0..state_count {
            let outgoing: Vec<f32> = (0..state_count)
                .map(|next| {
                    transitions[state][next]
                        + emissions[frame + 1][next]
                        + log_backward[frame + 1][next]
                })
                .collect();
            log_backward[frame][state] = logsumexp(&outgoing);
        }
    }

    let log_likelihood = logsumexp(&log_forward[frame_count - 1]);
    if !log_likelihood.is_finite() {
        return None;
    }
    let mut gamma = vec![vec![0.0f32; state_count]; frame_count];
    for frame in 0..frame_count {
        let mut row_sum = 0.0f32;
        for state in 0..state_count {
            let value =
                (log_forward[frame][state] + log_backward[frame][state] - log_likelihood).exp();
            gamma[frame][state] = value;
            row_sum += value;
        }
        if row_sum > f32::EPSILON {
            for value in &mut gamma[frame] {
                *value /= row_sum;
            }
        }
    }
    Some(HmmPosterior {
        gamma,
        log_forward,
        log_backward,
        log_likelihood,
    })
}

fn hmm_log_transitions(pi: &[f32], loop_prob: f32) -> Vec<Vec<f32>> {
    let loop_prob = loop_prob.clamp(1.0e-6, 1.0 - 1.0e-6);
    let change_prob = 1.0 - loop_prob;
    let state_count = pi.len();
    let mut transitions = vec![vec![0.0f32; state_count]; state_count];
    for (from, row) in transitions.iter_mut().enumerate() {
        let mut row_sum = 0.0f32;
        for (to, prior) in pi.iter().enumerate() {
            let mut probability = change_prob * prior.max(f32::EPSILON);
            if from == to {
                probability += loop_prob;
            }
            row[to] = probability;
            row_sum += probability;
        }
        if row_sum > f32::EPSILON {
            for value in row {
                *value = (*value / row_sum).max(f32::EPSILON).ln();
            }
        }
    }
    transitions
}

fn hmm_update_priors(
    log_forward: &[Vec<f32>],
    log_backward: &[Vec<f32>],
    emissions: &[Vec<f32>],
    log_likelihood: f32,
    pi: &[f32],
    loop_prob: f32,
) -> Vec<f32> {
    let frame_count = emissions.len();
    let state_count = pi.len();
    let mut updated = vec![0.0f32; state_count];
    if frame_count == 0 || state_count == 0 {
        return updated;
    }
    for (state, slot) in updated.iter_mut().enumerate() {
        *slot = (log_forward[0][state] + log_backward[0][state] - log_likelihood).exp();
    }
    let change_prob = 1.0 - loop_prob.clamp(1.0e-6, 1.0 - 1.0e-6);
    for frame in 1..frame_count {
        let prev_any = logsumexp(&log_forward[frame - 1]);
        for state in 0..state_count {
            let target_posterior =
                (prev_any + emissions[frame][state] + log_backward[frame][state] - log_likelihood)
                    .exp();
            updated[state] += change_prob * pi[state].max(f32::EPSILON) * target_posterior;
        }
    }
    normalize_distribution(&mut updated);
    updated
}

fn normalize_distribution(values: &mut [f32]) {
    let sum = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .sum::<f32>();
    if sum > f32::EPSILON {
        for value in values {
            *value /= sum;
        }
    } else if !values.is_empty() {
        values.fill(1.0 / values.len() as f32);
    }
}

fn initial_responsibilities(
    initial: &[usize],
    state_count: usize,
    smoothing: f32,
) -> Vec<Vec<f32>> {
    let assigned_weight = smoothing.exp();
    let denom = assigned_weight + (state_count.saturating_sub(1)) as f32;
    initial
        .iter()
        .map(|&state| {
            let mut row = vec![1.0 / denom; state_count];
            if let Some(slot) = row.get_mut(state) {
                *slot = assigned_weight / denom;
            }
            row
        })
        .collect()
}

fn logsumexp(values: &[f32]) -> f32 {
    let max = values
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, |acc, value| acc.max(value));
    if !max.is_finite() {
        return max;
    }
    max + values
        .iter()
        .map(|value| (value - max).exp())
        .sum::<f32>()
        .ln()
}

fn assign_regions_from_dense(
    context: &[ClusterContext],
    initial: &[SpeakerId],
    windows: &[DenseWindow],
    responsibilities: &[Vec<f32>],
    states: &[SpeakerId],
) -> Vec<SpeakerId> {
    let mut labels: Vec<SpeakerId> = context
        .iter()
        .enumerate()
        .map(|(region_index, item)| {
            let mut scores = vec![0.0f64; states.len()];
            for (window, gamma) in windows.iter().zip(responsibilities) {
                let overlap = item.range.intersection_s(&window.range);
                if overlap > 0.0 {
                    for (score, &responsibility) in scores.iter_mut().zip(gamma) {
                        *score += overlap * responsibility as f64;
                    }
                }
            }
            scores
                .iter()
                .enumerate()
                .max_by(|left, right| {
                    left.1
                        .total_cmp(right.1)
                        .then_with(|| states[right.0].cmp(&states[left.0]))
                })
                .filter(|(_, score)| **score > 0.0)
                .map(|(state, _)| states[state])
                .unwrap_or(initial[region_index])
        })
        .collect();
    restore_cannot_link_labels(context, initial, &mut labels);
    labels
}

fn restore_cannot_link_labels(
    context: &[ClusterContext],
    initial: &[SpeakerId],
    labels: &mut [SpeakerId],
) {
    for left in 0..labels.len() {
        for right in (left + 1)..labels.len() {
            if distinct_local_context_overlap(&context[left], &context[right])
                && labels[left] == labels[right]
                && initial[left] != initial[right]
            {
                labels[left] = initial[left];
                labels[right] = initial[right];
            }
        }
    }
}

fn distinct_local_context_overlap(left: &ClusterContext, right: &ClusterContext) -> bool {
    matches!(
        (left.local_speaker, right.local_speaker),
        (Some(left_speaker), Some(right_speaker))
            if left_speaker != right_speaker && left.range.overlaps(&right.range)
    )
}

fn state_means(embeddings: &[Vec<f32>], labels: &[SpeakerId]) -> Vec<(SpeakerId, Vec<f32>)> {
    let mut sums: BTreeMap<SpeakerId, (Vec<f32>, usize)> = BTreeMap::new();
    for (embedding, label) in embeddings.iter().zip(labels) {
        let entry = sums
            .entry(*label)
            .or_insert_with(|| (vec![0.0; PLDA_DIM], 0));
        for (acc, value) in entry.0.iter_mut().zip(embedding) {
            *acc += *value;
        }
        entry.1 += 1;
    }
    sums.into_iter()
        .map(|(label, (mut sum, count))| {
            if count > 0 {
                let scale = 1.0 / count as f32;
                for value in &mut sum {
                    *value *= scale;
                }
            }
            (label, sum)
        })
        .collect()
}

fn states_have_cannot_link(
    context: &[ClusterContext],
    labels: &[SpeakerId],
    left: SpeakerId,
    right: SpeakerId,
) -> bool {
    labels.iter().enumerate().any(|(left_index, &left_label)| {
        left_label == left
            && labels
                .iter()
                .enumerate()
                .any(|(right_index, &right_label)| {
                    right_label == right
                        && context[left_index]
                            .range
                            .overlaps(&context[right_index].range)
                })
    })
}

fn compact_time_order_labels(labels: &[SpeakerId], context: &[ClusterContext]) -> Vec<SpeakerId> {
    let mut first_start: BTreeMap<SpeakerId, f64> = BTreeMap::new();
    for (label, item) in labels.iter().zip(context) {
        first_start
            .entry(*label)
            .and_modify(|start| *start = start.min(item.range.start_s))
            .or_insert(item.range.start_s);
    }
    let mut order: Vec<_> = first_start.into_iter().collect();
    order.sort_by(|left, right| {
        left.1
            .total_cmp(&right.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    let remap: BTreeMap<SpeakerId, SpeakerId> = order
        .into_iter()
        .enumerate()
        .map(|(index, (old, _))| (old, SpeakerId(index as u32)))
        .collect();
    labels
        .iter()
        .map(|label| remap.get(label).copied().unwrap_or(SpeakerId(0)))
        .collect()
}

fn compact_arrival_labels(labels: &[SpeakerId]) -> Vec<SpeakerId> {
    let mut remap = BTreeMap::new();
    let mut next = 0u32;
    labels
        .iter()
        .map(|label| {
            *remap.entry(*label).or_insert_with(|| {
                let label = SpeakerId(next);
                next += 1;
                label
            })
        })
        .collect()
}

fn speaker_count(labels: &[SpeakerId]) -> usize {
    labels.iter().copied().collect::<BTreeSet<_>>().len()
}

fn l2_normalize(values: &mut [f32]) {
    let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        for value in values {
            *value /= norm;
        }
    }
}

fn read_exact<'a>(bytes: &'a [u8], offset: &mut usize, len: usize) -> Option<&'a [u8]> {
    let end = offset.checked_add(len)?;
    let slice = bytes.get(*offset..end)?;
    *offset = end;
    Some(slice)
}

fn read_u32(bytes: &[u8], offset: &mut usize) -> Option<u32> {
    let slice = read_exact(bytes, offset, 4)?;
    Some(u32::from_le_bytes(slice.try_into().ok()?))
}

fn read_f32_vec(bytes: &[u8], offset: &mut usize, len: usize) -> Option<Vec<f32>> {
    let raw = read_exact(bytes, offset, len.checked_mul(4)?)?;
    raw.chunks_exact(4)
        .map(|chunk| Some(f32::from_le_bytes(chunk.try_into().ok()?)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diarize::contract::TimeRange;

    #[test]
    fn community_plda_asset_loads() {
        let model = CommunityPlda::from_asset().expect("PLDA asset");
        assert_eq!(model.mean1.len(), INPUT_DIM);
        assert_eq!(model.mean2.len(), PLDA_DIM);
        assert_eq!(model.lda.len(), INPUT_DIM * PLDA_DIM);
        assert_eq!(model.mu.len(), PLDA_DIM);
        assert_eq!(model.tr.len(), PLDA_DIM * PLDA_DIM);
        assert_eq!(model.psi.len(), PLDA_DIM);
        assert_eq!(model.psi_weight.len(), PLDA_DIM);
    }

    #[test]
    fn community_plda_transform_matches_pyannote_reference() {
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("assets/community1_plda_parity.json"))
                .expect("parity fixture");
        let inputs: Vec<Vec<f32>> =
            serde_json::from_value(fixture["inputs"].clone()).expect("inputs");
        let expected: Vec<Vec<f32>> =
            serde_json::from_value(fixture["transformed"].clone()).expect("transformed");
        let scores = fixture["scores"].as_array().expect("scores");
        let model = CommunityPlda::from_asset().expect("PLDA asset");

        let actual: Vec<Vec<f32>> = inputs
            .iter()
            .map(|input| model.transform(input).expect("transform"))
            .collect();
        assert_eq!(actual.len(), expected.len());
        let mut max_abs_diff = 0.0f32;
        for (actual_row, expected_row) in actual.iter().zip(&expected) {
            assert_eq!(actual_row.len(), expected_row.len());
            for (&actual_value, &expected_value) in actual_row.iter().zip(expected_row) {
                max_abs_diff = max_abs_diff.max((actual_value - expected_value).abs());
            }
        }
        assert!(
            max_abs_diff <= 2.0e-4,
            "max_abs_transform_diff={max_abs_diff}"
        );

        let mut max_score_diff = 0.0f32;
        for score in scores {
            let left = score["left"].as_u64().expect("left") as usize;
            let right = score["right"].as_u64().expect("right") as usize;
            let expected_score = score["score"].as_f64().expect("score") as f32;
            let actual_score = model.score(&actual[left], &actual[right]);
            max_score_diff = max_score_diff.max((actual_score - expected_score).abs());
        }
        assert!(max_score_diff <= 2.0e-3, "max_score_diff={max_score_diff}");
    }

    #[test]
    fn vbx_gate_is_dense_wespeaker_context_only() {
        let embedding = SpeakerEmbedding(vec![0.0; INPUT_DIM]);
        let context = ClusterContext {
            range: TimeRange::new(0.0, 1.0),
            local_speaker: Some(SpeakerId(0)),
            overlap: false,
        };
        assert!(!should_refine(
            &[embedding.clone(), embedding],
            &[context, context],
            &[SpeakerId(0), SpeakerId(1)]
        ));
    }

    #[test]
    fn compact_labels_follow_time_order() {
        let context = vec![
            ClusterContext {
                range: TimeRange::new(5.0, 6.0),
                local_speaker: None,
                overlap: false,
            },
            ClusterContext {
                range: TimeRange::new(1.0, 2.0),
                local_speaker: None,
                overlap: false,
            },
        ];
        let labels = compact_time_order_labels(&[SpeakerId(9), SpeakerId(4)], &context);
        assert_eq!(labels, vec![SpeakerId(1), SpeakerId(0)]);
    }

    #[test]
    fn dense_initial_responsibilities_are_smoothed_one_hot() {
        let gamma = initial_responsibilities(&[0, 2], 3, DENSE_VBX_INIT_SMOOTHING);
        assert_eq!(gamma.len(), 2);
        for row in &gamma {
            let sum = row.iter().sum::<f32>();
            assert!((sum - 1.0).abs() < 1.0e-6);
        }
        assert!(gamma[0][0] > gamma[0][1]);
        assert!(gamma[1][2] > gamma[1][0]);
    }

    #[test]
    fn dense_vbx_gamma_matches_python_reference_fixture() {
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("assets/dense_vbx_gamma_parity.json"))
                .expect("gamma fixture");
        assert_eq!(
            fixture["dimension"].as_u64().expect("dimension") as usize,
            PLDA_DIM
        );
        let state_count = fixture["state_count"].as_u64().expect("state_count") as usize;
        let initial: Vec<usize> =
            serde_json::from_value(fixture["initial"].clone()).expect("initial");
        let features = sparse_fixture_features(&fixture["sparse_features"]);
        let params = DenseVbxParams {
            fa: fixture["params"]["fa"].as_f64().expect("fa") as f32,
            fb: fixture["params"]["fb"].as_f64().expect("fb") as f32,
            max_iters: fixture["params"]["max_iters"].as_u64().expect("max_iters") as usize,
            init_smoothing: fixture["params"]["init_smoothing"]
                .as_f64()
                .expect("init_smoothing") as f32,
            loop_prob: fixture["params"]["loop_prob"].as_f64().expect("loop_prob") as f32,
        };
        let model = CommunityPlda::from_asset().expect("PLDA asset");
        let mixture_expected: Vec<Vec<f32>> =
            serde_json::from_value(fixture["mixture_gamma"].clone()).expect("mixture gamma");
        let hmm_expected: Vec<Vec<f32>> =
            serde_json::from_value(fixture["hmm_gamma"].clone()).expect("hmm gamma");

        let mixture =
            plda_mixture_responsibilities(&features, &initial, state_count, &model, params)
                .expect("mixture responsibilities");
        let hmm = hmm_vbx_responsibilities(&features, &initial, state_count, &model, params)
            .expect("hmm responsibilities");

        assert_gamma_close(&mixture, &mixture_expected, 1.0e-4);
        assert_gamma_close(&hmm, &hmm_expected, 1.0e-4);
        let max_variant_delta = max_gamma_abs_diff(&mixture, &hmm);
        assert!(
            max_variant_delta > 0.05,
            "dense VBx variants unexpectedly converged on fixture: max_abs_delta={max_variant_delta}"
        );
    }

    #[test]
    fn dense_window_cache_rejects_mismatched_metadata() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("dense-cache.json");
        let metadata = DenseWindowCacheMetadata {
            schema_version: DENSE_WINDOW_CACHE_SCHEMA,
            audio_sha256: "sha256:audio".to_string(),
            sample_rate_hz: 16_000,
            embedder_fingerprint: "sha256:embedder".to_string(),
            embedder_dim: INPUT_DIM,
            window_s: DENSE_WINDOW_S,
            shift_s: DENSE_SHIFT_S,
            min_speech_overlap_s: DENSE_MIN_SPEECH_OVERLAP_S,
            plda_asset_id: "sha256:plda".to_string(),
        };
        let window = DenseWindow {
            range: TimeRange::new(0.0, DENSE_WINDOW_S),
            initial: SpeakerId(7),
            embedding: vec![0.125; PLDA_DIM],
        };
        write_dense_window_cache(&path, &metadata, std::slice::from_ref(&window));

        let loaded = read_dense_window_cache(&path, &metadata).expect("matching cache");
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].range, window.range);
        assert_eq!(loaded[0].initial, window.initial);
        assert_eq!(loaded[0].embedding, window.embedding);

        let mut mismatched = metadata.clone();
        mismatched.audio_sha256 = "sha256:other-audio".to_string();
        assert!(read_dense_window_cache(&path, &mismatched).is_none());

        let mut mismatched = metadata;
        mismatched.embedder_fingerprint = "sha256:other-embedder".to_string();
        assert!(read_dense_window_cache(&path, &mismatched).is_none());
    }

    fn sparse_fixture_features(value: &serde_json::Value) -> Vec<Vec<f32>> {
        value
            .as_array()
            .expect("sparse feature rows")
            .iter()
            .map(|row| {
                let mut feature = vec![0.0f32; PLDA_DIM];
                for entry in row.as_array().expect("sparse row") {
                    let pair = entry.as_array().expect("sparse pair");
                    let dim = pair[0].as_u64().expect("dim") as usize;
                    let value = pair[1].as_f64().expect("value") as f32;
                    feature[dim] = value;
                }
                feature
            })
            .collect()
    }

    fn assert_gamma_close(actual: &[Vec<f32>], expected: &[Vec<f32>], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        let max_abs_diff = max_gamma_abs_diff(actual, expected);
        assert!(
            max_abs_diff <= tolerance,
            "max_abs_gamma_diff={max_abs_diff}"
        );
    }

    fn max_gamma_abs_diff(left: &[Vec<f32>], right: &[Vec<f32>]) -> f32 {
        assert_eq!(left.len(), right.len());
        let mut max_abs_diff = 0.0f32;
        for (actual_row, expected_row) in left.iter().zip(right) {
            assert_eq!(actual_row.len(), expected_row.len());
            for (&actual_value, &expected_value) in actual_row.iter().zip(expected_row) {
                max_abs_diff = max_abs_diff.max((actual_value - expected_value).abs());
            }
        }
        max_abs_diff
    }
}
