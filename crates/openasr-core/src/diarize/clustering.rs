//! Agglomerative speaker clustering (pure Rust, no weights).
//!
//! Average-linkage agglomerative hierarchical clustering on cosine
//! dissimilarity (`1 - cos`), the sherpa-onnx-style default. Embedding counts
//! are small (one per speech segment — tens to low hundreds), so the naive
//! O(n^3) merge loop is comfortably fast. When pyannote segmentation context is
//! available, clustering also honors overlap cannot-link constraints.

use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

use thiserror::Error;

use super::calibration::{
    ClusteringCalibrationProfile, ContextGapCalibrationProfile, REDIMNET_CALIBRATION,
};
use super::contract::{DiarizeHint, SpeakerEmbedding, SpeakerId, TimeRange};
use super::embed::SpeakerEmbedder;

/// Default threshold on cosine **dissimilarity** (`1 - cos`): clusters merge
/// while their average-linkage distance is below this.
///
/// This is the ReDimNet2-B6 plain AHC threshold. Valid range is `[0, 2]`.
pub const DEFAULT_MERGE_THRESHOLD: f32 = REDIMNET_CALIBRATION.clustering.plain_merge_threshold;
/// Context-aware auto clustering can safely use a looser stop because overlap
/// constraints prevent merging simultaneous speakers.
pub const CONTEXT_AUTO_MERGE_THRESHOLD: f32 =
    REDIMNET_CALIBRATION.clustering.context_auto_merge_threshold;

#[derive(Debug, Clone, Copy)]
pub struct ClusterContext {
    pub range: TimeRange,
    pub local_speaker: Option<SpeakerId>,
    pub overlap: bool,
}

/// Assigns each embedding to a session-relative [`SpeakerId`].
pub trait SpeakerClusterer: Send + Sync {
    fn cluster(&self, embeddings: &[SpeakerEmbedding], hint: DiarizeHint) -> Vec<SpeakerId>;

    fn cluster_with_context(
        &self,
        embeddings: &[SpeakerEmbedding],
        context: &[ClusterContext],
        hint: DiarizeHint,
    ) -> Vec<SpeakerId> {
        let _ = context;
        self.cluster(embeddings, hint)
    }
}

/// Average-linkage agglomerative clusterer over cosine dissimilarity.
#[derive(Debug, Clone, Copy)]
pub struct AgglomerativeClusterer {
    /// Merge stop threshold used for `Auto` / `Threshold` hints.
    pub threshold: f32,
    profile: ClusteringCalibrationProfile,
}

impl Default for AgglomerativeClusterer {
    fn default() -> Self {
        Self {
            threshold: DEFAULT_MERGE_THRESHOLD,
            profile: REDIMNET_CALIBRATION.clustering,
        }
    }
}

impl AgglomerativeClusterer {
    pub fn new(threshold: f32) -> Self {
        Self {
            threshold,
            profile: ClusteringCalibrationProfile {
                plain_merge_threshold: threshold,
                context_auto_merge_threshold: threshold,
                dense_context_min_embeddings: usize::MAX,
                dense_context_merge_threshold: threshold,
                context_gap: None,
            },
        }
    }

    pub fn for_embedder(embedder: &dyn SpeakerEmbedder) -> Self {
        Self::for_profile(embedder.calibration_profile().clustering)
    }

    pub(crate) fn for_profile(profile: ClusteringCalibrationProfile) -> Self {
        Self {
            threshold: profile.plain_merge_threshold,
            profile,
        }
    }
}

impl SpeakerClusterer for AgglomerativeClusterer {
    fn cluster(&self, embeddings: &[SpeakerEmbedding], hint: DiarizeHint) -> Vec<SpeakerId> {
        self.cluster_inner(embeddings, None, hint)
    }

    fn cluster_with_context(
        &self,
        embeddings: &[SpeakerEmbedding],
        context: &[ClusterContext],
        hint: DiarizeHint,
    ) -> Vec<SpeakerId> {
        self.cluster_inner(embeddings, Some(context), hint)
    }
}

impl AgglomerativeClusterer {
    fn cluster_inner(
        &self,
        embeddings: &[SpeakerEmbedding],
        context: Option<&[ClusterContext]>,
        hint: DiarizeHint,
    ) -> Vec<SpeakerId> {
        let n = embeddings.len();
        if n == 0 {
            return Vec::new();
        }
        if n == 1 {
            return vec![SpeakerId(0)];
        }

        // Pairwise cosine similarity (symmetric; diagonal unused).
        let mut sim = vec![0.0f32; n * n];
        for i in 0..n {
            for j in (i + 1)..n {
                let s = embeddings[i].cosine(&embeddings[j]);
                sim[i * n + j] = s;
                sim[j * n + i] = s;
            }
        }

        // Active clusters as member-index lists.
        let mut clusters: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
        let target = match hint {
            DiarizeHint::NumSpeakers(k) => (k as usize).max(1),
            _ => 1,
        };
        // Cosine dissimilarity over L2-normalized vectors is in [0, 2]; clamp so
        // an out-of-range client knob can't silently over-/under-split.
        let context = context.filter(|context| context.len() == n);
        let has_context_signal = context.is_some_and(context_has_real_signal);
        if matches!(hint, DiarizeHint::Auto)
            && has_context_signal
            && let Some(context) = context
            && let Some(labels) = self.cluster_by_context_gap(&sim, n, context)
        {
            return labels;
        }

        let stop_threshold = self.stop_threshold(n, has_context_signal, hint);

        while clusters.len() > target {
            // Closest pair by average-linkage cosine distance (1 - mean sim).
            let mut best = (0usize, 1usize);
            let mut best_dist = f32::INFINITY;
            for a in 0..clusters.len() {
                for b in (a + 1)..clusters.len() {
                    if context.is_some_and(|context| {
                        clusters_overlap(&clusters[a], &clusters[b], context)
                    }) {
                        continue;
                    }
                    let dist = 1.0 - average_similarity(&clusters[a], &clusters[b], &sim, n);
                    if dist < best_dist {
                        best_dist = dist;
                        best = (a, b);
                    }
                }
            }
            // For Auto/Threshold, stop once the closest clusters are too far.
            if !matches!(hint, DiarizeHint::NumSpeakers(_)) && best_dist > stop_threshold {
                break;
            }
            if !best_dist.is_finite() {
                break;
            }
            let (a, b) = best;
            let merged_b = clusters.remove(b);
            clusters[a].extend(merged_b);
        }

        if let Some(context) = context {
            assign_time_order_labels(&clusters, context, n)
        } else {
            assign_arrival_order_labels(&clusters, n)
        }
    }

    fn stop_threshold(&self, n: usize, has_context_signal: bool, hint: DiarizeHint) -> f32 {
        match hint {
            DiarizeHint::Threshold(t) => t.clamp(0.0, 2.0),
            DiarizeHint::Auto
                if has_context_signal && n >= self.profile.dense_context_min_embeddings =>
            {
                self.profile.dense_context_merge_threshold
            }
            DiarizeHint::Auto if has_context_signal => self.profile.context_auto_merge_threshold,
            _ => self.profile.plain_merge_threshold,
        }
    }

    fn cluster_by_context_gap(
        &self,
        sim: &[f32],
        n: usize,
        context: &[ClusterContext],
    ) -> Option<Vec<SpeakerId>> {
        let gap = self.profile.context_gap?;
        if n >= self.profile.dense_context_min_embeddings {
            return None;
        }
        let mut clusters: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
        let mut states: Vec<Option<Vec<Vec<usize>>>> = vec![None; n + 1];
        let mut merge_dist_by_k = vec![f32::NAN; n + 1];
        states[n] = Some(clusters.clone());

        while clusters.len() > 1 {
            let mut best = (0usize, 1usize);
            let mut best_dist = f32::INFINITY;
            for a in 0..clusters.len() {
                for b in (a + 1)..clusters.len() {
                    if clusters_overlap(&clusters[a], &clusters[b], context) {
                        continue;
                    }
                    let dist = 1.0 - average_similarity(&clusters[a], &clusters[b], sim, n);
                    if dist < best_dist {
                        best_dist = dist;
                        best = (a, b);
                    }
                }
            }
            if !best_dist.is_finite() {
                break;
            }
            let k = clusters.len();
            merge_dist_by_k[k] = best_dist;
            let (a, b) = best;
            let merged_b = clusters.remove(b);
            clusters[a].extend(merged_b);
            states[clusters.len()] = Some(clusters.clone());
        }

        let chosen_k = choose_context_gap_speaker_count(&merge_dist_by_k, &states, n, gap)?;
        let clusters = states[chosen_k].as_ref()?;
        Some(assign_time_order_labels(clusters, context, n))
    }
}

fn choose_context_gap_speaker_count(
    merge_dist_by_k: &[f32],
    states: &[Option<Vec<Vec<usize>>>],
    n: usize,
    gap: ContextGapCalibrationProfile,
) -> Option<usize> {
    let max_k = gap.max_speakers.min(n).max(1);
    let mut best_k = None;
    let mut best_gap = f32::NEG_INFINITY;
    for (k, state) in states.iter().enumerate().take(max_k + 1).skip(2) {
        let this_dist = merge_dist_by_k.get(k).copied().unwrap_or(f32::NAN);
        let prev_dist = merge_dist_by_k.get(k + 1).copied().unwrap_or(f32::NAN);
        if !this_dist.is_finite() || !prev_dist.is_finite() || state.is_none() {
            continue;
        }
        let candidate_gap = this_dist - prev_dist;
        if candidate_gap > best_gap {
            best_gap = candidate_gap;
            best_k = Some(k);
        }
    }

    if best_gap >= gap.min_gap {
        return best_k;
    }

    let fallback = gap.fallback_speakers.clamp(1, max_k);
    if states[fallback].is_some() {
        return Some(fallback);
    }
    (1..=max_k).rev().find(|&k| states[k].is_some())
}

fn clusters_overlap(a: &[usize], b: &[usize], context: &[ClusterContext]) -> bool {
    a.iter().any(|&i| {
        b.iter()
            .any(|&j| context[i].range.overlaps(&context[j].range))
    })
}

fn average_similarity(a: &[usize], b: &[usize], sim: &[f32], n: usize) -> f32 {
    let mut total = 0.0f32;
    for &i in a {
        for &j in b {
            total += sim[i * n + j];
        }
    }
    total / (a.len() * b.len()) as f32
}

fn context_has_real_signal(context: &[ClusterContext]) -> bool {
    context
        .iter()
        .any(|item| item.local_speaker.is_some() || item.overlap)
}

fn assign_time_order_labels(
    clusters: &[Vec<usize>],
    context: &[ClusterContext],
    n: usize,
) -> Vec<SpeakerId> {
    let mut order: Vec<usize> = (0..clusters.len()).collect();
    order.sort_by(|&left, &right| {
        let left_start = clusters[left]
            .iter()
            .map(|&member| context[member].range.start_s)
            .fold(f64::INFINITY, f64::min);
        let right_start = clusters[right]
            .iter()
            .map(|&member| context[member].range.start_s)
            .fold(f64::INFINITY, f64::min);
        left_start
            .partial_cmp(&right_start)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut labels = vec![SpeakerId(0); n];
    for (speaker_idx, &cluster_idx) in order.iter().enumerate() {
        for &member in &clusters[cluster_idx] {
            labels[member] = SpeakerId(speaker_idx as u32);
        }
    }
    labels
}

/// Label clusters by arrival order: the cluster whose earliest member index is
/// smallest gets `SPEAKER_00`, and so on, so labels are deterministic and
/// reflect when each speaker first appears.
fn assign_arrival_order_labels(clusters: &[Vec<usize>], n: usize) -> Vec<SpeakerId> {
    let mut order: Vec<usize> = (0..clusters.len()).collect();
    order.sort_by_key(|&c| clusters[c].iter().copied().min().unwrap_or(usize::MAX));

    let mut labels = vec![SpeakerId(0); n];
    for (speaker_idx, &cluster_idx) in order.iter().enumerate() {
        for &member in &clusters[cluster_idx] {
            labels[member] = SpeakerId(speaker_idx as u32);
        }
    }
    labels
}

/// Deterministic 3D-Speaker automatic clustering used by the recording-level
/// external diarizer. Short recordings use average-linkage AHC with a cosine
/// similarity cutoff of 0.4; recordings with at least 40 embeddings use the
/// p-pruned unnormalized-Laplacian spectral path with eigengap speaker-count
/// selection and deterministic k-means. Both paths then apply the reference
/// minor-cluster reassignment and centroid merge.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct AutomaticClusterer;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub(crate) enum AutomaticClusteringError {
    #[error("automatic speaker clustering was canceled")]
    Canceled,
}

const AUTOMATIC_SPECTRAL_MIN_EMBEDDINGS: usize = 40;
const AUTOMATIC_AHC_COSINE_THRESHOLD: f32 = 0.4;
const SPECTRAL_PVAL: f64 = 0.012;
const SPECTRAL_MIN_PNUM: usize = 6;
const SPECTRAL_MIN_SPEAKERS: usize = 1;
const SPECTRAL_MAX_SPEAKERS: usize = 15;
const MINOR_CLUSTER_MAX_SIZE: usize = 4;
const CENTROID_MERGE_COSINE_THRESHOLD: f32 = 0.8;
const AUTOMATIC_CANCEL_CHECK_INTERVAL: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AutomaticStrategy {
    Ahc,
    Spectral,
}

/// Test-only view of the production automatic-clustering route.
#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AutomaticClusteringStrategy {
    Ahc,
    Spectral,
}

#[cfg(test)]
impl From<AutomaticStrategy> for AutomaticClusteringStrategy {
    fn from(strategy: AutomaticStrategy) -> Self {
        match strategy {
            AutomaticStrategy::Ahc => Self::Ahc,
            AutomaticStrategy::Spectral => Self::Spectral,
        }
    }
}

/// Test-only snapshot of every decision boundary in automatic clustering.
///
/// The native fixture emitter consumes this seam instead of maintaining a
/// diagnostic copy of the algorithm. Labels are session-relative and preserve
/// the exact ordering produced by the production path.
#[cfg(test)]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct AutomaticClusteringDiagnostics {
    pub(crate) strategy: AutomaticClusteringStrategy,
    /// Smallest unnormalized-Laplacian eigenvalues used by the spectral path.
    /// Automatic clustering needs at most `SPECTRAL_MAX_SPEAKERS + 1` (16).
    /// Forced counts above that retain the same diagnostic prefix.
    pub(crate) spectral_eigenvalues: Vec<f64>,
    /// Eigengap-selected count for automatic spectral clustering. Forced
    /// counts and AHC deliberately leave this unset.
    pub(crate) eigengap_speakers: Option<usize>,
    /// Speaker count selected by the active route before post-processing.
    pub(crate) selected_speakers: usize,
    pub(crate) raw_labels: Vec<SpeakerId>,
    pub(crate) minor_filtered_labels: Vec<SpeakerId>,
    pub(crate) final_labels: Vec<SpeakerId>,
}

trait AutomaticClusteringObserver {
    fn record_strategy(&mut self, _strategy: AutomaticStrategy) {}

    fn record_spectral(&mut self, _eigenvalues: &[f64], _eigengap_speakers: Option<usize>) {}

    fn record_selected_speakers(&mut self, _speakers: usize) {}

    fn record_raw_labels(&mut self, _labels: &[usize]) {}

    fn record_minor_filtered_labels(&mut self, _labels: &[usize]) {}
}

struct IgnoreAutomaticClusteringDiagnostics;

impl AutomaticClusteringObserver for IgnoreAutomaticClusteringDiagnostics {}

#[cfg(test)]
#[derive(Default)]
struct CaptureAutomaticClusteringDiagnostics {
    strategy: Option<AutomaticClusteringStrategy>,
    spectral_eigenvalues: Vec<f64>,
    eigengap_speakers: Option<usize>,
    selected_speakers: Option<usize>,
    raw_labels: Vec<SpeakerId>,
    minor_filtered_labels: Vec<SpeakerId>,
}

#[cfg(test)]
impl CaptureAutomaticClusteringDiagnostics {
    fn finish(self, final_labels: Vec<SpeakerId>) -> AutomaticClusteringDiagnostics {
        AutomaticClusteringDiagnostics {
            strategy: self
                .strategy
                .expect("automatic clustering must record its strategy"),
            spectral_eigenvalues: self.spectral_eigenvalues,
            eigengap_speakers: self.eigengap_speakers,
            selected_speakers: self
                .selected_speakers
                .expect("automatic clustering must record its selected speaker count"),
            raw_labels: self.raw_labels,
            minor_filtered_labels: self.minor_filtered_labels,
            final_labels,
        }
    }
}

#[cfg(test)]
impl AutomaticClusteringObserver for CaptureAutomaticClusteringDiagnostics {
    fn record_strategy(&mut self, strategy: AutomaticStrategy) {
        self.strategy = Some(strategy.into());
    }

    fn record_spectral(&mut self, eigenvalues: &[f64], eigengap_speakers: Option<usize>) {
        self.spectral_eigenvalues = eigenvalues
            .iter()
            .take(SPECTRAL_MAX_SPEAKERS + 1)
            .copied()
            .collect();
        self.eigengap_speakers = eigengap_speakers;
    }

    fn record_selected_speakers(&mut self, speakers: usize) {
        self.selected_speakers = Some(speakers);
    }

    fn record_raw_labels(&mut self, labels: &[usize]) {
        self.raw_labels = speaker_ids_from_labels(labels);
    }

    fn record_minor_filtered_labels(&mut self, labels: &[usize]) {
        self.minor_filtered_labels = speaker_ids_from_labels(labels);
    }
}

impl AutomaticClusterer {
    pub(crate) fn cluster(
        &self,
        embeddings: &[SpeakerEmbedding],
        hint: DiarizeHint,
        canceled: &dyn Fn() -> bool,
    ) -> Result<Vec<SpeakerId>, AutomaticClusteringError> {
        let mut observer = IgnoreAutomaticClusteringDiagnostics;
        self.cluster_observed(embeddings, hint, canceled, &mut observer)
    }

    /// Run the exact production clustering path while retaining its
    /// intermediate decisions for the ignored native fixture emitter.
    #[cfg(test)]
    pub(crate) fn diagnostics(
        &self,
        embeddings: &[SpeakerEmbedding],
        hint: DiarizeHint,
        canceled: &dyn Fn() -> bool,
    ) -> Result<AutomaticClusteringDiagnostics, AutomaticClusteringError> {
        let mut observer = CaptureAutomaticClusteringDiagnostics::default();
        let final_labels = self.cluster_observed(embeddings, hint, canceled, &mut observer)?;
        Ok(observer.finish(final_labels))
    }

    fn cluster_observed<O: AutomaticClusteringObserver>(
        &self,
        embeddings: &[SpeakerEmbedding],
        hint: DiarizeHint,
        canceled: &dyn Fn() -> bool,
        observer: &mut O,
    ) -> Result<Vec<SpeakerId>, AutomaticClusteringError> {
        automatic_cancel_checkpoint(canceled)?;
        let strategy = match hint {
            DiarizeHint::NumSpeakers(_) => AutomaticStrategy::Spectral,
            DiarizeHint::Threshold(_) => AutomaticStrategy::Ahc,
            DiarizeHint::Auto => Self::strategy_for_len(embeddings.len()),
        };
        observer.record_strategy(strategy);

        if embeddings.len() <= 1 {
            let labels = vec![0usize; embeddings.len()];
            observer.record_selected_speakers(embeddings.len());
            observer.record_raw_labels(&labels);
            observer.record_minor_filtered_labels(&labels);
            return Ok(as_speaker_ids(labels));
        }

        let labels = match hint {
            DiarizeHint::NumSpeakers(speakers) => spectral_labels(
                embeddings,
                Some((speakers as usize).clamp(1, embeddings.len())),
                canceled,
                observer,
            )?,
            DiarizeHint::Threshold(distance) => {
                let labels = ahc_labels(embeddings, 1.0 - distance.clamp(0.0, 2.0), 1, canceled)?;
                observer.record_selected_speakers(label_count(&labels));
                labels
            }
            DiarizeHint::Auto => match strategy {
                AutomaticStrategy::Ahc => {
                    let labels =
                        ahc_labels(embeddings, AUTOMATIC_AHC_COSINE_THRESHOLD, 1, canceled)?;
                    observer.record_selected_speakers(label_count(&labels));
                    labels
                }
                AutomaticStrategy::Spectral => {
                    spectral_labels(embeddings, None, canceled, observer)?
                }
            },
        };
        automatic_cancel_checkpoint(canceled)?;
        observer.record_raw_labels(&labels);

        if matches!(hint, DiarizeHint::NumSpeakers(_)) {
            observer.record_minor_filtered_labels(&labels);
            return Ok(as_speaker_ids(compact_labels(&labels)));
        }
        let labels = filter_minor_clusters(&labels, embeddings, MINOR_CLUSTER_MAX_SIZE, canceled)?;
        observer.record_minor_filtered_labels(&labels);
        Ok(as_speaker_ids(compact_labels(&merge_similar_centroids(
            &labels,
            embeddings,
            CENTROID_MERGE_COSINE_THRESHOLD,
            canceled,
        )?)))
    }

    fn strategy_for_len(len: usize) -> AutomaticStrategy {
        if len < AUTOMATIC_SPECTRAL_MIN_EMBEDDINGS {
            AutomaticStrategy::Ahc
        } else {
            AutomaticStrategy::Spectral
        }
    }
}

fn as_speaker_ids(labels: Vec<usize>) -> Vec<SpeakerId> {
    labels
        .into_iter()
        .map(|label| SpeakerId(label as u32))
        .collect()
}

#[cfg(test)]
fn speaker_ids_from_labels(labels: &[usize]) -> Vec<SpeakerId> {
    labels
        .iter()
        .map(|&label| SpeakerId(label as u32))
        .collect()
}

fn label_count(labels: &[usize]) -> usize {
    labels.iter().copied().max().map_or(0, |label| label + 1)
}

fn ahc_labels(
    embeddings: &[SpeakerEmbedding],
    cosine_threshold: f32,
    target_speakers: usize,
    canceled: &dyn Fn() -> bool,
) -> Result<Vec<usize>, AutomaticClusteringError> {
    let n = embeddings.len();
    let mut similarity = vec![0.0f32; n * n];
    for left in 0..n {
        automatic_cancel_checkpoint(canceled)?;
        similarity[left * n + left] = 1.0;
        for right in (left + 1)..n {
            if right % AUTOMATIC_CANCEL_CHECK_INTERVAL == 0 {
                automatic_cancel_checkpoint(canceled)?;
            }
            let value = embeddings[left].cosine(&embeddings[right]);
            similarity[left * n + right] = value;
            similarity[right * n + left] = value;
        }
    }
    let mut clusters: Vec<Vec<usize>> = (0..n).map(|index| vec![index]).collect();
    while clusters.len() > target_speakers.max(1) {
        automatic_cancel_checkpoint(canceled)?;
        let mut best = None;
        for left in 0..clusters.len() {
            automatic_cancel_checkpoint(canceled)?;
            for right in (left + 1)..clusters.len() {
                let mean = average_similarity(&clusters[left], &clusters[right], &similarity, n);
                let candidate = (mean, left, right);
                if best.is_none_or(|current: (f32, usize, usize)| {
                    candidate.0 > current.0
                        || (candidate.0 == current.0
                            && (candidate.1, candidate.2) < (current.1, current.2))
                }) {
                    best = Some(candidate);
                }
            }
        }
        let Some((mean, left, right)) = best else {
            break;
        };
        if mean < cosine_threshold && target_speakers <= 1 {
            break;
        }
        let merged = clusters.remove(right);
        clusters[left].extend(merged);
    }
    automatic_cancel_checkpoint(canceled)?;
    Ok(raw_labels_from_clusters(&clusters, n))
}

fn spectral_labels<O: AutomaticClusteringObserver>(
    embeddings: &[SpeakerEmbedding],
    forced_speakers: Option<usize>,
    canceled: &dyn Fn() -> bool,
    observer: &mut O,
) -> Result<Vec<usize>, AutomaticClusteringError> {
    let affinity = pruned_affinity(embeddings, canceled)?;
    spectral_labels_from_affinity(&affinity, embeddings, forced_speakers, canceled, observer)
}

fn spectral_labels_from_affinity<O: AutomaticClusteringObserver>(
    affinity: &SparseAffinity,
    embeddings: &[SpeakerEmbedding],
    forced_speakers: Option<usize>,
    canceled: &dyn Fn() -> bool,
    observer: &mut O,
) -> Result<Vec<usize>, AutomaticClusteringError> {
    let vector_count = forced_speakers
        .unwrap_or(SPECTRAL_MAX_SPEAKERS + 1)
        .min(embeddings.len());
    let (eigenvalues, eigenvectors) =
        smallest_laplacian_eigenvectors(affinity, vector_count, canceled)?;
    let eigengap_speakers = forced_speakers.is_none().then(|| {
        choose_eigengap_speakers(&eigenvalues, SPECTRAL_MIN_SPEAKERS, SPECTRAL_MAX_SPEAKERS)
    });
    let speakers = forced_speakers
        .or(eigengap_speakers)
        .unwrap_or(1)
        .clamp(1, embeddings.len());
    observer.record_spectral(&eigenvalues, eigengap_speakers);
    observer.record_selected_speakers(speakers);
    let mut features = Vec::with_capacity(embeddings.len() * speakers);
    for row in 0..embeddings.len() {
        if row % AUTOMATIC_CANCEL_CHECK_INTERVAL == 0 {
            automatic_cancel_checkpoint(canceled)?;
        }
        features
            .extend_from_slice(&eigenvectors[row * vector_count..row * vector_count + speakers]);
    }
    deterministic_kmeans(&features, embeddings.len(), speakers, speakers, canceled)
}

#[derive(Debug, Clone, PartialEq)]
struct SparseAffinity {
    rows: Vec<Vec<(usize, f64)>>,
    degree: Vec<f64>,
}

#[derive(Debug, Clone, Copy)]
struct RankedSimilarity {
    similarity: f32,
    column: usize,
}

impl PartialEq for RankedSimilarity {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for RankedSimilarity {}

impl PartialOrd for RankedSimilarity {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for RankedSimilarity {
    fn cmp(&self, other: &Self) -> Ordering {
        self.similarity
            .total_cmp(&other.similarity)
            .then_with(|| self.column.cmp(&other.column))
    }
}

fn spectral_retained_per_row(n: usize) -> usize {
    let remove =
        (((1.0 - SPECTRAL_PVAL) * n as f64) as usize).min(n.saturating_sub(SPECTRAL_MIN_PNUM));
    n - remove
}

fn pruned_affinity(
    embeddings: &[SpeakerEmbedding],
    canceled: &dyn Fn() -> bool,
) -> Result<SparseAffinity, AutomaticClusteringError> {
    let directed = pruned_directed_rows(embeddings, canceled)?;
    symmetrize_pruned_affinity(&directed, canceled)
}

/// Retains only the exact p-pruned directed neighborhood for each row.
///
/// The reference ranks every row by `(similarity, column)` and keeps the last
/// `max(ceil(p * n), min_pnum)` entries, including self. A fixed-capacity
/// min-heap produces the same selection without materializing either an
/// `n * n` directed matrix or a full `n`-entry ranked row.
fn pruned_directed_rows(
    embeddings: &[SpeakerEmbedding],
    canceled: &dyn Fn() -> bool,
) -> Result<Vec<Vec<RankedSimilarity>>, AutomaticClusteringError> {
    let n = embeddings.len();
    let retain = spectral_retained_per_row(n);
    let mut directed = Vec::with_capacity(n);
    for row in 0..n {
        automatic_cancel_checkpoint(canceled)?;
        let mut retained = BinaryHeap::with_capacity(retain);
        for column in 0..n {
            if column % AUTOMATIC_CANCEL_CHECK_INTERVAL == 0 {
                automatic_cancel_checkpoint(canceled)?;
            }
            let candidate = RankedSimilarity {
                similarity: embeddings[row].cosine(&embeddings[column]),
                column,
            };
            if retained.len() < retain {
                retained.push(Reverse(candidate));
            } else if retained
                .peek()
                .is_some_and(|Reverse(current)| candidate > *current)
            {
                retained.pop();
                retained.push(Reverse(candidate));
            }
        }
        automatic_cancel_checkpoint(canceled)?;
        let mut retained: Vec<_> = retained
            .into_iter()
            .map(|Reverse(candidate)| candidate)
            .collect();
        retained.sort_unstable_by_key(|candidate| candidate.column);
        automatic_cancel_checkpoint(canceled)?;
        debug_assert_eq!(retained.len(), retain);
        directed.push(retained);
    }
    Ok(directed)
}

fn symmetrize_pruned_affinity(
    directed: &[Vec<RankedSimilarity>],
    canceled: &dyn Fn() -> bool,
) -> Result<SparseAffinity, AutomaticClusteringError> {
    let n = directed.len();
    let mut rows = vec![Vec::new(); n];
    for source in 0..n {
        automatic_cancel_checkpoint(canceled)?;
        for (edge_index, candidate) in directed[source].iter().enumerate() {
            if edge_index % AUTOMATIC_CANCEL_CHECK_INTERVAL == 0 {
                automatic_cancel_checkpoint(canceled)?;
            }
            let target = candidate.column;
            if source == target {
                continue;
            }
            let reverse = directed_similarity(&directed[target], source);
            if source > target && reverse.is_some() {
                continue;
            }
            let weight = 0.5 * (candidate.similarity as f64 + reverse.unwrap_or(0.0) as f64);
            if weight == 0.0 {
                continue;
            }
            rows[source].push((target, weight));
            rows[target].push((source, weight));
        }
    }
    let mut degree = Vec::with_capacity(n);
    for row in &mut rows {
        automatic_cancel_checkpoint(canceled)?;
        row.sort_unstable_by_key(|&(neighbor, _)| neighbor);
        degree.push(row.iter().map(|&(_, weight)| weight.abs()).sum());
    }
    Ok(SparseAffinity { rows, degree })
}

fn directed_similarity(row: &[RankedSimilarity], column: usize) -> Option<f32> {
    row.binary_search_by_key(&column, |candidate| candidate.column)
        .ok()
        .map(|index| row[index].similarity)
}

/// Orthogonal iteration on a shifted sparse Laplacian computes only the
/// smallest 16 eigenvectors needed by the speaker-count contract, rather than
/// paying for an O(n^3) dense decomposition of every recording.
fn smallest_laplacian_eigenvectors(
    affinity: &SparseAffinity,
    vectors: usize,
    canceled: &dyn Fn() -> bool,
) -> Result<(Vec<f64>, Vec<f64>), AutomaticClusteringError> {
    automatic_cancel_checkpoint(canceled)?;
    let n = affinity.rows.len();
    let q = vectors.min(n).max(1);
    let shift = affinity
        .degree
        .iter()
        .copied()
        .fold(0.0f64, f64::max)
        .mul_add(2.0, 1.0e-6);
    let mut basis = vec![0.0f64; n * q];
    for row in 0..n {
        if row % AUTOMATIC_CANCEL_CHECK_INTERVAL == 0 {
            automatic_cancel_checkpoint(canceled)?;
        }
        basis[row * q] = 1.0;
        for column in 1..q {
            let phase = (row + 1) as f64 * (column + 1) as f64;
            basis[row * q + column] =
                (phase * 0.754_877_666_246_692_7).sin() + (phase * 0.569_840_290_998_053_2).cos();
        }
    }
    orthonormalize_columns(&mut basis, n, q, canceled)?;

    let mut next = vec![0.0f64; n * q];
    for _ in 0..160 {
        automatic_cancel_checkpoint(canceled)?;
        next.fill(0.0);
        for row in 0..n {
            if row % AUTOMATIC_CANCEL_CHECK_INTERVAL == 0 {
                automatic_cancel_checkpoint(canceled)?;
            }
            for column in 0..q {
                next[row * q + column] = (shift - affinity.degree[row]) * basis[row * q + column];
            }
            for &(neighbor, weight) in &affinity.rows[row] {
                for column in 0..q {
                    next[row * q + column] += weight * basis[neighbor * q + column];
                }
            }
        }
        orthonormalize_columns(&mut next, n, q, canceled)?;
        std::mem::swap(&mut basis, &mut next);
    }

    let laplacian_basis = apply_laplacian(affinity, &basis, q, canceled)?;
    let mut projected = vec![0.0f64; q * q];
    for left in 0..q {
        automatic_cancel_checkpoint(canceled)?;
        for right in left..q {
            let value = (0..n)
                .map(|row| basis[row * q + left] * laplacian_basis[row * q + right])
                .sum();
            projected[left * q + right] = value;
            projected[right * q + left] = value;
        }
    }
    let (values, rotation) = jacobi_symmetric_eigen(projected, q);
    let mut order: Vec<usize> = (0..q).collect();
    order.sort_by(|&left, &right| {
        values[left]
            .total_cmp(&values[right])
            .then_with(|| left.cmp(&right))
    });
    let eigenvalues = order.iter().map(|&index| values[index]).collect();
    let mut eigenvectors = vec![0.0f64; n * q];
    for row in 0..n {
        if row % AUTOMATIC_CANCEL_CHECK_INTERVAL == 0 {
            automatic_cancel_checkpoint(canceled)?;
        }
        for (output, &input) in order.iter().enumerate() {
            eigenvectors[row * q + output] = (0..q)
                .map(|inner| basis[row * q + inner] * rotation[inner * q + input])
                .sum();
        }
    }
    automatic_cancel_checkpoint(canceled)?;
    Ok((eigenvalues, eigenvectors))
}

fn apply_laplacian(
    affinity: &SparseAffinity,
    input: &[f64],
    columns: usize,
    canceled: &dyn Fn() -> bool,
) -> Result<Vec<f64>, AutomaticClusteringError> {
    let mut output = vec![0.0f64; input.len()];
    for row in 0..affinity.rows.len() {
        if row % AUTOMATIC_CANCEL_CHECK_INTERVAL == 0 {
            automatic_cancel_checkpoint(canceled)?;
        }
        for column in 0..columns {
            output[row * columns + column] = affinity.degree[row] * input[row * columns + column];
        }
        for &(neighbor, weight) in &affinity.rows[row] {
            for column in 0..columns {
                output[row * columns + column] -= weight * input[neighbor * columns + column];
            }
        }
    }
    automatic_cancel_checkpoint(canceled)?;
    Ok(output)
}

fn orthonormalize_columns(
    matrix: &mut [f64],
    rows: usize,
    columns: usize,
    canceled: &dyn Fn() -> bool,
) -> Result<(), AutomaticClusteringError> {
    for column in 0..columns {
        automatic_cancel_checkpoint(canceled)?;
        for _ in 0..2 {
            for previous in 0..column {
                let projection: f64 = (0..rows)
                    .map(|row| matrix[row * columns + column] * matrix[row * columns + previous])
                    .sum();
                for row in 0..rows {
                    if row % AUTOMATIC_CANCEL_CHECK_INTERVAL == 0 {
                        automatic_cancel_checkpoint(canceled)?;
                    }
                    matrix[row * columns + column] -= projection * matrix[row * columns + previous];
                }
            }
        }
        let norm = (0..rows)
            .map(|row| matrix[row * columns + column].powi(2))
            .sum::<f64>()
            .sqrt();
        if norm <= f64::EPSILON {
            for row in 0..rows {
                matrix[row * columns + column] = usize::from(row == column % rows) as f64;
            }
            continue;
        }
        for row in 0..rows {
            if row % AUTOMATIC_CANCEL_CHECK_INTERVAL == 0 {
                automatic_cancel_checkpoint(canceled)?;
            }
            matrix[row * columns + column] /= norm;
        }
    }
    automatic_cancel_checkpoint(canceled)?;
    Ok(())
}

fn jacobi_symmetric_eigen(mut matrix: Vec<f64>, n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut vectors = vec![0.0f64; n * n];
    for index in 0..n {
        vectors[index * n + index] = 1.0;
    }
    for _ in 0..(n * n * 32).max(1) {
        let mut pivot = (0usize, 0usize);
        let mut magnitude = 0.0f64;
        for left in 0..n {
            for right in (left + 1)..n {
                let value = matrix[left * n + right].abs();
                if value > magnitude {
                    magnitude = value;
                    pivot = (left, right);
                }
            }
        }
        if magnitude < 1.0e-10 {
            break;
        }
        let (left, right) = pivot;
        let angle = 0.5
            * (2.0 * matrix[left * n + right])
                .atan2(matrix[right * n + right] - matrix[left * n + left]);
        let (sine, cosine) = angle.sin_cos();
        for index in 0..n {
            if index == left || index == right {
                continue;
            }
            let a = matrix[index * n + left];
            let b = matrix[index * n + right];
            let new_left = cosine * a - sine * b;
            let new_right = sine * a + cosine * b;
            matrix[index * n + left] = new_left;
            matrix[left * n + index] = new_left;
            matrix[index * n + right] = new_right;
            matrix[right * n + index] = new_right;
        }
        let a = matrix[left * n + left];
        let b = matrix[right * n + right];
        let c = matrix[left * n + right];
        matrix[left * n + left] = cosine * cosine * a - 2.0 * sine * cosine * c + sine * sine * b;
        matrix[right * n + right] = sine * sine * a + 2.0 * sine * cosine * c + cosine * cosine * b;
        matrix[left * n + right] = 0.0;
        matrix[right * n + left] = 0.0;
        for row in 0..n {
            let a = vectors[row * n + left];
            let b = vectors[row * n + right];
            vectors[row * n + left] = cosine * a - sine * b;
            vectors[row * n + right] = sine * a + cosine * b;
        }
    }
    let values = (0..n).map(|index| matrix[index * n + index]).collect();
    (values, vectors)
}

fn choose_eigengap_speakers(values: &[f64], min_speakers: usize, max_speakers: usize) -> usize {
    let upper = max_speakers.min(values.len().saturating_sub(1));
    (min_speakers..=upper)
        .max_by(|&left, &right| {
            let left_gap = values[left] - values[left - 1];
            let right_gap = values[right] - values[right - 1];
            left_gap
                .total_cmp(&right_gap)
                .then_with(|| right.cmp(&left))
        })
        .unwrap_or(1)
}

fn deterministic_kmeans(
    points: &[f64],
    rows: usize,
    dimensions: usize,
    clusters: usize,
    canceled: &dyn Fn() -> bool,
) -> Result<Vec<usize>, AutomaticClusteringError> {
    automatic_cancel_checkpoint(canceled)?;
    if clusters <= 1 {
        return Ok(vec![0; rows]);
    }
    debug_assert_eq!(points.len(), rows * dimensions);
    debug_assert!(clusters <= rows);
    let centers =
        deterministic_greedy_kmeans_plus_plus(points, rows, dimensions, clusters, canceled)?;
    let mut centroids: Vec<f64> = centers
        .iter()
        .flat_map(|&row| {
            points[row * dimensions..(row + 1) * dimensions]
                .iter()
                .copied()
        })
        .collect();
    let mut labels = vec![usize::MAX; rows];
    for _ in 0..300 {
        automatic_cancel_checkpoint(canceled)?;
        let mut changed = false;
        for row in 0..rows {
            if row % AUTOMATIC_CANCEL_CHECK_INTERVAL == 0 {
                automatic_cancel_checkpoint(canceled)?;
            }
            let previous = labels[row];
            let label = (0..clusters)
                .min_by(|&left, &right| {
                    squared_distance(
                        &points[row * dimensions..(row + 1) * dimensions],
                        &centroids[left * dimensions..(left + 1) * dimensions],
                    )
                    .total_cmp(&squared_distance(
                        &points[row * dimensions..(row + 1) * dimensions],
                        &centroids[right * dimensions..(right + 1) * dimensions],
                    ))
                    .then_with(|| {
                        usize::from(left != previous).cmp(&usize::from(right != previous))
                    })
                    .then_with(|| left.cmp(&right))
                })
                .unwrap_or(0);
            changed |= labels[row] != label;
            labels[row] = label;
        }
        let mut sums = vec![0.0f64; clusters * dimensions];
        let mut counts = vec![0usize; clusters];
        for row in 0..rows {
            if row % AUTOMATIC_CANCEL_CHECK_INTERVAL == 0 {
                automatic_cancel_checkpoint(canceled)?;
            }
            counts[labels[row]] += 1;
            for dimension in 0..dimensions {
                sums[labels[row] * dimensions + dimension] += points[row * dimensions + dimension];
            }
        }
        for cluster in 0..clusters {
            automatic_cancel_checkpoint(canceled)?;
            if counts[cluster] == 0 {
                let replacement = (0..rows)
                    .filter(|&row| counts[labels[row]] > 1)
                    .max_by(|&left, &right| {
                        let left_label = labels[left];
                        let right_label = labels[right];
                        squared_distance(
                            &points[left * dimensions..(left + 1) * dimensions],
                            &centroids[left_label * dimensions..(left_label + 1) * dimensions],
                        )
                        .total_cmp(&squared_distance(
                            &points[right * dimensions..(right + 1) * dimensions],
                            &centroids[right_label * dimensions..(right_label + 1) * dimensions],
                        ))
                        .then_with(|| right.cmp(&left))
                    })
                    .expect("clusters <= rows guarantees a donor for every empty cluster");
                let donor = labels[replacement];
                counts[donor] -= 1;
                counts[cluster] = 1;
                labels[replacement] = cluster;
                for dimension in 0..dimensions {
                    let value = points[replacement * dimensions + dimension];
                    sums[donor * dimensions + dimension] -= value;
                    sums[cluster * dimensions + dimension] = value;
                }
                changed = true;
            }
        }
        for cluster in 0..clusters {
            debug_assert!(counts[cluster] > 0);
            for dimension in 0..dimensions {
                centroids[cluster * dimensions + dimension] =
                    sums[cluster * dimensions + dimension] / counts[cluster] as f64;
            }
        }
        if !changed {
            break;
        }
    }
    debug_assert_eq!(label_count(&labels), clusters);
    automatic_cancel_checkpoint(canceled)?;
    Ok(compact_labels(&labels))
}

/// Fixed-seed greedy k-means++ initialization, matching the reference
/// 3D-Speaker/sklearn algorithm class without depending on process-global RNG.
fn deterministic_greedy_kmeans_plus_plus(
    points: &[f64],
    rows: usize,
    dimensions: usize,
    clusters: usize,
    canceled: &dyn Fn() -> bool,
) -> Result<Vec<usize>, AutomaticClusteringError> {
    automatic_cancel_checkpoint(canceled)?;
    let mut rng = SplitMix64::new(0);
    let mut centers = Vec::with_capacity(clusters);
    centers.push(rng.index(rows));
    let mut closest = Vec::with_capacity(rows);
    for row in 0..rows {
        if row % AUTOMATIC_CANCEL_CHECK_INTERVAL == 0 {
            automatic_cancel_checkpoint(canceled)?;
        }
        closest.push(point_distance(points, dimensions, row, centers[0]));
    }
    let local_trials = 2 + (clusters as f64).ln() as usize;

    while centers.len() < clusters {
        automatic_cancel_checkpoint(canceled)?;
        let potential: f64 = closest.iter().sum();
        let next = if potential <= f64::EPSILON {
            (0..rows)
                .find(|row| !centers.contains(row))
                .expect("clusters <= rows guarantees an unused center")
        } else {
            let mut cumulative = Vec::with_capacity(rows);
            let mut total = 0.0;
            for (row, &distance) in closest.iter().enumerate() {
                if row % AUTOMATIC_CANCEL_CHECK_INTERVAL == 0 {
                    automatic_cancel_checkpoint(canceled)?;
                }
                total += distance;
                cumulative.push(total);
            }
            let mut best: Option<(f64, usize)> = None;
            for _ in 0..local_trials {
                automatic_cancel_checkpoint(canceled)?;
                let target = rng.unit_f64() * potential;
                let candidate = cumulative
                    .partition_point(|&value| value <= target)
                    .min(rows - 1);
                let candidate_potential =
                    kmeans_candidate_potential(points, dimensions, &closest, candidate, canceled)?;
                if best.is_none_or(|(best_potential, best_candidate)| {
                    candidate_potential
                        .total_cmp(&best_potential)
                        .then_with(|| candidate.cmp(&best_candidate))
                        == Ordering::Less
                }) {
                    best = Some((candidate_potential, candidate));
                }
            }
            best.expect("greedy k-means++ always evaluates a candidate")
                .1
        };
        centers.push(next);
        for (row, distance) in closest.iter_mut().enumerate() {
            if row % AUTOMATIC_CANCEL_CHECK_INTERVAL == 0 {
                automatic_cancel_checkpoint(canceled)?;
            }
            *distance = distance.min(point_distance(points, dimensions, row, next));
        }
    }
    automatic_cancel_checkpoint(canceled)?;
    Ok(centers)
}

fn kmeans_candidate_potential(
    points: &[f64],
    dimensions: usize,
    closest: &[f64],
    candidate: usize,
    canceled: &dyn Fn() -> bool,
) -> Result<f64, AutomaticClusteringError> {
    let mut potential = 0.0;
    for (row, &distance) in closest.iter().enumerate() {
        if row % AUTOMATIC_CANCEL_CHECK_INTERVAL == 0 {
            automatic_cancel_checkpoint(canceled)?;
        }
        potential += distance.min(point_distance(points, dimensions, row, candidate));
    }
    Ok(potential)
}

fn point_distance(points: &[f64], dimensions: usize, left: usize, right: usize) -> f64 {
    squared_distance(
        &points[left * dimensions..(left + 1) * dimensions],
        &points[right * dimensions..(right + 1) * dimensions],
    )
}

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn index(&mut self, upper: usize) -> usize {
        (self.next_u64() % upper as u64) as usize
    }

    fn unit_f64(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
    }
}

fn squared_distance(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(a, b)| (a - b).powi(2)).sum()
}

fn filter_minor_clusters(
    labels: &[usize],
    embeddings: &[SpeakerEmbedding],
    max_minor_size: usize,
    canceled: &dyn Fn() -> bool,
) -> Result<Vec<usize>, AutomaticClusteringError> {
    automatic_cancel_checkpoint(canceled)?;
    let cluster_count = labels.iter().copied().max().map_or(0, |label| label + 1);
    let mut sizes = vec![0usize; cluster_count];
    for (index, &label) in labels.iter().enumerate() {
        if index % AUTOMATIC_CANCEL_CHECK_INTERVAL == 0 {
            automatic_cancel_checkpoint(canceled)?;
        }
        sizes[label] += 1;
    }
    let major: Vec<usize> = sizes
        .iter()
        .enumerate()
        .filter_map(|(label, &size)| (size > max_minor_size).then_some(label))
        .collect();
    if major.is_empty() {
        return Ok(vec![0; labels.len()]);
    }
    let centroids = embedding_centroids(labels, embeddings, canceled)?;
    let mut filtered = Vec::with_capacity(labels.len());
    for (index, &label) in labels.iter().enumerate() {
        if index % AUTOMATIC_CANCEL_CHECK_INTERVAL == 0 {
            automatic_cancel_checkpoint(canceled)?;
        }
        let selected = if sizes[label] > max_minor_size {
            label
        } else {
            major
                .iter()
                .copied()
                .max_by(|&left, &right| {
                    embeddings[index]
                        .cosine(&centroids[left])
                        .total_cmp(&embeddings[index].cosine(&centroids[right]))
                        .then_with(|| right.cmp(&left))
                })
                .unwrap_or(0)
        };
        filtered.push(selected);
    }
    automatic_cancel_checkpoint(canceled)?;
    Ok(filtered)
}

fn merge_similar_centroids(
    labels: &[usize],
    embeddings: &[SpeakerEmbedding],
    threshold: f32,
    canceled: &dyn Fn() -> bool,
) -> Result<Vec<usize>, AutomaticClusteringError> {
    automatic_cancel_checkpoint(canceled)?;
    let mut labels = compact_labels(labels);
    loop {
        automatic_cancel_checkpoint(canceled)?;
        let centroids = embedding_centroids(&labels, embeddings, canceled)?;
        if centroids.len() <= 1 {
            break;
        }
        let mut best = None;
        for left in 0..centroids.len() {
            automatic_cancel_checkpoint(canceled)?;
            for right in (left + 1)..centroids.len() {
                let similarity = centroids[left].cosine(&centroids[right]);
                let candidate = (similarity, left, right);
                if best.is_none_or(|current: (f32, usize, usize)| {
                    candidate.0 > current.0
                        || (candidate.0 == current.0
                            && (candidate.1, candidate.2) < (current.1, current.2))
                }) {
                    best = Some(candidate);
                }
            }
        }
        let Some((similarity, keep, merge)) = best else {
            break;
        };
        if similarity < threshold {
            break;
        }
        for label in &mut labels {
            if *label == merge {
                *label = keep;
            }
        }
        labels = compact_labels(&labels);
    }
    automatic_cancel_checkpoint(canceled)?;
    Ok(labels)
}

fn embedding_centroids(
    labels: &[usize],
    embeddings: &[SpeakerEmbedding],
    canceled: &dyn Fn() -> bool,
) -> Result<Vec<SpeakerEmbedding>, AutomaticClusteringError> {
    automatic_cancel_checkpoint(canceled)?;
    let count = labels.iter().copied().max().map_or(0, |label| label + 1);
    let dimensions = embeddings.first().map_or(0, SpeakerEmbedding::dim);
    let mut sums = vec![vec![0.0f32; dimensions]; count];
    for (index, (&label, embedding)) in labels.iter().zip(embeddings).enumerate() {
        if index % AUTOMATIC_CANCEL_CHECK_INTERVAL == 0 {
            automatic_cancel_checkpoint(canceled)?;
        }
        for (sum, &value) in sums[label].iter_mut().zip(&embedding.0) {
            *sum += value;
        }
    }
    automatic_cancel_checkpoint(canceled)?;
    Ok(sums
        .into_iter()
        .map(SpeakerEmbedding::l2_normalized)
        .collect())
}

fn automatic_cancel_checkpoint(
    canceled: &dyn Fn() -> bool,
) -> Result<(), AutomaticClusteringError> {
    if canceled() {
        Err(AutomaticClusteringError::Canceled)
    } else {
        Ok(())
    }
}

fn compact_labels(labels: &[usize]) -> Vec<usize> {
    let mut mapping = std::collections::BTreeMap::new();
    let mut next = 0usize;
    labels
        .iter()
        .map(|label| {
            *mapping.entry(*label).or_insert_with(|| {
                let assigned = next;
                next += 1;
                assigned
            })
        })
        .collect()
}

fn raw_labels_from_clusters(clusters: &[Vec<usize>], n: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..clusters.len()).collect();
    order.sort_by_key(|&cluster| {
        clusters[cluster]
            .iter()
            .copied()
            .min()
            .unwrap_or(usize::MAX)
    });
    let mut labels = vec![0usize; n];
    for (label, &cluster) in order.iter().enumerate() {
        for &member in &clusters[cluster] {
            labels[member] = label;
        }
    }
    labels
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::*;

    fn emb(v: Vec<f32>) -> SpeakerEmbedding {
        SpeakerEmbedding::l2_normalized(v)
    }

    fn never_cancel() -> bool {
        false
    }

    fn dense_pruned_affinity_oracle(embeddings: &[SpeakerEmbedding]) -> SparseAffinity {
        let n = embeddings.len();
        let remove = n - spectral_retained_per_row(n);
        let mut directed = vec![vec![0.0f64; n]; n];
        for row in 0..n {
            let mut ranked: Vec<(f32, usize)> = (0..n)
                .map(|column| (embeddings[row].cosine(&embeddings[column]), column))
                .collect();
            ranked.sort_by(|left, right| {
                left.0
                    .total_cmp(&right.0)
                    .then_with(|| left.1.cmp(&right.1))
            });
            for &(similarity, column) in ranked.iter().skip(remove) {
                directed[row][column] = similarity as f64;
            }
        }

        let mut rows = vec![Vec::new(); n];
        let mut degree = vec![0.0f64; n];
        for left in 0..n {
            for right in (left + 1)..n {
                let weight = 0.5 * (directed[left][right] + directed[right][left]);
                if weight == 0.0 {
                    continue;
                }
                rows[left].push((right, weight));
                rows[right].push((left, weight));
                degree[left] += weight.abs();
                degree[right] += weight.abs();
            }
        }
        SparseAffinity { rows, degree }
    }

    fn tied_signed_embeddings(n: usize) -> Vec<SpeakerEmbedding> {
        const PATTERNS: [[f32; 3]; 8] = [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        (0..n)
            .map(|index| emb(PATTERNS[index % PATTERNS.len()].to_vec()))
            .collect()
    }

    fn ctx(
        start_s: f64,
        end_s: f64,
        local_speaker: Option<SpeakerId>,
        overlap: bool,
    ) -> ClusterContext {
        ClusterContext {
            range: TimeRange::new(start_s, end_s),
            local_speaker,
            overlap,
        }
    }

    #[test]
    fn empty_and_single() {
        let clusterer = AgglomerativeClusterer::default();
        assert!(clusterer.cluster(&[], DiarizeHint::Auto).is_empty());
        assert_eq!(
            clusterer.cluster(&[emb(vec![1.0, 0.0])], DiarizeHint::Auto),
            vec![SpeakerId(0)]
        );
    }

    #[test]
    fn separates_two_clear_speakers_by_threshold() {
        // Two tight groups around orthogonal directions.
        let embeddings = vec![
            emb(vec![1.0, 0.05]),
            emb(vec![1.0, 0.0]),
            emb(vec![0.0, 1.0]),
            emb(vec![0.05, 1.0]),
        ];
        let labels = AgglomerativeClusterer::default().cluster(&embeddings, DiarizeHint::Auto);
        assert_eq!(labels[0], labels[1], "group A together");
        assert_eq!(labels[2], labels[3], "group B together");
        assert_ne!(labels[0], labels[2], "groups distinct");
        // Arrival order: first group is SPEAKER_00.
        assert_eq!(labels[0], SpeakerId(0));
        assert_eq!(labels[2], SpeakerId(1));
    }

    #[test]
    fn num_speakers_hint_forces_exact_count() {
        let embeddings = vec![
            emb(vec![1.0, 0.0]),
            emb(vec![0.9, 0.1]),
            emb(vec![0.0, 1.0]),
        ];
        let labels =
            AgglomerativeClusterer::default().cluster(&embeddings, DiarizeHint::NumSpeakers(1));
        assert!(labels.iter().all(|l| *l == SpeakerId(0)), "all one speaker");
    }

    #[test]
    fn high_threshold_merges_everything() {
        let embeddings = vec![emb(vec![1.0, 0.0]), emb(vec![0.0, 1.0])];
        let labels =
            AgglomerativeClusterer::default().cluster(&embeddings, DiarizeHint::Threshold(2.0));
        assert_eq!(labels[0], labels[1]);
    }

    #[test]
    fn context_without_pyannote_signal_uses_plain_threshold() {
        let embeddings = vec![emb(vec![1.0, 0.0]), emb(vec![0.5, 0.866_025_4])];
        let context = vec![ctx(0.0, 1.0, None, false), ctx(1.0, 2.0, None, false)];

        let labels = AgglomerativeClusterer::default().cluster_with_context(
            &embeddings,
            &context,
            DiarizeHint::Auto,
        );

        assert_ne!(
            labels[0], labels[1],
            "plain TimeRange context must not unlock the loose context threshold"
        );
    }

    #[test]
    fn local_slot_aba_is_not_repaired_after_clustering() {
        let embeddings = vec![
            emb(vec![1.0, 0.0]),
            emb(vec![0.4, 0.916_515_1]),
            emb(vec![1.0, 0.0]),
        ];
        let context = vec![
            ctx(0.0, 5.0, Some(SpeakerId(7)), false),
            ctx(5.0, 7.0, Some(SpeakerId(7)), false),
            ctx(7.0, 12.0, Some(SpeakerId(7)), false),
        ];
        let labels = AgglomerativeClusterer::default().cluster_with_context(
            &embeddings,
            &context,
            DiarizeHint::Auto,
        );

        assert_eq!(
            labels,
            vec![SpeakerId(0), SpeakerId(1), SpeakerId(0)],
            "A-B-A local-slot islands are left to the clustering evidence"
        );
    }

    #[test]
    fn automatic_strategy_switches_at_forty_embeddings() {
        assert_eq!(
            AutomaticClusterer::strategy_for_len(39),
            AutomaticStrategy::Ahc
        );
        assert_eq!(
            AutomaticClusterer::strategy_for_len(40),
            AutomaticStrategy::Spectral
        );
    }

    #[test]
    fn streaming_affinity_matches_dense_oracle_at_p_pruning_boundaries() {
        for n in [499, 500, 501] {
            let embeddings = tied_signed_embeddings(n);
            let streaming = pruned_affinity(&embeddings, &never_cancel).unwrap();
            let dense = dense_pruned_affinity_oracle(&embeddings);

            assert_eq!(streaming, dense, "affinity drifted at n={n}");
            assert_eq!(
                spectral_retained_per_row(n),
                if n <= 500 { 6 } else { 7 },
                "p/min-six boundary drifted at n={n}"
            );
        }
    }

    #[test]
    fn streaming_affinity_preserves_dense_oracle_spectral_labels() {
        let embeddings = tied_signed_embeddings(64);
        let dense = dense_pruned_affinity_oracle(&embeddings);
        let mut observer = IgnoreAutomaticClusteringDiagnostics;
        let dense_labels = spectral_labels_from_affinity(
            &dense,
            &embeddings,
            Some(3),
            &never_cancel,
            &mut observer,
        )
        .unwrap();
        let streaming_labels = AutomaticClusterer
            .cluster(&embeddings, DiarizeHint::NumSpeakers(3), &never_cancel)
            .unwrap();

        assert_eq!(
            streaming_labels,
            as_speaker_ids(compact_labels(&dense_labels))
        );
    }

    #[test]
    fn one_hour_scale_retention_is_linear_in_selected_edges_and_early_cancelable() {
        let n = 4_800;
        let retained_per_row = spectral_retained_per_row(n);
        let retained_entries = n * retained_per_row;
        assert_eq!(retained_per_row, 58);
        assert_eq!(retained_entries, 278_400);
        assert!(retained_entries * 64 < n * n);

        let embeddings = tied_signed_embeddings(n);
        let checks = Cell::new(0usize);
        let canceled = || {
            let next = checks.get() + 1;
            checks.set(next);
            next >= 4
        };
        assert_eq!(
            pruned_directed_rows(&embeddings, &canceled),
            Err(AutomaticClusteringError::Canceled)
        );
        assert_eq!(checks.get(), 4, "cancel should stop within the first row");
    }

    #[test]
    fn cancellation_is_typed_across_affinity_solver_kmeans_and_postprocess() {
        let embeddings = tied_signed_embeddings(40);

        let checks = Cell::new(0usize);
        let cancel_after_first_row_sort = || {
            let next = checks.get() + 1;
            checks.set(next);
            next >= 4
        };
        assert_eq!(
            pruned_directed_rows(&embeddings[..8], &cancel_after_first_row_sort),
            Err(AutomaticClusteringError::Canceled)
        );

        let directed = pruned_directed_rows(&embeddings[..8], &never_cancel).unwrap();
        assert_eq!(
            symmetrize_pruned_affinity(&directed, &|| true),
            Err(AutomaticClusteringError::Canceled)
        );

        let affinity = symmetrize_pruned_affinity(&directed, &never_cancel).unwrap();
        assert_eq!(
            smallest_laplacian_eigenvectors(&affinity, 2, &|| true),
            Err(AutomaticClusteringError::Canceled)
        );
        assert_eq!(
            deterministic_kmeans(&[1.0, 0.0, 0.0, 1.0], 2, 2, 2, &|| true),
            Err(AutomaticClusteringError::Canceled)
        );

        let labels: Vec<_> = (0..40).map(|index| index % 2).collect();
        assert_eq!(
            filter_minor_clusters(&labels, &embeddings, 4, &|| true),
            Err(AutomaticClusteringError::Canceled)
        );
        assert_eq!(
            merge_similar_centroids(&labels, &embeddings, 0.8, &|| true),
            Err(AutomaticClusteringError::Canceled)
        );
    }

    #[test]
    fn automatic_clustering_is_deterministic() {
        let embeddings: Vec<_> = (0..48)
            .map(|index| {
                let group = index % 3;
                let mut values = vec![0.01 * index as f32, 0.0, 0.0];
                values[group] += 1.0;
                emb(values)
            })
            .collect();
        let clusterer = AutomaticClusterer;
        let expected = clusterer
            .cluster(&embeddings, DiarizeHint::Auto, &never_cancel)
            .unwrap();
        for _ in 0..4 {
            assert_eq!(
                clusterer
                    .cluster(&embeddings, DiarizeHint::Auto, &never_cancel)
                    .unwrap(),
                expected
            );
        }
    }

    #[test]
    fn deterministic_kmeans_preserves_the_requested_nonempty_count() {
        // Identical points force the empty-cluster path. The old implementation
        // copied a centroid without moving its label, then compacted k=3 to k=1.
        let points = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let labels = deterministic_kmeans(&points, 4, 2, 3, &never_cancel).unwrap();

        assert_eq!(label_count(&labels), 3);
        assert_eq!(
            labels,
            deterministic_kmeans(&points, 4, 2, 3, &never_cancel).unwrap()
        );
    }

    #[test]
    fn automatic_diagnostics_final_labels_exactly_match_production() {
        let ahc_embeddings: Vec<_> = (0..12)
            .map(|index| {
                if index < 6 {
                    emb(vec![1.0, 0.01 * index as f32, 0.0])
                } else {
                    emb(vec![0.01 * (index - 6) as f32, 1.0, 0.0])
                }
            })
            .collect();
        let spectral_embeddings: Vec<_> = (0..48)
            .map(|index| {
                let group = index % 3;
                let mut values = vec![0.01 * index as f32, 0.0, 0.0];
                values[group] += 1.0;
                emb(values)
            })
            .collect();
        let clusterer = AutomaticClusterer;

        let ahc = clusterer
            .diagnostics(&ahc_embeddings, DiarizeHint::Auto, &never_cancel)
            .unwrap();
        assert_eq!(
            ahc.final_labels,
            clusterer
                .cluster(&ahc_embeddings, DiarizeHint::Auto, &never_cancel)
                .unwrap()
        );
        assert_eq!(ahc.strategy, AutomaticClusteringStrategy::Ahc);
        assert!(ahc.spectral_eigenvalues.is_empty());
        assert_eq!(ahc.eigengap_speakers, None);
        assert_eq!(ahc.selected_speakers, 2);
        assert_eq!(ahc.raw_labels.len(), ahc_embeddings.len());
        assert_eq!(ahc.minor_filtered_labels.len(), ahc_embeddings.len());

        let spectral = clusterer
            .diagnostics(&spectral_embeddings, DiarizeHint::Auto, &never_cancel)
            .unwrap();
        assert_eq!(
            spectral.final_labels,
            clusterer
                .cluster(&spectral_embeddings, DiarizeHint::Auto, &never_cancel)
                .unwrap()
        );
        assert_eq!(spectral.strategy, AutomaticClusteringStrategy::Spectral);
        assert_eq!(
            spectral.spectral_eigenvalues.len(),
            SPECTRAL_MAX_SPEAKERS + 1
        );
        assert_eq!(spectral.eigengap_speakers, Some(spectral.selected_speakers));
        assert_eq!(spectral.raw_labels.len(), spectral_embeddings.len());
        assert_eq!(
            spectral.minor_filtered_labels.len(),
            spectral_embeddings.len()
        );
    }

    #[test]
    fn automatic_diagnostics_distinguish_forced_and_degenerate_paths() {
        let embeddings: Vec<_> = (0..40)
            .map(|index| {
                if index < 20 {
                    emb(vec![1.0, 0.01 * index as f32])
                } else {
                    emb(vec![0.01 * (index - 20) as f32, 1.0])
                }
            })
            .collect();
        let clusterer = AutomaticClusterer;

        let forced = clusterer
            .diagnostics(&embeddings, DiarizeHint::NumSpeakers(2), &never_cancel)
            .unwrap();
        assert_eq!(forced.strategy, AutomaticClusteringStrategy::Spectral);
        assert_eq!(forced.spectral_eigenvalues.len(), 2);
        assert_eq!(forced.eigengap_speakers, None);
        assert_eq!(forced.selected_speakers, 2);
        assert_eq!(forced.raw_labels, forced.minor_filtered_labels);
        assert_eq!(
            forced.final_labels,
            clusterer
                .cluster(&embeddings, DiarizeHint::NumSpeakers(2), &never_cancel,)
                .unwrap()
        );

        let singleton = vec![emb(vec![1.0, 0.0])];
        let short = clusterer
            .diagnostics(&singleton, DiarizeHint::Auto, &never_cancel)
            .unwrap();
        assert_eq!(short.strategy, AutomaticClusteringStrategy::Ahc);
        assert!(short.spectral_eigenvalues.is_empty());
        assert_eq!(short.eigengap_speakers, None);
        assert_eq!(short.selected_speakers, 1);
        assert_eq!(short.raw_labels, vec![SpeakerId(0)]);
        assert_eq!(short.minor_filtered_labels, short.raw_labels);
        assert_eq!(short.final_labels, short.raw_labels);
    }

    #[test]
    fn forced_spectral_clustering_uses_the_selected_eigenvector_count() {
        let embeddings: Vec<_> = (0..40)
            .map(|index| {
                if index < 20 {
                    emb(vec![1.0, 0.01 * index as f32])
                } else {
                    emb(vec![0.01 * (index - 20) as f32, 1.0])
                }
            })
            .collect();
        let labels = AutomaticClusterer
            .cluster(&embeddings, DiarizeHint::NumSpeakers(2), &never_cancel)
            .unwrap();
        assert!(labels[..20].iter().all(|label| *label == labels[0]));
        assert!(labels[20..].iter().all(|label| *label == labels[20]));
        assert_ne!(labels[0], labels[20]);
    }

    #[test]
    fn minor_clusters_reassign_before_centroid_merge() {
        let mut embeddings = vec![emb(vec![1.0, 0.0]); 5];
        embeddings.extend(vec![emb(vec![0.0, 1.0]); 5]);
        embeddings.push(emb(vec![0.98, 0.02]));
        let labels = vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2];
        let filtered = filter_minor_clusters(&labels, &embeddings, 4, &never_cancel).unwrap();
        assert_eq!(filtered[10], 0);
    }
}
