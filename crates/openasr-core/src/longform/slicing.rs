use thiserror::Error;

use super::options::{LongFormMode, LongFormOptions};
use super::timeline::{TimelineAnchor, TimelineMap};
use super::vad::EnergyLongFormVadProvider;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioSliceKind {
    Full,
    Fixed,
    Energy,
    Vad,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AudioSlice {
    pub index: usize,
    pub kind: AudioSliceKind,
    pub start_sample: usize,
    pub end_sample: usize,
    pub content_start_sample: usize,
    pub content_end_sample: usize,
}

impl AudioSlice {
    pub fn duration_samples(&self) -> usize {
        self.end_sample.saturating_sub(self.start_sample)
    }

    pub fn content_duration_samples(&self) -> usize {
        self.content_end_sample
            .saturating_sub(self.content_start_sample)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LongFormVadSlice {
    pub start_sample: usize,
    pub end_sample: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LongFormVadProviderKind {
    Custom,
    EnergyLike,
}

pub trait LongFormVadProvider: Send + Sync {
    fn provider_kind(&self) -> LongFormVadProviderKind {
        LongFormVadProviderKind::Custom
    }

    fn compute_speech_slices(
        &self,
        samples: &[f32],
        sample_rate_hz: u32,
        options: &LongFormOptions,
    ) -> Result<Vec<LongFormVadSlice>, String>;
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LongFormSliceStats {
    pub chunk_count: usize,
    pub skipped_silent_chunks: usize,
    pub duplicate_merge_count: usize,
    pub provenance: Vec<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LongFormBenchmarkMetadata {
    pub chunk_count: usize,
    pub skipped_silent_chunks: usize,
    pub duplicate_merge_count: usize,
    pub provenance: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LongFormSlicePlan {
    pub sample_rate_hz: u32,
    pub total_samples: usize,
    pub slices: Vec<AudioSlice>,
    pub processed_audio: Option<Vec<f32>>,
    pub timeline: TimelineMap,
    pub stats: LongFormSliceStats,
}

#[derive(Debug, Clone, PartialEq)]
struct LongFormPlanningLayout {
    slices: Vec<AudioSlice>,
    processed_audio: Option<Vec<f32>>,
    packed_audio_plan: Option<PackedAudioMaterializationPlan>,
    timeline: TimelineMap,
    selection_provenance: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PackedAudioMaterializationPlan {
    spans: Vec<LongFormVadSlice>,
    seam_samples: usize,
    processed_samples: usize,
}

#[derive(Debug, Error, Clone, PartialEq)]
pub enum LongFormSliceError {
    #[error("longform sample_rate_hz must be > 0")]
    InvalidSampleRate,
    #[error("longform options are invalid: {reason}")]
    InvalidOptions { reason: String },
    #[error("longform mode 'vad' requested but no VAD provider is configured")]
    VadUnavailable,
    #[error("longform VAD provider failed: {reason}")]
    VadFailed { reason: String },
}

pub fn plan_longform_slices(
    samples: &[f32],
    sample_rate_hz: u32,
    options: &LongFormOptions,
    vad_provider: Option<&dyn LongFormVadProvider>,
) -> Result<LongFormSlicePlan, LongFormSliceError> {
    if sample_rate_hz == 0 {
        return Err(LongFormSliceError::InvalidSampleRate);
    }
    options
        .validate()
        .map_err(|error| LongFormSliceError::InvalidOptions {
            reason: error.to_string(),
        })?;
    if samples.is_empty() {
        return Ok(LongFormSlicePlan {
            sample_rate_hz,
            total_samples: 0,
            slices: Vec::new(),
            processed_audio: None,
            timeline: TimelineMap::identity(),
            stats: LongFormSliceStats::default(),
        });
    }
    let total_samples = samples.len();
    let mut layout = match options.mode {
        LongFormMode::Off => layout_from_identity_slices(vec![full_slice(total_samples)]),
        LongFormMode::Fixed => {
            layout_from_identity_slices(plan_fixed_slices(total_samples, sample_rate_hz, options))
        }
        LongFormMode::Energy => {
            layout_from_identity_slices(plan_energy_slices(samples, sample_rate_hz, options))
        }
        LongFormMode::Vad => layout_from_identity_slices(plan_vad_slices(
            samples,
            sample_rate_hz,
            options,
            vad_provider,
        )?),
        LongFormMode::Auto => plan_auto_slices(samples, sample_rate_hz, options, vad_provider)?,
    };
    if layout.processed_audio.is_none() {
        if let Some(materialization_plan) = layout.packed_audio_plan.take() {
            layout.processed_audio = Some(materialize_packed_audio(samples, &materialization_plan));
        } else {
            apply_padding(
                &mut layout.slices,
                total_samples,
                sample_rate_hz,
                options.padding_seconds,
            );
        }
    }
    let stats = LongFormSliceStats {
        chunk_count: layout.slices.len(),
        skipped_silent_chunks: 0,
        duplicate_merge_count: 0,
        provenance: layout.selection_provenance.clone(),
    };
    Ok(LongFormSlicePlan {
        sample_rate_hz,
        total_samples,
        slices: layout.slices,
        processed_audio: layout.processed_audio,
        timeline: layout.timeline,
        stats,
    })
}

fn layout_from_identity_slices(slices: Vec<AudioSlice>) -> LongFormPlanningLayout {
    LongFormPlanningLayout {
        slices,
        processed_audio: None,
        packed_audio_plan: None,
        timeline: TimelineMap::identity(),
        selection_provenance: Vec::new(),
    }
}

fn layout_uses_packed_timeline(layout: &LongFormPlanningLayout) -> bool {
    layout.processed_audio.is_some() || layout.packed_audio_plan.is_some()
}

fn materialize_packed_audio(samples: &[f32], plan: &PackedAudioMaterializationPlan) -> Vec<f32> {
    let mut processed_audio = Vec::with_capacity(plan.processed_samples);
    for (index, span) in plan.spans.iter().enumerate() {
        if index > 0 {
            processed_audio.resize(processed_audio.len() + plan.seam_samples, 0.0);
        }
        processed_audio.extend_from_slice(&samples[span.start_sample..span.end_sample]);
    }
    processed_audio
}

fn full_slice(total_samples: usize) -> AudioSlice {
    AudioSlice {
        index: 0,
        kind: AudioSliceKind::Full,
        start_sample: 0,
        end_sample: total_samples,
        content_start_sample: 0,
        content_end_sample: total_samples,
    }
}

fn plan_fixed_slices(
    total_samples: usize,
    sample_rate_hz: u32,
    options: &LongFormOptions,
) -> Vec<AudioSlice> {
    let chunk_samples = seconds_to_samples(options.chunk_seconds, sample_rate_hz);
    let overlap_samples = seconds_to_samples(options.overlap_seconds, sample_rate_hz);
    let min_chunk_samples = seconds_to_samples(options.min_chunk_seconds, sample_rate_hz);
    let step = chunk_samples.saturating_sub(overlap_samples).max(1);
    let mut start = 0usize;
    let mut slices: Vec<AudioSlice> = Vec::new();
    while start < total_samples {
        let end = (start + chunk_samples).min(total_samples);
        if end.saturating_sub(start) < min_chunk_samples && !slices.is_empty() {
            let last = slices.last_mut().expect("non-empty");
            last.content_end_sample = total_samples;
            break;
        }
        slices.push(AudioSlice {
            index: slices.len(),
            kind: AudioSliceKind::Fixed,
            start_sample: start,
            end_sample: end,
            content_start_sample: start,
            content_end_sample: end,
        });
        if end == total_samples {
            break;
        }
        let next = start + step;
        if next <= start {
            start += 1;
        } else {
            start = next;
        }
    }
    slices
}

fn plan_auto_slices(
    samples: &[f32],
    sample_rate_hz: u32,
    options: &LongFormOptions,
    vad_provider: Option<&dyn LongFormVadProvider>,
) -> Result<LongFormPlanningLayout, LongFormSliceError> {
    let total_samples = samples.len();
    let chunk_samples = seconds_to_samples(options.chunk_seconds, sample_rate_hz);
    if total_samples <= chunk_samples {
        return Ok(layout_from_identity_slices(vec![full_slice(total_samples)]));
    }

    let mut candidates = Vec::with_capacity(3);
    candidates.push(build_auto_plan_candidate(
        AudioSliceKind::Energy,
        layout_from_identity_slices(plan_energy_slices(samples, sample_rate_hz, options)),
        samples,
        total_samples,
        sample_rate_hz,
        options,
    ));
    if let Some(packed_energy_layout) = plan_packed_energy_layout(samples, sample_rate_hz, options)
    {
        candidates.push(build_auto_plan_candidate(
            AudioSliceKind::Energy,
            packed_energy_layout,
            samples,
            total_samples,
            sample_rate_hz,
            options,
        ));
    }

    let fixed_slices = plan_fixed_slices(total_samples, sample_rate_hz, options);
    if !fixed_slices.is_empty() {
        candidates.push(build_auto_plan_candidate(
            AudioSliceKind::Fixed,
            layout_from_identity_slices(fixed_slices),
            samples,
            total_samples,
            sample_rate_hz,
            options,
        ));
    }

    if let Some(provider) = vad_provider
        && provider.provider_kind() != LongFormVadProviderKind::EnergyLike
    {
        let vad_spans = provider
            .compute_speech_slices(samples, sample_rate_hz, options)
            .map_err(|reason| LongFormSliceError::VadFailed { reason })?;
        if let Some(packed_vad_layout) = plan_packed_layout_from_speech_spans(
            samples,
            sample_rate_hz,
            options,
            AudioSliceKind::Vad,
            vad_spans.clone(),
        ) {
            candidates.push(build_auto_plan_candidate(
                AudioSliceKind::Vad,
                packed_vad_layout,
                samples,
                total_samples,
                sample_rate_hz,
                options,
            ));
        }
        let vad_slices =
            plan_vad_slices_from_speech_spans(samples, sample_rate_hz, options, vad_spans);
        if !vad_slices.is_empty() {
            candidates.push(build_auto_plan_candidate(
                AudioSliceKind::Vad,
                layout_from_identity_slices(vad_slices),
                samples,
                total_samples,
                sample_rate_hz,
                options,
            ));
        }
    }

    prune_dominated_vad_candidates(&mut candidates);
    let mut selection_provenance =
        apply_marginal_packed_penalties(&mut candidates, total_samples, sample_rate_hz, options);
    selection_provenance.extend(apply_marginal_vad_penalties(
        &mut candidates,
        total_samples,
        sample_rate_hz,
        options,
    ));
    selection_provenance.extend(apply_material_vad_boundary_credits(
        &mut candidates,
        sample_rate_hz,
        options,
    ));
    candidates.sort_by(compare_auto_plan_candidates);
    selection_provenance.extend(auto_selection_provenance(&candidates));
    Ok(candidates
        .into_iter()
        .next()
        .map(|mut candidate| {
            candidate.layout.selection_provenance = selection_provenance;
            candidate.layout
        })
        .unwrap_or_else(|| layout_from_identity_slices(vec![full_slice(total_samples)])))
}

fn plan_energy_slices(
    samples: &[f32],
    sample_rate_hz: u32,
    options: &LongFormOptions,
) -> Vec<AudioSlice> {
    plan_energy_slices_contiguous(samples, 0, sample_rate_hz, options)
}

fn plan_packed_energy_layout(
    samples: &[f32],
    sample_rate_hz: u32,
    options: &LongFormOptions,
) -> Option<LongFormPlanningLayout> {
    let provider = EnergyLongFormVadProvider;
    let speech_spans = provider
        .compute_speech_slices(samples, sample_rate_hz, options)
        .ok()?;
    plan_packed_layout_from_speech_spans(
        samples,
        sample_rate_hz,
        options,
        AudioSliceKind::Energy,
        speech_spans,
    )
}

fn plan_packed_layout_from_speech_spans(
    samples: &[f32],
    sample_rate_hz: u32,
    options: &LongFormOptions,
    kind: AudioSliceKind,
    speech_spans: Vec<LongFormVadSlice>,
) -> Option<LongFormPlanningLayout> {
    if speech_spans.len() < 2 {
        return None;
    }
    let target_chunk_samples = seconds_to_samples(options.chunk_seconds, sample_rate_hz).max(1);
    let min_chunk_samples = seconds_to_samples(options.min_chunk_seconds, sample_rate_hz).max(1);
    let gap_bridge_samples = seconds_to_samples(vad_coalesce_gap_seconds(options), sample_rate_hz);
    let keep_spans = coalesce_vad_slices(
        speech_spans,
        target_chunk_samples,
        min_chunk_samples,
        gap_bridge_samples,
        samples.len(),
    );
    let pad_samples = seconds_to_samples(options.padding_seconds, sample_rate_hz);
    let padded_spans = expand_and_merge_keep_spans(keep_spans, samples.len(), pad_samples);
    let (processed_audio, timeline, packed_spans) = build_packed_audio_materialization_plan(
        &padded_spans,
        samples.len(),
        sample_rate_hz,
        options,
    )?;
    let mut packed_options = options.clone();
    packed_options.padding_seconds = 0.0;
    let packed_windows =
        pack_processed_spans_into_windows(&packed_spans, sample_rate_hz, &packed_options);
    let slices: Vec<AudioSlice> = packed_windows
        .into_iter()
        .enumerate()
        .map(|(index, window)| AudioSlice {
            index,
            kind,
            start_sample: window.start_sample,
            end_sample: window.end_sample,
            content_start_sample: window.start_sample,
            content_end_sample: window.end_sample,
        })
        .collect();
    if slices.is_empty() {
        return None;
    }
    Some(LongFormPlanningLayout {
        slices,
        processed_audio: None,
        packed_audio_plan: Some(processed_audio),
        timeline,
        selection_provenance: Vec::new(),
    })
}

fn plan_energy_slices_contiguous(
    samples: &[f32],
    start_offset: usize,
    sample_rate_hz: u32,
    options: &LongFormOptions,
) -> Vec<AudioSlice> {
    let mut slices = Vec::new();
    extend_energy_slices_for_span(
        &mut slices,
        samples,
        start_offset,
        start_offset + samples.len(),
        sample_rate_hz,
        options,
    );
    slices
}

fn extend_energy_slices_for_span(
    slices: &mut Vec<AudioSlice>,
    samples: &[f32],
    span_start: usize,
    span_end: usize,
    sample_rate_hz: u32,
    options: &LongFormOptions,
) {
    if span_end <= span_start {
        return;
    }
    let chunk_samples = seconds_to_samples(options.chunk_seconds, sample_rate_hz);
    let max_chunk_samples = seconds_to_samples(options.max_chunk_seconds, sample_rate_hz);
    let overlap_samples = seconds_to_samples(options.overlap_seconds, sample_rate_hz);
    let min_chunk_samples = seconds_to_samples(options.min_chunk_seconds, sample_rate_hz);
    let search_samples = seconds_to_samples(options.energy_split_search_seconds, sample_rate_hz);
    let total_samples = samples.len();
    let mut start = span_start.min(total_samples);
    let limit = span_end.min(total_samples);
    while start < limit {
        let desired = (start + chunk_samples).min(limit);
        let hard_end = (start + max_chunk_samples).min(limit);
        if desired == limit {
            slices.push(AudioSlice {
                index: slices.len(),
                kind: AudioSliceKind::Energy,
                start_sample: start,
                end_sample: limit,
                content_start_sample: start,
                content_end_sample: limit,
            });
            break;
        }
        let search_start = desired
            .saturating_sub(search_samples)
            .max(start + min_chunk_samples);
        let search_end = (desired + search_samples).min(hard_end);
        let split = find_lowest_energy_split(samples, search_start, search_end)
            .unwrap_or(desired)
            .max(start + min_chunk_samples)
            .min(hard_end);
        slices.push(AudioSlice {
            index: slices.len(),
            kind: AudioSliceKind::Energy,
            start_sample: start,
            end_sample: split,
            content_start_sample: start,
            content_end_sample: split,
        });
        if split >= limit {
            break;
        }
        start = split.saturating_sub(overlap_samples);
        if let Some(last) = slices.last()
            && start <= last.content_start_sample
        {
            start = last.content_end_sample;
        }
    }
}

fn plan_vad_slices(
    samples: &[f32],
    sample_rate_hz: u32,
    options: &LongFormOptions,
    vad_provider: Option<&dyn LongFormVadProvider>,
) -> Result<Vec<AudioSlice>, LongFormSliceError> {
    let Some(provider) = vad_provider else {
        if options.fallback_to_energy_when_vad_unavailable {
            return Ok(plan_energy_slices(samples, sample_rate_hz, options));
        }
        return Err(LongFormSliceError::VadUnavailable);
    };
    let vad_slices = provider
        .compute_speech_slices(samples, sample_rate_hz, options)
        .map_err(|reason| LongFormSliceError::VadFailed { reason })?;
    if vad_slices.is_empty() {
        if options.fallback_to_energy_when_vad_empty {
            return Ok(plan_energy_slices(samples, sample_rate_hz, options));
        }
        return Ok(Vec::new());
    }
    let slices = plan_vad_slices_from_speech_spans(samples, sample_rate_hz, options, vad_slices);
    if slices.is_empty() && options.fallback_to_energy_when_vad_empty {
        return Ok(plan_energy_slices(samples, sample_rate_hz, options));
    }
    Ok(slices)
}

fn plan_vad_slices_from_speech_spans(
    samples: &[f32],
    sample_rate_hz: u32,
    options: &LongFormOptions,
    vad_slices: Vec<LongFormVadSlice>,
) -> Vec<AudioSlice> {
    let max_chunk_samples = seconds_to_samples(options.chunk_seconds, sample_rate_hz);
    let min_chunk_samples = seconds_to_samples(options.min_chunk_seconds, sample_rate_hz);
    let gap_bridge_samples = seconds_to_samples(vad_coalesce_gap_seconds(options), sample_rate_hz);
    let overlap_samples = seconds_to_samples(options.overlap_seconds, sample_rate_hz);
    let coalesced_slices = coalesce_vad_slices(
        vad_slices,
        max_chunk_samples.max(1),
        min_chunk_samples.max(1),
        gap_bridge_samples,
        samples.len(),
    );
    let mut slices = Vec::new();
    for vad_slice in coalesced_slices {
        if vad_slice.end_sample <= vad_slice.start_sample {
            continue;
        }
        let mut start = vad_slice.start_sample.min(samples.len());
        let end = vad_slice.end_sample.min(samples.len());
        while start < end {
            let next_end = (start + max_chunk_samples).min(end);
            slices.push(AudioSlice {
                index: slices.len(),
                kind: AudioSliceKind::Vad,
                start_sample: start,
                end_sample: next_end,
                content_start_sample: start,
                content_end_sample: next_end,
            });
            if next_end >= end {
                break;
            }
            start = next_end.saturating_sub(overlap_samples);
            if start >= next_end {
                start = next_end;
            }
        }
    }
    slices
}

fn coalesce_vad_slices(
    mut input: Vec<LongFormVadSlice>,
    target_chunk_samples: usize,
    min_chunk_samples: usize,
    gap_bridge_samples: usize,
    total_samples: usize,
) -> Vec<LongFormVadSlice> {
    if input.is_empty() {
        return input;
    }
    input.sort_by_key(|slice| slice.start_sample);
    let mut out = Vec::with_capacity(input.len());
    let mut current = LongFormVadSlice {
        start_sample: input[0].start_sample.min(total_samples),
        end_sample: input[0].end_sample.min(total_samples),
    };
    for next in input.into_iter().skip(1) {
        let next_start = next.start_sample.min(total_samples);
        let next_end = next.end_sample.min(total_samples);
        if next_end <= next_start {
            continue;
        }
        let current_len = current.end_sample.saturating_sub(current.start_sample);
        let merged_len = next_end.saturating_sub(current.start_sample);
        let gap = next_start.saturating_sub(current.end_sample);
        let should_merge = merged_len <= target_chunk_samples
            && (current_len < min_chunk_samples || gap <= gap_bridge_samples);
        if should_merge {
            current.end_sample = current.end_sample.max(next_end);
            continue;
        }
        if current.end_sample > current.start_sample {
            out.push(current);
        }
        current = LongFormVadSlice {
            start_sample: next_start,
            end_sample: next_end,
        };
    }
    if current.end_sample > current.start_sample {
        out.push(current);
    }
    out
}

#[derive(Debug)]
struct AutoPlanCandidate {
    kind: AudioSliceKind,
    score: u128,
    processed_samples: usize,
    short_slice_penalty: usize,
    boundary_penalty: usize,
    elision_penalty: usize,
    gap_edge_penalty: usize,
    seam_penalty: usize,
    extra_chunk_penalty: u128,
    stability_bias: u128,
    contextual_credit: u128,
    contextual_penalty: u128,
    layout: LongFormPlanningLayout,
}

fn auto_candidate_timeline_kind(candidate: &AutoPlanCandidate) -> &'static str {
    if layout_uses_packed_timeline(&candidate.layout) {
        "packed"
    } else {
        "identity"
    }
}

fn auto_candidate_label(candidate: &AutoPlanCandidate) -> String {
    let kind = match candidate.kind {
        AudioSliceKind::Full => "full",
        AudioSliceKind::Fixed => "fixed",
        AudioSliceKind::Energy => "energy",
        AudioSliceKind::Vad => "vad",
    };
    format!("{kind}-{}", auto_candidate_timeline_kind(candidate))
}

fn auto_selection_provenance(candidates: &[AutoPlanCandidate]) -> Vec<String> {
    let mut provenance = Vec::with_capacity(candidates.len().min(4) + 1);
    if let Some(selected) = candidates.first() {
        provenance.push(format!(
            "core.longform.auto.selected:{}:score={}:processed_samples={}:chunks={}:short_penalty={}:boundary_penalty={}:elision_penalty={}:gap_edge_penalty={}:seam_penalty={}:chunk_penalty={}:stability_bias={}:contextual_credit={}:contextual_penalty={}",
            auto_candidate_label(selected),
            selected.score,
            selected.processed_samples,
            selected.layout.slices.len(),
            selected.short_slice_penalty,
            selected.boundary_penalty,
            selected.elision_penalty,
            selected.gap_edge_penalty,
            selected.seam_penalty,
            selected.extra_chunk_penalty,
            selected.stability_bias,
            selected.contextual_credit,
            selected.contextual_penalty,
        ));
    }
    for (index, candidate) in candidates.iter().take(3).enumerate() {
        provenance.push(format!(
            "core.longform.auto.candidate[{index}]:{}:score={}:processed_samples={}:chunks={}:short_penalty={}:boundary_penalty={}:elision_penalty={}:gap_edge_penalty={}:seam_penalty={}:chunk_penalty={}:stability_bias={}:contextual_credit={}:contextual_penalty={}",
            auto_candidate_label(candidate),
            candidate.score,
            candidate.processed_samples,
            candidate.layout.slices.len(),
            candidate.short_slice_penalty,
            candidate.boundary_penalty,
            candidate.elision_penalty,
            candidate.gap_edge_penalty,
            candidate.seam_penalty,
            candidate.extra_chunk_penalty,
            candidate.stability_bias,
            candidate.contextual_credit,
            candidate.contextual_penalty,
        ));
    }
    provenance
}

fn build_auto_plan_candidate(
    kind: AudioSliceKind,
    layout: LongFormPlanningLayout,
    samples: &[f32],
    total_samples: usize,
    sample_rate_hz: u32,
    options: &LongFormOptions,
) -> AutoPlanCandidate {
    let processed_samples = estimate_layout_processed_samples(
        &layout,
        total_samples,
        sample_rate_hz,
        options.padding_seconds,
    );
    let short_slice_penalty = estimate_short_slice_penalty(&layout.slices, sample_rate_hz, options);
    let boundary_penalty = estimate_boundary_penalty(samples, &layout, sample_rate_hz, options);
    let elision_penalty = estimate_elision_penalty(samples, &layout, sample_rate_hz, options);
    let gap_edge_penalty = estimate_gap_edge_penalty(samples, &layout, sample_rate_hz, options);
    let seam_penalty = estimate_seam_penalty(&layout, sample_rate_hz, options);
    let chunk_samples = seconds_to_samples(options.chunk_seconds, sample_rate_hz).max(1) as u128;
    let per_chunk_overhead =
        (chunk_samples / 12).max(seconds_to_samples(1.0, sample_rate_hz).max(1) as u128);
    let extra_chunk_penalty = layout.slices.len().saturating_sub(1) as u128 * per_chunk_overhead;
    let stability_bias = match kind {
        AudioSliceKind::Energy => 0,
        AudioSliceKind::Vad => (chunk_samples / 64).max(1),
        AudioSliceKind::Fixed => (chunk_samples / 48).max(1),
        AudioSliceKind::Full => 0,
    };
    let score = processed_samples as u128
        + short_slice_penalty as u128
        + boundary_penalty as u128
        + elision_penalty as u128
        + gap_edge_penalty as u128
        + seam_penalty as u128
        + extra_chunk_penalty
        + stability_bias;
    AutoPlanCandidate {
        kind,
        score,
        processed_samples,
        short_slice_penalty,
        boundary_penalty,
        elision_penalty,
        gap_edge_penalty,
        seam_penalty,
        extra_chunk_penalty,
        stability_bias,
        contextual_credit: 0,
        contextual_penalty: 0,
        layout,
    }
}

fn prune_dominated_vad_candidates(candidates: &mut Vec<AutoPlanCandidate>) {
    if !candidates
        .iter()
        .any(|candidate| candidate.kind == AudioSliceKind::Vad)
    {
        return;
    }
    let energy_candidates: Vec<(bool, usize, usize, usize, usize, usize, usize, usize)> =
        candidates
            .iter()
            .filter(|candidate| candidate.kind == AudioSliceKind::Energy)
            .map(|candidate| {
                (
                    layout_uses_packed_timeline(&candidate.layout),
                    candidate.processed_samples,
                    candidate.layout.slices.len(),
                    candidate.short_slice_penalty,
                    candidate.boundary_penalty,
                    candidate.elision_penalty,
                    candidate.gap_edge_penalty,
                    candidate.seam_penalty,
                )
            })
            .collect();
    candidates.retain(|candidate| {
        if candidate.kind != AudioSliceKind::Vad {
            return true;
        }
        let packed = layout_uses_packed_timeline(&candidate.layout);
        !energy_candidates.iter().any(
            |(
                energy_packed,
                energy_processed,
                energy_chunks,
                energy_short_penalty,
                energy_boundary_penalty,
                energy_elision_penalty,
                energy_gap_edge_penalty,
                energy_seam_penalty,
            )| {
                *energy_packed == packed
                    && *energy_processed <= candidate.processed_samples
                    && *energy_chunks <= candidate.layout.slices.len()
                    && *energy_short_penalty <= candidate.short_slice_penalty
                    && *energy_boundary_penalty <= candidate.boundary_penalty
                    && *energy_elision_penalty <= candidate.elision_penalty
                    && *energy_gap_edge_penalty <= candidate.gap_edge_penalty
                    && *energy_seam_penalty <= candidate.seam_penalty
            },
        )
    });
}

fn apply_marginal_packed_penalties(
    candidates: &mut [AutoPlanCandidate],
    total_samples: usize,
    sample_rate_hz: u32,
    options: &LongFormOptions,
) -> Vec<String> {
    let mut provenance = Vec::new();
    if candidates.len() < 2 {
        return provenance;
    }
    let chunk_samples = seconds_to_samples(options.chunk_seconds, sample_rate_hz).max(1);
    let marginal_savings_threshold = (total_samples / 20).max(chunk_samples / 8);
    let identity_by_kind: Vec<(
        AudioSliceKind,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        u128,
    )> = candidates
        .iter()
        .filter(|candidate| !layout_uses_packed_timeline(&candidate.layout))
        .map(|candidate| {
            (
                candidate.kind,
                candidate.processed_samples,
                candidate.layout.slices.len(),
                candidate.boundary_penalty,
                candidate.elision_penalty,
                candidate.gap_edge_penalty,
                candidate.seam_penalty,
                candidate.score,
            )
        })
        .collect();
    for candidate in candidates.iter_mut() {
        if !layout_uses_packed_timeline(&candidate.layout) {
            continue;
        }
        let packed_chunks = candidate.layout.slices.len();
        let packed_processed = candidate.processed_samples;
        let penalty = identity_by_kind.iter().find_map(
            |(
                kind,
                identity_processed,
                identity_chunks,
                identity_boundary_penalty,
                identity_elision_penalty,
                identity_gap_edge_penalty,
                identity_seam_penalty,
                identity_score,
            )| {
                let savings = identity_processed.saturating_sub(packed_processed);
                let extra_chunk_count = packed_chunks.saturating_sub(*identity_chunks);
                let savings_threshold = if extra_chunk_count == 0 {
                    marginal_savings_threshold
                } else {
                    marginal_savings_threshold
                        .saturating_add(chunk_samples.saturating_mul(extra_chunk_count))
                };
                if *kind == candidate.kind
                    && *identity_chunks <= packed_chunks
                    && *identity_boundary_penalty <= candidate.boundary_penalty
                    && *identity_elision_penalty <= candidate.elision_penalty
                    && *identity_gap_edge_penalty <= candidate.gap_edge_penalty
                    && *identity_seam_penalty <= candidate.seam_penalty
                    && *identity_processed > packed_processed
                    && savings < savings_threshold
                {
                    Some((
                        identity_score
                            .saturating_sub(candidate.score)
                            .saturating_add(1),
                        *identity_chunks,
                        savings_threshold,
                    ))
                } else {
                    None
                }
            },
        );
        if let Some((penalty, identity_chunks, savings_threshold)) = penalty {
            candidate.contextual_penalty = candidate.contextual_penalty.saturating_add(penalty);
            candidate.score = candidate.score.saturating_add(penalty);
            provenance.push(format!(
                "core.longform.auto.penalized:{}:identity_chunks={}:packed_chunks={}:penalty={}:threshold={}",
                auto_candidate_label(candidate),
                identity_chunks,
                packed_chunks,
                penalty,
                savings_threshold,
            ));
        }
    }
    provenance
}

fn apply_marginal_vad_penalties(
    candidates: &mut [AutoPlanCandidate],
    total_samples: usize,
    sample_rate_hz: u32,
    options: &LongFormOptions,
) -> Vec<String> {
    let mut provenance = Vec::new();
    if candidates.len() < 2 {
        return provenance;
    }
    let chunk_samples = seconds_to_samples(options.chunk_seconds, sample_rate_hz).max(1);
    let marginal_savings_threshold = (total_samples / 40).max(chunk_samples / 10);
    let energy_by_timeline: Vec<(bool, usize, usize, usize, usize, usize, usize, usize, u128)> =
        candidates
            .iter()
            .filter(|candidate| candidate.kind == AudioSliceKind::Energy)
            .map(|candidate| {
                (
                    layout_uses_packed_timeline(&candidate.layout),
                    candidate.processed_samples,
                    candidate.layout.slices.len(),
                    candidate.short_slice_penalty,
                    candidate.boundary_penalty,
                    candidate.elision_penalty,
                    candidate.gap_edge_penalty,
                    candidate.seam_penalty,
                    candidate.score,
                )
            })
            .collect();
    for candidate in candidates.iter_mut() {
        if candidate.kind != AudioSliceKind::Vad {
            continue;
        }
        let packed = layout_uses_packed_timeline(&candidate.layout);
        let penalty = energy_by_timeline.iter().find_map(
            |(
                energy_packed,
                energy_processed,
                energy_chunks,
                energy_short_penalty,
                energy_boundary_penalty,
                energy_elision_penalty,
                energy_gap_edge_penalty,
                energy_seam_penalty,
                energy_score,
            )| {
                let savings = energy_processed.saturating_sub(candidate.processed_samples);
                if *energy_packed == packed
                    && *energy_chunks == candidate.layout.slices.len()
                    && *energy_short_penalty <= candidate.short_slice_penalty
                    && *energy_boundary_penalty <= candidate.boundary_penalty
                    && *energy_elision_penalty <= candidate.elision_penalty
                    && *energy_gap_edge_penalty <= candidate.gap_edge_penalty
                    && *energy_seam_penalty <= candidate.seam_penalty
                    && *energy_processed > candidate.processed_samples
                    && savings < marginal_savings_threshold
                {
                    Some(
                        energy_score
                            .saturating_sub(candidate.score)
                            .saturating_add(1),
                    )
                } else {
                    None
                }
            },
        );
        if let Some(penalty) = penalty {
            candidate.contextual_penalty = candidate.contextual_penalty.saturating_add(penalty);
            candidate.score = candidate.score.saturating_add(penalty);
            provenance.push(format!(
                "core.longform.auto.penalized:{}:marginal_vad_savings_below_threshold:same_chunks={}:penalty={}:threshold={}",
                auto_candidate_label(candidate),
                candidate.layout.slices.len(),
                penalty,
                marginal_savings_threshold,
            ));
        }
    }
    provenance
}

fn apply_material_vad_boundary_credits(
    candidates: &mut [AutoPlanCandidate],
    sample_rate_hz: u32,
    options: &LongFormOptions,
) -> Vec<String> {
    let mut provenance = Vec::new();
    if candidates.len() < 2 {
        return provenance;
    }
    let chunk_samples = seconds_to_samples(options.chunk_seconds, sample_rate_hz).max(1);
    let boundary_gain_threshold = (chunk_samples / 1920).max((sample_rate_hz as usize) / 80);
    let energy_by_timeline: Vec<(bool, usize, usize, usize, usize, usize)> = candidates
        .iter()
        .filter(|candidate| candidate.kind == AudioSliceKind::Energy)
        .map(|candidate| {
            (
                layout_uses_packed_timeline(&candidate.layout),
                candidate.layout.slices.len(),
                candidate.boundary_penalty,
                candidate.gap_edge_penalty,
                candidate.short_slice_penalty,
                candidate.seam_penalty,
            )
        })
        .collect();
    for candidate in candidates.iter_mut() {
        if candidate.kind != AudioSliceKind::Vad {
            continue;
        }
        let packed = layout_uses_packed_timeline(&candidate.layout);
        let chunk_count = candidate.layout.slices.len();
        let candidate_boundary_cost = candidate
            .boundary_penalty
            .saturating_add(candidate.gap_edge_penalty);
        let credit = energy_by_timeline
            .iter()
            .filter_map(
                |(
                    energy_packed,
                    energy_chunk_count,
                    energy_boundary_penalty,
                    energy_gap_edge_penalty,
                    energy_short_penalty,
                    energy_seam_penalty,
                )| {
                    if *energy_packed != packed || *energy_chunk_count != chunk_count {
                        return None;
                    }
                    let energy_boundary_cost =
                        energy_boundary_penalty.saturating_add(*energy_gap_edge_penalty);
                    let boundary_gain =
                        energy_boundary_cost.saturating_sub(candidate_boundary_cost);
                    let topology_overhead = candidate
                        .short_slice_penalty
                        .saturating_sub(*energy_short_penalty)
                        .saturating_add(
                            candidate.seam_penalty.saturating_sub(*energy_seam_penalty),
                        );
                    let net_quality_gain = boundary_gain.saturating_sub(topology_overhead);
                    if net_quality_gain < boundary_gain_threshold {
                        return None;
                    }
                    Some(net_quality_gain.saturating_sub(boundary_gain_threshold / 2) as u128)
                },
            )
            .max();
        if let Some(credit) = credit
            && credit > 0
        {
            candidate.contextual_credit = candidate.contextual_credit.saturating_add(credit);
            candidate.score = candidate.score.saturating_sub(credit);
            provenance.push(format!(
                "core.longform.auto.rewarded:{}:material_boundary_gain:same_chunks={}:credit={}:threshold={}",
                auto_candidate_label(candidate),
                chunk_count,
                credit,
                boundary_gain_threshold,
            ));
        }
    }
    provenance
}

fn compare_auto_plan_candidates(
    left: &AutoPlanCandidate,
    right: &AutoPlanCandidate,
) -> std::cmp::Ordering {
    left.score
        .cmp(&right.score)
        .then_with(|| left.processed_samples.cmp(&right.processed_samples))
        .then_with(|| left.short_slice_penalty.cmp(&right.short_slice_penalty))
        .then_with(|| left.layout.slices.len().cmp(&right.layout.slices.len()))
        .then_with(|| auto_plan_kind_rank(left.kind).cmp(&auto_plan_kind_rank(right.kind)))
}

fn auto_plan_kind_rank(kind: AudioSliceKind) -> u8 {
    match kind {
        AudioSliceKind::Energy => 0,
        AudioSliceKind::Vad => 1,
        AudioSliceKind::Fixed => 2,
        AudioSliceKind::Full => 3,
    }
}

fn estimate_layout_processed_samples(
    layout: &LongFormPlanningLayout,
    total_samples: usize,
    sample_rate_hz: u32,
    padding_seconds: f32,
) -> usize {
    if let Some(processed_audio) = layout.processed_audio.as_ref() {
        return processed_audio.len();
    }
    if let Some(processed_audio) = layout.packed_audio_plan.as_ref() {
        let sliced_samples: usize = layout.slices.iter().map(AudioSlice::duration_samples).sum();
        return sliced_samples.max(processed_audio.processed_samples);
    }
    {
        let mut estimated = layout.slices.clone();
        apply_padding(
            &mut estimated,
            total_samples,
            sample_rate_hz,
            padding_seconds,
        );
        estimated.iter().map(AudioSlice::duration_samples).sum()
    }
}

fn expand_and_merge_keep_spans(
    spans: Vec<LongFormVadSlice>,
    total_samples: usize,
    pad_samples: usize,
) -> Vec<LongFormVadSlice> {
    let mut expanded: Vec<LongFormVadSlice> = Vec::with_capacity(spans.len());
    for span in spans {
        if span.end_sample <= span.start_sample {
            continue;
        }
        let start_sample = span.start_sample.saturating_sub(pad_samples);
        let end_sample = (span.end_sample + pad_samples).min(total_samples);
        if let Some(previous) = expanded.last_mut()
            && start_sample <= previous.end_sample
        {
            previous.end_sample = previous.end_sample.max(end_sample);
            continue;
        }
        expanded.push(LongFormVadSlice {
            start_sample,
            end_sample,
        });
    }
    expanded
}

fn build_packed_audio_materialization_plan(
    spans: &[LongFormVadSlice],
    total_samples: usize,
    sample_rate_hz: u32,
    options: &LongFormOptions,
) -> Option<(
    PackedAudioMaterializationPlan,
    TimelineMap,
    Vec<LongFormVadSlice>,
)> {
    if spans.is_empty() {
        return None;
    }
    let seam_seconds = options.padding_seconds.clamp(0.05, 0.20);
    let seam_samples = seconds_to_samples(seam_seconds, sample_rate_hz);
    let mut processed_spans = Vec::with_capacity(spans.len());
    let mut anchors = Vec::with_capacity(spans.len() * 3);
    let mut previous_original_end = 0usize;
    let mut cursor = 0usize;
    for (index, span) in spans.iter().enumerate() {
        if span.end_sample <= span.start_sample || span.end_sample > total_samples {
            continue;
        }
        if index == 0 {
            anchors.push(timeline_anchor_from_samples(
                0,
                span.start_sample,
                sample_rate_hz,
            ));
        } else {
            anchors.push(timeline_anchor_from_samples(
                cursor,
                previous_original_end,
                sample_rate_hz,
            ));
            cursor += seam_samples;
            anchors.push(timeline_anchor_from_samples(
                cursor,
                span.start_sample,
                sample_rate_hz,
            ));
        }
        let processed_start = cursor;
        cursor += span.end_sample.saturating_sub(span.start_sample);
        processed_spans.push(LongFormVadSlice {
            start_sample: processed_start,
            end_sample: cursor,
        });
        anchors.push(timeline_anchor_from_samples(
            cursor,
            span.end_sample,
            sample_rate_hz,
        ));
        previous_original_end = span.end_sample;
    }
    if cursor == 0 {
        return None;
    }
    Some((
        PackedAudioMaterializationPlan {
            spans: spans.to_vec(),
            seam_samples,
            processed_samples: cursor,
        },
        TimelineMap::from_anchors(anchors),
        processed_spans,
    ))
}

fn pack_processed_spans_into_windows(
    spans: &[LongFormVadSlice],
    sample_rate_hz: u32,
    options: &LongFormOptions,
) -> Vec<LongFormVadSlice> {
    if spans.is_empty() {
        return Vec::new();
    }
    let target_chunk_samples = seconds_to_samples(options.chunk_seconds, sample_rate_hz).max(1);
    let min_chunk_samples = seconds_to_samples(options.min_chunk_seconds, sample_rate_hz).max(1);
    let overlap_samples = seconds_to_samples(options.overlap_seconds, sample_rate_hz)
        .min(target_chunk_samples.saturating_sub(1));
    let mut windows = Vec::new();
    let mut current_start = spans[0].start_sample;
    let mut current_end = spans[0].end_sample;
    for span in spans.iter().skip(1) {
        let prospective_end = span.end_sample;
        let prospective_len = prospective_end.saturating_sub(current_start);
        let current_len = current_end.saturating_sub(current_start);
        if prospective_len > target_chunk_samples && current_len >= min_chunk_samples {
            windows.push(LongFormVadSlice {
                start_sample: current_start,
                end_sample: current_end,
            });
            current_start = current_end.saturating_sub(overlap_samples);
        }
        current_end = prospective_end;
    }
    windows.push(LongFormVadSlice {
        start_sample: current_start,
        end_sample: current_end,
    });
    windows
}

fn timeline_anchor_from_samples(
    processed_sample: usize,
    original_sample: usize,
    sample_rate_hz: u32,
) -> TimelineAnchor {
    TimelineAnchor {
        processed_seconds: processed_sample as f32 / sample_rate_hz as f32,
        original_seconds: original_sample as f32 / sample_rate_hz as f32,
    }
}

fn estimate_short_slice_penalty(
    slices: &[AudioSlice],
    sample_rate_hz: u32,
    options: &LongFormOptions,
) -> usize {
    let chunk_samples = seconds_to_samples(options.chunk_seconds, sample_rate_hz).max(1);
    let min_desired = chunk_samples
        .saturating_div(2)
        .max(seconds_to_samples(options.min_chunk_seconds, sample_rate_hz).saturating_mul(2));
    slices
        .iter()
        .map(|slice| min_desired.saturating_sub(slice.content_duration_samples().min(min_desired)))
        .sum()
}

fn estimate_boundary_penalty(
    samples: &[f32],
    layout: &LongFormPlanningLayout,
    sample_rate_hz: u32,
    options: &LongFormOptions,
) -> usize {
    if layout.slices.len() <= 1 || samples.is_empty() {
        return 0;
    }
    let chunk_samples = seconds_to_samples(options.chunk_seconds, sample_rate_hz).max(1);
    let window_samples = seconds_to_samples(0.20, sample_rate_hz)
        .max(seconds_to_samples(
            options.overlap_seconds.min(0.20),
            sample_rate_hz,
        ))
        .max(1);
    let per_boundary_scale = (chunk_samples / 8).max(window_samples);
    layout
        .slices
        .iter()
        .take(layout.slices.len().saturating_sub(1))
        .map(|slice| {
            let processed_seconds = slice.content_end_sample as f32 / sample_rate_hz as f32;
            let original_seconds = layout
                .timeline
                .map_processed_to_original_seconds(processed_seconds);
            let boundary_sample =
                (original_seconds * sample_rate_hz as f32).round().max(0.0) as usize;
            let half_window = window_samples / 2;
            let start = boundary_sample
                .saturating_sub(half_window)
                .min(samples.len());
            let end = (boundary_sample + half_window).min(samples.len());
            if end <= start {
                return 0usize;
            }
            let boundary_rms = rms(&samples[start..end]);
            (boundary_rms * per_boundary_scale as f32).round() as usize
        })
        .sum()
}

fn estimate_elision_penalty(
    samples: &[f32],
    layout: &LongFormPlanningLayout,
    _sample_rate_hz: u32,
    options: &LongFormOptions,
) -> usize {
    let Some(plan) = layout.packed_audio_plan.as_ref() else {
        return 0;
    };
    if plan.spans.len() < 2 || samples.is_empty() {
        return 0;
    }
    let silence_threshold_linear = 10.0_f32.powf(options.energy_silence_threshold_db / 20.0);
    plan.spans
        .windows(2)
        .map(|window| {
            let gap_start = window[0].end_sample.min(samples.len());
            let gap_end = window[1].start_sample.min(samples.len());
            if gap_end <= gap_start {
                return 0usize;
            }
            let gap_rms = rms(&samples[gap_start..gap_end]);
            if gap_rms <= silence_threshold_linear {
                return 0usize;
            }
            let gap_len = gap_end.saturating_sub(gap_start);
            let excess_ratio = (gap_rms / silence_threshold_linear).max(1.0) - 1.0;
            (excess_ratio * gap_len as f32).round() as usize
        })
        .sum()
}

fn estimate_gap_edge_penalty(
    samples: &[f32],
    layout: &LongFormPlanningLayout,
    sample_rate_hz: u32,
    options: &LongFormOptions,
) -> usize {
    let Some(plan) = layout.packed_audio_plan.as_ref() else {
        return 0;
    };
    if plan.spans.len() < 2 || samples.is_empty() {
        return 0;
    }
    let silence_threshold_linear = 10.0_f32.powf(options.energy_silence_threshold_db / 20.0);
    let edge_window = seconds_to_samples(0.15, sample_rate_hz).max(1);
    let chunk_samples = seconds_to_samples(options.chunk_seconds, sample_rate_hz).max(1);
    let per_edge_scale = (chunk_samples / 16).max(edge_window);
    plan.spans
        .windows(2)
        .map(|window| {
            let gap_start = window[0].end_sample.min(samples.len());
            let gap_end = window[1].start_sample.min(samples.len());
            if gap_end <= gap_start {
                return 0usize;
            }
            let gap_len = gap_end.saturating_sub(gap_start);
            let edge_len = edge_window.min(gap_len.max(1) / 2).max(1);
            let left_end = (gap_start + edge_len).min(samples.len());
            let right_start = gap_end.saturating_sub(edge_len).min(samples.len());
            if left_end <= gap_start || gap_end <= right_start {
                return 0usize;
            }
            let left_rms = rms(&samples[gap_start..left_end]);
            let right_rms = rms(&samples[right_start..gap_end]);
            let left_excess = (left_rms / silence_threshold_linear).max(1.0) - 1.0;
            let right_excess = (right_rms / silence_threshold_linear).max(1.0) - 1.0;
            ((left_excess + right_excess) * per_edge_scale as f32).round() as usize
        })
        .sum()
}

fn estimate_seam_penalty(
    layout: &LongFormPlanningLayout,
    sample_rate_hz: u32,
    options: &LongFormOptions,
) -> usize {
    let Some(plan) = layout.packed_audio_plan.as_ref() else {
        return 0;
    };
    let seam_count = plan.spans.len().saturating_sub(layout.slices.len());
    if seam_count == 0 {
        return 0;
    }
    let chunk_samples = seconds_to_samples(options.chunk_seconds, sample_rate_hz).max(1);
    let per_seam_penalty = (chunk_samples / 48)
        .max(plan.seam_samples)
        .max(seconds_to_samples(0.10, sample_rate_hz));
    seam_count.saturating_mul(per_seam_penalty)
}

fn vad_coalesce_gap_seconds(options: &LongFormOptions) -> f32 {
    let detector_gap_seconds = options.vad.min_silence_duration_ms as f32 / 1000.0;
    let packing_gap_seconds = (options.chunk_seconds * 0.10).clamp(0.5, 3.0);
    detector_gap_seconds
        .max(packing_gap_seconds)
        .max(options.padding_seconds * 2.0)
        .max(options.overlap_seconds * 2.0)
}

fn apply_padding(
    slices: &mut [AudioSlice],
    total_samples: usize,
    sample_rate_hz: u32,
    padding_seconds: f32,
) {
    if slices.is_empty() {
        return;
    }
    let pad = seconds_to_samples(padding_seconds, sample_rate_hz);
    for slice in slices.iter_mut() {
        slice.start_sample = slice.content_start_sample.saturating_sub(pad);
        slice.end_sample = (slice.content_end_sample + pad).min(total_samples);
    }
}

fn find_lowest_energy_split(samples: &[f32], start: usize, end: usize) -> Option<usize> {
    if start >= end {
        return None;
    }
    let frame = 1600usize;
    let mut best_index = None;
    let mut best_energy = f32::INFINITY;
    let mut index = start;
    while index < end {
        let right = (index + frame).min(samples.len()).min(end);
        if right <= index {
            break;
        }
        let rms = rms(&samples[index..right]);
        if rms < best_energy {
            best_energy = rms;
            best_index = Some(index + (right - index) / 2);
        }
        index = right;
    }
    best_index
}

fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let mut sum = 0.0_f64;
    for sample in samples {
        let value = *sample as f64;
        sum += value * value;
    }
    (sum / samples.len() as f64).sqrt() as f32
}

fn seconds_to_samples(seconds: f32, sample_rate_hz: u32) -> usize {
    ((seconds.max(0.0)) * sample_rate_hz as f32).round() as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::longform::{LongFormMode, LongFormOptions};

    fn options_with_mode(mode: LongFormMode) -> LongFormOptions {
        LongFormOptions {
            mode,
            ..LongFormOptions::default()
        }
    }

    struct FixedVadProvider;

    impl LongFormVadProvider for FixedVadProvider {
        fn compute_speech_slices(
            &self,
            samples: &[f32],
            _sample_rate_hz: u32,
            _options: &LongFormOptions,
        ) -> Result<Vec<LongFormVadSlice>, String> {
            let end = samples.len().min(16_000);
            Ok(vec![LongFormVadSlice {
                start_sample: 0,
                end_sample: end,
            }])
        }
    }

    fn tone(samples: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(samples);
        for i in 0..samples {
            let t = i as f32 / 16_000.0;
            out.push((t * 2.0 * std::f32::consts::PI * 220.0).sin() * 0.2);
        }
        out
    }

    fn scaled_tone(samples: usize, scale: f32) -> Vec<f32> {
        tone(samples)
            .into_iter()
            .map(|sample| sample * scale)
            .collect()
    }

    #[test]
    fn fixed_mode_generates_multiple_slices() {
        let mut options = options_with_mode(LongFormMode::Fixed);
        options.chunk_seconds = 2.0;
        options.overlap_seconds = 0.5;
        let plan = plan_longform_slices(&tone(16_000 * 6), 16_000, &options, None).unwrap();
        assert!(plan.slices.len() >= 3);
        assert_eq!(plan.slices[0].content_start_sample, 0);
    }

    #[test]
    fn energy_mode_splits_long_audio() {
        let mut samples = tone(16_000 * 6);
        for sample in samples
            .iter_mut()
            .take(16_000 * 3 + 2000)
            .skip(16_000 * 3 - 2000)
        {
            *sample = 0.0;
        }
        let mut options = options_with_mode(LongFormMode::Energy);
        options.chunk_seconds = 2.0;
        let plan = plan_longform_slices(&samples, 16_000, &options, None).unwrap();
        assert!(plan.slices.len() >= 2);
    }

    #[test]
    fn packed_energy_candidate_removes_long_silence_gaps() {
        let mut samples = tone(16_000);
        samples.extend(vec![0.0; 16_000 * 12]);
        samples.extend(tone(16_000));
        let options = LongFormOptions::default();
        let layout = plan_packed_energy_layout(&samples, 16_000, &options).expect("packed");
        assert_eq!(layout.slices.len(), 1);
        assert!(layout.processed_audio.is_none());
        assert!(
            layout
                .packed_audio_plan
                .as_ref()
                .expect("materialization plan")
                .processed_samples
                < samples.len() / 2
        );
        let timeline = layout.timeline;
        assert!(timeline.map_processed_to_original_seconds(0.0) < 0.5);
        assert!(timeline.map_processed_to_original_seconds(1.5) > 10.0);
    }

    #[test]
    fn energy_mode_keeps_moderate_pauses_inside_one_chunk() {
        let mut samples = tone(16_000 * 10);
        samples.extend(vec![0.0; 16_000 * 3]);
        samples.extend(tone(16_000 * 10));
        let options = LongFormOptions::default();
        assert!(plan_packed_energy_layout(&samples, 16_000, &options).is_some());
        let plan = plan_longform_slices(&samples, 16_000, &options, None).unwrap();
        assert_eq!(plan.slices.len(), 1);
        assert!(plan.processed_audio.is_none());
    }

    #[test]
    fn empty_audio_returns_empty_plan() {
        let plan = plan_longform_slices(&[], 16_000, &LongFormOptions::default(), None).unwrap();
        assert!(plan.slices.is_empty());
    }

    #[test]
    fn invalid_sample_rate_fails_closed() {
        let error =
            plan_longform_slices(&tone(1600), 0, &LongFormOptions::default(), None).unwrap_err();
        assert!(matches!(error, LongFormSliceError::InvalidSampleRate));
    }

    #[test]
    fn auto_mode_prefers_vad_provider_for_long_audio() {
        let mut options = options_with_mode(LongFormMode::Auto);
        options.chunk_seconds = 1.0;
        let mut samples = tone(16_000);
        samples.extend(vec![0.0; 16_000 * 2]);
        samples.extend(tone(16_000));
        let plan =
            plan_longform_slices(&samples, 16_000, &options, Some(&FixedVadProvider)).unwrap();
        assert_eq!(plan.slices.len(), 1);
        assert_eq!(plan.slices[0].kind, AudioSliceKind::Vad);
    }

    #[test]
    fn vad_mode_falls_back_to_energy_when_provider_is_unavailable() {
        let mut options = options_with_mode(LongFormMode::Vad);
        options.chunk_seconds = 2.0;
        options.fallback_to_energy_when_vad_unavailable = true;
        let samples = tone(16_000 * 6);
        let plan = plan_longform_slices(&samples, 16_000, &options, None).unwrap();
        assert!(!plan.slices.is_empty());
        assert_eq!(plan.slices[0].kind, AudioSliceKind::Energy);
    }

    #[test]
    fn vad_mode_fails_closed_when_provider_is_unavailable_and_fallback_is_disabled() {
        let mut options = options_with_mode(LongFormMode::Vad);
        options.fallback_to_energy_when_vad_unavailable = false;
        let samples = tone(16_000 * 2);
        let error = plan_longform_slices(&samples, 16_000, &options, None).unwrap_err();
        assert!(matches!(error, LongFormSliceError::VadUnavailable));
    }

    #[test]
    fn vad_mode_coalesces_short_adjacent_speech_chunks() {
        struct FragmentedVadProvider;
        impl LongFormVadProvider for FragmentedVadProvider {
            fn compute_speech_slices(
                &self,
                _samples: &[f32],
                _sample_rate_hz: u32,
                _options: &LongFormOptions,
            ) -> Result<Vec<LongFormVadSlice>, String> {
                Ok(vec![
                    LongFormVadSlice {
                        start_sample: 0,
                        end_sample: 16_000,
                    },
                    LongFormVadSlice {
                        start_sample: 16_400,
                        end_sample: 32_000,
                    },
                    LongFormVadSlice {
                        start_sample: 40_000,
                        end_sample: 56_000,
                    },
                ])
            }
        }

        let mut options = options_with_mode(LongFormMode::Vad);
        options.chunk_seconds = 4.0;
        options.min_chunk_seconds = 2.5;
        options.vad.min_silence_duration_ms = 100;
        let samples = tone(16_000 * 6);
        let plan =
            plan_longform_slices(&samples, 16_000, &options, Some(&FragmentedVadProvider)).unwrap();
        assert!(
            plan.slices.len() <= 2,
            "coalesced slices: {}",
            plan.slices.len()
        );
        assert_eq!(plan.slices[0].kind, AudioSliceKind::Vad);
    }

    #[test]
    fn vad_mode_packs_adjacent_speech_regions_across_moderate_pauses() {
        struct PausedVadProvider;
        impl LongFormVadProvider for PausedVadProvider {
            fn compute_speech_slices(
                &self,
                _samples: &[f32],
                _sample_rate_hz: u32,
                _options: &LongFormOptions,
            ) -> Result<Vec<LongFormVadSlice>, String> {
                Ok(vec![
                    LongFormVadSlice {
                        start_sample: 0,
                        end_sample: 16_000 * 4,
                    },
                    LongFormVadSlice {
                        start_sample: 16_000 * 5,
                        end_sample: 16_000 * 9,
                    },
                    LongFormVadSlice {
                        start_sample: 16_000 * 11,
                        end_sample: 16_000 * 15,
                    },
                    LongFormVadSlice {
                        start_sample: 16_000 * 25,
                        end_sample: 16_000 * 29,
                    },
                ])
            }
        }

        let mut options = options_with_mode(LongFormMode::Vad);
        options.chunk_seconds = 30.0;
        options.padding_seconds = 0.25;
        options.overlap_seconds = 0.5;
        options.vad.min_silence_duration_ms = 450;
        let samples = tone(16_000 * 30);
        let plan =
            plan_longform_slices(&samples, 16_000, &options, Some(&PausedVadProvider)).unwrap();
        assert_eq!(plan.slices.len(), 2);
        assert_eq!(plan.slices[0].kind, AudioSliceKind::Vad);
        assert!(plan.slices[0].content_end_sample >= 16_000 * 15);
        assert!(plan.slices[1].content_start_sample >= 16_000 * 25);
    }

    #[test]
    fn auto_mode_keeps_best_energy_plan_when_custom_vad_is_over_fragmented() {
        struct OverFragmentedVadProvider;
        impl LongFormVadProvider for OverFragmentedVadProvider {
            fn compute_speech_slices(
                &self,
                _samples: &[f32],
                _sample_rate_hz: u32,
                _options: &LongFormOptions,
            ) -> Result<Vec<LongFormVadSlice>, String> {
                let mut slices = Vec::new();
                for index in 0..9 {
                    let start = index * 16_000 * 3;
                    slices.push(LongFormVadSlice {
                        start_sample: start,
                        end_sample: start + 16_000,
                    });
                }
                Ok(slices)
            }
        }

        let mut options = options_with_mode(LongFormMode::Auto);
        options.chunk_seconds = 8.0;
        options.min_chunk_seconds = 1.0;
        let mut samples = Vec::new();
        for index in 0..9 {
            samples.extend(tone(16_000));
            if index < 8 {
                samples.extend(scaled_tone(16_000 * 2, 0.1));
            }
        }
        let plan =
            plan_longform_slices(&samples, 16_000, &options, Some(&OverFragmentedVadProvider))
                .unwrap();
        let provenance = plan.stats.provenance.join("\n");
        assert!(!plan.slices.is_empty());
        assert_eq!(plan.slices[0].kind, AudioSliceKind::Energy, "{provenance}");
        assert!(plan.processed_audio.is_some(), "{provenance}");
        assert!(provenance.contains("energy-packed"), "{provenance}");
    }

    #[test]
    fn auto_mode_can_choose_packed_vad_candidate_for_large_gaps() {
        struct SparseVadProvider;
        impl LongFormVadProvider for SparseVadProvider {
            fn compute_speech_slices(
                &self,
                _samples: &[f32],
                _sample_rate_hz: u32,
                _options: &LongFormOptions,
            ) -> Result<Vec<LongFormVadSlice>, String> {
                Ok(vec![
                    LongFormVadSlice {
                        start_sample: 16_000 * 2,
                        end_sample: 16_000 * 3,
                    },
                    LongFormVadSlice {
                        start_sample: 16_000 * 20,
                        end_sample: 16_000 * 21,
                    },
                    LongFormVadSlice {
                        start_sample: 16_000 * 35,
                        end_sample: 16_000 * 36,
                    },
                ])
            }
        }

        let mut options = options_with_mode(LongFormMode::Auto);
        options.chunk_seconds = 30.0;
        let samples = vec![0.0; 16_000 * 40];
        let plan =
            plan_longform_slices(&samples, 16_000, &options, Some(&SparseVadProvider)).unwrap();
        assert!(plan.processed_audio.is_some());
        assert_eq!(plan.slices[0].kind, AudioSliceKind::Vad);
        assert!(plan.slices.len() <= 2);
        assert!(plan.processed_audio.as_ref().expect("processed").len() < samples.len() / 3);
    }

    #[test]
    fn auto_mode_prefers_custom_vad_when_energy_keeps_noisy_bridges() {
        struct SpeechOnlyVadProvider;
        impl LongFormVadProvider for SpeechOnlyVadProvider {
            fn compute_speech_slices(
                &self,
                _samples: &[f32],
                _sample_rate_hz: u32,
                _options: &LongFormOptions,
            ) -> Result<Vec<LongFormVadSlice>, String> {
                Ok(vec![
                    LongFormVadSlice {
                        start_sample: 0,
                        end_sample: 16_000 * 12,
                    },
                    LongFormVadSlice {
                        start_sample: 16_000 * 22,
                        end_sample: 16_000 * 34,
                    },
                    LongFormVadSlice {
                        start_sample: 16_000 * 44,
                        end_sample: 16_000 * 56,
                    },
                ])
            }
        }

        let mut options = options_with_mode(LongFormMode::Auto);
        options.chunk_seconds = 30.0;
        options.min_chunk_seconds = 1.0;
        options.padding_seconds = 0.0;
        options.vad.threshold = 0.1;

        let mut samples = tone(16_000 * 12);
        samples.extend(scaled_tone(16_000 * 10, 0.4));
        samples.extend(tone(16_000 * 12));
        samples.extend(scaled_tone(16_000 * 10, 0.4));
        samples.extend(tone(16_000 * 12));

        let plan =
            plan_longform_slices(&samples, 16_000, &options, Some(&SpeechOnlyVadProvider)).unwrap();
        let provenance = plan.stats.provenance.join("\n");
        assert_eq!(plan.slices[0].kind, AudioSliceKind::Vad, "{provenance}");
        assert!(
            provenance.contains("core.longform.auto.selected:vad-"),
            "{provenance}"
        );
    }

    #[test]
    fn auto_mode_considers_custom_vad_without_energy_prefilter() {
        struct SparseVadProvider;
        impl LongFormVadProvider for SparseVadProvider {
            fn compute_speech_slices(
                &self,
                _samples: &[f32],
                _sample_rate_hz: u32,
                _options: &LongFormOptions,
            ) -> Result<Vec<LongFormVadSlice>, String> {
                Ok(vec![
                    LongFormVadSlice {
                        start_sample: 16_000 * 2,
                        end_sample: 16_000 * 3,
                    },
                    LongFormVadSlice {
                        start_sample: 16_000 * 20,
                        end_sample: 16_000 * 21,
                    },
                    LongFormVadSlice {
                        start_sample: 16_000 * 35,
                        end_sample: 16_000 * 36,
                    },
                ])
            }
        }

        let mut options = options_with_mode(LongFormMode::Auto);
        options.chunk_seconds = 30.0;
        let samples = tone(16_000 * 40);
        let plan =
            plan_longform_slices(&samples, 16_000, &options, Some(&SparseVadProvider)).unwrap();
        let provenance = plan.stats.provenance.join("\n");
        assert!(provenance.contains("vad-"), "{provenance}");
    }

    #[test]
    fn auto_mode_prunes_vad_candidate_when_energy_packed_dominates_it() {
        let mut options = options_with_mode(LongFormMode::Auto);
        options.chunk_seconds = 8.0;
        options.padding_seconds = 0.0;
        let packed_slices = vec![AudioSlice {
            index: 0,
            kind: AudioSliceKind::Energy,
            start_sample: 0,
            end_sample: 16_000 * 4,
            content_start_sample: 0,
            content_end_sample: 16_000 * 4,
        }];
        let samples = tone(16_000 * 4);
        let energy_candidate = build_auto_plan_candidate(
            AudioSliceKind::Energy,
            LongFormPlanningLayout {
                slices: packed_slices.clone(),
                processed_audio: Some(samples.clone()),
                packed_audio_plan: None,
                timeline: TimelineMap::identity(),
                selection_provenance: Vec::new(),
            },
            &samples,
            16_000 * 20,
            16_000,
            &options,
        );
        let vad_candidate = build_auto_plan_candidate(
            AudioSliceKind::Vad,
            LongFormPlanningLayout {
                slices: packed_slices,
                processed_audio: Some(samples.clone()),
                packed_audio_plan: None,
                timeline: TimelineMap::identity(),
                selection_provenance: Vec::new(),
            },
            &samples,
            16_000 * 20,
            16_000,
            &options,
        );
        let mut candidates = vec![vad_candidate, energy_candidate];
        prune_dominated_vad_candidates(&mut candidates);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].kind, AudioSliceKind::Energy);
    }

    #[test]
    fn auto_mode_penalizes_marginal_packed_candidate_when_identity_has_same_chunk_count() {
        let mut options = options_with_mode(LongFormMode::Auto);
        options.chunk_seconds = 30.0;
        let samples = tone(16_000 * 10);
        let slices = vec![AudioSlice {
            index: 0,
            kind: AudioSliceKind::Energy,
            start_sample: 0,
            end_sample: 16_000 * 10,
            content_start_sample: 0,
            content_end_sample: 16_000 * 10,
        }];
        let identity_candidate = build_auto_plan_candidate(
            AudioSliceKind::Energy,
            LongFormPlanningLayout {
                slices: slices.clone(),
                processed_audio: None,
                packed_audio_plan: None,
                timeline: TimelineMap::identity(),
                selection_provenance: Vec::new(),
            },
            &samples,
            16_000 * 120,
            16_000,
            &options,
        );
        let marginal_packed_candidate = build_auto_plan_candidate(
            AudioSliceKind::Energy,
            LongFormPlanningLayout {
                slices,
                processed_audio: Some(tone(identity_candidate.processed_samples - 16_000 * 3)),
                packed_audio_plan: None,
                timeline: TimelineMap::identity(),
                selection_provenance: Vec::new(),
            },
            &samples,
            16_000 * 120,
            16_000,
            &options,
        );
        let mut candidates = vec![marginal_packed_candidate, identity_candidate];
        let provenance =
            apply_marginal_packed_penalties(&mut candidates, 16_000 * 120, 16_000, &options);
        candidates.sort_by(compare_auto_plan_candidates);
        assert_eq!(candidates.len(), 2);
        assert!(!layout_uses_packed_timeline(&candidates[0].layout));
        assert!(provenance.iter().any(|entry| {
            entry.contains("core.longform.auto.penalized:energy-packed")
                && entry.contains("identity_chunks=1:packed_chunks=1")
        }));
        assert!(candidates[1].contextual_penalty > 0);
    }

    #[test]
    fn auto_mode_penalizes_packed_candidate_when_it_adds_chunks_without_chunk_scale_savings() {
        let mut options = options_with_mode(LongFormMode::Auto);
        options.chunk_seconds = 30.0;

        let identity_candidate = AutoPlanCandidate {
            kind: AudioSliceKind::Energy,
            score: 1_000,
            processed_samples: 16_000 * 139,
            short_slice_penalty: 0,
            boundary_penalty: 400,
            elision_penalty: 0,
            gap_edge_penalty: 0,
            seam_penalty: 0,
            extra_chunk_penalty: 160_000,
            stability_bias: 0,
            contextual_credit: 0,
            contextual_penalty: 0,
            layout: LongFormPlanningLayout {
                slices: (0..5)
                    .map(|index| AudioSlice {
                        index,
                        kind: AudioSliceKind::Energy,
                        start_sample: index * 16_000 * 30,
                        end_sample: (index + 1) * 16_000 * 30,
                        content_start_sample: index * 16_000 * 30,
                        content_end_sample: (index + 1) * 16_000 * 30,
                    })
                    .collect(),
                processed_audio: None,
                packed_audio_plan: None,
                timeline: TimelineMap::identity(),
                selection_provenance: Vec::new(),
            },
        };
        let packed_candidate = AutoPlanCandidate {
            kind: AudioSliceKind::Energy,
            score: 900,
            processed_samples: 16_000 * 118,
            short_slice_penalty: 170_000,
            boundary_penalty: 2_200,
            elision_penalty: 52_000,
            gap_edge_penalty: 0,
            seam_penalty: 10_000,
            extra_chunk_penalty: 200_000,
            stability_bias: 0,
            contextual_credit: 0,
            contextual_penalty: 0,
            layout: LongFormPlanningLayout {
                slices: (0..6)
                    .map(|index| AudioSlice {
                        index,
                        kind: AudioSliceKind::Energy,
                        start_sample: index * 16_000 * 24,
                        end_sample: (index + 1) * 16_000 * 24,
                        content_start_sample: index * 16_000 * 24,
                        content_end_sample: (index + 1) * 16_000 * 24,
                    })
                    .collect(),
                processed_audio: None,
                packed_audio_plan: Some(PackedAudioMaterializationPlan {
                    spans: vec![
                        LongFormVadSlice {
                            start_sample: 0,
                            end_sample: 16_000 * 20,
                        };
                        6
                    ],
                    seam_samples: 16_000 / 10,
                    processed_samples: 16_000 * 118,
                }),
                timeline: TimelineMap::identity(),
                selection_provenance: Vec::new(),
            },
        };

        let mut candidates = vec![packed_candidate, identity_candidate];
        let provenance =
            apply_marginal_packed_penalties(&mut candidates, 16_000 * 139, 16_000, &options);
        candidates.sort_by(compare_auto_plan_candidates);

        assert_eq!(candidates[0].kind, AudioSliceKind::Energy);
        assert!(
            !layout_uses_packed_timeline(&candidates[0].layout),
            "{candidates:#?}"
        );
        assert!(
            provenance
                .iter()
                .any(|entry| entry.contains("identity_chunks=5:packed_chunks=6")),
            "{provenance:?}"
        );
    }

    #[test]
    fn auto_mode_penalizes_marginal_vad_candidate_when_energy_shape_matches() {
        let mut options = options_with_mode(LongFormMode::Auto);
        options.chunk_seconds = 30.0;
        let mut samples = tone(16_000 * 4);
        samples.extend(vec![0.0; 16_000]);
        samples.extend(tone(16_000 * 4));
        let energy_candidate = build_auto_plan_candidate(
            AudioSliceKind::Energy,
            LongFormPlanningLayout {
                slices: vec![
                    AudioSlice {
                        index: 0,
                        kind: AudioSliceKind::Energy,
                        start_sample: 0,
                        end_sample: 16_000 * 3 + 16_000 / 2,
                        content_start_sample: 0,
                        content_end_sample: 16_000 * 3 + 16_000 / 2,
                    },
                    AudioSlice {
                        index: 1,
                        kind: AudioSliceKind::Energy,
                        start_sample: 16_000 * 3 + 16_000 / 2,
                        end_sample: 16_000 * 9,
                        content_start_sample: 16_000 * 3 + 16_000 / 2,
                        content_end_sample: 16_000 * 9,
                    },
                ],
                processed_audio: Some(samples.clone()),
                packed_audio_plan: None,
                timeline: TimelineMap::identity(),
                selection_provenance: Vec::new(),
            },
            &samples,
            16_000 * 9,
            16_000,
            &options,
        );
        let vad_candidate = build_auto_plan_candidate(
            AudioSliceKind::Vad,
            LongFormPlanningLayout {
                slices: vec![
                    AudioSlice {
                        index: 0,
                        kind: AudioSliceKind::Vad,
                        start_sample: 0,
                        end_sample: 16_000 * 4 + 16_000 / 2,
                        content_start_sample: 0,
                        content_end_sample: 16_000 * 4 + 16_000 / 2,
                    },
                    AudioSlice {
                        index: 1,
                        kind: AudioSliceKind::Vad,
                        start_sample: 16_000 * 5,
                        end_sample: 16_000 * 9,
                        content_start_sample: 16_000 * 5,
                        content_end_sample: 16_000 * 9,
                    },
                ],
                processed_audio: Some(samples[..(16_000 * 8 + 16_000 / 2)].to_vec()),
                packed_audio_plan: None,
                timeline: TimelineMap::identity(),
                selection_provenance: Vec::new(),
            },
            &samples,
            16_000 * 9,
            16_000,
            &options,
        );
        assert!(vad_candidate.boundary_penalty < energy_candidate.boundary_penalty);
        let mut candidates = vec![vad_candidate, energy_candidate];
        let provenance =
            apply_marginal_vad_penalties(&mut candidates, 16_000 * 9, 16_000, &options);
        candidates.sort_by(compare_auto_plan_candidates);
        assert!(provenance.is_empty());
        assert_eq!(candidates[0].kind, AudioSliceKind::Vad);
        assert_eq!(candidates[0].contextual_penalty, 0);
    }

    #[test]
    fn auto_mode_rewards_material_vad_boundary_gain_for_matching_shape() {
        let mut options = options_with_mode(LongFormMode::Auto);
        options.chunk_seconds = 30.0;

        let energy_candidate = AutoPlanCandidate {
            kind: AudioSliceKind::Energy,
            score: 1000,
            processed_samples: 16_000 * 30,
            short_slice_penalty: 0,
            boundary_penalty: 700,
            elision_penalty: 0,
            gap_edge_penalty: 0,
            seam_penalty: 0,
            extra_chunk_penalty: 40_000,
            stability_bias: 0,
            contextual_credit: 0,
            contextual_penalty: 0,
            layout: LongFormPlanningLayout {
                slices: vec![
                    AudioSlice {
                        index: 0,
                        kind: AudioSliceKind::Energy,
                        start_sample: 0,
                        end_sample: 16_000 * 15,
                        content_start_sample: 0,
                        content_end_sample: 16_000 * 15,
                    },
                    AudioSlice {
                        index: 1,
                        kind: AudioSliceKind::Energy,
                        start_sample: 16_000 * 15,
                        end_sample: 16_000 * 30,
                        content_start_sample: 16_000 * 15,
                        content_end_sample: 16_000 * 30,
                    },
                ],
                processed_audio: None,
                packed_audio_plan: Some(PackedAudioMaterializationPlan {
                    spans: vec![
                        LongFormVadSlice {
                            start_sample: 0,
                            end_sample: 16_000 * 15,
                        },
                        LongFormVadSlice {
                            start_sample: 16_000 * 15,
                            end_sample: 16_000 * 30,
                        },
                    ],
                    seam_samples: 0,
                    processed_samples: 16_000 * 30,
                }),
                timeline: TimelineMap::identity(),
                selection_provenance: Vec::new(),
            },
        };
        let vad_candidate = AutoPlanCandidate {
            kind: AudioSliceKind::Vad,
            score: 1200,
            processed_samples: 16_000 * 30,
            short_slice_penalty: 0,
            boundary_penalty: 100,
            elision_penalty: 0,
            gap_edge_penalty: 0,
            seam_penalty: 0,
            extra_chunk_penalty: 40_000,
            stability_bias: 0,
            contextual_credit: 0,
            contextual_penalty: 0,
            layout: LongFormPlanningLayout {
                slices: vec![
                    AudioSlice {
                        index: 0,
                        kind: AudioSliceKind::Vad,
                        start_sample: 0,
                        end_sample: 16_000 * 15,
                        content_start_sample: 0,
                        content_end_sample: 16_000 * 15,
                    },
                    AudioSlice {
                        index: 1,
                        kind: AudioSliceKind::Vad,
                        start_sample: 16_000 * 15,
                        end_sample: 16_000 * 30,
                        content_start_sample: 16_000 * 15,
                        content_end_sample: 16_000 * 30,
                    },
                ],
                processed_audio: None,
                packed_audio_plan: Some(PackedAudioMaterializationPlan {
                    spans: vec![
                        LongFormVadSlice {
                            start_sample: 0,
                            end_sample: 16_000 * 15,
                        },
                        LongFormVadSlice {
                            start_sample: 16_000 * 15,
                            end_sample: 16_000 * 30,
                        },
                    ],
                    seam_samples: 0,
                    processed_samples: 16_000 * 30,
                }),
                timeline: TimelineMap::identity(),
                selection_provenance: Vec::new(),
            },
        };

        assert!(vad_candidate.boundary_penalty < energy_candidate.boundary_penalty);
        assert!(vad_candidate.score > energy_candidate.score);

        let mut candidates = vec![vad_candidate, energy_candidate];
        let provenance = apply_material_vad_boundary_credits(&mut candidates, 16_000, &options);
        candidates.sort_by(compare_auto_plan_candidates);

        assert!(
            provenance
                .iter()
                .any(|entry| entry.contains("auto.rewarded:vad-")),
            "{provenance:?}"
        );
        assert_eq!(candidates[0].kind, AudioSliceKind::Vad, "{candidates:#?}");
        assert!(candidates[0].contextual_credit > 0);
    }

    #[test]
    fn auto_mode_does_not_reward_vad_boundary_gain_when_fragmentation_overhead_cancels_it() {
        let mut options = options_with_mode(LongFormMode::Auto);
        options.chunk_seconds = 30.0;

        let energy_candidate = AutoPlanCandidate {
            kind: AudioSliceKind::Energy,
            score: 1000,
            processed_samples: 16_000 * 30,
            short_slice_penalty: 0,
            boundary_penalty: 600,
            elision_penalty: 0,
            gap_edge_penalty: 0,
            seam_penalty: 0,
            extra_chunk_penalty: 40_000,
            stability_bias: 0,
            contextual_credit: 0,
            contextual_penalty: 0,
            layout: LongFormPlanningLayout {
                slices: vec![
                    AudioSlice {
                        index: 0,
                        kind: AudioSliceKind::Energy,
                        start_sample: 0,
                        end_sample: 16_000 * 15,
                        content_start_sample: 0,
                        content_end_sample: 16_000 * 15,
                    },
                    AudioSlice {
                        index: 1,
                        kind: AudioSliceKind::Energy,
                        start_sample: 16_000 * 15,
                        end_sample: 16_000 * 30,
                        content_start_sample: 16_000 * 15,
                        content_end_sample: 16_000 * 30,
                    },
                ],
                processed_audio: None,
                packed_audio_plan: Some(PackedAudioMaterializationPlan {
                    spans: vec![
                        LongFormVadSlice {
                            start_sample: 0,
                            end_sample: 16_000 * 15,
                        },
                        LongFormVadSlice {
                            start_sample: 16_000 * 15,
                            end_sample: 16_000 * 30,
                        },
                    ],
                    seam_samples: 0,
                    processed_samples: 16_000 * 30,
                }),
                timeline: TimelineMap::identity(),
                selection_provenance: Vec::new(),
            },
        };
        let vad_candidate = AutoPlanCandidate {
            kind: AudioSliceKind::Vad,
            score: 1100,
            processed_samples: 16_000 * 30,
            short_slice_penalty: 350,
            boundary_penalty: 300,
            elision_penalty: 0,
            gap_edge_penalty: 0,
            seam_penalty: 120,
            extra_chunk_penalty: 40_000,
            stability_bias: 0,
            contextual_credit: 0,
            contextual_penalty: 0,
            layout: LongFormPlanningLayout {
                slices: vec![
                    AudioSlice {
                        index: 0,
                        kind: AudioSliceKind::Vad,
                        start_sample: 0,
                        end_sample: 16_000 * 15,
                        content_start_sample: 0,
                        content_end_sample: 16_000 * 15,
                    },
                    AudioSlice {
                        index: 1,
                        kind: AudioSliceKind::Vad,
                        start_sample: 16_000 * 15,
                        end_sample: 16_000 * 30,
                        content_start_sample: 16_000 * 15,
                        content_end_sample: 16_000 * 30,
                    },
                ],
                processed_audio: None,
                packed_audio_plan: Some(PackedAudioMaterializationPlan {
                    spans: vec![
                        LongFormVadSlice {
                            start_sample: 0,
                            end_sample: 16_000 * 8,
                        },
                        LongFormVadSlice {
                            start_sample: 16_000 * 8,
                            end_sample: 16_000 * 15,
                        },
                        LongFormVadSlice {
                            start_sample: 16_000 * 15,
                            end_sample: 16_000 * 22,
                        },
                        LongFormVadSlice {
                            start_sample: 16_000 * 22,
                            end_sample: 16_000 * 30,
                        },
                    ],
                    seam_samples: 16_000 / 10,
                    processed_samples: 16_000 * 30,
                }),
                timeline: TimelineMap::identity(),
                selection_provenance: Vec::new(),
            },
        };

        let mut candidates = vec![vad_candidate, energy_candidate];
        let provenance = apply_material_vad_boundary_credits(&mut candidates, 16_000, &options);
        candidates.sort_by(compare_auto_plan_candidates);

        assert!(provenance.is_empty(), "{provenance:?}");
        assert_eq!(
            candidates[0].kind,
            AudioSliceKind::Energy,
            "{candidates:#?}"
        );
        assert_eq!(candidates[1].contextual_credit, 0);
    }

    #[test]
    fn boundary_penalty_prefers_quieter_internal_cuts() {
        let mut options = options_with_mode(LongFormMode::Auto);
        options.chunk_seconds = 30.0;
        let mut samples = tone(16_000 * 4);
        samples.extend(vec![0.0; 16_000]);
        samples.extend(tone(16_000 * 4));
        let loud_cut = LongFormPlanningLayout {
            slices: vec![
                AudioSlice {
                    index: 0,
                    kind: AudioSliceKind::Energy,
                    start_sample: 0,
                    end_sample: 16_000 * 3 + 16_000 / 2,
                    content_start_sample: 0,
                    content_end_sample: 16_000 * 3 + 16_000 / 2,
                },
                AudioSlice {
                    index: 1,
                    kind: AudioSliceKind::Energy,
                    start_sample: 16_000 * 3 + 16_000 / 2,
                    end_sample: 16_000 * 9,
                    content_start_sample: 16_000 * 3 + 16_000 / 2,
                    content_end_sample: 16_000 * 9,
                },
            ],
            processed_audio: None,
            packed_audio_plan: None,
            timeline: TimelineMap::identity(),
            selection_provenance: Vec::new(),
        };
        let quiet_cut = LongFormPlanningLayout {
            slices: vec![
                AudioSlice {
                    index: 0,
                    kind: AudioSliceKind::Vad,
                    start_sample: 0,
                    end_sample: 16_000 * 4 + 16_000 / 2,
                    content_start_sample: 0,
                    content_end_sample: 16_000 * 4 + 16_000 / 2,
                },
                AudioSlice {
                    index: 1,
                    kind: AudioSliceKind::Vad,
                    start_sample: 16_000 * 5,
                    end_sample: 16_000 * 9,
                    content_start_sample: 16_000 * 5,
                    content_end_sample: 16_000 * 9,
                },
            ],
            processed_audio: None,
            packed_audio_plan: None,
            timeline: TimelineMap::identity(),
            selection_provenance: Vec::new(),
        };
        let loud_penalty = estimate_boundary_penalty(&samples, &loud_cut, 16_000, &options);
        let quiet_penalty = estimate_boundary_penalty(&samples, &quiet_cut, 16_000, &options);
        assert!(
            quiet_penalty < loud_penalty,
            "{quiet_penalty} !< {loud_penalty}"
        );
    }

    #[test]
    fn elision_penalty_only_charges_non_silent_removed_gaps() {
        let mut options = options_with_mode(LongFormMode::Auto);
        options.chunk_seconds = 30.0;
        options.padding_seconds = 0.0;

        let mut silent_gap_samples = tone(16_000 * 4);
        silent_gap_samples.extend(vec![0.0; 16_000 * 2]);
        silent_gap_samples.extend(tone(16_000 * 4));

        let mut loud_gap_samples = tone(16_000 * 4);
        loud_gap_samples.extend(tone(16_000 * 2));
        loud_gap_samples.extend(tone(16_000 * 4));

        let packed_layout = LongFormPlanningLayout {
            slices: vec![AudioSlice {
                index: 0,
                kind: AudioSliceKind::Energy,
                start_sample: 0,
                end_sample: 16_000 * 8,
                content_start_sample: 0,
                content_end_sample: 16_000 * 8,
            }],
            processed_audio: None,
            packed_audio_plan: Some(PackedAudioMaterializationPlan {
                spans: vec![
                    LongFormVadSlice {
                        start_sample: 0,
                        end_sample: 16_000 * 4,
                    },
                    LongFormVadSlice {
                        start_sample: 16_000 * 6,
                        end_sample: 16_000 * 10,
                    },
                ],
                seam_samples: 0,
                processed_samples: 16_000 * 8,
            }),
            timeline: TimelineMap::identity(),
            selection_provenance: Vec::new(),
        };

        let silent_penalty =
            estimate_elision_penalty(&silent_gap_samples, &packed_layout, 16_000, &options);
        let loud_penalty =
            estimate_elision_penalty(&loud_gap_samples, &packed_layout, 16_000, &options);
        assert_eq!(silent_penalty, 0);
        assert!(loud_penalty > 0, "{loud_penalty}");
    }

    #[test]
    fn gap_edge_penalty_only_charges_non_quiet_gap_edges() {
        let mut options = options_with_mode(LongFormMode::Auto);
        options.chunk_seconds = 30.0;
        options.padding_seconds = 0.0;

        let mut quiet_edge_gap_samples = tone(16_000 * 4);
        quiet_edge_gap_samples.extend(vec![0.0; 16_000 / 2]);
        quiet_edge_gap_samples.extend(scaled_tone(16_000, 0.35));
        quiet_edge_gap_samples.extend(vec![0.0; 16_000 / 2]);
        quiet_edge_gap_samples.extend(tone(16_000 * 4));

        let mut loud_edge_gap_samples = tone(16_000 * 4);
        loud_edge_gap_samples.extend(scaled_tone(16_000 / 2, 0.35));
        loud_edge_gap_samples.extend(vec![0.0; 16_000]);
        loud_edge_gap_samples.extend(scaled_tone(16_000 / 2, 0.35));
        loud_edge_gap_samples.extend(tone(16_000 * 4));

        let packed_layout = LongFormPlanningLayout {
            slices: vec![AudioSlice {
                index: 0,
                kind: AudioSliceKind::Energy,
                start_sample: 0,
                end_sample: 16_000 * 8,
                content_start_sample: 0,
                content_end_sample: 16_000 * 8,
            }],
            processed_audio: None,
            packed_audio_plan: Some(PackedAudioMaterializationPlan {
                spans: vec![
                    LongFormVadSlice {
                        start_sample: 0,
                        end_sample: 16_000 * 4,
                    },
                    LongFormVadSlice {
                        start_sample: 16_000 * 6,
                        end_sample: 16_000 * 10,
                    },
                ],
                seam_samples: 0,
                processed_samples: 16_000 * 8,
            }),
            timeline: TimelineMap::identity(),
            selection_provenance: Vec::new(),
        };

        let quiet_penalty =
            estimate_gap_edge_penalty(&quiet_edge_gap_samples, &packed_layout, 16_000, &options);
        let loud_penalty =
            estimate_gap_edge_penalty(&loud_edge_gap_samples, &packed_layout, 16_000, &options);
        assert_eq!(quiet_penalty, 0);
        assert!(loud_penalty > 0, "{loud_penalty}");
    }

    #[test]
    fn seam_penalty_prefers_fewer_splices_for_same_chunk_layout() {
        let mut options = options_with_mode(LongFormMode::Auto);
        options.chunk_seconds = 30.0;
        options.padding_seconds = 0.25;

        let fewer_seams = LongFormPlanningLayout {
            slices: vec![AudioSlice {
                index: 0,
                kind: AudioSliceKind::Energy,
                start_sample: 0,
                end_sample: 16_000 * 8,
                content_start_sample: 0,
                content_end_sample: 16_000 * 8,
            }],
            processed_audio: None,
            packed_audio_plan: Some(PackedAudioMaterializationPlan {
                spans: vec![
                    LongFormVadSlice {
                        start_sample: 0,
                        end_sample: 16_000 * 4,
                    },
                    LongFormVadSlice {
                        start_sample: 16_000 * 6,
                        end_sample: 16_000 * 10,
                    },
                ],
                seam_samples: seconds_to_samples(0.10, 16_000),
                processed_samples: 16_000 * 8 + seconds_to_samples(0.10, 16_000),
            }),
            timeline: TimelineMap::identity(),
            selection_provenance: Vec::new(),
        };
        let more_seams = LongFormPlanningLayout {
            slices: vec![AudioSlice {
                index: 0,
                kind: AudioSliceKind::Energy,
                start_sample: 0,
                end_sample: 16_000 * 8,
                content_start_sample: 0,
                content_end_sample: 16_000 * 8,
            }],
            processed_audio: None,
            packed_audio_plan: Some(PackedAudioMaterializationPlan {
                spans: vec![
                    LongFormVadSlice {
                        start_sample: 0,
                        end_sample: 16_000 * 2,
                    },
                    LongFormVadSlice {
                        start_sample: 16_000 * 3,
                        end_sample: 16_000 * 5,
                    },
                    LongFormVadSlice {
                        start_sample: 16_000 * 6,
                        end_sample: 16_000 * 8,
                    },
                    LongFormVadSlice {
                        start_sample: 16_000 * 9,
                        end_sample: 16_000 * 11,
                    },
                ],
                seam_samples: seconds_to_samples(0.10, 16_000),
                processed_samples: 16_000 * 8 + seconds_to_samples(0.10, 16_000) * 3,
            }),
            timeline: TimelineMap::identity(),
            selection_provenance: Vec::new(),
        };
        let fewer_penalty = estimate_seam_penalty(&fewer_seams, 16_000, &options);
        let more_penalty = estimate_seam_penalty(&more_seams, 16_000, &options);
        assert!(
            more_penalty > fewer_penalty,
            "{more_penalty} !> {fewer_penalty}"
        );
    }

    #[test]
    fn auto_mode_skips_duplicate_energy_like_vad_candidates() {
        let provider = EnergyLongFormVadProvider;
        let mut samples = tone(16_000 * 4);
        samples.extend(vec![0.0; 16_000 * 12]);
        samples.extend(tone(16_000 * 4));
        let mut options = options_with_mode(LongFormMode::Auto);
        options.chunk_seconds = 8.0;
        let plan = plan_longform_slices(&samples, 16_000, &options, Some(&provider)).unwrap();
        let provenance = plan.stats.provenance.join("\n");
        assert!(provenance.contains("energy-packed"), "{provenance}");
        assert!(!provenance.contains("vad-"), "{provenance}");
    }

    #[test]
    fn auto_mode_prefers_packed_timeline_for_large_silence_gaps() {
        let mut samples = Vec::new();
        for _ in 0..5 {
            samples.extend(tone(16_000 * 4));
            samples.extend(vec![0.0; 16_000 * 12]);
        }
        let mut options = options_with_mode(LongFormMode::Auto);
        options.chunk_seconds = 30.0;
        let plan = plan_longform_slices(&samples, 16_000, &options, None).unwrap();
        assert!(plan.processed_audio.is_some());
        assert!(plan.slices.len() <= 3);
        assert!(plan.processed_audio.as_ref().expect("processed").len() < samples.len() / 2);
    }

    #[test]
    fn auto_mode_can_elide_cumulative_moderate_gaps() {
        let mut samples = Vec::new();
        for index in 0..4 {
            samples.extend(tone(16_000 * 4));
            if index < 3 {
                samples.extend(vec![0.0; 16_000 * 6]);
            }
        }
        let mut options = options_with_mode(LongFormMode::Auto);
        options.chunk_seconds = 30.0;
        let plan = plan_longform_slices(&samples, 16_000, &options, None).unwrap();
        assert!(
            plan.processed_audio.is_some(),
            "expected packed timeline candidate"
        );
        assert!(plan.processed_audio.as_ref().expect("processed").len() < samples.len());
        assert!(
            plan.slices.len() <= 2,
            "packed slices: {}",
            plan.slices.len()
        );
    }

    #[test]
    fn auto_mode_prefers_identity_when_same_chunk_packed_savings_are_marginal() {
        let mut samples = Vec::new();
        for index in 0..5 {
            samples.extend(tone(16_000 * 6));
            if index < 4 {
                samples.extend(vec![0.0; 16_000 * 3]);
            }
        }
        let mut options = options_with_mode(LongFormMode::Auto);
        options.chunk_seconds = 30.0;
        let plan = plan_longform_slices(&samples, 16_000, &options, None).unwrap();
        let provenance = plan.stats.provenance.join("\n");
        assert!(plan.processed_audio.is_none(), "{provenance}");
        assert_eq!(plan.slices[0].kind, AudioSliceKind::Energy, "{provenance}");
        assert!(
            provenance.contains("core.longform.auto.penalized:energy-packed"),
            "{provenance}"
        );
        assert!(
            provenance.contains("core.longform.auto.selected:energy-identity"),
            "{provenance}"
        );
    }

    #[test]
    fn auto_mode_prefers_packed_for_material_quiet_gap_savings() {
        let mut options = options_with_mode(LongFormMode::Auto);
        options.chunk_seconds = 30.0;
        options.padding_seconds = 0.0;

        let mut samples = tone(16_000 * 18);
        samples.extend(vec![0.0; 16_000 * 6]);
        samples.extend(tone(16_000 * 18));

        let plan = plan_longform_slices(&samples, 16_000, &options, None).unwrap();
        let provenance = plan.stats.provenance.join("\n");
        assert!(plan.processed_audio.is_some(), "{provenance}");
        assert_eq!(plan.slices[0].kind, AudioSliceKind::Energy, "{provenance}");
        assert!(
            provenance.contains("core.longform.auto.selected:energy-packed"),
            "{provenance}"
        );
        assert!(
            !provenance.contains("core.longform.auto.penalized:energy-packed"),
            "{provenance}"
        );
    }

    #[test]
    fn auto_mode_prefers_identity_when_packed_gap_edges_are_loud() {
        let mut options = options_with_mode(LongFormMode::Auto);
        options.chunk_seconds = 30.0;
        options.padding_seconds = 0.0;

        let mut samples = tone(16_000 * 18);
        samples.extend(scaled_tone(16_000, 0.6));
        samples.extend(vec![0.0; 16_000 * 4]);
        samples.extend(scaled_tone(16_000, 0.6));
        samples.extend(tone(16_000 * 18));

        let plan = plan_longform_slices(&samples, 16_000, &options, None).unwrap();
        let provenance = plan.stats.provenance.join("\n");
        assert!(plan.processed_audio.is_none(), "{provenance}");
        assert_ne!(plan.slices[0].kind, AudioSliceKind::Vad, "{provenance}");
        assert!(
            provenance.contains("core.longform.auto.selected:fixed-identity")
                || provenance.contains("core.longform.auto.selected:energy-identity"),
            "{provenance}"
        );
        assert!(provenance.contains("energy-packed"), "{provenance}");
    }

    #[test]
    fn packed_windows_apply_configured_overlap_between_chunks() {
        let mut options = options_with_mode(LongFormMode::Auto);
        options.chunk_seconds = 30.0;
        options.min_chunk_seconds = 15.0;
        options.overlap_seconds = 0.5;

        let spans = vec![
            LongFormVadSlice {
                start_sample: 0,
                end_sample: 16_000 * 18,
            },
            LongFormVadSlice {
                start_sample: 16_000 * 18,
                end_sample: 16_000 * 36,
            },
            LongFormVadSlice {
                start_sample: 16_000 * 36,
                end_sample: 16_000 * 54,
            },
        ];
        let windows = pack_processed_spans_into_windows(&spans, 16_000, &options);
        assert_eq!(windows.len(), 3, "{windows:#?}");
        assert!(
            windows[1].start_sample < windows[0].end_sample,
            "{windows:#?}"
        );
        assert!(
            windows[2].start_sample < windows[1].end_sample,
            "{windows:#?}"
        );
        assert_eq!(
            windows[1].start_sample,
            windows[0].end_sample.saturating_sub(16_000 / 2),
            "{windows:#?}"
        );
    }

    #[test]
    fn packed_layout_processed_samples_include_window_overlap_cost() {
        let layout = LongFormPlanningLayout {
            slices: vec![
                AudioSlice {
                    index: 0,
                    kind: AudioSliceKind::Energy,
                    start_sample: 0,
                    end_sample: 16_000 * 30,
                    content_start_sample: 0,
                    content_end_sample: 16_000 * 30,
                },
                AudioSlice {
                    index: 1,
                    kind: AudioSliceKind::Energy,
                    start_sample: 16_000 * 30 - 16_000 / 2,
                    end_sample: 16_000 * 40,
                    content_start_sample: 16_000 * 30 - 16_000 / 2,
                    content_end_sample: 16_000 * 40,
                },
            ],
            processed_audio: None,
            packed_audio_plan: Some(PackedAudioMaterializationPlan {
                spans: vec![
                    LongFormVadSlice {
                        start_sample: 0,
                        end_sample: 16_000 * 20,
                    },
                    LongFormVadSlice {
                        start_sample: 16_000 * 20,
                        end_sample: 16_000 * 40,
                    },
                ],
                seam_samples: 0,
                processed_samples: 16_000 * 40,
            }),
            timeline: TimelineMap::identity(),
            selection_provenance: Vec::new(),
        };

        let estimated = estimate_layout_processed_samples(&layout, 16_000 * 40, 16_000, 0.0);
        assert_eq!(estimated, 16_000 * 40 + 16_000 / 2);
    }

    #[test]
    fn packed_candidates_do_not_get_implicit_score_bonus() {
        let mut options = options_with_mode(LongFormMode::Auto);
        options.chunk_seconds = 1.0;
        options.min_chunk_seconds = 0.5;
        options.padding_seconds = 0.0;
        let slices = vec![AudioSlice {
            index: 0,
            kind: AudioSliceKind::Energy,
            start_sample: 0,
            end_sample: 16_000,
            content_start_sample: 0,
            content_end_sample: 16_000,
        }];
        let identity_candidate = build_auto_plan_candidate(
            AudioSliceKind::Energy,
            LongFormPlanningLayout {
                slices: slices.clone(),
                processed_audio: None,
                packed_audio_plan: None,
                timeline: TimelineMap::identity(),
                selection_provenance: Vec::new(),
            },
            &tone(16_000),
            16_000,
            16_000,
            &options,
        );
        let packed_candidate = build_auto_plan_candidate(
            AudioSliceKind::Energy,
            LongFormPlanningLayout {
                slices,
                processed_audio: Some(tone(16_000)),
                packed_audio_plan: None,
                timeline: TimelineMap::identity(),
                selection_provenance: Vec::new(),
            },
            &tone(16_000),
            16_000,
            16_000,
            &options,
        );
        assert_eq!(
            packed_candidate.processed_samples,
            identity_candidate.processed_samples
        );
        assert_eq!(
            packed_candidate.short_slice_penalty,
            identity_candidate.short_slice_penalty
        );
        assert_eq!(packed_candidate.score, identity_candidate.score);
    }
}
