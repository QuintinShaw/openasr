use std::{
    collections::BTreeMap,
    num::NonZeroU32,
    ops::Deref,
    path::PathBuf,
    sync::{Arc, Mutex},
};

use thiserror::Error;

use crate::api::backend::DecodeTruncation;
use crate::ggml_runtime::{
    GgufRuntimeSourcePreflight, RequestBackendPreference, install_request_backend_override,
    request_backend_override,
};
use crate::models::ggml_family_adapter::GgmlAdapterBindingStrategy;
use crate::{
    GgmlExecutionCapability, GgmlFamilyAdapterDescriptor, GgmlRuntimeSource, LongFormOptions,
    NativeAsrBackpressurePolicy, NativeAsrSession, PcmSlice, PhraseBiasConfig, RealtimeAudioFormat,
    RequestExecutionContext, Transcription, TranscriptionTask,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgmlAsrBackendPreference {
    CpuOnly,
    /// Force the GPU-class backend (Metal on macOS). Conversion layers
    /// hard-error earlier when no GPU device exists, so this never silently
    /// downgrades.
    Accelerated,
    Auto,
}

impl GgmlAsrBackendPreference {
    /// The thread-local override the shared dispatch's backend resolution
    /// consults; `Auto` installs nothing (env/global default decides).
    pub(crate) fn request_backend_override(self) -> Option<RequestBackendPreference> {
        match self {
            Self::CpuOnly => Some(RequestBackendPreference::CpuOnly),
            Self::Accelerated => Some(RequestBackendPreference::Accelerated),
            Self::Auto => None,
        }
    }
}

/// Stable cache/engine identity for reusable native runtime state.
///
/// `pack_content_id` is a content proof, never a bare path, and is the
/// *entire* identity: two `RuntimeBuildIdentity` values with the same
/// content id, route, and options fingerprint are always interchangeable.
/// There is deliberately no invalidation generation/epoch here -- baking a
/// shared process-wide counter into this identity was an audited bug (one
/// idle unload / serve-batch owner shutdown / pack replace anywhere in the
/// process invalidated every resident identity, not just the one that
/// actually changed; see `runtime_cache_coordinator`'s module doc comment).
/// A pack replace already changes `pack_content_id` on its own; idle unload
/// and serve-batch owner shutdown now evict their own registries/caches
/// explicitly (see each family's `unload_idle_state` /
/// `shutdown_*_serve_batch_engines`) instead of relying on this identity
/// going stale.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RuntimeBuildIdentity {
    /// Content identity: `sha256:<hex>` for real pack bytes, or an explicit
    /// verified/fake id supplied by tests / future coordinator bindings.
    pub pack_content_id: String,
    /// Resolved execution route (family + backend lane) that owns the reusable
    /// graph shape.
    pub route: String,
    /// Adapter/options fingerprint that changes the lowered graph without
    /// changing pack bytes (for example an active `.oadp` adapter path).
    pub options_fingerprint: String,
}

impl RuntimeBuildIdentity {
    pub fn new(
        pack_content_id: impl Into<String>,
        route: impl Into<String>,
        options_fingerprint: impl Into<String>,
    ) -> Self {
        Self {
            pack_content_id: pack_content_id.into(),
            route: route.into(),
            options_fingerprint: options_fingerprint.into(),
        }
    }

    /// Builds the effective identity for one offline request.
    ///
    /// Prefer an explicit verified/fake content id from the request when present.
    /// Otherwise use the caller-supplied content id (production always passes a
    /// content-derived id from [`crate::GgmlRuntimeSource::content_id`]).
    pub fn resolve_for_request(
        request_identity: Option<&RuntimeBuildIdentity>,
        route: impl Into<String>,
        options_fingerprint: impl Into<String>,
        content_id: impl Into<String>,
    ) -> Self {
        let route = route.into();
        let options_fingerprint = options_fingerprint.into();
        match request_identity {
            Some(identity) => Self {
                pack_content_id: identity.pack_content_id.clone(),
                route,
                options_fingerprint,
            },
            None => Self {
                pack_content_id: content_id.into(),
                route,
                options_fingerprint,
            },
        }
    }

    /// Formats a content id from a lowercase hex sha256 digest.
    pub fn content_id_from_sha256_hex(sha256_hex: &str) -> String {
        crate::models::runtime_cache_coordinator::content_id_from_sha256_hex(sha256_hex)
    }
}

/// Builds the effective serve-batch / runtime-cache identity for one request.
///
/// Always binds a content-derived pack id, taken from `runtime_source`'s
/// already-open handle (`GgmlRuntimeSource::content_id`) -- never re-derived
/// from a bare path, which would reopen a file this request already has open
/// and admitted. Explicit request identities override the content id only
/// when the caller already supplies a verified/fake binding.
pub(crate) fn serve_batch_build_identity_for_request(
    options: &GgmlAsrExecutionOptions,
    family: &str,
    backend: crate::ggml_runtime::GgmlCpuGraphBackend,
    runtime_source: &GgmlRuntimeSource,
) -> RuntimeBuildIdentity {
    let options_fingerprint = match options.adapter_path.as_ref() {
        Some(path) => format!("adapter={}", path.display()),
        None => "adapter=none".to_string(),
    };
    RuntimeBuildIdentity::resolve_for_request(
        options.runtime_build_identity.as_ref(),
        format!("{family}:{backend:?}"),
        options_fingerprint,
        runtime_source.content_id(),
    )
}

/// Supplies a verified runtime identity to cache-owning execution components.
/// The pack resolver owns the content proof; consumers only carry it through
/// keys and invalidate when the content id itself changes.
pub trait RuntimeBuildIdentitySource {
    fn runtime_build_identity(&self) -> Option<RuntimeBuildIdentity>;
}

#[derive(Debug, Clone, PartialEq)]
pub struct GgmlAsrPreparedAudio {
    pub sample_rate_hz: u32,
    pub channels: u16,
    pub samples_f32: Vec<f32>,
}

impl GgmlAsrPreparedAudio {
    pub fn mono_16khz(samples_f32: Vec<f32>) -> Self {
        Self {
            sample_rate_hz: 16_000,
            channels: 1,
            samples_f32,
        }
    }

    fn as_view(&self) -> GgmlAsrPreparedAudioView<'_> {
        GgmlAsrPreparedAudioView {
            sample_rate_hz: self.sample_rate_hz,
            channels: self.channels,
            samples_f32: GgmlAsrSamplesView::Borrowed(&self.samples_f32),
        }
    }
}

/// Zero-copy audio view used only inside the native runtime.
///
/// The public [`GgmlAsrPreparedAudio`] remains the stable owned DTO. Native
/// long-form requests use the shared variant below so every slice and retry
/// references one immutable PCM allocation; an out-of-tree executor sees this
/// only through the dispatch's owned compatibility adapter.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct GgmlAsrPreparedAudioView<'a> {
    pub(crate) sample_rate_hz: u32,
    pub(crate) channels: u16,
    pub(crate) samples_f32: GgmlAsrSamplesView<'a>,
}

impl GgmlAsrPreparedAudioView<'static> {
    pub(crate) fn mono_16khz(samples_f32: Vec<f32>) -> Self {
        Self::mono_16khz_shared(samples_f32.into())
    }

    pub(crate) fn mono_16khz_shared(samples_f32: PcmSlice) -> Self {
        Self {
            sample_rate_hz: 16_000,
            channels: 1,
            samples_f32: GgmlAsrSamplesView::Shared(samples_f32),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum GgmlAsrSamplesView<'a> {
    Borrowed(&'a [f32]),
    Shared(PcmSlice),
}

impl Deref for GgmlAsrSamplesView<'_> {
    type Target = [f32];

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Borrowed(samples) => samples,
            Self::Shared(samples) => samples.as_slice(),
        }
    }
}

impl AsRef<[f32]> for GgmlAsrSamplesView<'_> {
    fn as_ref(&self) -> &[f32] {
        self
    }
}

impl GgmlAsrSamplesView<'_> {
    /// Produces a Send-safe owned view for a dedicated runtime actor. Native
    /// requests already carry [`PcmSlice`], so their hot path is only an Arc
    /// clone; the borrowed compatibility API pays the one unavoidable copy.
    pub(crate) fn to_owned_pcm_slice(&self) -> PcmSlice {
        match self {
            Self::Borrowed(samples) => samples.to_vec().into(),
            Self::Shared(samples) => samples.clone(),
        }
    }
}

#[cfg(test)]
impl GgmlAsrSamplesView<'_> {
    pub(crate) fn range(&self) -> std::ops::Range<usize> {
        match self {
            Self::Borrowed(samples) => 0..samples.len(),
            Self::Shared(samples) => samples.range(),
        }
    }

    pub(crate) fn backing_identity(&self) -> usize {
        match self {
            Self::Borrowed(samples) => samples.as_ptr() as usize,
            Self::Shared(samples) => samples.backing_identity(),
        }
    }
}

impl From<Vec<f32>> for GgmlAsrSamplesView<'static> {
    fn from(samples: Vec<f32>) -> Self {
        Self::Shared(samples.into())
    }
}

impl From<PcmSlice> for GgmlAsrSamplesView<'static> {
    fn from(samples: PcmSlice) -> Self {
        Self::Shared(samples)
    }
}

/// Pure semantic decoder-state plan.
///
/// This value proves token/state geometry only. Physical memory is admitted
/// later, after an execution route has been selected, and its committed lease
/// is owned by the Rust/native allocation it accounts for. Keeping physical
/// ownership out of this cheaply-cloned request value prevents a request guard
/// from refunding memory before the real buffer drops (or charging a
/// resident-only route for a host cache it never allocates).
#[derive(Debug)]
pub(crate) struct GgmlAsrPlannedDecoderState {
    plan: crate::capacity::topology::DecoderStatePlan,
    envelope: crate::capacity::topology::InvocationEnvelope,
}

impl GgmlAsrPlannedDecoderState {
    fn new(
        plan: crate::capacity::topology::DecoderStatePlan,
        envelope: crate::capacity::topology::InvocationEnvelope,
    ) -> Self {
        Self { plan, envelope }
    }
}

impl Deref for GgmlAsrPlannedDecoderState {
    type Target = crate::capacity::topology::DecoderStatePlan;

    fn deref(&self) -> &Self::Target {
        &self.plan
    }
}

impl PartialEq for GgmlAsrPlannedDecoderState {
    fn eq(&self, other: &Self) -> bool {
        self.plan == other.plan && self.envelope == other.envelope
    }
}

impl Eq for GgmlAsrPlannedDecoderState {}

/// The only two legal decoder-state states carried across execution seams.
///
/// `NoPersistentState` is an affirmative family declaration, not an
/// unplanned placeholder. Decoder families must carry a validated plan; an
/// optional plan would make "not planned yet" indistinguishable from "this
/// architecture owns no persistent decoder state".
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum GgmlAsrDecoderState {
    NoPersistentState,
    Planned(Arc<GgmlAsrPlannedDecoderState>),
}

#[cfg(test)]
impl GgmlAsrDecoderState {
    pub(crate) fn planned_for_test(
        plan: crate::capacity::topology::DecoderStatePlan,
        envelope: crate::capacity::topology::InvocationEnvelope,
    ) -> Self {
        Self::Planned(Arc::new(GgmlAsrPlannedDecoderState::new(plan, envelope)))
    }
}

impl GgmlAsrDecoderState {
    pub(crate) fn planned(
        plan: crate::capacity::topology::DecoderStatePlan,
        envelope: crate::capacity::topology::InvocationEnvelope,
    ) -> Self {
        Self::Planned(Arc::new(GgmlAsrPlannedDecoderState::new(plan, envelope)))
    }

    pub(crate) fn invocation_envelope(
        &self,
    ) -> Option<crate::capacity::topology::InvocationEnvelope> {
        match self {
            Self::NoPersistentState => None,
            Self::Planned(state) => Some(state.envelope),
        }
    }

    pub(crate) fn with_resident_demands_from(
        self,
        resident_template: &Self,
    ) -> Result<Self, GgmlAsrDecoderStatePlanningError> {
        match (self, resident_template) {
            (Self::NoPersistentState, Self::NoPersistentState) => Ok(Self::NoPersistentState),
            (Self::Planned(logical), Self::Planned(resident)) => logical
                .plan
                .with_resident_demands_from(&resident.plan)
                .map(|plan| {
                    Self::Planned(Arc::new(GgmlAsrPlannedDecoderState::new(
                        plan,
                        resident.envelope,
                    )))
                })
                .map_err(|source| GgmlAsrDecoderStatePlanningError::ResidentRebind { source }),
            _ => Err(GgmlAsrDecoderStatePlanningError::StateClassChanged),
        }
    }
}

/// Inputs common to every family-owned decoder-state planner. The logical
/// invocation and stable session envelope are deliberately separate: the
/// former drives masks/decode bounds, while only the latter sizes reusable
/// resident storage.
#[derive(Debug, Clone, Copy)]
pub(crate) struct GgmlAsrDecoderStatePlanningInput<'a> {
    pub(crate) preflight: &'a GgufRuntimeSourcePreflight,
    pub(crate) invocation: crate::capacity::topology::InvocationShapeInput,
    pub(crate) envelope: crate::capacity::topology::InvocationEnvelope,
    pub(crate) request_options: &'a GgmlAsrExecutionOptions,
    pub(crate) backend: crate::ggml_runtime::GgmlCpuGraphBackend,
}

impl<'a> GgmlAsrDecoderStatePlanningInput<'a> {
    /// Build the exact current invocation and the largest configured
    /// long-form slice envelope using integer samples. `max_chunk_seconds`
    /// is the ceiling for the fully padded buffer handed to an executor (the
    /// slicer shrinks padding when content reaches that ceiling), so adding
    /// padding again here would overstate the legal invocation envelope.
    #[cfg(test)]
    pub(crate) fn for_offline_request(
        preflight: &'a GgufRuntimeSourcePreflight,
        prepared_audio: &GgmlAsrPreparedAudio,
        request_options: &'a GgmlAsrExecutionOptions,
        backend: crate::ggml_runtime::GgmlCpuGraphBackend,
    ) -> Result<Self, GgmlAsrDecoderStatePlanningError> {
        Self::for_offline_audio_shape(
            preflight,
            prepared_audio.sample_rate_hz,
            prepared_audio.samples_f32.len(),
            request_options,
            backend,
        )
    }

    pub(crate) fn for_offline_view_request(
        preflight: &'a GgufRuntimeSourcePreflight,
        prepared_audio: &GgmlAsrPreparedAudioView<'_>,
        request_options: &'a GgmlAsrExecutionOptions,
        backend: crate::ggml_runtime::GgmlCpuGraphBackend,
    ) -> Result<Self, GgmlAsrDecoderStatePlanningError> {
        Self::for_offline_audio_shape(
            preflight,
            prepared_audio.sample_rate_hz,
            prepared_audio.samples_f32.len(),
            request_options,
            backend,
        )
    }

    fn for_offline_audio_shape(
        preflight: &'a GgufRuntimeSourcePreflight,
        sample_rate_hz: u32,
        sample_count: usize,
        request_options: &'a GgmlAsrExecutionOptions,
        backend: crate::ggml_runtime::GgmlCpuGraphBackend,
    ) -> Result<Self, GgmlAsrDecoderStatePlanningError> {
        let sample_rate_hz = NonZeroU32::new(sample_rate_hz)
            .ok_or(GgmlAsrDecoderStatePlanningError::InvalidSampleRate { sample_rate_hz })?;
        let invocation =
            crate::capacity::topology::InvocationShapeInput::new(sample_rate_hz, sample_count)
                .map_err(|source| GgmlAsrDecoderStatePlanningError::InvalidShape { source })?;
        let configured_samples =
            offline_invocation_envelope_samples(request_options, sample_rate_hz, sample_count)?;
        let max_prompt_tokens = request_options.max_longform_prompt_tokens();
        let envelope =
            crate::capacity::topology::InvocationEnvelope::new(sample_rate_hz, configured_samples)
                .map_err(|source| GgmlAsrDecoderStatePlanningError::InvalidShape { source })?
                .with_max_prompt_tokens(max_prompt_tokens);
        Ok(Self {
            preflight,
            invocation,
            envelope,
            request_options,
            backend,
        })
    }

    /// Streaming snapshot decoders retain at most the shared 30-second
    /// incremental window. Session construction therefore plans that stable
    /// maximum up front; per-frame requests reuse the same resident envelope.
    pub(crate) fn for_streaming_session(
        preflight: &'a GgufRuntimeSourcePreflight,
        request_options: &'a GgmlAsrExecutionOptions,
        backend: crate::ggml_runtime::GgmlCpuGraphBackend,
    ) -> Result<Self, GgmlAsrDecoderStatePlanningError> {
        let sample_rate_hz = NonZeroU32::new(16_000).expect("16 kHz is non-zero");
        let max_prompt_tokens = request_options.max_longform_prompt_tokens();
        let envelope = crate::capacity::topology::InvocationEnvelope::from_milliseconds(
            sample_rate_hz,
            NonZeroU32::new(30_000).expect("30 seconds is non-zero"),
        )
        .map_err(|source| GgmlAsrDecoderStatePlanningError::InvalidShape { source })?
        .with_max_prompt_tokens(max_prompt_tokens);
        Ok(Self {
            preflight,
            invocation: envelope.maximum_invocation(),
            envelope,
            request_options,
            backend,
        })
    }

    pub(crate) fn for_streaming_decode_view(
        preflight: &'a GgufRuntimeSourcePreflight,
        prepared_audio: &GgmlAsrPreparedAudioView<'_>,
        envelope: crate::capacity::topology::InvocationEnvelope,
        request_options: &'a GgmlAsrExecutionOptions,
        backend: crate::ggml_runtime::GgmlCpuGraphBackend,
    ) -> Result<Self, GgmlAsrDecoderStatePlanningError> {
        let sample_rate_hz = NonZeroU32::new(prepared_audio.sample_rate_hz).ok_or(
            GgmlAsrDecoderStatePlanningError::InvalidSampleRate {
                sample_rate_hz: prepared_audio.sample_rate_hz,
            },
        )?;
        let invocation = crate::capacity::topology::InvocationShapeInput::new(
            sample_rate_hz,
            prepared_audio.samples_f32.len(),
        )
        .map_err(|source| GgmlAsrDecoderStatePlanningError::InvalidShape { source })?;
        Ok(Self {
            preflight,
            invocation,
            envelope,
            request_options,
            backend,
        })
    }
}

fn offline_invocation_envelope_samples(
    request_options: &GgmlAsrExecutionOptions,
    sample_rate_hz: NonZeroU32,
    actual_samples: usize,
) -> Result<usize, GgmlAsrDecoderStatePlanningError> {
    let configured_samples = if request_options.longform_mode_enabled() {
        request_options
            .longform
            .as_ref()
            .map(|options| duration_samples_ceil(options.max_chunk_seconds, sample_rate_hz))
            .transpose()?
            .unwrap_or(actual_samples)
    } else {
        actual_samples
    };
    Ok(configured_samples)
}

fn duration_samples_ceil(
    seconds: f32,
    sample_rate_hz: NonZeroU32,
) -> Result<usize, GgmlAsrDecoderStatePlanningError> {
    // Decode the finite positive f32 at the configuration boundary into its
    // exact binary rational, then perform ceil(rate * seconds) in integers.
    // No float is allowed into family topology/oracle arithmetic.
    let bits = seconds.to_bits();
    let sign = bits >> 31;
    let exponent_bits = (bits >> 23) & 0xff;
    let fraction = bits & 0x7f_ff_ff;
    if sign != 0 || exponent_bits == 0xff || (exponent_bits == 0 && fraction == 0) {
        return Err(GgmlAsrDecoderStatePlanningError::InvalidEnvelopeDuration {
            value: seconds.to_string(),
        });
    }
    let (significand, exponent_two): (u128, i32) = if exponent_bits == 0 {
        (u128::from(fraction), -149)
    } else {
        (
            u128::from((1 << 23) | fraction),
            exponent_bits as i32 - 127 - 23,
        )
    };
    let scaled = significand
        .checked_mul(u128::from(sample_rate_hz.get()))
        .ok_or_else(
            || GgmlAsrDecoderStatePlanningError::InvalidEnvelopeDuration {
                value: seconds.to_string(),
            },
        )?;
    let samples = if exponent_two >= 0 {
        scaled.checked_shl(exponent_two as u32).ok_or_else(|| {
            GgmlAsrDecoderStatePlanningError::InvalidEnvelopeDuration {
                value: seconds.to_string(),
            }
        })?
    } else {
        let shift = exponent_two.unsigned_abs();
        if shift >= u128::BITS {
            1
        } else {
            let denominator = 1_u128 << shift;
            scaled.checked_add(denominator - 1).ok_or_else(|| {
                GgmlAsrDecoderStatePlanningError::InvalidEnvelopeDuration {
                    value: seconds.to_string(),
                }
            })? / denominator
        }
    };
    usize::try_from(samples).map_err(|_| {
        GgmlAsrDecoderStatePlanningError::InvalidEnvelopeDuration {
            value: seconds.to_string(),
        }
    })
}

pub(crate) type GgmlAsrDecoderStatePlanner = for<'a> fn(
    &GgmlAsrDecoderStatePlanningInput<'a>,
) -> Result<
    crate::capacity::topology::DecoderStatePlan,
    GgmlAsrDecoderStatePlanningError,
>;

/// Stable identity and semantic kind of one persistent state stream promised
/// by a family planner. Runtime dispatch validates the complete stream set,
/// so a non-empty but semantically wrong plan cannot cross the executor seam.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GgmlAsrDecoderStateStreamContract {
    pub(crate) id: &'static str,
    pub(crate) kind: crate::capacity::topology::StateKind,
}

impl GgmlAsrDecoderStateStreamContract {
    pub(crate) const fn new(id: &'static str, kind: crate::capacity::topology::StateKind) -> Self {
        Self { id, kind }
    }
}

/// Compile-time-required topology declaration returned by every executor.
/// Planner function pointers keep the derivation family-owned while allowing
/// model-agnostic dispatch to invoke it without an architecture switch.
#[derive(Debug, Clone, Copy)]
pub(crate) enum GgmlAsrDecoderStateContract {
    NoPersistentState,
    Planned {
        planner: GgmlAsrDecoderStatePlanner,
        streams: &'static [GgmlAsrDecoderStateStreamContract],
    },
}

impl GgmlAsrDecoderStateContract {
    pub(crate) const fn planned(
        planner: GgmlAsrDecoderStatePlanner,
        streams: &'static [GgmlAsrDecoderStateStreamContract],
    ) -> Self {
        Self::Planned { planner, streams }
    }

    pub(crate) fn plan(
        self,
        input: &GgmlAsrDecoderStatePlanningInput<'_>,
    ) -> Result<GgmlAsrDecoderState, GgmlAsrDecoderStatePlanningError> {
        match self {
            Self::NoPersistentState => Ok(GgmlAsrDecoderState::NoPersistentState),
            Self::Planned { planner, .. } => {
                planner(input).map(|plan| GgmlAsrDecoderState::planned(plan, input.envelope))
            }
        }
    }

    fn validates(self, state: &GgmlAsrDecoderState) -> bool {
        match (self, state) {
            (Self::NoPersistentState, GgmlAsrDecoderState::NoPersistentState) => true,
            (Self::Planned { streams, .. }, GgmlAsrDecoderState::Planned(plan)) => {
                !streams.is_empty()
                    && streams.len() == plan.allocations().len()
                    && streams.iter().enumerate().all(|(index, stream)| {
                        !streams[..index]
                            .iter()
                            .any(|existing| existing.id == stream.id)
                            && plan.allocations().iter().any(|allocation| {
                                allocation.logical.id == stream.id
                                    && allocation.logical.kind == stream.kind
                                    && allocation.reserve.id == stream.id
                                    && allocation.reserve.kind == stream.kind
                            })
                    })
            }
            _ => false,
        }
    }
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub(crate) enum GgmlAsrDecoderStatePlanningError {
    #[error("decoder-state planning received invalid sample rate {sample_rate_hz} Hz")]
    InvalidSampleRate { sample_rate_hz: u32 },
    #[error("decoder-state planning received invalid envelope duration '{value}' seconds")]
    InvalidEnvelopeDuration { value: String },
    #[error("decoder-state planning shape is invalid: {source}")]
    InvalidShape {
        #[source]
        source: crate::capacity::topology::TopologyError,
    },
    #[error("model family '{family}' decoder-state metadata is unavailable: {reason}")]
    MetadataUnavailable {
        family: &'static str,
        reason: String,
    },
    #[error("model family '{family}' exact prompt token count is unavailable: {reason}")]
    PromptTokenCountUnavailable {
        family: &'static str,
        reason: String,
    },
    #[error("model family '{family}' decoder-state topology failed: {source}")]
    Topology {
        family: &'static str,
        #[source]
        source: crate::capacity::topology::TopologyError,
    },
    #[error(
        "streaming decoder-state logical demand no longer fits the admitted resident session envelope: {source}"
    )]
    ResidentRebind {
        #[source]
        source: crate::capacity::topology::TopologyError,
    },
    #[error("streaming decoder-state family changed persistent-state class within one session")]
    StateClassChanged,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct GgmlAsrExecutionOptions {
    pub language: Option<String>,
    /// Speech task. Default `Transcribe` keeps the legacy byte-identical path;
    /// only whisper acts on `Translate` (other families reject it post-selection).
    pub task: TranscriptionTask,
    pub prompt: Option<String>,
    pub prompt_token_ids: Option<Vec<u32>>,
    pub phrase_bias: Option<PhraseBiasConfig>,
    pub inference_threads: Option<usize>,
    pub word_timestamps: bool,
    /// True when `word_timestamps` was forced on solely to obtain word anchors
    /// for VAD diarization (the caller did not request word timestamps). Only
    /// whisper acts on this: it keeps the decode path byte-identical to a
    /// non-diarized run (cross flash attention unchanged, no cross-attention
    /// collection) and derives anchors post hoc from the generated tokens
    /// instead of the higher-fidelity cross-attention alignment.
    pub word_timestamps_forced_for_diarization: bool,
    /// Whether this family's own decode should carry speaker structure. Set
    /// only for a family whose `arch::SpeakerSegmentationSource` is
    /// `InDecoder` and only when the request asked for Voice ID; the external
    /// VAD + speaker-embedder path never sets it, which is what keeps the two
    /// segmentation sources mutually exclusive.
    pub in_decoder_speakers: bool,
    pub longform: Option<LongFormOptions>,
    pub longform_chunk_count_hint: Option<usize>,
    /// Set from the architecture descriptor when the arch signals that multi-chunk
    /// longform on Metal should prefer the CPU decoder path. Avoids per-executor
    /// re-derivation of this policy flag.
    pub prefer_cpu_decoder_for_multichunk_metal: bool,
    /// Server-owned offline batching policy. The CLI and every non-server call
    /// retain `serial`; only the server derives this from its native-session
    /// admission limit.
    pub(crate) serve_batch: crate::models::serve_batch_env::ServeBatchPolicy,
    /// Verified cache identity for reusable native runtime state. Absent until
    /// the pack-content resolver supplies one; executors must not substitute a
    /// path-only identity.
    pub runtime_build_identity: Option<RuntimeBuildIdentity>,
    /// OADP Phase 0: request-level `.oadp` adapter pack path (CLI `--adapter`
    /// plumbs it here). `None` falls back to the server-side `OPENASR_ADAPTER`
    /// process environment variable.
    pub adapter_path: Option<PathBuf>,
}

impl RuntimeBuildIdentitySource for GgmlAsrExecutionOptions {
    fn runtime_build_identity(&self) -> Option<RuntimeBuildIdentity> {
        self.runtime_build_identity.clone()
    }
}

impl GgmlAsrExecutionOptions {
    pub(crate) fn longform_mode_enabled(&self) -> bool {
        self.longform
            .as_ref()
            .is_some_and(|options| !matches!(options.mode, crate::LongFormMode::Off))
    }

    pub(crate) fn longform_prompt_carry_enabled(&self) -> bool {
        self.longform_mode_enabled()
            && self
                .longform
                .as_ref()
                .is_some_and(|options| options.carry_prompt_across_slices)
    }

    pub(crate) fn max_longform_prompt_tokens(&self) -> usize {
        self.longform
            .as_ref()
            .filter(|_| self.longform_prompt_carry_enabled())
            .map_or(0, |options| options.max_context_tokens)
    }

    pub fn from_transcription_request(
        language: Option<String>,
        prompt: Option<String>,
        longform: Option<LongFormOptions>,
    ) -> Self {
        Self::from_transcription_request_with_phrase_bias(language, prompt, None, longform)
    }

    pub fn from_transcription_request_with_phrase_bias(
        language: Option<String>,
        prompt: Option<String>,
        phrase_bias: Option<PhraseBiasConfig>,
        longform: Option<LongFormOptions>,
    ) -> Self {
        Self {
            language,
            task: TranscriptionTask::default(),
            prompt,
            prompt_token_ids: None,
            phrase_bias,
            inference_threads: None,
            word_timestamps: false,
            word_timestamps_forced_for_diarization: false,
            in_decoder_speakers: false,
            longform,
            longform_chunk_count_hint: None,
            prefer_cpu_decoder_for_multichunk_metal: false,
            serve_batch: crate::models::serve_batch_env::ServeBatchPolicy::serial(),
            runtime_build_identity: None,
            adapter_path: None,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct GgmlAsrCarryContext {
    pub prompt_text: Option<String>,
    pub prompt_token_ids: Option<Vec<u32>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GgmlAsrExecutionRequest {
    /// Process-owned execution state that admitted and owns every resource
    /// used by this request. Required so dispatch, cached weights, and memory
    /// accounting cannot silently come from different ambient singletons.
    pub execution_services: Arc<crate::models::native_execution_services::NativeExecutionServices>,
    /// Capacity-planner output for decoder-resident state. The topology
    /// integration fills this after request/session planning is wired.
    pub(crate) decoder_state: GgmlAsrDecoderState,
    /// Package/runtime proof built at the untrusted-path ingress. Carrying the
    /// full proof, rather than its structural preflight projection, prevents
    /// callers from constructing an executable request from an arbitrary GGUF.
    pub verified_pack: crate::models::pack_verifier::VerifiedPack,
    pub selected_family: GgmlFamilyAdapterDescriptor,
    pub prepared_audio: GgmlAsrPreparedAudio,
    pub request_options: GgmlAsrExecutionOptions,
    /// The caller's raw execution-target choice. Still consulted by a few
    /// pre-existing, unrelated thread-local readers that install/read the
    /// override directly (the longform multichunk-metal probe, a family's
    /// own post-hoc RAM-fit check) -- but the family's own resolved backend
    /// is carried on `resolved_runtime` below, not derived from this field
    /// via any thread-local at decode time.
    pub backend_preference: GgmlAsrBackendPreference,
    /// This family's backend, already resolved from `backend_preference` and
    /// the family's own `AutoGpuPolicy` (see
    /// [`crate::ggml_runtime::ResolvedFamilyRuntimeInput::resolve`]) by
    /// whoever built this request. A required, explicit field -- not a
    /// thread-local an executor reads out of band -- so every graph-build
    /// call site an executor threads this value to (directly, or via a
    /// sub-request/job object copying it forward) observes the identical
    /// value the request was built with, including across an OS-thread
    /// boundary such as qwen's serve-batch worker.
    pub resolved_runtime: crate::ggml_runtime::ResolvedFamilyRuntimeInput,
    /// Cancel/pause/resume control and request id for this decode, carried
    /// explicitly rather than through the (removed) thread-local
    /// transcription control. Required: a caller with nothing to cancel
    /// still passes `RequestExecutionContext::uncancellable(reason)`.
    pub execution_context: Arc<RequestExecutionContext>,
}

/// Runtime request used inside the built-in dispatch.
///
/// This is a deep internal seam: all non-audio request state keeps the same
/// shape as [`GgmlAsrExecutionRequest`], while audio may either borrow the
/// public owned DTO or retain a shared native PCM range. Keeping that choice
/// here prevents storage ownership from leaking into every model adapter.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct GgmlAsrExecutionViewRequest<'a> {
    pub(crate) execution_services:
        Arc<crate::models::native_execution_services::NativeExecutionServices>,
    pub(crate) decoder_state: GgmlAsrDecoderState,
    pub(crate) verified_pack: crate::models::pack_verifier::VerifiedPack,
    pub(crate) selected_family: GgmlFamilyAdapterDescriptor,
    pub(crate) prepared_audio: GgmlAsrPreparedAudioView<'a>,
    pub(crate) request_options: GgmlAsrExecutionOptions,
    pub(crate) backend_preference: GgmlAsrBackendPreference,
    pub(crate) resolved_runtime: crate::ggml_runtime::ResolvedFamilyRuntimeInput,
    pub(crate) execution_context: Arc<RequestExecutionContext>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GgmlAsrStreamingSessionConfig {
    pub audio_format: RealtimeAudioFormat,
    pub backpressure: NativeAsrBackpressurePolicy,
    pub partial_results: bool,
    pub word_timestamps: bool,
    pub min_partial_interval_ms: Option<u32>,
}

impl GgmlAsrStreamingSessionConfig {
    /// Effective partial-decode floor (ms): the client override if set, else the
    /// per-family default. Fed only to `PartialDecodeCadence`, which gates PARTIAL
    /// re-decodes (never the FINAL), so it cannot affect transcript parity.
    pub(crate) fn partial_floor_ms(&self, family_default: u32) -> u64 {
        u64::from(self.min_partial_interval_ms.unwrap_or(family_default))
    }
}

impl From<crate::NativeAsrStreamingSessionConfig> for GgmlAsrStreamingSessionConfig {
    fn from(config: crate::NativeAsrStreamingSessionConfig) -> Self {
        Self {
            audio_format: config.audio_format,
            backpressure: config.backpressure,
            partial_results: config.partial_results,
            word_timestamps: config.word_timestamps,
            min_partial_interval_ms: config.min_partial_interval_ms,
        }
    }
}

type GgmlAsrStreamingFinalTextProcessor =
    Arc<dyn Fn(&str) -> Result<String, String> + Send + Sync + 'static>;

/// A session-stable seam through which an auxiliary FINAL-text runtime is
/// installed after the primary ASR session has constructed successfully.
///
/// The slot is shared by every semantics-equivalent reconstruction of the ASR
/// session. This lets ASR warm-up move to a different candidate without
/// rebuilding or rebinding the auxiliary runtime, while still guaranteeing
/// that no audio is accepted before the owning policy wrapper initializes the
/// slot.
#[derive(Clone, Default)]
pub(crate) struct GgmlAsrStreamingFinalTextProcessorSlot {
    processor: Arc<Mutex<Option<GgmlAsrStreamingFinalTextProcessor>>>,
}

impl GgmlAsrStreamingFinalTextProcessorSlot {
    pub(crate) fn install(
        &self,
        processor: GgmlAsrStreamingFinalTextProcessor,
    ) -> Result<(), &'static str> {
        let mut current = self
            .processor
            .lock()
            .map_err(|_| "streaming final-text processor slot is poisoned")?;
        if current.is_some() {
            return Err("streaming final-text processor is already installed");
        }
        *current = Some(processor);
        Ok(())
    }

    pub(crate) fn process(&self, text: &str) -> Result<String, String> {
        let current = self
            .processor
            .lock()
            .map_err(|_| "streaming final-text processor slot is poisoned".to_string())?;
        match current.as_ref() {
            Some(processor) => processor(text),
            // Construction intentionally precedes auxiliary initialization.
            // The policy wrapper initializes before audio; preserving the
            // input here keeps a direct low-level test/session caller safe.
            None => Ok(text.to_string()),
        }
    }
}

impl std::fmt::Debug for GgmlAsrStreamingFinalTextProcessorSlot {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("GgmlAsrStreamingFinalTextProcessorSlot")
            .field(
                "installed",
                &self
                    .processor
                    .lock()
                    .map(|processor| processor.is_some())
                    .ok(),
            )
            .finish()
    }
}

impl PartialEq for GgmlAsrStreamingFinalTextProcessorSlot {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.processor, &other.processor)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct GgmlAsrStreamingSessionRequest {
    /// The same explicit service root must be copied into every per-frame
    /// execution request built for this session.
    pub execution_services: Arc<crate::models::native_execution_services::NativeExecutionServices>,
    pub(crate) decoder_state: GgmlAsrDecoderState,
    /// Required on every production session; see
    /// [`GgmlAsrExecutionRequest::verified_pack`].
    pub verified_pack: crate::models::pack_verifier::VerifiedPack,
    pub selected_family: GgmlFamilyAdapterDescriptor,
    pub request_options: GgmlAsrExecutionOptions,
    pub configured_diarize: bool,
    pub backend_preference: GgmlAsrBackendPreference,
    /// This family's backend, resolved once for the whole session by
    /// whoever built this request (see `GgmlAsrExecutionRequest::resolved_runtime`'s
    /// doc comment for why this is a required field, not a thread-local).
    /// The shared streaming drivers copy it into every per-frame
    /// `GgmlAsrExecutionRequest` they build for the life of the session.
    pub resolved_runtime: crate::ggml_runtime::ResolvedFamilyRuntimeInput,
    /// Optional session-stable auxiliary FINAL-text processor. It has its own
    /// execution plan/lane and is never derived from `resolved_runtime`.
    pub(crate) final_text_processor: Option<GgmlAsrStreamingFinalTextProcessorSlot>,
    pub session_context: crate::NativeAsrSessionContext,
    pub session_config: GgmlAsrStreamingSessionConfig,
}

impl GgmlAsrExecutionRequest {
    pub fn runtime_source_preflight(&self) -> &GgufRuntimeSourcePreflight {
        self.verified_pack.preflight()
    }

    pub(crate) fn as_view(&self) -> GgmlAsrExecutionViewRequest<'_> {
        GgmlAsrExecutionViewRequest {
            execution_services: Arc::clone(&self.execution_services),
            decoder_state: self.decoder_state.clone(),
            verified_pack: self.verified_pack.clone(),
            selected_family: self.selected_family.clone(),
            prepared_audio: self.prepared_audio.as_view(),
            request_options: self.request_options.clone(),
            backend_preference: self.backend_preference,
            resolved_runtime: self.resolved_runtime,
            execution_context: Arc::clone(&self.execution_context),
        }
    }
}

impl GgmlAsrExecutionViewRequest<'_> {
    pub(crate) fn runtime_source_preflight(&self) -> &GgufRuntimeSourcePreflight {
        self.verified_pack.preflight()
    }

    fn to_owned_request(&self) -> GgmlAsrExecutionRequest {
        GgmlAsrExecutionRequest {
            execution_services: Arc::clone(&self.execution_services),
            decoder_state: self.decoder_state.clone(),
            verified_pack: self.verified_pack.clone(),
            selected_family: self.selected_family.clone(),
            prepared_audio: GgmlAsrPreparedAudio {
                sample_rate_hz: self.prepared_audio.sample_rate_hz,
                channels: self.prepared_audio.channels,
                samples_f32: self.prepared_audio.samples_f32.to_vec(),
            },
            request_options: self.request_options.clone(),
            backend_preference: self.backend_preference,
            resolved_runtime: self.resolved_runtime,
            execution_context: Arc::clone(&self.execution_context),
        }
    }
}

impl GgmlAsrStreamingSessionRequest {
    pub fn runtime_source_preflight(&self) -> &GgufRuntimeSourcePreflight {
        self.verified_pack.preflight()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct GgmlAsrExecutionResult {
    pub transcription: Transcription,
    pub carry_context: Option<GgmlAsrCarryContext>,
    /// Set when this decode stopped short of the audio it was given.
    ///
    /// A truncated decode is otherwise indistinguishable from a complete one --
    /// same shape, same success status -- so without this the caller cannot
    /// tell a transcript that covers its audio from one that gave up partway.
    /// Both the long-form loop and the single-pass path stamp it onto the
    /// returned [`Transcription`], and it is the signal a slice-level retry or
    /// degrade would key on. `None` means the decode ended on its own terms.
    ///
    /// Every seq2seq family derives this from the shared driver's stop reason
    /// via `Seq2SeqGreedyDecodeStopReason::into_decode_truncation`; CTC and
    /// transducer families never reach the greedy driver's guard and leave it
    /// `None`.
    pub decode_truncation: Option<DecodeTruncation>,
}

impl GgmlAsrExecutionResult {
    pub fn into_transcription(self) -> Transcription {
        self.transcription
    }
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum GgmlAsrExecutionError {
    #[error(
        "ggml execution capability is unsupported for adapter '{adapter_id}': backend preference '{backend_preference}'"
    )]
    UnsupportedCapability {
        adapter_id: &'static str,
        backend_preference: &'static str,
    },
    #[error(
        "no ggml executor is registered for adapter '{adapter_id}' (family '{model_family}') and capability '{capability}'"
    )]
    ExecutorUnavailable {
        adapter_id: &'static str,
        model_family: &'static str,
        capability: &'static str,
    },
    #[error(
        "verified pack route does not match selected family '{model_family}' (architecture '{model_architecture}')"
    )]
    VerifiedPackRouteMismatch {
        model_family: &'static str,
        model_architecture: &'static str,
    },
    #[allow(private_interfaces)]
    #[error(transparent)]
    DecoderStatePlanning(#[from] GgmlAsrDecoderStatePlanningError),
    #[error(
        "executor '{executor_id}' decoder-state contract does not match the state carried by adapter '{adapter_id}'"
    )]
    DecoderStateContractMismatch {
        executor_id: &'static str,
        adapter_id: &'static str,
    },
    #[error(
        "phrase bias / hotword boosting is unsupported for adapter '{adapter_id}' (family '{model_family}')"
    )]
    PhraseBiasUnsupported {
        adapter_id: &'static str,
        model_family: &'static str,
    },
    #[error("ggml executor '{executor_id}' failed for adapter '{adapter_id}': {reason}")]
    ExecutorFailed {
        executor_id: &'static str,
        adapter_id: &'static str,
        reason: String,
    },
    /// OADP Phase 0: an adapter is active (request `--adapter` or the
    /// server-side `OPENASR_ADAPTER` env var) but the selected family has no
    /// dynamic adapter support. Fail-closed: an adapter the user asked for is
    /// never silently ignored.
    #[error(
        "an adapter pack is active ('{adapter_path}') but model family '{model_family}' does not \
         implement an adapter-binding strategy; fail-closed"
    )]
    AdapterUnsupportedForFamily {
        model_family: &'static str,
        adapter_path: String,
    },
    #[error(
        "adapter binding contract mismatch for family '{model_family}': descriptor declares '{declared}', executor provides '{provided}'"
    )]
    AdapterBindingContractMismatch {
        model_family: &'static str,
        declared: &'static str,
        provided: &'static str,
    },
    /// Typed Exact/preferred device failure from graph backend init. Kept as a
    /// first-class variant so `dispatch_error_to_backend` can surface
    /// `BackendError::ExecutionDevice*` without string recovery.
    #[error(transparent)]
    ExecutionRoute(#[from] crate::device::execution_route::ExecutionRouteError),
    /// A transient serve-batch failure (queue saturation / owner gone / GPU step
    /// hung) carried out of the executor so the backend can map it to a retryable
    /// HTTP status instead of a generic 500. `retryable == true` => queue full
    /// (429); `retryable == false` => owner disconnected / reply timed out (503).
    #[error("{reason}")]
    ServeBatchUnavailable { reason: String, retryable: bool },
}

impl GgmlAsrExecutionError {
    pub(crate) fn executor_failed(
        executor_id: &'static str,
        adapter_id: &'static str,
        reason: impl Into<String>,
    ) -> Self {
        Self::ExecutorFailed {
            executor_id,
            adapter_id,
            reason: reason.into(),
        }
    }

    /// Preserve typed route failures from graph init; stringify everything else.
    /// Prefer this at family `GgmlCpuGraphError` boundaries so dispatch does not
    /// need Display recovery. Covered by unit tests; production call sites migrate
    /// family-by-family.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn from_ggml_cpu_graph_error(
        executor_id: &'static str,
        adapter_id: &'static str,
        error: crate::ggml_runtime::GgmlCpuGraphError,
    ) -> Self {
        match error {
            crate::ggml_runtime::GgmlCpuGraphError::ExecutionRoute(error) => {
                Self::ExecutionRoute(error)
            }
            other => Self::executor_failed(executor_id, adapter_id, other.to_string()),
        }
    }
}

#[allow(private_interfaces)]
pub trait GgmlAsrExecutor: Send + Sync {
    fn executor_id(&self) -> &'static str;
    fn adapter_binding_strategy(&self) -> GgmlAdapterBindingStrategy {
        GgmlAdapterBindingStrategy::Unsupported
    }
    fn supports_phrase_bias(&self) -> bool;
    /// Mandatory family-owned persistent-state declaration. There is no
    /// default: onboarding an executor cannot compile until it explicitly
    /// chooses no state, causal self-KV, or encoder-decoder self+cross KV and
    /// supplies the corresponding planner.
    fn decoder_state_contract(
        &self,
        selected_family: &GgmlFamilyAdapterDescriptor,
    ) -> Result<GgmlAsrDecoderStateContract, GgmlAsrExecutionError>;
    /// Recompute one streaming snapshot's logical demand. The default invokes
    /// the family contract directly. Families whose exact prompt oracle owns
    /// a large tokenizer may override this to borrow the already-admitted
    /// prepared runtime; the operation must remain a non-building cache probe.
    fn replan_streaming_decoder_state(
        &self,
        selected_family: &GgmlFamilyAdapterDescriptor,
        input: &GgmlAsrDecoderStatePlanningInput<'_>,
    ) -> Result<GgmlAsrDecoderState, GgmlAsrExecutionError> {
        self.decoder_state_contract(selected_family)?
            .plan(input)
            .map_err(Into::into)
    }
    fn execute(
        &self,
        request: &GgmlAsrExecutionRequest,
    ) -> Result<GgmlAsrExecutionResult, GgmlAsrExecutionError>;
    /// Drops this executor's process-lifetime cached prepared runtime(s)
    /// (mmap + materialized tensors + Metal/CPU graph context), if it caches
    /// one at all. Called by the daemon's idle-unload reaper (`idle_unload`
    /// preference). Resident mutable runtimes are service-owned actors, so
    /// every caching executor must clear its own owners here; the default
    /// no-op is only for executors with no resident state.
    fn unload_idle_state(&self) {}
}

/// Required zero-copy contract for executors owned by the built-in runtime.
///
/// This trait deliberately stays crate-private. Public extensions continue to
/// implement the unchanged owned [`GgmlAsrExecutor`] contract; dispatch stores
/// those in a compatibility slot and materializes owned PCM only when an
/// internal shared view must cross that extension boundary. Built-ins cannot
/// enter the native registry without implementing this view contract.
pub(crate) trait GgmlAsrViewExecutor: Send + Sync {
    fn executor_id(&self) -> &'static str;
    fn adapter_binding_strategy(&self) -> GgmlAdapterBindingStrategy {
        GgmlAdapterBindingStrategy::Unsupported
    }
    /// Resolve adapter binding after architecture selection.
    ///
    /// Most executors own exactly one architecture and inherit the default.
    /// A composed capability executor must override this method and delegate
    /// to the selected child; one wrapper-wide strategy cannot truthfully
    /// represent children such as Qwen and Moonshine.
    fn adapter_binding_strategy_for(
        &self,
        _selected_family: &GgmlFamilyAdapterDescriptor,
    ) -> Result<GgmlAdapterBindingStrategy, GgmlAsrExecutionError> {
        Ok(self.adapter_binding_strategy())
    }
    #[cfg_attr(not(test), allow(dead_code))]
    fn supports_phrase_bias(&self) -> bool;
    fn decoder_state_contract(
        &self,
        selected_family: &GgmlFamilyAdapterDescriptor,
    ) -> Result<GgmlAsrDecoderStateContract, GgmlAsrExecutionError>;
    fn replan_streaming_decoder_state(
        &self,
        selected_family: &GgmlFamilyAdapterDescriptor,
        input: &GgmlAsrDecoderStatePlanningInput<'_>,
    ) -> Result<GgmlAsrDecoderState, GgmlAsrExecutionError> {
        self.decoder_state_contract(selected_family)?
            .plan(input)
            .map_err(Into::into)
    }
    fn execute_view(
        &self,
        request: &GgmlAsrExecutionViewRequest<'_>,
    ) -> Result<GgmlAsrExecutionResult, GgmlAsrExecutionError>;
    /// Evicts prepared state for exactly one admitted pack identity.
    ///
    /// This is required for every built-in view executor: adding a family
    /// cannot compile until it explicitly implements content replacement.
    /// Implementations must target this exact identity; coarse whole-family
    /// eviction is not a supported built-in strategy.
    fn evict_prepared_runtime_content_id(&self, pack_content_id: &str);
    fn unload_idle_state(&self) {}
}

enum GgmlAsrExecutorSlot {
    OwnedCompatibility(Arc<dyn GgmlAsrExecutor>),
    SharedView(Arc<dyn GgmlAsrViewExecutor>),
}

impl GgmlAsrExecutorSlot {
    fn executor_id(&self) -> &'static str {
        match self {
            Self::OwnedCompatibility(executor) => executor.executor_id(),
            Self::SharedView(executor) => executor.executor_id(),
        }
    }

    fn adapter_binding_strategy(
        &self,
        selected_family: &GgmlFamilyAdapterDescriptor,
    ) -> Result<GgmlAdapterBindingStrategy, GgmlAsrExecutionError> {
        match self {
            Self::OwnedCompatibility(executor) => Ok(executor.adapter_binding_strategy()),
            Self::SharedView(executor) => executor.adapter_binding_strategy_for(selected_family),
        }
    }

    fn decoder_state_contract(
        &self,
        selected_family: &GgmlFamilyAdapterDescriptor,
    ) -> Result<GgmlAsrDecoderStateContract, GgmlAsrExecutionError> {
        match self {
            Self::OwnedCompatibility(executor) => executor.decoder_state_contract(selected_family),
            Self::SharedView(executor) => executor.decoder_state_contract(selected_family),
        }
    }

    fn replan_streaming_decoder_state(
        &self,
        selected_family: &GgmlFamilyAdapterDescriptor,
        input: &GgmlAsrDecoderStatePlanningInput<'_>,
    ) -> Result<GgmlAsrDecoderState, GgmlAsrExecutionError> {
        match self {
            Self::OwnedCompatibility(executor) => {
                executor.replan_streaming_decoder_state(selected_family, input)
            }
            Self::SharedView(executor) => {
                executor.replan_streaming_decoder_state(selected_family, input)
            }
        }
    }

    fn execute_owned(
        &self,
        request: &GgmlAsrExecutionRequest,
    ) -> Result<GgmlAsrExecutionResult, GgmlAsrExecutionError> {
        match self {
            Self::OwnedCompatibility(executor) => executor.execute(request),
            Self::SharedView(executor) => executor.execute_view(&request.as_view()),
        }
    }

    fn execute_view(
        &self,
        request: &GgmlAsrExecutionViewRequest<'_>,
    ) -> Result<GgmlAsrExecutionResult, GgmlAsrExecutionError> {
        match self {
            Self::OwnedCompatibility(executor) => executor.execute(&request.to_owned_request()),
            Self::SharedView(executor) => executor.execute_view(request),
        }
    }

    fn unload_idle_state(&self) {
        match self {
            Self::OwnedCompatibility(executor) => executor.unload_idle_state(),
            Self::SharedView(executor) => executor.unload_idle_state(),
        }
    }

    fn evict_prepared_runtime_content_id(&self, pack_content_id: &str) {
        match self {
            // Public compatibility executors do not expose the built-in
            // content-id Interface. Coarse unload remains the safe fallback.
            Self::OwnedCompatibility(executor) => executor.unload_idle_state(),
            Self::SharedView(executor) => {
                executor.evict_prepared_runtime_content_id(pack_content_id)
            }
        }
    }
}

pub trait GgmlAsrStreamingExecutor: Send + Sync {
    fn executor_id(&self) -> &'static str;
    fn adapter_binding_strategy(&self) -> GgmlAdapterBindingStrategy {
        GgmlAdapterBindingStrategy::Unsupported
    }
    fn start_streaming_session(
        &self,
        request: &GgmlAsrStreamingSessionRequest,
    ) -> Result<Box<dyn NativeAsrSession>, GgmlAsrExecutionError>;
    /// Streaming-side counterpart of [`GgmlAsrExecutor::unload_idle_state`].
    /// Families registered on both dispatches (offline + streaming) hold two
    /// independent executor instances with two independent caches, so both
    /// must be evicted for `idle_unload` to actually free the resident model.
    fn unload_idle_state(&self) {}
}

/// Partial-result granularity of a registered streaming executor. Declared on
/// the architecture descriptor and derived into this dispatch at
/// builtin registration time -- see [`crate::arch::StreamingPartialGranularity`].
pub use crate::arch::StreamingPartialGranularity;

#[derive(Default)]
pub struct GgmlAsrExecutionDispatch {
    executors_by_adapter_id: BTreeMap<&'static str, GgmlAsrExecutorSlot>,
    executors_by_capability: BTreeMap<&'static str, GgmlAsrExecutorSlot>,
    streaming_executors_by_adapter_id: BTreeMap<&'static str, Arc<dyn GgmlAsrStreamingExecutor>>,
    streaming_executors_by_capability: BTreeMap<&'static str, Arc<dyn GgmlAsrStreamingExecutor>>,
    streaming_partial_granularity_by_adapter_id:
        BTreeMap<&'static str, StreamingPartialGranularity>,
    streaming_partial_granularity_by_capability:
        BTreeMap<&'static str, StreamingPartialGranularity>,
}

impl GgmlAsrExecutionDispatch {
    pub fn with_executor_for_adapter(
        mut self,
        adapter_id: &'static str,
        executor: Arc<dyn GgmlAsrExecutor>,
    ) -> Self {
        self.executors_by_adapter_id.insert(
            adapter_id,
            GgmlAsrExecutorSlot::OwnedCompatibility(executor),
        );
        self
    }

    pub(crate) fn with_view_executor_for_adapter(
        mut self,
        adapter_id: &'static str,
        executor: Arc<dyn GgmlAsrViewExecutor>,
    ) -> Self {
        self.executors_by_adapter_id
            .insert(adapter_id, GgmlAsrExecutorSlot::SharedView(executor));
        self
    }

    pub fn with_executor_for_capability(
        mut self,
        capability: GgmlExecutionCapability,
        executor: Arc<dyn GgmlAsrExecutor>,
    ) -> Self {
        self.executors_by_capability.insert(
            capability_label(capability),
            GgmlAsrExecutorSlot::OwnedCompatibility(executor),
        );
        self
    }

    pub(crate) fn with_view_executor_for_capability(
        mut self,
        capability: GgmlExecutionCapability,
        executor: Arc<dyn GgmlAsrViewExecutor>,
    ) -> Self {
        self.executors_by_capability.insert(
            capability_label(capability),
            GgmlAsrExecutorSlot::SharedView(executor),
        );
        self
    }

    pub fn with_streaming_executor_for_adapter(
        mut self,
        adapter_id: &'static str,
        executor: Arc<dyn GgmlAsrStreamingExecutor>,
    ) -> Self {
        self.streaming_executors_by_adapter_id
            .insert(adapter_id, executor);
        self
    }

    pub fn with_streaming_executor_for_capability(
        mut self,
        capability: GgmlExecutionCapability,
        executor: Arc<dyn GgmlAsrStreamingExecutor>,
    ) -> Self {
        self.streaming_executors_by_capability
            .insert(capability_label(capability), executor);
        self
    }

    /// Declares the partial-result granularity of the streaming executor
    /// registered for `adapter_id`. This is orthogonal to (and does not
    /// require) registering the executor itself here -- it only records the
    /// granularity fact so capability derivation can answer
    /// [`Self::is_frame_sync_for`] without touching model-family code.
    pub fn with_streaming_partial_granularity_for_adapter(
        mut self,
        adapter_id: &'static str,
        granularity: StreamingPartialGranularity,
    ) -> Self {
        self.streaming_partial_granularity_by_adapter_id
            .insert(adapter_id, granularity);
        self
    }

    /// Capability-keyed counterpart of
    /// [`Self::with_streaming_partial_granularity_for_adapter`], mirroring the
    /// adapter-id/capability duality used by the executor maps above.
    pub fn with_streaming_partial_granularity_for_capability(
        mut self,
        capability: GgmlExecutionCapability,
        granularity: StreamingPartialGranularity,
    ) -> Self {
        self.streaming_partial_granularity_by_capability
            .insert(capability_label(capability), granularity);
        self
    }

    pub fn with_native_graph_lowering_v1(mut self, executor: Arc<dyn GgmlAsrExecutor>) -> Self {
        self = self
            .with_executor_for_capability(GgmlExecutionCapability::NativeGraphLoweringV1, executor);
        self
    }

    pub fn execute(
        &self,
        request: &GgmlAsrExecutionRequest,
    ) -> Result<GgmlAsrExecutionResult, GgmlAsrExecutionError> {
        let _execution_scope =
            crate::models::native_execution_services::install_native_execution_services(
                request.execution_services.as_ref(),
            );
        ensure_verified_pack_matches_family(&request.verified_pack, &request.selected_family)?;
        // Honor the request's execution preference for the few remaining
        // thread-local readers unrelated to backend resolution proper (the
        // longform multichunk-metal probe, a family's own post-hoc RAM-fit
        // check): this override is what makes execution_target truthful for
        // them. The family's own resolved backend is NOT computed here --
        // it already arrived as the required, explicit `request.resolved_runtime`
        // field, filled in by whoever built this request.
        let attempt_override =
            crate::models::native_execution_services::current_execution_placement()
                .and_then(|_| request_backend_override());
        let _backend_guard = install_request_backend_override(
            attempt_override.or_else(|| request.backend_preference.request_backend_override()),
        );

        if let Some(executor) = self.executor_for(&request.selected_family) {
            ensure_adapter_binding_for_executor(
                &request.selected_family,
                executor.adapter_binding_strategy(&request.selected_family)?,
                request.request_options.adapter_path.as_deref(),
            )?;
            ensure_dispatch_not_canceled(
                &request.execution_context,
                executor.executor_id(),
                request.selected_family.adapter_id,
            )?;
            let contract = executor.decoder_state_contract(&request.selected_family)?;
            if !contract.validates(&request.decoder_state) {
                return Err(GgmlAsrExecutionError::DecoderStateContractMismatch {
                    executor_id: executor.executor_id(),
                    adapter_id: request.selected_family.adapter_id,
                });
            }
            let result = executor.execute_owned(request)?;
            ensure_dispatch_not_canceled(
                &request.execution_context,
                executor.executor_id(),
                request.selected_family.adapter_id,
            )?;
            return Ok(result);
        }

        Err(GgmlAsrExecutionError::ExecutorUnavailable {
            adapter_id: request.selected_family.adapter_id,
            model_family: request.selected_family.model_family,
            capability: capability_label(request.selected_family.execution_capability),
        })
    }

    pub(crate) fn execute_view(
        &self,
        request: &GgmlAsrExecutionViewRequest<'_>,
    ) -> Result<GgmlAsrExecutionResult, GgmlAsrExecutionError> {
        let _execution_scope =
            crate::models::native_execution_services::install_native_execution_services(
                request.execution_services.as_ref(),
            );
        ensure_verified_pack_matches_family(&request.verified_pack, &request.selected_family)?;
        let attempt_override =
            crate::models::native_execution_services::current_execution_placement()
                .and_then(|_| request_backend_override());
        let _backend_guard = install_request_backend_override(
            attempt_override.or_else(|| request.backend_preference.request_backend_override()),
        );

        if let Some(executor) = self.executor_for(&request.selected_family) {
            ensure_adapter_binding_for_executor(
                &request.selected_family,
                executor.adapter_binding_strategy(&request.selected_family)?,
                request.request_options.adapter_path.as_deref(),
            )?;
            ensure_dispatch_not_canceled(
                &request.execution_context,
                executor.executor_id(),
                request.selected_family.adapter_id,
            )?;
            let contract = executor.decoder_state_contract(&request.selected_family)?;
            if !contract.validates(&request.decoder_state) {
                return Err(GgmlAsrExecutionError::DecoderStateContractMismatch {
                    executor_id: executor.executor_id(),
                    adapter_id: request.selected_family.adapter_id,
                });
            }
            let result = executor.execute_view(request)?;
            ensure_dispatch_not_canceled(
                &request.execution_context,
                executor.executor_id(),
                request.selected_family.adapter_id,
            )?;
            return Ok(result);
        }

        Err(GgmlAsrExecutionError::ExecutorUnavailable {
            adapter_id: request.selected_family.adapter_id,
            model_family: request.selected_family.model_family,
            capability: capability_label(request.selected_family.execution_capability),
        })
    }

    fn executor_for(
        &self,
        descriptor: &GgmlFamilyAdapterDescriptor,
    ) -> Option<&GgmlAsrExecutorSlot> {
        self.executors_by_adapter_id
            .get(descriptor.adapter_id)
            .or_else(|| {
                self.executors_by_capability
                    .get(capability_label(descriptor.execution_capability))
            })
    }

    /// Invoke the selected executor's family-owned planner. Dispatch only
    /// selects the registered component; it contains no model-family switch
    /// and no metadata constants.
    pub(crate) fn plan_decoder_state(
        &self,
        descriptor: &GgmlFamilyAdapterDescriptor,
        input: &GgmlAsrDecoderStatePlanningInput<'_>,
    ) -> Result<GgmlAsrDecoderState, GgmlAsrExecutionError> {
        let executor =
            self.executor_for(descriptor)
                .ok_or(GgmlAsrExecutionError::ExecutorUnavailable {
                    adapter_id: descriptor.adapter_id,
                    model_family: descriptor.model_family,
                    capability: capability_label(descriptor.execution_capability),
                })?;
        executor
            .decoder_state_contract(descriptor)?
            .plan(input)
            .map_err(Into::into)
    }

    pub(crate) fn replan_streaming_decoder_state(
        &self,
        descriptor: &GgmlFamilyAdapterDescriptor,
        input: &GgmlAsrDecoderStatePlanningInput<'_>,
    ) -> Result<GgmlAsrDecoderState, GgmlAsrExecutionError> {
        let executor =
            self.executor_for(descriptor)
                .ok_or(GgmlAsrExecutionError::ExecutorUnavailable {
                    adapter_id: descriptor.adapter_id,
                    model_family: descriptor.model_family,
                    capability: capability_label(descriptor.execution_capability),
                })?;
        executor.replan_streaming_decoder_state(descriptor, input)
    }

    pub fn start_streaming_session(
        &self,
        request: &GgmlAsrStreamingSessionRequest,
    ) -> Result<Box<dyn NativeAsrSession>, GgmlAsrExecutionError> {
        let _execution_scope =
            crate::models::native_execution_services::install_native_execution_services(
                request.execution_services.as_ref(),
            );
        ensure_verified_pack_matches_family(&request.verified_pack, &request.selected_family)?;
        // Same reasoning as `execute` above: the family's resolved backend
        // is `request.resolved_runtime`, filled in by whoever built this
        // session request. The shared streaming drivers copy that value
        // into every per-frame `GgmlAsrExecutionRequest` they build for the
        // life of the session (see `build_streaming_driver`/
        // `build_ctc_streaming_driver`), so no re-resolution is needed here.
        if let Some(executor) = self
            .streaming_executors_by_adapter_id
            .get(request.selected_family.adapter_id)
        {
            ensure_adapter_binding_for_executor(
                &request.selected_family,
                executor.adapter_binding_strategy(),
                request.request_options.adapter_path.as_deref(),
            )?;
            return executor.start_streaming_session(request);
        }

        if let Some(executor) = self.streaming_executors_by_capability.get(capability_label(
            request.selected_family.execution_capability,
        )) {
            ensure_adapter_binding_for_executor(
                &request.selected_family,
                executor.adapter_binding_strategy(),
                request.request_options.adapter_path.as_deref(),
            )?;
            return executor.start_streaming_session(request);
        }

        Err(GgmlAsrExecutionError::ExecutorUnavailable {
            adapter_id: request.selected_family.adapter_id,
            model_family: request.selected_family.model_family,
            capability: capability_label(request.selected_family.execution_capability),
        })
    }

    pub fn has_streaming_executor_for(&self, descriptor: &GgmlFamilyAdapterDescriptor) -> bool {
        self.streaming_executors_by_adapter_id
            .contains_key(descriptor.adapter_id)
            || self
                .streaming_executors_by_capability
                .contains_key(capability_label(descriptor.execution_capability))
    }

    #[cfg(test)]
    pub(crate) fn streaming_executor_id_for(
        &self,
        descriptor: &GgmlFamilyAdapterDescriptor,
    ) -> Option<&'static str> {
        self.streaming_executors_by_adapter_id
            .get(descriptor.adapter_id)
            .or_else(|| {
                self.streaming_executors_by_capability
                    .get(capability_label(descriptor.execution_capability))
            })
            .map(|executor| executor.executor_id())
    }

    /// True only when the streaming executor registered for `descriptor` was
    /// declared frame-sync at registration time. Unregistered granularity
    /// (including families with no streaming executor at all) reads as
    /// `false` -- fail closed to the buffered/no-partial-guarantee default
    /// rather than assume low-latency partials.
    pub fn is_frame_sync_for(&self, descriptor: &GgmlFamilyAdapterDescriptor) -> bool {
        matches!(
            self.streaming_partial_granularity_for(descriptor),
            Some(StreamingPartialGranularity::FrameSync)
        )
    }

    /// Returns the partial-result granularity registered for `descriptor`, if
    /// any. Builtin construction derives this from the architecture descriptor;
    /// unregistered families yield `None`.
    pub fn streaming_partial_granularity_for(
        &self,
        descriptor: &GgmlFamilyAdapterDescriptor,
    ) -> Option<StreamingPartialGranularity> {
        self.streaming_partial_granularity_by_adapter_id
            .get(descriptor.adapter_id)
            .copied()
            .or_else(|| {
                self.streaming_partial_granularity_by_capability
                    .get(capability_label(descriptor.execution_capability))
                    .copied()
            })
    }

    /// Idle-unload: evicts every registered executor's process-lifetime
    /// cached prepared runtime. Safe to call opportunistically (e.g. from a
    /// background reaper) -- executors with nothing resident, or whose
    /// caching is per-thread and self-managed, just no-op.
    pub fn unload_all(&self) {
        for executor in self.executors_by_adapter_id.values() {
            executor.unload_idle_state();
        }
        for executor in self.executors_by_capability.values() {
            executor.unload_idle_state();
        }
        for executor in self.streaming_executors_by_adapter_id.values() {
            executor.unload_idle_state();
        }
        for executor in self.streaming_executors_by_capability.values() {
            executor.unload_idle_state();
        }
    }

    /// Evicts one admitted pack identity from every registered offline
    /// executor. Built-in offline and streaming dispatches share the same
    /// service-owned executor allocations, so one pass is sufficient and
    /// avoids the fixed per-family list that previously drifted.
    pub(crate) fn evict_prepared_runtime_content_id(&self, pack_content_id: &str) {
        for executor in self.executors_by_adapter_id.values() {
            executor.evict_prepared_runtime_content_id(pack_content_id);
        }
        for executor in self.executors_by_capability.values() {
            executor.evict_prepared_runtime_content_id(pack_content_id);
        }
    }
}

/// The verifier proves a route from the package bytes; the selected adapter
/// is a separate inventory projection. Bind both at the last shared execution
/// seam so neither a direct library caller nor a future ingress can pair a
/// valid pack with the wrong family implementation.
fn ensure_verified_pack_matches_family(
    verified_pack: &crate::models::pack_verifier::VerifiedPack,
    selected_family: &GgmlFamilyAdapterDescriptor,
) -> Result<(), GgmlAsrExecutionError> {
    if verified_pack.proves_asr_family(
        selected_family.model_family,
        selected_family.model_architecture,
    ) {
        return Ok(());
    }
    Err(GgmlAsrExecutionError::VerifiedPackRouteMismatch {
        model_family: selected_family.model_family,
        model_architecture: selected_family.model_architecture,
    })
}

/// Universal cancellation fence around every built-in offline executor.
///
/// Family/topology code may add finer checkpoints for latency, but it cannot
/// start after cancellation or publish a successful result once cancellation
/// wins the race. This is an execution-module invariant, not a per-family
/// capability declaration.
fn ensure_dispatch_not_canceled(
    execution_context: &RequestExecutionContext,
    executor_id: &'static str,
    adapter_id: &'static str,
) -> Result<(), GgmlAsrExecutionError> {
    if execution_context.is_canceled() {
        return Err(GgmlAsrExecutionError::executor_failed(
            executor_id,
            adapter_id,
            "transcription canceled at shared execution boundary",
        ));
    }
    Ok(())
}

/// OADP Phase 0 fail-closed gate: when an adapter is active (request-level
/// adapter path, falling back to the server-side `OPENASR_ADAPTER` env var),
/// only families with an implemented LoRA binding contract may execute; the
/// adapter is then validated against the base pack inside that executor.
/// Every other family hard-errors instead of silently ignoring the adapter.
fn ensure_adapter_binding_for_executor(
    selected_family: &GgmlFamilyAdapterDescriptor,
    executor_binding: GgmlAdapterBindingStrategy,
    request_adapter_path: Option<&std::path::Path>,
) -> Result<(), GgmlAsrExecutionError> {
    if selected_family.adapter_binding != executor_binding {
        return Err(GgmlAsrExecutionError::AdapterBindingContractMismatch {
            model_family: selected_family.model_family,
            declared: selected_family.adapter_binding.label(),
            provided: executor_binding.label(),
        });
    }
    let Some(adapter_path) = crate::adapter_pack::active_adapter_path(request_adapter_path) else {
        return Ok(());
    };
    if executor_binding.is_supported() {
        return Ok(());
    }
    Err(GgmlAsrExecutionError::AdapterUnsupportedForFamily {
        model_family: selected_family.model_family,
        adapter_path: adapter_path.display().to_string(),
    })
}

const fn capability_label(capability: GgmlExecutionCapability) -> &'static str {
    match capability {
        GgmlExecutionCapability::DedicatedRuntimeExecutorV1 => "dedicated-runtime-executor-v1",
        GgmlExecutionCapability::NativeGraphLoweringV1 => "native-graph-lowering-v1",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::{
        QWEN3_ASR_GGML_ADAPTER_ID, WHISPER_GGML_ADAPTER_ID, builtin_adapter_descriptor,
    };

    #[test]
    fn duration_boundary_ceil_uses_exact_integer_binary_rational() {
        let rate = NonZeroU32::new(16_000).unwrap();
        assert_eq!(duration_samples_ceil(30.0, rate).unwrap(), 480_000);
        assert_eq!(duration_samples_ceil(30.5, rate).unwrap(), 488_000);
        // The stored f32 value is slightly greater than decimal 0.1. Exact
        // integer ceil must retain that conservative final sample.
        assert_eq!(duration_samples_ceil(0.1, rate).unwrap(), 1_601);
        assert_eq!(duration_samples_ceil(f32::from_bits(1), rate).unwrap(), 1);
        assert!(matches!(
            duration_samples_ceil(0.0, rate),
            Err(GgmlAsrDecoderStatePlanningError::InvalidEnvelopeDuration { .. })
        ));
        assert!(matches!(
            duration_samples_ceil(f32::NAN, rate),
            Err(GgmlAsrDecoderStatePlanningError::InvalidEnvelopeDuration { .. })
        ));
    }

    #[test]
    fn offline_envelope_does_not_add_padding_above_the_fed_window_cap() {
        let options = GgmlAsrExecutionOptions {
            longform: Some(crate::LongFormOptions {
                max_chunk_seconds: 30.0,
                padding_seconds: 0.25,
                ..crate::LongFormOptions::default()
            }),
            ..GgmlAsrExecutionOptions::default()
        };
        assert_eq!(
            offline_invocation_envelope_samples(
                &options,
                NonZeroU32::new(16_000).unwrap(),
                160_000,
            )
            .unwrap(),
            480_000
        );
    }

    #[test]
    fn active_longform_envelope_does_not_expand_to_an_illegal_slice() {
        let options = GgmlAsrExecutionOptions {
            longform: Some(crate::LongFormOptions {
                mode: crate::LongFormMode::Fixed,
                max_chunk_seconds: 30.0,
                ..crate::LongFormOptions::default()
            }),
            ..GgmlAsrExecutionOptions::default()
        };
        assert_eq!(
            offline_invocation_envelope_samples(
                &options,
                NonZeroU32::new(16_000).unwrap(),
                480_001,
            )
            .unwrap(),
            480_000,
            "the planner must reject an oversized invocation against this envelope"
        );

        let off = GgmlAsrExecutionOptions {
            longform: Some(crate::LongFormOptions {
                mode: crate::LongFormMode::Off,
                ..crate::LongFormOptions::default()
            }),
            ..GgmlAsrExecutionOptions::default()
        };
        assert_eq!(
            offline_invocation_envelope_samples(&off, NonZeroU32::new(16_000).unwrap(), 480_001,)
                .unwrap(),
            480_001,
            "explicit longform-off uses the direct invocation as its envelope"
        );
    }

    #[derive(Debug)]
    struct TestSelfOnlyTopology;

    impl crate::capacity::topology::DecoderStateTopology for TestSelfOnlyTopology {
        fn demands(
            &self,
            scope: crate::capacity::topology::DecoderStateDemandScope<
                crate::capacity::topology::InvocationShapeInput,
                crate::capacity::topology::InvocationEnvelope,
            >,
        ) -> Result<
            Vec<crate::capacity::topology::StateDemand>,
            crate::capacity::topology::TopologyError,
        > {
            let invocation = match scope {
                crate::capacity::topology::DecoderStateDemandScope::ExactInvocation(invocation) => {
                    invocation
                }
                crate::capacity::topology::DecoderStateDemandScope::StableEnvelope(envelope) => {
                    envelope.maximum_invocation()
                }
            };
            Ok(vec![crate::capacity::topology::StateDemand::new(
                "test.self_kv",
                crate::capacity::topology::StateKind::SelfAttentionKv,
                invocation.samples(),
                1_000,
                crate::capacity::topology::StateBytes {
                    host: invocation.samples() as u64,
                    resident: invocation.samples() as u64,
                },
                crate::capacity::topology::PositionBoundProof::Exact,
            )?])
        }
    }

    #[test]
    fn contract_distinguishes_affirmative_no_state_from_planned_state() {
        let rate = NonZeroU32::new(16_000).unwrap();
        let invocation = crate::capacity::topology::InvocationShapeInput::new(rate, 100).unwrap();
        let envelope = crate::capacity::topology::InvocationEnvelope::new(rate, 200).unwrap();
        let plan = crate::capacity::topology::DecoderStatePlan::build(
            &TestSelfOnlyTopology,
            invocation,
            envelope,
        )
        .unwrap();
        let planned = GgmlAsrDecoderState::planned_for_test(plan, envelope);
        assert!(
            GgmlAsrDecoderStateContract::NoPersistentState
                .validates(&GgmlAsrDecoderState::NoPersistentState)
        );
        assert!(!GgmlAsrDecoderStateContract::NoPersistentState.validates(&planned));
        const MATCHING_STREAMS: &[GgmlAsrDecoderStateStreamContract] =
            &[GgmlAsrDecoderStateStreamContract::new(
                "test.self_kv",
                crate::capacity::topology::StateKind::SelfAttentionKv,
            )];
        const WRONG_STREAMS: &[GgmlAsrDecoderStateStreamContract] =
            &[GgmlAsrDecoderStateStreamContract::new(
                "test.cross_kv",
                crate::capacity::topology::StateKind::CrossAttentionKv,
            )];
        assert!(
            GgmlAsrDecoderStateContract::planned(|_| unreachable!(), MATCHING_STREAMS)
                .validates(&planned)
        );
        assert!(
            !GgmlAsrDecoderStateContract::planned(|_| unreachable!(), WRONG_STREAMS)
                .validates(&planned),
            "a non-empty plan with the wrong stream identity/kind must fail closed"
        );
    }

    #[test]
    fn execution_options_distinguish_longform_mode_from_prompt_carry() {
        let disabled = GgmlAsrExecutionOptions {
            longform: Some(crate::LongFormOptions {
                mode: crate::LongFormMode::Off,
                ..crate::LongFormOptions::default()
            }),
            ..GgmlAsrExecutionOptions::default()
        };
        assert!(!disabled.longform_mode_enabled());
        assert!(!disabled.longform_prompt_carry_enabled());
        assert_eq!(disabled.max_longform_prompt_tokens(), 0);

        let no_carry = GgmlAsrExecutionOptions {
            longform: Some(crate::LongFormOptions {
                mode: crate::LongFormMode::Fixed,
                carry_prompt_across_slices: false,
                ..crate::LongFormOptions::default()
            }),
            ..GgmlAsrExecutionOptions::default()
        };
        assert!(no_carry.longform_mode_enabled());
        assert!(!no_carry.longform_prompt_carry_enabled());
        assert_eq!(no_carry.max_longform_prompt_tokens(), 0);

        let carry = GgmlAsrExecutionOptions {
            longform: Some(crate::LongFormOptions {
                mode: crate::LongFormMode::Fixed,
                max_context_tokens: 37,
                ..crate::LongFormOptions::default()
            }),
            ..GgmlAsrExecutionOptions::default()
        };
        assert!(carry.longform_mode_enabled());
        assert!(carry.longform_prompt_carry_enabled());
        assert_eq!(carry.max_longform_prompt_tokens(), 37);
    }

    #[test]
    fn runtime_build_identity_separates_same_route_content_replacements() {
        let route = "whisper:metal:base";
        let options = "adapter=none";
        let first = RuntimeBuildIdentity::new("verified-content-a", route, options);
        let replacement = RuntimeBuildIdentity::new("verified-content-b", route, options);
        assert_ne!(
            first, replacement,
            "same path/route must not reuse replacement content"
        );
        assert_ne!(
            first,
            RuntimeBuildIdentity::new("verified-content-a", route, "adapter=/tmp/a.oadp"),
            "adapter/options fingerprint must rebuild the engine"
        );
        // Same content id/route/options must always compare equal -- there is
        // no generation/epoch field left to make an otherwise-identical
        // identity spuriously distinct (that was the audited bug).
        assert_eq!(
            first,
            RuntimeBuildIdentity::new("verified-content-a", route, options)
        );
    }

    #[test]
    fn runtime_build_identity_resolve_prefers_explicit_request_content_id() {
        let verified = RuntimeBuildIdentity::new("verified-content-a", "old", "old");
        let resolved = RuntimeBuildIdentity::resolve_for_request(
            Some(&verified),
            "whisper:gpu",
            "adapter=none",
            "sha256:should-not-win",
        );
        assert_eq!(resolved.pack_content_id, "verified-content-a");
        assert_eq!(resolved.route, "whisper:gpu");
        assert_eq!(resolved.options_fingerprint, "adapter=none");
    }

    #[test]
    fn production_pack_content_id_misses_same_path_byte_replacement() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("same-path.gguf");
        let write = |payload: &[u8]| {
            let mut bytes = b"GGUF".to_vec();
            bytes.extend_from_slice(payload);
            std::fs::write(&path, bytes).expect("write pack");
        };
        let source_content_id = |path: &std::path::Path| -> String {
            crate::validate_ggml_runtime_source_path(path)
                .expect("validate runtime source")
                .content_id()
                .to_string()
        };

        write(b"content-a-bytes");
        let id_a = source_content_id(&path);
        write(b"content-b-bytes-different");
        let id_b = source_content_id(&path);
        assert!(id_a.starts_with("sha256:"), "got {id_a}");
        assert!(id_b.starts_with("sha256:"), "got {id_b}");
        assert_ne!(
            id_a, id_b,
            "same path with different pack bytes must not share content id"
        );

        let options = GgmlAsrExecutionOptions::default();
        write(b"content-a-bytes");
        let identity_a = serve_batch_build_identity_for_request(
            &options,
            "whisper",
            crate::ggml_runtime::GgmlCpuGraphBackend::Gpu,
            &crate::validate_ggml_runtime_source_path(&path).expect("validate a"),
        );
        write(b"content-b-bytes-different");
        let identity_b = serve_batch_build_identity_for_request(
            &options,
            "whisper",
            crate::ggml_runtime::GgmlCpuGraphBackend::Gpu,
            &crate::validate_ggml_runtime_source_path(&path).expect("validate b"),
        );
        assert_eq!(identity_a.pack_content_id, id_a);
        assert_eq!(identity_b.pack_content_id, id_b);
        assert_ne!(identity_a.pack_content_id, identity_b.pack_content_id);
        assert_eq!(identity_a.route, identity_b.route);

        // Re-resolving (a fresh source, exactly like a new request) against
        // unchanged (post-rewrite) bytes must return an identity equal to
        // `identity_b` -- nothing left to bump spuriously.
        let identity_again = serve_batch_build_identity_for_request(
            &options,
            "whisper",
            crate::ggml_runtime::GgmlCpuGraphBackend::Gpu,
            &crate::validate_ggml_runtime_source_path(&path).expect("validate again"),
        );
        assert_eq!(identity_again, identity_b);
    }

    /// Structural proof that `execution_context` is required, not optional:
    /// this compiles only because the field's type is the concrete
    /// `Arc<RequestExecutionContext>`, not `Option<Arc<RequestExecutionContext>>`
    /// -- an `Option` field would fail to type-check against
    /// `require_concrete_execution_context`'s parameter. Never called; exists
    /// purely so `cargo check`/`clippy` re-verify the contract on every build.
    #[allow(dead_code)]
    fn require_concrete_execution_context(_: std::sync::Arc<crate::RequestExecutionContext>) {}

    #[allow(dead_code)]
    fn assert_ggml_asr_execution_request_requires_execution_context(
        request: GgmlAsrExecutionRequest,
    ) {
        let GgmlAsrExecutionRequest {
            execution_context, ..
        } = request;
        require_concrete_execution_context(execution_context);
    }

    fn request_for_architecture(
        model_architecture: &'static str,
        backend_preference: GgmlAsrBackendPreference,
    ) -> GgmlAsrExecutionRequest {
        let verified_pack = crate::models::runtime_preflight::verified_pack_from_preflight_for_test(
            crate::models::runtime_preflight::leaked_tiny_runtime_source_preflight(),
            model_architecture,
        );
        GgmlAsrExecutionRequest {
            execution_services:
                crate::models::native_execution_services::test_native_execution_services(),
            decoder_state: crate::models::ggml_asr_executor::GgmlAsrDecoderState::NoPersistentState,
            verified_pack,
            selected_family: builtin_adapter_descriptor(model_architecture),
            prepared_audio: GgmlAsrPreparedAudio::mono_16khz(vec![0.0, 0.1]),
            request_options: GgmlAsrExecutionOptions::default(),
            backend_preference,
            resolved_runtime: crate::ggml_runtime::ResolvedFamilyRuntimeInput::resolve(
                backend_preference.request_backend_override(),
                crate::ggml_runtime::AutoGpuPolicy::AllBackends,
            ),
            execution_context: std::sync::Arc::new(crate::RequestExecutionContext::uncancellable(
                "test fixture",
            )),
        }
    }

    fn whisper_request(backend_preference: GgmlAsrBackendPreference) -> GgmlAsrExecutionRequest {
        request_for_architecture(
            crate::arch::WHISPER_GGML_ARCHITECTURE_ID,
            backend_preference,
        )
    }

    fn successful_execution_result(text: &str) -> GgmlAsrExecutionResult {
        GgmlAsrExecutionResult {
            transcription: Transcription {
                truncated_decodes: Vec::new(),
                unnamed_speakers: Vec::new(),
                text: text.to_string(),
                segments: Vec::new(),
                longform: None,
                language: None,
            },
            carry_context: None,
            decode_truncation: None,
        }
    }

    #[test]
    fn execute_rejects_a_verified_pack_paired_with_the_wrong_family() {
        let mut request = whisper_request(GgmlAsrBackendPreference::CpuOnly);
        request.selected_family =
            builtin_adapter_descriptor(crate::arch::QWEN3_ASR_GGML_ARCHITECTURE_ID);

        let error = GgmlAsrExecutionDispatch::default()
            .execute(&request)
            .expect_err("route mismatch must fail before executor lookup");
        assert!(matches!(
            error,
            GgmlAsrExecutionError::VerifiedPackRouteMismatch {
                model_family: "qwen3-asr",
                model_architecture: crate::arch::QWEN3_ASR_GGML_ARCHITECTURE_ID,
            }
        ));
    }

    #[test]
    fn supported_adapter_binding_cannot_be_self_certified_by_the_descriptor() {
        let qwen = builtin_adapter_descriptor(crate::arch::QWEN3_ASR_GGML_ARCHITECTURE_ID);
        let error = ensure_adapter_binding_for_executor(
            &qwen,
            GgmlAdapterBindingStrategy::Unsupported,
            None,
        )
        .expect_err("a supported descriptor requires a matching concrete executor binding");
        assert!(matches!(
            error,
            GgmlAsrExecutionError::AdapterBindingContractMismatch {
                declared: "qwen3-asr-lora-v1",
                provided: "unsupported",
                ..
            }
        ));
    }

    #[test]
    fn public_prepared_audio_retains_the_mutable_vec_contract() {
        let mut audio = GgmlAsrPreparedAudio {
            sample_rate_hz: 16_000,
            channels: 1,
            samples_f32: vec![0.25],
        };
        audio.samples_f32.push(-0.5);
        assert_eq!(audio.samples_f32, vec![0.25, -0.5]);
    }

    #[test]
    fn shared_view_materializes_only_at_an_owned_extension_boundary() {
        struct OwnedExtension {
            observed: Arc<std::sync::Mutex<(usize, Vec<f32>)>>,
        }

        impl GgmlAsrExecutor for OwnedExtension {
            fn executor_id(&self) -> &'static str {
                "owned-extension"
            }

            fn supports_phrase_bias(&self) -> bool {
                false
            }

            fn decoder_state_contract(
                &self,
                _selected_family: &GgmlFamilyAdapterDescriptor,
            ) -> Result<GgmlAsrDecoderStateContract, GgmlAsrExecutionError> {
                Ok(GgmlAsrDecoderStateContract::NoPersistentState)
            }

            fn execute(
                &self,
                request: &GgmlAsrExecutionRequest,
            ) -> Result<GgmlAsrExecutionResult, GgmlAsrExecutionError> {
                *self.observed.lock().unwrap() = (
                    request.prepared_audio.samples_f32.as_ptr() as usize,
                    request.prepared_audio.samples_f32.clone(),
                );
                Ok(successful_execution_result("owned"))
            }
        }

        let backing = crate::PcmBuffer::from_vec(vec![0.25, -0.5, 0.75]);
        let shared_pointer = backing.as_ptr() as usize;
        let owned = whisper_request(GgmlAsrBackendPreference::CpuOnly);
        let mut view = owned.as_view();
        view.prepared_audio = GgmlAsrPreparedAudioView::mono_16khz_shared(backing.full_slice());
        let observed = Arc::new(std::sync::Mutex::new((0, Vec::new())));
        let dispatch = GgmlAsrExecutionDispatch::default().with_executor_for_adapter(
            WHISPER_GGML_ADAPTER_ID,
            Arc::new(OwnedExtension {
                observed: Arc::clone(&observed),
            }),
        );

        let result = dispatch
            .execute_view(&view)
            .expect("compatibility dispatch");
        assert_eq!(result.transcription.text, "owned");
        let (observed_pointer, observed_samples) = observed.lock().unwrap().clone();
        assert_eq!(observed_samples, backing.as_slice());
        assert_ne!(
            observed_pointer, shared_pointer,
            "the owned extension boundary must receive its own Vec"
        );
    }

    #[test]
    fn native_view_slot_preserves_pcm_for_shared_and_public_owned_requests() {
        struct ViewExecutor {
            observed_pointers: Arc<std::sync::Mutex<Vec<usize>>>,
        }

        impl GgmlAsrViewExecutor for ViewExecutor {
            fn executor_id(&self) -> &'static str {
                "view-executor"
            }

            fn supports_phrase_bias(&self) -> bool {
                false
            }

            fn evict_prepared_runtime_content_id(&self, _pack_content_id: &str) {}

            fn decoder_state_contract(
                &self,
                _selected_family: &GgmlFamilyAdapterDescriptor,
            ) -> Result<GgmlAsrDecoderStateContract, GgmlAsrExecutionError> {
                Ok(GgmlAsrDecoderStateContract::NoPersistentState)
            }

            fn execute_view(
                &self,
                request: &GgmlAsrExecutionViewRequest<'_>,
            ) -> Result<GgmlAsrExecutionResult, GgmlAsrExecutionError> {
                self.observed_pointers
                    .lock()
                    .unwrap()
                    .push(request.prepared_audio.samples_f32.as_ptr() as usize);
                Ok(successful_execution_result("view"))
            }
        }

        let observed_pointers = Arc::new(std::sync::Mutex::new(Vec::new()));
        let dispatch = GgmlAsrExecutionDispatch::default().with_view_executor_for_adapter(
            WHISPER_GGML_ADAPTER_ID,
            Arc::new(ViewExecutor {
                observed_pointers: Arc::clone(&observed_pointers),
            }),
        );

        let backing = crate::PcmBuffer::from_vec(vec![0.1, 0.2, 0.3]);
        let shared_pointer = backing.as_ptr() as usize;
        let holder = whisper_request(GgmlAsrBackendPreference::CpuOnly);
        let mut shared_request = holder.as_view();
        shared_request.prepared_audio =
            GgmlAsrPreparedAudioView::mono_16khz_shared(backing.full_slice());
        dispatch
            .execute_view(&shared_request)
            .expect("shared view dispatch");

        let owned_request = whisper_request(GgmlAsrBackendPreference::CpuOnly);
        let owned_pointer = owned_request.prepared_audio.samples_f32.as_ptr() as usize;
        dispatch.execute(&owned_request).expect("owned dispatch");

        assert_eq!(
            observed_pointers.lock().unwrap().as_slice(),
            &[shared_pointer, owned_pointer]
        );
    }

    #[test]
    fn dedicated_tdt_cannot_bypass_the_shared_pre_execution_cancel_fence() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct NeverCalledExecutor(Arc<AtomicUsize>);

        impl GgmlAsrExecutor for NeverCalledExecutor {
            fn executor_id(&self) -> &'static str {
                "tdt-cancel-conformance"
            }

            fn supports_phrase_bias(&self) -> bool {
                false
            }

            fn decoder_state_contract(
                &self,
                _selected_family: &GgmlFamilyAdapterDescriptor,
            ) -> Result<GgmlAsrDecoderStateContract, GgmlAsrExecutionError> {
                Ok(GgmlAsrDecoderStateContract::NoPersistentState)
            }

            fn execute(
                &self,
                _request: &GgmlAsrExecutionRequest,
            ) -> Result<GgmlAsrExecutionResult, GgmlAsrExecutionError> {
                self.0.fetch_add(1, Ordering::SeqCst);
                Ok(successful_execution_result("must not execute"))
            }
        }

        let architecture = crate::arch::OpenAsrArchitectureRegistry::with_builtins()
            .find_by_model_architecture(crate::arch::PARAKEET_TDT_GGML_ARCHITECTURE_ID)
            .expect("TDT descriptor");
        let selected_family = architecture.ggml_family_adapter_descriptor();
        let control = Arc::new(crate::api::backend::TranscriptionControl::new());
        control.request_cancel();
        let mut request = request_for_architecture(
            crate::arch::PARAKEET_TDT_GGML_ARCHITECTURE_ID,
            GgmlAsrBackendPreference::CpuOnly,
        );
        request.execution_context = Arc::new(crate::RequestExecutionContext::new(
            Some("tdt-cancel-conformance".to_string()),
            control,
        ));
        let calls = Arc::new(AtomicUsize::new(0));
        let dispatch = GgmlAsrExecutionDispatch::default().with_executor_for_adapter(
            selected_family.adapter_id,
            Arc::new(NeverCalledExecutor(Arc::clone(&calls))),
        );

        let error = dispatch
            .execute(&request)
            .expect_err("pre-canceled TDT request must fail before its dedicated executor");
        assert!(error.to_string().contains("canceled"));
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }

    fn streaming_request_for_architecture(
        model_architecture: &'static str,
        backend_preference: GgmlAsrBackendPreference,
    ) -> GgmlAsrStreamingSessionRequest {
        let verified_pack = crate::models::runtime_preflight::verified_pack_from_preflight_for_test(
            crate::models::runtime_preflight::leaked_tiny_runtime_source_preflight(),
            model_architecture,
        );
        GgmlAsrStreamingSessionRequest {
            execution_services:
                crate::models::native_execution_services::test_native_execution_services(),
            decoder_state: crate::models::ggml_asr_executor::GgmlAsrDecoderState::NoPersistentState,
            verified_pack,
            selected_family: builtin_adapter_descriptor(model_architecture),
            request_options: GgmlAsrExecutionOptions::default(),
            configured_diarize: false,
            backend_preference,
            resolved_runtime: crate::ggml_runtime::ResolvedFamilyRuntimeInput::resolve(
                backend_preference.request_backend_override(),
                crate::ggml_runtime::AutoGpuPolicy::AllBackends,
            ),
            final_text_processor: None,
            session_context: crate::NativeAsrSessionContext::new("rt_ggml_streaming"),
            session_config: crate::NativeAsrStreamingSessionConfig::new().into(),
        }
    }

    fn whisper_streaming_request(
        backend_preference: GgmlAsrBackendPreference,
    ) -> GgmlAsrStreamingSessionRequest {
        streaming_request_for_architecture(
            crate::arch::WHISPER_GGML_ARCHITECTURE_ID,
            backend_preference,
        )
    }

    struct StubNativeSession {
        session_id: String,
    }

    impl crate::NativeAsrSession for StubNativeSession {
        fn session_id(&self) -> &str {
            &self.session_id
        }

        fn push_audio(
            &mut self,
            _frame: crate::RealtimeAudioFrame,
        ) -> Result<Vec<crate::RealtimeEventEnvelope>, crate::NativeAsrError> {
            Ok(Vec::new())
        }

        fn poll_events(
            &mut self,
        ) -> Result<Vec<crate::RealtimeEventEnvelope>, crate::NativeAsrError> {
            Ok(Vec::new())
        }

        fn finish(&mut self) -> Result<Vec<crate::RealtimeEventEnvelope>, crate::NativeAsrError> {
            Ok(Vec::new())
        }

        fn cancel(&mut self) -> Result<Vec<crate::RealtimeEventEnvelope>, crate::NativeAsrError> {
            Ok(Vec::new())
        }
    }

    #[test]
    fn dispatch_fails_closed_when_executor_is_not_registered() {
        let dispatch = GgmlAsrExecutionDispatch::default();
        let request = whisper_request(GgmlAsrBackendPreference::CpuOnly);

        let error = dispatch
            .execute(&request)
            .expect_err("missing executor must fail closed");
        assert!(matches!(
            error,
            GgmlAsrExecutionError::ExecutorUnavailable {
                adapter_id: "ggml-family-whisper-runtime-v1",
                model_family: "whisper",
                capability: "dedicated-runtime-executor-v1"
            }
        ));
        assert!(
            error
                .to_string()
                .contains("no ggml executor is registered for adapter")
        );
    }

    #[test]
    fn dispatch_accepts_auto_backend_preference() {
        struct StubExecutor;
        impl GgmlAsrExecutor for StubExecutor {
            fn executor_id(&self) -> &'static str {
                "stub"
            }

            fn supports_phrase_bias(&self) -> bool {
                true
            }

            fn decoder_state_contract(
                &self,
                _selected_family: &GgmlFamilyAdapterDescriptor,
            ) -> Result<GgmlAsrDecoderStateContract, GgmlAsrExecutionError> {
                Ok(GgmlAsrDecoderStateContract::NoPersistentState)
            }

            fn execute(
                &self,
                _request: &GgmlAsrExecutionRequest,
            ) -> Result<GgmlAsrExecutionResult, GgmlAsrExecutionError> {
                Ok(GgmlAsrExecutionResult {
                    transcription: Transcription {
                        truncated_decodes: Vec::new(),
                        unnamed_speakers: Vec::new(),
                        text: "ok".to_string(),
                        segments: Vec::new(),
                        longform: None,
                        language: None,
                    },
                    carry_context: None,
                    decode_truncation: None,
                })
            }
        }

        let request = whisper_request(GgmlAsrBackendPreference::Auto);
        let dispatch = GgmlAsrExecutionDispatch::default()
            .with_executor_for_adapter(WHISPER_GGML_ADAPTER_ID, Arc::new(StubExecutor));
        let result = dispatch.execute(&request).expect("auto should dispatch");
        assert_eq!(result.transcription.text, "ok");
    }

    #[test]
    fn dispatch_allows_phrase_bias_to_reach_registered_executor() {
        struct StubExecutor;
        impl GgmlAsrExecutor for StubExecutor {
            fn executor_id(&self) -> &'static str {
                "phrase-bias-stub"
            }

            fn supports_phrase_bias(&self) -> bool {
                true
            }

            fn decoder_state_contract(
                &self,
                _selected_family: &GgmlFamilyAdapterDescriptor,
            ) -> Result<GgmlAsrDecoderStateContract, GgmlAsrExecutionError> {
                Ok(GgmlAsrDecoderStateContract::NoPersistentState)
            }

            fn execute(
                &self,
                request: &GgmlAsrExecutionRequest,
            ) -> Result<GgmlAsrExecutionResult, GgmlAsrExecutionError> {
                assert!(request.request_options.phrase_bias.is_some());
                Ok(GgmlAsrExecutionResult {
                    transcription: Transcription {
                        truncated_decodes: Vec::new(),
                        unnamed_speakers: Vec::new(),
                        text: "biased".to_string(),
                        segments: Vec::new(),
                        longform: None,
                        language: None,
                    },
                    carry_context: None,
                    decode_truncation: None,
                })
            }
        }

        let mut request = whisper_request(GgmlAsrBackendPreference::Auto);
        request.request_options.phrase_bias = Some(
            crate::PhraseBiasConfig::from_phrases([("OpenASR", 2.0)])
                .expect("phrase bias fixture must validate"),
        );
        let dispatch = GgmlAsrExecutionDispatch::default()
            .with_executor_for_adapter(WHISPER_GGML_ADAPTER_ID, Arc::new(StubExecutor));

        let result = dispatch
            .execute(&request)
            .expect("registered executor receives phrase bias");

        assert_eq!(result.transcription.text, "biased");
    }

    #[test]
    fn dispatch_fails_closed_when_qwen_executor_is_not_registered() {
        let request = request_for_architecture(
            crate::arch::QWEN3_ASR_GGML_ARCHITECTURE_ID,
            GgmlAsrBackendPreference::CpuOnly,
        );
        let dispatch = GgmlAsrExecutionDispatch::default();
        let error = dispatch
            .execute(&request)
            .expect_err("missing qwen executor must fail closed");
        assert!(matches!(
            error,
            GgmlAsrExecutionError::ExecutorUnavailable {
                adapter_id: QWEN3_ASR_GGML_ADAPTER_ID,
                model_family: crate::QWEN3_ASR_MODEL_FAMILY,
                capability: "native-graph-lowering-v1"
            }
        ));
    }

    #[test]
    fn dispatch_allows_qwen_lora_and_rejects_unsupported_families() {
        struct StubExecutor {
            adapter_binding: GgmlAdapterBindingStrategy,
        }
        impl GgmlAsrExecutor for StubExecutor {
            fn executor_id(&self) -> &'static str {
                "adapter-gate-stub"
            }

            fn adapter_binding_strategy(&self) -> GgmlAdapterBindingStrategy {
                self.adapter_binding
            }

            fn supports_phrase_bias(&self) -> bool {
                true
            }

            fn decoder_state_contract(
                &self,
                _selected_family: &GgmlFamilyAdapterDescriptor,
            ) -> Result<GgmlAsrDecoderStateContract, GgmlAsrExecutionError> {
                Ok(GgmlAsrDecoderStateContract::NoPersistentState)
            }

            fn execute(
                &self,
                _request: &GgmlAsrExecutionRequest,
            ) -> Result<GgmlAsrExecutionResult, GgmlAsrExecutionError> {
                Ok(GgmlAsrExecutionResult {
                    transcription: Transcription {
                        truncated_decodes: Vec::new(),
                        unnamed_speakers: Vec::new(),
                        text: "qwen-lora".to_string(),
                        segments: Vec::new(),
                        longform: None,
                        language: None,
                    },
                    carry_context: None,
                    decode_truncation: None,
                })
            }
        }

        // Qwen has a native LoRA binding contract, so it must reach the
        // registered executor instead of dying in the family gate.
        let mut request = request_for_architecture(
            crate::arch::QWEN3_ASR_GGML_ARCHITECTURE_ID,
            GgmlAsrBackendPreference::CpuOnly,
        );
        request.request_options.adapter_path = Some(PathBuf::from("/tmp/fixture.oadp"));
        let dispatch = GgmlAsrExecutionDispatch::default().with_native_graph_lowering_v1(Arc::new(
            StubExecutor {
                adapter_binding: GgmlAdapterBindingStrategy::Qwen3AsrLoraV1,
            },
        ));

        let result = dispatch
            .execute(&request)
            .expect("Qwen adapter request must reach its executor");
        assert_eq!(result.transcription.text, "qwen-lora");

        let mut unsupported_whisper_request = whisper_request(GgmlAsrBackendPreference::CpuOnly);
        unsupported_whisper_request.request_options.adapter_path =
            Some(PathBuf::from("/tmp/fixture.oadp"));
        let unsupported_dispatch = GgmlAsrExecutionDispatch::default().with_executor_for_adapter(
            WHISPER_GGML_ADAPTER_ID,
            Arc::new(StubExecutor {
                adapter_binding: GgmlAdapterBindingStrategy::Unsupported,
            }),
        );
        let error = unsupported_dispatch
            .execute(&unsupported_whisper_request)
            .expect_err("unsupported family must fail closed before execution");
        assert!(matches!(
            error,
            GgmlAsrExecutionError::AdapterUnsupportedForFamily {
                model_family: "whisper",
                ..
            }
        ));

        // The same adapter on the moonshine family passes the descriptor gate:
        // with no moonshine executor registered it must reach executor lookup
        // and fail with ExecutorUnavailable, NOT AdapterUnsupportedForFamily.
        let mut moonshine_request = request_for_architecture(
            crate::arch::MOONSHINE_GGML_ARCHITECTURE_ID,
            GgmlAsrBackendPreference::CpuOnly,
        );
        moonshine_request.request_options.adapter_path = Some(PathBuf::from("/tmp/fixture.oadp"));
        let error = GgmlAsrExecutionDispatch::default()
            .execute(&moonshine_request)
            .expect_err("no moonshine executor registered");
        assert!(matches!(
            error,
            GgmlAsrExecutionError::ExecutorUnavailable { .. }
        ));
    }

    #[test]
    fn streaming_dispatch_fails_closed_when_adapter_is_active_for_non_lora_family() {
        struct StubStreamingExecutor;
        impl GgmlAsrStreamingExecutor for StubStreamingExecutor {
            fn executor_id(&self) -> &'static str {
                "non-lora-streaming-stub"
            }

            fn start_streaming_session(
                &self,
                _request: &GgmlAsrStreamingSessionRequest,
            ) -> Result<Box<dyn crate::NativeAsrSession>, GgmlAsrExecutionError> {
                unreachable!("the adapter gate must reject before session construction")
            }
        }

        let mut request = whisper_streaming_request(GgmlAsrBackendPreference::Auto);
        request.request_options.adapter_path = Some(PathBuf::from("/tmp/fixture.oadp"));
        let dispatch = GgmlAsrExecutionDispatch::default().with_streaming_executor_for_adapter(
            WHISPER_GGML_ADAPTER_ID,
            Arc::new(StubStreamingExecutor),
        );

        let error = match dispatch.start_streaming_session(&request) {
            Ok(_) => panic!("adapter on a non-LoRA family must fail closed"),
            Err(error) => error,
        };
        assert!(matches!(
            error,
            GgmlAsrExecutionError::AdapterUnsupportedForFamily {
                model_family: "whisper",
                ..
            }
        ));
    }

    #[test]
    fn dispatch_falls_back_to_capability_executor() {
        struct StubExecutor;
        impl GgmlAsrExecutor for StubExecutor {
            fn executor_id(&self) -> &'static str {
                "native-graph-lowering-stub"
            }

            fn adapter_binding_strategy(&self) -> GgmlAdapterBindingStrategy {
                GgmlAdapterBindingStrategy::Qwen3AsrLoraV1
            }

            fn supports_phrase_bias(&self) -> bool {
                true
            }

            fn decoder_state_contract(
                &self,
                _selected_family: &GgmlFamilyAdapterDescriptor,
            ) -> Result<GgmlAsrDecoderStateContract, GgmlAsrExecutionError> {
                Ok(GgmlAsrDecoderStateContract::NoPersistentState)
            }

            fn execute(
                &self,
                _request: &GgmlAsrExecutionRequest,
            ) -> Result<GgmlAsrExecutionResult, GgmlAsrExecutionError> {
                Ok(GgmlAsrExecutionResult {
                    transcription: Transcription {
                        truncated_decodes: Vec::new(),
                        unnamed_speakers: Vec::new(),
                        text: "ok".to_string(),
                        segments: Vec::new(),
                        longform: None,
                        language: None,
                    },
                    carry_context: None,
                    decode_truncation: None,
                })
            }
        }

        let request = request_for_architecture(
            crate::arch::QWEN3_ASR_GGML_ARCHITECTURE_ID,
            GgmlAsrBackendPreference::Auto,
        );
        let dispatch = GgmlAsrExecutionDispatch::default()
            .with_native_graph_lowering_v1(Arc::new(StubExecutor));

        let result = dispatch
            .execute(&request)
            .expect("capability executor should dispatch");
        assert_eq!(result.transcription.text, "ok");
    }

    #[test]
    fn streaming_dispatch_fails_closed_when_executor_is_not_registered() {
        let dispatch = GgmlAsrExecutionDispatch::default();
        let request = whisper_streaming_request(GgmlAsrBackendPreference::Auto);

        let error = match dispatch.start_streaming_session(&request) {
            Ok(_) => panic!("missing streaming executor must fail closed"),
            Err(error) => error,
        };

        assert!(matches!(
            error,
            GgmlAsrExecutionError::ExecutorUnavailable {
                adapter_id: "ggml-family-whisper-runtime-v1",
                model_family: "whisper",
                capability: "dedicated-runtime-executor-v1"
            }
        ));
    }

    #[test]
    fn streaming_dispatch_routes_registered_adapter_executor() {
        struct StubStreamingExecutor;
        impl GgmlAsrStreamingExecutor for StubStreamingExecutor {
            fn executor_id(&self) -> &'static str {
                "streaming-stub"
            }

            fn start_streaming_session(
                &self,
                request: &GgmlAsrStreamingSessionRequest,
            ) -> Result<Box<dyn crate::NativeAsrSession>, GgmlAsrExecutionError> {
                assert_eq!(request.selected_family.adapter_id, WHISPER_GGML_ADAPTER_ID);
                Ok(Box::new(StubNativeSession {
                    session_id: request.session_context.session_id.0.clone(),
                }))
            }
        }

        let request = whisper_streaming_request(GgmlAsrBackendPreference::Auto);
        let dispatch = GgmlAsrExecutionDispatch::default().with_streaming_executor_for_adapter(
            WHISPER_GGML_ADAPTER_ID,
            Arc::new(StubStreamingExecutor),
        );

        let session = dispatch
            .start_streaming_session(&request)
            .expect("registered streaming executor should dispatch");

        assert_eq!(session.session_id(), "rt_ggml_streaming");
    }

    #[test]
    fn streaming_dispatch_reports_executor_coverage() {
        struct StubStreamingExecutor;
        impl GgmlAsrStreamingExecutor for StubStreamingExecutor {
            fn executor_id(&self) -> &'static str {
                "streaming-coverage-stub"
            }

            fn start_streaming_session(
                &self,
                request: &GgmlAsrStreamingSessionRequest,
            ) -> Result<Box<dyn crate::NativeAsrSession>, GgmlAsrExecutionError> {
                Ok(Box::new(StubNativeSession {
                    session_id: request.session_context.session_id.0.clone(),
                }))
            }
        }

        let whisper = builtin_adapter_descriptor(crate::arch::WHISPER_GGML_ARCHITECTURE_ID);
        let qwen = builtin_adapter_descriptor(crate::arch::QWEN3_ASR_GGML_ARCHITECTURE_ID);
        let empty_dispatch = GgmlAsrExecutionDispatch::default();
        assert!(!empty_dispatch.has_streaming_executor_for(&whisper));
        assert!(!empty_dispatch.has_streaming_executor_for(&qwen));

        let adapter_dispatch = GgmlAsrExecutionDispatch::default()
            .with_streaming_executor_for_adapter(
                whisper.adapter_id,
                Arc::new(StubStreamingExecutor),
            );
        assert!(adapter_dispatch.has_streaming_executor_for(&whisper));
        assert!(!adapter_dispatch.has_streaming_executor_for(&qwen));

        let capability_dispatch = GgmlAsrExecutionDispatch::default()
            .with_streaming_executor_for_capability(
                qwen.execution_capability,
                Arc::new(StubStreamingExecutor),
            );
        assert!(capability_dispatch.has_streaming_executor_for(&qwen));
    }

    #[test]
    fn is_frame_sync_for_reports_registered_granularity_and_defaults_closed() {
        let whisper = builtin_adapter_descriptor(crate::arch::WHISPER_GGML_ARCHITECTURE_ID);
        let qwen = builtin_adapter_descriptor(crate::arch::QWEN3_ASR_GGML_ARCHITECTURE_ID);

        // No granularity registered at all: fails closed to "not frame-sync",
        // matching the treatment of an unregistered streaming executor.
        let empty_dispatch = GgmlAsrExecutionDispatch::default();
        assert!(!empty_dispatch.is_frame_sync_for(&whisper));
        assert!(!empty_dispatch.is_frame_sync_for(&qwen));

        let mixed_dispatch = GgmlAsrExecutionDispatch::default()
            .with_streaming_partial_granularity_for_adapter(
                whisper.adapter_id,
                StreamingPartialGranularity::FrameSync,
            )
            .with_streaming_partial_granularity_for_adapter(
                qwen.adapter_id,
                StreamingPartialGranularity::Buffered,
            );
        assert!(mixed_dispatch.is_frame_sync_for(&whisper));
        assert!(!mixed_dispatch.is_frame_sync_for(&qwen));

        let capability_dispatch = GgmlAsrExecutionDispatch::default()
            .with_streaming_partial_granularity_for_capability(
                qwen.execution_capability,
                StreamingPartialGranularity::FrameSync,
            );
        assert!(capability_dispatch.is_frame_sync_for(&qwen));
    }

    #[test]
    fn unload_all_reaches_every_registered_executor_map() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        // One stub per registration slot (offline adapter-id, offline
        // capability, streaming adapter-id, streaming capability), each
        // bumping its own counter from `unload_idle_state` -- proves
        // `unload_all` walks all four maps, not just the offline/adapter-id
        // one every other test in this file happens to exercise.
        struct CountingExecutor(Arc<AtomicUsize>);
        impl GgmlAsrExecutor for CountingExecutor {
            fn executor_id(&self) -> &'static str {
                "counting-offline-stub"
            }
            fn supports_phrase_bias(&self) -> bool {
                false
            }
            fn decoder_state_contract(
                &self,
                _selected_family: &GgmlFamilyAdapterDescriptor,
            ) -> Result<GgmlAsrDecoderStateContract, GgmlAsrExecutionError> {
                Ok(GgmlAsrDecoderStateContract::NoPersistentState)
            }
            fn execute(
                &self,
                _request: &GgmlAsrExecutionRequest,
            ) -> Result<GgmlAsrExecutionResult, GgmlAsrExecutionError> {
                unreachable!("this test never executes a request")
            }
            fn unload_idle_state(&self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }
        struct CountingStreamingExecutor(Arc<AtomicUsize>);
        impl GgmlAsrStreamingExecutor for CountingStreamingExecutor {
            fn executor_id(&self) -> &'static str {
                "counting-streaming-stub"
            }
            fn start_streaming_session(
                &self,
                _request: &GgmlAsrStreamingSessionRequest,
            ) -> Result<Box<dyn crate::NativeAsrSession>, GgmlAsrExecutionError> {
                unreachable!("this test never starts a streaming session")
            }
            fn unload_idle_state(&self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }

        let offline_adapter_calls = Arc::new(AtomicUsize::new(0));
        let offline_capability_calls = Arc::new(AtomicUsize::new(0));
        let streaming_adapter_calls = Arc::new(AtomicUsize::new(0));
        let streaming_capability_calls = Arc::new(AtomicUsize::new(0));

        let dispatch = GgmlAsrExecutionDispatch::default()
            .with_executor_for_adapter(
                WHISPER_GGML_ADAPTER_ID,
                Arc::new(CountingExecutor(Arc::clone(&offline_adapter_calls))),
            )
            .with_executor_for_capability(
                GgmlExecutionCapability::NativeGraphLoweringV1,
                Arc::new(CountingExecutor(Arc::clone(&offline_capability_calls))),
            )
            .with_streaming_executor_for_adapter(
                WHISPER_GGML_ADAPTER_ID,
                Arc::new(CountingStreamingExecutor(Arc::clone(
                    &streaming_adapter_calls,
                ))),
            )
            .with_streaming_executor_for_capability(
                GgmlExecutionCapability::NativeGraphLoweringV1,
                Arc::new(CountingStreamingExecutor(Arc::clone(
                    &streaming_capability_calls,
                ))),
            );

        dispatch.unload_all();

        assert_eq!(offline_adapter_calls.load(Ordering::SeqCst), 1);
        assert_eq!(offline_capability_calls.load(Ordering::SeqCst), 1);
        assert_eq!(streaming_adapter_calls.load(Ordering::SeqCst), 1);
        assert_eq!(streaming_capability_calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn unload_idle_state_default_is_a_no_op() {
        // Any executor that does not override `unload_idle_state` (every
        // family whose only caching is per-thread/bounded) must tolerate
        // being told to unload -- the default no-op must not panic.
        struct NoCacheExecutor;
        impl GgmlAsrExecutor for NoCacheExecutor {
            fn executor_id(&self) -> &'static str {
                "no-cache-stub"
            }
            fn supports_phrase_bias(&self) -> bool {
                false
            }
            fn decoder_state_contract(
                &self,
                _selected_family: &GgmlFamilyAdapterDescriptor,
            ) -> Result<GgmlAsrDecoderStateContract, GgmlAsrExecutionError> {
                Ok(GgmlAsrDecoderStateContract::NoPersistentState)
            }
            fn execute(
                &self,
                _request: &GgmlAsrExecutionRequest,
            ) -> Result<GgmlAsrExecutionResult, GgmlAsrExecutionError> {
                unreachable!("this test never executes a request")
            }
        }

        let dispatch = GgmlAsrExecutionDispatch::default()
            .with_executor_for_adapter(WHISPER_GGML_ADAPTER_ID, Arc::new(NoCacheExecutor));
        dispatch.unload_all();
    }

    #[test]
    fn from_ggml_cpu_graph_error_preserves_execution_route() {
        use crate::device::execution_route::ExecutionRouteError;
        use crate::ggml_runtime::GgmlCpuGraphError;

        let route_error = ExecutionRouteError::init_failed("provider=cuda stable_id=CUDA0");
        let mapped = GgmlAsrExecutionError::from_ggml_cpu_graph_error(
            "test-executor",
            "test-adapter",
            GgmlCpuGraphError::ExecutionRoute(route_error.clone()),
        );
        assert_eq!(mapped, GgmlAsrExecutionError::ExecutionRoute(route_error));

        let other = GgmlAsrExecutionError::from_ggml_cpu_graph_error(
            "test-executor",
            "test-adapter",
            GgmlCpuGraphError::CpuBackendUnavailable,
        );
        assert!(matches!(
            other,
            GgmlAsrExecutionError::ExecutorFailed {
                executor_id: "test-executor",
                adapter_id: "test-adapter",
                ..
            }
        ));
    }

    /// A single `execute()` call must resolve this family's backend exactly
    /// once and hand every graph-build call site the SAME value -- not let
    /// some sites read a gated resolution and others an ungated one. The
    /// observable seam is the request's own `resolved_runtime` field (not a
    /// global/thread-local getter): a fake executor reads
    /// `_request.resolved_runtime.backend()` at multiple simulated call
    /// sites (mirroring how a real family reads it once per cache key /
    /// graph config) and records every read.
    ///
    /// The request is built on one OS thread and executed on a second,
    /// distinct OS thread -- the case a thread-local channel gets wrong
    /// (the value would either fail to cross or silently read the executing
    /// thread's own unrelated installation) but an explicit struct field
    /// gets right by construction, since it rides along with the value.
    #[test]
    fn dispatch_resolves_family_backend_once_and_consistently_across_call_sites() {
        use crate::ggml_runtime::GgmlCpuGraphBackend;
        use std::sync::Mutex;

        struct RecordingExecutor {
            observed: Arc<Mutex<Vec<GgmlCpuGraphBackend>>>,
        }
        impl GgmlAsrExecutor for RecordingExecutor {
            fn executor_id(&self) -> &'static str {
                "resolved-backend-consistency-stub"
            }

            fn supports_phrase_bias(&self) -> bool {
                false
            }

            fn decoder_state_contract(
                &self,
                _selected_family: &GgmlFamilyAdapterDescriptor,
            ) -> Result<GgmlAsrDecoderStateContract, GgmlAsrExecutionError> {
                Ok(GgmlAsrDecoderStateContract::NoPersistentState)
            }

            fn execute(
                &self,
                request: &GgmlAsrExecutionRequest,
            ) -> Result<GgmlAsrExecutionResult, GgmlAsrExecutionError> {
                // Three independent reads, standing in for three real
                // call sites within one family decode (e.g. an audio-encoder
                // cache key, a decoder cache key, and a graph-config
                // builder) -- all inside the SAME `execute()` call, all
                // reading the same explicit field on `request`.
                let mut observed = self.observed.lock().unwrap();
                for _ in 0..3 {
                    observed.push(request.resolved_runtime.backend());
                }
                Ok(GgmlAsrExecutionResult {
                    transcription: Transcription {
                        truncated_decodes: Vec::new(),
                        unnamed_speakers: Vec::new(),
                        text: "ok".to_string(),
                        segments: Vec::new(),
                        longform: None,
                        language: None,
                    },
                    carry_context: None,
                    decode_truncation: None,
                })
            }
        }

        // Whisper's policy is `AllBackends` (a no-op gate), so the resolved
        // value must equal the independent generic resolution exactly --
        // host-independent equality, not a fixed backend.
        let expected = crate::ggml_runtime::GgmlCpuGraphConfig::runtime_default().backend;

        // Built on the submitting thread; `resolved_runtime` is materialized
        // into the request right here, before it ever crosses a thread
        // boundary.
        let request = whisper_request(GgmlAsrBackendPreference::Auto);
        let resolved_on_submitting_thread = request.resolved_runtime.backend();

        let observed = Arc::new(Mutex::new(Vec::new()));
        let dispatch = GgmlAsrExecutionDispatch::default().with_executor_for_adapter(
            WHISPER_GGML_ADAPTER_ID,
            Arc::new(RecordingExecutor {
                observed: Arc::clone(&observed),
            }),
        );

        // Hand the already-resolved request to a second, distinct OS
        // thread and execute it there. If the resolved value depended on
        // any per-thread state instead of riding along on `request`, this
        // would be the boundary where it would go stale or diverge.
        std::thread::spawn(move || {
            dispatch
                .execute(&request)
                .expect("recording executor always succeeds");
        })
        .join()
        .expect("execution thread must not panic");

        let observed = observed.lock().unwrap();
        assert_eq!(observed.len(), 3, "all three call sites must have run");
        assert!(
            observed.iter().all(|backend| *backend == observed[0]),
            "every call site within one request must observe the identical resolved backend, got {observed:?}"
        );
        assert_eq!(
            observed[0], resolved_on_submitting_thread,
            "the backend observed on the execution thread must be identical to the value \
             resolved on the submitting thread -- it must ride the request across the \
             thread boundary, not be re-derived from execution-thread-local state"
        );
        assert_eq!(
            observed[0], expected,
            "resolved backend must match the family's (AllBackends) generic resolution"
        );
    }

    /// A family whose descriptor declares a gated `AutoGpuPolicy`
    /// (xasr-zipformer's real `ExceptMetal`) must never observe a backend
    /// the gate forbids, even though the shared
    /// dispatch is the one doing the resolving now, not the family itself.
    /// Uses a fake executor substituted for the real xasr-zipformer one so
    /// the assertion is purely about dispatch's resolution, independent of
    /// xasr-zipformer's own graph-building code.
    #[test]
    fn dispatch_honors_gated_family_auto_policy_for_registered_architecture() {
        use crate::ggml_runtime::GgmlCpuGraphBackend;
        use std::sync::Mutex;

        struct RecordingExecutor {
            observed: Arc<Mutex<Option<GgmlCpuGraphBackend>>>,
        }
        impl GgmlAsrExecutor for RecordingExecutor {
            fn executor_id(&self) -> &'static str {
                "gated-policy-stub"
            }

            fn supports_phrase_bias(&self) -> bool {
                false
            }

            fn decoder_state_contract(
                &self,
                _selected_family: &GgmlFamilyAdapterDescriptor,
            ) -> Result<GgmlAsrDecoderStateContract, GgmlAsrExecutionError> {
                Ok(GgmlAsrDecoderStateContract::NoPersistentState)
            }

            fn execute(
                &self,
                request: &GgmlAsrExecutionRequest,
            ) -> Result<GgmlAsrExecutionResult, GgmlAsrExecutionError> {
                *self.observed.lock().unwrap() = Some(request.resolved_runtime.backend());
                Ok(GgmlAsrExecutionResult {
                    transcription: Transcription {
                        truncated_decodes: Vec::new(),
                        unnamed_speakers: Vec::new(),
                        text: "ok".to_string(),
                        segments: Vec::new(),
                        longform: None,
                        language: None,
                    },
                    carry_context: None,
                    decode_truncation: None,
                })
            }
        }

        let descriptor =
            builtin_adapter_descriptor(crate::arch::XASR_ZIPFORMER_GGML_ARCHITECTURE_ID);
        let auto_gpu_policy = crate::arch::family_auto_gpu_policy_for_model_architecture(
            descriptor.model_architecture,
        );
        assert_eq!(
            auto_gpu_policy,
            crate::ggml_runtime::AutoGpuPolicy::ExceptMetal,
            "this regression only pins something if xasr-zipformer stays ExceptMetal"
        );

        let generic_auto = crate::ggml_runtime::GgmlCpuGraphConfig::runtime_default().backend;
        let observed = Arc::new(Mutex::new(None));
        let dispatch = GgmlAsrExecutionDispatch::default().with_executor_for_adapter(
            descriptor.adapter_id,
            Arc::new(RecordingExecutor {
                observed: Arc::clone(&observed),
            }),
        );
        let request = GgmlAsrExecutionRequest {
            execution_services:
                crate::models::native_execution_services::test_native_execution_services(),
            decoder_state: crate::models::ggml_asr_executor::GgmlAsrDecoderState::NoPersistentState,
            verified_pack:
                crate::models::pack_verifier::VerifiedPack::from_unverified_preflight_for_test(
                    crate::models::runtime_preflight::leaked_tiny_runtime_source_preflight(),
                    crate::arch::XASR_ZIPFORMER_GGML_ARCHITECTURE_ID,
                ),
            selected_family: descriptor,
            prepared_audio: GgmlAsrPreparedAudio::mono_16khz(vec![0.0, 0.1]),
            request_options: GgmlAsrExecutionOptions::default(),
            backend_preference: GgmlAsrBackendPreference::Auto,
            // The dispatch resolves against this family's OWN declared gate
            // (`ExceptMetal`, asserted above), not the generic `AllBackends`
            // policy -- an ungated resolution here would defeat the whole
            // point of this regression (it would let Auto pick Metal, which
            // `assert_ne!` below exists to catch).
            resolved_runtime: crate::ggml_runtime::ResolvedFamilyRuntimeInput::resolve(
                GgmlAsrBackendPreference::Auto.request_backend_override(),
                auto_gpu_policy,
            ),
            execution_context: std::sync::Arc::new(crate::RequestExecutionContext::uncancellable(
                "test fixture",
            )),
        };
        dispatch
            .execute(&request)
            .expect("recording executor always succeeds");

        let observed = observed.lock().unwrap().expect("executor must have run");
        // The gate never lets Auto pick Metal specifically for this family --
        // this is the exact defect-A shape: an ungated read here would have
        // reported whatever the generic resolver picked, including Metal.
        assert_ne!(observed, GgmlCpuGraphBackend::Metal);
        if generic_auto == GgmlCpuGraphBackend::Metal {
            assert_eq!(observed, GgmlCpuGraphBackend::Cpu);
        } else {
            assert_eq!(observed, generic_auto);
        }
    }
}
