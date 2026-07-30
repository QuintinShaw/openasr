//! Derived single-decode capacity model.
//!
//! Everything a family needs to answer "how much audio fits one decode" is
//! computed from pack metadata plus family constants -- never hand-written
//! arithmetic beside a constant (the failure mode this module replaced:
//! margin-note arithmetic next to `integral_seconds` /
//! `MOSS_TD_MAX_KV_CACHE_POSITIONS` that was wrong twice without anyone
//! noticing).
//!
//! # Invariants (design review 2026-07-27, settled)
//!
//! 1. **Machine independence, reject-not-degrade.** The integral window (the
//!    longest recording a family folds into a single decode) depends ONLY on
//!    pack metadata and family constants, never on host memory. If it depended
//!    on RAM, the same recording would be decoded whole on one machine and
//!    sliced on another -- different transcripts and different speaker
//!    numbering for identical input, making every golden and WER table
//!    machine-conditional. Resource adaptation belongs on the ADMISSION side
//!    (request refused, or a per-request clamp with stated provenance), never
//!    inside the single-request compute semantics. vLLM (`max_model_len` is a
//!    model fact; memory pressure triggers preemption/recompute, not a
//!    different algorithm), llama.cpp (`n_ctx` declared, OOM = failure) and
//!    `references/transcribe.cpp` (`docs/input-limits.md`: "does not silently
//!    shrink the context and retry") all draw the line in the same place.
//!    Refusing costs ~nothing at current numbers: 300s at the worst-case
//!    `LlmKvCacheSpec::DEFAULT` footprint is ~2.63 GiB of KV, which fits an
//!    8 GiB min-spec machine's 75% budget with room to spare.
//! 2. **Bytes per position is split host / resident, never summed into one
//!    number.** The two copies scale and are budgeted differently: the
//!    resident arena carries the `n_seq` dimension (per-lane concurrency
//!    cost) and lives in device memory on discrete GPUs, while the host copy
//!    lives in RAM. One caller wants the sum, another wants only the resident
//!    half; summing here would lock both out.
//! 3. **Bytes per position is a `(pack, backend, env)` function.**
//!    [`crate::nn::decoder::resolve_production_llm_kv_cache_policy`] can fall
//!    back to `DEFAULT` at runtime (discrete GPU, no native GQA, no flash,
//!    wrong head_dim, or `OPENASR_QWEN_KV_CACHE_F32=1`), so no static number
//!    may assume q8_0. Static reasoning must take the worst case
//!    (`DEFAULT`: 336 KiB/position for a 28L/8-kv-head/head-dim-128 decoder).
//! 4. **Frontend geometry comes from the versioned-id registry below, not
//!    from prose.** Documentation constraints have zero enforcement -- the
//!    300s constant always had a comment beside it and was still wrong twice.
//!    Derivation reads frontend facts only from [`frontend_capacity_basis`],
//!    keyed by the family's versioned `audio_frontend_id` (a fail-closed
//!    contract id); the family integration audit refuses a `Derived` family
//!    whose id has no row.
//!
//! # Phase status
//!
//! Phase 0 (this module today): pure derivation + regression anchors, ZERO
//! production callers. Tests assert the derived values equal the declared
//! constants that production still reads (moss: `integral_seconds == 300.0`,
//! the 8192-position preallocation cap, the KV byte figures). Phase 1 moves
//! the derived value onto the loaded pack (transcribe.cpp's `LimitsBasis`
//! load-time pattern) and switches production over; the pin tests are the
//! safety net that makes that swap behavior-preserving.
//!
//! The dead-code allowance is load-bearing Phase 0 semantics, not slop: in a
//! release build the derivation surface genuinely has no caller yet (only the
//! declaration enum and the frontend registry are wired, via the family
//! integration audit). Remove it when Phase 1 attaches production callers.
#![cfg_attr(not(test), allow(dead_code))]

use std::num::NonZeroU32;

use crate::nn::decoder::LlmKvCacheSpec;

/// Decoder KV-cache geometry. Every field is a pack runtime-metadata fact
/// (each family's `runtime_contract` parses them fail-closed at import), so
/// capacity questions are answered from the pack actually loaded, never from
/// a literal that could drift.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct KvGeometry {
    /// Decoder layers, each contributing one K and one V row per position.
    pub n_layers: usize,
    /// KV heads per layer (post-GQA; equals `n_heads` for MHA decoders).
    pub kv_heads: usize,
    /// Values per head row.
    pub head_dim: usize,
}

/// Bytes of KV cache one decoder position costs for one sequence, split by
/// storage copy (invariant 2 above -- never return the sum alone).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct KvBytesPerPosition {
    /// Host-side copy (`Qwen3AsrLayerKvCacheState`).
    pub host: u64,
    /// Device-resident copy (`allocate_zeroed_llm_resident_kv_arena`), the
    /// buffer Metal physically wires and the one that scales with `n_seq`.
    pub resident: u64,
}

impl KvBytesPerPosition {
    /// Both copies together: the figure to compare against a unified-memory
    /// budget (Apple Silicon wires the resident arena from the same physical
    /// pool as the host copy).
    pub(crate) fn total(&self) -> u64 {
        self.host.saturating_add(self.resident)
    }

    fn checked_mul(self, factor: u64) -> Option<Self> {
        Some(Self {
            host: self.host.checked_mul(factor)?,
            resident: self.resident.checked_mul(factor)?,
        })
    }
}

/// Per-position KV cost of `geometry` under `spec`, one row each for K and V
/// of every (layer, kv-head). Fails closed on a zero field or a head_dim the
/// element type cannot represent.
pub(crate) fn kv_bytes_per_position(
    geometry: &KvGeometry,
    spec: LlmKvCacheSpec,
) -> Result<KvBytesPerPosition, String> {
    if geometry.n_layers == 0 || geometry.kv_heads == 0 {
        return Err(format!(
            "kv geometry must have positive n_layers and kv_heads (got {geometry:?})"
        ));
    }
    let rows_per_position = geometry
        .n_layers
        .checked_mul(2)
        .and_then(|kv_rows| kv_rows.checked_mul(geometry.kv_heads))
        .ok_or_else(|| format!("kv geometry row count overflowed: {geometry:?}"))?;
    let host_row = spec.host.row_nbytes(geometry.head_dim)?;
    let resident_row = spec.resident.row_nbytes(geometry.head_dim)?;
    Ok(KvBytesPerPosition {
        host: u64::try_from(host_row)
            .ok()
            .and_then(|row| row.checked_mul(rows_per_position as u64))
            .ok_or_else(|| "kv host byte count overflowed".to_string())?,
        resident: u64::try_from(resident_row)
            .ok()
            .and_then(|row| row.checked_mul(rows_per_position as u64))
            .ok_or_else(|| "kv resident byte count overflowed".to_string())?,
    })
}

/// Total KV cost of `positions` positions for one sequence (the admission
/// figure: `positions` = prompt + granted generation budget).
pub(crate) fn kv_bytes_at_positions(
    geometry: &KvGeometry,
    spec: LlmKvCacheSpec,
    positions: usize,
) -> Result<KvBytesPerPosition, String> {
    kv_bytes_per_position(geometry, spec)?
        .checked_mul(u64::try_from(positions).unwrap_or(u64::MAX))
        .ok_or_else(|| format!("kv byte count overflowed at {positions} positions"))
}

/// Versioned frontend geometry whose audio-token rate is the stride product
/// `sample_rate_hz / (hop_length * encoder_conv_stride * adaptor_merge_size)`
/// (the `_compute_audio_token_length` decomposition `references/transcribe.cpp`'s
/// `LimitsBasis::ms_per_audio_token` uses identically).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FrontendGeometry {
    pub sample_rate_hz: usize,
    pub hop_length: usize,
    pub encoder_conv_stride: usize,
    pub adaptor_merge_size: usize,
}

/// How a frontend's audio-token rate is established, keyed by
/// `audio_frontend_id` in [`FRONTEND_CAPACITY_REGISTRY`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AudioFrontendCapacityBasis {
    /// The rate is a Rust-side architecture constant (the frontend's shape is
    /// fixed by the family, not carried per pack). Derivation reads it here.
    Constant(FrontendGeometry),
    /// The rate's inputs are required pack metadata, fail-closed at import
    /// (the family's `runtime_contract` refuses a pack missing them). When
    /// this family grows a derived integral window, its derivation reads the
    /// parsed pack metadata -- the registry records WHERE the facts live so
    /// the obligation is visible, not prose.
    PackCarried { provenance: &'static str },
}

/// Audio tokens per second of input audio this frontend emits.
pub(crate) fn audio_tokens_per_second(geometry: &FrontendGeometry) -> f32 {
    let divisor = (geometry.hop_length.max(1) * geometry.encoder_conv_stride.max(1))
        .saturating_mul(geometry.adaptor_merge_size.max(1));
    geometry.sample_rate_hz as f32 / divisor as f32
}

/// Frontend capacity facts keyed by the family's versioned
/// `audio_frontend_id`. Derivation reads frontend geometry ONLY from here
/// (invariant 4); the family integration audit refuses a
/// [`crate::capacity::CapacityModelDeclaration::Derived`] family whose id has
/// no row. The versioned-id convention forbids the one remaining exposure: a
/// pack re-cut that changed frontend parameters while keeping an old id would
/// be a contract violation by construction. Moving the constants into the
/// pack itself (transcribe.cpp reads hop/sample-rate from GGUF) is strictly
/// better and queued behind each family's next natural re-cut.
const FRONTEND_CAPACITY_REGISTRY: &[(&str, AudioFrontendCapacityBasis)] = &[
    (
        crate::arch::MOSS_TD_AUDIO_FRONTEND_ID,
        AudioFrontendCapacityBasis::Constant(FrontendGeometry {
            // `WhisperFeatureExtractor`'s 16kHz/160-hop mel + the Whisper conv
            // stem's 2x stride + the adaptor's 4x time-merge -- the same
            // architecture constants `moss_transcribe_diarize::executor`'s
            // `SAMPLE_RATE_HZ` / `HOP_LENGTH` / `WHISPER_ENCODER_CONV_STRIDE`
            // state (pinned equal by that family's capacity tests) and
            // `moss_transcribe_diarize::decode_prompt`'s 12.5 tokens/s
            // marker cadence is derived from.
            sample_rate_hz: 16_000,
            hop_length: 160,
            encoder_conv_stride: 2,
            adaptor_merge_size: 4,
        }),
    ),
    (
        crate::arch::QWEN3_ASR_AUDIO_FRONTEND_ID,
        AudioFrontendCapacityBasis::PackCarried {
            provenance: "sample_rate_hz and hop_length are required pack metadata \
                         (qwen::runtime_contract), fail-closed at import; the encoder's \
                         2x conv-stem downsample is an architecture constant",
        },
    ),
    (
        crate::arch::COHERE_TRANSCRIBE_AUDIO_FRONTEND_ID,
        AudioFrontendCapacityBasis::PackCarried {
            provenance: "sample_rate and hop_length are required pack metadata \
                         (cohere::runtime_contract COHERE_TRANSCRIBE_AUDIO_SAMPLE_RATE_KEY / \
                         COHERE_TRANSCRIBE_AUDIO_HOP_LENGTH_KEY), fail-closed at import",
        },
    ),
    (
        crate::arch::FIRERED_LLM_AUDIO_FRONTEND_ID,
        AudioFrontendCapacityBasis::PackCarried {
            provenance: "fbank frontend contract shared with firered-aed; the adapter's \
                         frame-stacking factor is required pack metadata \
                         (firered_llm.adapter.downsample_rate), fail-closed at import",
        },
    ),
    (
        crate::arch::MIMO_ASR_AUDIO_FRONTEND_ID,
        AudioFrontendCapacityBasis::PackCarried {
            provenance: "mel sample_rate/hop (mimo.mel.*) and tokenizer conv strides \
                         (mimo.tok.*) are required pack metadata, fail-closed at import; \
                         the RVQ tokenizer runs at 25Hz frames and the group-4 input-local \
                         downcast sets the final audio-token rate",
        },
    ),
];

/// The capacity basis a frontend id declares, if any. `None` for families
/// whose capacity model is not `Derived` (they have nothing to derive from)
/// and for unknown ids.
pub(crate) fn frontend_capacity_basis(
    audio_frontend_id: &str,
) -> Option<&'static AudioFrontendCapacityBasis> {
    FRONTEND_CAPACITY_REGISTRY
        .iter()
        .find(|(id, _)| *id == audio_frontend_id)
        .map(|(_, basis)| basis)
}

/// How a family's single-decode capacity is established -- a MANDATORY field
/// of [`crate::arch::OpenAsrArchitectureDescriptor`], so a new architecture
/// cannot compile without placing itself in exactly one bucket.
///
/// Deliberately NOT an `Option`: a `None` could not distinguish "evaluated and
/// confirmed this family needs no capacity derivation" from "never thought
/// about it", and that ambiguity is exactly the silent-regression entry this
/// declaration closes. Leaving the field out of a `const` descriptor entry is
/// a compile error, which is stronger than any visibility convention.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CapacityModelDeclaration {
    /// This family's single-decode capacity derives from pack metadata plus
    /// the frontend registry ([`frontend_capacity_basis`]): KV geometry from
    /// the loaded pack, frontend rate from its versioned id, family constants
    /// for the rest. The family integration audit refuses a `Derived` family
    /// whose `audio_frontend_id` has no registry row, and the owning family
    /// pins its derived figures against its declared constants (moss:
    /// `moss_transcribe_diarize::capacity`'s regression anchors).
    Derived(CapacityModelDescriptor),
    /// Evaluated: this family has no autoregressive decoder KV cache (CTC /
    /// transducer shapes), so there is no per-position capacity to derive --
    /// the question this module answers does not arise.
    NoDecoderKv,
    /// Evaluated: this family has a decoder KV cache, but audio length in one
    /// decode is bounded by something else (`by` names it -- a fixed encoder
    /// window, an encoder positional span, an output-context cap). Deriving
    /// the KV figure would answer a question that does not bind this family.
    BoundedElsewhere { by: &'static str },
}

/// Compile-time facts a `Derived` family declares up front (the derived
/// VALUES live on the loaded pack -- Phase 1 resolves them at import time,
/// transcribe.cpp's `LimitsBasis` pattern; the descriptor stays a `const`
/// array of compile-time facts).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CapacityModelDescriptor {
    /// What bounds audio length in a single decode for this family --
    /// transcribe.cpp `LimitsBasis::audio_from_caps`'s distinction, which
    /// OpenASR previously did not model: for most LLM-decoder families the
    /// audio tokens are spliced into the decoder prompt, so the decoder's KV
    /// context binds audio length ([`CapacityAudioBound::DecoderContext`]);
    /// for others the ENCODER's positional span binds audio while the decoder
    /// context still binds KV memory ([`CapacityAudioBound::EncoderSpan`]),
    /// and the two constraints resolve separately.
    pub audio_bound: CapacityAudioBound,
}

/// Which side of the model bounds audio length in one decode (see
/// [`CapacityModelDescriptor::audio_bound`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CapacityAudioBound {
    /// Audio tokens feed the decoder prompt, so the decoder KV context is the
    /// binding audio-length constraint (moss, qwen3-asr, firered-llm,
    /// mimo-asr).
    DecoderContext,
    /// The encoder's positional table bounds audio length; the decoder
    /// context still bounds KV memory but is not what limits the recording
    /// (cohere-transcribe).
    EncoderSpan,
}

/// Decimal digits in `value` -- one BPE token each for a family's time-marker
/// track (moss's tokenizer emits exactly one token per digit; verified against
/// the golden fixtures' decoded marker runs).
fn decimal_digit_count(mut value: u64) -> usize {
    let mut digits = 0;
    loop {
        digits += 1;
        value /= 10;
        if value == 0 {
            return digits;
        }
    }
}

/// Token cost of the time-anchor marker track for an audio span of
/// `audio_token_count` tokens: a marker firing every `marker_every_seconds`
/// (at 5, 10, 15, ... elapsed seconds, inclusive of the endpoint) whose cost
/// is the digit count of its seconds value. Mirrors
/// `moss_transcribe_diarize::decode_prompt::audio_span_ids`'s firing rule
/// exactly (`sec <= duration` on the truncated whole-second duration), so the
/// derivation counts the same tokens the real prompt contains -- the flat
/// overhead constant this replaced only happened to stay conservative below
/// ~1000s and silently stopped being conservative past it.
pub(crate) fn marker_digit_tokens(
    audio_token_count: usize,
    audio_tokens_per_second: f32,
    marker_every_seconds: NonZeroU32,
) -> usize {
    if audio_tokens_per_second <= 0.0 {
        return 0;
    }
    let duration_seconds = (audio_token_count as f32 / audio_tokens_per_second) as u64;
    let every = marker_every_seconds.get() as u64;
    let marker_count = duration_seconds / every;
    (1..=marker_count)
        .map(|index| decimal_digit_count(index * every))
        .sum()
}

/// Family-agnostic inputs for deriving a decoder-context-bound family's
/// integral window. Every value is a pack-metadata or family-constant fact
/// (invariant 1); none comes from the host.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct IntegralWindowDerivation {
    /// The most decoder positions one decode may occupy: the pack's advertised
    /// ceiling clamped by the family's preallocation cap (moss:
    /// `min(pack max_positions, 8192)`).
    pub kv_position_ceiling: usize,
    /// Encoder chunk quantum; windows are only meaningful in whole chunks (a
    /// partial chunk still costs a full one's audio tokens).
    pub chunk_seconds: f32,
    /// Post-merge audio tokens one full encoder chunk contributes to the
    /// prompt.
    pub audio_tokens_per_chunk: usize,
    /// Audio tokens per second of input (drives the marker track; see
    /// [`audio_tokens_per_second`]).
    pub audio_tokens_per_second: f32,
    /// Tokens of the fixed prompt wrapper (chat template + instruction +
    /// audio-span delimiters), measured from the family's real tokenized
    /// prompt -- never a round-number guess.
    pub fixed_prompt_tokens: usize,
    /// Time-anchor marker cadence, if the family's prompt has one (`None`
    /// removes the marker term).
    pub marker_every_seconds: Option<NonZeroU32>,
    /// Densest generation demand actually measured for the family; the budget
    /// a window must leave room for.
    pub densest_generated_tokens_per_second: f32,
    /// The family's runaway-generation backstop: required generation is
    /// `min(ceil(window * densest), this)`. A FIRST-CLASS input, not a comment
    /// -- at current numbers this backstop (moss: 4096), not the position
    /// ceiling alone, is what stops the window growing past 300s. Whether the
    /// backstop itself may rise is a MODEL-BEHAVIOR question (checkpoint
    /// reference configuration, runaway-repetition risk), not a capacity one;
    /// capacity engineering never varies it.
    pub max_generated_tokens: usize,
}

impl IntegralWindowDerivation {
    /// Prompt tokens a `chunks`-chunk request costs: fixed wrapper + one full
    /// chunk's audio tokens per chunk + the marker track over the whole span.
    pub(crate) fn prompt_tokens_for_chunks(&self, chunks: usize) -> usize {
        let audio_tokens = self.audio_tokens_per_chunk.saturating_mul(chunks);
        let markers = match self.marker_every_seconds {
            Some(every) => marker_digit_tokens(audio_tokens, self.audio_tokens_per_second, every),
            None => 0,
        };
        self.fixed_prompt_tokens
            .saturating_add(audio_tokens)
            .saturating_add(markers)
    }

    /// Decoder positions a `chunks`-chunk single decode requires: the prompt
    /// plus a generation budget covering the densest measured demand, capped
    /// at the runaway backstop. Saturates (rather than overflows) so a
    /// pathological input fails closed against the ceiling instead of
    /// wrapping into a small number that "fits".
    pub(crate) fn required_positions_for_chunks(&self, chunks: usize) -> usize {
        let window_seconds = chunks as f32 * self.chunk_seconds;
        let densest_demand =
            (window_seconds * self.densest_generated_tokens_per_second).ceil() as usize;
        self.prompt_tokens_for_chunks(chunks)
            .saturating_add(densest_demand.min(self.max_generated_tokens))
    }
}

/// The largest whole-chunk window whose required positions still fit the
/// ceiling -- the integral (no-slicing) window, derived. `None` is the
/// fail-closed result: not even one chunk fits, so the family cannot serve a
/// single decode at this geometry and callers must refuse rather than guess.
///
/// Required positions grow strictly with the chunk count (every term is
/// non-decreasing and the audio-token term is strictly increasing), so the
/// first window that does not fit ends the search.
pub(crate) fn derive_integral_seconds(derivation: &IntegralWindowDerivation) -> Option<f32> {
    if derivation.kv_position_ceiling == 0
        || derivation.audio_tokens_per_chunk == 0
        || !derivation.chunk_seconds.is_finite()
        || derivation.chunk_seconds <= 0.0
    {
        return None;
    }
    let mut largest_fitting = None;
    let mut chunks = 1usize;
    while derivation.required_positions_for_chunks(chunks) <= derivation.kv_position_ceiling {
        largest_fitting = Some(chunks);
        chunks = chunks.checked_add(1)?;
    }
    largest_fitting.map(|chunks| chunks as f32 * derivation.chunk_seconds)
}

/// A family's decoder KV footprint for THIS request clearly does not fit this
/// host's memory budget -- the reject-not-degrade admission outcome (design
/// review 2026-07-27, point A: refuse rather than silently reslice/requant,
/// which would make the same recording produce a different transcript on a
/// 16 GiB machine than on a 64 GiB one). Every field the caller needs to
/// render both a compact machine-readable trailer and a human paragraph is
/// kept here rather than collapsed to a bool, so `Display` never has to be
/// reverse-engineered by a test asserting on message text alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct HostMemoryCapacityRejection {
    /// KV bytes at `required_positions` plus the pack's own on-disk bytes --
    /// the two quantities [`evaluate_host_memory_admission`] actually knows.
    pub needed_bytes: u64,
    /// `host_total_memory_bytes * 3/4` (`crate::host::host_memory_budget_bytes`).
    pub budget_bytes: u64,
    pub host_total_memory_bytes: u64,
    pub pack_bytes_on_disk: u64,
    pub required_positions: usize,
}

impl HostMemoryCapacityRejection {
    /// Compact `key:value,k=v,...` trailer, the same shape the
    /// `core.native.longform.policy:...` provenance strings already use
    /// elsewhere in this crate -- the full arithmetic a bug report needs,
    /// in one line.
    pub(crate) fn provenance(&self) -> String {
        format!(
            "core.native.capacity.admission:reject,needed_bytes={},budget_bytes={},host_total_memory_bytes={},pack_bytes_on_disk={},required_positions={}",
            self.needed_bytes,
            self.budget_bytes,
            self.host_total_memory_bytes,
            self.pack_bytes_on_disk,
            self.required_positions,
        )
    }

    /// User-facing English paragraph: what this request needed, what this
    /// host offers, and what to try instead -- a clear error is not enough on
    /// its own (the bar is "the user knows the next step"), and mirrors the
    /// numbers-plus-suggestion shape `firered_llm::executor`'s existing
    /// pack-does-not-fit-this-host's-budget message already uses.
    pub(crate) fn user_message(&self) -> String {
        const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
        format!(
            "This request needs about {needed_gib:.1} GiB (the model pack plus its decode \
             context for {positions} decoder positions), but this host's admission budget is \
             only {budget_gib:.1} GiB ({host_gib:.1} GiB total RAM x 75%).\n\
             Try a smaller quantization (q8_0 or q4_k instead of fp16), a smaller model, or \
             close other memory-heavy applications and retry.\n\
             The request was rejected before building the decode graph, instead of failing \
             later with an opaque ggml allocation error.\n\
             {provenance}",
            needed_gib = self.needed_bytes as f64 / GIB,
            positions = self.required_positions,
            budget_gib = self.budget_bytes as f64 / GIB,
            host_gib = self.host_total_memory_bytes as f64 / GIB,
            provenance = self.provenance(),
        )
    }
}

/// Evaluate whether `required_positions` decoder positions at `spec` for
/// `geometry`, alongside `pack_bytes_on_disk` already resident from the
/// mmap'd pack file, fit this host's memory budget
/// (`crate::host::host_memory_budget_bytes`: 75% of total RAM, never
/// `host_available_memory_bytes` -- see this module's invariants and
/// `host_available_memory_bytes`'s own doc comment forbidding admission use).
///
/// Fails OPEN on a degenerate geometry (`kv_bytes_at_positions` erring):
/// geometry validity is enforced elsewhere (pack import), not here, and
/// "uncertain" always resolves to "allow" per invariant 1 -- refusing only
/// when certain it will not fit. Deliberately omits any reserve for encoder
/// activations / mel / compute graph beyond the two quantities exactly known
/// here: inventing a family-general "working set" constant with no measured
/// figure to draw from would either be large enough to risk rejecting a
/// request that would actually fit (the one outcome invariant 1 forbids) or
/// small enough not to matter, so the 25% of total memory the 75% budget
/// already sets aside is left to cover it instead of a second guessed number.
///
/// Assumes unified memory: `needed_bytes` sums KV cache and pack bytes as if
/// both draw from the same host RAM pool the mmap'd pack resides in and the
/// budget bounds. On a host with a discrete GPU, resident decoder state can
/// instead live in VRAM (a separate budget this function does not see) while
/// the mmap'd pack's pages stay reclaimable host-side, so this arithmetic is
/// conservative-to-a-fault there rather than wrong in the unsafe direction --
/// but it has not been exercised against a discrete-GPU host yet (moss-
/// transcribe-diarize, the only family wired to this check so far, stays
/// small enough that it does not matter in practice). Before wiring a family
/// that runs meaningfully large on a discrete GPU (e.g. qwen3-asr at fp16,
/// ~4.7 GiB), this needs device-aware accounting: a separate VRAM budget for
/// resident state, and the pack's on-disk bytes should not be charged against
/// host RAM at their full mmap size once they are known to be reclaimable.
pub(crate) fn evaluate_host_memory_admission(
    geometry: &KvGeometry,
    spec: LlmKvCacheSpec,
    required_positions: usize,
    pack_bytes_on_disk: u64,
    host_total_memory_bytes: u64,
) -> Result<(), HostMemoryCapacityRejection> {
    let Ok(kv_bytes) = kv_bytes_at_positions(geometry, spec, required_positions) else {
        return Ok(());
    };
    let needed_bytes = kv_bytes.total().saturating_add(pack_bytes_on_disk);
    let budget_bytes = crate::host::host_memory_budget_bytes(host_total_memory_bytes);
    if needed_bytes <= budget_bytes {
        return Ok(());
    }
    Err(HostMemoryCapacityRejection {
        needed_bytes,
        budget_bytes,
        host_total_memory_bytes,
        pack_bytes_on_disk,
        required_positions,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn moss_geometry() -> KvGeometry {
        // The real checkpoint's decoder (same values as
        // moss_transcribe_diarize::runtime_contract::tests::full_metadata).
        KvGeometry {
            n_layers: 28,
            kv_heads: 8,
            head_dim: 128,
        }
    }

    #[test]
    fn kv_bytes_per_position_splits_host_and_resident_copies() {
        // f32 host row = 128 * 4 = 512 B; f16 resident row = 128 * 2 = 256 B;
        // 28 layers * 2 (K+V) * 8 kv-heads = 448 rows per position.
        let default =
            kv_bytes_per_position(&moss_geometry(), LlmKvCacheSpec::DEFAULT).expect("default");
        assert_eq!(default.host, 448 * 512); // 224 KiB -- the historical figure, host copy ONLY
        assert_eq!(default.resident, 448 * 256); // 112 KiB
        assert_eq!(default.total(), 448 * 768); // 336 KiB -- the real DEFAULT cost

        // q8_0 row = 128 / 32 * 34 = 136 B, BOTH copies q8_0 under this spec.
        let q8_0 = kv_bytes_per_position(&moss_geometry(), LlmKvCacheSpec::Q8_0).expect("q8_0");
        assert_eq!(q8_0.host, 448 * 136); // 59.5 KiB
        assert_eq!(q8_0.resident, 448 * 136);
        assert_eq!(q8_0.total(), 448 * 272); // 119 KiB -- 2.8x under DEFAULT, not ~4x
    }

    #[test]
    fn kv_bytes_per_position_rejects_degenerate_geometry() {
        let bad = KvGeometry {
            n_layers: 0,
            kv_heads: 8,
            head_dim: 128,
        };
        assert!(kv_bytes_per_position(&bad, LlmKvCacheSpec::DEFAULT).is_err());
        // q8_0 cannot represent a head_dim that is not a multiple of 32.
        let unaligned = KvGeometry {
            n_layers: 28,
            kv_heads: 8,
            head_dim: 100,
        };
        assert!(kv_bytes_per_position(&unaligned, LlmKvCacheSpec::Q8_0).is_err());
    }

    #[test]
    fn kv_bytes_at_positions_scales_linearly() {
        let at_8192 =
            kv_bytes_at_positions(&moss_geometry(), LlmKvCacheSpec::DEFAULT, 8192).expect("bytes");
        assert_eq!(at_8192.host, 448 * 512 * 8192); // 1.75 GiB host
        assert_eq!(at_8192.resident, 448 * 256 * 8192); // 0.875 GiB resident
        assert_eq!(at_8192.total(), 448 * 768 * 8192); // 2.625 GiB total
    }

    #[test]
    fn frontend_registry_carries_every_derivable_family() {
        let moss =
            frontend_capacity_basis(crate::arch::MOSS_TD_AUDIO_FRONTEND_ID).expect("moss row");
        assert_eq!(
            *moss,
            AudioFrontendCapacityBasis::Constant(FrontendGeometry {
                sample_rate_hz: 16_000,
                hop_length: 160,
                encoder_conv_stride: 2,
                adaptor_merge_size: 4,
            })
        );
        // The other four derivable families declare WHERE their facts live.
        for id in [
            crate::arch::QWEN3_ASR_AUDIO_FRONTEND_ID,
            crate::arch::COHERE_TRANSCRIBE_AUDIO_FRONTEND_ID,
            crate::arch::FIRERED_LLM_AUDIO_FRONTEND_ID,
            crate::arch::MIMO_ASR_AUDIO_FRONTEND_ID,
        ] {
            assert!(
                matches!(
                    frontend_capacity_basis(id),
                    Some(AudioFrontendCapacityBasis::PackCarried { provenance })
                        if !provenance.is_empty()
                ),
                "'{id}' must carry a non-empty PackCarried provenance"
            );
        }
        assert_eq!(frontend_capacity_basis("not-a-frontend-id"), None);
    }

    #[test]
    fn constant_frontend_rows_are_well_formed() {
        for (id, basis) in FRONTEND_CAPACITY_REGISTRY {
            if let AudioFrontendCapacityBasis::Constant(geometry) = basis {
                assert!(
                    geometry.sample_rate_hz > 0
                        && geometry.hop_length > 0
                        && geometry.encoder_conv_stride > 0
                        && geometry.adaptor_merge_size > 0,
                    "'{id}' frontend geometry must have no zero field: {geometry:?}"
                );
            }
        }
    }

    #[test]
    fn audio_tokens_per_second_is_the_stride_product() {
        let AudioFrontendCapacityBasis::Constant(geometry) =
            frontend_capacity_basis(crate::arch::MOSS_TD_AUDIO_FRONTEND_ID).expect("moss")
        else {
            panic!("moss frontend must be a Constant");
        };
        // 16000 / (160 * 2 * 4) = 12.5 -- the rate the family's marker cadence
        // and limit messages already state.
        assert_eq!(audio_tokens_per_second(geometry), 12.5);
    }

    #[test]
    fn marker_digit_tokens_matches_the_real_prompt_construction() {
        let every = NonZeroU32::new(5).expect("cadence");
        // jfk.wav's real span: 138 audio tokens at 12.5/s = 11.04s -> markers
        // at 5 and 10 -> '5' (1 digit) + '10' (2 digits) = 3, exactly the
        // three non-pad tokens decode_prompt's golden fixture verifies.
        assert_eq!(marker_digit_tokens(138, 12.5, every), 3);
        // 300s: 60 markers -> 1 one-digit + 18 two-digit + 41 three-digit
        // = 1 + 36 + 123 = 160.
        assert_eq!(marker_digit_tokens(3750, 12.5, every), 160);
        // 330s: 66 markers -> 1 + 36 + 141 = 178.
        assert_eq!(marker_digit_tokens(4125, 12.5, every), 178);
        // Under the first marker: nothing fires.
        assert_eq!(marker_digit_tokens(50, 12.5, every), 0);
    }

    #[test]
    fn decimal_digit_counts_cover_the_ranges() {
        assert_eq!(decimal_digit_count(5), 1);
        assert_eq!(decimal_digit_count(10), 2);
        assert_eq!(decimal_digit_count(95), 2);
        assert_eq!(decimal_digit_count(100), 3);
        assert_eq!(decimal_digit_count(300), 3);
    }

    fn moss_shaped_derivation() -> IntegralWindowDerivation {
        // The real moss input (moss_transcribe_diarize::capacity assembles it
        // from pack metadata; this copy keeps the module's own tests
        // self-contained).
        IntegralWindowDerivation {
            kv_position_ceiling: 8192,
            chunk_seconds: 30.0,
            audio_tokens_per_chunk: 375,
            audio_tokens_per_second: 12.5,
            fixed_prompt_tokens: 86,
            marker_every_seconds: NonZeroU32::new(5),
            densest_generated_tokens_per_second: 12.7,
            max_generated_tokens: 4096,
        }
    }

    #[test]
    fn required_positions_walks_the_moss_window_from_both_sides() {
        let derivation = moss_shaped_derivation();
        // 300s: 86 fixed + 3750 audio + 160 markers = 3996 prompt,
        // ceil(300 * 12.7) = 3810 generation -> 7806 <= 8192.
        assert_eq!(derivation.prompt_tokens_for_chunks(10), 3996);
        assert_eq!(derivation.required_positions_for_chunks(10), 7806);
        // 330s: 86 + 4125 + 178 = 4389 prompt, backstop-clamped 4096
        // generation -> 8485 > 8192.
        assert_eq!(derivation.required_positions_for_chunks(11), 8485);
    }

    #[test]
    fn derives_the_moss_integral_window() {
        assert_eq!(
            derive_integral_seconds(&moss_shaped_derivation()),
            Some(300.0)
        );
    }

    #[test]
    fn derivation_fails_closed_on_degenerate_input() {
        let mut derivation = moss_shaped_derivation();
        derivation.kv_position_ceiling = 0;
        assert_eq!(derive_integral_seconds(&derivation), None);
        let mut derivation = moss_shaped_derivation();
        derivation.audio_tokens_per_chunk = 0;
        assert_eq!(derive_integral_seconds(&derivation), None);
        // A ceiling that cannot serve the fixed wrapper alone admits nothing.
        let mut derivation = moss_shaped_derivation();
        derivation.kv_position_ceiling = 10;
        assert_eq!(derive_integral_seconds(&derivation), None);
    }

    #[test]
    fn derivation_without_markers_omits_the_term() {
        let mut derivation = moss_shaped_derivation();
        derivation.marker_every_seconds = None;
        // 300s prompt drops the 160 marker tokens: 3996 - 160 = 3836.
        assert_eq!(derivation.prompt_tokens_for_chunks(10), 3836);
        // The window still lands on 300s: 3836 + 3810 = 7646 <= 8192 and
        // 330s = (86 + 4125) + 4096 = 8307 > 8192.
        assert_eq!(derive_integral_seconds(&derivation), Some(300.0));
    }

    /// The pack-alone-does-not-fit shape: a multi-gigabyte pack on a host
    /// whose budget cannot possibly hold it, independent of how many KV
    /// positions the request needs -- the pack bytes alone (qwen3-asr-1.7b's
    /// real shipped fp16 size, used here only as a realistic oversized-pack
    /// figure; this admission check does not run for qwen3-asr yet) already
    /// exceed a 1 GiB host's 75% budget. Reproduces the scenario with an
    /// injected test-constructed budget rather than a real 16 GiB machine.
    #[test]
    fn host_memory_admission_rejects_a_pack_that_plainly_does_not_fit_a_tiny_host() {
        let geometry = moss_geometry();
        let tiny_host_total_memory_bytes: u64 = 1024 * 1024 * 1024; // 1 GiB
        let oversized_pack_bytes_on_disk: u64 = 4_704_801_920; // qwen3-asr-1.7b fp16 .oasr
        let rejection = evaluate_host_memory_admission(
            &geometry,
            LlmKvCacheSpec::DEFAULT,
            512,
            oversized_pack_bytes_on_disk,
            tiny_host_total_memory_bytes,
        )
        .expect_err("a multi-GiB pack cannot fit a 1 GiB host's budget");
        assert_eq!(
            rejection.budget_bytes,
            crate::host::host_memory_budget_bytes(tiny_host_total_memory_bytes)
        );
        assert!(rejection.needed_bytes > rejection.budget_bytes);

        let message = rejection.user_message();
        assert!(message.contains("needs about"), "{message}");
        assert!(message.contains("admission budget is only"), "{message}");
        assert!(message.contains("Try a smaller quantization"), "{message}");
        assert!(
            message.contains("core.native.capacity.admission:reject"),
            "{message}"
        );
        assert!(
            message.contains(&format!("needed_bytes={}", rejection.needed_bytes)),
            "{message}"
        );
        assert!(
            message.contains(&format!("budget_bytes={}", rejection.budget_bytes)),
            "{message}"
        );
    }

    /// Reverse case: the family's own worst-case ceiling (8192 positions,
    /// `DEFAULT` policy) plus a modest pack size must still fit the repo's own
    /// 8 GiB min-spec floor -- the same fact
    /// `declared_position_cap_fits_min_spec_budget_under_every_policy` in
    /// `moss_transcribe_diarize::capacity` already pins, exercised here
    /// through the admission function a real request calls rather than the
    /// raw KV arithmetic. Currently-working combos on a min-spec machine must
    /// stay working.
    #[test]
    fn host_memory_admission_allows_the_shipped_worst_case_on_a_min_spec_host() {
        let geometry = moss_geometry();
        let modest_pack_bytes_on_disk: u64 = 1_500_000_000; // ~1.4 GiB, a q4_k-class pack
        evaluate_host_memory_admission(
            &geometry,
            LlmKvCacheSpec::DEFAULT,
            8192,
            modest_pack_bytes_on_disk,
            crate::host::MIN_SPEC_TOTAL_MEMORY_BYTES,
        )
        .expect(
            "the shipped worst-case ceiling plus a modest pack must fit the 8 GiB min-spec budget",
        );
    }

    /// Fail open, not closed: a degenerate geometry is a different question
    /// (`kv_bytes_per_position` already rejects it at pack-import time), so
    /// admission must not ALSO refuse on it -- "uncertain" resolves to
    /// "allow" per invariant 1.
    #[test]
    fn host_memory_admission_fails_open_on_degenerate_geometry() {
        let degenerate = KvGeometry {
            n_layers: 0,
            kv_heads: 8,
            head_dim: 128,
        };
        assert!(
            evaluate_host_memory_admission(&degenerate, LlmKvCacheSpec::DEFAULT, 8192, 0, 1024)
                .is_ok()
        );
    }
}
