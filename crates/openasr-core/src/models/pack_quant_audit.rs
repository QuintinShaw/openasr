//! Quantization-strategy self-check for `.oasr` packs: replay the current
//! tensor-quantization policy against a pack's own tensor index and fail
//! closed when an audio-encoder tensor sits below the Q8_0 safety floor (or
//! any tensor exceeds the tier the pack claims).
//!
//! This is the structural lesson of the q4_k over-quantization incident:
//! sub-Q8 block quantization of the acoustic encoder is a behavioral cliff
//! (long-audio greedy decode collapses into repetition or empty output), and
//! nothing in a pack used to answer "was this built under the policy that
//! floors the encoder at Q8_0?". The audit works on the GGUF header alone --
//! metadata + tensor names/dims/types, all in a file's first few megabytes --
//! so it also runs against published, remotely-hosted, read-only packs via an
//! HTTP `Range` prefix fetch, with no source weights and no inference.
//!
//! The encoder/decoder split is keyed on the RUNTIME tensor names written
//! into the pack and on the pack's `openasr.model.architecture` metadata, so
//! the check needs no source checkout -- but it is NOT a hand-copied guess at
//! each family's importer classification. `audio_encoder_tensors_for_architecture`
//! below references each family's own `AUDIO_ENCODER_TENSOR_NAME_PREFIXES`
//! constant (declared beside that family's `QuantComponent` classification in
//! its `package_import` module), so the two can never drift apart silently:
//! a renamed encoder namespace moves both call sites at once, and a family
//! with no such constant simply does not compile. The tier-ceiling check
//! (`declared_tier_allows`) is likewise derived by calling the shared policy
//! function directly (see below) instead of re-deriving its per-tier rung
//! table by hand. The floor policy itself is
//! [`crate::models::pack_quant::classify_quant_tensor`]; a pack that violates
//! it was built by code predating the floor (or by a regression), and must
//! not ship.

use std::borrow::Cow;
use std::io::Read;
use std::path::{Path, PathBuf};

use thiserror::Error;

use crate::ggml_runtime::GgufWriteTensorType;
use crate::ggml_runtime::gguf_header::{GgufHeaderError, GgufHeaderView, parse_gguf_header};
use crate::models::pack_quant::{PackQuant, QuantComponent, classify_quant_tensor};

// --- ggml type ids (stable ggml ABI wire values) ---------------------------

pub const GGML_TYPE_F32: u32 = 0;
pub const GGML_TYPE_F16: u32 = 1;
pub const GGML_TYPE_Q4_0: u32 = 2;
pub const GGML_TYPE_Q4_1: u32 = 3;
pub const GGML_TYPE_Q5_0: u32 = 6;
pub const GGML_TYPE_Q5_1: u32 = 7;
pub const GGML_TYPE_Q8_0: u32 = 8;
pub const GGML_TYPE_Q8_1: u32 = 9;
pub const GGML_TYPE_Q2_K: u32 = 10;
pub const GGML_TYPE_Q3_K: u32 = 11;
pub const GGML_TYPE_Q4_K: u32 = 12;
pub const GGML_TYPE_Q5_K: u32 = 13;
pub const GGML_TYPE_Q6_K: u32 = 14;
pub const GGML_TYPE_Q8_K: u32 = 15;
pub const GGML_TYPE_IQ2_XXS: u32 = 16;
pub const GGML_TYPE_IQ2_XS: u32 = 17;
pub const GGML_TYPE_IQ3_XXS: u32 = 18;
pub const GGML_TYPE_IQ1_S: u32 = 19;
pub const GGML_TYPE_IQ4_NL: u32 = 20;
pub const GGML_TYPE_IQ3_S: u32 = 21;
pub const GGML_TYPE_IQ2_S: u32 = 22;
pub const GGML_TYPE_IQ4_XS: u32 = 23;
pub const GGML_TYPE_I8: u32 = 24;
pub const GGML_TYPE_I16: u32 = 25;
pub const GGML_TYPE_I32: u32 = 26;
pub const GGML_TYPE_I64: u32 = 27;
pub const GGML_TYPE_F64: u32 = 28;
pub const GGML_TYPE_IQ1_M: u32 = 29;
pub const GGML_TYPE_BF16: u32 = 30;

/// Human-readable ggml type name for diagnostics.
pub fn ggml_type_name(ggml_type: u32) -> Cow<'static, str> {
    match ggml_type {
        GGML_TYPE_F32 => "f32".into(),
        GGML_TYPE_F16 => "f16".into(),
        GGML_TYPE_Q4_0 => "q4_0".into(),
        GGML_TYPE_Q4_1 => "q4_1".into(),
        GGML_TYPE_Q5_0 => "q5_0".into(),
        GGML_TYPE_Q5_1 => "q5_1".into(),
        GGML_TYPE_Q8_0 => "q8_0".into(),
        GGML_TYPE_Q8_1 => "q8_1".into(),
        GGML_TYPE_Q2_K => "q2_k".into(),
        GGML_TYPE_Q3_K => "q3_k".into(),
        GGML_TYPE_Q4_K => "q4_k".into(),
        GGML_TYPE_Q5_K => "q5_k".into(),
        GGML_TYPE_Q6_K => "q6_k".into(),
        GGML_TYPE_Q8_K => "q8_k".into(),
        GGML_TYPE_IQ2_XXS => "iq2_xxs".into(),
        GGML_TYPE_IQ2_XS => "iq2_xs".into(),
        GGML_TYPE_IQ3_XXS => "iq3_xxs".into(),
        GGML_TYPE_IQ1_S => "iq1_s".into(),
        GGML_TYPE_IQ4_NL => "iq4_nl".into(),
        GGML_TYPE_IQ3_S => "iq3_s".into(),
        GGML_TYPE_IQ2_S => "iq2_s".into(),
        GGML_TYPE_IQ4_XS => "iq4_xs".into(),
        GGML_TYPE_I8 => "i8".into(),
        GGML_TYPE_I16 => "i16".into(),
        GGML_TYPE_I32 => "i32".into(),
        GGML_TYPE_I64 => "i64".into(),
        GGML_TYPE_F64 => "f64".into(),
        GGML_TYPE_IQ1_M => "iq1_m".into(),
        GGML_TYPE_BF16 => "bf16".into(),
        other => format!("ggml-type-{other}").into(),
    }
}

/// True for block-quantized storage (the lossy Q*/IQ* ggml types). F16/F32/
/// BF16/F64 are higher-precision float storage and integer arrays (I8..I64)
/// are non-neural data; neither is a block quant. An UNKNOWN type id is
/// treated as a block quant so the audit fails closed on types it cannot
/// account for rather than waving them through.
pub fn is_block_quant_type(ggml_type: u32) -> bool {
    !matches!(
        ggml_type,
        GGML_TYPE_F32
            | GGML_TYPE_F16
            | GGML_TYPE_F64
            | GGML_TYPE_BF16
            | GGML_TYPE_I8
            | GGML_TYPE_I16
            | GGML_TYPE_I32
            | GGML_TYPE_I64
    )
}

/// The wire ggml type ids that satisfy a given writer tensor type's *safety*
/// class, for the audit's purposes. Q8_0-class output accepts ANY 8-bit
/// block-quant rung (Q8_0/Q8_1/Q8_K): the audit also reads packs this repo's
/// own writer never produced (published packs, future writers), so it accepts
/// any rung at the writer's precision class rather than demanding the exact
/// ggml type `classify_quant_tensor` happens to emit today. Each K-quant rung
/// above Q8 has no such sibling -- it maps onto exactly its own wire id.
fn wire_types_equivalent_to(tensor_type: GgufWriteTensorType) -> &'static [u32] {
    match tensor_type {
        GgufWriteTensorType::Q8_0 => &[GGML_TYPE_Q8_0, GGML_TYPE_Q8_1, GGML_TYPE_Q8_K],
        GgufWriteTensorType::Q3_K => &[GGML_TYPE_Q3_K],
        GgufWriteTensorType::Q4_K => &[GGML_TYPE_Q4_K],
        GgufWriteTensorType::Q5_K => &[GGML_TYPE_Q5_K],
        GgufWriteTensorType::Q6_K => &[GGML_TYPE_Q6_K],
        // `classify_quant_tensor` never returns these for a block-quantized
        // tensor (F32/F16 are its "keep the fp16-mode representation" `None`
        // case, not a `Some` block-quant type); listed for exhaustiveness.
        GgufWriteTensorType::F32 | GgufWriteTensorType::F16 => &[],
    }
}

/// True when the type satisfies the audio-encoder Q8_0 floor. Anything below
/// -- the Q2..Q6 and IQ* rungs -- is the behavioral cliff the floor exists to
/// prevent.
pub fn meets_encoder_q8_floor(ggml_type: u32) -> bool {
    wire_types_equivalent_to(GgufWriteTensorType::Q8_0).contains(&ggml_type)
}

/// Representative `ne0` values covering both alignment classes
/// `classify_quant_tensor` branches on: 32-aligned-but-not-256 (falls back to
/// the Q8_0 alignment rung) and 256-aligned (unlocks a tier's own K-quant
/// rung, for `Decoder`).
const REPRESENTATIVE_NE0_VALUES: [u64; 2] = [32, 256];
const QUANT_COMPONENTS: [QuantComponent; 2] = [QuantComponent::Encoder, QuantComponent::Decoder];

/// The block-quant rungs a declared pack tier may contain, DERIVED from
/// [`crate::models::pack_quant::classify_quant_tensor`] -- the same policy
/// function every importer's build-time quantization calls -- rather than a
/// hand-copied per-tier table. `classify_quant_tensor` returns `None` for
/// every `(ne0, component)` sample under `PackQuant::Fp16`, so the `Fp16`
/// ceiling falls out of the same loop as an empty set (no block quants
/// allowed) with no special-casing needed here.
fn declared_tier_allows(declared: PackQuant, ggml_type: u32) -> bool {
    REPRESENTATIVE_NE0_VALUES.iter().any(|&ne0| {
        QUANT_COMPONENTS.iter().any(|&component| {
            classify_quant_tensor(ne0, declared, component).is_some_and(|tensor_type| {
                wire_types_equivalent_to(tensor_type).contains(&ggml_type)
            })
        })
    })
}

// --- per-architecture audio-encoder tensor rules ---------------------------

/// Which tensors of a pack are audio-encoder weights for the Q8_0 floor,
/// keyed on the RUNTIME tensor names written into the pack.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AudioEncoderTensors {
    /// Encoder tensors are the rank-2 weights under these name prefixes.
    /// Sourced from each family's own `AUDIO_ENCODER_TENSOR_NAME_PREFIXES`
    /// constant (declared beside that family's `QuantComponent`
    /// classification in its `package_import` module) -- never a literal
    /// re-typed here -- so the audit can never silently drift from what the
    /// importer actually classifies as `QuantComponent::Encoder`.
    NamePrefixes(&'static [&'static str]),
    /// The whole pack is the acoustic path (e.g. redimnet2's speaker
    /// embedder, which has no separate decoder): every block-quant tensor
    /// carries the floor.
    EntirePack,
    /// No audio encoder in the ASR-floor sense (translation / punctuation /
    /// segmentation packs): the floor is vacuous.
    NoAudioEncoder,
}

// The architecture ids are the stable wire values written into each pack's
// `openasr.model.architecture` / `general.architecture` metadata (see
// crate::arch and each family's package_import). Every arm keys on the
// family's own constant -- no string literals -- so a renamed architecture
// id moves the audit table with it at compile time.
use AudioEncoderTensors as Rule;

/// The encoder rule for a pack architecture, or `None` when the architecture
/// is unknown to this table. `None` is NOT "no encoder" -- it is "cannot
/// verify", and the audit fails closed when such a pack contains any
/// block-quant tensor at all.
///
/// Coverage over every shipped architecture (ASR decode families from
/// `crate::arch::OpenAsrArchitectureRegistry` plus the auxiliary families
/// from `crate::models::aux_pack_registry`) is enforced by
/// `every_shipped_architecture_has_an_encoder_rule` below, derived from those
/// two registries rather than a hand-counted literal list.
pub fn audio_encoder_tensors_for_architecture(architecture: &str) -> Option<AudioEncoderTensors> {
    let rule = match architecture {
        crate::arch::WHISPER_GGML_ARCHITECTURE_ID => {
            Rule::NamePrefixes(crate::models::whisper::AUDIO_ENCODER_TENSOR_NAME_PREFIXES)
        }
        crate::arch::QWEN3_ASR_GGML_ARCHITECTURE_ID => {
            Rule::NamePrefixes(crate::models::qwen::AUDIO_ENCODER_TENSOR_NAME_PREFIXES)
        }
        crate::models::qwen::QWEN3_FORCED_ALIGNER_GGML_ARCHITECTURE_ID => {
            Rule::NamePrefixes(crate::models::qwen::AUDIO_ENCODER_TENSOR_NAME_PREFIXES)
        }
        crate::arch::DOLPHIN_GGML_ARCHITECTURE_ID => {
            Rule::NamePrefixes(crate::models::dolphin::package_import::AUDIO_ENCODER_TENSOR_NAME_PREFIXES)
        }
        crate::arch::SENSEVOICE_GGML_ARCHITECTURE_ID => Rule::NamePrefixes(
            crate::models::sensevoice::package_import::AUDIO_ENCODER_TENSOR_NAME_PREFIXES,
        ),
        crate::arch::COHERE_TRANSCRIBE_GGML_ARCHITECTURE_ID => {
            Rule::NamePrefixes(crate::models::cohere::AUDIO_ENCODER_TENSOR_NAME_PREFIXES)
        }
        crate::arch::MOONSHINE_GGML_ARCHITECTURE_ID => {
            Rule::NamePrefixes(crate::models::moonshine::AUDIO_ENCODER_TENSOR_NAME_PREFIXES)
        }
        crate::arch::PARAKEET_CTC_GGML_ARCHITECTURE_ID => Rule::NamePrefixes(
            crate::models::parakeet_ctc::package_import::AUDIO_ENCODER_TENSOR_NAME_PREFIXES,
        ),
        crate::arch::PARAKEET_TDT_GGML_ARCHITECTURE_ID => Rule::NamePrefixes(
            crate::models::parakeet_tdt::package_import::AUDIO_ENCODER_TENSOR_NAME_PREFIXES,
        ),
        crate::arch::WAV2VEC2_CTC_GGML_ARCHITECTURE_ID => Rule::NamePrefixes(
            crate::models::wav2vec2_ctc::package_import::AUDIO_ENCODER_TENSOR_NAME_PREFIXES,
        ),
        crate::arch::XASR_ZIPFORMER_GGML_ARCHITECTURE_ID => Rule::NamePrefixes(
            crate::models::xasr_zipformer::package_import::AUDIO_ENCODER_TENSOR_NAME_PREFIXES,
        ),
        crate::arch::FIRERED_AED_GGML_ARCHITECTURE_ID => Rule::NamePrefixes(
            crate::models::firered_aed::package_import::AUDIO_ENCODER_TENSOR_NAME_PREFIXES,
        ),
        // firered-llm combines TWO independently quantized halves into one
        // pack: the conformer encoder + encoder->LLM adapter (always floored
        // to Q8_0) and the separately-imported Qwen2 LLM backbone (`llm.*`,
        // which legitimately keeps the full requested rung, e.g. Q4_K). An
        // earlier `EntirePack` rule here incorrectly floored the `llm.*`
        // tensors too; `firered_llm::AUDIO_ENCODER_TENSOR_NAME_PREFIXES`
        // covers only the encoder/adapter half, reconciled against the real
        // importer in
        // `firered_llm::package_import::audio_encoder_tensor_name_prefixes_match_the_encoder_adapter_half_only`.
        crate::arch::FIRERED_LLM_GGML_ARCHITECTURE_ID => Rule::NamePrefixes(
            crate::models::firered_llm::package_import::AUDIO_ENCODER_TENSOR_NAME_PREFIXES,
        ),
        // funasr-nano: SAN-M encoder (`enc.`/`tp.`) + transformer adaptor
        // (`adaptor.`) are the audio-encoder half; the `blk.`/`token_embd`/
        // `output*` tensors are the Qwen3 decoder (LLM), not an audio encoder.
        crate::arch::FUNASR_NANO_GGML_ARCHITECTURE_ID => Rule::NamePrefixes(
            crate::models::funasr_nano::tensor_names::AUDIO_ENCODER_TENSOR_NAME_PREFIXES,
        ),
        crate::arch::MOSS_TD_GGML_ARCHITECTURE_ID => Rule::NamePrefixes(
            crate::models::moss_transcribe_diarize::package_import::AUDIO_ENCODER_TENSOR_NAME_PREFIXES,
        ),
        // granite-speech's acoustic path is the Conformer CTC encoder
        // (`encoder.*`) plus the Q-Former window projector (`projector.*`);
        // both must stay at/above the Q8_0 floor for encode fidelity, while
        // the Granite decoder-only LLM (`language_model.*`) legitimately keeps
        // the full requested rung. The importer ships fp16 today (see
        // `granite_speech::package_import`), so no real pack carries a
        // block-quant tensor here yet -- this rule floors the acoustic half of
        // any future quantized pack rather than shipping a decode cliff.
        crate::arch::GRANITE_SPEECH_GGML_ARCHITECTURE_ID => {
            Rule::NamePrefixes(&["encoder.", "projector."])
        }
        // MiMo's external converter quantizes ONLY the Qwen2 backbone; the
        // whole audio side (tokenizer encoder, input-local layers, speech
        // embeddings) stays f16/f32 for encode fidelity. Unlike the families
        // above, MiMo has no in-repo Rust importer to source a shared
        // constant from (see `models::mimo_asr`'s module docs), so this list
        // is declared here directly: a future converter that block-quantizes
        // the acoustic path fails the floor instead of shipping a decode
        // cliff.
        crate::arch::MIMO_ASR_GGML_ARCHITECTURE_ID => Rule::NamePrefixes(&[
            "audiotok.",
            "inlocal.",
            "speech_embd.",
            "speech_group_proj.",
        ]),
        // ReDimNet2-B6 (speaker embedder): ships F32-only today (see
        // `models::diarize_pack_import`, which has no `PackQuant` tier at
        // all), so no real pack should ever carry a block-quant tensor here.
        // `EntirePack` gives the right answer if it ever gains a quant tier:
        // the whole checkpoint is one acoustic-similarity backbone with no
        // separate decoder, so any block-quant tensor it does produce
        // correctly carries the Q8_0 floor.
        crate::models::aux_pack_registry::REDIMNET2_GGML_ARCHITECTURE_ID => Rule::EntirePack,
        // Translation / punctuation / segmentation packs have no acoustic
        // encoder in the ASR-floor sense.
        crate::models::hymt2::config::HUNYUAN_DENSE_ARCHITECTURE_VALUE => Rule::NoAudioEncoder,
        crate::models::pyannote::PYANNOTE_GGML_ARCHITECTURE_ID => Rule::NoAudioEncoder,
        crate::models::firered_punc::config::FIRERED_PUNC_ARCHITECTURE_VALUE => {
            Rule::NoAudioEncoder
        }
        _ => return None,
    };
    Some(rule)
}

// --- audit ------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuantFloorViolationKind {
    /// An audio-encoder tensor is block-quantized below Q8_0: the behavioral
    /// cliff the shared floor exists to prevent.
    EncoderBelowQ8Floor,
    /// A tensor's block quant exceeds the rung the pack's declared tier can
    /// produce (the pack is mislabeled, or was built outside its tier).
    ExceedsDeclaredTier,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QuantFloorViolation {
    pub tensor: String,
    pub dims: Vec<u64>,
    pub ggml_type: u32,
    pub kind: QuantFloorViolationKind,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct QuantFloorReport {
    /// The pack's `openasr.model.architecture` (falling back to
    /// `general.architecture`), when present.
    pub architecture: Option<String>,
    /// Build provenance (`openasr.build.commit`): the open-core commit whose
    /// quantization policy wrote the pack, when the builder recorded it.
    pub build_commit: Option<String>,
    pub tensor_count: u64,
    /// Block-quantized tensors in the pack.
    pub block_quant_tensors: usize,
    /// Block-quantized tensors classified as audio encoder.
    pub encoder_block_quant_tensors: usize,
    pub violations: Vec<QuantFloorViolation>,
}

impl QuantFloorReport {
    pub fn passed(&self) -> bool {
        self.violations.is_empty()
    }
}

#[derive(Debug, Error)]
pub enum QuantFloorAuditError {
    #[error(transparent)]
    Header(#[from] GgufHeaderError),
    #[error("could not read pack '{path}': {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("could not fetch pack header from '{url}': {reason}")]
    RemoteFetch { url: String, reason: String },
    #[error(
        "cannot verify the audio-encoder quant floor: architecture '{architecture}' has no encoder tensor rule but the pack contains {block_quant_tensors} block-quantized tensor(s); add a rule to pack_quant_audit before shipping"
    )]
    UnrecognizedArchitecture {
        architecture: String,
        block_quant_tensors: usize,
    },
}

fn pack_architecture(view: &GgufHeaderView) -> Option<String> {
    view.metadata_string(crate::models::oasr_metadata::OASR_METADATA_KEY_MODEL_ARCHITECTURE)
        .or_else(|| view.metadata_string("general.architecture"))
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

fn is_encoder_tensor(rule: AudioEncoderTensors, name: &str) -> bool {
    match rule {
        Rule::EntirePack => true,
        Rule::NoAudioEncoder => false,
        Rule::NamePrefixes(prefixes) => prefixes.iter().any(|prefix| name.starts_with(prefix)),
    }
}

/// Replay the current quantization policy against a parsed pack header.
///
/// Two independent invariants, both fail-closed:
/// 1. FLOOR -- no audio-encoder tensor may be a sub-Q8 block quant, for ANY
///    tier (the floor applies at every rung; see `classify_quant_tensor`).
/// 2. CEILING -- when `declared` names the tier the pack claims, no tensor
///    may carry a block quant that tier cannot produce.
///
/// A pack whose architecture has no encoder rule passes only if it contains
/// no block-quant tensors at all; otherwise verification is impossible and
/// the audit errors out rather than silently waving it through.
pub fn audit_quant_floor(
    view: &GgufHeaderView,
    declared: Option<PackQuant>,
) -> Result<QuantFloorReport, QuantFloorAuditError> {
    let architecture = pack_architecture(view);
    let rule = architecture
        .as_deref()
        .and_then(audio_encoder_tensors_for_architecture);

    let mut report = QuantFloorReport {
        build_commit: view
            .metadata_string(crate::models::oasr_metadata::OASR_METADATA_KEY_BUILD_COMMIT)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_string),
        architecture,
        tensor_count: view.tensor_count,
        ..QuantFloorReport::default()
    };

    for tensor in &view.tensors {
        if !is_block_quant_type(tensor.ggml_type) {
            continue;
        }
        report.block_quant_tensors += 1;

        let encoder = match rule {
            Some(rule) => is_encoder_tensor(rule, &tensor.name),
            None => false,
        };
        if encoder {
            report.encoder_block_quant_tensors += 1;
        }

        if encoder && !meets_encoder_q8_floor(tensor.ggml_type) {
            report.violations.push(QuantFloorViolation {
                tensor: tensor.name.clone(),
                dims: tensor.dims.clone(),
                ggml_type: tensor.ggml_type,
                kind: QuantFloorViolationKind::EncoderBelowQ8Floor,
            });
        }
        if let Some(declared) = declared
            && !declared_tier_allows(declared, tensor.ggml_type)
        {
            report.violations.push(QuantFloorViolation {
                tensor: tensor.name.clone(),
                dims: tensor.dims.clone(),
                ggml_type: tensor.ggml_type,
                kind: QuantFloorViolationKind::ExceedsDeclaredTier,
            });
        }
    }

    if rule.is_none() && report.block_quant_tensors > 0 {
        return Err(QuantFloorAuditError::UnrecognizedArchitecture {
            architecture: report
                .architecture
                .clone()
                .unwrap_or_else(|| "<missing>".to_string()),
            block_quant_tensors: report.block_quant_tensors,
        });
    }

    Ok(report)
}

/// Parse a complete GGUF header out of a growable byte source: parse the
/// window, and while it reports Truncated, ask the source for more bytes.
/// `read_prefix(len)` must return the file/URL's first `len` bytes (or all
/// remaining bytes when the object is shorter).
fn parse_header_with_growing_window(
    mut read_prefix: impl FnMut(usize) -> Result<Vec<u8>, QuantFloorAuditError>,
    initial_window: usize,
    max_window: usize,
) -> Result<GgufHeaderView, QuantFloorAuditError> {
    let mut window = initial_window.max(64 * 1024);
    loop {
        let bytes = read_prefix(window)?;
        match parse_gguf_header(&bytes) {
            Ok(view) => return Ok(view),
            Err(GgufHeaderError::Truncated { .. }) if window < max_window => {
                // The header is larger than the window; quadruple it, capped.
                window = (window.saturating_mul(4)).min(max_window);
                if bytes.len() < window && bytes.len() < max_window {
                    // The source is shorter than the requested window yet the
                    // header did not complete: the object itself is malformed
                    // or shorter than its own header claims. Surface the
                    // truncation verbatim on the next (final) parse.
                    window = max_window;
                }
            }
            Err(error) => return Err(error.into()),
        }
    }
}

const LOCAL_INITIAL_WINDOW: usize = 8 * 1024 * 1024;
const LOCAL_MAX_WINDOW: usize = 128 * 1024 * 1024;

/// Audit a pack on local disk, reading only its header prefix (never the
/// tensor data), growing the read window if an unusually large header needs
/// it.
pub fn audit_local_pack_quant_floor(
    path: impl AsRef<Path>,
    declared: Option<PackQuant>,
) -> Result<QuantFloorReport, QuantFloorAuditError> {
    let path = path.as_ref().to_path_buf();
    let view = parse_header_with_growing_window(
        |len| {
            let mut file =
                std::fs::File::open(&path).map_err(|source| QuantFloorAuditError::Io {
                    path: path.clone(),
                    source,
                })?;
            let mut buffer = vec![0u8; len];
            let mut filled = 0usize;
            while filled < buffer.len() {
                match file.read(&mut buffer[filled..]) {
                    Ok(0) => break,
                    Ok(count) => filled += count,
                    Err(source) if source.kind() == std::io::ErrorKind::Interrupted => continue,
                    Err(source) => {
                        return Err(QuantFloorAuditError::Io {
                            path: path.clone(),
                            source,
                        });
                    }
                }
            }
            buffer.truncate(filled);
            Ok(buffer)
        },
        LOCAL_INITIAL_WINDOW,
        LOCAL_MAX_WINDOW,
    )?;
    audit_quant_floor(&view, declared)
}

/// Fetch the leading `len` bytes of a remote pack with an HTTP `Range`
/// request (read-only; the server sees one ranged GET, no download).
fn fetch_remote_prefix(url: &str, len: usize) -> Result<Vec<u8>, QuantFloorAuditError> {
    use std::time::Duration;

    let client = crate::http::blocking_client(Duration::from_secs(30), Duration::from_secs(600))
        .map_err(|error| QuantFloorAuditError::RemoteFetch {
            url: url.to_string(),
            reason: crate::http::error_message(&error),
        })?;
    let response = client
        .get(url)
        .header(reqwest::header::RANGE, format!("bytes=0-{}", len - 1))
        .send()
        .map_err(|error| QuantFloorAuditError::RemoteFetch {
            url: url.to_string(),
            reason: crate::http::error_message(&error),
        })?;
    let status = response.status();
    // Require a genuine partial response. A server that ignores the Range
    // header answers 200 OK and starts streaming the ENTIRE pack body --
    // buffering a multi-GB model to audit its first megabytes is precisely
    // what prefix fetching exists to avoid, so fail closed instead of
    // reading. Published-pack CDNs honor ranges; anything else is not an
    // auditable endpoint.
    if status != reqwest::StatusCode::PARTIAL_CONTENT {
        return Err(QuantFloorAuditError::RemoteFetch {
            url: url.to_string(),
            reason: format!(
                "expected HTTP 206 Partial Content for the Range request, got {status}; \
                 the endpoint must honor byte ranges for a prefix-only audit"
            ),
        });
    }
    // Defense in depth: never read past the requested prefix, even if a
    // server over-delivers on its Content-Range.
    use std::io::Read;
    let mut bytes = Vec::new();
    response
        .take(len as u64)
        .read_to_end(&mut bytes)
        .map_err(|error| QuantFloorAuditError::RemoteFetch {
            url: url.to_string(),
            reason: error.to_string(),
        })?;
    Ok(bytes)
}

const REMOTE_INITIAL_WINDOW: usize = 8 * 1024 * 1024;
const REMOTE_MAX_WINDOW: usize = 64 * 1024 * 1024;

/// Audit a published pack over HTTP(S), fetching only the header prefix via
/// `Range` requests -- no source weights, no full download. Designed as the
/// whole-catalog periodic health check: point it at each catalog entry's
/// resolve URL.
pub fn audit_remote_pack_quant_floor(
    url: &str,
    declared: Option<PackQuant>,
) -> Result<QuantFloorReport, QuantFloorAuditError> {
    let view = parse_header_with_growing_window(
        |len| fetch_remote_prefix(url, len),
        REMOTE_INITIAL_WINDOW,
        REMOTE_MAX_WINDOW,
    )?;
    audit_quant_floor(&view, declared)
}

#[cfg(test)]
mod tests {
    use super::{
        AudioEncoderTensors, GGML_TYPE_F16, GGML_TYPE_F32, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K,
        GGML_TYPE_Q6_K, GGML_TYPE_Q8_0, GGML_TYPE_Q8_1, GGML_TYPE_Q8_K, QuantFloorAuditError,
        QuantFloorViolationKind, audio_encoder_tensors_for_architecture, audit_quant_floor,
        declared_tier_allows, ggml_type_name, is_block_quant_type, meets_encoder_q8_floor,
    };
    use crate::ggml_runtime::gguf_header::{GgufHeaderTensor, GgufHeaderView};
    use crate::models::pack_quant::PackQuant;
    use std::collections::BTreeMap;

    fn view_with(architecture: Option<&str>, tensors: Vec<GgufHeaderTensor>) -> GgufHeaderView {
        let mut string_metadata = BTreeMap::new();
        if let Some(architecture) = architecture {
            string_metadata.insert(
                "openasr.model.architecture".to_string(),
                architecture.to_string(),
            );
        }
        let tensor_count = tensors.len() as u64;
        GgufHeaderView {
            version: 3,
            tensor_count,
            string_metadata,
            tensors,
            header_len: 0,
        }
    }

    fn tensor(name: &str, ggml_type: u32) -> GgufHeaderTensor {
        GgufHeaderTensor {
            name: name.to_string(),
            dims: vec![256, 256],
            ggml_type,
            data_offset: 0,
        }
    }

    const QWEN_ARCH: &str = crate::arch::QWEN3_ASR_GGML_ARCHITECTURE_ID;

    #[test]
    fn type_classification_covers_the_ggml_abi() {
        assert!(!is_block_quant_type(GGML_TYPE_F32));
        assert!(!is_block_quant_type(GGML_TYPE_F16));
        assert!(is_block_quant_type(GGML_TYPE_Q8_0));
        assert!(is_block_quant_type(GGML_TYPE_Q4_K));
        // Unknown ids fail closed as block quants.
        assert!(is_block_quant_type(999));
        assert!(meets_encoder_q8_floor(GGML_TYPE_Q8_0));
        assert!(!meets_encoder_q8_floor(GGML_TYPE_Q6_K));
        assert!(!meets_encoder_q8_floor(GGML_TYPE_F16));
        assert_eq!(ggml_type_name(GGML_TYPE_Q4_K), "q4_k");
        assert_eq!(ggml_type_name(999), "ggml-type-999");
    }

    /// Pins `declared_tier_allows`'s derived rungs against the same
    /// expectations the old hand-written per-tier table encoded, so the
    /// switch to deriving from `classify_quant_tensor` is a proven
    /// behavior-preserving refactor, not just a structural one.
    #[test]
    fn declared_tier_allows_matches_the_pre_derivation_rung_table() {
        // fp16: no block quant is ever allowed.
        for ggml_type in [
            GGML_TYPE_Q8_0,
            GGML_TYPE_Q8_1,
            GGML_TYPE_Q8_K,
            GGML_TYPE_Q3_K,
            GGML_TYPE_Q4_K,
            GGML_TYPE_Q6_K,
        ] {
            assert!(!declared_tier_allows(PackQuant::Fp16, ggml_type));
        }
        // q8_0: only the 8-bit safety class.
        for ggml_type in [GGML_TYPE_Q8_0, GGML_TYPE_Q8_1, GGML_TYPE_Q8_K] {
            assert!(declared_tier_allows(PackQuant::Q8_0, ggml_type));
        }
        for ggml_type in [GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K] {
            assert!(!declared_tier_allows(PackQuant::Q8_0, ggml_type));
        }
        // q3_k: the 8-bit safety class plus its own rung.
        for ggml_type in [
            GGML_TYPE_Q8_0,
            GGML_TYPE_Q8_1,
            GGML_TYPE_Q8_K,
            GGML_TYPE_Q3_K,
        ] {
            assert!(declared_tier_allows(PackQuant::Q3_K, ggml_type));
        }
        assert!(!declared_tier_allows(PackQuant::Q3_K, GGML_TYPE_Q4_K));
        assert!(!declared_tier_allows(PackQuant::Q3_K, GGML_TYPE_Q6_K));
        // q4_k: the 8-bit safety class plus its own rung.
        for ggml_type in [
            GGML_TYPE_Q8_0,
            GGML_TYPE_Q8_1,
            GGML_TYPE_Q8_K,
            GGML_TYPE_Q4_K,
        ] {
            assert!(declared_tier_allows(PackQuant::Q4_K, ggml_type));
        }
        assert!(!declared_tier_allows(PackQuant::Q4_K, GGML_TYPE_Q3_K));
        assert!(!declared_tier_allows(PackQuant::Q4_K, GGML_TYPE_Q6_K));
    }

    /// Assert every architecture in `architectures` has a quant-floor
    /// encoder rule, panicking with the offending id otherwise. Factored out
    /// so `coverage_check_fails_closed_on_an_unruled_architecture` can prove
    /// this mechanism is not vacuous without needing to register a throwaway
    /// architecture in the real builtin/aux registries.
    fn assert_every_architecture_has_an_encoder_rule(
        architectures: impl Iterator<Item = &'static str>,
    ) {
        for architecture in architectures {
            assert!(
                audio_encoder_tensors_for_architecture(architecture).is_some(),
                "architecture '{architecture}' has no quant-floor encoder rule; add one to \
                 pack_quant_audit::audio_encoder_tensors_for_architecture before shipping"
            );
        }
    }

    /// Coverage over every shipped architecture, DERIVED from the two
    /// registries that between them enumerate every builtin family: ASR
    /// decode architectures (`OpenAsrArchitectureRegistry::with_builtins`)
    /// and auxiliary non-ASR families (`aux_pack_registry::aux_pack_architecture_ids`,
    /// diarization/translation/punctuation/forced-alignment). Unlike a
    /// hand-counted literal list, adding a new builtin family to either
    /// registry without adding its quant-floor rule here makes this test red
    /// -- see `coverage_check_fails_closed_on_an_unruled_architecture` for
    /// the proof that the mechanism actually fires.
    #[test]
    fn every_shipped_architecture_has_an_encoder_rule() {
        let asr_registry = crate::arch::OpenAsrArchitectureRegistry::with_builtins();
        assert_every_architecture_has_an_encoder_rule(
            asr_registry
                .descriptors()
                .iter()
                .map(|descriptor| descriptor.model_architecture),
        );
        assert_every_architecture_has_an_encoder_rule(
            crate::models::aux_pack_registry::aux_pack_architecture_ids(),
        );

        assert_eq!(
            audio_encoder_tensors_for_architecture(QWEN_ARCH),
            Some(AudioEncoderTensors::NamePrefixes(&["audio."]))
        );
        assert_eq!(audio_encoder_tensors_for_architecture("not-a-family"), None);
    }

    /// Proves `assert_every_architecture_has_an_encoder_rule` is not vacuous:
    /// an architecture id with no entry in `audio_encoder_tensors_for_architecture`
    /// makes the check panic. This is the permanent stand-in for "add a new
    /// builtin family without filling in its quant-floor rule" -- adding a
    /// throwaway descriptor to the real `BUILTIN_ARCHITECTURE_DESCRIPTORS` /
    /// `AUX_PACK_DESCRIPTORS` tables would also need to satisfy those
    /// registries' own unrelated exhaustiveness checks (capacity model,
    /// decode policy, tensor contract, ...), so this test isolates just the
    /// mechanism this change adds.
    #[test]
    #[should_panic(expected = "has no quant-floor encoder rule")]
    fn coverage_check_fails_closed_on_an_unruled_architecture() {
        assert_every_architecture_has_an_encoder_rule(
            ["definitely-not-a-real-architecture-id"].into_iter(),
        );
    }

    #[test]
    fn sub_q8_encoder_tensor_fails_the_floor() {
        let view = view_with(
            Some(QWEN_ARCH),
            vec![
                tensor("audio.blk.0.attn_q.weight", GGML_TYPE_Q4_K),
                tensor("blk.0.ffn_gate.weight", GGML_TYPE_Q4_K),
                tensor("token_embd.weight", GGML_TYPE_Q8_0),
            ],
        );
        let report = audit_quant_floor(&view, Some(PackQuant::Q4_K)).expect("auditable");
        assert_eq!(report.block_quant_tensors, 3);
        assert_eq!(report.encoder_block_quant_tensors, 1);
        assert_eq!(report.violations.len(), 1);
        assert_eq!(report.violations[0].tensor, "audio.blk.0.attn_q.weight");
        assert_eq!(
            report.violations[0].kind,
            QuantFloorViolationKind::EncoderBelowQ8Floor
        );
    }

    #[test]
    fn q8_floored_pack_passes_under_q4_k() {
        // The post-floor shape: encoder at Q8_0, decoder at the requested rung.
        let view = view_with(
            Some(QWEN_ARCH),
            vec![
                tensor("audio.blk.0.attn_q.weight", GGML_TYPE_Q8_0),
                tensor("audio.proj.weight", GGML_TYPE_Q8_0),
                tensor("blk.0.ffn_gate.weight", GGML_TYPE_Q4_K),
                tensor("output.weight", GGML_TYPE_F16),
            ],
        );
        let report = audit_quant_floor(&view, Some(PackQuant::Q4_K)).expect("auditable");
        assert!(report.passed(), "violations: {:?}", report.violations);
        assert_eq!(report.encoder_block_quant_tensors, 2);
    }

    #[test]
    fn declared_tier_ceiling_catches_mislabeled_packs() {
        // Claims q8_0 but carries a Q4_K decoder tensor: impossible under the
        // q8_0 tier (a ceiling violation, distinct from the encoder floor).
        let view = view_with(
            Some(QWEN_ARCH),
            vec![
                tensor("audio.blk.0.attn_q.weight", GGML_TYPE_Q8_0),
                tensor("blk.0.ffn_gate.weight", GGML_TYPE_Q4_K),
            ],
        );
        let report = audit_quant_floor(&view, Some(PackQuant::Q8_0)).expect("auditable");
        assert_eq!(report.violations.len(), 1);
        assert_eq!(report.violations[0].tensor, "blk.0.ffn_gate.weight");
        assert_eq!(
            report.violations[0].kind,
            QuantFloorViolationKind::ExceedsDeclaredTier
        );
    }

    #[test]
    fn unknown_architecture_with_block_quants_fails_closed() {
        let view = view_with(
            Some("brand-new-family"),
            vec![tensor("blk.0.ffn_gate.weight", GGML_TYPE_Q4_K)],
        );
        let error = audit_quant_floor(&view, Some(PackQuant::Q4_K))
            .expect_err("must fail closed without an encoder rule");
        assert!(matches!(
            error,
            QuantFloorAuditError::UnrecognizedArchitecture { .. }
        ));
    }

    #[test]
    fn unknown_architecture_without_block_quants_passes_vacuously() {
        let view = view_with(
            Some("brand-new-family"),
            vec![tensor("some.weight", GGML_TYPE_F16)],
        );
        let report = audit_quant_floor(&view, None).expect("auditable");
        assert!(report.passed());
        assert_eq!(report.block_quant_tensors, 0);
    }

    #[test]
    fn translation_pack_decoder_quants_are_not_encoder_floor_violations() {
        let view = view_with(
            Some(crate::models::hymt2::config::HUNYUAN_DENSE_ARCHITECTURE_VALUE),
            vec![tensor("blk.0.ffn_gate.weight", GGML_TYPE_Q4_K)],
        );
        let report = audit_quant_floor(&view, Some(PackQuant::Q4_K)).expect("auditable");
        assert!(report.passed());
        assert_eq!(report.encoder_block_quant_tensors, 0);
    }

    #[test]
    fn entire_pack_rule_floors_every_tensor() {
        // redimnet2 (speaker embedder): a single acoustic backbone with no
        // decoder side, so EVERY block-quant tensor is encoder regardless of
        // name.
        let view = view_with(
            Some(crate::models::aux_pack_registry::REDIMNET2_GGML_ARCHITECTURE_ID),
            vec![
                tensor("resnet.tdnn.0.weight", GGML_TYPE_Q8_0),
                tensor("resnet.pool.weight", GGML_TYPE_Q6_K),
            ],
        );
        let report = audit_quant_floor(&view, Some(PackQuant::Q4_K)).expect("auditable");
        // The pool tensor is still acoustic path: a sub-Q8 rung anywhere in
        // this pack violates BOTH the encoder floor and the declared q4_k
        // ceiling (Q6_K is outside its rung set), so it counts twice.
        assert_eq!(report.violations.len(), 2);
        assert!(
            report
                .violations
                .iter()
                .all(|violation| violation.tensor == "resnet.pool.weight")
        );
        let kinds: Vec<_> = report.violations.iter().map(|v| v.kind).collect();
        assert!(kinds.contains(&QuantFloorViolationKind::EncoderBelowQ8Floor));
        assert!(kinds.contains(&QuantFloorViolationKind::ExceedsDeclaredTier));
    }

    /// The bug this change fixes: firered-llm's `llm.*` Qwen2 decoder tensors
    /// keep the full requested rung and must NOT be floored, even though an
    /// earlier hand-written `EntirePack` rule treated the whole pack
    /// (including `llm.*`) as encoder. This is the real shape of the
    /// published `firered2-llm-q4_k.oasr` pack: `enc.*`/`adapter.*` at Q8_0,
    /// `llm.*` at Q4_K.
    #[test]
    fn firered_llm_decoder_tensors_keep_the_declared_tier_and_are_not_floored() {
        let view = view_with(
            Some(crate::arch::FIRERED_LLM_GGML_ARCHITECTURE_ID),
            vec![
                tensor("enc.blk.0.attn.q.weight", GGML_TYPE_Q8_0),
                tensor("adapter.linear1.weight", GGML_TYPE_Q8_0),
                tensor("llm.blk.0.attn_q.weight", GGML_TYPE_Q4_K),
                tensor("llm.lm_head.weight", GGML_TYPE_Q4_K),
            ],
        );
        let report = audit_quant_floor(&view, Some(PackQuant::Q4_K)).expect("auditable");
        assert!(report.passed(), "violations: {:?}", report.violations);
        assert_eq!(report.block_quant_tensors, 4);
        // Only the two encoder/adapter tensors are floor-relevant; the two
        // `llm.*` Q4_K tensors must not be counted as encoder.
        assert_eq!(report.encoder_block_quant_tensors, 2);
    }

    /// A regression pin for the specific bug: an `EntirePack`-style rule
    /// would flag this `llm.*` Q4_K tensor as `EncoderBelowQ8Floor`.
    #[test]
    fn firered_llm_llm_prefix_is_never_classified_as_encoder() {
        let view = view_with(
            Some(crate::arch::FIRERED_LLM_GGML_ARCHITECTURE_ID),
            vec![tensor("llm.blk.3.ffn_gate.weight", GGML_TYPE_Q4_K)],
        );
        let report = audit_quant_floor(&view, Some(PackQuant::Q4_K)).expect("auditable");
        assert!(report.passed(), "violations: {:?}", report.violations);
        assert_eq!(report.encoder_block_quant_tensors, 0);
    }

    #[test]
    fn whisper_encoder_prefix_is_model_encoder() {
        let view = view_with(
            Some(crate::arch::WHISPER_GGML_ARCHITECTURE_ID),
            vec![
                tensor(
                    "model.encoder.layers.0.self_attn.q_proj.weight",
                    GGML_TYPE_Q4_K,
                ),
                tensor(
                    "model.decoder.layers.0.self_attn.q_proj.weight",
                    GGML_TYPE_Q4_K,
                ),
            ],
        );
        let report = audit_quant_floor(&view, Some(PackQuant::Q4_K)).expect("auditable");
        assert_eq!(report.violations.len(), 1);
        assert!(report.violations[0].tensor.starts_with("model.encoder."));
    }

    /// A one-shot raw HTTP server: reply to the first request with the
    /// prepared bytes, then close. Lets the prefix-fetch contract tests pin
    /// exact status-line behavior without a framework.
    fn spawn_raw_http_server(raw_response: Vec<u8>) -> String {
        use std::io::{Read, Write};
        use std::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").expect("bind test server");
        let url = format!("http://{}/pack.oasr", listener.local_addr().unwrap());
        std::thread::spawn(move || {
            if let Ok((mut stream, _)) = listener.accept() {
                let mut request = vec![0u8; 4096];
                let _ = stream.read(&mut request); // request headers; content irrelevant
                let _ = stream.write_all(&raw_response);
            }
        });
        url
    }

    fn raw_response(status_line: &str, body: &[u8]) -> Vec<u8> {
        let mut raw = format!(
            "HTTP/1.1 {status_line}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            body.len()
        )
        .into_bytes();
        raw.extend_from_slice(body);
        raw
    }

    #[test]
    fn remote_prefix_fetch_rejects_a_full_body_response() {
        // A server that ignores the Range header answers 200 and streams the
        // whole (multi-GB) pack. The fetch must fail closed rather than
        // buffer it.
        let url = spawn_raw_http_server(raw_response(
            "200 OK",
            b"the entire pack body would follow this header",
        ));
        let error = super::fetch_remote_prefix(&url, 8).expect_err("must fail closed on 200");
        match error {
            QuantFloorAuditError::RemoteFetch { reason, .. } => {
                assert!(reason.contains("206"), "reason was: {reason}");
            }
            other => panic!("expected RemoteFetch, got {other:?}"),
        }
    }

    #[test]
    fn remote_prefix_fetch_caps_the_read_at_the_requested_prefix() {
        // A 206 that over-delivers on Content-Range still yields exactly the
        // requested prefix.
        let url = spawn_raw_http_server(raw_response("206 Partial Content", b"0123456789abcdef"));
        let bytes = super::fetch_remote_prefix(&url, 8).expect("206 prefix fetch");
        assert_eq!(bytes, b"01234567");
    }
}
