//! Single dispatch point for auxiliary (non-ASR) runtime pack contracts.
//!
//! ASR families are looked up through one data-driven table --
//! [`crate::arch::OpenAsrArchitectureRegistry`] -- keyed by `general.architecture`
//! and cross-checked (`openasr.model.family` / audio-frontend / decode-policy /
//! tokenizer) before an adapter is selected. Auxiliary packs (speaker
//! embedder, speaker segmenter, translation, punctuation, forced alignment)
//! are not ASR transcription architectures -- they have no audio frontend or
//! decode policy in that sense -- so forcing them into
//! `OpenAsrArchitectureDescriptor` would model a shape they don't have (see
//! `models::pyannote` module docs, which already say so explicitly). They still
//! deserve **one** table instead of an ad hoc chain of `if let Some(...)` calls
//! in `api::backend::native`, so this module is that table: one
//! `general.architecture` value per aux family, matched by a single lookup,
//! fail-closed (`None` when no aux entry matches, so the caller falls through
//! to ASR adapter selection, which then fails closed on its own if the pack
//! matches nothing at all).
//!
//! [`aux_pack_architecture_ids_are_unique_and_disjoint_from_asr`] is the safety
//! net a hand-rolled chain never had: it fails the test suite if a future aux
//! family ever reuses a `general.architecture` value already claimed by an ASR
//! descriptor (which would otherwise silently shadow one or the other,
//! depending on chain order, instead of raising `Ambiguous`).

use std::path::Path;

use crate::GgufMetadata;
use crate::arch::GENERAL_ARCHITECTURE_KEY;

/// Which pull-time error prefix a matched aux family reports, preserving the
/// exact wording `api::backend::native`'s tests assert on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AuxPackKind {
    /// Speaker embedder (ReDimNet2-B6) / speaker segmenter (pyannote) diarization
    /// support packs.
    Diarization,
    /// Translation runtime packs (Hy-MT2).
    Translation,
    /// Punctuation-restoration packs (FireRedPunc).
    Punctuation,
    /// Forced-alignment word-timestamp refiner packs (Qwen3-ForcedAligner).
    ForcedAlignment,
}

impl AuxPackKind {
    /// The `"<label> failed: <error>"` prefix `validate_native_runtime_model_pack_contract`
    /// reports for this kind (unchanged from the pre-consolidation call sites).
    pub(crate) fn validation_failure_label(self) -> &'static str {
        match self {
            AuxPackKind::Diarization => "diarization pack validation failed",
            AuxPackKind::Translation => "translation pack validation failed",
            AuxPackKind::Punctuation => "punctuation pack validation failed",
            AuxPackKind::ForcedAlignment => "forced-alignment pack validation failed",
        }
    }
}

struct AuxPackDescriptor {
    /// `general.architecture` value that identifies this aux family's packs.
    architecture_id: &'static str,
    kind: AuxPackKind,
    /// Cheap pull-time contract probe: constructs/parses just enough of the
    /// pack to prove the runtime loader can build from it, without
    /// materializing full weights for execution.
    validate: fn(&Path, &GgufMetadata) -> Result<(), String>,
}

fn validate_pyannote(path: &Path, _metadata: &GgufMetadata) -> Result<(), String> {
    crate::diarize::segment::PyannoteSegmenter::from_oasr(path)
        .map(|_| ())
        .map_err(|error| error.to_string())
}

fn validate_hymt2(path: &Path, _metadata: &GgufMetadata) -> Result<(), String> {
    crate::models::hymt2::Hymt2Runtime::probe_path(path)
        .map(|_| ())
        .map_err(|error| error.to_string())
}

fn validate_redimnet2(path: &Path, _metadata: &GgufMetadata) -> Result<(), String> {
    crate::diarize::embed::RedimNet2Embedder::from_oasr(path)
        .map(|_| ())
        .map_err(|error| error.to_string())
}

fn validate_firered_punc(_path: &Path, metadata: &GgufMetadata) -> Result<(), String> {
    crate::models::firered_punc::runtime_contract::parse_and_validate_firered_punc_metadata(
        metadata,
    )
    .map(|_| ())
    .map_err(|error| error.to_string())
}

fn validate_forced_aligner(_path: &Path, metadata: &GgufMetadata) -> Result<(), String> {
    crate::models::qwen::validate_forced_aligner_runtime_pack_contract(metadata)
        .map_err(|error| error.to_string())
}

/// `general.architecture` value ReDimNet2-B6 speaker-embedder packs carry.
/// No dedicated `models::redimnet2` module owns this constant (the model
/// forward pass lives in `crate::diarize::embed::redimnet`, and packaging is
/// the family-agnostic `models::diarize_pack_import`), so this aux registry
/// -- the only production reader of the string -- is its home. Referenced by
/// `models::pack_quant_audit` too, so both stay in sync.
pub(crate) const REDIMNET2_GGML_ARCHITECTURE_ID: &str = "redimnet2";

const AUX_PACK_DESCRIPTORS: &[AuxPackDescriptor] = &[
    AuxPackDescriptor {
        architecture_id: REDIMNET2_GGML_ARCHITECTURE_ID,
        kind: AuxPackKind::Diarization,
        validate: validate_redimnet2,
    },
    AuxPackDescriptor {
        architecture_id: crate::models::pyannote::PYANNOTE_GGML_ARCHITECTURE_ID,
        kind: AuxPackKind::Diarization,
        validate: validate_pyannote,
    },
    AuxPackDescriptor {
        architecture_id: crate::models::hymt2::config::HUNYUAN_DENSE_ARCHITECTURE_VALUE,
        kind: AuxPackKind::Translation,
        validate: validate_hymt2,
    },
    AuxPackDescriptor {
        architecture_id: crate::models::firered_punc::config::FIRERED_PUNC_ARCHITECTURE_VALUE,
        kind: AuxPackKind::Punctuation,
        validate: validate_firered_punc,
    },
    AuxPackDescriptor {
        architecture_id: crate::models::qwen::QWEN3_FORCED_ALIGNER_GGML_ARCHITECTURE_ID,
        kind: AuxPackKind::ForcedAlignment,
        validate: validate_forced_aligner,
    },
];

/// Every aux family's `general.architecture` id. Lets a caller that needs the
/// full non-ASR family list (e.g. `models::pack_quant_audit`'s quant-floor
/// coverage test) enumerate it without depending on `AuxPackKind` or
/// `validate_aux_runtime_pack_contract`'s metadata-driven dispatch.
/// Test-only today (no non-test caller needs the full list yet).
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn aux_pack_architecture_ids() -> impl Iterator<Item = &'static str> {
    AUX_PACK_DESCRIPTORS
        .iter()
        .map(|descriptor| descriptor.architecture_id)
}

/// Pull-time contract dispatch for auxiliary (non-ASR) runtime packs.
///
/// Returns `None` when `metadata` does not declare one of the known aux
/// `general.architecture` values, so the caller (`validate_native_runtime_model_pack_contract`)
/// falls through to ASR family-adapter selection -- which then fails closed on
/// its own for a pack that matches neither table. Returns `Some((kind,
/// result))` when an aux family claims the pack, `result` being that family's
/// cheap runtime-loader probe (no weight materialization).
pub(crate) fn validate_aux_runtime_pack_contract(
    path: &Path,
    metadata: &GgufMetadata,
) -> Option<(AuxPackKind, Result<(), String>)> {
    let architecture = metadata.get_string(GENERAL_ARCHITECTURE_KEY)?.trim();
    let descriptor = AUX_PACK_DESCRIPTORS
        .iter()
        .find(|descriptor| descriptor.architecture_id == architecture)?;
    Some((descriptor.kind, (descriptor.validate)(path, metadata)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::OpenAsrArchitectureRegistry;

    /// Fail-closed safety net the previous hand-rolled `if let Some(...)` chain
    /// in `api::backend::native` never had: every aux `general.architecture`
    /// value must be unique among aux families AND disjoint from every ASR
    /// `OpenAsrArchitectureDescriptor::model_architecture`. A collision would
    /// otherwise be resolved by chain/table iteration order instead of an
    /// explicit `Ambiguous` error -- exactly the silent-shadowing failure mode
    /// `GgmlFamilyRegistry::select_from_fields` refuses to allow within the ASR
    /// table.
    #[test]
    fn aux_pack_architecture_ids_are_unique_and_disjoint_from_asr() {
        let mut seen: Vec<&'static str> = Vec::new();
        for descriptor in AUX_PACK_DESCRIPTORS {
            assert!(
                !seen.contains(&descriptor.architecture_id),
                "duplicate aux architecture id: {}",
                descriptor.architecture_id
            );
            seen.push(descriptor.architecture_id);
        }

        let asr_registry = OpenAsrArchitectureRegistry::with_builtins();
        for descriptor in AUX_PACK_DESCRIPTORS {
            assert!(
                asr_registry
                    .find_by_model_architecture(descriptor.architecture_id)
                    .is_none(),
                "aux architecture id '{}' collides with a registered ASR architecture",
                descriptor.architecture_id
            );
        }
    }

    #[test]
    fn dispatch_returns_none_for_unknown_architecture() {
        let mut values = std::collections::BTreeMap::new();
        values.insert(
            GENERAL_ARCHITECTURE_KEY.to_string(),
            crate::ggml_runtime::GgufMetadataValue::String("totally-unknown-arch".to_string()),
        );
        let metadata = GgufMetadata::from_values_for_test(values);
        assert!(validate_aux_runtime_pack_contract(Path::new("/nonexistent"), &metadata).is_none());
    }

    /// A complete, minimal set of `qwen3_forced_aligner.*` + tokenizer keys --
    /// mirrors exactly what a real published pack carries (verified against
    /// the rebuilt `qwen3-forced-aligner-0.6b-q4_k.oasr` pack's GGUF header:
    /// no `openasr.audio.frontend` / `openasr.decode.policy`, only these).
    fn valid_forced_aligner_metadata() -> GgufMetadata {
        use crate::ggml_runtime::GgufMetadataValue;
        let mut values = std::collections::BTreeMap::new();
        values.insert(
            GENERAL_ARCHITECTURE_KEY.to_string(),
            GgufMetadataValue::String(
                crate::models::qwen::QWEN3_FORCED_ALIGNER_GGML_ARCHITECTURE_ID.to_string(),
            ),
        );
        for key in [
            "qwen3_forced_aligner.audio.sample_rate_hz",
            "qwen3_forced_aligner.audio.n_mels",
            "qwen3_forced_aligner.audio.n_fft",
            "qwen3_forced_aligner.audio.win_length",
            "qwen3_forced_aligner.audio.hop_length",
            "qwen3_forced_aligner.audio.n_layers",
            "qwen3_forced_aligner.audio.d_model",
            "qwen3_forced_aligner.audio.n_heads",
            "qwen3_forced_aligner.llm.n_layers",
            "qwen3_forced_aligner.llm.d_model",
            "qwen3_forced_aligner.llm.n_heads",
            "qwen3_forced_aligner.llm.n_kv_heads",
            "qwen3_forced_aligner.llm.head_dim",
            "qwen3_forced_aligner.llm.embed_vocab_size",
            "qwen3_forced_aligner.llm.classify_num",
            "qwen3_forced_aligner.llm.max_positions",
            "qwen3_forced_aligner.audio_start_token_id",
            "qwen3_forced_aligner.audio_end_token_id",
            "qwen3_forced_aligner.audio_pad_token_id",
            "qwen3_forced_aligner.timestamp_token_id",
            "qwen3_forced_aligner.timestamp_segment_time_ms",
        ] {
            values.insert(key.to_string(), GgufMetadataValue::U32(1));
        }
        values.insert(
            "tokenizer.ggml.tokens".to_string(),
            GgufMetadataValue::StringArray(vec!["<pad>".to_string()]),
        );
        values.insert(
            "tokenizer.ggml.merges".to_string(),
            GgufMetadataValue::StringArray(Vec::new()),
        );
        GgufMetadata::from_values_for_test(values)
    }

    /// Positive direction: a forced-aligner pack that carries every metadata
    /// key the runtime loader needs is routed to the aux table and accepted,
    /// never rejected by ASR runtime adapter selection (which would happen if
    /// this architecture were not registered here -- see
    /// `native.rs::pull_contract_validation_routes_diarize_packs_to_their_loader`
    /// for the same shape of proof on the diarization aux kind).
    #[test]
    fn forced_aligner_pack_with_complete_metadata_is_accepted() {
        let metadata = valid_forced_aligner_metadata();
        let (kind, result) =
            validate_aux_runtime_pack_contract(Path::new("/nonexistent"), &metadata)
                .expect("forced-aligner architecture must be claimed by the aux table");
        assert_eq!(kind, AuxPackKind::ForcedAlignment);
        assert!(result.is_ok(), "got: {result:?}");
    }

    /// Negative direction: a forced-aligner pack missing a required
    /// `qwen3_forced_aligner.*` key must still be claimed by the aux table
    /// (so it is never silently accepted by ASR adapter selection instead)
    /// but must fail validation -- the actual bug this module closes: before
    /// this architecture was registered, a real published pack (which has
    /// never carried `openasr.audio.frontend`) fell through the aux table
    /// entirely and was rejected by unrelated ASR-adapter-selection metadata
    /// requirements instead of this family's own contract.
    #[test]
    fn forced_aligner_pack_missing_required_metadata_is_rejected() {
        let mut values = valid_forced_aligner_metadata().values().clone();
        values.remove("qwen3_forced_aligner.llm.classify_num");
        let metadata = GgufMetadata::from_values_for_test(values);

        let (kind, result) =
            validate_aux_runtime_pack_contract(Path::new("/nonexistent"), &metadata)
                .expect("forced-aligner architecture must still be claimed by the aux table");
        assert_eq!(kind, AuxPackKind::ForcedAlignment);
        let error = result.expect_err("pack missing a required metadata key must be rejected");
        assert!(
            error.contains("qwen3_forced_aligner.llm.classify_num"),
            "got: {error}"
        );

        // Also missing a tokenizer array (present in every real pack but not
        // covered by `parse_forced_aligner_runtime_metadata`'s scalar keys)
        // must independently fail closed.
        let mut values_no_tokens = valid_forced_aligner_metadata().values().clone();
        values_no_tokens.remove("tokenizer.ggml.tokens");
        let metadata_no_tokens = GgufMetadata::from_values_for_test(values_no_tokens);
        let (_, result_no_tokens) =
            validate_aux_runtime_pack_contract(Path::new("/nonexistent"), &metadata_no_tokens)
                .expect("forced-aligner architecture must still be claimed by the aux table");
        let error_no_tokens =
            result_no_tokens.expect_err("pack missing the BPE tokenizer array must be rejected");
        assert!(
            error_no_tokens.contains("tokenizer.ggml.tokens"),
            "got: {error_no_tokens}"
        );
    }
}
