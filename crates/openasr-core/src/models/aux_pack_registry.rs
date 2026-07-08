//! Single dispatch point for auxiliary (non-ASR) runtime pack contracts.
//!
//! ASR families are looked up through one data-driven table --
//! [`crate::arch::OpenAsrArchitectureRegistry`] -- keyed by `general.architecture`
//! and cross-checked (`openasr.model.family` / audio-frontend / decode-policy /
//! tokenizer) before an adapter is selected. Auxiliary packs (speaker
//! embedder, speaker segmenter, translation, punctuation) are not ASR
//! transcription architectures -- they have no audio frontend, tokenizer, or
//! decode policy in that sense -- so forcing them into
//! `OpenAsrArchitectureDescriptor` would model a shape they don't have (see
//! `models::pyannote`/`models::wespeaker`'s module docs, which already say so
//! explicitly). They still deserve **one** table instead of an ad hoc chain of
//! `if let Some(...)` calls in `api::backend::native`, so this module is that
//! table: one `general.architecture` value per aux family, matched by a single
//! lookup, fail-closed (`None` when no aux entry matches, so the caller falls
//! through to ASR adapter selection, which then fails closed on its own if the
//! pack matches nothing at all).
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
    /// Speaker embedder (WeSpeaker) / speaker segmenter (pyannote) diarization
    /// support packs.
    Diarization,
    /// Translation runtime packs (Hy-MT2).
    Translation,
    /// Punctuation-restoration packs (FireRedPunc).
    Punctuation,
}

impl AuxPackKind {
    /// The `"<label> failed: <error>"` prefix `validate_native_runtime_model_pack_contract`
    /// reports for this kind (unchanged from the pre-consolidation call sites).
    pub(crate) fn validation_failure_label(self) -> &'static str {
        match self {
            AuxPackKind::Diarization => "diarization pack validation failed",
            AuxPackKind::Translation => "translation pack validation failed",
            AuxPackKind::Punctuation => "punctuation pack validation failed",
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

fn validate_wespeaker(path: &Path, _metadata: &GgufMetadata) -> Result<(), String> {
    crate::diarize::embed::WeSpeakerEmbedder::from_oasr(path)
        .map(|_| ())
        .map_err(|error| error.to_string())
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

fn validate_firered_punc(_path: &Path, metadata: &GgufMetadata) -> Result<(), String> {
    crate::models::firered_punc::runtime_contract::parse_and_validate_firered_punc_metadata(
        metadata,
    )
    .map(|_| ())
    .map_err(|error| error.to_string())
}

const AUX_PACK_DESCRIPTORS: &[AuxPackDescriptor] = &[
    AuxPackDescriptor {
        architecture_id: crate::models::wespeaker::WESPEAKER_GGML_ARCHITECTURE_ID,
        kind: AuxPackKind::Diarization,
        validate: validate_wespeaker,
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
];

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
}
