//! Realtime-translation pack capability, selection, and Hy-MT2 pack
//! validation. Pure code-motion from `lib.rs`; shared crate-root items come
//! via `use crate::*`, translation/pack types from `openasr_core`.

use std::{
    env, fs,
    path::{Path, PathBuf},
};

use openasr_core::{InstalledPack, RealtimeTranslationCapability};

use crate::*;

pub(crate) const HYMT2_TRANSLATION_MODEL_ID: &str =
    RealtimeTranslationCapability::MODEL_ID_HYMT2_1_8B_Q4_K_M;
const HYMT2_TRANSLATION_MODEL_ALIASES: &[&str] = &[
    RealtimeTranslationCapability::MODEL_ID_HYMT2_1_8B_Q4_K_M,
    "hymt2-1.8b",
    "hy-mt2-1.8b",
    "Hy-MT2-1.8B-Q4_K_M",
];
const HYMT2_REAL_PACK_ENV: &str = "OPENASR_HYMT2_REAL_PACK";

#[derive(Debug, Clone)]
pub(crate) struct TranslationPackSelection {
    pub(crate) path: PathBuf,
}

pub(crate) fn translation_capability_for_distribution(
    distribution: &DistributionContext,
) -> RealtimeTranslationCapability {
    match resolve_translation_pack_selection(distribution, None) {
        Ok(_) => RealtimeTranslationCapability::installed_hymt2(),
        Err(reason)
            if reason.starts_with(RealtimeTranslationCapability::REASON_MODEL_UNSUPPORTED) =>
        {
            RealtimeTranslationCapability::unavailable(
                RealtimeTranslationCapability::REASON_MODEL_UNSUPPORTED,
            )
        }
        Err(_) => RealtimeTranslationCapability::unavailable(
            RealtimeTranslationCapability::REASON_PACK_MISSING,
        ),
    }
}

pub(crate) fn resolve_translation_pack_selection(
    distribution: &DistributionContext,
    requested_model: Option<&str>,
) -> Result<TranslationPackSelection, String> {
    if let Some(model) = requested_model
        && !translation_model_ref_supported(model)
    {
        return Err(format!(
            "{}: realtime translation MVP only supports {} (aliases: hymt2-1.8b) for zh->en",
            RealtimeTranslationCapability::REASON_MODEL_UNSUPPORTED,
            HYMT2_TRANSLATION_MODEL_ID
        ));
    }

    if let Some(path) = env::var_os(HYMT2_REAL_PACK_ENV).map(PathBuf::from)
        && path.exists()
    {
        validate_hymt2_translation_pack(&path)?;
        return Ok(TranslationPackSelection { path });
    }

    let home = distribution.openasr_home().map_err(|error| {
        format!(
            "{}: {error}",
            RealtimeTranslationCapability::REASON_PACK_MISSING
        )
    })?;
    let Some(pack) = find_installed_hymt2_translation_pack(&home, requested_model) else {
        return Err(format!(
            "{}: install {} before enabling realtime translation",
            RealtimeTranslationCapability::REASON_PACK_MISSING,
            HYMT2_TRANSLATION_MODEL_ID
        ));
    };
    validate_hymt2_installed_pack_revision(&pack)?;
    validate_hymt2_translation_pack(&pack.path)?;
    Ok(TranslationPackSelection { path: pack.path })
}

pub(crate) fn translation_model_ref_supported(model: &str) -> bool {
    let normalized = normalize_translation_model_ref(model);
    HYMT2_TRANSLATION_MODEL_ALIASES
        .iter()
        .map(|alias| normalize_translation_model_ref(alias))
        .any(|alias| alias == normalized)
}

fn normalize_translation_model_ref(model: &str) -> String {
    model
        .trim()
        .trim_end_matches(".oasr")
        .replace('_', "-")
        .to_ascii_lowercase()
}

fn find_installed_hymt2_translation_pack(
    home: &Path,
    requested_model: Option<&str>,
) -> Option<InstalledPack> {
    let root = home.join("models");
    let model_dirs = fs::read_dir(root).ok()?;
    let mut candidates = Vec::new();
    for model_dir in model_dirs.flatten() {
        let Ok(quant_dirs) = fs::read_dir(model_dir.path()) else {
            continue;
        };
        for quant_dir in quant_dirs.flatten() {
            let metadata_path = quant_dir.path().join("installed.json");
            let Ok(contents) = fs::read_to_string(metadata_path) else {
                continue;
            };
            let Ok(pack) = serde_json::from_str::<InstalledPack>(&contents) else {
                continue;
            };
            if !translation_installed_pack_matches(&pack, requested_model) {
                continue;
            }
            if fs::symlink_metadata(&pack.path)
                .ok()
                .is_some_and(|metadata| metadata.is_file() && !metadata.file_type().is_symlink())
            {
                candidates.push(pack);
            }
        }
    }
    candidates.sort_by(|left, right| left.pull.cmp(&right.pull));
    candidates.into_iter().next()
}

fn translation_installed_pack_matches(pack: &InstalledPack, requested_model: Option<&str>) -> bool {
    if let Some(requested) = requested_model
        && !translation_model_ref_supported(requested)
    {
        return false;
    }
    translation_model_ref_supported(&pack.model_id)
        || translation_model_ref_supported(&pack.pull)
        || translation_model_ref_supported(&pack.filename)
}

fn validate_hymt2_installed_pack_revision(pack: &InstalledPack) -> Result<(), String> {
    if pack.hf_revision.len() == 40
        && pack
            .hf_revision
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit())
    {
        Ok(())
    } else {
        Err(format!(
            "{}: installed {} pack metadata is missing a pinned hf_revision",
            RealtimeTranslationCapability::REASON_MODEL_UNSUPPORTED,
            HYMT2_TRANSLATION_MODEL_ID
        ))
    }
}

fn validate_hymt2_translation_pack(path: &Path) -> Result<(), String> {
    openasr_core::Hymt2Runtime::probe_path(path)
        .map(|_| ())
        .map_err(|error| {
            format!(
                "{}: {}",
                RealtimeTranslationCapability::REASON_MODEL_UNSUPPORTED,
                error
            )
        })
}
