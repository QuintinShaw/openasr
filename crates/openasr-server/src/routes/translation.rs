//! Realtime-translation pack capability, selection, and Hy-MT2 pack
//! validation. Pure code-motion from `lib.rs`; shared crate-root items come
//! via `use crate::*`, translation/pack types from `openasr_core`.

use std::{
    env,
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
    // Reuse `list_installed_packs`'s trust-boundary validation (safe
    // relative model_id/quant/filename components, no path traversal, the
    // record's declared `path` must literally be `quant_dir/filename`, and
    // the file must be a real non-symlink regular file) instead of a
    // hand-rolled directory scan that trusted `installed.json`'s `path`
    // field outright. Already sorted by `pull`, so `.find()` preserves the
    // same "first by pull order" tie-break the old manual scan applied.
    let packs = openasr_core::list_installed_packs(home).ok()?;
    packs
        .into_iter()
        .find(|pack| translation_installed_pack_matches(pack, requested_model))
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn stub_installed_pack(path: PathBuf, filename: &str) -> InstalledPack {
        InstalledPack {
            model_id: HYMT2_TRANSLATION_MODEL_ID.to_string(),
            display_name: "Hy-MT2 1.8B".to_string(),
            quant: "q4_k_m".to_string(),
            suffix: "q4_k_m".to_string(),
            pull: HYMT2_TRANSLATION_MODEL_ID.to_string(),
            filename: filename.to_string(),
            path,
            url: "https://example.invalid/model.oasr".to_string(),
            hf_revision: "a".repeat(40),
            sha256: "b".repeat(64),
            size_bytes: 0,
            installed_at_unix_seconds: 0,
            source: None,
        }
    }

    /// Aligns `find_installed_hymt2_translation_pack` with the trust
    /// boundary `list_installed_packs` already enforces: an `installed.json`
    /// record whose declared `path` does not resolve to
    /// `<quant_dir>/<filename>` must be rejected outright, not trusted just
    /// because the path happens to exist and isn't a symlink. The
    /// hand-rolled directory scan this replaced only checked the latter, so
    /// a record like this would previously have been accepted.
    #[test]
    fn rejects_installed_record_whose_declared_path_escapes_its_quant_dir() {
        let home = tempfile::tempdir().unwrap();
        let quant_dir = home
            .path()
            .join("models")
            .join(HYMT2_TRANSLATION_MODEL_ID)
            .join("q4_k_m");
        fs::create_dir_all(&quant_dir).unwrap();

        // A real, non-symlink file elsewhere under `home` -- the crafted
        // record points `path` at this instead of `quant_dir/model.oasr`.
        let escaped_path = home.path().join("escaped.oasr");
        fs::write(&escaped_path, b"not a real pack, just needs to exist").unwrap();

        let pack = stub_installed_pack(escaped_path, "model.oasr");
        fs::write(
            quant_dir.join("installed.json"),
            serde_json::to_string(&pack).unwrap(),
        )
        .unwrap();

        assert!(
            find_installed_hymt2_translation_pack(home.path(), None).is_none(),
            "an installed.json record whose declared path escapes its own quant \
             directory must be rejected, not silently trusted"
        );
    }
}
