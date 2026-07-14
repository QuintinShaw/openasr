//! Single authority for resolving and persisting the "default model" state,
//! which spans two files under the OpenASR home: `config.json`'s
//! `default_model` field (the user's explicit choice) and `default.json` (a
//! pointer recording the last pack a default write installed). Before this
//! module existed, the server, the CLI, and the config layer each carried
//! their own reading of these two files, and only the server's read the
//! `default.json` pointer as a fallback -- see `docs/default-model-resolution.md`
//! for the contract this module now owns for every caller (server routes,
//! CLI serve/transcribe pack lookup, and eventually the desktop shell).
//!
//! Fail-closed by design: `resolve` never invents a default when nothing is
//! configured, and never substitutes a different installed pack when the
//! configured one is missing. Silently picking "some" installed model would
//! defeat the point of a "default" (the user chose *this* model) and could
//! route audio through an unexpected model/quant.

use std::path::Path;

use thiserror::Error;

use crate::{
    CatalogError, ConfigError, InstalledPack, LaunchPackRequest, ModelCatalog, PullError,
    QuantPreference, default_pack_pointer_path, host_quant_recommendation_profile,
    list_installed_packs, load_config_document, load_model_catalog, persist_default_pack_pointer,
    read_default_pack_pointer, resolve_launch_pack, save_config_document,
    save_default_model_selection,
};

#[derive(Debug, Error)]
pub enum DefaultSelectionError {
    #[error(transparent)]
    Config(#[from] ConfigError),
    #[error(transparent)]
    Pull(#[from] PullError),
    #[error(transparent)]
    Catalog(#[from] CatalogError),
}

/// The result of resolving the persisted default model against the packs
/// actually installed on disk. A bare `Option<InstalledPack>` cannot tell
/// "nothing configured" apart from "configured but not installed" -- both
/// collapse to `None` -- yet callers (the desktop default-model banner, the
/// `GET /v1/models/default` status field) need to tell those apart to show
/// the right prompt ("choose a model" vs. "reinstall your default").
// `resolve` returns this by value at most once per call (never in a hot loop
// or a large collection), so the `Installed(InstalledPack)` / `NotInstalled
// (String)` size delta doesn't warrant boxing every caller's match arm --
// server routes and the CLI both want to destructure it directly.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DefaultModelResolution {
    /// `config.default_model` (or the `default.json` pointer as a fallback)
    /// names a model that has a matching pack installed.
    Installed(InstalledPack),
    /// A default is configured, but no installed pack matches it (removed,
    /// never installed, or the wrong quant with no fallback available).
    NotInstalled(String),
    /// Neither `config.default_model` nor the `default.json` pointer is set.
    Unset,
}

impl DefaultModelResolution {
    pub fn installed_pack(&self) -> Option<&InstalledPack> {
        match self {
            Self::Installed(pack) => Some(pack),
            Self::NotInstalled(_) | Self::Unset => None,
        }
    }

    pub fn into_installed_pack(self) -> Option<InstalledPack> {
        match self {
            Self::Installed(pack) => Some(pack),
            Self::NotInstalled(_) | Self::Unset => None,
        }
    }
}

/// Resolves the persisted default model against installed packs, loading the
/// catalog from `catalog_url` first (a bare `None` skips catalog loading
/// entirely rather than falling back to the bundled catalog -- see
/// `resolve_with_catalog` for the shared logic and why callers that already
/// hold a loaded catalog, like the CLI, should call that instead).
///
/// Priority (ported verbatim from the original server-only resolver):
/// `config.default_model` wins when set; the `default.json` pointer's model
/// id is a fallback only when `config.default_model` is unset. When
/// `preferences.quant_preference` is `Pinned` and a pointer exists, the
/// pointer's quant is tried first (falling back to the best installed quant
/// if that exact quant was removed) -- this keeps `openasr pull <id>:<quant>`
/// sticky across quant changes without re-filtering the candidate list to a
/// single quant (which would break the Pinned-missing fallback ladder).
pub fn resolve(
    home: &Path,
    catalog_url: Option<&str>,
) -> Result<DefaultModelResolution, DefaultSelectionError> {
    let catalog = catalog_url
        .map(|catalog_url| load_model_catalog(Some(catalog_url), home))
        .transpose()?;
    resolve_with_catalog(home, catalog.as_ref())
}

/// Same resolution as `resolve`, but against an already-loaded catalog
/// (or `None` to resolve without catalog-assisted alias matching). Lets a
/// caller that owns its own catalog-loading policy -- the CLI's
/// `OPENASR_CATALOG_URL`/local-file override in `load_cli_model_catalog`,
/// distinct from the server's `catalog_url` override -- share this resolver
/// without loading the catalog twice or adopting the server's policy.
pub fn resolve_with_catalog(
    home: &Path,
    catalog: Option<&ModelCatalog>,
) -> Result<DefaultModelResolution, DefaultSelectionError> {
    let packs = list_installed_packs(home)?;
    let document = load_config_document(home)?;
    let pointer = read_default_pack_pointer(home)?;

    if matches!(
        document.preferences.quant_preference,
        QuantPreference::Pinned { .. }
    ) && let Some(pointer) = pointer.as_ref()
    {
        let pointer_preference = QuantPreference::pinned(&pointer.quant);
        let reference = document
            .config
            .default_model
            .as_deref()
            .unwrap_or(pointer.model_id.as_str());
        return Ok(select(&packs, reference, &pointer_preference, catalog));
    }

    let Some(default_model) = document
        .config
        .default_model
        .as_deref()
        .or_else(|| pointer.as_ref().map(|pointer| pointer.model_id.as_str()))
    else {
        return Ok(DefaultModelResolution::Unset);
    };
    Ok(select(
        &packs,
        default_model,
        &document.preferences.quant_preference,
        catalog,
    ))
}

fn select(
    packs: &[InstalledPack],
    reference: &str,
    preference: &QuantPreference,
    catalog: Option<&ModelCatalog>,
) -> DefaultModelResolution {
    let request = LaunchPackRequest {
        model_ref: reference,
        preference,
        catalog,
        host_profile: host_quant_recommendation_profile(),
    };
    match resolve_launch_pack(packs, &request) {
        Ok(selection) => DefaultModelResolution::Installed(selection.pack),
        Err(_) => DefaultModelResolution::NotInstalled(reference.to_string()),
    }
}

/// Persists `pack` as the default model: writes `config.json`'s
/// `default_model` (bare model id) and the `default.json` pointer, in that
/// order. Callers must go through this single function rather than calling
/// the two underlying writes separately, so the two files never drift.
pub fn persist(
    home: &Path,
    pack: &InstalledPack,
    quant_preference: QuantPreference,
) -> Result<(), DefaultSelectionError> {
    save_default_model_selection(home, pack.model_id.clone(), quant_preference)?;
    persist_default_pack_pointer(home, pack)?;
    Ok(())
}

/// Clears the persisted default: resets `config.json`'s `default_model` and
/// `quant_preference` to their unset states and removes the `default.json`
/// pointer file (a missing pointer file is not an error).
pub fn clear(home: &Path) -> Result<(), DefaultSelectionError> {
    let mut document = load_config_document(home)?;
    document.config.default_model = None;
    document.preferences.quant_preference = QuantPreference::Auto;
    save_config_document(home, &document)?;

    let path = default_pack_pointer_path(home);
    match std::fs::remove_file(&path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(source) => Err(DefaultSelectionError::Pull(PullError::Io { path, source })),
    }
}

#[cfg(test)]
#[path = "default_selection_tests.rs"]
mod default_selection_tests;
