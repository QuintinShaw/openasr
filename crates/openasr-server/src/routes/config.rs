//! `/v1/config` get/put handlers and the preferences-patch + validation
//! helpers. Pure code-motion from `lib.rs`; shared crate-root items come via
//! `use crate::*`, config-document types are imported directly from
//! `openasr_core::config`.

use axum::{Extension, Json};
use openasr_core::config::{
    OpenAsrConfigDocument, Preferences, load_config_document, save_config_document,
};

use crate::*;

pub(crate) async fn get_config(
    Extension(distribution): Extension<DistributionContext>,
) -> Result<Json<OpenAsrConfigDocument>, ApiError> {
    let home = distribution.openasr_home()?;
    let document = load_config_document(&home).map_err(ApiError::Config)?;
    validate_config_document(&document, &distribution)?;
    Ok(Json(document))
}

pub(crate) async fn put_config(
    Extension(distribution): Extension<DistributionContext>,
    Json(payload): Json<serde_json::Value>,
) -> Result<Json<OpenAsrConfigDocument>, ApiError> {
    let home = distribution.openasr_home()?;
    let document = config_document_from_update_payload(&home, payload)?;
    validate_config_document(&document, &distribution)?;
    save_config_document(&home, &document).map_err(ApiError::Config)?;
    let saved = load_config_document(&home).map_err(ApiError::Config)?;
    validate_config_document(&saved, &distribution)?;
    Ok(Json(saved))
}

fn config_document_from_update_payload(
    home: &std::path::Path,
    payload: serde_json::Value,
) -> Result<OpenAsrConfigDocument, ApiError> {
    if payload_has_config_fields(&payload) {
        return serde_json::from_value(payload).map_err(|error| {
            ApiError::Config(openasr_core::ConfigError::InvalidPreference {
                field: "config",
                reason: error.to_string(),
            })
        });
    }

    // The desktop preferences client owns only the nested `preferences` object.
    // Treat preferences-only requests as patches over the stored document so a
    // narrow update like `{ "preferences": { "onboarded": true } }` cannot
    // reset shortcut, tray, inference, model, or mirror settings to defaults.
    let mut document = load_config_document(home).map_err(ApiError::Config)?;
    merge_preferences_patch(
        &mut document.preferences,
        preferences_patch_payload(&payload)?,
    )?;
    Ok(document)
}

fn preferences_patch_payload(payload: &serde_json::Value) -> Result<&serde_json::Value, ApiError> {
    if let Some(preferences) = payload.get("preferences") {
        return Ok(preferences);
    }
    Ok(payload)
}

fn merge_preferences_patch(
    preferences: &mut Preferences,
    patch: &serde_json::Value,
) -> Result<(), ApiError> {
    let patch = patch.as_object().ok_or_else(|| {
        ApiError::Config(openasr_core::ConfigError::InvalidPreference {
            field: "preferences",
            reason: "must be a JSON object".to_string(),
        })
    })?;
    let mut merged = serde_json::to_value(&*preferences).map_err(ApiError::Serialize)?;
    let merged_object = merged.as_object_mut().ok_or_else(|| {
        ApiError::Config(openasr_core::ConfigError::InvalidPreference {
            field: "preferences",
            reason: "could not serialize existing preferences".to_string(),
        })
    })?;
    for (key, value) in patch {
        merged_object.insert(key.clone(), value.clone());
    }
    *preferences = serde_json::from_value(merged).map_err(|error| {
        ApiError::Config(openasr_core::ConfigError::InvalidPreference {
            field: "preferences",
            reason: error.to_string(),
        })
    })?;
    Ok(())
}

/// Whether a `/v1/config` request body carries the daemon/CLI-managed config
/// portion (vs a preferences-only update from the desktop preferences client).
fn payload_has_config_fields(payload: &serde_json::Value) -> bool {
    payload.as_object().is_some_and(|object| {
        [
            "default_model",
            "default_backend",
            "media",
            "download_source",
        ]
        .iter()
        .any(|key| object.contains_key(*key))
    })
}

pub(crate) fn validate_config_document(
    document: &OpenAsrConfigDocument,
    distribution: &DistributionContext,
) -> Result<(), ApiError> {
    let registry = load_registry(default_registry_dir()).map_err(ApiError::Registry)?;
    let home = distribution.openasr_home()?;
    let catalog = load_runtime_model_catalog(distribution.catalog_url(), &home)?;
    document
        .validate_with_catalog(&registry, catalog.as_ref())
        .map_err(ApiError::Config)
}
