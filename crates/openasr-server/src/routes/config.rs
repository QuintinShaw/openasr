//! `/v1/config` get/put handlers and the preferences-patch + validation
//! helpers. Pure code-motion from `lib.rs`; shared crate-root items come via
//! `use crate::*`, config-document types are imported directly from
//! `openasr_core::config`.

use axum::{Extension, Json};
use openasr_core::config::{OpenAsrConfigDocument, Preferences, load_config_document};

use crate::*;

pub(crate) async fn get_config(
    Extension(distribution): Extension<DistributionContext>,
) -> Result<Json<OpenAsrConfigDocument>, ApiError> {
    let home = distribution.openasr_home()?;
    let mut document = load_config_document(&home).map_err(ApiError::Config)?;
    document.config.default_model = openasr_core::default_selection::current_default_model(&home)?;
    validate_config_document(&document, &distribution)?;
    Ok(Json(document))
}

pub(crate) async fn put_config(
    Extension(distribution): Extension<DistributionContext>,
    Extension(idle_controller): Extension<IdleUnloadController>,
    Json(payload): Json<serde_json::Value>,
) -> Result<Json<OpenAsrConfigDocument>, ApiError> {
    // All read/validate/save/reload/publish work is synchronous. Keep this
    // router-scoped lock out of await points so concurrent preference patches
    // serialize their read-modify-write and policy publication.
    let _write = idle_controller.config_write_lock();
    let home = distribution.openasr_home()?;
    let document = config_document_from_update_payload(&home, payload)?;
    validate_config_document(&document, &distribution)?;
    openasr_core::default_selection::save_config_document_preserving_v2_selection(
        &home, &document,
    )?;
    let mut saved = load_config_document(&home).map_err(ApiError::Config)?;
    saved.config.default_model = openasr_core::default_selection::current_default_model(&home)?;
    validate_config_document(&saved, &distribution)?;
    idle_controller.publish(saved.preferences.idle_unload.idle_threshold());
    Ok(Json(saved))
}

pub(crate) fn config_document_from_update_payload(
    home: &std::path::Path,
    payload: serde_json::Value,
) -> Result<OpenAsrConfigDocument, ApiError> {
    let current_default = openasr_core::default_selection::current_default_model(home)?;
    let mut payload = payload;
    if let Some(object) = payload.as_object_mut()
        && let Some(default_model) = object.get("default_model")
    {
        if !default_model.is_null() && current_default.as_deref() != default_model.as_str() {
            return Err(ApiError::Config(
                openasr_core::ConfigError::InvalidPreference {
                    field: "default_model",
                    reason: "default_model must be changed through /v1/models/default".to_string(),
                },
            ));
        }
        object.remove("default_model");
    }
    if payload_has_config_fields(&payload) {
        let mut document: OpenAsrConfigDocument =
            serde_json::from_value(payload).map_err(|error| {
                ApiError::Config(openasr_core::ConfigError::InvalidPreference {
                    field: "config",
                    reason: error.to_string(),
                })
            })?;
        document.config.default_model = current_default;
        return Ok(document);
    }

    // The desktop preferences client owns only the nested `preferences` object.
    // Treat preferences-only requests as patches over the stored document.
    let mut document = load_config_document(home).map_err(ApiError::Config)?;
    merge_preferences_patch(
        &mut document.preferences,
        preferences_patch_payload(&payload)?,
    )?;
    document.config.default_model = current_default;
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
            "models_dir",
        ]
        .iter()
        .any(|key| object.contains_key(*key))
    })
}

pub(crate) fn validate_config_document(
    document: &OpenAsrConfigDocument,
    distribution: &DistributionContext,
) -> Result<(), ApiError> {
    let home = distribution.openasr_home()?;
    let catalog = load_runtime_model_catalog(distribution.catalog_source(), &home)?;
    let registry = runtime_registry(catalog.as_ref()).map_err(ApiError::from)?;
    document
        .validate_with_catalog(&registry, catalog.as_ref())
        .map_err(ApiError::Config)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn isolated_distribution(home: &std::path::Path) -> DistributionContext {
        DistributionContext::new(DistributionRuntime {
            openasr_home: Some(home.to_path_buf()),
            catalog_url: None,
            catalog_local_override: None,
        })
    }

    #[tokio::test]
    async fn idle_policy_validation_failure_preserves_config_and_live_policy() {
        let home = tempfile::tempdir().unwrap();
        let distribution = isolated_distribution(home.path());
        let controller = IdleUnloadController::new(None);
        let receiver = controller.subscribe();
        let before = load_config_document(home.path()).unwrap();
        let result = put_config(
            Extension(distribution),
            Extension(controller.clone()),
            Json(serde_json::json!({"preferences": {"idle_unload": "now", "language": " "}})),
        )
        .await;
        assert!(result.unwrap_err().to_string().contains("language"));
        assert_eq!(*receiver.borrow(), None);
        assert!(!receiver.has_changed().unwrap());
        assert_eq!(load_config_document(home.path()).unwrap(), before);
    }

    #[tokio::test]
    async fn idle_policy_save_failure_does_not_publish_unpersisted_policy() {
        let home = tempfile::tempdir().unwrap();
        let distribution = isolated_distribution(home.path());
        let controller = IdleUnloadController::new(None);
        let receiver = controller.subscribe();
        let payload = serde_json::json!({"preferences": {"idle_unload": "now"}});
        let document = config_document_from_update_payload(home.path(), payload.clone()).unwrap();
        validate_config_document(&document, &distribution).unwrap();
        // A directory at the lock-file path makes the real persistence operation
        // fail on every platform, without mutating global permissions or hooks.
        std::fs::create_dir(home.path().join(".openasr-default-selection.lock")).unwrap();
        let result = put_config(
            Extension(distribution),
            Extension(controller.clone()),
            Json(payload),
        )
        .await;
        assert!(result.is_err());
        assert!(!home.path().join("config.json").exists());
        assert_eq!(*receiver.borrow(), None);
        assert!(!receiver.has_changed().unwrap());
    }

    #[test]
    fn idle_policy_concurrent_patches_preserve_fields_and_persisted_policy() {
        let home = tempfile::tempdir().unwrap();
        let distribution = isolated_distribution(home.path());
        let controller = IdleUnloadController::new(None);
        let receiver = controller.subscribe();
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(3));
        std::thread::scope(|scope| {
            for payload in [
                serde_json::json!({"preferences": {"idle_unload": "now"}}),
                serde_json::json!({"preferences": {"language": "en", "history_retention": "forever"}}),
            ] {
                let distribution = distribution.clone();
                let controller = controller.clone();
                let barrier = barrier.clone();
                scope.spawn(move || {
                    let runtime = tokio::runtime::Builder::new_current_thread()
                        .build()
                        .unwrap();
                    barrier.wait();
                    let _ = runtime
                        .block_on(put_config(
                            Extension(distribution),
                            Extension(controller),
                            Json(payload),
                        ))
                        .unwrap();
                });
            }
            barrier.wait();
        });
        let saved = load_config_document(home.path()).unwrap();
        assert_eq!(saved.preferences.language.as_deref(), Some("en"));
        assert_eq!(
            saved.preferences.history_retention,
            openasr_core::config::HistoryRetentionPolicy::Forever
        );
        assert_eq!(
            saved.preferences.idle_unload,
            openasr_core::config::IdleUnloadPolicy::Now
        );
        assert_eq!(
            *receiver.borrow(),
            saved.preferences.idle_unload.idle_threshold()
        );
    }

    #[test]
    fn generic_config_writer_echoes_current_active_default_without_rewriting_it() {
        let temp = tempfile::tempdir().unwrap();
        let mut document = OpenAsrConfigDocument::default();
        document.config.default_model = Some("active-model".to_string());
        openasr_core::save_config_document(temp.path(), &document).unwrap();

        let saved = config_document_from_update_payload(
            temp.path(),
            serde_json::json!({"default_model": "active-model", "default_backend": "mock"}),
        )
        .unwrap();

        assert_eq!(saved.config.default_model.as_deref(), Some("active-model"));
    }

    #[test]
    fn generic_config_write_preserves_authoritative_v2_default_after_stale_read() {
        let temp = tempfile::tempdir().unwrap();
        let record = openasr_core::default_selection::ActiveModelSelectionV2 {
            schema_version:
                openasr_core::default_selection::ACTIVE_MODEL_SELECTION_V2_SCHEMA_VERSION,
            selection_generation: 0,
            status: openasr_core::default_selection::ActiveModelSelectionStatus::NotInstalled,
            pull: Some("authoritative-model:q8".to_string()),
            model_id: Some("authoritative-model".to_string()),
            quant: Some("q8_0".to_string()),
            architecture_id: None,
            expected_pack: None,
            quant_preference: openasr_core::QuantPreference::pinned("q8_0"),
            execution_intent: "auto".to_string(),
            checksum: String::new(),
        };
        openasr_core::default_selection::persist_v2_record(temp.path(), record).unwrap();

        let mut stale = OpenAsrConfigDocument::default();
        stale.config.default_model = Some("stale-model".to_string());
        stale.preferences.quant_preference = openasr_core::QuantPreference::Auto;
        openasr_core::default_selection::save_config_document_preserving_v2_selection(
            temp.path(),
            &stale,
        )
        .unwrap();

        let saved = load_config_document(temp.path()).unwrap();
        assert_eq!(
            saved.config.default_model.as_deref(),
            Some("authoritative-model")
        );
        assert_eq!(
            saved.preferences.quant_preference,
            openasr_core::QuantPreference::pinned("q8_0")
        );
    }

    #[test]
    fn generic_config_writer_rejects_default_model() {
        let temp = tempfile::tempdir().unwrap();
        let error = config_document_from_update_payload(
            temp.path(),
            serde_json::json!({"default_model": "whisper-small"}),
        )
        .unwrap_err();
        assert!(error.to_string().contains("/v1/models/default"));
    }
}
