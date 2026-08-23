use std::fs;
use std::path::Path;

use super::*;
use crate::testing::{TinyGgufFixtureSpec, write_tiny_gguf_runtime_source};
use crate::{OpenAsrConfigDocument, config_path, save_config_document};
use sha2::Digest;

/// Installs `model_id`/`quant` the way the store actually holds it: an immutable
/// object plus the ref that names it.
///
/// The ref is re-validated on every lookup (`InstalledModelStore` checks the
/// object exists, is a regular file, and matches the recorded size), so a ref
/// with no backing object is silently dropped rather than "installed". The
/// backing bytes use the graph-complete whisper fixture because installs enforce
/// `verify_native_runtime_model_pack_path`, which the bare non-graph spec
/// fails.
fn write_installed_pack(home: &Path, model_id: &str, quant: &str, suffix: &str) -> InstalledPack {
    let filename = format!("{model_id}-{quant}.oasr");
    let models = home.join("models");

    let staged = models.join("fixture-source").join(&filename);
    fs::create_dir_all(staged.parent().expect("staged parent")).expect("create fixture dir");
    let spec = TinyGgufFixtureSpec::whisper_oasr_v1_encoder_graph_one_layer(model_id);
    write_tiny_gguf_runtime_source(&staged, &spec).expect("write tiny gguf runtime source");
    let bytes = fs::read(&staged).expect("read fixture pack");
    fs::remove_dir_all(models.join("fixture-source")).expect("drop fixture staging dir");

    let sha256 = format!("{:x}", sha2::Sha256::digest(&bytes));
    let path = models.join("objects/sha256").join(&sha256).join("content");
    fs::create_dir_all(path.parent().expect("object parent")).expect("create object dir");
    fs::write(&path, &bytes).expect("write object");

    let pack = InstalledPack {
        model_id: model_id.to_string(),
        display_name: model_id.to_string(),
        quant: quant.to_string(),
        suffix: suffix.to_string(),
        pull: format!("{model_id}:{suffix}"),
        filename,
        path,
        url: format!("https://example.test/{model_id}-{quant}.oasr"),
        hf_revision: "0123456789abcdef0123456789abcdef01234567".to_string(),
        sha256,
        size_bytes: bytes.len() as u64,
        installed_at_unix_seconds: 1,
        source: None,
    };
    let ref_path = models
        .join("refs")
        .join(model_id)
        .join(format!("{quant}.json"));
    fs::create_dir_all(ref_path.parent().expect("ref parent")).expect("create ref dir");
    fs::write(
        &ref_path,
        serde_json::to_string_pretty(&pack).expect("serialize installed pack"),
    )
    .expect("write model ref");
    pack
}

fn write_config_default_model(home: &Path, model_id: &str) {
    let document = OpenAsrConfigDocument {
        config: crate::OpenAsrConfig {
            default_model: Some(model_id.to_string()),
            ..crate::OpenAsrConfig::default()
        },
        ..OpenAsrConfigDocument::default()
    };
    save_config_document(home, &document).expect("save config document");
}

#[test]
fn resolve_is_unset_with_no_config_and_no_pointer() {
    let temp = tempfile::tempdir().unwrap();

    let resolution = resolve(temp.path(), None).unwrap();

    assert_eq!(resolution, DefaultModelResolution::Unset);
}

#[test]
fn resolve_is_installed_when_config_default_matches_an_installed_pack() {
    let temp = tempfile::tempdir().unwrap();
    let pack = write_installed_pack(temp.path(), "whisper-small", "q8_0", "q8");
    write_config_default_model(temp.path(), "whisper-small");

    let resolution = resolve(temp.path(), None).unwrap();

    assert_eq!(resolution, DefaultModelResolution::Installed(pack));
}

#[test]
fn resolve_is_not_installed_when_configured_model_has_no_matching_pack() {
    let temp = tempfile::tempdir().unwrap();
    write_config_default_model(temp.path(), "whisper-small");

    let resolution = resolve(temp.path(), None).unwrap();

    assert_eq!(
        resolution,
        DefaultModelResolution::NotInstalled("whisper-small".to_string())
    );
}

/// Fail-closed core assertion: a configured-but-uninstalled default model
/// must resolve to `NotInstalled`, never silently substitute a different
/// pack that happens to be on disk (even with no pointer file at all). This
/// is the exact bug class described in the refactor brief: a fresh install
/// with a stale/unreachable `default_model` must not fall back to "whatever
/// is installed".
#[test]
fn resolve_does_not_fall_back_to_a_different_installed_pack() {
    let temp = tempfile::tempdir().unwrap();
    // A different model is installed on disk...
    write_installed_pack(temp.path(), "dolphin-base", "q8_0", "q8");
    // ...but the configured default points elsewhere, and there is no
    // default.json pointer to fall back to.
    write_config_default_model(temp.path(), "whisper-small");
    assert!(
        !crate::default_pack_pointer_path(temp.path()).exists(),
        "test setup must not have a pointer file"
    );

    let resolution = resolve(temp.path(), None).unwrap();

    assert_eq!(
        resolution,
        DefaultModelResolution::NotInstalled("whisper-small".to_string())
    );
    assert!(resolution.installed_pack().is_none());
}

#[test]
fn resolve_falls_back_to_pointer_model_id_when_config_default_is_unset() {
    let temp = tempfile::tempdir().unwrap();
    let pack = write_installed_pack(temp.path(), "whisper-small", "q8_0", "q8");
    persist_default_pack_pointer(temp.path(), &pack).unwrap();
    // config.default_model stays None (fresh config document).

    let resolution = resolve(temp.path(), None).unwrap();

    assert_eq!(resolution, DefaultModelResolution::Installed(pack));
}

#[test]
fn persist_writes_config_and_pointer_together() {
    let temp = tempfile::tempdir().unwrap();
    let pack = write_installed_pack(temp.path(), "whisper-small", "q8_0", "q8");

    persist(temp.path(), &pack, QuantPreference::pinned("q8_0")).unwrap();

    let document = load_config_document(temp.path()).unwrap();
    assert_eq!(
        document.config.default_model.as_deref(),
        Some("whisper-small")
    );
    let pointer = read_default_pack_pointer(temp.path()).unwrap().unwrap();
    assert_eq!(pointer.model_id, "whisper-small");
    assert_eq!(
        resolve(temp.path(), None).unwrap(),
        DefaultModelResolution::Installed(pack)
    );
}

#[test]
fn clear_resets_config_and_removes_pointer() {
    let temp = tempfile::tempdir().unwrap();
    let pack = write_installed_pack(temp.path(), "whisper-small", "q8_0", "q8");
    persist(temp.path(), &pack, QuantPreference::pinned("q8_0")).unwrap();
    assert!(config_path(temp.path()).exists());

    clear(temp.path()).unwrap();

    let document = load_config_document(temp.path()).unwrap();
    assert_eq!(document.config.default_model, None);
    assert_eq!(document.preferences.quant_preference, QuantPreference::Auto);
    assert!(!crate::default_pack_pointer_path(temp.path()).exists());
    assert_eq!(
        resolve(temp.path(), None).unwrap(),
        DefaultModelResolution::Unset
    );
}

#[test]
fn clear_is_idempotent_without_a_pointer_file() {
    let temp = tempfile::tempdir().unwrap();

    clear(temp.path()).unwrap();
    clear(temp.path()).unwrap();
}

#[test]
fn v2_persist_is_self_checksumming_and_contains_no_absolute_path() {
    let temp = tempfile::tempdir().unwrap();
    let pack = write_installed_pack(temp.path(), "whisper-small", "q8_0", "q8");

    persist(temp.path(), &pack, QuantPreference::Auto).unwrap();

    let path = active_model_selection_v2_path(temp.path());
    let value: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap();
    assert_eq!(
        value["schema_version"],
        ACTIVE_MODEL_SELECTION_V2_SCHEMA_VERSION
    );
    assert_eq!(value["selection_generation"], 1);
    assert_eq!(value["status"], "installed");
    assert!(value.get("path").is_none());
    assert!(
        value
            .to_string()
            .find(temp.path().to_str().unwrap())
            .is_none()
    );
}

#[test]
fn v2_generation_is_monotonic_and_unset_is_a_complete_record() {
    let temp = tempfile::tempdir().unwrap();
    let pack = write_installed_pack(temp.path(), "whisper-small", "q8_0", "q8");

    persist(temp.path(), &pack, QuantPreference::Auto).unwrap();
    clear(temp.path()).unwrap();

    let record: ActiveModelSelectionV2 = serde_json::from_str(
        &fs::read_to_string(active_model_selection_v2_path(temp.path())).unwrap(),
    )
    .unwrap();
    assert_eq!(record.selection_generation, 2);
    assert_eq!(record.status, ActiveModelSelectionStatus::Unset);
    assert!(record.checksum_for_record().unwrap() == record.checksum);
    assert_eq!(
        resolve(temp.path(), None).unwrap(),
        DefaultModelResolution::Unset
    );
}

#[test]
fn v2_unset_does_not_fall_back_to_stale_legacy_projection() {
    let temp = tempfile::tempdir().unwrap();
    let pack = write_installed_pack(temp.path(), "whisper-small", "q8_0", "q8");
    write_config_default_model(temp.path(), "whisper-small");
    persist_default_pack_pointer(temp.path(), &pack).unwrap();

    let record = ActiveModelSelectionV2 {
        schema_version: ACTIVE_MODEL_SELECTION_V2_SCHEMA_VERSION,
        selection_generation: 0,
        status: ActiveModelSelectionStatus::Unset,
        pull: None,
        model_id: None,
        quant: None,
        architecture_id: None,
        expected_pack: None,
        quant_preference: QuantPreference::Auto,
        execution_intent: "auto".to_string(),
        checksum: String::new(),
    };
    persist_v2_record(temp.path(), record).unwrap();

    assert_eq!(
        current_default_model(temp.path()).unwrap(),
        None,
        "an existing V2 Unset record must hide stale legacy projections"
    );
    assert_eq!(
        resolve(temp.path(), None).unwrap(),
        DefaultModelResolution::Unset
    );
}

#[test]
fn v2_checksum_corruption_fails_closed_without_legacy_fallback() {
    let temp = tempfile::tempdir().unwrap();
    let pack = write_installed_pack(temp.path(), "whisper-small", "q8_0", "q8");
    persist(temp.path(), &pack, QuantPreference::Auto).unwrap();
    let path = active_model_selection_v2_path(temp.path());
    let mut value: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
    value["model_id"] = serde_json::Value::String("tampered-model".to_string());
    fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();

    let error = resolve(temp.path(), None).unwrap_err();
    assert!(error.to_string().contains("checksum"));
}

#[test]
fn v2_rejects_platform_path_lookalikes_in_logical_fields() {
    let temp = tempfile::tempdir().unwrap();
    let record = ActiveModelSelectionV2 {
        schema_version: ACTIVE_MODEL_SELECTION_V2_SCHEMA_VERSION,
        selection_generation: 0,
        status: ActiveModelSelectionStatus::NotInstalled,
        pull: Some("C:/models:q8".to_string()),
        model_id: Some("../escape".to_string()),
        quant: Some("q8_0".to_string()),
        architecture_id: Some("whisper".to_string()),
        expected_pack: None,
        quant_preference: QuantPreference::Auto,
        execution_intent: "auto".to_string(),
        checksum: String::new(),
    };
    assert!(persist_v2_record(temp.path(), record).is_err());
}

#[test]
fn concurrent_writers_serialize_generation_and_leave_valid_v2() {
    let temp = tempfile::tempdir().unwrap();
    let pack = write_installed_pack(temp.path(), "whisper-small", "q8_0", "q8");
    let barrier = std::sync::Arc::new(std::sync::Barrier::new(3));
    let home = temp.path().to_path_buf();
    let first_pack = pack.clone();
    let first_barrier = barrier.clone();
    let first = std::thread::spawn(move || {
        first_barrier.wait();
        persist(&home, &first_pack, QuantPreference::Auto).unwrap()
    });
    let home = temp.path().to_path_buf();
    let second_pack = pack.clone();
    let second_barrier = barrier.clone();
    let second = std::thread::spawn(move || {
        second_barrier.wait();
        persist(&home, &second_pack, QuantPreference::Auto).unwrap()
    });
    barrier.wait();
    first.join().unwrap();
    second.join().unwrap();

    let record = read_active_model_selection_v2(temp.path())
        .unwrap()
        .unwrap();
    assert_eq!(record.selection_generation, 2);
    assert_eq!(
        resolve(temp.path(), None).unwrap(),
        DefaultModelResolution::Installed(pack)
    );
}

#[test]
fn clear_reports_v2_committed_when_legacy_projection_fails() {
    let temp = tempfile::tempdir().unwrap();
    let pack = write_installed_pack(temp.path(), "whisper-small", "q8_0", "q8");
    persist(temp.path(), &pack, QuantPreference::Auto).unwrap();
    fs::remove_file(config_path(temp.path())).unwrap();
    fs::create_dir_all(config_path(temp.path())).unwrap();

    let outcome = clear_detailed(temp.path()).unwrap();

    assert!(matches!(
        outcome,
        DefaultSelectionCommitOutcome::V2CommittedProjectionFailed { .. }
    ));
    assert_eq!(
        resolve(temp.path(), None).unwrap(),
        DefaultModelResolution::Unset
    );
}

#[test]
fn persist_reports_v2_committed_when_legacy_projection_fails() {
    let temp = tempfile::tempdir().unwrap();
    let pack = write_installed_pack(temp.path(), "whisper-small", "q8_0", "q8");
    fs::create_dir_all(config_path(temp.path())).unwrap();

    assert!(persist(temp.path(), &pack, QuantPreference::Auto).is_ok());
    let outcome = persist_detailed(temp.path(), &pack, QuantPreference::Auto).unwrap();

    assert!(matches!(
        outcome,
        DefaultSelectionCommitOutcome::V2CommittedProjectionFailed { .. }
    ));
    assert_eq!(
        resolve(temp.path(), None).unwrap(),
        DefaultModelResolution::Installed(pack)
    );
}

#[test]
fn legacy_migration_reuses_pointer_quant_policy_and_writes_v2() {
    let temp = tempfile::tempdir().unwrap();
    let auto_pack = write_installed_pack(temp.path(), "whisper-small", "q4_k", "q4");
    let pinned_pack = write_installed_pack(temp.path(), "whisper-small", "q8_0", "q8");
    let mut document = OpenAsrConfigDocument::default();
    document.config.default_model = Some("whisper-small".to_string());
    document.preferences.quant_preference = QuantPreference::Pinned {
        quant: "q8_0".to_string(),
    };
    save_config_document(temp.path(), &document).unwrap();
    persist_default_pack_pointer(temp.path(), &pinned_pack).unwrap();

    let migrated = migrate_legacy_to_v2(temp.path()).unwrap().unwrap();

    assert_eq!(migrated.status, ActiveModelSelectionStatus::Installed);
    assert_eq!(migrated.pull.as_deref(), Some(pinned_pack.pull.as_str()));
    assert_ne!(migrated.pull.as_deref(), Some(auto_pack.pull.as_str()));
    assert_eq!(
        resolve(temp.path(), None).unwrap(),
        DefaultModelResolution::Installed(pinned_pack)
    );
}

#[test]
fn legacy_migration_uses_catalog_alias_and_explicit_quant_tag() {
    let temp = tempfile::tempdir().unwrap();
    let pack = write_installed_pack(temp.path(), "whisper-small", "q8_0", "q8");
    let mut document = OpenAsrConfigDocument::default();
    document.config.default_model = Some("legacy-whisper:q8".to_string());
    save_config_document(temp.path(), &document).unwrap();

    let mut catalog = crate::load_embedded_signed_catalog(temp.path()).unwrap();
    catalog
        .models
        .iter_mut()
        .find(|model| model.id == "whisper-small")
        .expect("embedded catalog contains whisper-small")
        .aliases
        .push("legacy-whisper".to_string());

    let migrated = migrate_legacy_to_v2_with_catalog(temp.path(), Some(&catalog))
        .unwrap()
        .unwrap();

    assert_eq!(migrated.status, ActiveModelSelectionStatus::Installed);
    assert_eq!(migrated.model_id.as_deref(), Some("whisper-small"));
    assert_eq!(migrated.quant.as_deref(), Some("q8_0"));
    assert_eq!(migrated.pull.as_deref(), Some(pack.pull.as_str()));
}

#[test]
fn legacy_alias_without_pack_migrates_to_canonical_and_resolves_after_install() {
    let temp = tempfile::tempdir().unwrap();
    let mut document = OpenAsrConfigDocument::default();
    document.config.default_model = Some("legacy-whisper".to_string());
    crate::config::save_config_document_unlocked(temp.path(), &document).unwrap();

    let mut catalog = crate::load_embedded_signed_catalog(temp.path()).unwrap();
    catalog
        .models
        .iter_mut()
        .find(|model| model.id == "whisper-small")
        .expect("embedded catalog contains whisper-small")
        .aliases
        .push("legacy-whisper".to_string());

    let migrated = migrate_legacy_to_v2_with_catalog(temp.path(), Some(&catalog))
        .unwrap()
        .unwrap();
    assert_eq!(migrated.status, ActiveModelSelectionStatus::NotInstalled);
    assert_eq!(migrated.model_id.as_deref(), Some("whisper-small"));
    assert_eq!(migrated.pull.as_deref(), Some("whisper-small"));
    assert_eq!(migrated.quant, None);

    let pack = write_installed_pack(temp.path(), "whisper-small", "q8_0", "q8");
    assert_eq!(
        resolve_with_catalog(temp.path(), Some(&catalog)).unwrap(),
        DefaultModelResolution::Installed(pack)
    );
}

#[test]
fn legacy_alias_with_quant_without_pack_keeps_canonical_identity() {
    let temp = tempfile::tempdir().unwrap();
    let mut document = OpenAsrConfigDocument::default();
    document.config.default_model = Some("legacy-whisper:q8".to_string());
    crate::config::save_config_document_unlocked(temp.path(), &document).unwrap();

    let mut catalog = crate::load_embedded_signed_catalog(temp.path()).unwrap();
    catalog
        .models
        .iter_mut()
        .find(|model| model.id == "whisper-small")
        .expect("embedded catalog contains whisper-small")
        .aliases
        .push("legacy-whisper".to_string());

    let migrated = migrate_legacy_to_v2_with_catalog(temp.path(), Some(&catalog))
        .unwrap()
        .unwrap();
    assert_eq!(migrated.status, ActiveModelSelectionStatus::NotInstalled);
    assert_eq!(migrated.model_id.as_deref(), Some("whisper-small"));
    assert_eq!(migrated.pull.as_deref(), Some("whisper-small:q8_0"));
    assert_eq!(migrated.quant.as_deref(), Some("q8_0"));

    let pack = write_installed_pack(temp.path(), "whisper-small", "q8_0", "q8");
    assert_eq!(
        resolve_with_catalog(temp.path(), Some(&catalog)).unwrap(),
        DefaultModelResolution::Installed(pack)
    );
}

#[test]
fn corrupt_recovery_evidence_copy_failure_keeps_original_v2_in_place() {
    let temp = tempfile::tempdir().unwrap();
    let path = active_model_selection_v2_path(temp.path());
    let original = br#"{"schema_version":999,"checksum":"bad"}"#;
    fs::write(&path, original).unwrap();
    set_recovery_fail_after_evidence(true);

    let error = recover_corrupt_v2(temp.path()).unwrap_err();
    set_recovery_fail_after_evidence(false);

    assert!(error.to_string().contains("evidence copy"));
    assert_eq!(fs::read(&path).unwrap(), original);
    assert_eq!(
        fs::read_dir(temp.path())
            .unwrap()
            .filter_map(Result::ok)
            .filter(|entry| entry
                .file_name()
                .to_string_lossy()
                .starts_with("default-selection.corrupt."))
            .count(),
        1
    );
}

#[test]
fn legacy_migration_preserves_not_installed_config_intent() {
    let temp = tempfile::tempdir().unwrap();
    write_config_default_model(temp.path(), "missing-model");

    let migrated = migrate_legacy_to_v2(temp.path()).unwrap().unwrap();

    assert_eq!(migrated.status, ActiveModelSelectionStatus::NotInstalled);
    assert_eq!(migrated.model_id.as_deref(), Some("missing-model"));
    assert_eq!(
        resolve(temp.path(), None).unwrap(),
        DefaultModelResolution::NotInstalled("missing-model".to_string())
    );
}

#[test]
fn corrupt_v2_recovery_preserves_evidence_and_does_not_use_legacy() {
    let temp = tempfile::tempdir().unwrap();
    write_config_default_model(temp.path(), "legacy-model");
    let path = active_model_selection_v2_path(temp.path());
    fs::write(&path, br#"{"schema_version":999,"checksum":"bad"}"#).unwrap();

    let recovered = recover_corrupt_v2(temp.path()).unwrap();

    assert_eq!(recovered.status, ActiveModelSelectionStatus::Unset);
    assert_eq!(
        resolve(temp.path(), None).unwrap(),
        DefaultModelResolution::Unset
    );
    assert_eq!(
        fs::read_dir(temp.path())
            .unwrap()
            .filter_map(Result::ok)
            .filter(|entry| entry
                .file_name()
                .to_string_lossy()
                .starts_with("default-selection.corrupt."))
            .count(),
        1
    );
}

#[cfg(unix)]
#[test]
fn corrupt_recovery_evidence_is_owner_only() {
    use std::os::unix::fs::PermissionsExt;

    let temp = tempfile::tempdir().unwrap();
    let path = active_model_selection_v2_path(temp.path());
    fs::write(&path, br#"{"schema_version":999,"checksum":"bad"}"#).unwrap();

    let outcome = recover_corrupt_v2_detailed(temp.path()).unwrap();
    let evidence_path = match outcome {
        DefaultSelectionRecoveryOutcome::Committed { evidence_path, .. }
        | DefaultSelectionRecoveryOutcome::ProjectionFailed { evidence_path, .. } => evidence_path,
    };
    let mode = fs::metadata(evidence_path).unwrap().permissions().mode();
    assert_eq!(mode & 0o777, 0o600);
}

#[test]
fn corrupt_recovery_reports_projection_failure_and_repair() {
    let temp = tempfile::tempdir().unwrap();
    let path = active_model_selection_v2_path(temp.path());
    fs::write(&path, br#"{"schema_version":999,"checksum":"bad"}"#).unwrap();
    fs::create_dir_all(config_path(temp.path())).unwrap();

    let outcome = recover_corrupt_v2_detailed(temp.path()).unwrap();
    assert!(matches!(
        outcome,
        DefaultSelectionRecoveryOutcome::ProjectionFailed { .. }
    ));
    assert_eq!(
        resolve(temp.path(), None).unwrap(),
        DefaultModelResolution::Unset
    );

    fs::remove_dir(config_path(temp.path())).unwrap();
    assert_eq!(
        repair_compat_projection(temp.path()).unwrap(),
        DefaultSelectionCommitOutcome::V2Committed
    );
}

#[test]
fn v2_exact_checksum_mismatch_resolves_not_installed() {
    let temp = tempfile::tempdir().unwrap();
    let record = ActiveModelSelectionV2 {
        schema_version: ACTIVE_MODEL_SELECTION_V2_SCHEMA_VERSION,
        selection_generation: 0,
        status: ActiveModelSelectionStatus::NotInstalled,
        pull: Some("whisper-small:q8".to_string()),
        model_id: Some("whisper-small".to_string()),
        quant: Some("q8_0".to_string()),
        architecture_id: Some("whisper".to_string()),
        expected_pack: Some(ExpectedPackIdentityV2 {
            sha256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_string(),
            size_bytes: 1,
        }),
        quant_preference: QuantPreference::Auto,
        execution_intent: "auto".to_string(),
        checksum: String::new(),
    };
    persist_v2_record(temp.path(), record).unwrap();

    assert_eq!(
        resolve(temp.path(), None).unwrap(),
        DefaultModelResolution::NotInstalled("whisper-small".to_string())
    );
}

#[test]
fn v2_execution_intent_wire_round_trips_every_selector_shape() {
    use crate::device::{
        execution_policy::{AcceleratedDeviceConstraint, ExecutionIntent},
        execution_route::{
            ExactDeviceSelector, ExecutionHardwareVendor, ExecutionProvider, PhysicalResourceKey,
        },
    };

    let intents = vec![
        ExecutionIntent::Auto,
        ExecutionIntent::CpuOnly,
        ExecutionIntent::AcceleratedOnly,
        ExecutionIntent::ConstrainedAcceleratedOnly(AcceleratedDeviceConstraint::Provider(
            ExecutionProvider::Cuda,
        )),
        ExecutionIntent::ConstrainedAcceleratedOnly(AcceleratedDeviceConstraint::HardwareVendor(
            ExecutionHardwareVendor::Amd,
        )),
        ExecutionIntent::Exact(ExactDeviceSelector::PhysicalKey(
            PhysicalResourceKey::new("0000:c1:00.0").unwrap(),
        )),
        ExecutionIntent::Exact(ExactDeviceSelector::StableId {
            provider: Some(ExecutionProvider::Vulkan),
            stable_id: "Vulkan device/0: discrete".to_string(),
        }),
        ExecutionIntent::Exact(ExactDeviceSelector::StableId {
            provider: None,
            stable_id: "provider-local 设备".to_string(),
        }),
    ];

    for intent in intents {
        let wire = execution_intent_to_v2_wire(&intent);
        assert!(is_logical_intent(&wire), "wire must stay logical: {wire}");
        assert_eq!(execution_intent_from_v2_wire(&wire).unwrap(), intent);
    }
}

#[test]
fn v2_execution_intent_wire_rejects_malformed_exact_identifiers() {
    for wire in [
        "exact_physical:",
        "exact_physical:0",
        "exact_stable:cuda:",
        "exact_stable:not-a-provider:43505530",
        "unknown_intent",
    ] {
        assert!(
            execution_intent_from_v2_wire(wire).is_err(),
            "malformed wire unexpectedly accepted: {wire}"
        );
    }
}
