use std::fs;
use std::path::Path;

use super::*;
use crate::testing::{TinyGgufFixtureSpec, write_tiny_gguf_runtime_source};
use crate::{OpenAsrConfigDocument, config_path};

/// Writes an `installed.json` plus a real backing pack file for `model_id`/
/// `quant`. `list_installed_packs` re-validates every on-disk pack on each
/// lookup (`installed_pack_matches_quant_dir` in pull.rs checks the file
/// exists, its size matches `size_bytes`, and it passes
/// `validate_native_runtime_model_pack_contract`) -- a bare `installed.json`
/// with no backing file is silently dropped, not "installed". Mirror the
/// graph-complete whisper fixture `openasr-server`'s tests use for the same
/// reason (the bare non-graph spec omits required whisper runtime keys and
/// fails contract validation).
fn write_installed_pack(home: &Path, model_id: &str, quant: &str, suffix: &str) -> InstalledPack {
    let filename = format!("{model_id}-{quant}.oasr");
    let dir = home.join("models").join(model_id).join(quant);
    fs::create_dir_all(&dir).expect("create installed pack dir");
    let path = dir.join(&filename);
    let spec = TinyGgufFixtureSpec::whisper_oasr_v1_encoder_graph_one_layer(model_id);
    write_tiny_gguf_runtime_source(&path, &spec).expect("write tiny gguf runtime source");
    let size_bytes = fs::metadata(&path)
        .expect("stat installed pack fixture")
        .len();
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
        sha256: "a".repeat(64),
        size_bytes,
        installed_at_unix_seconds: 1,
        source: None,
    };
    fs::write(
        dir.join("installed.json"),
        serde_json::to_string_pretty(&pack).expect("serialize installed pack"),
    )
    .expect("write installed pack metadata");
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
