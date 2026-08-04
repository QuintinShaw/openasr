use std::fs;
use std::path::Path;

use super::*;
use crate::testing::{TinyGgufFixtureSpec, write_tiny_gguf_runtime_source};
use crate::{OpenAsrConfigDocument, config_path};
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
