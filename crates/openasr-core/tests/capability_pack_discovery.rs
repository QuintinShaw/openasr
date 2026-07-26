//! Voice ID's capability probe against a real, isolated `OPENASR_HOME`.
//!
//! `diarize::vad_diarization_available()` is the gate the whole speaker-identity
//! layer hangs off: when it answers `false` the identity stage is skipped and a
//! transcript comes back with segment-level speakers only -- no error, no
//! warning. That made a content-addressed install look like "diarization is just
//! bad" rather than "diarization never ran".
//!
//! This drives the real public function in a dedicated test process, so the
//! `OPENASR_HOME` override is genuine rather than a reconstruction of the lookup.

use std::{fs, path::Path};

use sha2::{Digest, Sha256};

/// Install a capability pack the way a pull does: an immutable object plus the
/// ref that names it. No `models/<model_id>/` directory is ever created -- that
/// absence is the whole point.
fn install_content_addressed(home: &Path, model_id: &str, quant: &str, bytes: &[u8]) {
    let models = home.join("models");
    let digest = format!("{:x}", Sha256::digest(bytes));
    let object = models.join("objects/sha256").join(&digest).join("content");
    fs::create_dir_all(object.parent().expect("object parent")).unwrap();
    fs::write(&object, bytes).unwrap();

    let record = serde_json::json!({
        "model_id": model_id,
        "display_name": model_id,
        "quant": quant,
        "suffix": quant,
        "pull": format!("{model_id}:{quant}"),
        "filename": format!("{model_id}-{quant}.oasr"),
        "path": object,
        "url": "https://example.invalid/pack.oasr",
        "hf_revision": "test",
        "sha256": digest,
        "size_bytes": bytes.len(),
        "installed_at_unix_seconds": 1,
        "source": serde_json::Value::Null,
    });
    let ref_path = models
        .join("refs")
        .join(model_id)
        .join(format!("{quant}.json"));
    fs::create_dir_all(ref_path.parent().expect("ref parent")).unwrap();
    fs::write(&ref_path, serde_json::to_string(&record).unwrap()).unwrap();
}

/// Single test function on purpose: `OPENASR_HOME` is process-global, so the
/// stages have to run in a known order inside one process.
#[test]
fn voice_id_capability_probe_sees_a_content_addressed_embedder_pack() {
    let home = tempfile::tempdir().unwrap();
    // SAFETY: this test binary is the sole owner of its process environment, and
    // the variable is set once before any probe runs.
    unsafe { std::env::set_var("OPENASR_HOME", home.path()) };

    assert!(
        !openasr_core::diarize::vad_diarization_available(),
        "an empty home must report the embedder as absent"
    );

    install_content_addressed(
        home.path(),
        "redimnet2-b6-cn",
        "fp16",
        b"GGUF-redimnet-pack",
    );

    // The shape measured on a real machine: no `models/*redimnet*` directory
    // exists, only a ref.
    let models = home.path().join("models");
    let redimnet_dirs = fs::read_dir(&models)
        .unwrap()
        .flatten()
        .filter(|entry| entry.file_name().to_string_lossy().contains("redimnet"))
        .count();
    assert_eq!(
        redimnet_dirs, 0,
        "a content-addressed install creates no per-model directory"
    );
    assert!(models.join("refs/redimnet2-b6-cn").is_dir());

    assert!(
        openasr_core::diarize::vad_diarization_available(),
        "Voice ID must be available from a content-addressed embedder install; \
         answering false here is the silent no-op that reported a 4-person \
         meeting as 11-17 speakers"
    );

    // A legacy install of the same pack must keep working too, so an upgrading
    // home does not lose the capability between install and migration.
    let legacy_home = tempfile::tempdir().unwrap();
    let legacy_dir = legacy_home.path().join("models/redimnet2-b6-cn/fp16");
    fs::create_dir_all(&legacy_dir).unwrap();
    fs::write(legacy_dir.join("redimnet2-b6-cn-fp16.oasr"), b"GGUF-legacy").unwrap();
    unsafe { std::env::set_var("OPENASR_HOME", legacy_home.path()) };
    assert!(
        openasr_core::diarize::vad_diarization_available(),
        "the legacy layout must stay discoverable until migration has run"
    );

    unsafe { std::env::remove_var("OPENASR_HOME") };
}
