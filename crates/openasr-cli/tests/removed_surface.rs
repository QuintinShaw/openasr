use assert_cmd::Command;
use predicates::prelude::*;
use std::{
    path::PathBuf,
    sync::{
        OnceLock,
        atomic::{AtomicUsize, Ordering},
    },
};
use tempfile::TempDir;

fn openasr() -> Command {
    let mut command = Command::cargo_bin("openasr").expect("openasr binary");
    command.env("OPENASR_HOME", isolated_openasr_home());
    command
}

fn isolated_openasr_home() -> PathBuf {
    static ROOT: OnceLock<PathBuf> = OnceLock::new();
    static COUNTER: AtomicUsize = AtomicUsize::new(0);

    let root = ROOT.get_or_init(|| {
        let path = std::env::temp_dir().join(format!(
            "openasr-removed-surface-tests-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&path).expect("create shared test root");
        path
    });
    let path = root.join(format!("case-{}", COUNTER.fetch_add(1, Ordering::Relaxed)));
    let _ = std::fs::remove_dir_all(&path);
    std::fs::create_dir_all(&path).expect("create isolated OPENASR_HOME");
    path
}

fn temp_home() -> TempDir {
    tempfile::tempdir().expect("temporary OPENASR_HOME")
}

fn temp_input_wav() -> tempfile::NamedTempFile {
    let file = tempfile::Builder::new()
        .prefix("openasr-test-")
        .suffix(".wav")
        .tempfile()
        .expect("temporary wav");
    std::fs::write(file.path(), b"not a real wav").expect("write sample");
    file
}

#[test]
fn clean_legacy_runtime_cache_subcommand_is_removed() {
    openasr()
        .args(["clean-legacy-runtime-cache", "--yes"])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "unrecognized subcommand 'clean-legacy-runtime-cache'",
        ));
}

#[test]
fn clean_legacy_cache_subcommand_is_removed() {
    openasr()
        .args(["clean-legacy-cache", "--yes"])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "unrecognized subcommand 'clean-legacy-cache'",
        ));
}

#[test]
fn batch_subcommand_is_removed() {
    openasr()
        .args(["batch", "/tmp/in", "--output", "/tmp/out"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("unrecognized subcommand 'batch'"));
}

#[test]
fn benchmark_subcommand_is_removed() {
    openasr()
        .args(["benchmark", "fixtures/jfk.wav"])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "unrecognized subcommand 'benchmark'",
        ));
}

#[test]
fn models_subcommand_group_is_removed() {
    openasr()
        .args(["models", "list"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("unrecognized subcommand 'models'"));
}

#[test]
fn runtime_subcommand_is_removed() {
    openasr()
        .args(["runtime", "list"])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "unrecognized subcommand 'runtime'",
        ));
}

#[test]
fn model_cache_subcommand_is_removed() {
    openasr()
        .args(["model-cache", "inspect"])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "unrecognized subcommand 'model-cache'",
        ));
}

#[test]
fn model_pack_pack_subcommand_is_fail_closed() {
    openasr()
        .args(["model-pack", "pack", "in.openasr", "out.oasr"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("unrecognized subcommand 'pack'"));
}

#[test]
fn model_pack_quantize_subcommand_is_fail_closed() {
    openasr()
        .args(["model-pack", "quantize", "in.openasr", "out.openasr"])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "unrecognized subcommand 'quantize'",
        ));
}

#[test]
fn config_rejects_removed_backend_as_default() {
    let home = temp_home();
    openasr()
        .env("OPENASR_HOME", home.path())
        .args(["config", "set", "default_backend", "whisper.cpp"])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "Unsupported backend 'whisper.cpp'",
        ));
}

#[test]
fn config_accepts_native_as_persisted_default_backend() {
    // native is the default backend now and a valid persisted value.
    let home = temp_home();
    openasr()
        .env("OPENASR_HOME", home.path())
        .args(["config", "set", "default_backend", "native"])
        .assert()
        .success();
}

#[test]
fn transcribe_rejects_saved_default_backend_when_legacy_value_is_present() {
    let home = temp_home();
    std::fs::write(
        home.path().join("config.json"),
        r#"{
  "default_model": "whisper-large-v3-turbo",
  "default_backend": "whisper.cpp",
  "media": {}
}
"#,
    )
    .expect("write config");

    let input = temp_input_wav();
    openasr()
        .env("OPENASR_HOME", home.path())
        .args([
            "transcribe",
            &input.path().display().to_string(),
            "--model",
            "whisper-large-v3-turbo",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "Saved default backend 'whisper.cpp' is retired and no longer executable.",
        ));
}

#[test]
fn transcribe_rejects_unknown_saved_default_backend_value() {
    let home = temp_home();
    std::fs::write(
        home.path().join("config.json"),
        r#"{
  "default_model": "whisper-large-v3-turbo",
  "default_backend": "mokk",
  "media": {}
}
"#,
    )
    .expect("write config");

    let input = temp_input_wav();
    openasr()
        .env("OPENASR_HOME", home.path())
        .args([
            "transcribe",
            &input.path().display().to_string(),
            "--model",
            "whisper-large-v3-turbo",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("Unsupported backend 'mokk'"));
}
