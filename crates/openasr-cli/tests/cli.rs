use assert_cmd::Command;
use openasr_core::api::backend::transcribe_with_mock_backend;
use openasr_core::testing::{
    TinyGgufFixtureSpec, write_reserved_oasr_container, write_tiny_gguf_runtime_source,
};
use openasr_core::{ResponseFormat, TranscriptionRequest, render_transcription};
use predicates::prelude::*;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    path::{Path, PathBuf},
    sync::{
        OnceLock,
        atomic::{AtomicUsize, Ordering},
    },
};
use tempfile::TempDir;

fn openasr() -> Command {
    let mut command = Command::cargo_bin("openasr").expect("openasr binary");
    command.env("OPENASR_HOME", isolated_openasr_home());
    clear_inherited_openasr_env(&mut command);
    command
}

fn openasr_with_home(home: &Path) -> Command {
    let mut command = Command::cargo_bin("openasr").expect("openasr binary");
    command.env("OPENASR_HOME", home);
    clear_inherited_openasr_env(&mut command);
    command
}

/// Keeps tests deterministic regardless of the developer's shell: the clap
/// `env` fallbacks (OPENASR_MODEL/OPENASR_ADDR) and consent env switches must
/// not bleed in from the parent process.
fn clear_inherited_openasr_env(command: &mut Command) {
    for key in [
        "OPENASR_MODEL",
        "OPENASR_ADDR",
        "OPENASR_ASSUME_YES",
        "OPENASR_OFFLINE",
    ] {
        command.env_remove(key);
    }
}

fn isolated_openasr_home() -> PathBuf {
    static ROOT: OnceLock<PathBuf> = OnceLock::new();
    static COUNTER: AtomicUsize = AtomicUsize::new(0);

    let root = ROOT.get_or_init(|| {
        let path = std::env::temp_dir().join(format!("openasr-cli-tests-{}", std::process::id()));
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

fn sample_wav_fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/jfk.wav")
        .canonicalize()
        .expect("sample wav fixture path must exist")
}

fn expected_mock_rendered_transcription(
    model: &str,
    file_name: &str,
    format: ResponseFormat,
) -> String {
    let transcription = transcribe_with_mock_backend(
        TranscriptionRequest::new(PathBuf::from(file_name), model)
            .with_display_file_name(Some(file_name.to_string())),
    )
    .expect("mock transcription");
    render_transcription(&transcription, format).expect("render transcription")
}

fn write_gguf_package(path: &std::path::Path) {
    let spec = TinyGgufFixtureSpec::new(Default::default());
    write_tiny_gguf_runtime_source(path, &spec).expect("write mock gguf runtime source");
}

fn write_whisper_oasr_v1_fixture(path: &std::path::Path, model_id: &str) {
    let spec = TinyGgufFixtureSpec::whisper_oasr_v1_graph_ready_for_runtime_fail_closed(model_id);
    write_tiny_gguf_runtime_source(path, &spec).expect("write whisper gguf runtime source");
}

fn write_reserved_oasr_package(path: &std::path::Path) {
    write_reserved_oasr_container(path).expect("write reserved oasr package fixture");
}

fn catalog_model_fixture(
    id: &str,
    display_name: &str,
    family: &str,
    aliases: Vec<&str>,
    size: &str,
    license: &str,
    license_url: &str,
    license_class: &str,
    sha256: &str,
    size_bytes: u64,
) -> Value {
    const REVISION: &str = "0123456789abcdef0123456789abcdef01234567";
    let pull_alias = aliases.first().map(|value| (*value).to_string());
    let aliases = aliases.into_iter().map(str::to_string).collect::<Vec<_>>();
    json!({
      "id": id,
      "display_name": display_name,
      "family": family,
      "aliases": aliases,
      "pull_alias": pull_alias,
      "size": size,
      "languages": ["en"],
      "vendor": "Useful Sensors",
      "license": license,
      "license_url": license_url,
      "license_class": license_class,
      "hf_repo": format!("OpenASR/{id}"),
      "hf_revision": REVISION,
      "public": true,
      "min_cli_version": "0.1.0",
      "recommended_quant": "q8_0",
      "pull_recommended": format!("{id}:q8"),
      "quants": [
        {
          "quant": "q8_0",
          "suffix": "q8",
          "pull": format!("{id}:q8"),
          "filename": format!("{id}-q8_0.oasr"),
          "url": format!("https://huggingface.co/OpenASR/{id}/resolve/{REVISION}/{id}-q8_0.oasr"),
          "sha256": sha256,
          "size_bytes": size_bytes,
          "recommended": true
        }
      ]
    })
}

fn write_catalog_models_fixture(path: &std::path::Path, models: Vec<Value>) {
    let catalog = json!({
      "schema_version": 1,
      "generated_at": "2026-05-31T00:00:00Z",
      "catalog_url": "https://catalog.openasr.org/v1/catalog.json",
      "models": models
    });
    let json = serde_json::to_string_pretty(&catalog).expect("serialize catalog fixture");
    std::fs::write(path, json).expect("write catalog fixture");
}

fn write_catalog_fixture(path: &std::path::Path, sha256: &str, size_bytes: u64) {
    write_catalog_models_fixture(
        path,
        vec![catalog_model_fixture(
            "moonshine-tiny",
            "Moonshine Tiny",
            "moonshine",
            vec!["moonshine"],
            "tiny",
            "MIT",
            "https://huggingface.co/UsefulSensors/moonshine-tiny",
            "permissive",
            sha256,
            size_bytes,
        )],
    );
}

fn write_unsupported_catalog_schema_fixture(path: &std::path::Path) {
    let catalog = json!({
      "schema_version": 99,
      "generated_at": "2026-05-31T00:00:00Z",
      "catalog_url": "https://catalog.openasr.org/v1/catalog.json",
      "models": []
    });
    let json = serde_json::to_string_pretty(&catalog).expect("serialize catalog fixture");
    std::fs::write(path, json).expect("write catalog fixture");
}

fn write_ambiguous_moonshine_catalog_fixture(
    path: &std::path::Path,
    tiny_sha256: &str,
    tiny_size_bytes: u64,
) {
    write_catalog_models_fixture(
        path,
        vec![
            catalog_model_fixture(
                "moonshine-tiny",
                "Moonshine Tiny",
                "moonshine",
                vec!["moonshine"],
                "tiny",
                "MIT",
                "https://huggingface.co/UsefulSensors/moonshine-tiny",
                "permissive",
                tiny_sha256,
                tiny_size_bytes,
            ),
            catalog_model_fixture(
                "moonshine-base",
                "Moonshine Base",
                "moonshine",
                vec!["moonshine"],
                "base",
                "MIT",
                "https://huggingface.co/UsefulSensors/moonshine-base",
                "permissive",
                "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
                1,
            ),
        ],
    );
}

fn write_gated_catalog_fixture(path: &std::path::Path) {
    write_catalog_models_fixture(
        path,
        vec![catalog_model_fixture(
            "parakeet-ctc-0.6b",
            "Parakeet CTC 0.6B",
            "parakeet-ctc",
            vec!["parakeet"],
            "0.6b",
            "NVIDIA model license",
            "https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/parakeet-ctc-0_6b",
            "gated",
            "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
            1,
        )],
    );
}

#[test]
fn help_does_not_list_removed_legacy_backends() {
    // --backend is hidden from the default help now (native is the default; mock
    // is a testing-only affordance), so the help surfaces no backend names at all
    // -- least of all the removed legacy ones. The advanced longform/VAD knobs are
    // hidden too, keeping the default help newcomer-friendly.
    openasr()
        .args(["transcribe", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("sensevoice-onnx").not())
        .stdout(predicate::str::contains("whisper.cpp").not())
        .stdout(predicate::str::contains("vad-threshold-db").not());
}

#[test]
fn serve_help_documents_local_default_and_remote_security() {
    openasr()
        .args(["serve", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "Defaults to local HTTP on 127.0.0.1",
        ))
        .stdout(predicate::str::contains("HTTPS/WSS"))
        .stdout(predicate::str::contains("--tls-self-signed"))
        .stdout(predicate::str::contains("--pairing-admin-token-env"));
}

#[test]
fn model_pack_validate_accepts_oasr_file() {
    let temp = tempfile::tempdir().expect("tempdir");
    let package = temp.path().join("fixture-model.oasr");
    write_gguf_package(&package);

    openasr()
        .args(["verify", &package.display().to_string()])
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "Validated local ggml model package",
        ))
        .stdout(predicate::str::contains(
            "No downloads or inference were performed.",
        ));
}

#[test]
fn model_pack_validate_accepts_oasr_extension_when_magic_is_gguf() {
    let temp = tempfile::tempdir().expect("tempdir");
    let package = temp.path().join("fixture-model.oasr");
    write_gguf_package(&package);

    openasr()
        .args(["verify", &package.display().to_string()])
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "Validated local ggml model package",
        ));
}

#[test]
fn model_pack_inspect_prints_ggml_probe_summary() {
    let temp = tempfile::tempdir().expect("tempdir");
    let package = temp.path().join("fixture-model.oasr");
    write_gguf_package(&package);

    openasr()
        .args(["show", &package.display().to_string()])
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "Format: .oasr (OpenASR native pack)",
        ))
        .stdout(predicate::str::contains("Extension hint: .oasr"))
        .stdout(predicate::str::contains("Warnings: none"));
}

#[test]
fn model_pack_validate_rejects_reserved_oasr_magic() {
    let temp = tempfile::tempdir().expect("tempdir");
    let package = temp.path().join("fixture-model.oasr");
    write_reserved_oasr_package(&package);

    openasr()
        .args(["verify", &package.display().to_string()])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "reserved non-GGUF container magic",
        ));
}

#[test]
fn model_pack_validate_rejects_remote_looking_path() {
    openasr()
        .args(["verify", "https://example.invalid/model.gguf"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("remote URLs are not supported"));
}

#[test]
fn model_pack_validate_rejects_missing_path() {
    let missing =
        std::env::temp_dir().join(format!("missing-model-pack-{}.oasr", std::process::id()));
    let _ = std::fs::remove_dir_all(&missing);

    openasr()
        .args(["verify", &missing.display().to_string()])
        .assert()
        .failure()
        .stderr(predicate::str::contains("does not exist"));
}

#[test]
fn model_pack_validate_rejects_parent_alias_path() {
    let temp = tempfile::tempdir().expect("tempdir");
    let package = temp.path().join("fixture-model.oasr");
    write_gguf_package(&package);
    let parent_alias = format!("{}/..", package.display());

    openasr()
        .args(["verify", &parent_alias])
        .assert()
        .failure()
        .stderr(
            predicate::str::contains("Model package path").and(
                predicate::str::contains("does not exist")
                    .or(predicate::str::contains("must be a local .oasr file")),
            ),
        );
}

#[test]
fn model_pack_validate_rejects_directory_path() {
    let temp = tempfile::tempdir().expect("tempdir");
    let dir_path = temp.path().join("fixture-model.openasr");
    std::fs::create_dir_all(&dir_path).expect("create directory");

    openasr()
        .args(["verify", &dir_path.display().to_string()])
        .assert()
        .failure()
        .stderr(predicate::str::contains("must be a local .oasr file"));
}

#[test]
fn model_pack_validate_rejects_unknown_magic() {
    let temp = tempfile::tempdir().unwrap();
    let package = temp.path().join("fixture-model.oasr");
    std::fs::write(&package, b"ABCDfixture").expect("write unknown magic fixture");

    openasr()
        .args(["verify", &package.display().to_string()])
        .assert()
        .failure()
        .stderr(predicate::str::contains("unknown magic bytes"));
}

// --- New local-import subcommands (parakeet-ctc / wav2vec2-ctc / moonshine) ---
//
// These cover the CLI surface (parser wiring, required flags, quantization
// default) and the `.oasr`-only output contract at the CLI boundary, without
// needing a multi-GB HF source on disk: the suffix gate runs before any source
// read, so a non-.oasr output fails fast. The importer's heavy round-trip is
// covered by the (`#[ignore]`d) core round-trip tests + the bench suite.

#[test]
fn import_parakeet_ctc_local_help_lists_flags_and_quant_default() {
    openasr()
        .args(["model-pack", "import", "parakeet-ctc", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--package-id"))
        .stdout(predicate::str::contains("--quantization"))
        .stdout(predicate::str::contains("[default: fp16]"))
        .stdout(predicate::str::contains("q4-k"));
}

#[test]
fn import_wav2vec2_ctc_local_help_lists_flags_and_quant_default() {
    openasr()
        .args(["model-pack", "import", "wav2vec2-ctc", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--package-id"))
        .stdout(predicate::str::contains("--quantization"))
        .stdout(predicate::str::contains("[default: q4-k]"));
}

#[test]
fn import_moonshine_local_help_lists_flags_and_quant_default() {
    openasr()
        .args(["model-pack", "import", "moonshine", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--package-id"))
        .stdout(predicate::str::contains("--quantization"))
        .stdout(predicate::str::contains("[default: fp16]"));
}

#[test]
fn import_whisper_local_rejects_non_oasr_output() {
    let temp = tempfile::tempdir().expect("tempdir");
    let source = temp.path().join("nonexistent-src");
    let output = temp.path().join("model.gguf");

    openasr()
        .args([
            "model-pack",
            "import",
            "whisper",
            &source.display().to_string(),
            &output.display().to_string(),
            "--package-id",
            "whisper-small",
            "--source-revision",
            "main",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("must end with .oasr"));
}

#[test]
fn import_qwen_local_rejects_non_oasr_output() {
    let temp = tempfile::tempdir().expect("tempdir");
    let source = temp.path().join("nonexistent-src");
    let output = temp.path().join("model.gguf");

    openasr()
        .args([
            "model-pack",
            "import",
            "qwen",
            &source.display().to_string(),
            &output.display().to_string(),
            "--package-id",
            "qwen3-asr-0.6b",
            "--source-revision",
            "main",
            "--license-source",
            "https://example.com/license",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("must end with .oasr"));
}

#[test]
fn import_cohere_local_rejects_non_oasr_output() {
    let temp = tempfile::tempdir().expect("tempdir");
    let source = temp.path().join("nonexistent-src");
    let output = temp.path().join("model.gguf");

    openasr()
        .args([
            "model-pack",
            "import",
            "cohere",
            &source.display().to_string(),
            &output.display().to_string(),
            "--package-id",
            "cohere-transcribe-03-2026",
            "--source-revision",
            "2026-03",
            "--license-source",
            "https://example.com/license",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("must end with .oasr"));
}

#[test]
fn import_parakeet_ctc_local_requires_package_id() {
    let temp = tempfile::tempdir().expect("tempdir");
    let source = temp.path().join("src");
    let output = temp.path().join("out.oasr");

    openasr()
        .args([
            "model-pack",
            "import",
            "parakeet-ctc",
            &source.display().to_string(),
            &output.display().to_string(),
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("--package-id"));
}

#[test]
fn import_parakeet_ctc_local_rejects_non_oasr_output() {
    let temp = tempfile::tempdir().expect("tempdir");
    let source = temp.path().join("nonexistent-src");
    let output = temp.path().join("model.gguf");

    openasr()
        .args([
            "model-pack",
            "import",
            "parakeet-ctc",
            &source.display().to_string(),
            &output.display().to_string(),
            "--package-id",
            "parakeet-ctc-0.6b",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("must end with .oasr"));
}

#[test]
fn import_wav2vec2_ctc_local_rejects_non_oasr_output() {
    let temp = tempfile::tempdir().expect("tempdir");
    let source = temp.path().join("nonexistent-src");
    let output = temp.path().join("model.gguf");

    openasr()
        .args([
            "model-pack",
            "import",
            "wav2vec2-ctc",
            &source.display().to_string(),
            &output.display().to_string(),
            "--package-id",
            "wav2vec2-base-960h",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("must end with .oasr"));
}

#[test]
fn import_moonshine_local_rejects_non_oasr_output() {
    let temp = tempfile::tempdir().expect("tempdir");
    let source = temp.path().join("nonexistent-src");
    let output = temp.path().join("model.gguf");

    openasr()
        .args([
            "model-pack",
            "import",
            "moonshine",
            &source.display().to_string(),
            &output.display().to_string(),
            "--package-id",
            "moonshine-tiny",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("must end with .oasr"));
}

#[test]
fn transcribe_mock_still_works() {
    let input = temp_input_wav();
    openasr()
        .args([
            "transcribe",
            &input.path().display().to_string(),
            "--backend",
            "mock",
            "--model",
            "whisper-large-v3-turbo",
            "--format",
            "text",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("OpenASR mock transcription"));
}

#[test]
fn transcribe_mock_formats_match_core_renderers() {
    let input = sample_wav_fixture_path();
    for format in [
        ResponseFormat::Text,
        ResponseFormat::Json,
        ResponseFormat::VerboseJson,
        ResponseFormat::Srt,
        ResponseFormat::Vtt,
        ResponseFormat::Markdown,
    ] {
        let expected =
            expected_mock_rendered_transcription("whisper-large-v3-turbo", "jfk.wav", format);
        let assert = openasr()
            .args([
                "transcribe",
                &input.display().to_string(),
                "--backend",
                "mock",
                "--model",
                "whisper-large-v3-turbo",
                "--format",
                format.as_str(),
            ])
            .assert()
            .success();
        let output = String::from_utf8(assert.get_output().stdout.clone()).expect("utf8 stdout");
        assert_eq!(
            output,
            expected,
            "unexpected CLI output for {}",
            format.as_str()
        );
    }
}

#[test]
fn transcribe_native_requires_local_model_pack_path() {
    let input = temp_input_wav();
    openasr()
        .args([
            "transcribe",
            &input.path().display().to_string(),
            "--backend",
            "native",
            "--model",
            "whisper-large-v3-turbo",
            "--format",
            "text",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("is not installed"));
}

#[test]
fn transcribe_native_without_model_uses_runtime_auto_model_selection() {
    let input = sample_wav_fixture_path();
    let temp = tempfile::tempdir().unwrap();
    let pack_root = temp.path().join("whisper-runtime.oasr");
    write_whisper_oasr_v1_fixture(&pack_root, "whisper-runtime");

    openasr()
        .args([
            "transcribe",
            &input.display().to_string(),
            "--backend",
            "native",
            "--model-pack",
            &pack_root.display().to_string(),
            "--format",
            "text",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("Native ASR Core"))
        .stderr(predicate::str::contains("fail-closed"))
        .stderr(predicate::str::contains("requires --model to match local source id").not());
}

#[test]
fn transcribe_rejects_model_pack_with_mock_backend() {
    let input = temp_input_wav();
    let temp = tempfile::tempdir().unwrap();
    let pack_root = temp.path().join("native-pack.oasr");
    write_gguf_package(&pack_root);

    // Native is the default now, so the "--model-pack needs native" rejection is
    // exercised by forcing the mock backend.
    openasr()
        .args([
            "transcribe",
            &input.path().display().to_string(),
            "--backend",
            "mock",
            "--model-pack",
            &pack_root.display().to_string(),
            "--format",
            "text",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "--model-pack is only supported with --backend native",
        ));
}

#[test]
fn transcribe_native_fails_closed_when_fixture_lacks_tokenizer_kv() {
    let input = sample_wav_fixture_path();
    let temp = tempfile::tempdir().unwrap();
    let pack_root = temp.path().join("whisper-runtime.oasr");
    write_whisper_oasr_v1_fixture(&pack_root, "whisper-runtime");

    openasr()
        .args([
            "transcribe",
            &input.display().to_string(),
            "--backend",
            "native",
            "--model-pack",
            &pack_root.display().to_string(),
            "--model",
            "whisper-runtime",
            "--format",
            "text",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("Native ASR Core"))
        .stderr(predicate::str::contains("fail-closed"))
        .stderr(predicate::str::contains("ggml-family-whisper-runtime-v1"))
        .stderr(predicate::str::contains("tokenizer is missing"))
        .stderr(predicate::str::contains(
            "Whisper GGUF tokenizer is missing required key 'tokenizer.ggml.model'",
        ))
        .stderr(predicate::str::contains("could not read gguf metadata").not())
        .stderr(predicate::str::contains("missing required OASR v1 key").not())
        .stderr(predicate::str::contains("missing required GGUF metadata key").not())
        .stderr(predicate::str::contains("missing required GGUF tensor").not())
        .stderr(predicate::str::contains(".openasr").not())
        .stderr(predicate::str::contains("legacy pack").not());
}

#[test]
fn transcribe_native_rejects_model_id_mismatch_with_local_runtime_source() {
    let input = temp_input_wav();
    let temp = tempfile::tempdir().unwrap();
    let pack_root = temp.path().join("whisper-runtime.oasr");
    write_whisper_oasr_v1_fixture(&pack_root, "whisper-runtime");

    openasr()
        .args([
            "transcribe",
            &input.path().display().to_string(),
            "--backend",
            "native",
            "--model-pack",
            &pack_root.display().to_string(),
            "--model",
            // A genuinely different base id (not a quant-pin of the pack id):
            // since 07bc0f728 a `name:quant` request matches a bare local id, so
            // `whisper-runtime:typo` is no longer a mismatch. Use a distinct base
            // so the test still exercises model-id-mismatch rejection.
            "not-whisper-runtime",
            "--format",
            "text",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "does not match local runtime source model id",
        ));
}

#[test]
fn serve_native_rejects_model_id_mismatch_with_local_runtime_source() {
    let temp = tempfile::tempdir().unwrap();
    let pack_root = temp.path().join("whisper-runtime.oasr");
    write_whisper_oasr_v1_fixture(&pack_root, "whisper-runtime");

    // A `name:quant` request matches a bare local id under the bare-id
    // contract (same tolerant matcher as transcribe/server), so mismatch
    // rejection needs a genuinely different family base.
    openasr()
        .args([
            "serve",
            "--backend",
            "native",
            "--model-pack",
            &pack_root.display().to_string(),
            "--model",
            "not-whisper-runtime",
            "--addr",
            "127.0.0.1:0",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "requires --model to match local source id",
        ));
}

#[test]
fn serve_native_accepts_quant_pinned_model_ref_for_bare_local_runtime_source() {
    // Regression guard for the serve startup gate: the catalog resolves a
    // requested id to a quant-pinned ref (e.g. `whisper-tiny` ->
    // `whisper-tiny:q8_0`) while the pack's runtime id stays bare, so the gate
    // must use the tolerant bare-id matcher, not string equality -- strict
    // equality rejected every catalog-installed pack it was about to serve.
    let temp = tempfile::tempdir().unwrap();
    let pack_root = temp.path().join("whisper-runtime.oasr");
    write_whisper_oasr_v1_fixture(&pack_root, "whisper-runtime");

    let reserved = std::net::TcpListener::bind("127.0.0.1:0").expect("reserve ephemeral port");
    let addr = reserved.local_addr().expect("reserved addr").to_string();
    drop(reserved);

    let mut child = std::process::Command::new(env!("CARGO_BIN_EXE_openasr"))
        .env("OPENASR_HOME", temp.path())
        .env_remove("OPENASR_MODEL")
        .env_remove("OPENASR_ADDR")
        .args([
            "serve",
            "--backend",
            "native",
            "--model-pack",
            &pack_root.display().to_string(),
            "--model",
            "whisper-runtime:q8_0",
            "--addr",
            &addr,
        ])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("spawn openasr serve");

    let stdout = child.stdout.take().expect("piped stdout");
    let mut reader = std::io::BufReader::new(stdout);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    loop {
        use std::io::BufRead;
        let mut line = String::new();
        let bytes_read = reader.read_line(&mut line).expect("read stdout line");
        if bytes_read == 0 {
            let status = child.wait().expect("child exit status");
            let mut stderr = String::new();
            if let Some(mut handle) = child.stderr.take() {
                use std::io::Read;
                let _ = handle.read_to_string(&mut stderr);
            }
            panic!(
                "openasr serve rejected a quant-pinned ref for a bare local source id (status: {status:?}, stderr: {stderr})"
            );
        }
        if line
            .trim_end()
            .starts_with("OpenASR server listening on http://")
        {
            break;
        }
        if std::time::Instant::now() > deadline {
            let _ = child.kill();
            let _ = child.wait();
            panic!("openasr serve did not report listening within 10s");
        }
    }
    let _ = child.kill();
    let _ = child.wait();
}

/// Spawns the real `openasr serve` binary against `home` and blocks until it
/// prints its "listening on http://<addr>" line, returning the bound address.
/// Panics with the child's stderr if it exits before ever reporting ready --
/// the exact old failure mode where a fresh, model-less install's daemon
/// process died before the HTTP listener bound.
// The happy path intentionally returns the still-running child to the
// caller, which owns killing and waiting on it once its assertions are done
// (every call site does); clippy's zombie-process heuristic can't see across
// that boundary.
#[allow(clippy::zombie_processes)]
fn spawn_serve_and_wait_until_listening(home: &Path) -> (std::process::Child, String) {
    // `--addr 127.0.0.1:0` asks the OS for an ephemeral port; `serve` reports
    // back the listener's actual bound address (not the `:0` it was given
    // verbatim), so the real port is parsed straight from that banner line
    // instead of pre-reserving one ourselves (which was also a race, in
    // principle, against the reserved port being reused before `serve` binds).
    let mut command = std::process::Command::new(env!("CARGO_BIN_EXE_openasr"));
    command
        .env("OPENASR_HOME", home)
        .env_remove("OPENASR_MODEL")
        .env_remove("OPENASR_ADDR")
        .env_remove("OPENASR_ASSUME_YES")
        .env_remove("OPENASR_OFFLINE")
        .args(["serve", "--backend", "native", "--addr", "127.0.0.1:0"])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    let mut child = command.spawn().expect("spawn openasr serve");
    let stdout = child.stdout.take().expect("piped stdout");
    let mut reader = std::io::BufReader::new(stdout);

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    loop {
        use std::io::BufRead;
        let mut line = String::new();
        let bytes_read = reader.read_line(&mut line).expect("read stdout line");
        if bytes_read == 0 {
            let status = child.wait().expect("child exit status");
            let mut stderr = String::new();
            if let Some(mut handle) = child.stderr.take() {
                use std::io::Read;
                let _ = handle.read_to_string(&mut stderr);
            }
            panic!(
                "openasr serve exited before reporting it was listening (status: {status:?}, stderr: {stderr})"
            );
        }
        let trimmed = line.trim_end();
        if let Some(addr) = trimmed.strip_prefix("OpenASR server listening on http://") {
            assert_ne!(
                addr, "127.0.0.1:0",
                "serve must report the listener's real bound port, not the \
                 requested wildcard address, or every caller of this helper \
                 would try to connect to the unusable port 0"
            );
            return (child, addr.to_string());
        }
        if std::time::Instant::now() > deadline {
            let _ = child.kill();
            let _ = child.wait();
            panic!("openasr serve did not report listening within 10s");
        }
    }
}

fn raw_http_request(addr: &str, request: &[u8]) -> String {
    use std::io::{Read, Write};
    let stream = std::net::TcpStream::connect(addr).expect("connect to daemon");
    stream
        .set_read_timeout(Some(std::time::Duration::from_secs(5)))
        .unwrap();
    (&stream).write_all(request).unwrap();
    let mut response = Vec::new();
    (&stream).read_to_end(&mut response).expect("read response");
    String::from_utf8_lossy(&response).into_owned()
}

#[test]
fn serve_native_without_installed_model_starts_and_answers_health() {
    // Root-cause regression, exercised through the real `openasr` binary: a
    // fresh install with zero pulled models must still start the daemon and
    // answer /health. Before the fix, `serve` bailed with "is not installed"
    // before the HTTP listener ever bound, so the daemon process exited
    // immediately -- and desktop's health poll just watched a process that
    // was already dead until its 30s timeout gave up.
    let temp = tempfile::tempdir().unwrap();
    let (mut child, addr) = spawn_serve_and_wait_until_listening(temp.path());

    let response = raw_http_request(
        &addr,
        format!("GET /health HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\n\r\n").as_bytes(),
    );
    assert!(
        response.starts_with("HTTP/1.1 200"),
        "expected a healthy 200 response, got: {response}"
    );
    assert!(
        response.contains("\"model_installed\":false"),
        "expected /health to honestly report no model installed, got: {response}"
    );

    let _ = child.kill();
    let _ = child.wait();
}

#[test]
fn transcriptions_via_daemon_with_no_installed_model_return_clear_400() {
    // The other half of the same regression: once the daemon is up (see
    // `serve_native_without_installed_model_starts_and_answers_health`), an
    // actual transcription request with no model installed must fail closed
    // with a clear, structured client error naming the model id -- not a
    // connection error and not a 500.
    let temp = tempfile::tempdir().unwrap();
    let (mut child, addr) = spawn_serve_and_wait_until_listening(temp.path());

    let boundary = "openasr-nomodel-test-boundary";
    let mut body = format!(
        "--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"sample.wav\"\r\nContent-Type: audio/wav\r\n\r\nnot a real wav\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nqwen3-asr-0.6b\r\n--{boundary}--\r\n"
    )
    .into_bytes();
    let mut request = format!(
        "POST /v1/audio/transcriptions HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\nContent-Type: multipart/form-data; boundary={boundary}\r\nContent-Length: {}\r\n\r\n",
        body.len()
    )
    .into_bytes();
    request.append(&mut body);

    let response = raw_http_request(&addr, &request);
    assert!(
        response.starts_with("HTTP/1.1 400"),
        "expected a fail-closed 400 for an uninstalled model, got: {response}"
    );
    assert!(
        response.contains("qwen3-asr-0.6b") && response.contains("not installed"),
        "expected a clear 'model not installed' message naming the model id, got: {response}"
    );

    let _ = child.kill();
    let _ = child.wait();
}

#[test]
fn serve_rejects_model_pack_with_mock_backend() {
    // `--model-pack` is only meaningful for the native runtime. Native is the
    // default now, so the rejection is exercised by forcing `--backend mock`.
    let temp = tempfile::tempdir().unwrap();
    let pack_root = temp.path().join("native-pack.oasr");
    write_gguf_package(&pack_root);

    openasr()
        .args([
            "serve",
            "--backend",
            "mock",
            "--model-pack",
            &pack_root.display().to_string(),
            "--addr",
            "127.0.0.1:0",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "--model-pack is only supported with --backend native",
        ));
}

#[test]
fn transcribe_rejects_removed_whisper_cpp_backend_value() {
    let input = temp_input_wav();
    openasr()
        .args([
            "transcribe",
            &input.path().display().to_string(),
            "--backend",
            "whisper.cpp",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "Unsupported backend 'whisper.cpp'",
        ));
}

#[test]
fn transcribe_rejects_removed_sensevoice_backend_value() {
    let input = temp_input_wav();
    openasr()
        .args([
            "transcribe",
            &input.path().display().to_string(),
            "--backend",
            "sensevoice-onnx",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "Unsupported backend 'sensevoice-onnx'",
        ));
}

#[test]
fn transcribe_dir_rejects_removed_whisper_cpp_backend_value() {
    let input_dir = temp_home();
    let output_dir = temp_home();
    openasr()
        .args([
            "transcribe",
            &input_dir.path().display().to_string(),
            "--output",
            &output_dir.path().display().to_string(),
            "--backend",
            "whisper.cpp",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "Unsupported backend 'whisper.cpp'",
        ));
}

#[test]
fn transcribe_dir_mock_formats_match_core_renderers() {
    let source = sample_wav_fixture_path();
    for format in [
        ResponseFormat::Text,
        ResponseFormat::Json,
        ResponseFormat::VerboseJson,
        ResponseFormat::Srt,
        ResponseFormat::Vtt,
        ResponseFormat::Markdown,
    ] {
        let input_dir = tempfile::tempdir().unwrap();
        let output_dir = tempfile::tempdir().unwrap();
        let input_file = input_dir.path().join("sample.wav");
        std::fs::copy(&source, &input_file).unwrap();

        let expected =
            expected_mock_rendered_transcription("whisper-large-v3-turbo", "sample.wav", format);
        openasr()
            .args([
                "transcribe",
                &input_dir.path().display().to_string(),
                "--output",
                &output_dir.path().display().to_string(),
                "--backend",
                "mock",
                "--model",
                "whisper-large-v3-turbo",
                "--format",
                format.as_str(),
            ])
            .assert()
            .success();

        let output_path = output_dir
            .path()
            .join(format!("sample.wav.{}", format.output_extension()));
        let rendered = std::fs::read_to_string(&output_path).unwrap();
        assert_eq!(
            rendered,
            expected,
            "unexpected batch output for {}",
            format.as_str()
        );
    }
}

#[test]
fn transcribe_dir_native_requires_local_model_pack_path() {
    let input_dir = tempfile::tempdir().unwrap();
    let output_dir = tempfile::tempdir().unwrap();
    std::fs::write(input_dir.path().join("sample.wav"), b"not a real wav").unwrap();
    openasr()
        .args([
            "transcribe",
            &input_dir.path().display().to_string(),
            "--output",
            &output_dir.path().display().to_string(),
            "--backend",
            "native",
            "--model",
            "whisper-large-v3-turbo",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("is not installed"));
}

#[test]
fn transcribe_benchmark_rejects_removed_sensevoice_backend_value() {
    let input = temp_input_wav();
    openasr()
        .args([
            "transcribe",
            &input.path().display().to_string(),
            "--benchmark",
            "--backend",
            "sensevoice-onnx",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "Unsupported backend 'sensevoice-onnx'",
        ));
}

#[test]
fn transcribe_benchmark_native_requires_local_model_pack_path() {
    let input = temp_input_wav();
    openasr()
        .args([
            "transcribe",
            &input.path().display().to_string(),
            "--benchmark",
            "--backend",
            "native",
            "--model",
            "whisper-large-v3-turbo",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("is not installed"));
}

#[test]
fn transcribe_benchmark_renders_timing_on_mock() {
    let input = sample_wav_fixture_path();
    openasr()
        .args([
            "transcribe",
            &input.display().to_string(),
            "--benchmark",
            "--backend",
            "mock",
            "--model",
            "whisper-large-v3-turbo",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("OpenASR benchmark"))
        .stdout(predicate::str::contains("Real-time factor:"));
}

#[test]
fn transcribe_benchmark_rejects_multiple_inputs() {
    let input = sample_wav_fixture_path();
    openasr()
        .args([
            "transcribe",
            &input.display().to_string(),
            &input.display().to_string(),
            "--benchmark",
            "--backend",
            "mock",
            "--model",
            "whisper-large-v3-turbo",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("takes exactly one input file"));
}

#[test]
fn transcribe_benchmark_rejects_request_shaping_flags() {
    let input = sample_wav_fixture_path();
    openasr()
        .args([
            "transcribe",
            &input.display().to_string(),
            "--benchmark",
            "--diarize",
            "--backend",
            "mock",
            "--model",
            "whisper-large-v3-turbo",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "measures plain transcription timing",
        ));
}

#[test]
fn transcribe_multiple_files_write_per_file_outputs() {
    let source = sample_wav_fixture_path();
    let dir = tempfile::tempdir().unwrap();
    let a = dir.path().join("a.wav");
    let b = dir.path().join("b.wav");
    std::fs::copy(&source, &a).unwrap();
    std::fs::copy(&source, &b).unwrap();
    let out = tempfile::tempdir().unwrap();
    openasr()
        .args([
            "transcribe",
            &a.display().to_string(),
            &b.display().to_string(),
            "--output",
            &out.path().display().to_string(),
            "--backend",
            "mock",
            "--model",
            "whisper-large-v3-turbo",
        ])
        .assert()
        .success();
    assert!(out.path().join("a.wav.txt").exists());
    assert!(out.path().join("b.wav.txt").exists());
}

#[test]
fn transcribe_multiple_inputs_require_output_dir() {
    let input = sample_wav_fixture_path();
    openasr()
        .args([
            "transcribe",
            &input.display().to_string(),
            &input.display().to_string(),
            "--backend",
            "mock",
            "--model",
            "whisper-large-v3-turbo",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("require --output"));
}

#[test]
fn transcribe_per_file_rejects_single_only_flags() {
    let source = sample_wav_fixture_path();
    let dir = tempfile::tempdir().unwrap();
    let a = dir.path().join("a.wav");
    let b = dir.path().join("b.wav");
    std::fs::copy(&source, &a).unwrap();
    std::fs::copy(&source, &b).unwrap();
    let out = tempfile::tempdir().unwrap();
    openasr()
        .args([
            "transcribe",
            &a.display().to_string(),
            &b.display().to_string(),
            "--output",
            &out.path().display().to_string(),
            "--word-timestamps",
            "--backend",
            "mock",
            "--model",
            "whisper-large-v3-turbo",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("single input only"));
}

#[test]
fn transcribe_continue_on_error_reports_failures() {
    let source = sample_wav_fixture_path();
    let dir = tempfile::tempdir().unwrap();
    let good = dir.path().join("good.wav");
    std::fs::copy(&source, &good).unwrap();
    let missing = dir.path().join("missing.wav");
    let out = tempfile::tempdir().unwrap();
    openasr()
        .args([
            "transcribe",
            &good.display().to_string(),
            &missing.display().to_string(),
            "--output",
            &out.path().display().to_string(),
            "--continue-on-error",
            "--backend",
            "mock",
            "--model",
            "whisper-large-v3-turbo",
        ])
        .assert()
        .failure()
        .stdout(predicate::str::contains("Files failed: 1"));
    assert!(out.path().join("good.wav.txt").exists());
}

#[test]
fn transcribe_multiple_formats_write_sidecars_next_to_input() {
    let source = sample_wav_fixture_path();
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("clip.wav");
    std::fs::copy(&source, &input).unwrap();
    openasr()
        .args([
            "transcribe",
            &input.display().to_string(),
            "-f",
            "srt",
            "-f",
            "vtt",
            "--backend",
            "mock",
            "--model",
            "whisper-large-v3-turbo",
        ])
        .assert()
        .success();
    assert!(dir.path().join("clip.wav.srt").exists());
    assert!(dir.path().join("clip.wav.vtt").exists());
}

#[test]
fn transcribe_multiple_formats_write_into_output_dir() {
    let source = sample_wav_fixture_path();
    let input_dir = tempfile::tempdir().unwrap();
    let input = input_dir.path().join("clip.wav");
    std::fs::copy(&source, &input).unwrap();
    let out = tempfile::tempdir().unwrap();
    openasr()
        .args([
            "transcribe",
            &input.display().to_string(),
            "-f",
            "json",
            "-f",
            "srt",
            "-o",
            &out.path().display().to_string(),
            "--backend",
            "mock",
            "--model",
            "whisper-large-v3-turbo",
        ])
        .assert()
        .success();
    assert!(out.path().join("clip.wav.json").exists());
    assert!(out.path().join("clip.wav.srt").exists());
}

#[test]
fn transcribe_reads_wav_from_stdin() {
    let bytes = std::fs::read(sample_wav_fixture_path()).unwrap();
    openasr()
        .args([
            "transcribe",
            "-",
            "--backend",
            "mock",
            "--model",
            "whisper-large-v3-turbo",
        ])
        .write_stdin(bytes)
        .assert()
        .success()
        .stdout(predicate::str::contains("OpenASR mock transcription"));
}

#[test]
fn live_rejects_removed_whisper_cpp_backend_value() {
    openasr()
        .args(["live", "--source", "mic", "--backend", "whisper.cpp"])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "Unsupported backend 'whisper.cpp'",
        ));
}

#[test]
fn serve_rejects_removed_sensevoice_backend_value() {
    openasr()
        .args(["serve", "--backend", "sensevoice-onnx"])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "Unsupported backend 'sensevoice-onnx'",
        ));
}

#[test]
fn pull_subcommand_is_available_for_model_distribution() {
    openasr()
        .args(["pull", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "Download a local OpenASR model pack",
        ))
        .stdout(predicate::str::contains("<id>:<quant>"))
        .stdout(predicate::str::contains("--catalog-url"));
}

#[test]
fn hidden_gguf_c_parser_probe_emits_metadata_and_tensor_index_json() {
    let temp = tempfile::tempdir().expect("tempdir");
    let pack = temp.path().join("probe.oasr");
    write_whisper_oasr_v1_fixture(&pack, "whisper-small");

    openasr()
        .args([
            openasr_core::GGUF_C_PARSER_SANDBOX_HELPER_ARG,
            pack.to_str().expect("pack path"),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains(r#""metadata""#))
        .stdout(predicate::str::contains(r#""tensor_index""#))
        .stdout(predicate::str::contains("whisper-small"));
}

#[test]
fn pull_installs_local_pack_from_catalog_reference() {
    let home = temp_home();
    let temp = tempfile::tempdir().expect("tempdir");
    let pack = temp.path().join("moonshine-tiny-q8_0.oasr");
    write_whisper_oasr_v1_fixture(&pack, "moonshine-tiny");
    let bytes = std::fs::read(&pack).expect("read pack fixture");
    let sha256 = format!("{:x}", Sha256::digest(&bytes));
    let catalog = temp.path().join("catalog.json");
    write_catalog_fixture(&catalog, &sha256, bytes.len() as u64);
    let catalog_url = format!("file://{}", catalog.display());

    openasr_with_home(home.path())
        .args([
            "pull",
            "moonshine-tiny:q8",
            "--catalog-url",
            &catalog_url,
            "--from",
            pack.to_str().expect("pack path"),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("moonshine-tiny:q8"))
        .stdout(predicate::str::contains(&sha256));

    openasr_with_home(home.path())
        .args(["list"])
        .assert()
        .success()
        .stdout(predicate::str::contains("moonshine-tiny:q8"));
}

#[test]
fn pull_gated_catalog_entry_requires_license_acceptance_or_local_pack() {
    let home = temp_home();
    let temp = tempfile::tempdir().expect("tempdir");
    let catalog = temp.path().join("catalog.json");
    write_gated_catalog_fixture(&catalog);
    let catalog_url = format!("file://{}", catalog.display());

    openasr_with_home(home.path())
        .args([
            "pull",
            "parakeet-ctc-0.6b:q8",
            "--catalog-url",
            &catalog_url,
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "requires vendor license acceptance before download",
        ))
        .stderr(predicate::str::contains("Open vendor site:"))
        .stderr(predicate::str::contains(
            "Then rerun with --accept-license or --from <local-pack>.",
        ));

    openasr_with_home(home.path())
        .args(["list"])
        .assert()
        .success()
        .stdout(predicate::str::contains("No models installed"));
}

#[test]
fn pull_alias_with_size_and_quant_option_installs_resolved_catalog_pull() {
    let home = temp_home();
    let temp = tempfile::tempdir().expect("tempdir");
    let pack = temp.path().join("moonshine-tiny-q8_0.oasr");
    write_whisper_oasr_v1_fixture(&pack, "moonshine-tiny");
    let bytes = std::fs::read(&pack).expect("read pack fixture");
    let sha256 = format!("{:x}", Sha256::digest(&bytes));
    let catalog = temp.path().join("catalog.json");
    write_ambiguous_moonshine_catalog_fixture(&catalog, &sha256, bytes.len() as u64);
    let catalog_url = format!("file://{}", catalog.display());

    openasr_with_home(home.path())
        .args([
            "pull",
            "moonshine",
            "--size",
            "tiny",
            "--quant",
            "q8",
            "--catalog-url",
            &catalog_url,
            "--from",
            pack.to_str().expect("pack path"),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("moonshine-tiny:q8"))
        .stdout(predicate::str::contains(&sha256));

    openasr_with_home(home.path())
        .args(["list"])
        .assert()
        .success()
        .stdout(predicate::str::contains("moonshine-tiny:q8"))
        .stdout(predicate::str::contains("moonshine-base:q8").not());
}

#[test]
fn pull_from_local_pack_fails_closed_on_sha_mismatch() {
    let home = temp_home();
    let temp = tempfile::tempdir().expect("tempdir");
    let pack = temp.path().join("moonshine-tiny-q8_0.oasr");
    write_whisper_oasr_v1_fixture(&pack, "moonshine-tiny");
    let bytes = std::fs::read(&pack).expect("read pack fixture");
    let catalog = temp.path().join("catalog.json");
    write_catalog_fixture(
        &catalog,
        "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
        bytes.len() as u64,
    );
    let catalog_url = format!("file://{}", catalog.display());

    openasr_with_home(home.path())
        .args([
            "pull",
            "moonshine-tiny:q8",
            "--catalog-url",
            &catalog_url,
            "--from",
            pack.to_str().expect("pack path"),
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("sha256 mismatch"))
        .stderr(predicate::str::contains(
            "expected eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
        ));

    openasr_with_home(home.path())
        .args(["list"])
        .assert()
        .success()
        .stdout(predicate::str::contains("No models installed"));
}

#[test]
fn pull_rejects_unsupported_catalog_schema_before_download() {
    let home = temp_home();
    let temp = tempfile::tempdir().expect("tempdir");
    let catalog = temp.path().join("catalog.json");
    write_unsupported_catalog_schema_fixture(&catalog);
    let catalog_url = format!("file://{}", catalog.display());

    openasr_with_home(home.path())
        .args(["pull", "moonshine-tiny:q8", "--catalog-url", &catalog_url])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "Unsupported model catalog schema_version 99; update OpenASR to read this catalog.",
        ));

    openasr_with_home(home.path())
        .args(["list"])
        .assert()
        .success()
        .stdout(predicate::str::contains("No models installed"));
}

#[test]
fn models_rm_removes_installed_pack_by_model_id() {
    let home = temp_home();
    let temp = tempfile::tempdir().expect("tempdir");
    let pack = temp.path().join("moonshine-tiny-q8_0.oasr");
    write_whisper_oasr_v1_fixture(&pack, "moonshine-tiny");
    let bytes = std::fs::read(&pack).expect("read pack fixture");
    let sha256 = format!("{:x}", Sha256::digest(&bytes));
    let catalog = temp.path().join("catalog.json");
    write_catalog_fixture(&catalog, &sha256, bytes.len() as u64);
    let catalog_url = format!("file://{}", catalog.display());

    openasr_with_home(home.path())
        .args([
            "pull",
            "moonshine-tiny:q8",
            "--catalog-url",
            &catalog_url,
            "--from",
            pack.to_str().expect("pack path"),
        ])
        .assert()
        .success();

    openasr_with_home(home.path())
        .args(["rm", "moonshine-tiny"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Removed moonshine-tiny:q8"));

    openasr_with_home(home.path())
        .args(["list"])
        .assert()
        .success()
        .stdout(predicate::str::contains("No models installed"));
}

#[test]
fn models_rm_reports_missing_install() {
    let home = temp_home();

    openasr_with_home(home.path())
        .args(["rm", "moonshine-tiny:q8"])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "Model pack is not installed: moonshine-tiny:q8",
        ));
}

#[test]
fn remove_subcommand_is_removed() {
    openasr()
        .args(["remove", "whisper-large-v3-turbo"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("unrecognized subcommand 'remove'"));
}

#[test]
fn transcribe_rejects_unknown_saved_default_model_value() {
    let home = temp_home();
    std::fs::write(
        home.path().join("config.json"),
        r#"{
  "default_model": "not-a-model",
  "default_backend": "mock",
  "media": {}
}
"#,
    )
    .expect("write config");

    let input = temp_input_wav();
    openasr()
        .env("OPENASR_HOME", home.path())
        .args(["transcribe", &input.path().display().to_string()])
        .assert()
        .failure()
        .stderr(predicate::str::contains("Unknown model: not-a-model"));
}

#[test]
fn transcribe_rejects_saved_default_model_when_unknown_family_ref_is_present() {
    let home = temp_home();
    std::fs::write(
        home.path().join("config.json"),
        r#"{
  "default_model": "no-such-model-xyz",
  "default_backend": "mock",
  "media": {}
}
"#,
    )
    .expect("write config");

    let input = temp_input_wav();
    openasr()
        .env("OPENASR_HOME", home.path())
        .args(["transcribe", &input.path().display().to_string()])
        .assert()
        .failure()
        .stderr(predicate::str::contains("Unknown model: no-such-model-xyz"));
}

#[test]
fn doctor_reports_native_backend_line() {
    openasr()
        .arg("doctor")
        .assert()
        .success()
        .stdout(predicate::str::contains("- mock: ok"))
        .stdout(predicate::str::contains("native").or(predicate::str::contains("Backends")));
}

#[test]
fn doctor_marks_legacy_saved_default_backend_as_legacy() {
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
    .expect("write legacy backend config");

    openasr()
        .env("OPENASR_HOME", home.path())
        .arg("doctor")
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "Default backend: whisper.cpp (legacy)",
        ));
}

#[test]
fn catalog_fingerprint_prints_json_line_matching_embedded_signature() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../model-registry");
    let public_contents =
        std::fs::read_to_string(root.join("catalog.public.json")).expect("read public catalog");
    let manifest_contents = std::fs::read_to_string(root.join("catalog.public.signature.json"))
        .expect("read public catalog signature manifest");
    let manifest: Value = serde_json::from_str(&manifest_contents).expect("parse manifest");
    let expected_epoch = manifest["catalog_epoch"].as_u64().expect("catalog_epoch");
    let expected_sha256 = {
        let mut hasher = Sha256::new();
        hasher.update(public_contents.as_bytes());
        hasher
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
    };

    let output = openasr()
        .arg("catalog-fingerprint")
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let stdout = String::from_utf8(output).expect("utf8 stdout");
    let parsed: Value =
        serde_json::from_str(stdout.trim()).expect("catalog-fingerprint prints a single JSON line");

    assert_eq!(
        parsed["catalog_sha256"].as_str().unwrap(),
        expected_sha256,
        "fingerprint sha256 must be byte-identical to sha256(catalog.public.json)"
    );
    assert_eq!(
        parsed["catalog_epoch"]
            .as_str()
            .unwrap()
            .parse::<u64>()
            .unwrap(),
        expected_epoch,
        "fingerprint epoch must match the embedded signature manifest's epoch"
    );
}

#[test]
fn doctor_marks_sensevoice_cpp_saved_default_backend_as_legacy() {
    let home = temp_home();
    std::fs::write(
        home.path().join("config.json"),
        r#"{
  "default_model": "whisper-large-v3-turbo",
  "default_backend": "sensevoice.cpp",
  "media": {}
}
"#,
    )
    .expect("write legacy backend config");

    openasr()
        .env("OPENASR_HOME", home.path())
        .arg("doctor")
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "Default backend: sensevoice.cpp (legacy)",
        ));
}

#[test]
fn doctor_marks_unknown_saved_default_model_as_unknown() {
    let home = temp_home();
    std::fs::write(
        home.path().join("config.json"),
        r#"{
  "default_model": "no-such-model-xyz",
  "default_backend": "mock",
  "media": {}
}
"#,
    )
    .expect("write unknown model config");

    openasr()
        .env("OPENASR_HOME", home.path())
        .arg("doctor")
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "Default model: no-such-model-xyz (unknown)",
        ));
}

#[test]
fn doctor_marks_unknown_saved_default_backend_as_unknown() {
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
    .expect("write unknown backend config");

    openasr()
        .env("OPENASR_HOME", home.path())
        .arg("doctor")
        .assert()
        .success()
        .stdout(predicate::str::contains("Default backend: mokk (unknown)"));
}
