use axum::{
    Router,
    body::{Body, to_bytes},
    http::{Request, StatusCode, header},
};
use futures_util::StreamExt;
use openasr_core::api::backend::transcribe_with_mock_backend;
use openasr_core::testing::{
    TinyGgufFixtureSpec, write_local_dev_signed_catalog, write_reserved_oasr_container,
    write_tiny_gguf_runtime_source,
};
use openasr_core::{ResponseFormat, TranscriptionRequest, render_transcription};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    io::Write,
    sync::{Mutex, OnceLock},
    time::Duration,
};
use tower::ServiceExt;

const SERVER_INSTANCE_TOKEN_ENV: &str = "OPENASR_SERVER_INSTANCE_TOKEN";
const LIVE_PULL_FIXTURE_SIZE_BYTES: u64 = 64 * 1024 * 1024;

/// The product default `dictation_shortcut` for the host this test binary is
/// compiled for -- mirrors openasr-core's `default_dictation_shortcut()`
/// `#[cfg]` split (Ctrl+Win on Windows, Option alone elsewhere), so the
/// fresh-config assertion below tracks the real default on every platform
/// instead of a single hardcoded string.
#[cfg(windows)]
fn expected_default_dictation_shortcut() -> &'static str {
    "LControl+LCommand"
}

#[cfg(not(windows))]
fn expected_default_dictation_shortcut() -> &'static str {
    "Alt"
}

fn sample_wav_bytes() -> Vec<u8> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/jfk.wav");
    std::fs::read(path).unwrap()
}

struct ServerInstanceTokenEnvRestore {
    previous: Option<std::ffi::OsString>,
}

impl Drop for ServerInstanceTokenEnvRestore {
    fn drop(&mut self) {
        match self.previous.take() {
            Some(value) => {
                unsafe { std::env::set_var(SERVER_INSTANCE_TOKEN_ENV, value) };
            }
            None => {
                unsafe { std::env::remove_var(SERVER_INSTANCE_TOKEN_ENV) };
            }
        }
    }
}

fn with_server_instance_token_env<T>(value: Option<&str>, run: impl FnOnce() -> T) -> T {
    static INSTANCE_TOKEN_ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    let lock = INSTANCE_TOKEN_ENV_LOCK.get_or_init(|| Mutex::new(()));
    let _guard = lock.lock().expect("instance token env lock");
    let _restore = ServerInstanceTokenEnvRestore {
        previous: std::env::var_os(SERVER_INSTANCE_TOKEN_ENV),
    };
    match value {
        Some(value) => {
            unsafe { std::env::set_var(SERVER_INSTANCE_TOKEN_ENV, value) };
        }
        None => {
            unsafe { std::env::remove_var(SERVER_INSTANCE_TOKEN_ENV) };
        }
    }
    run()
}

fn write_content_addressed_moonshine_ref(home: &std::path::Path) -> std::path::PathBuf {
    std::fs::create_dir_all(home).unwrap();
    let staging = home.join("fixture-source.oasr");
    let spec = TinyGgufFixtureSpec::whisper_oasr_v1_encoder_graph_one_layer("moonshine-tiny");
    write_tiny_gguf_runtime_source(&staging, &spec).expect("write content-addressed fixture");
    let bytes = std::fs::read(&staging).unwrap();
    std::fs::remove_file(&staging).unwrap();
    let sha256 = format!("{:x}", Sha256::digest(&bytes));
    let object = home
        .join("models/objects/sha256")
        .join(&sha256)
        .join("content");
    std::fs::create_dir_all(object.parent().unwrap()).unwrap();
    std::fs::write(&object, &bytes).unwrap();

    let reference = home.join("models/refs/moonshine-tiny/q8_0.json");
    std::fs::create_dir_all(reference.parent().unwrap()).unwrap();
    let pack = openasr_core::InstalledPack {
        model_id: "moonshine-tiny".to_string(),
        display_name: "Moonshine Tiny".to_string(),
        quant: "q8_0".to_string(),
        suffix: "q8".to_string(),
        pull: "moonshine-tiny:q8".to_string(),
        filename: "moonshine-tiny-q8_0.oasr".to_string(),
        path: object.clone(),
        url: "https://example.invalid/moonshine-tiny-q8_0.oasr".to_string(),
        hf_revision: "test".to_string(),
        sha256,
        size_bytes: bytes.len() as u64,
        installed_at_unix_seconds: 1,
        source: None,
    };
    std::fs::write(reference, serde_json::to_vec_pretty(&pack).unwrap()).unwrap();
    object
}

fn write_mock_gguf_runtime_source(path: &std::path::Path, metadata_model_id: Option<&str>) {
    let spec = metadata_model_id.map_or_else(
        || TinyGgufFixtureSpec::new(Default::default()),
        TinyGgufFixtureSpec::whisper_oasr_v1_non_streaming_cpu,
    );
    write_tiny_gguf_runtime_source(path, &spec).expect("write mock gguf runtime source");
}

fn write_xasr_gguf_runtime_source(path: &std::path::Path, metadata_model_id: &str) {
    let mut metadata = BTreeMap::new();
    metadata.insert(
        "openasr.model.id".to_string(),
        metadata_model_id.to_string(),
    );
    metadata.insert(
        openasr_core::models::oasr_metadata::OASR_METADATA_KEY_PACKAGE_VERSION.to_string(),
        openasr_core::models::oasr_metadata::OASR_PACKAGE_VERSION_V1.to_string(),
    );
    metadata.insert(
        openasr_core::models::oasr_metadata::OASR_METADATA_KEY_MODEL_FAMILY.to_string(),
        "xasr-zipformer".to_string(),
    );
    metadata.insert(
        openasr_core::models::oasr_metadata::OASR_METADATA_KEY_MODEL_ARCHITECTURE.to_string(),
        openasr_core::XASR_ZIPFORMER_GGML_ARCHITECTURE_ID.to_string(),
    );
    metadata.insert(
        openasr_core::models::oasr_metadata::OASR_METADATA_KEY_AUDIO_FRONTEND.to_string(),
        openasr_core::XASR_ZIPFORMER_AUDIO_FRONTEND_ID.to_string(),
    );
    metadata.insert(
        openasr_core::models::oasr_metadata::OASR_METADATA_KEY_DECODE_POLICY.to_string(),
        openasr_core::XASR_ZIPFORMER_DECODE_POLICY_ID.to_string(),
    );
    metadata.insert(
        openasr_core::GGML_TOKENIZER_ID_KEY.to_string(),
        openasr_core::XASR_ZIPFORMER_TOKENIZER_ID.to_string(),
    );
    let spec = TinyGgufFixtureSpec::new(metadata);
    write_tiny_gguf_runtime_source(path, &spec).expect("write xasr gguf runtime source");
}

fn write_whisper_oasr_v1_fixture(path: &std::path::Path, model_id: &str) {
    let spec = TinyGgufFixtureSpec::whisper_oasr_v1_graph_ready_for_runtime_fail_closed(model_id);
    write_tiny_gguf_runtime_source(path, &spec).expect("write whisper gguf runtime source");
}

fn write_moonshine_pull_fixture(
    root: &std::path::Path,
) -> (std::path::PathBuf, openasr_server::DistributionRuntime) {
    let pack_path = root.join("moonshine-tiny-q8_0.oasr");
    // `whisper_oasr_v1_non_streaming_cpu` alone omits the whisper runtime
    // scalar contract keys; install-time validation now enforces them (see
    // `validate_native_runtime_model_pack_contract`), so this pull/import
    // stand-in pack must be contract-complete to keep installing.
    let spec = TinyGgufFixtureSpec::whisper_oasr_v1_encoder_graph_one_layer("moonshine-tiny");
    write_tiny_gguf_runtime_source(&pack_path, &spec).expect("write pull fixture");
    let bytes = std::fs::read(&pack_path).unwrap();
    let sha256 = format!("{:x}", Sha256::digest(&bytes));
    let revision = "0123456789abcdef0123456789abcdef01234567";
    let catalog = serde_json::json!({
        "schema_version": 1,
        "generated_at": "2026-05-31T00:00:00Z",
        "catalog_url": "file://test-catalog.json",
        "models": [{
            "id": "moonshine-tiny",
            "display_name": "Moonshine Tiny",
            "family": "moonshine",
            "aliases": ["moonshine"],
            "pull_alias": "moonshine",
            "size": "tiny",
            "languages": ["en"],
            "vendor": "Useful Sensors",
            "license": "MIT",
            "license_url": "https://huggingface.co/UsefulSensors/moonshine-tiny",
            "license_class": "permissive",
            "hf_repo": "OpenASR/moonshine-tiny",
            "hf_revision": revision,
            "public": true,
            "min_cli_version": "0.1.0",
            "recommended_quant": "q8_0",
            "pull_recommended": "moonshine-tiny:q8",
            "quants": [{
                "quant": "q8_0",
                "suffix": "q8",
                "pull": "moonshine-tiny:q8",
                "filename": "moonshine-tiny-q8_0.oasr",
                "url": format!("https://huggingface.co/OpenASR/moonshine-tiny/resolve/{revision}/moonshine-tiny-q8_0.oasr"),
                "sha256": sha256,
                "size_bytes": bytes.len() as u64,
                "recommended": true
            }]
        }]
    });
    let catalog_path = root.join("catalog.json");
    let catalog_json =
        String::from_utf8(serde_json::to_vec_pretty(&catalog).expect("serialize catalog fixture"))
            .expect("catalog fixture is valid utf-8");
    // A local `file://` catalog now requires the same signed sidecar a
    // production HTTPS catalog does; sign it with the public local-dev key.
    write_local_dev_signed_catalog(&catalog_path, &catalog_json, 1);

    (
        pack_path,
        openasr_server::DistributionRuntime {
            openasr_home: Some(root.join("home")),
            catalog_url: Some(format!("file://{}", catalog_path.display())),
            catalog_local_override: None,
        },
    )
}

fn pad_pull_fixture_pack_to(
    pack_path: &std::path::Path,
    distribution: &openasr_server::DistributionRuntime,
    min_size_bytes: u64,
) {
    let mut file = std::fs::OpenOptions::new()
        .append(true)
        .open(pack_path)
        .unwrap();
    let mut remaining = min_size_bytes.saturating_sub(file.metadata().unwrap().len());
    let zeros = vec![0_u8; 1024 * 1024];
    while remaining > 0 {
        let chunk_len = remaining.min(zeros.len() as u64) as usize;
        file.write_all(&zeros[..chunk_len]).unwrap();
        remaining -= chunk_len as u64;
    }
    drop(file);

    let bytes = std::fs::read(pack_path).unwrap();
    let sha256 = format!("{:x}", Sha256::digest(&bytes));
    let catalog_url = distribution.catalog_url.as_ref().unwrap();
    let catalog_path = std::path::Path::new(catalog_url.strip_prefix("file://").unwrap());
    let mut catalog: Value = serde_json::from_slice(&std::fs::read(catalog_path).unwrap()).unwrap();
    let quant = &mut catalog["models"][0]["quants"][0];
    quant["sha256"] = serde_json::json!(sha256);
    quant["size_bytes"] = serde_json::json!(bytes.len() as u64);
    let catalog_json = String::from_utf8(serde_json::to_vec_pretty(&catalog).unwrap()).unwrap();
    // Re-sign after mutating the catalog bytes in place: a stale sidecar
    // would now be treated as tampering, not a no-op (see
    // `write_local_dev_signed_catalog`'s doc comment).
    write_local_dev_signed_catalog(catalog_path, &catalog_json, 1);
}

async fn create_approved_pairing_credential(app: &Router, device_name: &str) -> (String, String) {
    let create = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/pairing/requests")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    serde_json::json!({ "device_name": device_name }).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(create.status(), StatusCode::ACCEPTED);
    let create_body = to_bytes(create.into_body(), 1024 * 64).await.unwrap();
    let create_json: Value = serde_json::from_slice(&create_body).unwrap();
    let request_id = create_json["request_id"].as_str().unwrap().to_string();

    let approve = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(format!("/v1/pairing/requests/{request_id}/approve"))
                .header(header::AUTHORIZATION, "Bearer admin-secret")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(approve.status(), StatusCode::OK);
    let approve_body = to_bytes(approve.into_body(), 1024 * 64).await.unwrap();
    let approve_json: Value = serde_json::from_slice(&approve_body).unwrap();
    let device_id = approve_json["device_id"].as_str().unwrap().to_string();

    let credential = app
        .clone()
        .oneshot(
            Request::builder()
                .uri(format!("/v1/pairing/requests/{request_id}/credential"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(credential.status(), StatusCode::OK);
    let credential_body = to_bytes(credential.into_body(), 1024 * 64).await.unwrap();
    let credential_json: Value = serde_json::from_slice(&credential_body).unwrap();
    let bearer_token = credential_json["bearer_token"]
        .as_str()
        .unwrap()
        .to_string();
    assert_eq!(credential_json["device_id"], device_id);

    (device_id, bearer_token)
}

/// Installed packs are immutable content-addressed objects:
/// `<models>/objects/sha256/<digest>/content`.
fn assert_installed_content_object(installed_path: &str) {
    let path = std::path::Path::new(installed_path);
    assert_eq!(
        path.file_name().and_then(|name| name.to_str()),
        Some("content"),
        "installed pack must be a content-addressed object: {installed_path}"
    );
    assert!(
        path.parent()
            .and_then(std::path::Path::parent)
            .is_some_and(|root| root.ends_with("objects/sha256")),
        "installed pack must live under objects/sha256: {installed_path}"
    );
}

fn write_complete_moonshine_partial(home: &std::path::Path, source_pack: &std::path::Path) -> u64 {
    let bytes = std::fs::read(source_pack).unwrap();
    let sha256 = format!("{:x}", Sha256::digest(&bytes));
    let revision = "0123456789abcdef0123456789abcdef01234567";
    let url = format!(
        "https://huggingface.co/OpenASR/moonshine-tiny/resolve/{revision}/moonshine-tiny-q8_0.oasr"
    );
    // In-flight downloads live in the shared staging directory, keyed by the
    // digest they are downloading, not in a per-model/quant directory.
    let staging_dir = home.join("models").join("staging");
    std::fs::create_dir_all(&staging_dir).unwrap();
    let partial_path = staging_dir.join(format!("{sha256}-moonshine-tiny-q8_0.oasr.partial"));
    let partial_meta_path = staging_dir.join(format!(
        "{sha256}-moonshine-tiny-q8_0.oasr.partial.meta.json"
    ));
    std::fs::write(&partial_path, &bytes).unwrap();
    std::fs::write(
        &partial_meta_path,
        serde_json::to_vec_pretty(&serde_json::json!({
            "model_id": "moonshine-tiny",
            "quant": "q8_0",
            "filename": "moonshine-tiny-q8_0.oasr",
            "url": url,
            "hf_revision": revision,
            "sha256": sha256,
            "size_bytes": bytes.len() as u64,
            "etag": null,
            "bytes_done": bytes.len() as u64,
            "updated_at_unix_seconds": 1
        }))
        .unwrap(),
    )
    .unwrap();
    bytes.len() as u64
}

fn write_persisted_pull_job(
    home: &std::path::Path,
    job_id: &str,
    state: &str,
    bytes_done: u64,
    bytes_total: u64,
) {
    let pulls_dir = home.join("pulls");
    std::fs::create_dir_all(&pulls_dir).unwrap();
    std::fs::write(
        pulls_dir.join(format!("{job_id}.json")),
        serde_json::to_vec_pretty(&serde_json::json!({
            "job_id": job_id,
            "state": state,
            "model_id": "moonshine-tiny",
            "display_name": "Moonshine Tiny",
            "quant": "q8_0",
            "pull": "moonshine-tiny:q8",
            "bytes_done": bytes_done,
            "bytes_total": bytes_total
        }))
        .unwrap(),
    )
    .unwrap();
}

fn write_persisted_pull_job_with_resolved(
    home: &std::path::Path,
    job_id: &str,
    state: &str,
    bytes_done: u64,
    bytes_total: u64,
    source_pack: &std::path::Path,
) {
    write_persisted_pull_job_with_resolved_and_source(
        home,
        job_id,
        state,
        bytes_done,
        bytes_total,
        source_pack,
        None,
    );
}

fn write_persisted_local_source_pull_job_with_resolved(
    home: &std::path::Path,
    job_id: &str,
    state: &str,
    bytes_done: u64,
    bytes_total: u64,
    source_pack: &std::path::Path,
) {
    write_persisted_pull_job_with_resolved_and_source(
        home,
        job_id,
        state,
        bytes_done,
        bytes_total,
        source_pack,
        Some(source_pack),
    );
}

fn write_persisted_pull_job_with_resolved_and_source(
    home: &std::path::Path,
    job_id: &str,
    state: &str,
    bytes_done: u64,
    bytes_total: u64,
    source_pack: &std::path::Path,
    source_path: Option<&std::path::Path>,
) {
    let bytes = std::fs::read(source_pack).unwrap();
    let sha256 = format!("{:x}", Sha256::digest(&bytes));
    let revision = "0123456789abcdef0123456789abcdef01234567";
    let url = source_path.map_or_else(
        || {
            format!(
                "https://huggingface.co/OpenASR/moonshine-tiny/resolve/{revision}/moonshine-tiny-q8_0.oasr"
            )
        },
        |_| "https://127.0.0.1:9/moonshine-tiny-q8_0.oasr".to_string(),
    );
    let pulls_dir = home.join("pulls");
    std::fs::create_dir_all(&pulls_dir).unwrap();
    let source_path = source_path.map(|path| path.to_path_buf());
    std::fs::write(
        pulls_dir.join(format!("{job_id}.json")),
        serde_json::to_vec_pretty(&serde_json::json!({
            "job_id": job_id,
            "state": state,
            "model_id": "moonshine-tiny",
            "display_name": "Moonshine Tiny",
            "quant": "q8_0",
            "pull": "moonshine-tiny:q8",
            "resolved": {
                "requested": "moonshine-tiny:q8",
                "model_id": "moonshine-tiny",
                "display_name": "Moonshine Tiny",
                "quant": "q8_0",
                "suffix": "q8",
                "pull": "moonshine-tiny:q8",
                "filename": "moonshine-tiny-q8_0.oasr",
                "url": url,
                "hf_revision": revision,
                "sha256": sha256,
                "size_bytes": bytes.len() as u64,
                "license": "MIT",
                "license_url": "https://huggingface.co/UsefulSensors/moonshine-tiny",
                "license_class": "permissive"
            },
            "source_path": source_path,
            "bytes_done": bytes_done,
            "bytes_total": bytes_total
        }))
        .unwrap(),
    )
    .unwrap();
}

fn mutate_fixture_catalog_pack_identity(distribution: &openasr_server::DistributionRuntime) {
    let catalog_url = distribution.catalog_url.as_ref().unwrap();
    let catalog_path = std::path::Path::new(catalog_url.strip_prefix("file://").unwrap());
    let mut catalog: Value = serde_json::from_slice(&std::fs::read(catalog_path).unwrap()).unwrap();
    let model = &mut catalog["models"][0];
    model["hf_revision"] = serde_json::json!("fedcba9876543210fedcba9876543210fedcba98");
    let quant = &mut model["quants"][0];
    quant["url"] = serde_json::json!(
        "https://huggingface.co/OpenASR/moonshine-tiny/resolve/fedcba9876543210fedcba9876543210fedcba98/moonshine-tiny-q8_0.oasr"
    );
    quant["sha256"] =
        serde_json::json!("cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc");
    quant["size_bytes"] = serde_json::json!(1);
    let catalog_json = String::from_utf8(serde_json::to_vec_pretty(&catalog).unwrap()).unwrap();
    // Re-sign after mutating the catalog bytes in place (see
    // `pad_pull_fixture_pack_to`'s matching comment).
    write_local_dev_signed_catalog(catalog_path, &catalog_json, 1);
}

fn write_reserved_oasr_runtime_source(path: &std::path::Path) {
    write_reserved_oasr_container(path).expect("write reserved oasr runtime source");
}

async fn job_snapshot(app: axum::Router, job_id: &str) -> Value {
    let response = app
        .oneshot(
            Request::builder()
                .uri(format!("/v1/models/pull/{job_id}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = to_bytes(response.into_body(), 1024 * 64).await.unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

async fn post_pull_control(app: axum::Router, uri: String) -> (StatusCode, Value) {
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(uri)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let bytes = to_bytes(response.into_body(), 1024 * 64).await.unwrap();
    (status, serde_json::from_slice(&bytes).unwrap())
}

async fn wait_for_terminal_job(app: axum::Router, job_id: &str) -> Value {
    let mut last = None;
    for _ in 0..40 {
        let parsed = job_snapshot(app.clone(), job_id).await;
        match parsed["state"].as_str() {
            Some("completed" | "already_installed" | "canceled" | "failed") => return parsed,
            _ => {
                last = Some(parsed);
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
        }
    }
    panic!(
        "pull job did not finish; last snapshot: {}",
        serde_json::to_string_pretty(&last).unwrap()
    );
}

#[tokio::test]
async fn catalog_endpoint_serves_configured_catalog() {
    let temp = tempfile::tempdir().unwrap();
    let (_, distribution) = write_moonshine_pull_fixture(temp.path());
    let app = openasr_server::app_with_runtime_and_distribution(
        openasr_server::ServerRuntime::default(),
        distribution,
    );

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/catalog")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let bytes = to_bytes(response.into_body(), 1024 * 256).await.unwrap();
    let parsed: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(parsed["models"][0]["id"], "moonshine-tiny");
    assert_eq!(
        parsed["models"][0]["quants"][0]["pull"],
        "moonshine-tiny:q8"
    );
}

#[tokio::test]
async fn config_endpoint_roundtrips_versioned_preferences() {
    let temp = tempfile::tempdir().unwrap();
    let distribution = openasr_server::DistributionRuntime {
        openasr_home: Some(temp.path().join("home")),
        catalog_url: None,
        catalog_local_override: None,
    };
    let home = distribution.openasr_home.as_ref().unwrap().clone();
    let app = openasr_server::app_with_runtime_and_distribution(
        openasr_server::ServerRuntime::default(),
        distribution,
    );

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/config")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = to_bytes(response.into_body(), 1024 * 64).await.unwrap();
    let mut document: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(
        document["preferences"]["version"],
        openasr_core::config::PREFERENCES_SCHEMA_VERSION
    );
    // Fresh-config product defaults surfaced to the desktop: Option (⌥) alone
    // on macOS/Linux, Ctrl+Win on Windows, push-to-talk on. These are what a
    // cleared-state first launch shows.
    assert_eq!(
        document["preferences"]["dictation_shortcut"],
        expected_default_dictation_shortcut()
    );
    assert_eq!(document["preferences"]["push_to_talk"], true);
    assert_eq!(document["preferences"]["word_timestamps"], false);

    document["preferences"]["language"] = serde_json::json!("en");
    document["preferences"]["auto_save"] = serde_json::json!(true);
    document["preferences"]["output_dir"] =
        serde_json::json!(temp.path().join("out").to_string_lossy());
    document["preferences"]["hotwords"] = serde_json::json!(["OpenASR"]);
    document["preferences"]["hotword_boost"] = serde_json::json!(3.5);
    document["preferences"]["theme"] = serde_json::json!("dark");
    document["preferences"]["density"] = serde_json::json!("compact");
    document["preferences"]["push_to_talk"] = serde_json::json!(true);
    document["preferences"]["inference_threads"] = serde_json::json!(2);

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("PUT")
                .uri("/v1/config")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(document.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = to_bytes(response.into_body(), 1024 * 64).await.unwrap();
    let saved: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(saved["preferences"]["language"], "en");
    assert_eq!(saved["preferences"]["hotwords"][0], "OpenASR");
    assert_eq!(saved["preferences"]["inference_threads"], 2);

    let file: Value =
        serde_json::from_slice(&std::fs::read(home.join("config.json")).unwrap()).unwrap();
    assert_eq!(file["preferences"]["theme"], "dark");
    assert_eq!(file["preferences"]["auto_save"], true);
}

#[tokio::test]
async fn config_endpoint_rejects_invalid_whole_object_update() {
    let temp = tempfile::tempdir().unwrap();
    let distribution = openasr_server::DistributionRuntime {
        openasr_home: Some(temp.path().join("home")),
        catalog_url: None,
        catalog_local_override: None,
    };
    let app = openasr_server::app_with_runtime_and_distribution(
        openasr_server::ServerRuntime::default(),
        distribution,
    );

    let mut document = serde_json::json!({
        "default_model": "whisper-large-v3-turbo",
        "default_backend": "bogus-xyz",
        "media": {},
        "preferences": {
            "version": openasr_core::config::PREFERENCES_SCHEMA_VERSION
        }
    });
    document["preferences"]["hotwords"] = serde_json::json!(["OpenASR", "openasr"]);
    let response = app
        .oneshot(
            Request::builder()
                .method("PUT")
                .uri("/v1/config")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(document.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let bytes = to_bytes(response.into_body(), 1024 * 64).await.unwrap();
    let body = String::from_utf8_lossy(&bytes);
    assert!(body.contains("Unsupported backend"));
}

#[tokio::test]
async fn preferences_only_put_preserves_daemon_managed_config() {
    let temp = tempfile::tempdir().unwrap();
    let distribution = openasr_server::DistributionRuntime {
        openasr_home: Some(temp.path().join("home")),
        catalog_url: None,
        catalog_local_override: None,
    };
    let app = openasr_server::app_with_runtime_and_distribution(
        openasr_server::ServerRuntime::default(),
        distribution,
    );

    // Establish a daemon/CLI-owned setting via a full-document PUT.
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/config")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let bytes = to_bytes(response.into_body(), 1024 * 64).await.unwrap();
    let mut document: Value = serde_json::from_slice(&bytes).unwrap();
    document["download_source"] = serde_json::json!({
        "mode": "pinned",
        "source": "hf-mirror"
    });
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("PUT")
                .uri("/v1/config")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(document.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = to_bytes(response.into_body(), 1024 * 64).await.unwrap();
    let saved: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(saved["download_source"]["source"], "hf-mirror");

    // The desktop preferences client sends preferences only (no config fields);
    // it must not reset the daemon-owned config back to defaults.
    let body = serde_json::json!({
        "preferences": {
            "version": openasr_core::config::PREFERENCES_SCHEMA_VERSION,
            "language": "en"
        }
    });
    let response = app
        .oneshot(
            Request::builder()
                .method("PUT")
                .uri("/v1/config")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = to_bytes(response.into_body(), 1024 * 64).await.unwrap();
    let after: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(after["download_source"]["source"], "hf-mirror");
    assert_eq!(after["preferences"]["language"], "en");
}

#[tokio::test]
async fn preferences_only_put_merges_partial_preferences() {
    let temp = tempfile::tempdir().unwrap();
    let distribution = openasr_server::DistributionRuntime {
        openasr_home: Some(temp.path().join("home")),
        catalog_url: None,
        catalog_local_override: None,
    };
    let home = distribution.openasr_home.as_ref().unwrap().clone();
    let app = openasr_server::app_with_runtime_and_distribution(
        openasr_server::ServerRuntime::default(),
        distribution,
    );

    let initial = serde_json::json!({
        "default_model": "whisper-large-v3-turbo",
        "default_backend": "mock",
        "media": { "ffmpeg_bin": null },
        "preferences": {
            "version": openasr_core::config::PREFERENCES_SCHEMA_VERSION,
            "language": "zh-CN",
            "auto_save": true,
            "tray_icon": false,
            "dictation_shortcut": "Alt",
            "push_to_talk": true,
            "inference_threads": 8,
            "theme": "dark",
            "accent_color": "#2fa663",
            "density": "compact"
        }
    });
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("PUT")
                .uri("/v1/config")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(initial.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let response = app
        .oneshot(
            Request::builder()
                .method("PUT")
                .uri("/v1/config")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    serde_json::json!({ "preferences": { "diarize": true } }).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = to_bytes(response.into_body(), 1024 * 64).await.unwrap();
    let after: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(after["preferences"]["diarize"], true);
    assert_eq!(after["preferences"]["language"], "zh-CN");
    assert_eq!(after["preferences"]["auto_save"], true);
    assert_eq!(after["preferences"]["tray_icon"], false);
    assert_eq!(after["preferences"]["dictation_shortcut"], "Alt");
    assert_eq!(after["preferences"]["push_to_talk"], true);
    assert_eq!(after["preferences"]["inference_threads"], 8);
    assert_eq!(after["preferences"]["theme"], "dark");
    assert_eq!(after["preferences"]["accent_color"], "#2fa663");
    assert_eq!(after["preferences"]["density"], "compact");

    let file: Value =
        serde_json::from_slice(&std::fs::read(home.join("config.json")).unwrap()).unwrap();
    assert_eq!(file["preferences"]["diarize"], true);
    assert_eq!(file["preferences"]["dictation_shortcut"], "Alt");
}

#[tokio::test]
async fn capabilities_endpoint_exposes_transcription_capability_contract() {
    // Hermetic: point the app at an empty TempDir so realtime.translation.installed
    // is deterministically false on any machine, regardless of whether hymt2 (or
    // any other translation pack) is installed in the real OpenASR home.
    let temp = tempfile::tempdir().unwrap();
    let response = openasr_server::app_with_runtime_and_distribution(
        openasr_server::ServerRuntime::default(),
        openasr_server::DistributionRuntime {
            openasr_home: Some(temp.path().to_path_buf()),
            catalog_url: None,
            catalog_local_override: None,
        },
    )
    .oneshot(
        Request::builder()
            .uri("/v1/capabilities")
            .body(Body::empty())
            .unwrap(),
    )
    .await
    .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let bytes = to_bytes(response.into_body(), 1024 * 64).await.unwrap();
    let parsed: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(parsed["object"], "capabilities");
    assert_eq!(parsed["transcription"]["backend"], "mock");
    assert_eq!(parsed["transcription"]["diarization"]["supported"], false);
    assert_eq!(
        parsed["transcription"]["diarization"]["behavior"],
        "reject_request"
    );
    assert_eq!(
        parsed["transcription"]["word_timestamps"]["behavior"],
        "supported"
    );
    assert_eq!(
        parsed["transcription"]["inference_threads"]["behavior"],
        "supported"
    );
    assert_eq!(parsed["realtime"]["mode"], "file_per_utterance_fallback");
    assert_eq!(parsed["realtime"]["phrase_bias"]["supported"], false);
    assert_eq!(
        parsed["realtime"]["phrase_bias"]["behavior"],
        "reject_request"
    );
    assert_eq!(parsed["realtime"]["word_timestamps"]["supported"], true);
    assert_eq!(
        parsed["realtime"]["word_timestamps"]["behavior"],
        "supported"
    );
    assert_eq!(parsed["realtime"]["diarization"]["supported"], false);
    assert_eq!(
        parsed["realtime"]["diarization"]["behavior"],
        "reject_request"
    );
    assert_eq!(
        parsed["realtime"]["diarization"]["reason"],
        "Voice ID is available only for file transcription; realtime sessions do not support diarize=true or --diarize."
    );
    assert_eq!(parsed["realtime"]["translation"]["supported"], false);
    assert_eq!(parsed["realtime"]["translation"]["installed"], false);
    assert_eq!(
        parsed["realtime"]["translation"]["reason"],
        "translation_pack_missing"
    );
}

#[tokio::test]
async fn capabilities_endpoint_reflects_active_xasr_phrase_bias_capability() {
    let temp = tempfile::tempdir().unwrap();
    let pack_root = temp.path().join("xasr-capability.oasr");
    write_xasr_gguf_runtime_source(&pack_root, "xasr-capability");
    let app = openasr_server::app_with_runtime(openasr_server::ServerRuntime {
        backend: openasr_core::BackendKind::Native,
        native_execution: openasr_server::NativeExecutionSupervisor::default(),
        ffmpeg_bin: None,
        ffmpeg_bin_explicit: false,
        model_pack_path: Some(pack_root),
    });

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/capabilities")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let bytes = to_bytes(response.into_body(), 1024 * 64).await.unwrap();
    let parsed: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(parsed["transcription"]["backend"], "native");
    assert_eq!(parsed["transcription"]["phrase_bias"]["supported"], false);
    assert_eq!(parsed["realtime"]["mode"], "true_streaming");
    assert_eq!(parsed["realtime"]["phrase_bias"]["supported"], false);
    assert_eq!(parsed["realtime"]["supports_partial_results"], true);
    // xasr-zipformer is the only family running the frame-sync append-only
    // streaming driver; every other true-streaming family re-decodes a
    // buffer and must not claim this.
    assert_eq!(parsed["realtime"]["frame_sync_partials"], true);
}

#[tokio::test]
async fn config_endpoint_reports_malformed_stored_config_as_server_error() {
    let temp = tempfile::tempdir().unwrap();
    let home = temp.path().join("home");
    std::fs::create_dir_all(&home).unwrap();
    std::fs::write(home.join("config.json"), b"{not json").unwrap();
    let app = openasr_server::app_with_runtime_and_distribution(
        openasr_server::ServerRuntime::default(),
        openasr_server::DistributionRuntime {
            openasr_home: Some(home),
            catalog_url: None,
            catalog_local_override: None,
        },
    );

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/config")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    let bytes = to_bytes(response.into_body(), 1024 * 64).await.unwrap();
    let body = String::from_utf8_lossy(&bytes);
    assert!(body.contains("Could not read or update OpenASR config"));
}

#[tokio::test]
async fn transcription_degrades_to_defaults_when_stored_config_is_malformed() {
    let temp = tempfile::tempdir().unwrap();
    let home = temp.path().join("home");
    std::fs::create_dir_all(&home).unwrap();
    std::fs::write(home.join("config.json"), b"{not json").unwrap();
    let app = openasr_server::app_with_runtime_and_distribution(
        openasr_server::ServerRuntime::default(),
        openasr_server::DistributionRuntime {
            openasr_home: Some(home),
            catalog_url: None,
            catalog_local_override: None,
        },
    );

    // A malformed daemon config must NOT fail a well-formed transcription: the
    // request succeeds with default preferences. (The /v1/config endpoint still
    // surfaces the corruption — see the test above.)
    let response = app
        .oneshot(multipart_request(
            "whisper-large-v3-turbo",
            "sample.wav",
            b"not a real wav",
        ))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn native_transcription_without_installed_model_fails_closed_and_never_downloads() {
    let temp = tempfile::tempdir().unwrap();
    let home = temp.path().join("home");
    std::fs::create_dir_all(&home).unwrap();
    // Native backend, no explicit pack, isolated empty home: the server must
    // fail closed rather than auto-pull a model. The server never downloads --
    // consent-pull lives only in the CLI handlers, so this is structurally true,
    // and this test locks it as a safety invariant.
    let app = openasr_server::app_with_runtime_and_distribution(
        openasr_server::ServerRuntime {
            backend: openasr_core::BackendKind::Native,
            native_execution: openasr_server::NativeExecutionSupervisor::default(),
            ffmpeg_bin: None,
            ffmpeg_bin_explicit: false,
            model_pack_path: None,
        },
        openasr_server::DistributionRuntime {
            openasr_home: Some(home.clone()),
            catalog_url: None,
            catalog_local_override: None,
        },
    );

    let response = app
        .oneshot(multipart_request(
            "qwen3-asr-0.6b",
            "sample.wav",
            b"not a real wav",
        ))
        .await
        .unwrap();

    // A missing model is a client error (400), not a crash and not a silent
    // 500 -- the daemon is up, it just needs the caller to install a model
    // first (see also `daemon_starts_and_reports_ready_with_zero_models_installed`,
    // which locks in that the same empty-home startup itself never fails).
    assert_eq!(
        response.status(),
        StatusCode::BAD_REQUEST,
        "expected a fail-closed 400 for an uninstalled model"
    );
    let bytes = to_bytes(response.into_body(), 1024 * 64).await.unwrap();
    let json: Value = serde_json::from_slice(&bytes).unwrap();
    let message = json["error"]["message"].as_str().unwrap();
    assert!(
        message.contains("qwen3-asr-0.6b") && message.contains("not installed"),
        "expected a clear 'model not installed' message naming the model id, got: {message}"
    );
    assert!(
        !home.join("models").exists(),
        "the server must never download a model"
    );
}

#[tokio::test]
async fn daemon_starts_and_reports_ready_with_zero_models_installed() {
    // The core regression this guards: a fresh install with zero pulled
    // models must not prevent the daemon from starting at all. Previously
    // `ServerRuntime::validate()` (run before the HTTP listener ever binds)
    // hard-failed for the native backend whenever no model pack was resolved,
    // so a brand-new install's daemon process exited immediately and the
    // desktop app's health poll just timed out waiting for a process that was
    // already dead. Starting the app (which runs the same validation the real
    // `serve_with_launch_options` entrypoint runs) and hitting /health must
    // succeed and honestly report that no model is installed yet.
    let temp = tempfile::tempdir().unwrap();
    let home = temp.path().join("home");
    std::fs::create_dir_all(&home).unwrap();
    let runtime = openasr_server::ServerRuntime {
        backend: openasr_core::BackendKind::Native,
        native_execution: openasr_server::NativeExecutionSupervisor::default(),
        ffmpeg_bin: None,
        ffmpeg_bin_explicit: false,
        model_pack_path: None,
    };
    runtime
        .validate()
        .expect("serve must not fail closed at startup just because zero models are installed");

    let app = openasr_server::app_with_runtime_and_distribution(
        runtime,
        openasr_server::DistributionRuntime {
            openasr_home: Some(home),
            catalog_url: None,
            catalog_local_override: None,
        },
    );

    let response = app
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let bytes = to_bytes(response.into_body(), 1024 * 64).await.unwrap();
    let parsed: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(parsed["status"], "ok");
    assert_eq!(
        parsed["model_installed"], false,
        "health must honestly report no model bound instead of just being unreachable"
    );
    assert_eq!(
        parsed["model_resident"], false,
        "nothing can be resident when no model is bound at all"
    );
    // Additive debug-observability fields (see `HealthResponse`'s doc
    // comments): with `model_pack_path: None`, `spawn_boot_native_warmup`
    // returns immediately without ever entering the native activity
    // tracker, so this fresh-in-process tracker reads deterministically
    // zero on both.
    assert_eq!(
        parsed["native_active_count"], 0,
        "nothing ever entered the native activity tracker when no model is bound"
    );
    assert_eq!(
        parsed["idle_seconds"], 0,
        "a tracker that has never seen any activity reads zero idle seconds, not stale/uninitialized"
    );
    assert_eq!(
        parsed["abandoned_worker_count"], 0,
        "no decode worker has hung, so the fail-loud abandonment counter reads zero"
    );
}

#[tokio::test]
async fn health_reports_model_bound_but_not_resident_before_any_load() {
    // A native pack can be bound (`model_installed: true`) at boot without
    // its runtime ever having been loaded yet -- the boot warm-up runs in
    // the background and this test never triggers it or any transcription.
    // `/health` must not conflate "bound" with "resident": a client polling
    // right after daemon start must see `model_resident: false` until an
    // actual load (warm-up or first request) completes.
    let temp = tempfile::tempdir().unwrap();
    let pack_root = temp.path().join("native-pack.oasr");
    write_mock_gguf_runtime_source(&pack_root, None);
    let app = openasr_server::app_with_runtime(openasr_server::ServerRuntime {
        backend: openasr_core::BackendKind::Native,
        native_execution: openasr_server::NativeExecutionSupervisor::default(),
        ffmpeg_bin: None,
        ffmpeg_bin_explicit: false,
        model_pack_path: Some(pack_root),
    });

    let response = app
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let bytes = to_bytes(response.into_body(), 1024 * 64).await.unwrap();
    let parsed: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(parsed["model_installed"], true, "a model pack is bound");
    assert_eq!(
        parsed["model_resident"], false,
        "a bound pack whose runtime has never been loaded this boot must not read resident"
    );
    // Additive debug-observability fields: unlike the zero-models-installed
    // test above, a background boot warm-up is spawned here (even though it
    // never completes a load in time to flip `model_resident`), so this only
    // pins the wire shape/type -- not an exact count/duration, which would
    // race the warm-up task's own scheduling.
    assert!(
        parsed["native_active_count"].is_u64(),
        "native_active_count must serialize as a JSON number"
    );
    assert!(
        parsed["idle_seconds"].is_u64(),
        "idle_seconds must serialize as a JSON number"
    );
    assert!(
        parsed["abandoned_worker_count"].is_u64(),
        "abandoned_worker_count must serialize as a JSON number"
    );
}

#[tokio::test]
async fn transcription_succeeds_when_history_cannot_be_recorded() {
    let temp = tempfile::tempdir().unwrap();
    // OPENASR_HOME points at a *file*, so the history store cannot create its
    // directory tree under it. `create_dir_all` fails for any user (root cannot
    // create a directory inside a regular file either), deterministically
    // exercising the history-write-failure path regardless of CI uid.
    let home = temp.path().join("home-as-file");
    std::fs::write(&home, b"not a directory").unwrap();
    let app = openasr_server::app_with_runtime_and_distribution(
        openasr_server::ServerRuntime::default(),
        openasr_server::DistributionRuntime {
            openasr_home: Some(home),
            catalog_url: None,
            catalog_local_override: None,
        },
    );

    // History is a best-effort audit side-write: its failure must not fail an
    // otherwise-successful transcription (this is the Docker-smoke 500 fix).
    let response = app
        .oneshot(multipart_request(
            "whisper-small",
            "sample.wav",
            b"not a real wav",
        ))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn pull_job_from_local_pack_installs_streams_and_deletes() {
    let temp = tempfile::tempdir().unwrap();
    let (source_pack, distribution) = write_moonshine_pull_fixture(temp.path());
    let app = openasr_server::app_with_runtime_and_distribution(
        openasr_server::ServerRuntime::default(),
        distribution,
    );

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/models/moonshine-tiny/pull")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    serde_json::json!({ "quant": "q8", "from": source_pack }).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::ACCEPTED);
    let bytes = to_bytes(response.into_body(), 1024 * 64).await.unwrap();
    let started: Value = serde_json::from_slice(&bytes).unwrap();
    let job_id = started["job_id"].as_str().unwrap();
    assert_eq!(started["source_path"], source_pack.to_str().unwrap());

    let completed = wait_for_terminal_job(app.clone(), job_id).await;
    assert_eq!(completed["state"], "completed");
    assert_eq!(completed["pull"], "moonshine-tiny:q8");
    assert_installed_content_object(completed["installed_path"].as_str().unwrap());

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/models/local")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = to_bytes(response.into_body(), 1024 * 64).await.unwrap();
    let parsed: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(parsed["data"][0]["pull"], "moonshine-tiny:q8");

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri(format!("/v1/models/pull/{job_id}/events"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get(header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok()),
        Some("text/event-stream")
    );
    let bytes = to_bytes(response.into_body(), 1024 * 64).await.unwrap();
    let body = String::from_utf8_lossy(&bytes);
    assert!(body.contains("event: snapshot"));
    assert!(body.contains("\"state\":\"completed\""));

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/models/moonshine-tiny/pull")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::json!({ "quant": "q8" }).to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = to_bytes(response.into_body(), 1024 * 64).await.unwrap();
    let parsed: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(parsed["state"], "already_installed");
    let already_installed_job_id = parsed["job_id"].as_str().unwrap();

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(format!("/v1/models/pull/{already_installed_job_id}/cancel"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = to_bytes(response.into_body(), 1024 * 64).await.unwrap();
    let parsed: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(parsed["state"], "already_installed");

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(format!("/v1/models/pull/{already_installed_job_id}/pause"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = to_bytes(response.into_body(), 1024 * 64).await.unwrap();
    let parsed: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(parsed["state"], "already_installed");

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(format!("/v1/models/pull/{already_installed_job_id}/resume"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = to_bytes(response.into_body(), 1024 * 64).await.unwrap();
    let parsed: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(parsed["state"], "already_installed");

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri("/v1/models/moonshine-tiny:q8")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = to_bytes(response.into_body(), 1024 * 64).await.unwrap();
    let parsed: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(parsed["deleted"], true);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/models/local")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = to_bytes(response.into_body(), 1024 * 64).await.unwrap();
    let parsed: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(parsed["data"].as_array().unwrap().len(), 0);
}

#[tokio::test]
async fn pull_job_events_stream_live_updates_while_pause_cancel_race_sets_flags() {
    let temp = tempfile::tempdir().unwrap();
    let (source_pack, distribution) = write_moonshine_pull_fixture(temp.path());
    pad_pull_fixture_pack_to(&source_pack, &distribution, LIVE_PULL_FIXTURE_SIZE_BYTES);
    let app = openasr_server::app_with_runtime_and_distribution(
        openasr_server::ServerRuntime::default(),
        distribution,
    );

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/models/moonshine-tiny/pull")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    serde_json::json!({ "quant": "q8", "from": source_pack }).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::ACCEPTED);
    let bytes = to_bytes(response.into_body(), 1024 * 64).await.unwrap();
    let started: Value = serde_json::from_slice(&bytes).unwrap();
    let job_id = started["job_id"].as_str().unwrap().to_string();

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri(format!("/v1/models/pull/{job_id}/events"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get(header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok()),
        Some("text/event-stream")
    );

    let mut event_stream = response.into_body().into_data_stream();
    let first_chunk = tokio::time::timeout(Duration::from_secs(5), event_stream.next())
        .await
        .expect("timed out waiting for first pull SSE event")
        .expect("pull SSE stream ended before first event")
        .expect("pull SSE body error");
    let mut events = String::from_utf8_lossy(&first_chunk).into_owned();
    assert!(events.contains("event: snapshot"));

    let pause_uri = format!("/v1/models/pull/{job_id}/pause");
    let cancel_uri = format!("/v1/models/pull/{job_id}/cancel");
    let ((pause_status, pause_body), (cancel_status, cancel_body)) = tokio::join!(
        post_pull_control(app.clone(), pause_uri),
        post_pull_control(app.clone(), cancel_uri),
    );
    assert_eq!(
        pause_status,
        StatusCode::ACCEPTED,
        "pause body: {pause_body}"
    );
    assert_eq!(
        cancel_status,
        StatusCode::ACCEPTED,
        "cancel body: {cancel_body}"
    );

    tokio::time::timeout(Duration::from_secs(30), async {
        while let Some(chunk) = event_stream.next().await {
            let chunk = chunk.expect("pull SSE body error");
            events.push_str(&String::from_utf8_lossy(&chunk));
        }
    })
    .await
    .expect("timed out waiting for terminal pull SSE event");

    let snapshot_events = events.matches("event: snapshot").count();
    assert!(
        snapshot_events > 1,
        "expected live SSE updates beyond the immediate snapshot, got {snapshot_events}: {events}"
    );
    assert!(
        events.contains("\"control_requested\":\"pause\"")
            || events.contains("\"control_requested\":\"cancel\"")
            || events.contains("Pause requested.")
            || events.contains("Cancellation requested."),
        "expected pause/cancel control state in streamed snapshots: {events}"
    );
    assert!(
        events.contains("\"state\":\"completed\"")
            || events.contains("\"state\":\"canceled\"")
            || events.contains("\"state\":\"failed\""),
        "expected terminal pull state in streamed snapshots: {events}"
    );
}

#[tokio::test]
async fn pull_request_catalog_url_body_field_is_ignored() {
    let temp = tempfile::tempdir().unwrap();
    let (source_pack, distribution) = write_moonshine_pull_fixture(temp.path());
    let untrusted_catalog = temp.path().join("untrusted-catalog.json");
    std::fs::write(
        &untrusted_catalog,
        serde_json::to_vec_pretty(&serde_json::json!({
            "schema_version": 1,
            "generated_at": "2026-05-31T00:00:00Z",
            "catalog_url": "file://untrusted-catalog.json",
            "models": []
        }))
        .unwrap(),
    )
    .unwrap();
    let app = openasr_server::app_with_runtime_and_distribution(
        openasr_server::ServerRuntime::default(),
        distribution,
    );

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/models/moonshine-tiny/pull")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    serde_json::json!({
                        "quant": "q8",
                        "from": source_pack,
                        "catalog_url": format!("file://{}", untrusted_catalog.display())
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::ACCEPTED);
    let bytes = to_bytes(response.into_body(), 1024 * 64).await.unwrap();
    let started: Value = serde_json::from_slice(&bytes).unwrap();
    let completed = wait_for_terminal_job(app, started["job_id"].as_str().unwrap()).await;
    assert_eq!(completed["state"], "completed");
    assert_eq!(completed["pull"], "moonshine-tiny:q8");
}

#[tokio::test]
async fn content_addressed_refs_drive_local_and_default_model_endpoints() {
    let temp = tempfile::tempdir().unwrap();
    let home = temp.path().join("home");
    let object = write_content_addressed_moonshine_ref(&home);
    std::fs::write(
        home.join("config.json"),
        r#"{"default_model":"moonshine-tiny"}"#,
    )
    .unwrap();
    let app = openasr_server::app_with_runtime_and_distribution(
        openasr_server::ServerRuntime::default(),
        openasr_server::DistributionRuntime {
            openasr_home: Some(home.clone()),
            catalog_url: None,
            catalog_local_override: None,
        },
    );

    let response = app
        .clone()
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/models/local")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let local: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), 1024 * 64).await.unwrap()).unwrap();
    assert_eq!(local["data"][0]["pull"], "moonshine-tiny:q8");
    assert_eq!(local["data"][0]["path"], object.display().to_string());
    assert_eq!(local["data"][0]["is_default"], true);

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/models/default")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let default: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), 1024 * 64).await.unwrap()).unwrap();
    assert_eq!(default["default_model_status"], "installed");
    assert_eq!(default["default_pull"], "moonshine-tiny:q8");

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri("/v1/models/moonshine-tiny:q8")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert!(
        !object.exists(),
        "the last ref naming an object collects it; nothing else referenced this digest"
    );
    assert!(!home.join("models/refs/moonshine-tiny/q8_0.json").exists());

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/models/local")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let local: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), 1024 * 64).await.unwrap()).unwrap();
    assert!(local["data"].as_array().unwrap().is_empty());
}

#[tokio::test]
async fn default_model_endpoint_marks_local_pack_and_clears_default_on_delete() {
    let temp = tempfile::tempdir().unwrap();
    let (source_pack, distribution) = write_moonshine_pull_fixture(temp.path());
    let home = distribution.openasr_home.as_ref().unwrap().clone();
    let app = openasr_server::app_with_runtime_and_distribution(
        openasr_server::ServerRuntime::default(),
        distribution,
    );

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/models/moonshine-tiny/pull")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    serde_json::json!({ "quant": "q8", "from": source_pack }).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::ACCEPTED);
    let bytes = to_bytes(response.into_body(), 1024 * 64).await.unwrap();
    let started: Value = serde_json::from_slice(&bytes).unwrap();
    let job_id = started["job_id"].as_str().unwrap();
    let completed = wait_for_terminal_job(app.clone(), job_id).await;
    assert_eq!(completed["state"], "completed");

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/models/default")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    serde_json::json!({ "pull": "moonshine-tiny:q8" }).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = to_bytes(response.into_body(), 1024 * 64).await.unwrap();
    let parsed: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(parsed["object"], "model.default");
    assert_eq!(parsed["default_model"], "moonshine-tiny");
    assert_eq!(parsed["default_pull"], "moonshine-tiny:q8");
    assert_eq!(parsed["pack"]["pull"], "moonshine-tiny:q8");

    let config: Value =
        serde_json::from_slice(&std::fs::read(home.join("config.json")).unwrap()).unwrap();
    assert_eq!(config["default_model"], "moonshine-tiny");
    let pointer: Value =
        serde_json::from_slice(&std::fs::read(home.join("default.json")).unwrap()).unwrap();
    assert_eq!(pointer["pull"], "moonshine-tiny:q8");

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/models/local")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = to_bytes(response.into_body(), 1024 * 64).await.unwrap();
    let parsed: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(parsed["data"][0]["pull"], "moonshine-tiny:q8");
    assert_eq!(parsed["data"][0]["is_default"], true);

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/models/default")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = to_bytes(response.into_body(), 1024 * 64).await.unwrap();
    let parsed: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(parsed["default_pull"], "moonshine-tiny:q8");

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri("/v1/models/moonshine-tiny:q8")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = to_bytes(response.into_body(), 1024 * 64).await.unwrap();
    let parsed: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(parsed["deleted"], true);
    assert_eq!(parsed["pack"]["pull"], "moonshine-tiny:q8");

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/models/local")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = to_bytes(response.into_body(), 1024 * 64).await.unwrap();
    let parsed: Value = serde_json::from_slice(&bytes).unwrap();
    assert!(parsed["data"].as_array().unwrap().is_empty());

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/models/default")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = to_bytes(response.into_body(), 1024 * 64).await.unwrap();
    let parsed: Value = serde_json::from_slice(&bytes).unwrap();
    assert!(parsed["default_model"].is_null());
    assert!(parsed["default_pull"].is_null());
    assert!(parsed["pack"].is_null());

    let config: Value =
        serde_json::from_slice(&std::fs::read(home.join("config.json")).unwrap()).unwrap();
    assert!(config["default_model"].is_null());
    assert!(!home.join("default.json").exists());
}

#[tokio::test]
async fn default_model_endpoint_rejects_uninstalled_pack() {
    let temp = tempfile::tempdir().unwrap();
    let (_, distribution) = write_moonshine_pull_fixture(temp.path());
    let app = openasr_server::app_with_runtime_and_distribution(
        openasr_server::ServerRuntime::default(),
        distribution,
    );

    let response = app
        .oneshot(
            Request::builder()
                .method("PUT")
                .uri("/v1/models/default")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    serde_json::json!({ "pull": "moonshine-tiny:q8" }).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let bytes = to_bytes(response.into_body(), 1024 * 64).await.unwrap();
    let parsed: Value = serde_json::from_slice(&bytes).unwrap();
    assert!(
        parsed["error"]["message"]
            .as_str()
            .unwrap()
            .contains("Installed model pack not found")
    );
}

#[tokio::test]
async fn pull_job_snapshot_survives_app_recreation() {
    let temp = tempfile::tempdir().unwrap();
    let (source_pack, distribution) = write_moonshine_pull_fixture(temp.path());
    let app = openasr_server::app_with_runtime_and_distribution(
        openasr_server::ServerRuntime::default(),
        distribution.clone(),
    );

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/models/moonshine-tiny/pull")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    serde_json::json!({ "quant": "q8", "from": source_pack }).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::ACCEPTED);
    let bytes = to_bytes(response.into_body(), 1024 * 64).await.unwrap();
    let started: Value = serde_json::from_slice(&bytes).unwrap();
    let job_id = started["job_id"].as_str().unwrap().to_string();
    let completed = wait_for_terminal_job(app, &job_id).await;
    assert_eq!(completed["state"], "completed");

    let recreated = openasr_server::app_with_runtime_and_distribution(
        openasr_server::ServerRuntime::default(),
        distribution,
    );
    let persisted = job_snapshot(recreated, &job_id).await;
    assert_eq!(persisted["state"], "completed");
    assert_eq!(persisted["pull"], "moonshine-tiny:q8");
}

#[tokio::test]
async fn interrupted_pull_job_resumes_after_app_recreation() {
    let temp = tempfile::tempdir().unwrap();
    let (source_pack, distribution) = write_moonshine_pull_fixture(temp.path());
    let home = distribution.openasr_home.as_ref().unwrap().clone();
    let bytes_total = write_complete_moonshine_partial(&home, &source_pack);
    write_persisted_pull_job_with_resolved(
        &home,
        "pull-restart-resume",
        "verifying",
        bytes_total,
        bytes_total,
        &source_pack,
    );

    let app = openasr_server::app_with_runtime_and_distribution(
        openasr_server::ServerRuntime::default(),
        distribution,
    );
    let completed = wait_for_terminal_job(app.clone(), "pull-restart-resume").await;
    assert_eq!(completed["state"], "completed");
    assert_eq!(completed["pull"], "moonshine-tiny:q8");
    assert_installed_content_object(completed["installed_path"].as_str().unwrap());

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/models/local")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = to_bytes(response.into_body(), 1024 * 64).await.unwrap();
    let parsed: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(parsed["data"][0]["pull"], "moonshine-tiny:q8");
}

#[tokio::test]
async fn interrupted_pull_job_resume_uses_persisted_resolved_spec_not_mutable_catalog() {
    let temp = tempfile::tempdir().unwrap();
    let (source_pack, distribution) = write_moonshine_pull_fixture(temp.path());
    let home = distribution.openasr_home.as_ref().unwrap().clone();
    let bytes_total = write_complete_moonshine_partial(&home, &source_pack);
    write_persisted_pull_job_with_resolved(
        &home,
        "pull-resume-stable-spec",
        "verifying",
        bytes_total,
        bytes_total,
        &source_pack,
    );
    mutate_fixture_catalog_pack_identity(&distribution);

    let app = openasr_server::app_with_runtime_and_distribution(
        openasr_server::ServerRuntime::default(),
        distribution,
    );
    let completed = wait_for_terminal_job(app.clone(), "pull-resume-stable-spec").await;

    assert_eq!(completed["state"], "completed");
    assert_eq!(
        completed["resolved"]["hf_revision"],
        "0123456789abcdef0123456789abcdef01234567"
    );
    assert_eq!(completed["resolved"]["size_bytes"], bytes_total);
    assert_installed_content_object(completed["installed_path"].as_str().unwrap());
}

#[tokio::test]
async fn interrupted_local_source_pull_job_resumes_from_persisted_source_path_after_app_recreation()
{
    let temp = tempfile::tempdir().unwrap();
    let (source_pack, distribution) = write_moonshine_pull_fixture(temp.path());
    let home = distribution.openasr_home.as_ref().unwrap().clone();
    let bytes_total = std::fs::metadata(&source_pack).unwrap().len();
    write_persisted_local_source_pull_job_with_resolved(
        &home,
        "pull-local-restart-resume",
        "verifying",
        0,
        bytes_total,
        &source_pack,
    );

    let app = openasr_server::app_with_runtime_and_distribution(
        openasr_server::ServerRuntime::default(),
        distribution,
    );
    let completed = wait_for_terminal_job(app, "pull-local-restart-resume").await;

    assert_eq!(completed["state"], "completed");
    assert_eq!(completed["source_path"], source_pack.to_str().unwrap());
    assert_installed_content_object(completed["installed_path"].as_str().unwrap());
}

#[tokio::test]
async fn paused_local_source_pull_job_manual_resume_uses_persisted_source_path() {
    let temp = tempfile::tempdir().unwrap();
    let (source_pack, distribution) = write_moonshine_pull_fixture(temp.path());
    let home = distribution.openasr_home.as_ref().unwrap().clone();
    let bytes_total = std::fs::metadata(&source_pack).unwrap().len();
    write_persisted_local_source_pull_job_with_resolved(
        &home,
        "pull-local-manual-resume",
        "paused",
        0,
        bytes_total,
        &source_pack,
    );

    let app = openasr_server::app_with_runtime_and_distribution(
        openasr_server::ServerRuntime::default(),
        distribution,
    );
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/models/pull/pull-local-manual-resume/resume")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::ACCEPTED);
    let bytes = to_bytes(response.into_body(), 1024 * 64).await.unwrap();
    let resumed: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(resumed["state"], "queued");
    assert_eq!(resumed["source_path"], source_pack.to_str().unwrap());

    let completed = wait_for_terminal_job(app, "pull-local-manual-resume").await;
    assert_eq!(completed["state"], "completed");
    assert_eq!(completed["source_path"], source_pack.to_str().unwrap());
}

#[tokio::test]
async fn restart_resumable_job_without_resolved_spec_fails_closed() {
    let temp = tempfile::tempdir().unwrap();
    let (_, distribution) = write_moonshine_pull_fixture(temp.path());
    let home = distribution.openasr_home.as_ref().unwrap().clone();
    write_persisted_pull_job(&home, "pull-old-snapshot", "verifying", 4, 10);

    let app = openasr_server::app_with_runtime_and_distribution(
        openasr_server::ServerRuntime::default(),
        distribution,
    );
    let failed = job_snapshot(app, "pull-old-snapshot").await;

    assert_eq!(failed["state"], "failed");
    assert!(
        failed["error"]
            .as_str()
            .unwrap()
            .contains("Refusing to re-resolve the mutable catalog")
    );
}

#[tokio::test]
async fn paused_pull_job_is_not_resumed_after_app_recreation() {
    let temp = tempfile::tempdir().unwrap();
    let (_, distribution) = write_moonshine_pull_fixture(temp.path());
    let home = distribution.openasr_home.as_ref().unwrap().clone();
    write_persisted_pull_job(&home, "pull-paused", "paused", 4, 10);

    let app = openasr_server::app_with_runtime_and_distribution(
        openasr_server::ServerRuntime::default(),
        distribution,
    );
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    let paused = job_snapshot(app, "pull-paused").await;
    assert_eq!(paused["state"], "paused");
    assert_eq!(paused["bytes_done"], 4);
}

fn multipart_request(model: &str, file_name: &str, bytes: &[u8]) -> Request<Body> {
    multipart_request_with_diarize(model, file_name, bytes, false)
}

/// Seeds a single Voice ID person directly through `VoiceIdStore::enroll_person`
/// -- the only enrollment path left post-migration-removal. Tests that just
/// need *a* person to exercise PATCH/list/rename semantics call this instead
/// of routing through HTTP enrollment (which requires a real embedder pack).
fn seed_voice_id_person(home: &std::path::Path) -> (String, u64) {
    let space = openasr_core::diarize::voice_id::EmbeddingSpace::from_parts(
        2,
        "sha256:test",
        "test",
        "test",
        "v1",
        openasr_core::diarize::voice_id::REDIMNET_FRONTEND_VERSION,
        openasr_core::diarize::calibration::REDIMNET_CALIBRATION_VERSION,
        openasr_core::diarize::voice_id::MATCHER_POLICY_VERSION,
    );
    let sample = openasr_core::diarize::voice_id::NewSampleInput {
        sample_id: openasr_core::diarize::voice_id::SampleId::generate(),
        capture_context: openasr_core::diarize::voice_id::CaptureContext {
            device_class: "test".to_string(),
            input_route: "mic".to_string(),
            environment_hint: None,
            sample_label: Some("clip".to_string()),
        },
        quality: openasr_core::diarize::voice_id::SampleQuality {
            speech_seconds: 5.25,
            snr_estimate: 20.0,
            clipping_ratio: 0.0,
            vad_coverage: 0.8,
            accepted_reason: "test".to_string(),
        },
        space,
        embedding: openasr_core::diarize::contract::SpeakerEmbedding::l2_normalized(vec![1.0, 0.0]),
    };
    let consent = openasr_core::diarize::voice_id::ConsentRecord {
        granted_at: openasr_core::diarize::voice_id::timestamp_now(),
        notice_version: "voice-id-notice-v1".to_string(),
        capture_method: "test".to_string(),
    };
    let store = openasr_core::diarize::voice_id::VoiceIdStore::open_checked(home).unwrap();
    let person = store
        .enroll_person("Alice", consent, vec![sample], None)
        .unwrap();
    (person.person_id, person.revision)
}

#[tokio::test]
async fn legacy_speaker_routes_are_not_registered() {
    let app = openasr_server::app_with_runtime_and_distribution(
        openasr_server::ServerRuntime::default(),
        openasr_server::DistributionRuntime::default(),
    );
    for (method, uri) in [
        ("GET", "/v1/speakers"),
        ("POST", "/v1/speakers"),
        ("PATCH", "/v1/speakers/vp_aaaaaaaaaaaaaaaa"),
        ("DELETE", "/v1/speakers/vp_aaaaaaaaaaaaaaaa"),
        ("POST", "/v1/speakers/vp_aaaaaaaaaaaaaaaa/reenroll"),
    ] {
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(method)
                    .uri(uri)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND, "{method} {uri}");
    }
}

#[tokio::test]
async fn voice_id_routes_require_operator_credentials_for_paired_devices() {
    let temp = tempfile::tempdir().unwrap();
    let app = openasr_server::app_with_runtime_and_distribution_and_launch_options(
        openasr_server::ServerRuntime::default(),
        openasr_server::DistributionRuntime {
            openasr_home: Some(temp.path().join("home")),
            catalog_url: None,
            catalog_local_override: None,
        },
        openasr_server::ServerLaunchOptions {
            auth: openasr_server::ServerAuth::pairing("admin-secret"),
            ..Default::default()
        },
    );
    let (_device_id, bearer_token) =
        create_approved_pairing_credential(&app, "Remote Compute Mac").await;

    let paired = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/voice-id/persons")
                .header(header::AUTHORIZATION, format!("Bearer {bearer_token}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(paired.status(), StatusCode::FORBIDDEN);

    let missing = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/voice-id/persons")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert!(
        missing.status() == StatusCode::UNAUTHORIZED || missing.status() == StatusCode::FORBIDDEN,
        "unauthenticated voice-id access must be 401/403, got {}",
        missing.status()
    );

    let operator = app
        .oneshot(
            Request::builder()
                .uri("/v1/voice-id/persons")
                .header(header::AUTHORIZATION, "Bearer admin-secret")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(operator.status(), StatusCode::OK);
}

#[tokio::test]
async fn voice_id_rename_revision_conflict_returns_409() {
    let temp = tempfile::tempdir().unwrap();
    let home = temp.path().join("home");
    // Seed a person directly via the store so rename has a target without
    // needing the embedder pack.
    let (person_id, revision) = seed_voice_id_person(&home);
    let app = openasr_server::app_with_runtime_and_distribution(
        openasr_server::ServerRuntime::default(),
        openasr_server::DistributionRuntime {
            openasr_home: Some(home),
            catalog_url: None,
            catalog_local_override: None,
        },
    );

    let conflict = app
        .oneshot(
            Request::builder()
                .method("PATCH")
                .uri(format!("/v1/voice-id/persons/{person_id}"))
                .header(header::CONTENT_TYPE, "application/json")
                .header(header::IF_MATCH, format!("\"{}\"", revision + 99))
                .body(Body::from(
                    serde_json::json!({ "display_name": "Alicia" }).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(conflict.status(), StatusCode::CONFLICT);
    let body = to_bytes(conflict.into_body(), 1024 * 64).await.unwrap();
    let json: Value = serde_json::from_slice(&body).unwrap();
    let message = json["error"]["message"].as_str().unwrap_or_default();
    assert!(
        message.to_ascii_lowercase().contains("revision")
            || message.to_ascii_lowercase().contains("conflict"),
        "unexpected conflict message: {message}"
    );
}

#[tokio::test]
async fn voice_id_person_patch_supports_atomic_name_and_color_edits() {
    let temp = tempfile::tempdir().unwrap();
    let home = temp.path().join("home");
    let (person_id, revision) = seed_voice_id_person(&home);
    let app = openasr_server::app_with_runtime_and_distribution(
        openasr_server::ServerRuntime::default(),
        openasr_server::DistributionRuntime {
            openasr_home: Some(home),
            catalog_url: None,
            catalog_local_override: None,
        },
    );

    let color_only = app
        .clone()
        .oneshot(
            Request::builder()
                .method("PATCH")
                .uri(format!("/v1/voice-id/persons/{person_id}"))
                .header(header::CONTENT_TYPE, "application/json")
                .header(header::IF_MATCH, format!("\"{revision}\""))
                .body(Body::from(r#"{"color_preference":"purple"}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(color_only.status(), StatusCode::OK);
    assert_eq!(color_only.headers().get(header::ETAG).unwrap(), "\"2\"");
    let color_body = to_bytes(color_only.into_body(), 1024 * 64).await.unwrap();
    let color_json: Value = serde_json::from_slice(&color_body).unwrap();
    assert_eq!(color_json["display_name"], "Alice");
    assert_eq!(color_json["color_preference"], "purple");
    assert_eq!(color_json["revision"], 2);

    let name_only = app
        .clone()
        .oneshot(
            Request::builder()
                .method("PATCH")
                .uri(format!("/v1/voice-id/persons/{person_id}"))
                .header(header::CONTENT_TYPE, "application/json")
                .header(header::IF_MATCH, "\"2\"")
                .body(Body::from(r#"{"display_name":"Alicia"}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(name_only.status(), StatusCode::OK);
    let name_body = to_bytes(name_only.into_body(), 1024 * 64).await.unwrap();
    let name_json: Value = serde_json::from_slice(&name_body).unwrap();
    assert_eq!(name_json["display_name"], "Alicia");
    assert_eq!(name_json["color_preference"], "purple");
    assert_eq!(name_json["revision"], 3);

    let both = app
        .clone()
        .oneshot(
            Request::builder()
                .method("PATCH")
                .uri(format!("/v1/voice-id/persons/{person_id}"))
                .header(header::CONTENT_TYPE, "application/json")
                .header(header::IF_MATCH, "\"3\"")
                .body(Body::from(
                    r#"{"display_name":"Alice Example","color_preference":"green"}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(both.status(), StatusCode::OK);
    assert_eq!(both.headers().get(header::ETAG).unwrap(), "\"4\"");
    let both_body = to_bytes(both.into_body(), 1024 * 64).await.unwrap();
    let both_json: Value = serde_json::from_slice(&both_body).unwrap();
    assert_eq!(both_json["display_name"], "Alice Example");
    assert_eq!(both_json["color_preference"], "green");
    assert_eq!(both_json["revision"], 4);
    assert!(both_json["samples"].is_array());

    for body in [
        r#"{}"#,
        r##"{"color_preference":"#12ab34"}"##,
        r#"{"display_name":"   "}"#,
    ] {
        let invalid = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("PATCH")
                    .uri(format!("/v1/voice-id/persons/{person_id}"))
                    .header(header::CONTENT_TYPE, "application/json")
                    .header(header::IF_MATCH, "\"4\"")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(invalid.status(), StatusCode::BAD_REQUEST);
    }

    let stale = app
        .clone()
        .oneshot(
            Request::builder()
                .method("PATCH")
                .uri(format!("/v1/voice-id/persons/{person_id}"))
                .header(header::CONTENT_TYPE, "application/json")
                .header(header::IF_MATCH, "\"3\"")
                .body(Body::from(r#"{"display_name":"Stale"}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(stale.status(), StatusCode::CONFLICT);

    let cleared = app
        .clone()
        .oneshot(
            Request::builder()
                .method("PATCH")
                .uri(format!("/v1/voice-id/persons/{person_id}"))
                .header(header::CONTENT_TYPE, "application/json")
                .header(header::IF_MATCH, "\"4\"")
                .body(Body::from(r#"{"color_preference":null}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(cleared.status(), StatusCode::OK);
    assert_eq!(cleared.headers().get(header::ETAG).unwrap(), "\"5\"");
    let cleared_body = to_bytes(cleared.into_body(), 1024 * 64).await.unwrap();
    let cleared_json: Value = serde_json::from_slice(&cleared_body).unwrap();
    assert!(cleared_json.get("color_preference").is_none());

    let omitted = app
        .oneshot(
            Request::builder()
                .method("PATCH")
                .uri(format!("/v1/voice-id/persons/{person_id}"))
                .header(header::CONTENT_TYPE, "application/json")
                .header(header::IF_MATCH, "\"5\"")
                .body(Body::from(r#"{"display_name":"Alice Final"}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(omitted.status(), StatusCode::OK);
    let omitted_body = to_bytes(omitted.into_body(), 1024 * 64).await.unwrap();
    let omitted_json: Value = serde_json::from_slice(&omitted_body).unwrap();
    assert_eq!(omitted_json["display_name"], "Alice Final");
    assert!(omitted_json.get("color_preference").is_none());
}

#[tokio::test]
async fn voice_id_sample_patch_updates_only_label_with_owner_etag() {
    let temp = tempfile::tempdir().unwrap();
    let home = temp.path().join("home");
    seed_voice_id_person(&home);
    let app = openasr_server::app_with_runtime_and_distribution(
        openasr_server::ServerRuntime::default(),
        openasr_server::DistributionRuntime {
            openasr_home: Some(home),
            catalog_url: None,
            catalog_local_override: None,
        },
    );

    // The sample-level PATCH tested here needs the sample_id/quality fields,
    // which seed_voice_id_person's direct return doesn't carry, so fetch them
    // through the same list endpoint the server exposes to real clients.
    let list = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/voice-id/persons")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let list_body = to_bytes(list.into_body(), 1024 * 64).await.unwrap();
    let list_json: Value = serde_json::from_slice(&list_body).unwrap();
    let person = &list_json["data"][0];
    let revision = person["revision"].as_u64().unwrap();
    let sample_id = person["samples"][0]["sample_id"]
        .as_str()
        .unwrap()
        .to_string();
    let quality = person["samples"][0]["quality"].clone();

    let renamed = app
        .clone()
        .oneshot(
            Request::builder()
                .method("PATCH")
                .uri(format!("/v1/voice-id/samples/{sample_id}"))
                .header(header::CONTENT_TYPE, "application/json")
                .header(header::IF_MATCH, format!("\"{revision}\""))
                .body(Body::from(r#"{"sample_label":"Desk microphone"}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(renamed.status(), StatusCode::OK);
    assert_eq!(renamed.headers().get(header::ETAG).unwrap(), "\"2\"");
    let renamed_body = to_bytes(renamed.into_body(), 1024 * 64).await.unwrap();
    let renamed_json: Value = serde_json::from_slice(&renamed_body).unwrap();
    assert_eq!(renamed_json["revision"], 2);
    assert_eq!(
        renamed_json["samples"][0]["sample_label"],
        "Desk microphone"
    );
    assert_eq!(
        renamed_json["samples"][0]["capture_context"]["sample_label"],
        "Desk microphone"
    );
    assert_eq!(renamed_json["samples"][0]["quality"], quality);
    assert!(renamed_json["samples"].is_array());

    let blank = app
        .clone()
        .oneshot(
            Request::builder()
                .method("PATCH")
                .uri(format!("/v1/voice-id/samples/{sample_id}"))
                .header(header::CONTENT_TYPE, "application/json")
                .header(header::IF_MATCH, "\"2\"")
                .body(Body::from(r#"{"sample_label":"  "}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(blank.status(), StatusCode::BAD_REQUEST);

    let stale = app
        .oneshot(
            Request::builder()
                .method("PATCH")
                .uri(format!("/v1/voice-id/samples/{sample_id}"))
                .header(header::CONTENT_TYPE, "application/json")
                .header(header::IF_MATCH, format!("\"{revision}\""))
                .body(Body::from(r#"{"sample_label":"Stale"}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(stale.status(), StatusCode::CONFLICT);
}

#[test]
fn native_server_runtime_with_no_model_pack_is_accepted_at_startup_validation() {
    // A native backend with no model pack bound (a fresh install with zero
    // models pulled) must still pass startup validation -- the daemon has to
    // come up and answer /health; "no model" is a fail-closed error at
    // transcription-request time, not a reason to refuse to serve at all.
    openasr_server::ServerRuntime {
        backend: openasr_core::BackendKind::Native,
        native_execution: openasr_server::NativeExecutionSupervisor::default(),
        ffmpeg_bin: None,
        ffmpeg_bin_explicit: false,
        model_pack_path: None,
    }
    .validate()
    .expect("zero installed models must not block server startup");
}

#[test]
fn native_server_runtime_falls_back_to_path_stem_when_metadata_model_id_is_retired() {
    let temp = tempfile::tempdir().unwrap();
    let pack_root = temp.path().join("native-pack.oasr");
    write_mock_gguf_runtime_source(&pack_root, Some("whisper-tiny:q4_0"));
    openasr_server::ServerRuntime {
        backend: openasr_core::BackendKind::Native,
        native_execution: openasr_server::NativeExecutionSupervisor::default(),
        ffmpeg_bin: None,
        ffmpeg_bin_explicit: false,
        model_pack_path: Some(pack_root),
    }
    .validate()
    .expect("runtime should fall back to path stem model id");
}

#[test]
fn native_server_runtime_falls_back_to_path_stem_when_metadata_model_id_is_invalid() {
    let temp = tempfile::tempdir().unwrap();
    let pack_root = temp.path().join("native-pack.oasr");
    write_mock_gguf_runtime_source(&pack_root, Some("bad::id"));
    openasr_server::ServerRuntime {
        backend: openasr_core::BackendKind::Native,
        native_execution: openasr_server::NativeExecutionSupervisor::default(),
        ffmpeg_bin: None,
        ffmpeg_bin_explicit: false,
        model_pack_path: Some(pack_root),
    }
    .validate()
    .expect("runtime should fall back to path stem model id");
}

#[test]
fn native_server_runtime_rejects_reserved_oasr_container_magic() {
    let temp = tempfile::tempdir().unwrap();
    let pack_root = temp.path().join("native-pack.oasr");
    write_reserved_oasr_runtime_source(&pack_root);
    let error = openasr_server::ServerRuntime {
        backend: openasr_core::BackendKind::Native,
        native_execution: openasr_server::NativeExecutionSupervisor::default(),
        ffmpeg_bin: None,
        ffmpeg_bin_explicit: false,
        model_pack_path: Some(pack_root),
    }
    .validate()
    .unwrap_err()
    .to_string();

    assert!(error.contains("reserved OASR container magic"));
}

fn multipart_request_with_diarize(
    model: &str,
    file_name: &str,
    bytes: &[u8],
    diarize: bool,
) -> Request<Body> {
    multipart_request_with_options(
        "/v1/audio/transcriptions",
        model,
        file_name,
        bytes,
        diarize,
        None,
    )
}

fn multipart_request_with_options(
    uri: &str,
    model: &str,
    file_name: &str,
    bytes: &[u8],
    diarize: bool,
    response_format: Option<&str>,
) -> Request<Body> {
    let boundary = "openasr-boundary";
    let mut body = Vec::new();
    body.extend_from_slice(
        format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{file_name}\"\r\nContent-Type: audio/wav\r\n\r\n"
        )
        .as_bytes(),
    );
    // Raw bytes, not a UTF-8 string: `bytes` is real (often non-UTF-8) PCM
    // audio, and lossily reinterpreting it as text before re-encoding would
    // silently corrupt every byte outside valid UTF-8 -- exactly the kind of
    // corruption the old `format!("{}", String::from_utf8_lossy(bytes))`
    // version of this helper used to introduce, invisible only because a
    // since-fixed audio-prep bug (over-lenient wav passthrough) never
    // actually looked at the decoded content either.
    body.extend_from_slice(bytes);
    body.extend_from_slice(
        format!(
            "\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\n{model}"
        )
        .as_bytes(),
    );
    if diarize {
        body.extend_from_slice(
            format!(
                "\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"diarize\"\r\n\r\ntrue"
            )
            .as_bytes(),
        );
    }
    if let Some(value) = response_format {
        body.extend_from_slice(
            format!(
                "\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"response_format\"\r\n\r\n{value}"
            )
            .as_bytes(),
        );
    }
    body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());

    Request::builder()
        .method("POST")
        .uri(uri)
        .header(
            header::CONTENT_TYPE,
            format!("multipart/form-data; boundary={boundary}"),
        )
        .body(Body::from(body))
        .unwrap()
}

fn multipart_request_with_extra_fields(
    uri: &str,
    model: &str,
    file_name: &str,
    bytes: &[u8],
    fields: &[(&str, &str)],
) -> Request<Body> {
    let boundary = "openasr-boundary";
    let mut body = Vec::new();
    body.extend_from_slice(
        format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{file_name}\"\r\nContent-Type: audio/wav\r\n\r\n"
        )
        .as_bytes(),
    );
    // See `multipart_request_with_options`'s matching comment: raw bytes,
    // never a lossy UTF-8 reinterpretation of binary audio.
    body.extend_from_slice(bytes);
    body.extend_from_slice(
        format!(
            "\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\n{model}"
        )
        .as_bytes(),
    );
    for (name, value) in fields {
        body.extend_from_slice(
            format!(
                "\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}"
            )
            .as_bytes(),
        );
    }
    body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());

    Request::builder()
        .method("POST")
        .uri(uri)
        .header(
            header::CONTENT_TYPE,
            format!("multipart/form-data; boundary={boundary}"),
        )
        .body(Body::from(body))
        .unwrap()
}

fn expected_mock_rendered_transcription(
    model: &str,
    file_name: &str,
    response_format: ResponseFormat,
) -> String {
    let transcription = transcribe_with_mock_backend(
        TranscriptionRequest::new(std::path::Path::new(file_name), model)
            .with_display_file_name(Some(file_name.to_string())),
    )
    .expect("mock transcription");
    render_transcription(&transcription, response_format).expect("render mock transcription")
}

#[tokio::test]
async fn health_catalog_degraded_is_null_with_no_recorded_status() {
    // A fresh $OPENASR_HOME with no catalog load recorded (or the most recent
    // load used the primary source) reads `catalog_degraded: null` -- the
    // common case.
    let temp = tempfile::tempdir().unwrap();
    let distribution = openasr_server::DistributionRuntime {
        openasr_home: Some(temp.path().join("home")),
        catalog_url: None,
        catalog_local_override: None,
    };
    let app = openasr_server::app_with_runtime_and_distribution(
        openasr_server::ServerRuntime::default(),
        distribution,
    );

    let response = app
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = to_bytes(response.into_body(), 1024 * 64).await.unwrap();
    let parsed: Value = serde_json::from_slice(&bytes).unwrap();
    assert!(parsed["catalog_degraded"].is_null());
}

#[tokio::test]
async fn health_surfaces_catalog_degraded_reason_when_recorded() {
    // /health reads the SAME degraded-status marker the catalog loaders
    // write when they fall back to a cache/embedded tier (see
    // docs/CATALOG_COMPATIBILITY.md) -- this pins the shell-facing surface
    // half of that contract without needing to reproduce a real fallback.
    let temp = tempfile::tempdir().unwrap();
    let home = temp.path().join("home");
    std::fs::create_dir_all(&home).unwrap();
    openasr_core::record_catalog_degraded(&home, "embedded", "network unreachable; using embedded");
    let distribution = openasr_server::DistributionRuntime {
        openasr_home: Some(home),
        catalog_url: None,
        catalog_local_override: None,
    };
    let app = openasr_server::app_with_runtime_and_distribution(
        openasr_server::ServerRuntime::default(),
        distribution,
    );

    let response = app
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = to_bytes(response.into_body(), 1024 * 64).await.unwrap();
    let parsed: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(
        parsed["catalog_degraded"],
        serde_json::json!("network unreachable; using embedded")
    );
}

#[tokio::test]
async fn health_returns_identity_json_without_instance_token() {
    let app = with_server_instance_token_env(None, openasr_server::app);
    let response = app
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let bytes = to_bytes(response.into_body(), 1024 * 64).await.unwrap();
    let parsed: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(parsed["status"], serde_json::json!("ok"));
    assert_eq!(
        parsed["server_version"],
        serde_json::json!(env!("CARGO_PKG_VERSION"))
    );
    assert_eq!(parsed["pid"], serde_json::json!(std::process::id()));
    assert!(parsed["instance_token"].is_null());
    assert_eq!(
        parsed["model_resident"], true,
        "the mock backend has no runtime to unload, so it reads resident whenever bound"
    );
    let recognized = parsed["recognized_audio_extensions"]
        .as_array()
        .expect("recognized_audio_extensions array");
    assert!(
        recognized.iter().any(|value| value == "wav"),
        "recognized_audio_extensions mirrors core's recognized_audio_extensions() and must list wav"
    );
    assert_eq!(
        parsed.as_object().expect("health response object").len(),
        12,
        "status/server_version/pid/instance_token/model_installed/model_resident \
         plus the 0.1.14 additive native_active_count/idle_seconds, the 0.1.15 \
         additive abandoned_worker_count debug field, the 0.1.16 additive \
         catalog_degraded field, the 0.1.25 additive \
         voice_id_min_enrollment_speech_seconds field, and the 0.1.26 additive \
         recognized_audio_extensions field"
    );
}

#[tokio::test]
async fn health_echoes_launch_instance_token_without_env() {
    let app = with_server_instance_token_env(None, || {
        openasr_server::app_with_runtime_and_distribution_and_launch_options(
            openasr_server::ServerRuntime::default(),
            openasr_server::DistributionRuntime::default(),
            openasr_server::ServerLaunchOptions {
                instance_token: Some("launch-health-token".to_string()),
                ..Default::default()
            },
        )
    });
    let response = app
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let bytes = to_bytes(response.into_body(), 1024 * 64).await.unwrap();
    let parsed: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(
        parsed["instance_token"],
        serde_json::json!("launch-health-token")
    );
    assert_eq!(
        parsed.as_object().expect("health response object").len(),
        12,
        "status/server_version/pid/instance_token/model_installed/model_resident \
         plus the 0.1.14 additive native_active_count/idle_seconds, the 0.1.15 \
         additive abandoned_worker_count debug field, the 0.1.16 additive \
         catalog_degraded field, the 0.1.25 additive \
         voice_id_min_enrollment_speech_seconds field, and the 0.1.26 additive \
         recognized_audio_extensions field"
    );
}

#[tokio::test]
async fn health_prefers_env_instance_token_over_launch_option() {
    let app = with_server_instance_token_env(Some("env-health-token"), || {
        openasr_server::app_with_runtime_and_distribution_and_launch_options(
            openasr_server::ServerRuntime::default(),
            openasr_server::DistributionRuntime::default(),
            openasr_server::ServerLaunchOptions {
                instance_token: Some("launch-health-token".to_string()),
                ..Default::default()
            },
        )
    });
    let response = app
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let bytes = to_bytes(response.into_body(), 1024 * 64).await.unwrap();
    let body = String::from_utf8(bytes.to_vec()).unwrap();
    assert!(!body.contains("launch-health-token"));
    let parsed: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(
        parsed["instance_token"],
        serde_json::json!("env-health-token")
    );
    assert_eq!(
        parsed.as_object().expect("health response object").len(),
        12,
        "status/server_version/pid/instance_token/model_installed/model_resident \
         plus the 0.1.14 additive native_active_count/idle_seconds, the 0.1.15 \
         additive abandoned_worker_count debug field, the 0.1.16 additive \
         catalog_degraded field, the 0.1.25 additive \
         voice_id_min_enrollment_speech_seconds field, and the 0.1.26 additive \
         recognized_audio_extensions field"
    );
}

#[tokio::test]
async fn bearer_auth_protects_v1_routes_when_enabled() {
    let app = openasr_server::app_with_runtime_and_distribution_and_launch_options(
        openasr_server::ServerRuntime::default(),
        openasr_server::DistributionRuntime::default(),
        openasr_server::ServerLaunchOptions {
            auth: openasr_server::ServerAuth::bearer("remote-secret"),
            ..Default::default()
        },
    );

    let unauthenticated = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/capabilities")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(unauthenticated.status(), StatusCode::UNAUTHORIZED);
    assert_eq!(
        unauthenticated
            .headers()
            .get(header::WWW_AUTHENTICATE)
            .and_then(|value| value.to_str().ok()),
        Some("Bearer")
    );

    let wrong = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/capabilities")
                .header(header::AUTHORIZATION, "Bearer wrong")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(wrong.status(), StatusCode::UNAUTHORIZED);

    let authorized = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/capabilities")
                .header(header::AUTHORIZATION, "Bearer remote-secret")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(authorized.status(), StatusCode::OK);

    let health = app
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(health.status(), StatusCode::OK);
}

#[tokio::test]
async fn pairing_auth_issues_and_revokes_device_bearer_credentials() {
    let app = openasr_server::app_with_runtime_and_distribution_and_launch_options(
        openasr_server::ServerRuntime::default(),
        openasr_server::DistributionRuntime::default(),
        openasr_server::ServerLaunchOptions {
            auth: openasr_server::ServerAuth::pairing("admin-secret"),
            ..Default::default()
        },
    );

    let unauthenticated = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/capabilities")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(unauthenticated.status(), StatusCode::UNAUTHORIZED);

    let create = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/pairing/requests")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(r#"{"device_name":"MacBook Pro"}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(create.status(), StatusCode::ACCEPTED);
    let create_body = to_bytes(create.into_body(), 1024 * 64).await.unwrap();
    let create_json: Value = serde_json::from_slice(&create_body).unwrap();
    let request_id = create_json["request_id"].as_str().unwrap();
    assert_eq!(request_id.len(), 32);
    assert!(request_id.bytes().all(|byte| byte.is_ascii_hexdigit()));
    assert_eq!(create_json["device_name"], "MacBook Pro");
    assert_eq!(create_json["status"], "pending");
    assert!(create_json["safety_code"].is_null());

    let listed_without_admin = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/pairing/requests")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(listed_without_admin.status(), StatusCode::UNAUTHORIZED);

    let listed = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/pairing/requests")
                .header(header::AUTHORIZATION, "Bearer admin-secret")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(listed.status(), StatusCode::OK);
    let listed_body = to_bytes(listed.into_body(), 1024 * 64).await.unwrap();
    let listed_json: Value = serde_json::from_slice(&listed_body).unwrap();
    assert_eq!(listed_json.as_array().unwrap().len(), 1);

    let reject_create = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/pairing/requests")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(r#"{"device_name":"Rejected Mac"}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(reject_create.status(), StatusCode::ACCEPTED);
    let reject_body = to_bytes(reject_create.into_body(), 1024 * 64)
        .await
        .unwrap();
    let reject_json: Value = serde_json::from_slice(&reject_body).unwrap();
    let rejected_request_id = reject_json["request_id"].as_str().unwrap();
    let rejected = app
        .clone()
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri(format!("/v1/pairing/requests/{rejected_request_id}"))
                .header(header::AUTHORIZATION, "Bearer admin-secret")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(rejected.status(), StatusCode::NO_CONTENT);

    let approve = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(format!("/v1/pairing/requests/{request_id}/approve"))
                .header(header::AUTHORIZATION, "Bearer admin-secret")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(approve.status(), StatusCode::OK);
    let approve_body = to_bytes(approve.into_body(), 1024 * 64).await.unwrap();
    let approve_json: Value = serde_json::from_slice(&approve_body).unwrap();
    let device_id = approve_json["device_id"].as_str().unwrap();
    assert_eq!(device_id.len(), 24);
    assert!(device_id.bytes().all(|byte| byte.is_ascii_hexdigit()));
    assert_eq!(approve_json["status"], "approved");
    assert!(approve_json["bearer_token"].is_null());

    let credential = app
        .clone()
        .oneshot(
            Request::builder()
                .uri(format!("/v1/pairing/requests/{request_id}/credential"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(credential.status(), StatusCode::OK);
    let credential_body = to_bytes(credential.into_body(), 1024 * 64).await.unwrap();
    let credential_json: Value = serde_json::from_slice(&credential_body).unwrap();
    assert_eq!(credential_json["device_id"], device_id);
    let bearer_token = credential_json["bearer_token"].as_str().unwrap();
    assert!(bearer_token.starts_with("oasr_"));
    assert_eq!(credential_json["device_name"], "MacBook Pro");

    let devices = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/pairing/credentials")
                .header(header::AUTHORIZATION, "Bearer admin-secret")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(devices.status(), StatusCode::OK);
    let devices_body = to_bytes(devices.into_body(), 1024 * 64).await.unwrap();
    let devices_json: Value = serde_json::from_slice(&devices_body).unwrap();
    assert_eq!(devices_json.as_array().unwrap().len(), 1);
    assert_eq!(devices_json[0]["device_id"], device_id);
    assert_eq!(devices_json[0]["device_name"], "MacBook Pro");
    assert!(devices_json[0].get("bearer_token").is_none());

    let authorized = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/capabilities")
                .header(header::AUTHORIZATION, format!("Bearer {bearer_token}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(authorized.status(), StatusCode::OK);

    let device_cannot_manage_pairing = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/pairing/requests")
                .header(header::AUTHORIZATION, format!("Bearer {bearer_token}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        device_cannot_manage_pairing.status(),
        StatusCode::UNAUTHORIZED
    );

    let revoke = app
        .clone()
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri(format!("/v1/pairing/credentials/{device_id}"))
                .header(header::AUTHORIZATION, "Bearer admin-secret")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(revoke.status(), StatusCode::NO_CONTENT);

    let devices = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/pairing/credentials")
                .header(header::AUTHORIZATION, "Bearer admin-secret")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(devices.status(), StatusCode::OK);
    let devices_body = to_bytes(devices.into_body(), 1024 * 64).await.unwrap();
    let devices_json: Value = serde_json::from_slice(&devices_body).unwrap();
    assert_eq!(devices_json.as_array().unwrap().len(), 0);

    let revoked = app
        .oneshot(
            Request::builder()
                .uri("/v1/capabilities")
                .header(header::AUTHORIZATION, format!("Bearer {bearer_token}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(revoked.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn pairing_credential_claim_stays_pending_until_admin_approval() {
    let app = openasr_server::app_with_runtime_and_distribution_and_launch_options(
        openasr_server::ServerRuntime::default(),
        openasr_server::DistributionRuntime::default(),
        openasr_server::ServerLaunchOptions {
            auth: openasr_server::ServerAuth::pairing("admin-secret"),
            ..Default::default()
        },
    );

    let create = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/pairing/requests")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(r#"{"device_name":"Waiting Mac"}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    let create_body = to_bytes(create.into_body(), 1024 * 64).await.unwrap();
    let create_json: Value = serde_json::from_slice(&create_body).unwrap();
    let request_id = create_json["request_id"].as_str().unwrap();

    let pending = app
        .oneshot(
            Request::builder()
                .uri(format!("/v1/pairing/requests/{request_id}/credential"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(pending.status(), StatusCode::ACCEPTED);
    let pending_body = to_bytes(pending.into_body(), 1024 * 64).await.unwrap();
    let pending_json: Value = serde_json::from_slice(&pending_body).unwrap();
    assert_eq!(pending_json["status"], "pending");
}

#[tokio::test]
async fn pairing_route_ids_are_normalized_and_fail_closed() {
    let app = openasr_server::app_with_runtime_and_distribution_and_launch_options(
        openasr_server::ServerRuntime::default(),
        openasr_server::DistributionRuntime::default(),
        openasr_server::ServerLaunchOptions {
            auth: openasr_server::ServerAuth::pairing("admin-secret"),
            ..Default::default()
        },
    );

    let create = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/pairing/requests")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(r#"{"device_name":"Case Mac"}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    let create_body = to_bytes(create.into_body(), 1024 * 64).await.unwrap();
    let create_json: Value = serde_json::from_slice(&create_body).unwrap();
    let request_id = create_json["request_id"].as_str().unwrap();
    let uppercase_request_id = request_id.to_ascii_uppercase();

    let approve = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(format!(
                    "/v1/pairing/requests/{uppercase_request_id}/approve"
                ))
                .header(header::AUTHORIZATION, "Bearer admin-secret")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(approve.status(), StatusCode::OK);
    let approve_body = to_bytes(approve.into_body(), 1024 * 64).await.unwrap();
    let approve_json: Value = serde_json::from_slice(&approve_body).unwrap();
    let device_id = approve_json["device_id"].as_str().unwrap();
    let uppercase_device_id = device_id.to_ascii_uppercase();

    let revoke = app
        .clone()
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri(format!("/v1/pairing/credentials/{uppercase_device_id}"))
                .header(header::AUTHORIZATION, "Bearer admin-secret")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(revoke.status(), StatusCode::NO_CONTENT);

    for (method, uri) in [
        ("POST", "/v1/pairing/requests/not-hex/approve"),
        ("DELETE", "/v1/pairing/requests/not-hex"),
        ("GET", "/v1/pairing/requests/not-hex/credential"),
        ("DELETE", "/v1/pairing/credentials/not-hex"),
    ] {
        let mut builder = Request::builder().method(method).uri(uri);
        if method != "GET" {
            builder = builder.header(header::AUTHORIZATION, "Bearer admin-secret");
        }
        let response = app
            .clone()
            .oneshot(builder.body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{method} {uri}");
    }
}

#[tokio::test]
async fn pairing_auth_returns_safety_code_derived_from_server_identity() {
    let safety_code = openasr_server::pairing_safety_code_for_certificate_fingerprint(
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    );
    let app = openasr_server::app_with_runtime_and_distribution_and_launch_options(
        openasr_server::ServerRuntime::default(),
        openasr_server::DistributionRuntime::default(),
        openasr_server::ServerLaunchOptions {
            auth: openasr_server::ServerAuth::pairing_with_safety_code(
                "admin-secret",
                Some(safety_code.clone()),
            ),
            ..Default::default()
        },
    );

    let create = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/pairing/requests")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(r#"{"device_name":"Remote Mac"}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(create.status(), StatusCode::ACCEPTED);
    let create_body = to_bytes(create.into_body(), 1024 * 64).await.unwrap();
    let create_json: Value = serde_json::from_slice(&create_body).unwrap();
    let request_id = create_json["request_id"].as_str().unwrap();
    assert_eq!(create_json["safety_code"], safety_code);

    let listed = app
        .oneshot(
            Request::builder()
                .uri("/v1/pairing/requests")
                .header(header::AUTHORIZATION, "Bearer admin-secret")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(listed.status(), StatusCode::OK);
    let listed_body = to_bytes(listed.into_body(), 1024 * 64).await.unwrap();
    let listed_json: Value = serde_json::from_slice(&listed_body).unwrap();
    assert_eq!(listed_json[0]["request_id"], request_id);
    assert_eq!(listed_json[0]["safety_code"], safety_code);
}

#[tokio::test]
async fn serve_rejects_non_loopback_http_bind_until_tls_is_available() {
    let err = openasr_server::serve_with_launch_options(
        "0.0.0.0:0".parse().unwrap(),
        openasr_server::ServerRuntime::default(),
        openasr_server::ServerLaunchOptions::default(),
    )
    .await
    .unwrap_err();

    assert!(
        err.to_string().contains("local-only until TLS/WSS"),
        "unexpected error: {err:?}"
    );
}

#[tokio::test]
async fn serve_rejects_non_loopback_tls_without_device_authentication() {
    let err = openasr_server::serve_with_launch_options(
        "0.0.0.0:0".parse().unwrap(),
        openasr_server::ServerRuntime::default(),
        openasr_server::ServerLaunchOptions {
            tls: openasr_server::ServerTlsConfig::self_signed(["localhost"]),
            ..Default::default()
        },
    )
    .await
    .unwrap_err();

    assert!(
        err.to_string().contains("requires device authentication"),
        "unexpected error: {err:?}"
    );
}

#[tokio::test]
async fn transcriptions_returns_mock_json_by_default() {
    let temp = tempfile::tempdir().unwrap();
    let app = openasr_server::app_with_runtime_and_distribution(
        openasr_server::ServerRuntime::default(),
        openasr_server::DistributionRuntime {
            openasr_home: Some(temp.path().join("home")),
            catalog_url: None,
            catalog_local_override: None,
        },
    );
    let request = multipart_request("whisper-large-v3-turbo", "sample.wav", b"not a real wav");
    let response = app.oneshot(request).await.unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let bytes = to_bytes(response.into_body(), 1024 * 256).await.unwrap();
    let parsed: Value = serde_json::from_slice(&bytes).unwrap();
    assert!(
        parsed["text"]
            .as_str()
            .unwrap()
            .contains("OpenASR mock transcription")
    );
}

/// A real-world upload filename containing CJK characters and a space must
/// parse as multipart/form-data like any other filename -- this previously
/// got misdiagnosed as a client encoding bug when the true cause was uploads
/// exceeding the server's body-size limit (see the oversized-upload test
/// below), not the filename itself.
#[tokio::test]
async fn transcriptions_accept_filename_with_cjk_characters_and_space() {
    let temp = tempfile::tempdir().unwrap();
    let app = openasr_server::app_with_runtime_and_distribution(
        openasr_server::ServerRuntime::default(),
        openasr_server::DistributionRuntime {
            openasr_home: Some(temp.path().join("home")),
            catalog_url: None,
            catalog_local_override: None,
        },
    );
    let request = multipart_request(
        "whisper-large-v3-turbo",
        "0511 博弘讨论配合问题.m4a",
        b"not a real m4a",
    );
    let response = app.oneshot(request).await.unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let bytes = to_bytes(response.into_body(), 1024 * 256).await.unwrap();
    let parsed: Value = serde_json::from_slice(&bytes).unwrap();
    assert!(
        parsed["text"]
            .as_str()
            .unwrap()
            .contains("OpenASR mock transcription")
    );
}

/// Regression guard for the real-world report this change fixes: a meeting
/// recording just over the *old* 64 MB ceiling used to fail with 413 even
/// though the daemon could transcribe it fine. The `file` field now streams
/// straight to disk (see `write_upload_temp_file_streaming` in
/// `routes/transcription.rs`), so a 65 MB upload must succeed end to end.
#[tokio::test]
async fn transcriptions_accept_upload_past_old_64mb_limit() {
    let temp = tempfile::tempdir().unwrap();
    let app = openasr_server::app_with_runtime_and_distribution(
        openasr_server::ServerRuntime::default(),
        openasr_server::DistributionRuntime {
            openasr_home: Some(temp.path().join("home")),
            catalog_url: None,
            catalog_local_override: None,
        },
    );
    let past_old_limit = vec![0u8; 65 * 1024 * 1024];
    let request = multipart_request("whisper-large-v3-turbo", "meeting.wav", &past_old_limit);
    let response = app.oneshot(request).await.unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let bytes = to_bytes(response.into_body(), 1024 * 256).await.unwrap();
    let parsed: Value = serde_json::from_slice(&bytes).unwrap();
    assert!(
        parsed["text"]
            .as_str()
            .unwrap()
            .contains("OpenASR mock transcription"),
        "expected a successful mock transcription, got: {parsed}"
    );
}

/// An upload past the server's (now multi-gigabyte) body-size ceiling must
/// still fail closed with a clear, actionable "file too large" message and
/// 413, not the generic "Error parsing `multipart/form-data` request" text
/// that `MultipartError`'s `Display` renders for every underlying `multer`
/// failure (including this one) -- see `multipart_error_message` in
/// `lib.rs`. The request body below is built from a handful of cheap
/// `Bytes` clones (refcount bumps, not copies) streamed past the ceiling, so
/// exercising the real multi-gigabyte limit doesn't require the test itself
/// to hold multiple gigabytes in memory.
#[tokio::test]
async fn transcriptions_reject_upload_past_body_limit_with_clear_message() {
    let temp = tempfile::tempdir().unwrap();
    let app = openasr_server::app_with_runtime_and_distribution(
        openasr_server::ServerRuntime::default(),
        openasr_server::DistributionRuntime {
            openasr_home: Some(temp.path().join("home")),
            catalog_url: None,
            catalog_local_override: None,
        },
    );
    // Mirrors `MAX_TRANSCRIPTION_UPLOAD_BYTES` in `lib.rs`; kept in sync by
    // hand since it's a private constant of the crate under test (same
    // convention the old 64 MB-scaled test above used).
    const CEILING_BYTES: u64 = 2 * 1024 * 1024 * 1024;
    let request = oversized_streaming_multipart_request(CEILING_BYTES + 8 * 1024 * 1024);
    let response = app.oneshot(request).await.unwrap();

    assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
    let bytes = to_bytes(response.into_body(), 1024 * 256).await.unwrap();
    let parsed: Value = serde_json::from_slice(&bytes).unwrap();
    let message = parsed["error"]["message"].as_str().unwrap();
    assert!(message.contains("too large"), "message was: {message}");
    assert!(
        message.contains("GB"),
        "message should cite the new GB-scale ceiling, not the old MB one: {message}"
    );
    assert!(
        !message.contains("Error parsing"),
        "message regressed to the generic multipart error text: {message}"
    );
}

/// Builds a `/v1/audio/transcriptions` request whose `file` field streams
/// `total_file_bytes` of zeroed content without ever materializing that much
/// memory in the test process: the chunk is a single `Bytes` allocation,
/// cloned (cheap, `Arc`-backed) for every repetition.
fn oversized_streaming_multipart_request(total_file_bytes: u64) -> Request<Body> {
    let boundary = "openasr-oversize-boundary";
    let preamble = format!(
        "--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nwhisper-large-v3-turbo\r\n\
         --{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"huge.wav\"\r\nContent-Type: audio/wav\r\n\r\n"
    );

    const CHUNK_BYTES: usize = 4 * 1024 * 1024;
    let chunk = axum::body::Bytes::from(vec![0u8; CHUNK_BYTES]);
    let chunk_count = (total_file_bytes / CHUNK_BYTES as u64) as usize + 1;

    let head = futures_util::stream::once(async move {
        Ok::<_, std::io::Error>(axum::body::Bytes::from(preamble.into_bytes()))
    });
    let tail = futures_util::stream::iter(std::iter::repeat_n(chunk, chunk_count))
        .map(Ok::<_, std::io::Error>)
        .then(yield_between_chunks);
    let body = Body::from_stream(head.chain(tail));

    Request::builder()
        .method("POST")
        .uri("/v1/audio/transcriptions")
        .header(
            header::CONTENT_TYPE,
            format!("multipart/form-data; boundary={boundary}"),
        )
        .body(body)
        .unwrap()
}

/// Real-world evidence that the upload path no longer buffers the whole
/// file in memory: a 200 MB upload's resident-set growth must stay far
/// below the file size, at roughly the same order of magnitude as a 1 MB
/// upload's. Before this change, `field.bytes()` held the entire multipart
/// field in memory, so this delta tracked file size almost 1:1.
///
/// This drives the *actual* running process (via `oneshot`, same process as
/// the test) with a body streamed from cheap `Bytes` clones -- so the input
/// construction itself doesn't add O(file) memory that would muddy the
/// measurement -- and samples RSS with `ps`, which is available in this
/// project's macOS and Linux CI targets.
#[tokio::test]
async fn transcriptions_large_upload_memory_stays_bounded() {
    let temp = tempfile::tempdir().unwrap();
    let build_app = || {
        openasr_server::app_with_runtime_and_distribution(
            openasr_server::ServerRuntime::default(),
            openasr_server::DistributionRuntime {
                openasr_home: Some(temp.path().join("home")),
                catalog_url: None,
                catalog_local_override: None,
            },
        )
    };
    const SMALL_FILE_BYTES: u64 = 1024 * 1024;
    const LARGE_FILE_BYTES: u64 = 200 * 1024 * 1024;

    // Warm up allocator/paging so the first request's one-time setup costs
    // don't get counted as part of either delta below.
    let warmup = multipart_request(
        "whisper-large-v3-turbo",
        "warm.wav",
        &vec![0u8; SMALL_FILE_BYTES as usize],
    );
    let response = build_app().oneshot(warmup).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let baseline_rss_kb = process_rss_kb();
    let small = multipart_request(
        "whisper-large-v3-turbo",
        "small.wav",
        &vec![0u8; SMALL_FILE_BYTES as usize],
    );
    let response = build_app().oneshot(small).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let after_small_rss_kb = process_rss_kb();

    let large = streamed_zero_multipart_request(LARGE_FILE_BYTES);
    let response = build_app().oneshot(large).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let after_large_rss_kb = process_rss_kb();

    let small_delta_kb = after_small_rss_kb.saturating_sub(baseline_rss_kb);
    let large_delta_kb = after_large_rss_kb.saturating_sub(after_small_rss_kb);
    let large_file_kb = LARGE_FILE_BYTES / 1024;

    // Generous ceiling (half the file size) to avoid flakiness from
    // allocator/page-cache noise, while still failing hard if the upload
    // path regresses back to buffering the whole file: full buffering would
    // put `large_delta_kb` within shouting distance of `large_file_kb`.
    assert!(
        large_delta_kb < large_file_kb / 2,
        "RSS grew by {large_delta_kb} KB for a {large_file_kb} KB upload \
         (small-file baseline delta was {small_delta_kb} KB) -- looks like \
         the upload is being buffered in memory again"
    );
}

fn process_rss_kb() -> u64 {
    let pid = std::process::id().to_string();
    let output = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &pid])
        .output()
        .expect("`ps` should be available to sample this test process's RSS");
    String::from_utf8_lossy(&output.stdout)
        .trim()
        .parse()
        .unwrap_or_else(|error| panic!("could not parse `ps -o rss=` output: {error}"))
}

/// Builds a request that streams `total_file_bytes` of zeroed content as a
/// well-formed, successfully-parseable multipart body (unlike
/// `oversized_streaming_multipart_request`, which is truncated because the
/// server is expected to reject it before the body ends).
fn streamed_zero_multipart_request(total_file_bytes: u64) -> Request<Body> {
    let boundary = "openasr-stream-boundary";
    let preamble = format!(
        "--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"large.wav\"\r\nContent-Type: audio/wav\r\n\r\n"
    );
    let trailer = format!(
        "\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nwhisper-large-v3-turbo\r\n--{boundary}--\r\n"
    );

    const CHUNK_BYTES: usize = 64 * 1024;
    let chunk = axum::body::Bytes::from(vec![0u8; CHUNK_BYTES]);
    let full_chunks = (total_file_bytes / CHUNK_BYTES as u64) as usize;

    let head = futures_util::stream::once(async move {
        Ok::<_, std::io::Error>(axum::body::Bytes::from(preamble.into_bytes()))
    });
    let body_chunks = futures_util::stream::iter(std::iter::repeat_n(chunk, full_chunks))
        .map(Ok::<_, std::io::Error>)
        .then(yield_between_chunks);
    let tail = futures_util::stream::once(async move {
        Ok::<_, std::io::Error>(axum::body::Bytes::from(trailer.into_bytes()))
    });
    let body = Body::from_stream(head.chain(body_chunks).chain(tail));

    Request::builder()
        .method("POST")
        .uri("/v1/audio/transcriptions")
        .header(
            header::CONTENT_TYPE,
            format!("multipart/form-data; boundary={boundary}"),
        )
        .body(body)
        .unwrap()
}

/// Inserts a real async yield point between streamed body chunks.
///
/// Without this, `futures_util::stream::iter` resolves every poll
/// immediately (`Poll::Ready`, never `Poll::Pending`), which is *not* how a
/// real network body behaves -- a real client's bytes arrive as separate
/// TCP reads with actual `Pending` gaps in between, and that backpressure is
/// exactly what makes `multer`'s field-parsing loop hand back a chunk as
/// soon as one is available instead of draining the whole stream in one
/// synchronous sweep. Skipping this made an earlier version of the
/// large-upload memory test below spuriously "prove" the upload was fully
/// buffered (`field.chunk()` returned the *entire* file as a single chunk)
/// even though the real server, driven by a real socket, streams correctly.
async fn yield_between_chunks(
    item: Result<axum::body::Bytes, std::io::Error>,
) -> Result<axum::body::Bytes, std::io::Error> {
    tokio::task::yield_now().await;
    item
}

#[tokio::test]
async fn transcriptions_accept_word_timestamp_granularity_for_json() {
    let temp = tempfile::tempdir().unwrap();
    let app = openasr_server::app_with_runtime_and_distribution(
        openasr_server::ServerRuntime::default(),
        openasr_server::DistributionRuntime {
            openasr_home: Some(temp.path().join("home")),
            catalog_url: None,
            catalog_local_override: None,
        },
    );
    let request = multipart_request_with_extra_fields(
        "/v1/audio/transcriptions",
        "whisper-large-v3-turbo",
        "sample.wav",
        b"not a real wav",
        &[("timestamp_granularities[]", "word")],
    );
    let response = app.oneshot(request).await.unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let bytes = to_bytes(response.into_body(), 1024 * 256).await.unwrap();
    let parsed: Value = serde_json::from_slice(&bytes).unwrap();
    let words = parsed["segments"][0]["words"].as_array().unwrap();
    assert!(!words.is_empty());
    assert_eq!(words[0]["word"], "OpenASR");
    assert_eq!(words[0]["start"], 0.0);
    assert_eq!(words.last().unwrap()["end"], 2.5);
}

/// Writes a config with auto-save off and last5 retention at `<temp>/home`,
/// locking in that history recording is governed by `history_retention` alone
/// (auto_save only controls transcript-file exports).
fn enable_history(temp: &tempfile::TempDir) {
    let home = temp.path().join("home");
    std::fs::create_dir_all(&home).unwrap();
    std::fs::write(
        home.join("config.json"),
        serde_json::json!({
            "preferences": { "auto_save": false, "history_retention": "last5" }
        })
        .to_string(),
    )
    .unwrap();
}

#[tokio::test]
async fn transcriptions_record_file_history_in_sqlite_store() {
    let temp = tempfile::tempdir().unwrap();
    let distribution = openasr_server::DistributionRuntime {
        openasr_home: Some(temp.path().join("home")),
        catalog_url: None,
        catalog_local_override: None,
    };
    let home = distribution.openasr_home.as_ref().unwrap().clone();
    // History recording is governed by history_retention alone; auto_save
    // stays false to lock in that it does not gate history.
    std::fs::create_dir_all(&home).unwrap();
    std::fs::write(
        home.join("config.json"),
        serde_json::json!({
            "preferences": { "auto_save": false, "history_retention": "last5" }
        })
        .to_string(),
    )
    .unwrap();
    let app = openasr_server::app_with_runtime_and_distribution(
        openasr_server::ServerRuntime::default(),
        distribution,
    );

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/history")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = to_bytes(response.into_body(), 1024 * 64).await.unwrap();
    let empty: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(empty["data"].as_array().unwrap().len(), 0);

    let request = multipart_request_with_options(
        "/v1/audio/transcriptions",
        "whisper-large-v3-turbo",
        "sample.wav",
        b"not a real wav",
        false,
        Some("srt"),
    );
    let response = app.clone().oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/history")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = to_bytes(response.into_body(), 1024 * 64).await.unwrap();
    let parsed: Value = serde_json::from_slice(&bytes).unwrap();
    let entry = &parsed["data"][0];
    let id = entry["id"].as_str().unwrap();
    assert_eq!(entry["kind"], "file");
    assert_eq!(entry["model"], "whisper-large-v3-turbo");
    assert_eq!(entry["source_name"], "sample.wav");
    assert!(entry["created_at"].is_null());
    assert!(entry["created_at_unix_seconds"].as_u64().is_some());
    assert!(entry["duration_seconds"].as_f64().is_some());
    assert_eq!(entry["output_format"], "srt");
    assert_eq!(entry["diarization_active"], false);
    assert_eq!(entry["provenance"], "recorded");
    assert!(entry["preview"].as_str().unwrap().contains("OpenASR mock"));
    // Transcript text lives in the SQLite row, not a filesystem sidecar; it
    // must not leak a path into the wire contract.
    assert!(entry.get("text_path").is_none());
    // `formats` is derived from the stored transcript: this file transcription
    // has timed segments, so the timing-dependent exports are advertised too
    // (no longer the old blanket ResponseFormat::ALL claim).
    let format_strs: Vec<&str> = entry["formats"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_str().unwrap())
        .collect();
    for expected in ["text", "json", "markdown", "srt", "vtt"] {
        assert!(
            format_strs.contains(&expected),
            "missing {expected}: {format_strs:?}"
        );
    }
    let history_db = home.join("history").join("history.db");
    assert!(history_db.exists());

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri(format!("/v1/history/{id}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = to_bytes(response.into_body(), 1024 * 64).await.unwrap();
    let detail: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(detail["id"], id);
    assert!(detail["transcript"].is_null());
    assert!(detail["response_format"].is_null());
    assert_eq!(detail["output_format"], "srt");
    assert_eq!(detail["diarization_active"], false);
    assert_eq!(detail["provenance"], "recorded");
    assert!(
        detail["text"]
            .as_str()
            .unwrap()
            .contains("OpenASR mock transcription")
    );
    // Detail carries the per-segment transcript in the transcription API's
    // JsonSegment shape so the desktop export UI can rebuild SRT/VTT/JSON.
    let detail_segments = detail["segments"].as_array().unwrap();
    assert!(!detail_segments.is_empty());
    assert!(detail_segments[0]["text"].as_str().is_some());
    assert!(detail_segments[0]["start"].as_f64().is_some());
    assert!(detail_segments[0]["end"].as_f64().is_some());

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri(format!("/v1/history/{id}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/history")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let bytes = to_bytes(response.into_body(), 1024 * 64).await.unwrap();
    let parsed: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(parsed["data"].as_array().unwrap().len(), 0);
}

#[tokio::test]
async fn transcriptions_skip_file_history_when_retention_off() {
    let temp = tempfile::tempdir().unwrap();
    let home = temp.path().join("home");
    std::fs::create_dir_all(&home).unwrap();
    // Even with auto_save enabled, "off" retention must skip the write:
    // history_retention is the only history switch.
    std::fs::write(
        home.join("config.json"),
        serde_json::json!({
            "preferences": { "auto_save": true, "history_retention": "off" }
        })
        .to_string(),
    )
    .unwrap();
    let app = openasr_server::app_with_runtime_and_distribution(
        openasr_server::ServerRuntime::default(),
        openasr_server::DistributionRuntime {
            openasr_home: Some(home.clone()),
            catalog_url: None,
            catalog_local_override: None,
        },
    );

    let response = app
        .clone()
        .oneshot(multipart_request(
            "whisper-large-v3-turbo",
            "sample.wav",
            b"not a real wav",
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/history")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = to_bytes(response.into_body(), 1024 * 64).await.unwrap();
    let parsed: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(parsed["data"].as_array().unwrap().len(), 0);
}

#[tokio::test]
async fn history_list_supports_search_pagination_and_kind_filter() {
    use openasr_core::realtime::history::{
        DaemonHistoryKind, DaemonHistoryRecord, DaemonHistoryStore,
    };

    let temp = tempfile::tempdir().unwrap();
    enable_history(&temp);
    let home = temp.path().join("home");
    let store = DaemonHistoryStore::open(&home);
    let record = |kind: DaemonHistoryKind, source: &str, text: &str| DaemonHistoryRecord {
        kind,
        model: "whisper-large-v3-turbo".to_string(),
        source_name: Some(source.to_string()),
        duration_seconds: None,
        output_format: Some(ResponseFormat::Text),
        diarization_active: Some(false),
        provenance: None,
        segments: Vec::new(),
        text: text.to_string(),
    };
    let oldest = store
        .record(record(
            DaemonHistoryKind::File,
            "notes.wav",
            "english meeting notes",
        ))
        .unwrap();
    let middle = store
        .record(record(
            DaemonHistoryKind::Live,
            "live-zh",
            "我们讨论了历史记录",
        ))
        .unwrap();
    let newest = store
        .record(record(
            DaemonHistoryKind::Live,
            "live-en",
            "quick live note",
        ))
        .unwrap();

    let app = openasr_server::app_with_runtime_and_distribution(
        openasr_server::ServerRuntime::default(),
        openasr_server::DistributionRuntime {
            openasr_home: Some(home),
            catalog_url: None,
            catalog_local_override: None,
        },
    );
    let list = |uri: String| {
        let app = app.clone();
        async move {
            let response = app
                .oneshot(Request::builder().uri(uri).body(Body::empty()).unwrap())
                .await
                .unwrap();
            let status = response.status();
            let bytes = to_bytes(response.into_body(), 1024 * 64).await.unwrap();
            (status, serde_json::from_slice::<Value>(&bytes).unwrap())
        }
    };

    // Default listing: newest first, additive pagination metadata present.
    let (status, parsed) = list("/v1/history".to_string()).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(parsed["object"], "list");
    assert_eq!(parsed["total"], 3);
    assert_eq!(parsed["limit"], 50);
    assert_eq!(parsed["offset"], 0);
    let ids: Vec<&str> = parsed["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|entry| entry["id"].as_str().unwrap())
        .collect();
    assert_eq!(ids, vec![&newest.id, &middle.id, &oldest.id]);

    // FTS search must handle CJK substrings (trigram tokenizer, not unicode61).
    let (status, parsed) = list("/v1/history?search=%E5%8E%86%E5%8F%B2".to_string()).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(parsed["total"], 1);
    assert_eq!(parsed["data"][0]["id"], middle.id.as_str());

    // Search also covers source_name and model, and misses return empty pages.
    let (status, parsed) = list("/v1/history?search=notes.wav".to_string()).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(parsed["total"], 1);
    assert_eq!(parsed["data"][0]["id"], oldest.id.as_str());
    let (_, parsed) = list("/v1/history?search=nonexistent-token".to_string()).await;
    assert_eq!(parsed["total"], 0);
    assert_eq!(parsed["data"].as_array().unwrap().len(), 0);

    // Kind filter.
    let (status, parsed) = list("/v1/history?kind=live".to_string()).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(parsed["total"], 2);
    for entry in parsed["data"].as_array().unwrap() {
        assert_eq!(entry["kind"], "live");
    }
    let (status, _) = list("/v1/history?kind=dictation".to_string()).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    // Pagination: stable newest-first order across pages, total unaffected.
    let (status, parsed) = list("/v1/history?limit=2&offset=0".to_string()).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(parsed["total"], 3);
    assert_eq!(parsed["limit"], 2);
    assert_eq!(parsed["offset"], 0);
    let page_one: Vec<&str> = parsed["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|entry| entry["id"].as_str().unwrap())
        .collect();
    assert_eq!(page_one, vec![&newest.id, &middle.id]);
    let (_, parsed) = list("/v1/history?limit=2&offset=2".to_string()).await;
    assert_eq!(parsed["total"], 3);
    assert_eq!(parsed["offset"], 2);
    let page_two: Vec<&str> = parsed["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|entry| entry["id"].as_str().unwrap())
        .collect();
    assert_eq!(page_two, vec![&oldest.id]);

    // Combined search + kind filter.
    let (status, parsed) = list("/v1/history?search=live&kind=live".to_string()).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(parsed["total"], 2);
}

#[tokio::test]
async fn history_routes_report_errors_for_corrupt_database_without_crashing() {
    let temp = tempfile::tempdir().unwrap();
    enable_history(&temp);
    let home = temp.path().join("home");
    let history_dir = home.join("history");
    std::fs::create_dir_all(&history_dir).unwrap();
    std::fs::write(history_dir.join("history.db"), b"not a sqlite database").unwrap();

    let app = openasr_server::app_with_runtime_and_distribution(
        openasr_server::ServerRuntime::default(),
        openasr_server::DistributionRuntime {
            openasr_home: Some(home),
            catalog_url: None,
            catalog_local_override: None,
        },
    );

    // History endpoints answer with a structured error instead of taking the
    // daemon down.
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/history")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    let bytes = to_bytes(response.into_body(), 1024 * 64).await.unwrap();
    let parsed: Value = serde_json::from_slice(&bytes).unwrap();
    assert!(
        parsed["error"]["message"]
            .as_str()
            .unwrap()
            .contains("history")
    );

    // Transcription (the daemon's main job) still succeeds; the failed
    // best-effort history side-write must not fail the request.
    let request = multipart_request("whisper-large-v3-turbo", "sample.wav", b"not a real wav");
    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn static_bearer_remote_compute_transcription_records_server_history() {
    let temp = tempfile::tempdir().unwrap();
    enable_history(&temp);
    let app = openasr_server::app_with_runtime_and_distribution_and_launch_options(
        openasr_server::ServerRuntime::default(),
        openasr_server::DistributionRuntime {
            openasr_home: Some(temp.path().join("home")),
            catalog_url: None,
            catalog_local_override: None,
        },
        openasr_server::ServerLaunchOptions {
            auth: openasr_server::ServerAuth::bearer("remote-secret"),
            ..Default::default()
        },
    );

    let mut request = multipart_request("whisper-large-v3-turbo", "sample.wav", b"not a real wav");
    request.headers_mut().insert(
        header::AUTHORIZATION,
        "Bearer remote-secret".parse().unwrap(),
    );
    request
        .headers_mut()
        .insert("x-openasr-remote-compute", "client".parse().unwrap());
    let response = app.clone().oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/history")
                .header(header::AUTHORIZATION, "Bearer remote-secret")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = to_bytes(response.into_body(), 1024 * 64).await.unwrap();
    let parsed: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(
        parsed["data"].as_array().unwrap().len(),
        1,
        "static bearer auth is not a paired remote-compute device token"
    );
}

#[tokio::test]
async fn paired_device_remote_compute_transcription_skips_history_and_honors_revoke() {
    let temp = tempfile::tempdir().unwrap();
    let app = openasr_server::app_with_runtime_and_distribution_and_launch_options(
        openasr_server::ServerRuntime::default(),
        openasr_server::DistributionRuntime {
            openasr_home: Some(temp.path().join("home")),
            catalog_url: None,
            catalog_local_override: None,
        },
        openasr_server::ServerLaunchOptions {
            auth: openasr_server::ServerAuth::pairing("admin-secret"),
            ..Default::default()
        },
    );
    let (device_id, bearer_token) =
        create_approved_pairing_credential(&app, "Remote Compute Mac").await;

    let mut request = multipart_request("whisper-large-v3-turbo", "sample.wav", b"not a real wav");
    request.headers_mut().insert(
        header::AUTHORIZATION,
        format!("Bearer {bearer_token}").parse().unwrap(),
    );
    request
        .headers_mut()
        .insert("x-openasr-remote-compute", "client".parse().unwrap());
    let response = app.clone().oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let history = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/history")
                .header(header::AUTHORIZATION, "Bearer admin-secret")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(history.status(), StatusCode::OK);
    let bytes = to_bytes(history.into_body(), 1024 * 64).await.unwrap();
    let parsed: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(parsed["data"].as_array().unwrap().len(), 0);

    let revoke = app
        .clone()
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri(format!("/v1/pairing/credentials/{device_id}"))
                .header(header::AUTHORIZATION, "Bearer admin-secret")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(revoke.status(), StatusCode::NO_CONTENT);

    let mut revoked_request =
        multipart_request("whisper-large-v3-turbo", "sample.wav", b"not a real wav");
    revoked_request.headers_mut().insert(
        header::AUTHORIZATION,
        format!("Bearer {bearer_token}").parse().unwrap(),
    );
    revoked_request
        .headers_mut()
        .insert("x-openasr-remote-compute", "client".parse().unwrap());
    let revoked = app.oneshot(revoked_request).await.unwrap();
    assert_eq!(revoked.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn paired_device_cannot_enable_voice_id_on_remote_compute_routes() {
    let temp = tempfile::tempdir().unwrap();
    let app = openasr_server::app_with_runtime_and_distribution_and_launch_options(
        openasr_server::ServerRuntime::default(),
        openasr_server::DistributionRuntime {
            openasr_home: Some(temp.path().join("home")),
            catalog_url: None,
            catalog_local_override: None,
        },
        openasr_server::ServerLaunchOptions {
            auth: openasr_server::ServerAuth::pairing("admin-secret"),
            ..Default::default()
        },
    );
    let (_device_id, bearer_token) =
        create_approved_pairing_credential(&app, "Remote Compute Mac").await;

    // Credential identity, not the advisory transport header, owns the
    // privacy boundary. Cover both normal clients and a missing-header request
    // so neither can reach the operator's local speaker-profile library.
    for (uri, include_remote_marker) in [
        ("/v1/audio/transcriptions", true),
        ("/v1/audio/transcriptions", false),
        ("/v1/audio/transcriptions?stream=true", true),
        ("/v1/audio/translations", true),
    ] {
        let mut request = multipart_request_with_options(
            uri,
            "whisper-large-v3-turbo",
            "sample.wav",
            b"not a real wav",
            true,
            None,
        );
        request.headers_mut().insert(
            header::AUTHORIZATION,
            format!("Bearer {bearer_token}").parse().unwrap(),
        );
        if include_remote_marker {
            request
                .headers_mut()
                .insert("x-openasr-remote-compute", "client".parse().unwrap());
        }

        let response = app.clone().oneshot(request).await.unwrap();
        assert_eq!(
            response.status(),
            StatusCode::BAD_REQUEST,
            "paired device Voice ID must fail closed for {uri} (marker={include_remote_marker})"
        );
        let body = to_bytes(response.into_body(), 1024 * 64).await.unwrap();
        let body = String::from_utf8_lossy(&body);
        assert!(body.contains("available only for local file transcription"));
        assert!(body.contains("omit diarize=true"));
    }
}

#[tokio::test]
async fn remote_compute_header_without_auth_still_records_server_history() {
    let temp = tempfile::tempdir().unwrap();
    enable_history(&temp);
    let app = openasr_server::app_with_runtime_and_distribution(
        openasr_server::ServerRuntime::default(),
        openasr_server::DistributionRuntime {
            openasr_home: Some(temp.path().join("home")),
            catalog_url: None,
            catalog_local_override: None,
        },
    );

    let mut request = multipart_request("whisper-large-v3-turbo", "sample.wav", b"not a real wav");
    request
        .headers_mut()
        .insert("x-openasr-remote-compute", "client".parse().unwrap());
    let response = app.clone().oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/history")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = to_bytes(response.into_body(), 1024 * 64).await.unwrap();
    let parsed: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(parsed["data"].as_array().unwrap().len(), 1);
}

#[tokio::test]
async fn transcriptions_with_mock_backend_unknown_model_returns_registry_error() {
    let request = multipart_request(
        "definitely-not-an-openasr-model",
        "sample.wav",
        b"not a real wav",
    );
    let response = openasr_server::app().oneshot(request).await.unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let bytes = to_bytes(response.into_body(), 1024 * 256).await.unwrap();
    let body = String::from_utf8_lossy(&bytes);
    assert!(body.contains("was not found in the registry"));
    assert!(body.contains("openasr list"));
}

#[tokio::test]
async fn transcriptions_mock_backend_formats_match_core_renderers() {
    let temp = tempfile::tempdir().unwrap();
    let app = openasr_server::app_with_runtime_and_distribution(
        openasr_server::ServerRuntime::default(),
        openasr_server::DistributionRuntime {
            openasr_home: Some(temp.path().join("home")),
            catalog_url: None,
            catalog_local_override: None,
        },
    );
    let wav_bytes = b"not a real wav";
    for (response_format, expected_content_type) in [
        (ResponseFormat::Text, "text/plain; charset=utf-8"),
        (ResponseFormat::Json, "application/json"),
        (ResponseFormat::VerboseJson, "application/json"),
        (ResponseFormat::Srt, "text/plain; charset=utf-8"),
        (ResponseFormat::Vtt, "text/plain; charset=utf-8"),
        (ResponseFormat::Markdown, "text/plain; charset=utf-8"),
    ] {
        let request = multipart_request_with_options(
            "/v1/audio/transcriptions",
            "whisper-large-v3-turbo",
            "sample.wav",
            wav_bytes,
            false,
            Some(response_format.as_str()),
        );
        let response = app.clone().oneshot(request).await.unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response
                .headers()
                .get(header::CONTENT_TYPE)
                .and_then(|value| value.to_str().ok()),
            Some(expected_content_type),
            "unexpected content-type for {}",
            response_format.as_str()
        );
        let bytes = to_bytes(response.into_body(), 1024 * 256).await.unwrap();
        let body = String::from_utf8(bytes.to_vec()).unwrap();
        let expected = expected_mock_rendered_transcription(
            "whisper-large-v3-turbo",
            "sample.wav",
            response_format,
        );
        assert_eq!(
            body,
            expected,
            "unexpected body for {}",
            response_format.as_str()
        );
    }
}

#[tokio::test]
async fn transcriptions_reject_hotword_fields_for_current_backends_fail_closed() {
    let request = multipart_request_with_extra_fields(
        "/v1/audio/transcriptions",
        "whisper-large-v3-turbo",
        "sample.wav",
        b"not a real wav",
        &[
            ("hotword", "OpenASR Core"),
            ("phrase_bias", "Qwen"),
            ("hotword_boost", "3.5"),
        ],
    );
    let response = openasr_server::app().oneshot(request).await.unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let bytes = to_bytes(response.into_body(), 1024 * 256).await.unwrap();
    let body = String::from_utf8_lossy(&bytes);
    assert!(body.contains("Phrase bias / hotword boosting is not supported"));
    assert!(body.contains("silently ignoring phrase_bias"));
}

#[tokio::test]
async fn transcriptions_reject_phrase_bias_alias_boost_for_current_backends_fail_closed() {
    let request = multipart_request_with_extra_fields(
        "/v1/audio/transcriptions",
        "whisper-large-v3-turbo",
        "sample.wav",
        b"not a real wav",
        &[
            ("phrase_bias", "OpenASR Core"),
            ("phrase_bias_boost", "3.5"),
        ],
    );
    let response = openasr_server::app().oneshot(request).await.unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let bytes = to_bytes(response.into_body(), 1024 * 256).await.unwrap();
    let body = String::from_utf8_lossy(&bytes);
    assert!(body.contains("Phrase bias / hotword boosting is not supported"));
    assert!(body.contains("silently ignoring phrase_bias"));
}

#[tokio::test]
async fn transcriptions_reject_conflicting_phrase_bias_boost_aliases() {
    let request = multipart_request_with_extra_fields(
        "/v1/audio/transcriptions",
        "whisper-large-v3-turbo",
        "sample.wav",
        b"not a real wav",
        &[
            ("hotword", "OpenASR Core"),
            ("hotword_boost", "3.5"),
            ("phrase_bias_boost", "4.0"),
        ],
    );
    let response = openasr_server::app().oneshot(request).await.unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let bytes = to_bytes(response.into_body(), 1024 * 256).await.unwrap();
    let body = String::from_utf8_lossy(&bytes);
    assert!(body.contains("Use only one phrase bias boost field"));
    assert!(body.contains("hotword_boost or phrase_bias_boost"));
}

#[tokio::test]
async fn transcriptions_reject_phrase_bias_boost_without_phrase() {
    let request = multipart_request_with_extra_fields(
        "/v1/audio/transcriptions",
        "whisper-large-v3-turbo",
        "sample.wav",
        b"not a real wav",
        &[("hotword_boost", "3.5")],
    );
    let response = openasr_server::app().oneshot(request).await.unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let bytes = to_bytes(response.into_body(), 1024 * 256).await.unwrap();
    let body = String::from_utf8_lossy(&bytes);
    assert!(body.contains("requires at least one hotword or phrase_bias"));
}

#[tokio::test]
async fn transcriptions_reject_openai_stream_form_field_fail_closed() {
    // The OpenAI SDK sends `stream=true` as a multipart form field. This server
    // only streams via the `?stream=true` query parameter (OpenASR realtime SSE
    // events, not OpenAI `transcript.text.*`), so the field must fail closed
    // with an actionable error instead of silently returning a JSON body an SDK
    // streaming client would hang on. Doubles as the error-envelope shape
    // check: OpenAI clients expect `error.{message,type,param,code}`.
    let request = multipart_request_with_extra_fields(
        "/v1/audio/transcriptions",
        "whisper-large-v3-turbo",
        "sample.wav",
        b"not a real wav",
        &[("stream", "true")],
    );
    let response = openasr_server::app().oneshot(request).await.unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let bytes = to_bytes(response.into_body(), 1024 * 256).await.unwrap();
    let parsed: Value = serde_json::from_slice(&bytes).unwrap();
    let message = parsed["error"]["message"].as_str().unwrap();
    assert!(message.contains("'stream' form field is not supported"));
    assert!(message.contains("?stream=true"));
    assert!(message.contains("transcript.text"));
    assert_eq!(parsed["error"]["type"], "invalid_request_error");
    assert!(parsed["error"]["param"].is_null());
    assert!(parsed["error"]["code"].is_null());
}

#[tokio::test]
async fn transcriptions_accept_explicit_stream_false_form_field() {
    // `stream=false` is what an OpenAI SDK caller sends when not streaming; it
    // must parse cleanly and run the normal non-streaming pipeline.
    let request = multipart_request_with_extra_fields(
        "/v1/audio/transcriptions",
        "whisper-large-v3-turbo",
        "sample.wav",
        b"not a real wav",
        &[("stream", "false")],
    );
    let response = openasr_server::app().oneshot(request).await.unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let bytes = to_bytes(response.into_body(), 1024 * 256).await.unwrap();
    let parsed: Value = serde_json::from_slice(&bytes).unwrap();
    assert!(parsed["text"].is_string());
}

#[tokio::test]
async fn transcriptions_reject_invalid_phrase_bias_boost_before_backend_dispatch() {
    let request = multipart_request_with_extra_fields(
        "/v1/audio/transcriptions",
        "whisper-large-v3-turbo",
        "sample.wav",
        b"not a real wav",
        &[("hotword", "OpenASR"), ("hotword_boost", "0")],
    );
    let response = openasr_server::app().oneshot(request).await.unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let bytes = to_bytes(response.into_body(), 1024 * 256).await.unwrap();
    let body = String::from_utf8_lossy(&bytes);
    assert!(body.contains("Invalid phrase bias request fields"));
    assert!(body.contains("boost must be finite, non-zero"));
}

#[tokio::test]
async fn transcriptions_with_native_backend_fail_closed() {
    let temp = tempfile::tempdir().unwrap();
    let pack_root = temp.path().join("whisper-runtime.oasr");
    write_whisper_oasr_v1_fixture(&pack_root, "whisper-runtime");
    let app = openasr_server::app_with_runtime(openasr_server::ServerRuntime {
        backend: openasr_core::BackendKind::Native,
        native_execution: openasr_server::NativeExecutionSupervisor::default(),
        ffmpeg_bin: None,
        ffmpeg_bin_explicit: false,
        model_pack_path: Some(pack_root.clone()),
    });
    let wav_bytes = sample_wav_bytes();
    let request = multipart_request("whisper-runtime", "sample.wav", &wav_bytes);
    let response = app.oneshot(request).await.unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let bytes = to_bytes(response.into_body(), 1024 * 256).await.unwrap();
    let body = String::from_utf8_lossy(&bytes);
    assert!(!body.trim().is_empty());
}

#[tokio::test]
async fn transcriptions_with_native_xasr_hotword_returns_model_unsupported_error() {
    let temp = tempfile::tempdir().unwrap();
    let model_id = "xasr-hotword-http";
    let pack_root = temp.path().join("xasr-hotword-http.oasr");
    write_xasr_gguf_runtime_source(&pack_root, model_id);
    let app = openasr_server::app_with_runtime(openasr_server::ServerRuntime {
        backend: openasr_core::BackendKind::Native,
        native_execution: openasr_server::NativeExecutionSupervisor::default(),
        ffmpeg_bin: None,
        ffmpeg_bin_explicit: false,
        model_pack_path: Some(pack_root),
    });
    let wav_bytes = sample_wav_bytes();
    let request = multipart_request_with_extra_fields(
        "/v1/audio/transcriptions",
        model_id,
        "sample.wav",
        &wav_bytes,
        &[("hotword", "OpenASR")],
    );
    let response = app.oneshot(request).await.unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let bytes = to_bytes(response.into_body(), 1024 * 256).await.unwrap();
    let body = String::from_utf8_lossy(&bytes);
    assert!(body.contains("Phrase bias / hotword boosting is not supported"));
    assert!(body.contains("'xasr-zipformer' native model family"));
    assert!(body.contains("ggml-family-xasr-zipformer-runtime-v1"));
    assert!(body.contains("silently ignoring phrase_bias"));
    assert!(!body.contains("stayed fail-closed"));
}

#[tokio::test]
async fn transcriptions_with_native_backend_model_mismatch_returns_bad_request() {
    let temp = tempfile::tempdir().unwrap();
    let pack_root = temp.path().join("whisper-runtime.oasr");
    write_whisper_oasr_v1_fixture(&pack_root, "whisper-runtime");
    let app = openasr_server::app_with_runtime(openasr_server::ServerRuntime {
        backend: openasr_core::BackendKind::Native,
        native_execution: openasr_server::NativeExecutionSupervisor::default(),
        ffmpeg_bin: None,
        ffmpeg_bin_explicit: false,
        model_pack_path: Some(pack_root.clone()),
    });
    let wav_bytes = sample_wav_bytes();
    // A genuinely different base id (not a quant-pin of the pack id): since
    // 07bc0f728 a `name:quant` request matches a bare local id, so
    // `whisper-runtime:typo` is no longer a mismatch. Use a distinct base so the
    // test still exercises model-id-mismatch rejection.
    let request = multipart_request("not-whisper-runtime", "sample.wav", &wav_bytes);
    let response = app.oneshot(request).await.unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let bytes = to_bytes(response.into_body(), 1024 * 256).await.unwrap();
    let body = String::from_utf8_lossy(&bytes);
    assert!(body.contains("does not match server native local runtime source id"));
    // The error must be diagnosable without a human canonicalizing quant
    // aliases by eye: it names both sides' normalized form.
    assert!(body.contains("requested model normalizes to 'not-whisper-runtime'"));
    assert!(body.contains("loaded native runtime source normalizes to 'whisper-runtime'"));
}

#[tokio::test]
async fn transcriptions_with_native_backend_accepts_quant_alias_against_legacy_hyphen_joined_metadata_id()
 {
    // Regression test for the reported bug: a real already-published pack
    // (mimo-v2.5-asr) has its `openasr.model.id` metadata baked by an older
    // conversion tool as `family-quant` (hyphen-joined) instead of the
    // catalog's `family:quant` colon convention. A request using any
    // recognized quant alias (`:q4`, the desktop UI's literal request) must
    // still resolve to that pack instead of a spurious 400.
    let temp = tempfile::tempdir().unwrap();
    let pack_root = temp.path().join("mimo-v2.5-asr-q4_k.oasr");
    write_whisper_oasr_v1_fixture(&pack_root, "mimo-v2.5-asr-q4_k");
    let app = openasr_server::app_with_runtime(openasr_server::ServerRuntime {
        backend: openasr_core::BackendKind::Native,
        native_execution: openasr_server::NativeExecutionSupervisor::default(),
        ffmpeg_bin: None,
        ffmpeg_bin_explicit: false,
        model_pack_path: Some(pack_root.clone()),
    });
    let wav_bytes = sample_wav_bytes();
    let request = multipart_request("mimo-v2.5-asr:q4", "sample.wav", &wav_bytes);
    let response = app.oneshot(request).await.unwrap();

    // Must clear the model-identity gate (whatever happens next in actual
    // native dispatch against a stand-in whisper fixture is not this test's
    // concern -- only that the mismatch rejection this bug caused is gone).
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let bytes = to_bytes(response.into_body(), 1024 * 256).await.unwrap();
    let body = String::from_utf8_lossy(&bytes);
    assert!(!body.contains("does not match server native local runtime source id"));
    assert!(!body.contains("retired legacy metadata id"));
}

#[tokio::test]
async fn transcriptions_with_native_backend_and_diarize_returns_bad_request() {
    let temp = tempfile::tempdir().unwrap();
    // Hermetic: diarization availability probes the installed ReDimNet2-B6 pack,
    // so pin the lookup to an empty home to keep the rejection deterministic.
    unsafe { std::env::remove_var("OPENASR_REDIMNET_PACK") };
    unsafe { std::env::set_var("OPENASR_HOME", temp.path()) };
    let pack_root = temp.path().join("whisper-runtime.oasr");
    write_whisper_oasr_v1_fixture(&pack_root, "whisper-runtime");
    let app = openasr_server::app_with_runtime(openasr_server::ServerRuntime {
        backend: openasr_core::BackendKind::Native,
        native_execution: openasr_server::NativeExecutionSupervisor::default(),
        ffmpeg_bin: None,
        ffmpeg_bin_explicit: false,
        model_pack_path: Some(pack_root.clone()),
    });
    let wav_bytes = sample_wav_bytes();
    let request = multipart_request_with_diarize("whisper-runtime", "sample.wav", &wav_bytes, true);
    let response = app.oneshot(request).await.unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let bytes = to_bytes(response.into_body(), 1024 * 256).await.unwrap();
    let body = String::from_utf8_lossy(&bytes);
    assert!(body.contains("speaker-embedder pack"));
    assert!(body.contains("redimnet2-b6-cn"));
}

#[tokio::test]
async fn transcriptions_with_native_backend_reject_retired_legacy_model_alias() {
    let app = openasr_server::app_with_runtime(openasr_server::ServerRuntime {
        backend: openasr_core::BackendKind::Native,
        native_execution: openasr_server::NativeExecutionSupervisor::default(),
        ffmpeg_bin: None,
        ffmpeg_bin_explicit: false,
        model_pack_path: None,
    });
    let wav_bytes = sample_wav_bytes();
    let request = multipart_request("whisper-tiny:q4_0", "sample.wav", &wav_bytes);
    let response = app.oneshot(request).await.unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let bytes = to_bytes(response.into_body(), 1024 * 256).await.unwrap();
    let body = String::from_utf8_lossy(&bytes);
    assert!(body.contains("retired legacy metadata id"));
}

#[tokio::test]
async fn transcriptions_with_native_backend_accepts_live_catalog_family_bare_metadata_id() {
    // Regression guard: a native pack's `openasr.model.id` metadata legitimately
    // carries the bare family id (no quant tag) per the "bare id" contract in
    // `native_model_refs_match`. `whisper-large-v3-turbo` is a live catalog
    // family (see model-registry/catalog.json), so it must not be treated as a
    // retired legacy id -- that would fail closed for every pack/pull of this
    // model. This must reach (and fail at) actual native execution, not the
    // retired-id or model-mismatch rejections.
    let temp = tempfile::tempdir().unwrap();
    let pack_root = temp.path().join("whisper-large-v3-turbo-q4_k.oasr");
    write_whisper_oasr_v1_fixture(&pack_root, "whisper-large-v3-turbo");
    let app = openasr_server::app_with_runtime(openasr_server::ServerRuntime {
        backend: openasr_core::BackendKind::Native,
        native_execution: openasr_server::NativeExecutionSupervisor::default(),
        ffmpeg_bin: None,
        ffmpeg_bin_explicit: false,
        model_pack_path: Some(pack_root.clone()),
    });
    let wav_bytes = sample_wav_bytes();
    let request = multipart_request("whisper-large-v3-turbo:q4_k", "sample.wav", &wav_bytes);
    let response = app.oneshot(request).await.unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let bytes = to_bytes(response.into_body(), 1024 * 256).await.unwrap();
    let body = String::from_utf8_lossy(&bytes);
    assert!(!body.contains("retired legacy metadata id"));
    assert!(!body.contains("does not match server native local runtime source id"));
}

#[tokio::test]
async fn stream_transcriptions_with_native_backend_reject_empty_model_form_value() {
    let temp = tempfile::tempdir().unwrap();
    let pack_root = temp.path().join("native-pack.oasr");
    write_mock_gguf_runtime_source(&pack_root, Some("native-pack"));
    let app = openasr_server::app_with_runtime(openasr_server::ServerRuntime {
        backend: openasr_core::BackendKind::Native,
        native_execution: openasr_server::NativeExecutionSupervisor::default(),
        ffmpeg_bin: None,
        ffmpeg_bin_explicit: false,
        model_pack_path: Some(pack_root),
    });
    let wav_bytes = sample_wav_bytes();
    let request = multipart_request_with_options(
        "/v1/audio/transcriptions?stream=true",
        "   ",
        "sample.wav",
        &wav_bytes,
        false,
        None,
    );
    let response = app.oneshot(request).await.unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let bytes = to_bytes(response.into_body(), 1024 * 256).await.unwrap();
    let body = String::from_utf8_lossy(&bytes);
    assert!(body.contains("must be a non-empty model id"));
}

#[tokio::test]
async fn stream_transcriptions_with_mock_backend_emits_protocol_events() {
    let app = openasr_server::app_with_runtime(openasr_server::ServerRuntime {
        backend: openasr_core::BackendKind::Mock,
        native_execution: openasr_server::NativeExecutionSupervisor::default(),
        ffmpeg_bin: None,
        ffmpeg_bin_explicit: false,
        model_pack_path: None,
    });
    let wav_bytes = sample_wav_bytes();
    let request = multipart_request_with_options(
        "/v1/audio/transcriptions?stream=true",
        "whisper-large-v3-turbo",
        "sample.wav",
        &wav_bytes,
        false,
        None,
    );
    let response = app.oneshot(request).await.unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get(header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok()),
        Some("text/event-stream")
    );
    let bytes = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
    let body = String::from_utf8_lossy(&bytes);
    assert!(body.contains("event: segment_start"));
    assert!(body.contains("event: final"));
    assert!(body.contains("event: segment_end"));
    assert!(body.contains("event: done"));
    assert!(body.contains("\"totalLatencyMs\":"));
}

#[tokio::test]
async fn stream_transcription_succeeds_when_history_cannot_be_recorded() {
    let temp = tempfile::tempdir().unwrap();
    let home = temp.path().join("home-as-file");
    std::fs::write(&home, b"not a directory").unwrap();
    let app = openasr_server::app_with_runtime_and_distribution(
        openasr_server::ServerRuntime {
            backend: openasr_core::BackendKind::Mock,
            native_execution: openasr_server::NativeExecutionSupervisor::default(),
            ffmpeg_bin: None,
            ffmpeg_bin_explicit: false,
            model_pack_path: None,
        },
        openasr_server::DistributionRuntime {
            openasr_home: Some(home),
            catalog_url: None,
            catalog_local_override: None,
        },
    );
    let wav_bytes = sample_wav_bytes();
    let request = multipart_request_with_options(
        "/v1/audio/transcriptions?stream=true",
        "whisper-large-v3-turbo",
        "sample.wav",
        &wav_bytes,
        false,
        None,
    );

    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
    let body = String::from_utf8_lossy(&body);
    assert!(body.contains("event: final"));
    assert!(body.contains("event: done"));
    assert!(!body.contains("event: error"));
    assert!(body.contains("\"status\":\"ok\""));
}

#[tokio::test]
async fn static_bearer_remote_compute_stream_transcription_records_server_history() {
    let temp = tempfile::tempdir().unwrap();
    enable_history(&temp);
    let app = openasr_server::app_with_runtime_and_distribution_and_launch_options(
        openasr_server::ServerRuntime {
            backend: openasr_core::BackendKind::Mock,
            native_execution: openasr_server::NativeExecutionSupervisor::default(),
            ffmpeg_bin: None,
            ffmpeg_bin_explicit: false,
            model_pack_path: None,
        },
        openasr_server::DistributionRuntime {
            openasr_home: Some(temp.path().join("home")),
            catalog_url: None,
            catalog_local_override: None,
        },
        openasr_server::ServerLaunchOptions {
            auth: openasr_server::ServerAuth::bearer("remote-secret"),
            ..Default::default()
        },
    );
    let wav_bytes = sample_wav_bytes();
    let mut request = multipart_request_with_options(
        "/v1/audio/transcriptions?stream=true",
        "whisper-large-v3-turbo",
        "sample.wav",
        &wav_bytes,
        false,
        None,
    );
    request.headers_mut().insert(
        header::AUTHORIZATION,
        "Bearer remote-secret".parse().unwrap(),
    );
    request
        .headers_mut()
        .insert("x-openasr-remote-compute", "client".parse().unwrap());

    let response = app.clone().oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
    assert!(String::from_utf8_lossy(&body).contains("event: done"));

    let history = app
        .oneshot(
            Request::builder()
                .uri("/v1/history")
                .header(header::AUTHORIZATION, "Bearer remote-secret")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(history.status(), StatusCode::OK);
    let bytes = to_bytes(history.into_body(), 1024 * 64).await.unwrap();
    let parsed: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(
        parsed["data"].as_array().unwrap().len(),
        1,
        "static bearer auth is not a paired remote-compute device token"
    );
}

#[tokio::test]
async fn stream_transcriptions_with_native_backend_reject_srt_response_format() {
    let temp = tempfile::tempdir().unwrap();
    let pack_root = temp.path().join("native-pack.oasr");
    write_mock_gguf_runtime_source(&pack_root, Some("native-pack"));
    let app = openasr_server::app_with_runtime(openasr_server::ServerRuntime {
        backend: openasr_core::BackendKind::Native,
        native_execution: openasr_server::NativeExecutionSupervisor::default(),
        ffmpeg_bin: None,
        ffmpeg_bin_explicit: false,
        model_pack_path: Some(pack_root),
    });
    let wav_bytes = sample_wav_bytes();
    let request = multipart_request_with_options(
        "/v1/audio/transcriptions?stream=true",
        "native-pack",
        "sample.wav",
        &wav_bytes,
        false,
        Some("srt"),
    );
    let response = app.oneshot(request).await.unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let bytes = to_bytes(response.into_body(), 1024 * 256).await.unwrap();
    let body = String::from_utf8_lossy(&bytes);
    assert!(body.contains("does not support SRT/VTT response_format"));
}

#[tokio::test]
async fn stream_transcriptions_with_native_backend_reject_model_mismatch() {
    let temp = tempfile::tempdir().unwrap();
    let pack_root = temp.path().join("native-pack.oasr");
    write_mock_gguf_runtime_source(&pack_root, Some("native-pack"));
    let app = openasr_server::app_with_runtime(openasr_server::ServerRuntime {
        backend: openasr_core::BackendKind::Native,
        native_execution: openasr_server::NativeExecutionSupervisor::default(),
        ffmpeg_bin: None,
        ffmpeg_bin_explicit: false,
        model_pack_path: Some(pack_root),
    });
    let wav_bytes = sample_wav_bytes();
    let request = multipart_request_with_options(
        "/v1/audio/transcriptions?stream=true",
        "native-pack:typo",
        "sample.wav",
        &wav_bytes,
        false,
        None,
    );
    let response = app.oneshot(request).await.unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let bytes = to_bytes(response.into_body(), 1024 * 256).await.unwrap();
    let body = String::from_utf8_lossy(&bytes);
    assert!(body.contains("event: error"));
    assert!(body.contains("\"status\":\"error\""));
}

#[tokio::test]
async fn stream_transcriptions_with_native_xasr_hotword_emits_model_unsupported_error() {
    let temp = tempfile::tempdir().unwrap();
    let model_id = "xasr-hotword-sse";
    let pack_root = temp.path().join("xasr-hotword-sse.oasr");
    write_xasr_gguf_runtime_source(&pack_root, model_id);
    let app = openasr_server::app_with_runtime(openasr_server::ServerRuntime {
        backend: openasr_core::BackendKind::Native,
        native_execution: openasr_server::NativeExecutionSupervisor::default(),
        ffmpeg_bin: None,
        ffmpeg_bin_explicit: false,
        model_pack_path: Some(pack_root),
    });
    let wav_bytes = sample_wav_bytes();
    let request = multipart_request_with_extra_fields(
        "/v1/audio/transcriptions?stream=true",
        model_id,
        "sample.wav",
        &wav_bytes,
        &[("hotword", "OpenASR")],
    );
    let response = app.oneshot(request).await.unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get(header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok()),
        Some("text/event-stream")
    );
    let bytes = to_bytes(response.into_body(), 1024 * 256).await.unwrap();
    let body = String::from_utf8_lossy(&bytes);
    assert!(body.contains("event: error"));
    assert!(body.contains("\"status\":\"error\""));
    assert!(body.contains("Phrase bias / hotword boosting is not supported"));
    assert!(body.contains("'xasr-zipformer' native model family"));
    assert!(body.contains("ggml-family-xasr-zipformer-runtime-v1"));
    assert!(body.contains("silently ignoring phrase_bias"));
    assert!(!body.contains("stayed fail-closed"));
}

#[tokio::test]
async fn stream_transcriptions_with_native_backend_reject_missing_model_pack_path() {
    let app = openasr_server::app_with_runtime(openasr_server::ServerRuntime {
        backend: openasr_core::BackendKind::Native,
        native_execution: openasr_server::NativeExecutionSupervisor::default(),
        ffmpeg_bin: None,
        ffmpeg_bin_explicit: false,
        model_pack_path: None,
    });
    let wav_bytes = sample_wav_bytes();
    // The streaming endpoint's synchronous multipart parse only runs the retired-id
    // check (missing model_pack_path is validated deeper, inside the spawned
    // transcribe task, so it never surfaces as a synchronous 400 here). Use a
    // still-retired tagged id -- not a live catalog family like
    // `whisper-large-v3-turbo`, which is no longer blacklisted -- so this keeps
    // exercising a real synchronous rejection instead of relying on that pack
    // path never being reached for an unrelated reason.
    let request = multipart_request_with_options(
        "/v1/audio/transcriptions?stream=true",
        "whisper-tiny:q4_0",
        "sample.wav",
        &wav_bytes,
        false,
        None,
    );
    let response = app.oneshot(request).await.unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let bytes = to_bytes(response.into_body(), 1024 * 256).await.unwrap();
    let body = String::from_utf8_lossy(&bytes);
    assert!(!body.trim().is_empty());
}

#[tokio::test]
async fn transcriptions_with_native_backend_srt_stays_fail_closed_for_unexecutable_runtime() {
    let temp = tempfile::tempdir().unwrap();
    let pack_root = temp.path().join("native-pack.oasr");
    write_mock_gguf_runtime_source(&pack_root, Some("native-pack"));
    let app = openasr_server::app_with_runtime(openasr_server::ServerRuntime {
        backend: openasr_core::BackendKind::Native,
        native_execution: openasr_server::NativeExecutionSupervisor::default(),
        ffmpeg_bin: None,
        ffmpeg_bin_explicit: false,
        model_pack_path: Some(pack_root.clone()),
    });
    let wav_bytes = sample_wav_bytes();
    let request = multipart_request_with_options(
        "/v1/audio/transcriptions",
        "native-pack",
        "sample.wav",
        &wav_bytes,
        false,
        Some("srt"),
    );
    let response = app.oneshot(request).await.unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let bytes = to_bytes(response.into_body(), 1024 * 256).await.unwrap();
    let body = String::from_utf8_lossy(&bytes);
    assert!(!body.trim().is_empty());
}

#[tokio::test]
async fn models_with_native_backend_lists_loaded_local_pack_id() {
    let temp = tempfile::tempdir().unwrap();
    let pack_root = temp.path().join("native-pack.oasr");
    write_mock_gguf_runtime_source(&pack_root, Some("native-pack"));
    let app = openasr_server::app_with_runtime(openasr_server::ServerRuntime {
        backend: openasr_core::BackendKind::Native,
        native_execution: openasr_server::NativeExecutionSupervisor::default(),
        ffmpeg_bin: None,
        ffmpeg_bin_explicit: false,
        model_pack_path: Some(pack_root),
    });
    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let bytes = to_bytes(response.into_body(), 1024 * 64).await.unwrap();
    let parsed: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(parsed["data"][0]["id"], "native-pack");
}

#[tokio::test]
async fn models_with_native_backend_and_no_pack_lists_empty_instead_of_erroring() {
    // Zero installed models is a normal listing result (an empty catalog of
    // "currently loaded" models), not an error -- /v1/models must not fail
    // closed the way an actual transcription request does.
    let app = openasr_server::app_with_runtime(openasr_server::ServerRuntime {
        backend: openasr_core::BackendKind::Native,
        native_execution: openasr_server::NativeExecutionSupervisor::default(),
        ffmpeg_bin: None,
        ffmpeg_bin_explicit: false,
        model_pack_path: None,
    });
    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let bytes = to_bytes(response.into_body(), 1024 * 64).await.unwrap();
    let parsed: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(parsed["data"].as_array().unwrap().len(), 0);
}

// ── Full bundled-catalog x quant-alias matrix for native model matching ─────
//
// Regression net for the `mimo-v2.5-asr:q4` alias-matching bug: instead of
// hand-writing one assertion per family (which only ever covers the families
// someone remembered to write a test for), this walks the bundled catalog
// (`model-registry/catalog.json`, the same file the daemon ships) and, for
// every model x every quant variant it declares, exercises every legal
// reference form (bare id, `id:<catalog suffix>`, `id:<catalog quant label>`,
// `id:<canonical tag>`, and every other recognized alias sharing that
// canonical tag) against both a colon-tagged and a bare native runtime source
// id. Any newly onboarded catalog model is covered automatically -- no
// per-family test to remember to add. Pure id-matching logic: no weights, no
// backend, no network.
mod native_quant_alias_catalog_matrix {
    use openasr_core::{canonical_quant_tag, native_runtime_model_refs_match};
    use serde_json::Value;

    // The full set of alternate spellings recognized by `canonical_quant_tag`
    // (crates/openasr-core/src/registry.rs `QUANT_ALIAS_GROUPS`), grouped by
    // the canonical tag they collapse onto. Kept here only to *generate test
    // inputs* (never to reimplement the alias decision) -- the assertions all
    // go back through `canonical_quant_tag` / `native_runtime_model_refs_match`
    // themselves, so this table drifting from the real one would only ever
    // make the matrix *weaker*, never spuriously fail.
    const ALL_ALIAS_SPELLINGS: &[&str] = &[
        "q8", "q8_0", "q4", "q4_k", "q4_k_m", "q4km", "q3", "q3_k", "fp16",
    ];

    fn bundled_catalog() -> Value {
        let raw = include_str!("../../../model-registry/catalog.json");
        serde_json::from_str(raw).expect("bundled catalog.json parses")
    }

    struct QuantCase {
        model_id: String,
        // Every legal request-side tag for this quant variant: the catalog's
        // own "quant" and "suffix" spellings plus every other recognized
        // alias sharing the same canonical target.
        request_tags: Vec<String>,
        canonical_tag: String,
    }

    fn catalog_quant_cases(catalog: &Value) -> Vec<QuantCase> {
        let mut cases = Vec::new();
        for model in catalog["models"].as_array().expect("models array") {
            let model_id = model["id"].as_str().expect("model id").to_string();
            let quants = match model["quants"].as_array() {
                Some(quants) => quants,
                None => continue,
            };
            for quant in quants {
                let quant_label = quant["quant"].as_str().expect("quant label");
                let suffix = quant["suffix"].as_str().expect("quant suffix");
                let canonical_tag = canonical_quant_tag(quant_label).to_string();

                let mut request_tags: Vec<String> = vec![
                    quant_label.to_string(),
                    suffix.to_string(),
                    canonical_tag.clone(),
                ];
                for alias in ALL_ALIAS_SPELLINGS {
                    if canonical_quant_tag(alias) == canonical_tag {
                        request_tags.push(alias.to_string());
                    }
                }
                request_tags.sort();
                request_tags.dedup();

                cases.push(QuantCase {
                    model_id: model_id.clone(),
                    request_tags,
                    canonical_tag,
                });
            }
        }
        cases
    }

    #[test]
    fn every_catalog_model_quant_alias_form_matches_its_own_tagged_and_bare_runtime_source_id() {
        let catalog = bundled_catalog();
        let cases = catalog_quant_cases(&catalog);
        assert!(
            cases.len() >= 40,
            "expected broad catalog coverage, got {} (model-registry/catalog.json shrank or failed to parse?)",
            cases.len()
        );

        let mut failures = Vec::new();
        for case in &cases {
            let tagged_runtime_source_id = format!("{}:{}", case.model_id, case.canonical_tag);
            let bare_runtime_source_id = case.model_id.clone();

            for tag in &case.request_tags {
                let requested = format!("{}:{}", case.model_id, tag);

                if !native_runtime_model_refs_match(&requested, &tagged_runtime_source_id) {
                    failures.push(format!(
                        "{requested} vs tagged runtime source {tagged_runtime_source_id}: expected match"
                    ));
                }
                // Bare-id contract: a quant-pinned request must also match a
                // pack whose metadata carries no quant tag at all (the common
                // real-world case -- most families burn no quant into
                // `openasr.model.id`).
                if !native_runtime_model_refs_match(&requested, &bare_runtime_source_id) {
                    failures.push(format!(
                        "{requested} vs bare runtime source {bare_runtime_source_id}: expected match (bare-id contract)"
                    ));
                }
            }

            // A bare request matches a bare runtime source id exactly...
            if !native_runtime_model_refs_match(&case.model_id, &bare_runtime_source_id) {
                failures.push(format!(
                    "{} vs bare runtime source {bare_runtime_source_id}: expected match",
                    case.model_id
                ));
            }
            // ...but must NOT match a quant-tagged one: an unpinned request
            // naming a family with a quant-pinned pack loaded is ambiguous
            // about which quant it means, so this stays a refusal by design,
            // not a bug.
            if native_runtime_model_refs_match(&case.model_id, &tagged_runtime_source_id) {
                failures.push(format!(
                    "{} vs tagged runtime source {tagged_runtime_source_id}: expected NO match (bare request against a quant-pinned pack is ambiguous)",
                    case.model_id
                ));
            }
        }

        assert!(
            failures.is_empty(),
            "quant-alias matrix regressions ({} of {} cases affected):\n{}",
            failures.len(),
            cases.len(),
            failures.join("\n")
        );
    }

    #[test]
    fn cross_family_and_cross_quant_requests_still_fail_closed() {
        let catalog = bundled_catalog();
        let cases = catalog_quant_cases(&catalog);

        // Cross-family: no two distinct catalog model ids may spuriously
        // match each other, even when one is a quant-pinned request.
        let mut sampled_families: Vec<&str> = Vec::new();
        for case in &cases {
            if !sampled_families.contains(&case.model_id.as_str()) {
                sampled_families.push(&case.model_id);
            }
        }
        assert!(
            sampled_families.len() >= 10,
            "expected multiple distinct catalog families to sample"
        );
        for window in sampled_families.windows(2) {
            let (a, b) = (window[0], window[1]);
            assert!(
                !native_runtime_model_refs_match(&format!("{a}:q8"), &format!("{b}:q8_0")),
                "{a}:q8 must not match unrelated family runtime source {b}:q8_0"
            );
        }

        // Cross-quant, same family: a model with more than one quant variant
        // must fail closed when the request pins a different quant than the
        // loaded tagged pack.
        for case in &cases {
            for other in &cases {
                if other.model_id == case.model_id && other.canonical_tag != case.canonical_tag {
                    let requested = format!("{}:{}", case.model_id, case.canonical_tag);
                    let other_tagged_runtime_source_id =
                        format!("{}:{}", other.model_id, other.canonical_tag);
                    assert!(
                        !native_runtime_model_refs_match(
                            &requested,
                            &other_tagged_runtime_source_id
                        ),
                        "{requested} must not match a differently-quantized loaded pack {other_tagged_runtime_source_id}"
                    );
                }
            }
        }
    }
}
