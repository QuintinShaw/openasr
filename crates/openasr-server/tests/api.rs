use axum::{
    Router,
    body::{Body, to_bytes},
    http::{Request, StatusCode, header},
};
use futures_util::StreamExt;
use openasr_core::api::backend::transcribe_with_mock_backend;
use openasr_core::testing::{
    TinyGgufFixtureSpec, write_reserved_oasr_container, write_tiny_gguf_runtime_source,
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

fn sample_wav_bytes() -> Vec<u8> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/jfk.wav");
    std::fs::read(path).unwrap()
}

struct EnvRestore {
    key: &'static str,
    previous: Option<std::ffi::OsString>,
}

impl EnvRestore {
    fn set(key: &'static str, value: Option<&std::path::Path>) -> Self {
        let restore = Self {
            key,
            previous: std::env::var_os(key),
        };
        match value {
            Some(path) => unsafe { std::env::set_var(key, path) },
            None => unsafe { std::env::remove_var(key) },
        }
        restore
    }
}

impl Drop for EnvRestore {
    fn drop(&mut self) {
        match self.previous.take() {
            Some(value) => unsafe { std::env::set_var(self.key, value) },
            None => unsafe { std::env::remove_var(self.key) },
        }
    }
}

async fn with_empty_openasr_home<T, F>(home: &std::path::Path, future: F) -> T
where
    F: std::future::Future<Output = T>,
{
    static ENV_LOCK: OnceLock<tokio::sync::Mutex<()>> = OnceLock::new();
    let lock = ENV_LOCK.get_or_init(|| tokio::sync::Mutex::new(()));
    let _guard = lock.lock().await;
    let _home = EnvRestore::set("OPENASR_HOME", Some(home));
    let _wespeaker = EnvRestore::set("OPENASR_WESPEAKER_PACK", None);
    future.await
}

fn pcm16_mono_wav_bytes(seconds: u32, sample_value: i16) -> Vec<u8> {
    let sample_rate = 16_000u32;
    let samples = sample_rate * seconds;
    let data_bytes = samples * 2;
    let mut out = Vec::with_capacity(44 + data_bytes as usize);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_bytes).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&(sample_rate * 2).to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_bytes.to_le_bytes());
    for _ in 0..samples {
        out.extend_from_slice(&sample_value.to_le_bytes());
    }
    out
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
    std::fs::write(
        &catalog_path,
        serde_json::to_vec_pretty(&catalog).expect("serialize catalog fixture"),
    )
    .unwrap();

    (
        pack_path,
        openasr_server::DistributionRuntime {
            openasr_home: Some(root.join("home")),
            catalog_url: Some(format!("file://{}", catalog_path.display())),
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
    std::fs::write(catalog_path, serde_json::to_vec_pretty(&catalog).unwrap()).unwrap();
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

fn write_complete_moonshine_partial(home: &std::path::Path, source_pack: &std::path::Path) -> u64 {
    let bytes = std::fs::read(source_pack).unwrap();
    let sha256 = format!("{:x}", Sha256::digest(&bytes));
    let revision = "0123456789abcdef0123456789abcdef01234567";
    let url = format!(
        "https://huggingface.co/OpenASR/moonshine-tiny/resolve/{revision}/moonshine-tiny-q8_0.oasr"
    );
    let model_dir = home.join("models").join("moonshine-tiny").join("q8_0");
    std::fs::create_dir_all(&model_dir).unwrap();
    let partial_path = model_dir.join("moonshine-tiny-q8_0.oasr.partial");
    let partial_meta_path = model_dir.join("moonshine-tiny-q8_0.oasr.partial.meta.json");
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
    std::fs::write(catalog_path, serde_json::to_vec_pretty(&catalog).unwrap()).unwrap();
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
    // Fresh-config product defaults surfaced to the desktop: Option (⌥) alone,
    // push-to-talk on. These are what a cleared-state first launch shows.
    assert_eq!(document["preferences"]["dictation_shortcut"], "Alt");
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
        ffmpeg_bin: None,
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
            ffmpeg_bin: None,
            model_pack_path: None,
        },
        openasr_server::DistributionRuntime {
            openasr_home: Some(home.clone()),
            catalog_url: None,
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

    assert_ne!(
        response.status(),
        StatusCode::OK,
        "native serve must not transcribe an uninstalled model"
    );
    assert!(
        response.status().is_client_error() || response.status().is_server_error(),
        "expected a fail-closed error status, got {}",
        response.status()
    );
    assert!(
        !home.join("models").exists(),
        "the server must never download a model"
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
    assert!(
        completed["installed_path"]
            .as_str()
            .unwrap()
            .ends_with(".oasr")
    );

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
    assert!(
        completed["installed_path"]
            .as_str()
            .unwrap()
            .ends_with("moonshine-tiny-q8_0.oasr")
    );

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
    assert!(
        completed["installed_path"]
            .as_str()
            .unwrap()
            .ends_with("moonshine-tiny-q8_0.oasr")
    );
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
    assert!(
        completed["installed_path"]
            .as_str()
            .unwrap()
            .ends_with("moonshine-tiny-q8_0.oasr")
    );
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

fn speaker_multipart_request(uri: &str, name: Option<&str>, wav_bytes: &[u8]) -> Request<Body> {
    let boundary = "openasr-speaker-boundary";
    let mut body = Vec::new();
    if let Some(name) = name {
        body.extend_from_slice(
            format!(
                "--{boundary}\r\nContent-Disposition: form-data; name=\"name\"\r\n\r\n{name}\r\n"
            )
            .as_bytes(),
        );
    }
    body.extend_from_slice(
        format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"wav\"; filename=\"speaker.wav\"\r\nContent-Type: audio/wav\r\n\r\n"
        )
        .as_bytes(),
    );
    body.extend_from_slice(wav_bytes);
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

fn write_voiceprint_store(home: &std::path::Path) -> String {
    let id = "vp_aaaaaaaaaaaaaaaa".to_string();
    let profile = openasr_core::diarize::enrollment::SpeakerProfile {
        id: id.clone(),
        name: "Alice".to_string(),
        created_at: "2026-06-11T00:00:00.000Z".to_string(),
        updated_at: "2026-06-11T00:00:00.000Z".to_string(),
        sample_seconds: 5.25,
        embedding_dim: 2,
        pack_fingerprint: "sha256:test".to_string(),
        match_similarity: 0.5,
        embedding: vec![1.0, 0.0],
    };
    let store = openasr_core::diarize::enrollment::VoiceprintStore {
        version: openasr_core::diarize::enrollment::VOICEPRINT_STORE_VERSION,
        profiles: vec![profile],
    };
    store
        .save(&home.join("diarize").join("voiceprints.json"))
        .unwrap();
    id
}

#[tokio::test]
async fn speaker_routes_require_operator_credentials_for_paired_devices() {
    let temp = tempfile::tempdir().unwrap();
    let app = openasr_server::app_with_runtime_and_distribution_and_launch_options(
        openasr_server::ServerRuntime::default(),
        openasr_server::DistributionRuntime {
            openasr_home: Some(temp.path().join("home")),
            catalog_url: None,
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
                .uri("/v1/speakers")
                .header(header::AUTHORIZATION, format!("Bearer {bearer_token}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(paired.status(), StatusCode::FORBIDDEN);

    let operator = app
        .oneshot(
            Request::builder()
                .uri("/v1/speakers")
                .header(header::AUTHORIZATION, "Bearer admin-secret")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(operator.status(), StatusCode::OK);
}

#[tokio::test]
async fn speaker_routes_list_rename_and_delete_profiles() {
    let temp = tempfile::tempdir().unwrap();
    let home = temp.path().join("home");
    let id = write_voiceprint_store(&home);
    let app = openasr_server::app_with_runtime_and_distribution(
        openasr_server::ServerRuntime::default(),
        openasr_server::DistributionRuntime {
            openasr_home: Some(home),
            catalog_url: None,
        },
    );

    let list = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/speakers")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(list.status(), StatusCode::OK);
    let body = to_bytes(list.into_body(), 1024 * 64).await.unwrap();
    let json: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["data"][0]["id"], id);
    assert_eq!(json["data"][0]["name"], "Alice");
    assert_eq!(json["data"][0]["sample_seconds"], 5.25);
    assert_eq!(json["data"][0]["compatible"], false);

    let renamed = app
        .clone()
        .oneshot(
            Request::builder()
                .method("PATCH")
                .uri(format!("/v1/speakers/{id}"))
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    serde_json::json!({ "name": "Alicia" }).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(renamed.status(), StatusCode::OK);
    let body = to_bytes(renamed.into_body(), 1024 * 64).await.unwrap();
    let json: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["id"], id);
    assert_eq!(json["name"], "Alicia");

    let deleted = app
        .clone()
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri(format!("/v1/speakers/{id}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(deleted.status(), StatusCode::OK);
    let body = to_bytes(deleted.into_body(), 1024 * 64).await.unwrap();
    let json: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["id"], id);
    assert_eq!(json["deleted"], true);

    let list = app
        .oneshot(
            Request::builder()
                .uri("/v1/speakers")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let body = to_bytes(list.into_body(), 1024 * 64).await.unwrap();
    let json: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["data"].as_array().unwrap().len(), 0);
}

#[tokio::test]
async fn speaker_enrollment_routes_reject_short_silent_and_missing_pack() {
    let temp = tempfile::tempdir().unwrap();
    let home = temp.path().join("home");
    let app = openasr_server::app_with_runtime_and_distribution(
        openasr_server::ServerRuntime::default(),
        openasr_server::DistributionRuntime {
            openasr_home: Some(home.clone()),
            catalog_url: None,
        },
    );

    with_empty_openasr_home(&home, async {
        let short = app
            .clone()
            .oneshot(speaker_multipart_request(
                "/v1/speakers",
                Some("Alice"),
                &pcm16_mono_wav_bytes(4, 1_000),
            ))
            .await
            .unwrap();
        assert_eq!(short.status(), StatusCode::BAD_REQUEST);
        let body = to_bytes(short.into_body(), 1024 * 64).await.unwrap();
        let json: Value = serde_json::from_slice(&body).unwrap();
        assert!(
            json["error"]["message"]
                .as_str()
                .unwrap()
                .contains("too short")
        );

        let silent = app
            .clone()
            .oneshot(speaker_multipart_request(
                "/v1/speakers",
                Some("Alice"),
                &pcm16_mono_wav_bytes(6, 0),
            ))
            .await
            .unwrap();
        assert_eq!(silent.status(), StatusCode::BAD_REQUEST);
        let body = to_bytes(silent.into_body(), 1024 * 64).await.unwrap();
        let json: Value = serde_json::from_slice(&body).unwrap();
        assert!(
            json["error"]["message"]
                .as_str()
                .unwrap()
                .contains("silent")
        );

        let missing_pack = app
            .clone()
            .oneshot(speaker_multipart_request(
                "/v1/speakers",
                Some("Alice"),
                &pcm16_mono_wav_bytes(6, 1_000),
            ))
            .await
            .unwrap();
        assert_eq!(missing_pack.status(), StatusCode::BAD_REQUEST);
        let body = to_bytes(missing_pack.into_body(), 1024 * 64).await.unwrap();
        let json: Value = serde_json::from_slice(&body).unwrap();
        assert!(
            json["error"]["message"]
                .as_str()
                .unwrap()
                .contains("speaker-embedder pack")
        );
    })
    .await;
}

#[tokio::test]
async fn speaker_reenroll_fails_closed_when_embedder_pack_is_missing() {
    let temp = tempfile::tempdir().unwrap();
    let home = temp.path().join("home");
    let id = write_voiceprint_store(&home);
    let app = openasr_server::app_with_runtime_and_distribution(
        openasr_server::ServerRuntime::default(),
        openasr_server::DistributionRuntime {
            openasr_home: Some(home.clone()),
            catalog_url: None,
        },
    );

    with_empty_openasr_home(&home, async {
        let response = app
            .oneshot(speaker_multipart_request(
                &format!("/v1/speakers/{id}/reenroll"),
                None,
                &pcm16_mono_wav_bytes(6, 1_000),
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = to_bytes(response.into_body(), 1024 * 64).await.unwrap();
        let json: Value = serde_json::from_slice(&body).unwrap();
        assert!(
            json["error"]["message"]
                .as_str()
                .unwrap()
                .contains("speaker-embedder pack")
        );
    })
    .await;
}

#[test]
fn native_server_runtime_is_rejected_at_startup_validation() {
    let error = openasr_server::ServerRuntime {
        backend: openasr_core::BackendKind::Native,
        ffmpeg_bin: None,
        model_pack_path: None,
    }
    .validate()
    .unwrap_err()
    .to_string();

    assert!(error.contains("requires an explicit local .oasr runtime pack path"));
}

#[test]
fn native_server_runtime_falls_back_to_path_stem_when_metadata_model_id_is_retired() {
    let temp = tempfile::tempdir().unwrap();
    let pack_root = temp.path().join("native-pack.oasr");
    write_mock_gguf_runtime_source(&pack_root, Some("whisper-tiny:q4_0"));
    openasr_server::ServerRuntime {
        backend: openasr_core::BackendKind::Native,
        ffmpeg_bin: None,
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
        ffmpeg_bin: None,
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
        ffmpeg_bin: None,
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
    let diarize_field = if diarize {
        format!("\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"diarize\"\r\n\r\ntrue")
    } else {
        String::new()
    };
    let response_format_field = response_format.map_or_else(String::new, |value| {
        format!(
            "\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"response_format\"\r\n\r\n{value}"
        )
    });
    let body = format!(
        "--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{file_name}\"\r\nContent-Type: audio/wav\r\n\r\n{}\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\n{model}{diarize_field}{response_format_field}\r\n--{boundary}--\r\n",
        String::from_utf8_lossy(bytes),
    );

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
    let extra_fields = fields
        .iter()
        .map(|(name, value)| {
            format!(
                "\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}"
            )
        })
        .collect::<String>();
    let body = format!(
        "--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{file_name}\"\r\nContent-Type: audio/wav\r\n\r\n{}\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\n{model}{extra_fields}\r\n--{boundary}--\r\n",
        String::from_utf8_lossy(bytes),
    );

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
    assert_eq!(parsed.as_object().expect("health response object").len(), 4);
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
    assert_eq!(parsed.as_object().expect("health response object").len(), 4);
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
    assert_eq!(parsed.as_object().expect("health response object").len(), 4);
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

/// An upload past the server's body-size ceiling must fail with a clear,
/// actionable "file too large" message and 413, not the generic "Error
/// parsing `multipart/form-data` request" text that `MultipartError`'s
/// `Display` renders for every underlying `multer` failure (including this
/// one) -- see `multipart_error_message` in `lib.rs`.
#[tokio::test]
async fn transcriptions_reject_upload_past_body_limit_with_clear_message() {
    let temp = tempfile::tempdir().unwrap();
    let app = openasr_server::app_with_runtime_and_distribution(
        openasr_server::ServerRuntime::default(),
        openasr_server::DistributionRuntime {
            openasr_home: Some(temp.path().join("home")),
            catalog_url: None,
        },
    );
    let oversized = vec![0u8; 65 * 1024 * 1024];
    let request = multipart_request("whisper-large-v3-turbo", "huge.wav", &oversized);
    let response = app.oneshot(request).await.unwrap();

    assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
    let bytes = to_bytes(response.into_body(), 1024 * 256).await.unwrap();
    let parsed: Value = serde_json::from_slice(&bytes).unwrap();
    let message = parsed["error"]["message"].as_str().unwrap();
    assert!(message.contains("too large"), "message was: {message}");
    assert!(
        !message.contains("Error parsing"),
        "message regressed to the generic multipart error text: {message}"
    );
}

#[tokio::test]
async fn transcriptions_accept_word_timestamp_granularity_for_json() {
    let temp = tempfile::tempdir().unwrap();
    let app = openasr_server::app_with_runtime_and_distribution(
        openasr_server::ServerRuntime::default(),
        openasr_server::DistributionRuntime {
            openasr_home: Some(temp.path().join("home")),
            catalog_url: None,
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
        formats: vec!["text".to_string()],
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
async fn remote_compute_header_without_auth_still_records_server_history() {
    let temp = tempfile::tempdir().unwrap();
    enable_history(&temp);
    let app = openasr_server::app_with_runtime_and_distribution(
        openasr_server::ServerRuntime::default(),
        openasr_server::DistributionRuntime {
            openasr_home: Some(temp.path().join("home")),
            catalog_url: None,
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
        ffmpeg_bin: None,
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
        ffmpeg_bin: None,
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
        ffmpeg_bin: None,
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
}

#[tokio::test]
async fn transcriptions_with_native_backend_and_diarize_returns_bad_request() {
    let temp = tempfile::tempdir().unwrap();
    // Hermetic: diarization availability probes the installed WeSpeaker pack,
    // so pin the lookup to an empty home to keep the rejection deterministic.
    unsafe { std::env::remove_var("OPENASR_WESPEAKER_PACK") };
    unsafe { std::env::set_var("OPENASR_HOME", temp.path()) };
    let pack_root = temp.path().join("whisper-runtime.oasr");
    write_whisper_oasr_v1_fixture(&pack_root, "whisper-runtime");
    let app = openasr_server::app_with_runtime(openasr_server::ServerRuntime {
        backend: openasr_core::BackendKind::Native,
        ffmpeg_bin: None,
        model_pack_path: Some(pack_root.clone()),
    });
    let wav_bytes = sample_wav_bytes();
    let request = multipart_request_with_diarize("whisper-runtime", "sample.wav", &wav_bytes, true);
    let response = app.oneshot(request).await.unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let bytes = to_bytes(response.into_body(), 1024 * 256).await.unwrap();
    let body = String::from_utf8_lossy(&bytes);
    assert!(body.contains("speaker-embedder pack"));
}

#[tokio::test]
async fn transcriptions_with_native_backend_reject_retired_legacy_model_alias() {
    let app = openasr_server::app_with_runtime(openasr_server::ServerRuntime {
        backend: openasr_core::BackendKind::Native,
        ffmpeg_bin: None,
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
        ffmpeg_bin: None,
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
        ffmpeg_bin: None,
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
        ffmpeg_bin: None,
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
            ffmpeg_bin: None,
            model_pack_path: None,
        },
        openasr_server::DistributionRuntime {
            openasr_home: Some(home),
            catalog_url: None,
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
            ffmpeg_bin: None,
            model_pack_path: None,
        },
        openasr_server::DistributionRuntime {
            openasr_home: Some(temp.path().join("home")),
            catalog_url: None,
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
        ffmpeg_bin: None,
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
        ffmpeg_bin: None,
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
        ffmpeg_bin: None,
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
        ffmpeg_bin: None,
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
        ffmpeg_bin: None,
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
        ffmpeg_bin: None,
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
