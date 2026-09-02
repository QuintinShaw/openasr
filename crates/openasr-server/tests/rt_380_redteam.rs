//! Red-team HTTP falsifiers for PR #380 WeSpeaker preference promises.
//!
//! Predicted observation if the PR holds: a persisted `voice_id_embedder=wespeaker`
//! preference fail-closes file JSON / stream / translations with the WeSpeaker
//! pack id in the error. Otherwise the ReDimNet probe or the stream skip loads
//! the default 192-d space / ReDimNet copy.

use axum::{
    body::{Body, to_bytes},
    http::{Request, StatusCode, header},
};
use openasr_core::testing::{TinyGgufFixtureSpec, write_tiny_gguf_runtime_source};
use tower::ServiceExt;

fn sample_wav_bytes() -> Vec<u8> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/jfk.wav");
    std::fs::read(path).unwrap()
}

fn write_whisper_oasr_v1_fixture(path: &std::path::Path, model_id: &str) {
    let spec = TinyGgufFixtureSpec::whisper_oasr_v1_graph_ready_for_runtime_fail_closed(model_id);
    write_tiny_gguf_runtime_source(path, &spec).expect("write whisper gguf runtime source");
}

fn multipart_diarize(uri: &str, model: &str, bytes: &[u8]) -> Request<Body> {
    let boundary = "openasr-rt380-boundary";
    let mut body = Vec::new();
    body.extend_from_slice(
        format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"sample.wav\"\r\nContent-Type: audio/wav\r\n\r\n"
        )
        .as_bytes(),
    );
    body.extend_from_slice(bytes);
    body.extend_from_slice(
        format!(
            "\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\n{model}\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"diarize\"\r\n\r\ntrue\r\n--{boundary}--\r\n"
        )
        .as_bytes(),
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

async fn post_diarize_with_wespeaker_preference(uri: &str) -> (StatusCode, String) {
    let temp = tempfile::tempdir().unwrap();
    unsafe { std::env::remove_var("OPENASR_REDIMNET_PACK") };
    unsafe { std::env::remove_var("OPENASR_WESPEAKER_PACK") };
    unsafe { std::env::set_var("OPENASR_HOME", temp.path()) };
    std::fs::write(
        temp.path().join("config.json"),
        r#"{"preferences":{"voice_id_embedder":"wespeaker"}}"#,
    )
    .unwrap();
    let pack_root = temp.path().join("whisper-runtime.oasr");
    write_whisper_oasr_v1_fixture(&pack_root, "whisper-runtime");
    let app = openasr_server::app_with_runtime(openasr_server::ServerRuntime {
        backend: openasr_core::BackendKind::Native,
        native_execution: openasr_server::NativeExecutionSupervisor::default(),
        ffmpeg_bin: None,
        ffmpeg_bin_explicit: false,
        model_pack_path: Some(pack_root).into(),
    });
    let response = app
        .oneshot(multipart_diarize(
            uri,
            "whisper-runtime",
            &sample_wav_bytes(),
        ))
        .await
        .unwrap();
    let status = response.status();
    let bytes = to_bytes(response.into_body(), 1024 * 256).await.unwrap();
    (status, String::from_utf8_lossy(&bytes).into_owned())
}

fn assert_wespeaker_fail_closed(status: StatusCode, body: &str, surface: &str) {
    assert!(
        status.is_client_error(),
        "{surface}: selected WeSpeaker missing must fail closed with 4xx, got {status}; body={body}"
    );
    assert!(
        body.contains("wespeaker-voxceleb-resnet34-lm") || body.contains("WeSpeaker ResNet"),
        "{surface}: error must name the selected WeSpeaker pack, got {body}"
    );
    assert!(
        !body.contains("redimnet2-b6-cn") || body.contains("wespeaker-voxceleb-resnet34-lm"),
        "{surface}: ReDimNet-only copy must not replace the selected WeSpeaker reason: {body}"
    );
}

/// File JSON transcription with a persisted WeSpeaker preference and no
/// WeSpeaker pack must 400 naming that pack, not the ReDimNet capability probe.
#[tokio::test]
#[ignore = "redteam: pr-380"]
async fn rt_380_http_json_transcription_wespeaker_missing_fail_closed() {
    let (status, body) = post_diarize_with_wespeaker_preference("/v1/audio/transcriptions").await;
    assert_wespeaker_fail_closed(status, &body, "POST /v1/audio/transcriptions");
}

/// `?stream=true` is still a file job and must honor the same preference.
/// Predicted miss: HTTP 200 SSE plus ReDimNet copy, because stream skips
/// apply_transcription_preferences and maps Backend errors into SSE.
#[tokio::test]
#[ignore = "redteam: pr-380"]
async fn rt_380_http_stream_transcription_wespeaker_missing_fail_closed() {
    let (status, body) =
        post_diarize_with_wespeaker_preference("/v1/audio/transcriptions?stream=true").await;
    assert_wespeaker_fail_closed(status, &body, "POST /v1/audio/transcriptions?stream=true");
}

/// Translations alias shares the file Voice ID contract.
#[tokio::test]
#[ignore = "redteam: pr-380"]
async fn rt_380_http_translations_wespeaker_missing_fail_closed() {
    let (status, body) = post_diarize_with_wespeaker_preference("/v1/audio/translations").await;
    assert_wespeaker_fail_closed(status, &body, "POST /v1/audio/translations");
}
