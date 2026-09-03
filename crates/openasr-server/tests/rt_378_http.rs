//! Red-team coverage for PR #378 HTTP / wire / exporter promises that currently
//! fail. Ignored so default `cargo nextest` stays green.

use axum::{
    body::{Body, to_bytes},
    http::{Request, header},
};
use openasr_core::{ResponseFormat, Segment, Transcription, WordTimestamp, render_transcription};
use std::{
    fs,
    path::{Path, PathBuf},
};
use tower::ServiceExt;

const TMP_ROOT: &str = "/Volumes/QuintinDocument/tmp";
const JFK_TRANSCRIPT: &str = "And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country.";

fn isolated_home() -> tempfile::TempDir {
    fs::create_dir_all(TMP_ROOT).expect("create large-volume tmp root");
    tempfile::Builder::new()
        .prefix("oasr-rt378.")
        .tempdir_in(TMP_ROOT)
        .expect("create isolated OPENASR_HOME")
}

fn isolate_process_home(home: &Path) {
    unsafe {
        std::env::set_var("OPENASR_HOME", home);
        std::env::remove_var("OPENASR_FORCED_ALIGNER_PACK");
        std::env::remove_var("OPENASR_MODELS_DIR");
    }
}

fn jfk_wav() -> Vec<u8> {
    fs::read(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/jfk.wav"))
        .expect("read fixtures/jfk.wav")
}

fn loopback_app(home: &Path) -> axum::Router {
    openasr_server::app_with_runtime_and_distribution(
        openasr_server::ServerRuntime::default(),
        openasr_server::DistributionRuntime {
            openasr_home: Some(home.to_path_buf()),
            catalog_url: None,
            catalog_local_override: None,
        },
    )
}

enum MultipartPart<'a> {
    File {
        filename: &'a str,
        content_type: &'a str,
        bytes: &'a [u8],
    },
    Text(&'a str),
}

fn multipart_body(fields: &[(&str, MultipartPart<'_>)]) -> (String, Vec<u8>) {
    let boundary = "rt378httpboundary";
    let mut body = Vec::new();
    for (name, part) in fields {
        match part {
            MultipartPart::File {
                filename,
                content_type,
                bytes,
            } => {
                body.extend_from_slice(
                    format!(
                        "--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"; filename=\"{filename}\"\r\nContent-Type: {content_type}\r\n\r\n"
                    )
                    .as_bytes(),
                );
                body.extend_from_slice(bytes);
                body.extend_from_slice(b"\r\n");
            }
            MultipartPart::Text(value) => {
                body.extend_from_slice(
                    format!(
                        "--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n"
                    )
                    .as_bytes(),
                );
            }
        }
    }
    body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
    (format!("multipart/form-data; boundary={boundary}"), body)
}

async fn post_multipart(
    app: axum::Router,
    uri: &str,
    content_type: &str,
    body: Vec<u8>,
) -> (axum::http::StatusCode, String, String) {
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(uri)
                .header(header::CONTENT_TYPE, content_type)
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let content_type_header = response
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or("")
        .to_string();
    let bytes = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
    (
        status,
        content_type_header,
        String::from_utf8_lossy(&bytes).into_owned(),
    )
}

fn looks_like_aligned_timeline(body: &str) -> bool {
    if body.contains(" --> ") {
        return true;
    }
    let Ok(value) = serde_json::from_str::<serde_json::Value>(body) else {
        return false;
    };
    value
        .get("words")
        .and_then(serde_json::Value::as_array)
        .is_some_and(|words| !words.is_empty())
}

#[tokio::test]
async fn rt_378_precise_timeline_stream_query_is_not_transcription_sse() {
    let home = isolated_home();
    isolate_process_home(home.path());
    let app = loopback_app(home.path());
    let wav = jfk_wav();
    let (content_type, body) = multipart_body(&[
        (
            "file",
            MultipartPart::File {
                filename: "jfk.wav",
                content_type: "audio/wav",
                bytes: &wav,
            },
        ),
        ("transcript", MultipartPart::Text(JFK_TRANSCRIPT)),
        ("language", MultipartPart::Text("en")),
        ("response_format", MultipartPart::Text("verbose_json")),
    ]);
    let (status, content_type_header, body_text) = post_multipart(
        app,
        "/v1/audio/precise-timeline?stream=true",
        &content_type,
        body,
    )
    .await;
    assert!(
        !content_type_header.contains("text/event-stream"),
        "precise-timeline?stream=true must not become transcription SSE (content-type {content_type_header})\n{body_text}"
    );
    assert_eq!(
        status,
        axum::http::StatusCode::BAD_REQUEST,
        "stream=true on precise-timeline must fail closed with 400 mentioning stream, got {status}\ncontent-type={content_type_header}\n{body_text}"
    );
    assert!(
        body_text.to_ascii_lowercase().contains("stream"),
        "400 body must mention stream rather than silently running alignment\n{body_text}"
    );
    assert!(
        !looks_like_aligned_timeline(&body_text),
        "stream=true must not return an aligned timeline\n{body_text}"
    );
}

#[test]
fn rt_378_http_wire_includes_precise_timeline_types() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("generated/http-wire");
    let mut names = Vec::new();
    let mut contents = String::new();
    for entry in fs::read_dir(&dir)
        .unwrap_or_else(|error| panic!("generated/http-wire missing at {}: {error}", dir.display()))
    {
        let entry = entry.unwrap();
        let name = entry.file_name().to_string_lossy().into_owned();
        if let Ok(text) = fs::read_to_string(entry.path()) {
            contents.push_str(&text);
            contents.push('\n');
        }
        names.push(name);
    }
    let name_haystack = names.join("\n").to_ascii_lowercase();
    let content_haystack = contents.to_ascii_lowercase();
    let has_endpoint_type = names.iter().any(|name| {
        let lower = name.to_ascii_lowercase();
        (lower.contains("precise") && lower.contains("timeline"))
            || lower == "transcription.ts"
            || lower.contains("verbosetranscription")
            || lower.contains("jsontranscription")
    }) || name_haystack.contains("precise-timeline")
        || content_haystack.contains("precise-timeline")
        || content_haystack.contains("precisetimeline");
    assert!(
        has_endpoint_type,
        "CI http-wire golden export set does not cover precise-timeline request/response/Transcription types; committed files: {names:?}"
    );
}

#[test]
fn rt_378_shared_exporter_rejects_zero_length_and_overlap_srt() {
    let transcription = Transcription {
        text: "hello world".into(),
        segments: vec![
            Segment {
                start: 1.0,
                end: 1.0,
                text: "hello".into(),
                speaker: None,
                speaker_label: None,
                speaker_person_id: None,
                speaker_snapshot_label: None,
                words: vec![WordTimestamp {
                    word: "hello".into(),
                    start: 1.0,
                    end: 1.0,
                    confidence: None,
                }],
            },
            Segment {
                start: 0.5,
                end: 2.0,
                text: "world".into(),
                speaker: None,
                speaker_label: None,
                speaker_person_id: None,
                speaker_snapshot_label: None,
                words: vec![WordTimestamp {
                    word: "world".into(),
                    start: 0.5,
                    end: 2.0,
                    confidence: None,
                }],
            },
        ],
        subtitle_cues: vec![
            Segment {
                start: 1.0,
                end: 1.0,
                text: "hello".into(),
                speaker: None,
                speaker_label: None,
                speaker_person_id: None,
                speaker_snapshot_label: None,
                words: Vec::new(),
            },
            Segment {
                start: 0.5,
                end: 2.0,
                text: "world".into(),
                speaker: None,
                speaker_label: None,
                speaker_person_id: None,
                speaker_snapshot_label: None,
                words: Vec::new(),
            },
        ],
        ..Default::default()
    };
    let srt =
        render_transcription(&transcription, ResponseFormat::Srt).expect("shared SRT renderer");
    let mut previous_end: Option<&str> = None;
    let mut illegal = false;
    for line in srt.lines() {
        let Some((start, end)) = line.split_once(" --> ") else {
            continue;
        };
        let start = start.trim();
        let end = end.trim();
        if start == end {
            illegal = true;
            break;
        }
        if let Some(previous) = previous_end
            && start < previous
        {
            illegal = true;
            break;
        }
        previous_end = Some(end);
    }
    assert!(
        !illegal,
        "shared exporter must not silently write zero-length or overlapping SRT, got:\n{srt}"
    );
}
