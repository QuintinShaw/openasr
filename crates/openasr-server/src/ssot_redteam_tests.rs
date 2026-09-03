//! Red-team HTTP coverage for remote-compute product invariants.
//!
//! The route matrix is a live pairing-mode HTTP sweep of every `.route(` in
//! `lib.rs`. Other tests in this file are SSOT falsifiers: they stay ignored
//! when live code currently violates the contract.

use std::collections::BTreeSet;

use axum::body::{Body, to_bytes};
use axum::http::{Request, header};
use tower::ServiceExt;

use super::*;
use crate::testing::{
    PAIRING_ADMIN_TOKEN, approve_loopback_pairing, bearer_auth_header, https_request,
    https_request_status, spawn_loopback_pairing_server,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct RouteExpect {
    method: &'static str,
    path: &'static str,
    none: u16,
    device: u16,
    operator: u16,
}

/// Complete pairing-mode permission table. Paths are the `lib.rs` templates;
/// status codes are the product contract for {no token, device, operator}.
/// Filled from a live pairing-mode sweep; device 200 on an operator-only
/// mutation is a contract failure, not a table bug.
const EXPECTED_ROUTE_MATRIX: &[RouteExpect] = &[
    RouteExpect {
        method: "DELETE",
        path: "/v1/history/{id}",
        none: 401,
        device: 403,
        operator: 404,
    },
    RouteExpect {
        method: "DELETE",
        path: "/v1/models/{id}",
        none: 401,
        device: 403,
        operator: 200,
    },
    RouteExpect {
        method: "DELETE",
        path: "/v1/pairing/credentials/{device_id}",
        none: 401,
        device: 401,
        operator: 404,
    },
    RouteExpect {
        method: "DELETE",
        path: "/v1/pairing/requests/{request_id}",
        none: 401,
        device: 401,
        operator: 404,
    },
    RouteExpect {
        method: "DELETE",
        path: "/v1/runtime/runs",
        none: 401,
        device: 403,
        operator: 204,
    },
    RouteExpect {
        method: "DELETE",
        path: "/v1/voice-id/persons/{person_id}",
        none: 401,
        device: 403,
        operator: 404,
    },
    RouteExpect {
        method: "DELETE",
        path: "/v1/voice-id/samples/{sample_id}",
        none: 401,
        device: 403,
        operator: 404,
    },
    RouteExpect {
        method: "GET",
        path: "/health",
        none: 200,
        device: 200,
        operator: 200,
    },
    RouteExpect {
        method: "GET",
        path: "/v1/audio/realtime",
        none: 401,
        device: 400,
        operator: 400,
    },
    RouteExpect {
        method: "GET",
        path: "/v1/audio/transcriptions/progress",
        none: 401,
        device: 200,
        operator: 200,
    },
    RouteExpect {
        method: "GET",
        path: "/v1/audio/transcriptions/{id}/progress",
        none: 401,
        device: 200,
        operator: 200,
    },
    RouteExpect {
        method: "GET",
        path: "/v1/capabilities",
        none: 401,
        device: 200,
        operator: 200,
    },
    RouteExpect {
        method: "GET",
        path: "/v1/capabilities/requests",
        none: 401,
        device: 403,
        operator: 200,
    },
    RouteExpect {
        method: "GET",
        path: "/v1/catalog",
        none: 401,
        device: 400,
        operator: 400,
    },
    RouteExpect {
        method: "GET",
        path: "/v1/config",
        none: 401,
        device: 403,
        operator: 200,
    },
    RouteExpect {
        method: "GET",
        path: "/v1/debug/runtime-receipts",
        none: 401,
        device: 403,
        operator: 200,
    },
    RouteExpect {
        method: "GET",
        path: "/v1/devices",
        none: 401,
        device: 200,
        operator: 200,
    },
    RouteExpect {
        method: "GET",
        path: "/v1/history",
        none: 401,
        device: 403,
        operator: 200,
    },
    RouteExpect {
        method: "GET",
        path: "/v1/history/{id}",
        none: 401,
        device: 403,
        operator: 404,
    },
    RouteExpect {
        method: "GET",
        path: "/v1/models",
        none: 401,
        device: 200,
        operator: 200,
    },
    RouteExpect {
        method: "GET",
        path: "/v1/models/default",
        none: 401,
        device: 200,
        operator: 200,
    },
    RouteExpect {
        method: "GET",
        path: "/v1/models/local",
        none: 401,
        device: 403,
        operator: 200,
    },
    RouteExpect {
        method: "GET",
        path: "/v1/models/pull/{job_id}",
        none: 401,
        device: 404,
        operator: 404,
    },
    RouteExpect {
        method: "GET",
        path: "/v1/models/pull/{job_id}/events",
        none: 401,
        device: 404,
        operator: 404,
    },
    RouteExpect {
        method: "GET",
        path: "/v1/models/pulls",
        none: 401,
        device: 403,
        operator: 200,
    },
    RouteExpect {
        method: "GET",
        path: "/v1/pairing/credentials",
        none: 401,
        device: 401,
        operator: 200,
    },
    RouteExpect {
        method: "GET",
        path: "/v1/pairing/requests",
        none: 401,
        device: 401,
        operator: 200,
    },
    RouteExpect {
        method: "GET",
        path: "/v1/pairing/requests/{request_id}/credential",
        none: 404,
        device: 404,
        operator: 404,
    },
    RouteExpect {
        method: "GET",
        path: "/v1/runtime/receipts",
        none: 401,
        device: 403,
        operator: 200,
    },
    RouteExpect {
        method: "GET",
        path: "/v1/runtime/runs",
        none: 401,
        device: 403,
        operator: 200,
    },
    RouteExpect {
        method: "GET",
        path: "/v1/voice-id/persons",
        none: 401,
        device: 403,
        operator: 200,
    },
    RouteExpect {
        method: "GET",
        path: "/v1/voice-id/persons/{person_id}",
        none: 401,
        device: 403,
        operator: 404,
    },
    RouteExpect {
        method: "PATCH",
        path: "/v1/voice-id/persons/{person_id}",
        none: 401,
        device: 403,
        operator: 400,
    },
    RouteExpect {
        method: "PATCH",
        path: "/v1/voice-id/samples/{sample_id}",
        none: 401,
        device: 403,
        operator: 400,
    },
    RouteExpect {
        method: "POST",
        path: "/v1/audio/precise-timeline",
        none: 401,
        device: 400,
        operator: 400,
    },
    RouteExpect {
        method: "POST",
        path: "/v1/audio/transcriptions",
        none: 401,
        device: 200,
        operator: 200,
    },
    RouteExpect {
        method: "POST",
        path: "/v1/audio/transcriptions/{id}/cancel",
        none: 401,
        device: 404,
        operator: 404,
    },
    RouteExpect {
        method: "POST",
        path: "/v1/audio/transcriptions/{id}/pause",
        none: 401,
        device: 404,
        operator: 404,
    },
    RouteExpect {
        method: "POST",
        path: "/v1/audio/transcriptions/{id}/resume",
        none: 401,
        device: 404,
        operator: 404,
    },
    RouteExpect {
        method: "POST",
        path: "/v1/audio/translations",
        none: 401,
        device: 200,
        operator: 200,
    },
    RouteExpect {
        method: "POST",
        path: "/v1/capabilities/requests",
        none: 401,
        device: 202,
        operator: 202,
    },
    RouteExpect {
        method: "POST",
        path: "/v1/capabilities/requests/approve",
        none: 401,
        device: 403,
        operator: 400,
    },
    RouteExpect {
        method: "POST",
        path: "/v1/history/{id}/transcript",
        none: 401,
        device: 403,
        operator: 422,
    },
    RouteExpect {
        method: "POST",
        path: "/v1/models/default",
        none: 401,
        device: 403,
        operator: 400,
    },
    RouteExpect {
        method: "POST",
        path: "/v1/models/default/idle-switch/cancel",
        none: 401,
        device: 403,
        operator: 200,
    },
    RouteExpect {
        method: "POST",
        path: "/v1/models/local/import",
        none: 401,
        device: 403,
        operator: 415,
    },
    RouteExpect {
        method: "POST",
        path: "/v1/models/pull/{job_id}/cancel",
        none: 401,
        device: 403,
        operator: 404,
    },
    RouteExpect {
        method: "POST",
        path: "/v1/models/pull/{job_id}/pause",
        none: 401,
        device: 403,
        operator: 404,
    },
    RouteExpect {
        method: "POST",
        path: "/v1/models/pull/{job_id}/resume",
        none: 401,
        device: 403,
        operator: 404,
    },
    RouteExpect {
        method: "POST",
        path: "/v1/models/{id}/pull",
        none: 401,
        device: 403,
        operator: 400,
    },
    RouteExpect {
        method: "POST",
        path: "/v1/pairing/requests",
        none: 202,
        device: 202,
        operator: 202,
    },
    RouteExpect {
        method: "POST",
        path: "/v1/pairing/requests/{request_id}/approve",
        none: 401,
        device: 401,
        operator: 404,
    },
    RouteExpect {
        method: "POST",
        path: "/v1/voice-id/export",
        none: 401,
        device: 403,
        operator: 200,
    },
    RouteExpect {
        method: "POST",
        path: "/v1/voice-id/persons",
        none: 401,
        device: 403,
        operator: 400,
    },
    RouteExpect {
        method: "POST",
        path: "/v1/voice-id/persons/from-audio",
        none: 401,
        device: 403,
        operator: 400,
    },
    RouteExpect {
        method: "POST",
        path: "/v1/voice-id/persons/{person_id}/consent/revoke",
        none: 401,
        device: 403,
        operator: 404,
    },
    RouteExpect {
        method: "POST",
        path: "/v1/voice-id/persons/{person_id}/samples",
        none: 401,
        device: 403,
        operator: 400,
    },
    RouteExpect {
        method: "POST",
        path: "/v1/voice-id/persons/{person_id}/samples/from-audio",
        none: 401,
        device: 403,
        operator: 400,
    },
    RouteExpect {
        method: "PUT",
        path: "/v1/config",
        none: 401,
        device: 403,
        operator: 200,
    },
    RouteExpect {
        method: "PUT",
        path: "/v1/history/{id}/speaker-assignments",
        none: 401,
        device: 403,
        operator: 422,
    },
    RouteExpect {
        method: "PUT",
        path: "/v1/models/default",
        none: 401,
        device: 403,
        operator: 400,
    },
];

fn extract_registered_routes(src: &str) -> BTreeSet<(String, String)> {
    let start = src
        .find("Router::new()")
        .expect("lib.rs must contain Router::new()");
    let rest = &src[start..];
    let end = rest
        .find(".layer(middleware::from_fn_with_state")
        .expect("router must be followed by the auth layer");
    let block = &rest[..end];
    let mut routes = BTreeSet::new();
    let mut remaining = block;
    while let Some(idx) = remaining.find(".route(") {
        let after_open = &remaining[idx + ".route(".len()..];
        let close = matching_paren_end(after_open);
        let args = &after_open[..close];
        let (path, methods_src) = split_route_args(args);
        for method in http_methods_in(methods_src) {
            routes.insert((method.to_string(), path.clone()));
        }
        remaining = &after_open[close + 1..];
    }
    routes
}

fn matching_paren_end(src: &str) -> usize {
    let mut depth = 1;
    let mut in_string = false;
    let mut escaped = false;
    for (i, ch) in src.char_indices() {
        if in_string {
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_string = false;
            }
            continue;
        }
        match ch {
            '"' => in_string = true,
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    return i;
                }
            }
            _ => {}
        }
    }
    panic!("unterminated .route( in lib.rs");
}

fn split_route_args(args: &str) -> (String, &str) {
    let trimmed = args.trim_start();
    assert!(
        trimmed.starts_with('"'),
        "route path must be a string literal, got {trimmed}"
    );
    let inner = &trimmed[1..];
    let end = inner.find('"').expect("unterminated route path");
    let path = inner[..end].to_string();
    (path, &inner[end + 1..])
}

fn http_methods_in(src: &str) -> Vec<&'static str> {
    let mut methods = Vec::new();
    for (i, ch) in src.char_indices() {
        if ch != '(' {
            continue;
        }
        let ident = ident_before(src, i);
        let method = match ident {
            "get" => "GET",
            "post" => "POST",
            "put" => "PUT",
            "delete" => "DELETE",
            "patch" => "PATCH",
            // Websocket upgrade is GET; `any()` is registered for that path.
            "any" => "GET",
            _ => continue,
        };
        methods.push(method);
    }
    assert!(
        !methods.is_empty(),
        "route methods must include get/post/put/delete/patch/any: {src}"
    );
    methods
}

fn ident_before(src: &str, paren_idx: usize) -> &str {
    let before = &src[..paren_idx];
    let start = before
        .rfind(|c: char| !c.is_ascii_alphanumeric() && c != '_')
        .map(|i| i + 1)
        .unwrap_or(0);
    &before[start..]
}

fn instantiate_path(template: &str) -> String {
    template
        .replace("{person_id}", "person_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
        .replace("{sample_id}", "sample_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
        .replace("{request_id}", "00000000000000000000000000000000")
        .replace("{device_id}", "000000000000000000000000")
        .replace("{job_id}", "missing-job")
        .replace("{id}", "missing-id")
}

fn request_payload(method: &str, path: &str) -> (Vec<(&'static str, String)>, Vec<u8>) {
    if matches!(method, "GET" | "DELETE") {
        return (Vec::new(), Vec::new());
    }
    let multipart = path.contains("/audio/transcriptions")
        && !path.contains("/progress")
        && !path.contains("/cancel")
        && !path.contains("/pause")
        && !path.contains("/resume")
        || path.contains("/audio/precise-timeline")
        || path.contains("/audio/translations")
        || path.contains("/from-audio")
        || path.ends_with("/samples")
        || path.ends_with("/import");
    if multipart {
        let boundary = "openasr-matrix-boundary";
        let body = format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"sample.wav\"\r\nContent-Type: audio/wav\r\n\r\nnot a real wav\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nwhisper-large-v3-turbo\r\n--{boundary}--\r\n"
        );
        return (
            vec![(
                "Content-Type",
                format!("multipart/form-data; boundary={boundary}"),
            )],
            body.into_bytes(),
        );
    }
    let json = if path == "/v1/pairing/requests" && method == "POST" {
        br#"{"device_name":"MatrixProbe"}"#.to_vec()
    } else if path == "/v1/capabilities/requests" && method == "POST" {
        br#"{"features":["speakers"]}"#.to_vec()
    } else if path == "/v1/models/default" {
        br#"{"pull":"whisper-tiny:q4"}"#.to_vec()
    } else {
        b"{}".to_vec()
    };
    (vec![("Content-Type", "application/json".to_string())], json)
}

fn dump_matrix(rows: &[(String, String, u16, u16, u16)]) -> String {
    let mut dump = String::from("const EXPECTED_ROUTE_MATRIX: &[RouteExpect] = &[\n");
    for (method, path, none, device, operator) in rows {
        dump.push_str(&format!(
            "    RouteExpect {{ method: \"{method}\", path: \"{path}\", none: {none}, device: {device}, operator: {operator} }},\n"
        ));
    }
    dump.push_str("];\n");
    dump
}

#[tokio::test(flavor = "multi_thread")]
async fn ssot_route_permission_matrix_covers_every_registered_route() {
    let extracted = extract_registered_routes(include_str!("lib.rs"));
    assert!(
        !extracted.is_empty(),
        "lib.rs router must register at least one .route("
    );

    let temp = tempfile::tempdir().unwrap();
    let server = spawn_loopback_pairing_server(temp.path()).await;
    let credential = approve_loopback_pairing(&server).await;
    let device_auth = bearer_auth_header(&credential.bearer_token);
    let operator_auth = bearer_auth_header(PAIRING_ADMIN_TOKEN);

    let mut live = Vec::new();
    for (method, path) in &extracted {
        let uri = instantiate_path(path);
        let (extra_headers, body) = request_payload(method, path);
        let none = hit(&server, method, &uri, &extra_headers, None, body.clone()).await;
        let device = hit(
            &server,
            method,
            &uri,
            &extra_headers,
            Some(device_auth.as_str()),
            body.clone(),
        )
        .await;
        let operator = hit(
            &server,
            method,
            &uri,
            &extra_headers,
            Some(operator_auth.as_str()),
            body,
        )
        .await;
        live.push((method.clone(), path.clone(), none, device, operator));
    }

    let expected_keys: BTreeSet<(String, String)> = EXPECTED_ROUTE_MATRIX
        .iter()
        .map(|row| (row.method.to_string(), row.path.to_string()))
        .collect();
    if expected_keys != extracted || EXPECTED_ROUTE_MATRIX.is_empty() {
        panic!(
            "route matrix set mismatch; paste this table into EXPECTED_ROUTE_MATRIX:\n{}",
            dump_matrix(&live)
        );
    }

    let mut mismatches = Vec::new();
    for (method, path, none, device, operator) in &live {
        let expect = EXPECTED_ROUTE_MATRIX
            .iter()
            .find(|row| row.method == method && row.path == path)
            .expect("expected table must cover extracted route");
        if (expect.none, expect.device, expect.operator) != (*none, *device, *operator) {
            mismatches.push(format!(
                "{method} {path}: live none={none} device={device} operator={operator}, expected none={} device={} operator={}",
                expect.none, expect.device, expect.operator
            ));
        }
        // Product contract: unauthenticated callers never receive 200 on
        // authenticated routes. /health and unauthenticated pairing create /
        // credential poll are the documented exceptions.
        let unauthenticated_ok = *path == "/health"
            || (*method == "POST" && *path == "/v1/pairing/requests")
            || (*method == "GET" && path.ends_with("/credential"));
        if !unauthenticated_ok && *none == 200 {
            mismatches.push(format!(
                "{method} {path}: unauthenticated caller received 200"
            ));
        }
        if is_device_forbidden_path(method, path) && *device != 403 {
            mismatches.push(format!(
                "{method} {path}: device token must be 403, got {device}"
            ));
        }
        if is_operator_manage_path(method, path) && *operator == 403 {
            mismatches.push(format!(
                "{method} {path}: operator token must be allowed to manage, got 403"
            ));
        }
    }
    if !mismatches.is_empty() {
        panic!(
            "route permission matrix failed:\n{}\n\nlive table:\n{}",
            mismatches.join("\n"),
            dump_matrix(&live)
        );
    }
}

fn is_device_forbidden_path(method: &str, path: &str) -> bool {
    if path == "/v1/models/default" {
        return method != "GET";
    }
    if path == "/v1/models/local"
        || path == "/v1/models/local/import"
        || path == "/v1/models/default/idle-switch/cancel"
        || (path.starts_with("/v1/models/") && path.ends_with("/pull") && method == "POST")
        || (path.starts_with("/v1/models/") && method == "DELETE" && path != "/v1/models/local")
        || path.starts_with("/v1/voice-id/")
        || path == "/v1/voice-id"
    {
        return true;
    }
    false
}

fn is_operator_manage_path(method: &str, path: &str) -> bool {
    is_device_forbidden_path(method, path)
        || path == "/v1/runtime/runs"
        || path == "/v1/models/pulls"
        || path == "/v1/capabilities/requests/approve"
        || (path == "/v1/capabilities/requests" && method == "GET")
        || path.starts_with("/v1/models/pull/") && method == "POST"
}

async fn hit(
    server: &crate::testing::LoopbackTlsServer,
    method: &str,
    path: &str,
    extra_headers: &[(&'static str, String)],
    bearer: Option<&str>,
    body: Vec<u8>,
) -> u16 {
    let mut headers: Vec<(&str, &str)> = Vec::new();
    let extra_owned: Vec<(&str, String)> = extra_headers.to_vec();
    let extra_refs: Vec<(&str, &str)> = extra_owned
        .iter()
        .map(|(name, value)| (*name, value.as_str()))
        .collect();
    headers.extend(extra_refs);
    if let Some(token) = bearer {
        headers.push(("Authorization", token));
    }
    let header_slice: Vec<(&str, &str)> = headers.clone();
    https_request_status(server.addr, method, path, &header_slice, body).await
}

fn policy_app(runtime: ServerRuntime, home: std::path::PathBuf) -> axum::Router {
    app_with_runtime_and_distribution(
        runtime,
        DistributionRuntime {
            openasr_home: Some(home),
            catalog_url: None,
            catalog_local_override: None,
        },
    )
}

fn sample_multipart(transcription_id: Option<&str>, stream: bool) -> (String, Vec<u8>) {
    let boundary = "openasr-redteam-boundary";
    let mut body = format!(
        "--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"sample.wav\"\r\nContent-Type: audio/wav\r\n\r\nnot a real wav\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nwhisper-large-v3-turbo\r\n"
    );
    if let Some(id) = transcription_id {
        body.push_str(&format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"transcription_id\"\r\n\r\n{id}\r\n"
        ));
    }
    body.push_str(&format!("--{boundary}--\r\n"));
    let uri_suffix = if stream { "?stream=true" } else { "" };
    let _ = uri_suffix;
    (
        format!("multipart/form-data; boundary={boundary}"),
        body.into_bytes(),
    )
}

async fn post_transcription(
    app: axum::Router,
    uri: &str,
    transcription_id: Option<&str>,
) -> axum::http::Response<Body> {
    let (content_type, body) = sample_multipart(transcription_id, uri.contains("stream=true"));
    app.oneshot(
        Request::builder()
            .method("POST")
            .uri(uri)
            .header(header::CONTENT_TYPE, content_type)
            .body(Body::from(body))
            .unwrap(),
    )
    .await
    .unwrap()
}

/// SSOT 13: `POST /v1/audio/transcriptions?stream=true` is still a new file
/// task. A pending idle switch must reject it, including operator-local calls.
///
/// If correct: 409 + PENDING_IDLE_SWITCH_MESSAGE. Otherwise Y: 200 (stream
/// path bypasses wait_for_file_admission on the mock backend).
#[tokio::test]
async fn ssot_13_pending_idle_switch_rejects_stream_file_jobs() {
    let temp = tempfile::tempdir().unwrap();
    let home = temp.path().join("home");
    std::fs::create_dir_all(&home).unwrap();
    let runtime = ServerRuntime::default();
    runtime
        .native_execution
        .remote_policy()
        .request_idle_switch("whisper-base:q4");
    let app = policy_app(runtime, home);
    let response = post_transcription(
        app,
        "/v1/audio/transcriptions?stream=true",
        Some("blocked-stream"),
    )
    .await;
    assert_eq!(response.status(), StatusCode::CONFLICT);
    let body = to_bytes(response.into_body(), 1024 * 64).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(
        json["error"]["message"].as_str(),
        Some(PENDING_IDLE_SWITCH_MESSAGE)
    );
}

/// SSOT 10: a busy server must queue cancelable file jobs. `?stream=true`
/// must not jump the FIFO.
///
/// If correct: queued (or 429 without starting a second native slot).
/// Otherwise Y: 200 while the native slot is still held.
#[tokio::test]
async fn ssot_10_stream_file_jobs_do_not_bypass_fifo() {
    let temp = tempfile::tempdir().unwrap();
    let home = temp.path().join("home");
    std::fs::create_dir_all(&home).unwrap();
    let runtime = ServerRuntime::default();
    let _permit = runtime
        .native_execution
        .try_acquire("hold-native-slot")
        .unwrap();
    let app = policy_app(runtime.clone(), home);
    let response = post_transcription(
        app,
        "/v1/audio/transcriptions?stream=true",
        Some("stream-queued"),
    )
    .await;
    assert_ne!(
        response.status(),
        StatusCode::OK,
        "stream=true must not run while the native slot is occupied: {:?}",
        response.status()
    );
    assert!(
        runtime
            .native_execution
            .remote_policy()
            .is_file_queued("stream-queued")
            || response.status() == StatusCode::TOO_MANY_REQUESTS
            || response.status() == StatusCode::CONFLICT,
        "busy stream file job must enter the FIFO or fail closed busy, got {}",
        response.status()
    );
}

/// SSOT 22: a client must be able to cancel a file job. The SSE stream path
/// currently uses an uncancellable execution context.
///
/// If correct: cancel returns 202 and the stream fails closed canceled.
/// Otherwise Y: cancel 404 / stream completes with status ok.
#[tokio::test]
async fn ssot_22_stream_file_jobs_are_cancellable() {
    let temp = tempfile::tempdir().unwrap();
    let home = temp.path().join("home");
    std::fs::create_dir_all(&home).unwrap();
    let _delay =
        openasr_core::testing::MockTranscribeDelayGuard::new(std::time::Duration::from_secs(2));
    let runtime = ServerRuntime::default();
    let app = policy_app(runtime, home);
    let (content_type, body) = sample_multipart(Some("stream-cancel"), true);
    let streamed = tokio::spawn({
        let app = app.clone();
        async move {
            app.oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/audio/transcriptions?stream=true")
                    .header(header::CONTENT_TYPE, content_type)
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap()
        }
    });
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    let cancel = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/audio/transcriptions/stream-cancel/cancel")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(cancel.status(), StatusCode::ACCEPTED);
    let streamed = streamed.await.unwrap();
    assert_ne!(
        streamed.status(),
        StatusCode::OK,
        "canceled stream jobs must not complete as success"
    );
}

/// SSOT 22: aborting a queued HTTP file job must release the FIFO so a later
/// job can run. If correct: abort clears the queued id and a new job is 200.
/// Otherwise Y: later jobs stay 409/429 or the queued id remains.
#[tokio::test(flavor = "multi_thread")]
async fn ssot_22_aborting_a_queued_file_job_releases_the_fifo() {
    let temp = tempfile::tempdir().unwrap();
    let home = temp.path().join("home");
    std::fs::create_dir_all(&home).unwrap();
    let runtime = ServerRuntime::default();
    let permit = runtime
        .native_execution
        .try_acquire("hold-native-slot")
        .unwrap();
    let app = policy_app(runtime.clone(), home);
    let (content_type, body) = sample_multipart(Some("file-aborted"), false);
    let queued = tokio::spawn({
        let app = app.clone();
        async move {
            app.oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/audio/transcriptions")
                    .header(header::CONTENT_TYPE, content_type)
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap()
        }
    });
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
    while !runtime
        .native_execution
        .remote_policy()
        .is_file_queued("file-aborted")
    {
        assert!(
            std::time::Instant::now() < deadline,
            "busy file job must enter the cancelable FIFO"
        );
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    queued.abort();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
    while runtime
        .native_execution
        .remote_policy()
        .is_file_queued("file-aborted")
        || runtime
            .native_execution
            .remote_policy()
            .file_running()
            .as_deref()
            == Some("file-aborted")
    {
        assert!(
            std::time::Instant::now() < deadline,
            "aborting the HTTP request must cancel_file the queued id"
        );
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    drop(permit);
    let admitted = post_transcription(app, "/v1/audio/transcriptions", Some("after-abort")).await;
    assert_eq!(admitted.status(), StatusCode::OK);
}

/// SSOT 22: aborting a running mock file job must call finish_file so the
/// server is not stuck busy. Mock transcribe delay is a blocking sleep, so a
/// dropped HTTP future cannot cancel it until the sleep ends.
///
/// If correct: a follow-up job is 200 within 200ms of abort.
/// Otherwise Y: follow-up stays 429/409 until the blocking decode finishes.
#[tokio::test(flavor = "multi_thread")]
async fn ssot_22_aborting_a_running_file_job_calls_finish_file() {
    let temp = tempfile::tempdir().unwrap();
    let home = temp.path().join("home");
    std::fs::create_dir_all(&home).unwrap();
    let _delay =
        openasr_core::testing::MockTranscribeDelayGuard::new(std::time::Duration::from_secs(2));
    let runtime = ServerRuntime::default();
    let app = policy_app(runtime.clone(), home);
    let (content_type, body) = sample_multipart(Some("file-running-abort"), false);
    let running = tokio::spawn({
        let app = app.clone();
        async move {
            app.oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/audio/transcriptions")
                    .header(header::CONTENT_TYPE, content_type)
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap()
        }
    });
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
    while runtime
        .native_execution
        .remote_policy()
        .file_running()
        .as_deref()
        != Some("file-running-abort")
    {
        assert!(
            std::time::Instant::now() < deadline,
            "delayed mock job must occupy the running FIFO slot"
        );
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    running.abort();
    let admitted = tokio::time::timeout(
        std::time::Duration::from_millis(300),
        post_transcription(app, "/v1/audio/transcriptions", Some("after-running-abort")),
    )
    .await
    .expect("aborting a running file job must finish_file promptly so the next job is not blocked");
    assert_eq!(
        admitted.status(),
        StatusCode::OK,
        "aborting a running file job must finish_file promptly, got {}",
        admitted.status()
    );
}

/// SSOT 21: a second paired device must not see the first device's name in a
/// busy file response. If correct: 429 + SERVER_BUSY_MESSAGE and the body
/// has no peer device name. Otherwise Y: body contains the other device id.
#[tokio::test(flavor = "multi_thread")]
async fn ssot_21_busy_file_response_omits_peer_device_identity() {
    let temp = tempfile::tempdir().unwrap();
    let server = spawn_loopback_pairing_server(temp.path()).await;
    let first = approve_loopback_pairing(&server).await;
    let create = https_request(
        server.addr,
        "POST",
        "/v1/pairing/requests",
        &[("Content-Type", "application/json")],
        br#"{"device_name":"Peer Phone"}"#.to_vec(),
    )
    .await;
    assert_eq!(create.status, 202);
    let create_json: serde_json::Value = serde_json::from_slice(&create.body).unwrap();
    let request_id = create_json["request_id"].as_str().unwrap();
    crate::testing::approve_pending_pairing_request(&server, request_id).await;
    let second = https_request(
        server.addr,
        "GET",
        &format!("/v1/pairing/requests/{request_id}/credential"),
        &[],
        Vec::new(),
    )
    .await;
    assert_eq!(second.status, 200);
    let second_json: serde_json::Value = serde_json::from_slice(&second.body).unwrap();
    let second_token = second_json["bearer_token"].as_str().unwrap();

    let _delay =
        openasr_core::testing::MockTranscribeDelayGuard::new(std::time::Duration::from_secs(2));
    let boundary = "openasr-busy-peer";
    let body = format!(
        "--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"secret-name.wav\"\r\nContent-Type: audio/wav\r\n\r\nnot a real wav\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nwhisper-large-v3-turbo\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"transcription_id\"\r\n\r\npeer-a\r\n--{boundary}--\r\n"
    );
    let first_auth = bearer_auth_header(&first.bearer_token);
    let content_type = format!("multipart/form-data; boundary={boundary}");
    let first_job = {
        let addr = server.addr;
        let auth = first_auth.clone();
        let content_type = content_type.clone();
        let body = body.into_bytes();
        tokio::spawn(async move {
            https_request(
                addr,
                "POST",
                "/v1/audio/transcriptions",
                &[
                    ("Authorization", auth.as_str()),
                    ("X-OpenASR-Remote-Compute", "client"),
                    ("Content-Type", content_type.as_str()),
                ],
                body,
            )
            .await
        })
    };
    tokio::time::sleep(std::time::Duration::from_millis(80)).await;
    let (content_type, body) = sample_multipart(None, false);
    let second_auth = bearer_auth_header(second_token);
    let busy = https_request(
        server.addr,
        "POST",
        "/v1/audio/transcriptions",
        &[
            ("Authorization", second_auth.as_str()),
            ("X-OpenASR-Remote-Compute", "client"),
            ("Content-Type", content_type.as_str()),
        ],
        body,
    )
    .await;
    let busy_text = String::from_utf8_lossy(&busy.body);
    assert_ne!(busy.status, 200, "second device must not start while busy");
    assert!(
        !busy_text.contains(&first.device_id),
        "busy body must not disclose the peer device id: {busy_text}"
    );
    assert!(
        !busy_text.contains("Loopback Mac"),
        "busy body must not disclose the peer device name: {busy_text}"
    );
    assert!(
        !busy_text.contains("secret-name.wav"),
        "busy body must not disclose the peer file name: {busy_text}"
    );
    first_job.abort();
}

/// SSOT 21 / 22: a paired device must not cancel another device's in-flight
/// file job. If correct: 403. Otherwise Y: 202 because owner lookup collapsed
/// to None.
#[tokio::test(flavor = "multi_thread")]
async fn ssot_21_device_cannot_cancel_another_device_file_job() {
    let temp = tempfile::tempdir().unwrap();
    let server = spawn_loopback_pairing_server(temp.path()).await;
    let first = approve_loopback_pairing(&server).await;
    let create = https_request(
        server.addr,
        "POST",
        "/v1/pairing/requests",
        &[("Content-Type", "application/json")],
        br#"{"device_name":"Peer Phone"}"#.to_vec(),
    )
    .await;
    assert_eq!(create.status, 202);
    let create_json: serde_json::Value = serde_json::from_slice(&create.body).unwrap();
    let request_id = create_json["request_id"].as_str().unwrap();
    crate::testing::approve_pending_pairing_request(&server, request_id).await;
    let second = https_request(
        server.addr,
        "GET",
        &format!("/v1/pairing/requests/{request_id}/credential"),
        &[],
        Vec::new(),
    )
    .await;
    let second_json: serde_json::Value = serde_json::from_slice(&second.body).unwrap();
    let second_token = second_json["bearer_token"].as_str().unwrap().to_string();

    let _delay =
        openasr_core::testing::MockTranscribeDelayGuard::new(std::time::Duration::from_secs(2));
    let (content_type, body) = sample_multipart(Some("owner-job"), false);
    let first_auth = bearer_auth_header(&first.bearer_token);
    let first_job = {
        let addr = server.addr;
        let auth = first_auth.clone();
        let content_type = content_type.clone();
        tokio::spawn(async move {
            https_request(
                addr,
                "POST",
                "/v1/audio/transcriptions",
                &[
                    ("Authorization", auth.as_str()),
                    ("X-OpenASR-Remote-Compute", "client"),
                    ("Content-Type", content_type.as_str()),
                ],
                body,
            )
            .await
        })
    };
    tokio::time::sleep(std::time::Duration::from_millis(150)).await;
    let second_auth = bearer_auth_header(&second_token);
    let cancel = https_request(
        server.addr,
        "POST",
        "/v1/audio/transcriptions/owner-job/cancel",
        &[("Authorization", second_auth.as_str())],
        Vec::new(),
    )
    .await;
    assert_eq!(
        cancel.status,
        403,
        "peer device must not control another device's job: {}",
        String::from_utf8_lossy(&cancel.body)
    );
    first_job.abort();
}

/// SSOT 20: DELETE /v1/runtime/runs must actually empty the operator log.
/// If correct: a follow-up GET returns data=[]. Otherwise Y: records remain
/// because `OperatorRunLog::clear` is a no-op.
#[tokio::test]
async fn ssot_20_clear_operator_runs_empties_the_log() {
    use axum::body::{Body, to_bytes};
    use tower::ServiceExt;

    let temp = tempfile::tempdir().unwrap();
    let home = temp.path().join("home");
    std::fs::create_dir_all(&home).unwrap();
    let app = policy_app(ServerRuntime::default(), home);
    let transcribed = post_transcription(
        app.clone(),
        "/v1/audio/transcriptions",
        Some("run-log-clear"),
    )
    .await;
    assert_eq!(transcribed.status(), StatusCode::OK);
    let cleared = app
        .clone()
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri("/v1/runtime/runs")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(cleared.status(), StatusCode::NO_CONTENT);
    let listed = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/runtime/runs")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(listed.status(), StatusCode::OK);
    let body = to_bytes(listed.into_body(), 1024 * 64).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(
        json["data"].as_array().map(Vec::len),
        Some(0),
        "immediate clear must drop every run record: {json}"
    );
}

/// SSOT 1: pairing mode must keep two device tokens concurrently valid.
#[tokio::test(flavor = "multi_thread")]
async fn ssot_1_pairing_accepts_multiple_device_tokens() {
    let temp = tempfile::tempdir().unwrap();
    let server = spawn_loopback_pairing_server(temp.path()).await;
    let first = approve_loopback_pairing(&server).await;
    let create = https_request(
        server.addr,
        "POST",
        "/v1/pairing/requests",
        &[("Content-Type", "application/json")],
        br#"{"device_name":"Second Laptop"}"#.to_vec(),
    )
    .await;
    assert_eq!(create.status, 202);
    let create_json: serde_json::Value = serde_json::from_slice(&create.body).unwrap();
    let request_id = create_json["request_id"].as_str().unwrap();
    crate::testing::approve_pending_pairing_request(&server, request_id).await;
    let second = https_request(
        server.addr,
        "GET",
        &format!("/v1/pairing/requests/{request_id}/credential"),
        &[],
        Vec::new(),
    )
    .await;
    assert_eq!(second.status, 200);
    let second_json: serde_json::Value = serde_json::from_slice(&second.body).unwrap();
    let second_token = second_json["bearer_token"].as_str().unwrap();
    let first_models = https_request(
        server.addr,
        "GET",
        "/v1/models",
        &[(
            "Authorization",
            bearer_auth_header(&first.bearer_token).as_str(),
        )],
        Vec::new(),
    )
    .await;
    let second_auth = bearer_auth_header(second_token);
    let second_models = https_request(
        server.addr,
        "GET",
        "/v1/models",
        &[("Authorization", second_auth.as_str())],
        Vec::new(),
    )
    .await;
    assert_eq!(first_models.status, 200);
    assert_eq!(second_models.status, 200);
    assert_ne!(first.bearer_token, second_token);
}

/// SSOT 5 / B6 / B7: device tokens cannot read Voice ID or mutate models.
/// Covered by the route matrix; this pins the product sentence with explicit
/// HTTPS calls through the pairing helper.
#[tokio::test(flavor = "multi_thread")]
async fn ssot_5_device_token_cannot_read_or_mutate_voice_id() {
    let temp = tempfile::tempdir().unwrap();
    let server = spawn_loopback_pairing_server(temp.path()).await;
    let credential = approve_loopback_pairing(&server).await;
    let auth = bearer_auth_header(&credential.bearer_token);
    let listed = https_request(
        server.addr,
        "GET",
        "/v1/voice-id/persons",
        &[("Authorization", auth.as_str())],
        Vec::new(),
    )
    .await;
    assert_eq!(listed.status, 403);
    let enrolled = https_request(
        server.addr,
        "POST",
        "/v1/voice-id/persons",
        &[
            ("Authorization", auth.as_str()),
            ("Content-Type", "application/json"),
        ],
        b"{}".to_vec(),
    )
    .await;
    assert_eq!(enrolled.status, 403);
}

/// SSOT 21: the owning device must be able to cancel its own file job.
/// Kills `set_transcription_owner` no-op and `transcription_owner -> None`.
#[tokio::test(flavor = "multi_thread")]
async fn device_can_cancel_its_own_file_job() {
    let temp = tempfile::tempdir().unwrap();
    let server = spawn_loopback_pairing_server(temp.path()).await;
    let first = approve_loopback_pairing(&server).await;
    let _delay =
        openasr_core::testing::MockTranscribeDelayGuard::new(std::time::Duration::from_secs(2));
    let (content_type, body) = sample_multipart(Some("own-job"), false);
    let first_auth = bearer_auth_header(&first.bearer_token);
    let first_job = {
        let addr = server.addr;
        let auth = first_auth.clone();
        let content_type = content_type.clone();
        tokio::spawn(async move {
            https_request(
                addr,
                "POST",
                "/v1/audio/transcriptions",
                &[
                    ("Authorization", auth.as_str()),
                    ("X-OpenASR-Remote-Compute", "client"),
                    ("Content-Type", content_type.as_str()),
                ],
                body,
            )
            .await
        })
    };
    tokio::time::sleep(std::time::Duration::from_millis(150)).await;
    let cancel = https_request(
        server.addr,
        "POST",
        "/v1/audio/transcriptions/own-job/cancel",
        &[("Authorization", first_auth.as_str())],
        Vec::new(),
    )
    .await;
    assert_eq!(
        cancel.status,
        202,
        "owning device must control its own job: {}",
        String::from_utf8_lossy(&cancel.body)
    );
    first_job.abort();
}

/// SSOT 6: stream file jobs from a paired device remap enrolled Voice ID to
/// anonymous labels. Deleting `!voice_id_allowed` would leave voice_id=true
/// and the mock backend would fail closed.
#[tokio::test(flavor = "multi_thread")]
async fn paired_device_stream_file_job_remaps_enrolled_voice_id() {
    let temp = tempfile::tempdir().unwrap();
    let server = spawn_loopback_pairing_server(temp.path()).await;
    let credential = approve_loopback_pairing(&server).await;
    let boundary = "openasr-stream-voice";
    let body = format!(
        "--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"sample.wav\"\r\nContent-Type: audio/wav\r\n\r\nnot a real wav\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nwhisper-large-v3-turbo\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"diarize\"\r\n\r\ntrue\r\n--{boundary}--\r\n"
    );
    let auth = bearer_auth_header(&credential.bearer_token);
    let streamed = https_request(
        server.addr,
        "POST",
        "/v1/audio/transcriptions?stream=true",
        &[
            ("Authorization", auth.as_str()),
            ("X-OpenASR-Remote-Compute", "client"),
            (
                "Content-Type",
                &format!("multipart/form-data; boundary={boundary}"),
            ),
        ],
        body.into_bytes(),
    )
    .await;
    assert_eq!(
        streamed.status,
        200,
        "paired stream diarize must remap to anonymous labels, got {}: {}",
        streamed.status,
        String::from_utf8_lossy(&streamed.body)
    );
    let body = String::from_utf8_lossy(&streamed.body);
    assert!(
        body.contains("event: done") && body.contains("\"status\":\"ok\""),
        "paired stream diarize must complete, got {body}"
    );
}

/// Local/operator stream still refuses enrolled Voice ID on the mock backend.
/// Kills flipping `!voice_id_allowed` so the operator path is stripped too.
#[tokio::test]
async fn operator_stream_file_job_keeps_enrolled_voice_id_fail_closed() {
    let temp = tempfile::tempdir().unwrap();
    let home = temp.path().join("home");
    std::fs::create_dir_all(&home).unwrap();
    let app = policy_app(ServerRuntime::default(), home);
    let boundary = "openasr-local-voice";
    let body = format!(
        "--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"sample.wav\"\r\nContent-Type: audio/wav\r\n\r\nnot a real wav\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nwhisper-large-v3-turbo\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"diarize\"\r\n\r\ntrue\r\n--{boundary}--\r\n"
    );
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/audio/transcriptions?stream=true")
                .header(
                    header::CONTENT_TYPE,
                    format!("multipart/form-data; boundary={boundary}"),
                )
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        response.status(),
        StatusCode::BAD_REQUEST,
        "operator stream enrolled Voice ID must fail closed over HTTP before SSE, got {}",
        response.status()
    );
    let body = to_bytes(response.into_body(), 1024 * 64).await.unwrap();
    let body = String::from_utf8_lossy(&body);
    assert!(
        body.to_ascii_lowercase().contains("voice")
            || body.to_ascii_lowercase().contains("diariz")
            || body.to_ascii_lowercase().contains("speaker"),
        "operator stream enrolled Voice ID must fail closed: {body}"
    );
}

/// SSOT 21: the id-less progress endpoint uses the same owner gate as
/// `{id}/progress`. A paired device must not observe an operator-local job.
#[tokio::test]
async fn legacy_progress_hides_operator_local_and_peer_jobs_from_device() {
    use openasr_core::api::backend::{
        ProgressBackendClass, ProgressPlan, ProgressPlanInput, ProgressReporter,
        ProgressSegmenterKind,
    };

    let plan = ProgressPlan::build(ProgressPlanInput {
        audio_duration_s: 1.0,
        voice_id: false,
        external_diarize: false,
        segmenter: ProgressSegmenterKind::Auto,
        punctuate: false,
        align: false,
        backend: ProgressBackendClass::AutoOrCpu,
        persist: false,
    });
    let _reporter = ProgressReporter::install(Some("progress-job".to_string()), plan);
    let auth = ServerAuth::pairing("admin-token");
    {
        let mut pairing = auth.lock_pairing();
        pairing.credentials.insert(
            "aaaaaaaaaaaaaaaaaaaaaaaa".to_string(),
            DeviceCredentialRecord {
                device_id: "aaaaaaaaaaaaaaaaaaaaaaaa".to_string(),
                device_name: "Phone".to_string(),
                token_hash: bearer_token_hash("device-token"),
                issued_at_unix_secs: 1,
                last_seen_unix_secs: None,
                revoked: false,
            },
        );
    }
    let distribution = DistributionContext::new(DistributionRuntime {
        openasr_home: None,
        catalog_url: None,
        catalog_local_override: None,
    });
    let mut headers = axum::http::HeaderMap::new();
    headers.insert(
        header::AUTHORIZATION,
        "Bearer device-token".parse().unwrap(),
    );

    let hidden = transcription_progress(
        axum::Extension(auth.clone()),
        axum::Extension(distribution.clone()),
        headers.clone(),
    )
    .await
    .expect("device reading operator-local progress must not error");
    let hidden_json = to_bytes(hidden.into_body(), 1024 * 64).await.unwrap();
    let hidden_value: serde_json::Value = serde_json::from_slice(&hidden_json).unwrap();
    assert_eq!(
        hidden_value["phase"],
        serde_json::Value::Null,
        "operator-local progress must read idle to a paired device: {hidden_value}"
    );

    distribution.set_transcription_owner("progress-job", Some("bbbbbbbbbbbbbbbbbbbbbbbb"));
    let peer = transcription_progress(
        axum::Extension(auth.clone()),
        axum::Extension(distribution.clone()),
        headers.clone(),
    )
    .await
    .expect("device reading a peer job must not error");
    let peer_json = to_bytes(peer.into_body(), 1024 * 64).await.unwrap();
    let peer_value: serde_json::Value = serde_json::from_slice(&peer_json).unwrap();
    assert_eq!(
        peer_value["phase"],
        serde_json::Value::Null,
        "peer progress must read idle: {peer_value}"
    );

    distribution.set_transcription_owner("progress-job", Some("aaaaaaaaaaaaaaaaaaaaaaaa"));
    let own = transcription_progress(
        axum::Extension(auth),
        axum::Extension(distribution),
        headers,
    )
    .await
    .expect("owner must read its own legacy progress");
    let own_json = to_bytes(own.into_body(), 1024 * 64).await.unwrap();
    let own_value: serde_json::Value = serde_json::from_slice(&own_json).unwrap();
    assert_eq!(
        own_value["phase"].as_str(),
        Some("decode"),
        "owning device must see live progress: {own_value}"
    );
}

#[test]
fn extract_registered_routes_reads_every_lib_rs_route() {
    let routes = extract_registered_routes(include_str!("lib.rs"));
    assert!(routes.contains(&("GET".to_string(), "/health".to_string())));
    assert!(routes.contains(&("POST".to_string(), "/v1/models/default".to_string())));
    assert!(routes.contains(&("GET".to_string(), "/v1/models/local".to_string())));
    assert!(routes.contains(&("GET".to_string(), "/v1/audio/realtime".to_string())));
    assert!(
        routes.len() >= 50,
        "lib.rs documents ~50 .route registrations; parser saw {}",
        routes.len()
    );
}
