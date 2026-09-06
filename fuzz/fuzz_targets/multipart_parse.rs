#![no_main]

use libfuzzer_sys::fuzz_target;

const BOUNDARY: &str = "fuzzboundary";

fn push_field(body: &mut Vec<u8>, name: &str, value: &[u8]) {
    body.extend_from_slice(b"--");
    body.extend_from_slice(BOUNDARY.as_bytes());
    body.extend_from_slice(b"\r\nContent-Disposition: form-data; name=\"");
    body.extend_from_slice(name.as_bytes());
    body.extend_from_slice(b"\"\r\n\r\n");
    body.extend_from_slice(value);
    body.extend_from_slice(b"\r\n");
}

fn push_file(body: &mut Vec<u8>, name: &str, filename: &str, value: &[u8]) {
    body.extend_from_slice(b"--");
    body.extend_from_slice(BOUNDARY.as_bytes());
    body.extend_from_slice(b"\r\nContent-Disposition: form-data; name=\"");
    body.extend_from_slice(name.as_bytes());
    body.extend_from_slice(b"\"; filename=\"");
    body.extend_from_slice(filename.as_bytes());
    body.extend_from_slice(b"\"\r\nContent-Type: application/octet-stream\r\n\r\n");
    body.extend_from_slice(value);
    body.extend_from_slice(b"\r\n");
}

fn structured_enroll_body(data: &[u8]) -> Vec<u8> {
    let mut body = Vec::with_capacity(data.len().saturating_add(256));
    push_field(&mut body, "display_name", b"fuzz");
    push_field(&mut body, "sample_label", data.get(..32).unwrap_or(data));
    push_file(&mut body, "wav", "clip.wav", data);
    if data.first().copied().unwrap_or(0) & 1 == 1 {
        push_file(&mut body, "sample", "empty.wav", b"");
    }
    body.extend_from_slice(b"--");
    body.extend_from_slice(BOUNDARY.as_bytes());
    body.extend_from_slice(b"--\r\n");
    body
}

fn structured_sample_body(data: &[u8]) -> Vec<u8> {
    let mut body = Vec::with_capacity(data.len().saturating_add(256));
    push_field(&mut body, "sample_label", b"fuzz");
    push_file(&mut body, "wav", "clip.wav", data);
    body.extend_from_slice(b"--");
    body.extend_from_slice(BOUNDARY.as_bytes());
    body.extend_from_slice(b"--\r\n");
    body
}

fuzz_target!(|data: &[u8]| {
    let content_type = format!("multipart/form-data; boundary={BOUNDARY}");
    let enroll = structured_enroll_body(data);
    openasr_server::fuzz::fuzz_parse_enroll_multipart(&enroll, &content_type);
    let sample = structured_sample_body(data);
    openasr_server::fuzz::fuzz_parse_sample_multipart(&sample, &content_type);
    openasr_server::fuzz::fuzz_parse_enroll_multipart(data, &content_type);
    openasr_server::fuzz::fuzz_parse_sample_multipart(data, "text/plain");
});
