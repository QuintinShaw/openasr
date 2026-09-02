//! Red-team coverage for PR #378 align promises that currently fail.
//! Ignored so default `cargo nextest` stays green.

use openasr_core::{
    ExecutionTarget, NativeExecutionServices, align_plain_transcript_to_audio,
    load_native_wav_16khz_mono_f32_v0,
};
use std::{
    io::Write,
    path::{Path, PathBuf},
    process::{Child, Stdio},
    time::{Duration, Instant},
};
use tempfile::TempDir;

const TMP_ROOT: &str = "/Volumes/QuintinDocument/tmp";
const PACK_SOURCE: &str = "/Users/quintinshaw/.openasr/models/objects/sha256/5b36662d373cbee279f168c2f88700a59a93246584bf3a5ade9e800e41c7807b/content";
const KANJI_ONLY_JAPANESE: &str = "日本国民";
const HTTP_TIMEOUT_ALIGN: Duration = Duration::from_secs(300);

fn jfk_wav() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/jfk.wav")
}

fn isolated_home() -> TempDir {
    std::fs::create_dir_all(TMP_ROOT).expect("create large-volume tmp root");
    tempfile::Builder::new()
        .prefix("oasr-rt378.")
        .tempdir_in(TMP_ROOT)
        .expect("create isolated OPENASR_HOME on large volume")
}

fn pack_source() -> PathBuf {
    let path = PathBuf::from(PACK_SOURCE);
    assert!(
        path.is_file(),
        "forced-aligner pack source missing (do not skip): {}",
        path.display()
    );
    path
}

fn copy_pack_into(home: &Path) -> PathBuf {
    let dest = home.join("qwen3-forced-aligner.oasr");
    let source = pack_source();
    std::fs::copy(&source, &dest).unwrap_or_else(|error| {
        panic!(
            "copy forced-aligner pack with cp semantics failed (hard links are forbidden): {error}"
        )
    });
    let source_len = std::fs::metadata(&source).expect("source metadata").len();
    let dest_len = std::fs::metadata(&dest)
        .expect("copied pack metadata")
        .len();
    assert_eq!(
        source_len, dest_len,
        "copied pack size mismatch: source {source_len} dest {dest_len}"
    );
    dest
}

fn isolate_process_env(home: &Path, pack: &Path) {
    unsafe {
        std::env::set_var("OPENASR_HOME", home);
        std::env::remove_var("OPENASR_MODELS_DIR");
        std::env::set_var("OPENASR_FORCED_ALIGNER_PACK", pack);
    }
}

struct ServeGuard {
    child: Child,
    addr: String,
}

impl Drop for ServeGuard {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

struct HttpResponse {
    status: u16,
    body: String,
}

fn parse_http(raw: &str) -> HttpResponse {
    let (head, body) = raw
        .split_once("\r\n\r\n")
        .or_else(|| raw.split_once("\n\n"))
        .unwrap_or((raw, ""));
    let status = head
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .and_then(|code| code.parse().ok())
        .unwrap_or(0);
    HttpResponse {
        status,
        body: body.to_string(),
    }
}

fn curl_http(
    addr: &str,
    method: &str,
    path: &str,
    content_type: Option<&str>,
    body: &[u8],
    timeout: Duration,
) -> HttpResponse {
    let mut command = std::process::Command::new("curl");
    command
        .args([
            "-sS",
            "-i",
            "--http1.1",
            "--max-time",
            &timeout.as_secs().max(1).to_string(),
            "-X",
            method,
            &format!("http://{addr}{path}"),
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if let Some(content_type) = content_type {
        command.args(["-H", &format!("Content-Type: {content_type}")]);
        command.arg("--data-binary");
        command.arg("@-");
        command.stdin(Stdio::piped());
    } else {
        command.stdin(Stdio::null());
    }
    let mut child = command.spawn().expect("spawn curl");
    if !body.is_empty() {
        if let Some(mut stdin) = child.stdin.take() {
            stdin.write_all(body).expect("write curl body");
        }
    }
    let output = child.wait_with_output().expect("curl exit");
    if !output.status.success() {
        return HttpResponse {
            status: 0,
            body: format!(
                "{}\n{}",
                String::from_utf8_lossy(&output.stderr),
                String::from_utf8_lossy(&output.stdout)
            ),
        };
    }
    parse_http(&String::from_utf8_lossy(&output.stdout))
}

#[allow(clippy::zombie_processes)]
fn spawn_serve(home: &Path, pack: &Path) -> ServeGuard {
    let mut command = std::process::Command::new(env!("CARGO_BIN_EXE_openasr"));
    command
        .env("OPENASR_HOME", home)
        .env("OPENASR_FORCED_ALIGNER_PACK", pack)
        .env("OPENASR_OFFLINE", "1")
        .env_remove("OPENASR_MODEL")
        .env_remove("OPENASR_ADDR")
        .env_remove("OPENASR_ASSUME_YES")
        .env_remove("OPENASR_CATALOG_URL")
        .env_remove("OPENASR_CATALOG_FILE")
        .env_remove("OPENASR_CATALOG_IDENTITY")
        .env_remove("OPENASR_MODELS_DIR")
        .args(["serve", "--backend", "native", "--addr", "127.0.0.1:0"])
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit());
    let mut child = command.spawn().expect("spawn openasr serve");
    let stdout = child.stdout.take().expect("piped stdout");
    let mut reader = std::io::BufReader::new(stdout);
    let deadline = Instant::now() + Duration::from_secs(15);
    let addr = loop {
        use std::io::BufRead;
        let mut line = String::new();
        let bytes_read = reader.read_line(&mut line).expect("read serve stdout");
        if bytes_read == 0 {
            let status = child.wait().expect("serve exit");
            panic!("openasr serve exited before listening: {status:?}");
        }
        if let Some(rest) = line
            .trim_end()
            .strip_prefix("OpenASR server listening on http://")
        {
            break rest.to_string();
        }
        if Instant::now() > deadline {
            let _ = child.kill();
            let _ = child.wait();
            panic!("openasr serve did not report listening within 15s");
        }
    };
    std::thread::spawn(move || {
        let mut sink = Vec::new();
        let _ = std::io::Read::read_to_end(&mut reader, &mut sink);
    });
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let health = curl_http(&addr, "GET", "/health", None, &[], Duration::from_secs(2));
        if health.status == 200 {
            break;
        }
        if Instant::now() > deadline {
            panic!(
                "openasr serve never answered /health (last status {})",
                health.status
            );
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    ServeGuard { child, addr }
}

fn multipart_precise_timeline(
    wav: &[u8],
    transcript: &str,
    extra: &[(&str, &str)],
) -> (String, Vec<u8>) {
    let boundary = "rt378boundary";
    let mut body = Vec::new();
    body.extend_from_slice(
        format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"jfk.wav\"\r\nContent-Type: audio/wav\r\n\r\n"
        )
        .as_bytes(),
    );
    body.extend_from_slice(wav);
    body.extend_from_slice(b"\r\n");
    let mut fields = vec![("transcript", transcript)];
    fields.extend_from_slice(extra);
    for (name, value) in fields {
        body.extend_from_slice(
            format!("--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n")
                .as_bytes(),
        );
        body.extend_from_slice(value.as_bytes());
        body.extend_from_slice(b"\r\n");
    }
    body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
    (format!("multipart/form-data; boundary={boundary}"), body)
}

fn japanese_fail_closed(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    lower.contains("japanese") || lower.contains("does not yet support")
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
        || value
            .get("segments")
            .and_then(serde_json::Value::as_array)
            .is_some_and(|segments| {
                segments.iter().any(|segment| {
                    segment
                        .get("words")
                        .and_then(serde_json::Value::as_array)
                        .is_some_and(|words| !words.is_empty())
                })
            })
}

#[test]
#[ignore = "redteam: pr-378"]
fn rt_378_kanji_only_japanese_tagged_en_or_auto_fails_closed() {
    let home = isolated_home();
    let pack = copy_pack_into(home.path());
    isolate_process_env(home.path(), &pack);
    let samples = load_native_wav_16khz_mono_f32_v0(&jfk_wav(), "rt-378", "rt-378")
        .expect("load jfk.wav samples");
    let services = NativeExecutionServices::for_local_process().expect("native execution services");

    let mut leaks = Vec::new();
    for language in [Some("en"), Some("auto"), None] {
        match align_plain_transcript_to_audio(
            KANJI_ONLY_JAPANESE.to_string(),
            &samples,
            &services,
            ExecutionTarget::Cpu,
            language,
            true,
        ) {
            Ok(transcription) => {
                let words: Vec<&str> = transcription
                    .segments
                    .iter()
                    .flat_map(|segment| segment.words.iter().map(|word| word.word.as_str()))
                    .collect();
                leaks.push(format!(
                    "public API language={language:?} returned text={:?} words={words:?}",
                    transcription.text
                ));
            }
            Err(error) => {
                let message = error.to_string();
                if !japanese_fail_closed(&message) {
                    leaks.push(format!(
                        "public API language={language:?} failed with non-Japanese error: {message}"
                    ));
                }
            }
        }
    }

    let serve = spawn_serve(home.path(), &pack);
    let wav = std::fs::read(jfk_wav()).expect("read jfk.wav");
    let (content_type, body) = multipart_precise_timeline(
        &wav,
        KANJI_ONLY_JAPANESE,
        &[
            ("language", "en"),
            ("response_format", "verbose_json"),
            ("execution_target", "cpu"),
        ],
    );
    let response = curl_http(
        &serve.addr,
        "POST",
        "/v1/audio/precise-timeline",
        Some(&content_type),
        &body,
        HTTP_TIMEOUT_ALIGN,
    );
    if response.status == 200 || looks_like_aligned_timeline(&response.body) {
        leaks.push(format!(
            "HTTP language=en status={} body={}",
            response.status, response.body
        ));
    } else if response.status != 400 || !japanese_fail_closed(&response.body) {
        leaks.push(format!(
            "HTTP language=en expected 400 Japanese fail-closed, got {} body={}",
            response.status, response.body
        ));
    }

    assert!(
        leaks.is_empty(),
        "kanji-only Japanese tagged en/auto must fail closed (FAQ: language tag or script); leaked timelines:\n{}",
        leaks.join("\n")
    );
}
