use std::{
    cell::Cell,
    collections::VecDeque,
    fs,
    io::{self, Cursor, Read, Write},
    net::TcpListener,
    path::Path,
    sync::{
        Arc, Barrier, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
};

use crate::{
    CATALOG_FEATURE_SPEAKER_DIARIZATION, CatalogBackendFile, CatalogBackendFileRole,
    CatalogBackendVendor, CatalogCapability, CatalogCapabilityRole, CatalogMirror, CatalogModel,
    CatalogModelKind, CatalogQuant, LicenseClass, ModelCatalog, ResolvedCatalogBackendPull,
    ResolvedCatalogPull,
    testing::{TinyGgufFixtureSpec, write_tiny_gguf_runtime_source},
};

use super::*;

#[cfg(unix)]
use std::os::unix::fs::symlink;

#[cfg(unix)]
use std::{ffi::CString, os::unix::ffi::OsStrExt, time::SystemTime, time::UNIX_EPOCH};

#[derive(Clone)]
struct ResponseSpec {
    status: u16,
    body: Vec<u8>,
}

#[derive(Clone, Default)]
struct FakeClient {
    responses: Arc<Mutex<VecDeque<ResponseSpec>>>,
    ranges: Arc<Mutex<Vec<Option<u64>>>>,
    urls: Arc<Mutex<Vec<String>>>,
}

impl FakeClient {
    fn with_responses(responses: Vec<ResponseSpec>) -> Self {
        Self {
            responses: Arc::new(Mutex::new(responses.into())),
            ranges: Arc::new(Mutex::new(Vec::new())),
            urls: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn ranges(&self) -> Vec<Option<u64>> {
        self.ranges.lock().unwrap().clone()
    }

    fn urls(&self) -> Vec<String> {
        self.urls.lock().unwrap().clone()
    }
}

impl DownloadClient for FakeClient {
    fn open(&mut self, url: &str, range: Option<ByteRange>) -> Result<DownloadResponse, PullError> {
        let range_start = range.map(|range| range.start);
        self.urls.lock().unwrap().push(url.to_string());
        self.ranges.lock().unwrap().push(range_start);
        let response = self
            .responses
            .lock()
            .unwrap()
            .pop_front()
            .expect("fake response");
        let content_length = response.body.len() as u64;
        let content_range = fake_content_range(response.status, range_start, content_length);
        Ok(DownloadResponse {
            status: response.status,
            content_length: Some(content_length),
            content_range,
            etag: Some("etag-test".to_string()),
            reader: Box::new(Cursor::new(response.body)),
        })
    }
}

fn fake_content_range(
    status: u16,
    range_start: Option<u64>,
    content_length: u64,
) -> Option<String> {
    if status != 206 || content_length == 0 {
        return None;
    }
    let start = range_start?;
    let end = start.checked_add(content_length)?.checked_sub(1)?;
    let total = end.checked_add(1)?;
    Some(format!("bytes {start}-{end}/{total}"))
}

enum FirstResponse {
    Timeout,
    SingleByte,
}

struct StalledThenSuccessClient {
    bytes: Vec<u8>,
    first_response: FirstResponse,
    attempts: usize,
    ranges: Vec<Option<u64>>,
}

impl StalledThenSuccessClient {
    fn new(bytes: Vec<u8>, first_response: FirstResponse) -> Self {
        Self {
            bytes,
            first_response,
            attempts: 0,
            ranges: Vec::new(),
        }
    }

    fn ranges(&self) -> Vec<Option<u64>> {
        self.ranges.clone()
    }
}

impl DownloadClient for StalledThenSuccessClient {
    fn open(
        &mut self,
        _url: &str,
        range: Option<ByteRange>,
    ) -> Result<DownloadResponse, PullError> {
        let range_start = range.map(|range| range.start);
        self.ranges.push(range_start);
        self.attempts += 1;
        let reader: Box<dyn Read> = match (&self.first_response, self.attempts) {
            (FirstResponse::Timeout, 1) => Box::new(TimedOutReader),
            (FirstResponse::SingleByte, 1) => Box::new(Cursor::new(vec![self.bytes[0]])),
            _ => Box::new(Cursor::new(self.bytes.clone())),
        };
        Ok(DownloadResponse {
            status: 200,
            content_length: Some(self.bytes.len() as u64),
            content_range: None,
            etag: Some("etag-test".to_string()),
            reader,
        })
    }
}

struct TimedOutReader;

impl Read for TimedOutReader {
    fn read(&mut self, _buf: &mut [u8]) -> io::Result<usize> {
        Err(io::Error::new(
            io::ErrorKind::TimedOut,
            "simulated stalled response body",
        ))
    }
}

struct PanicOnRead;

impl Read for PanicOnRead {
    fn read(&mut self, _buf: &mut [u8]) -> io::Result<usize> {
        panic!("content-length mismatch should fail before reading response body");
    }
}

struct InvalidRangeThenSuccessClient {
    bytes: Vec<u8>,
    split: usize,
    attempts: usize,
    ranges: Vec<Option<u64>>,
}

impl InvalidRangeThenSuccessClient {
    fn new(bytes: Vec<u8>, split: usize) -> Self {
        Self {
            bytes,
            split,
            attempts: 0,
            ranges: Vec::new(),
        }
    }

    fn ranges(&self) -> Vec<Option<u64>> {
        self.ranges.clone()
    }
}

impl DownloadClient for InvalidRangeThenSuccessClient {
    fn open(
        &mut self,
        _url: &str,
        range: Option<ByteRange>,
    ) -> Result<DownloadResponse, PullError> {
        let range_start = range.map(|range| range.start);
        self.ranges.push(range_start);
        self.attempts += 1;
        if self.attempts == 1 {
            let body_len = self.bytes.len() - self.split;
            let wrong_body = self.bytes[..body_len].to_vec();
            return Ok(DownloadResponse {
                status: 206,
                content_length: Some(body_len as u64),
                content_range: Some(format!("bytes 0-{}/{}", body_len - 1, self.bytes.len())),
                etag: Some("etag-test".to_string()),
                reader: Box::new(Cursor::new(wrong_body)),
            });
        }
        Ok(DownloadResponse {
            status: 200,
            content_length: Some(self.bytes.len() as u64),
            content_range: None,
            etag: Some("etag-test".to_string()),
            reader: Box::new(Cursor::new(self.bytes.clone())),
        })
    }
}

/// A range-aware mock `DownloadClient` for the concurrent chunked-download
/// tests. Unlike `FakeClient`'s fixed response queue (which assumes requests
/// arrive in a known sequential order), this serves any requested byte range
/// directly out of an in-memory buffer -- exactly how a real Range server
/// behaves -- so it gives deterministic, byte-correct responses regardless
/// of which order concurrent worker threads happen to issue requests in.
#[derive(Clone)]
struct RangeServerClient {
    bytes: Arc<Vec<u8>>,
    supports_range: Arc<AtomicBool>,
    /// ETags served in call order: the Nth call gets
    /// `etags[min(N, etags.len() - 1)]`, so a test can make the ETag change
    /// after a fixed number of requests to simulate a mid-download CDN swap.
    etags: Arc<Vec<String>>,
    calls: Arc<AtomicUsize>,
    requests: Arc<Mutex<Vec<(u64, Option<u64>)>>>,
}

impl RangeServerClient {
    fn new(bytes: Vec<u8>) -> Self {
        Self {
            bytes: Arc::new(bytes),
            supports_range: Arc::new(AtomicBool::new(true)),
            etags: Arc::new(vec!["etag-a".to_string()]),
            calls: Arc::new(AtomicUsize::new(0)),
            requests: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn without_range_support(self) -> Self {
        self.supports_range.store(false, Ordering::SeqCst);
        self
    }

    fn with_etag_sequence(mut self, etags: &[&str]) -> Self {
        self.etags = Arc::new(etags.iter().map(|etag| etag.to_string()).collect());
        self
    }

    fn requests(&self) -> Vec<(u64, Option<u64>)> {
        self.requests.lock().unwrap().clone()
    }

    fn call_count(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

impl DownloadClient for RangeServerClient {
    fn open(
        &mut self,
        _url: &str,
        range: Option<ByteRange>,
    ) -> Result<DownloadResponse, PullError> {
        let call_index = self.calls.fetch_add(1, Ordering::SeqCst);
        self.requests.lock().unwrap().push((
            range.map(|r| r.start).unwrap_or(0),
            range.and_then(|r| r.end),
        ));
        let etag = self.etags[call_index.min(self.etags.len() - 1)].clone();
        let total = self.bytes.len() as u64;
        if !self.supports_range.load(Ordering::SeqCst) || range.is_none() {
            return Ok(DownloadResponse {
                status: 200,
                content_length: Some(total),
                content_range: None,
                etag: Some(etag),
                reader: Box::new(Cursor::new(self.bytes.as_ref().clone())),
            });
        }
        let range = range.expect("checked above");
        let end = range
            .end
            .unwrap_or(total.saturating_sub(1))
            .min(total.saturating_sub(1));
        let start = range.start.min(end);
        let slice = self.bytes[start as usize..=end as usize].to_vec();
        Ok(DownloadResponse {
            status: 206,
            content_length: Some(slice.len() as u64),
            content_range: Some(format!("bytes {start}-{end}/{total}")),
            etag: Some(etag),
            reader: Box::new(Cursor::new(slice)),
        })
    }
}

/// A segment size that splits `total` bytes into roughly `segments` chunks,
/// for tests that need real multi-segment behavior without multi-hundred-MB
/// fixtures (see `PullOptions::parallel_segment_bytes_override`).
fn small_segment_bytes(total: usize, segments: u64) -> u64 {
    ((total as u64) / segments).max(1)
}

fn parallel_test_options(segment_bytes: u64) -> PullOptions {
    PullOptions {
        parallel_segment_bytes_override: Some(segment_bytes),
        ..PullOptions::default()
    }
}

fn tiny_pack_bytes() -> Vec<u8> {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("tiny.oasr");
    // `whisper_oasr_v1_non_streaming_cpu` alone deliberately omits the
    // whisper runtime scalar keys (block_count, head_count, ...) elsewhere
    // used to test fail-closed executor preflight; install now enforces that
    // same contract (see `validate_native_runtime_model_pack_contract`), so
    // this generic "any installable pack" fixture must be contract-complete.
    let spec = TinyGgufFixtureSpec::whisper_oasr_v1_encoder_graph_one_layer("moonshine-tiny");
    write_tiny_gguf_runtime_source(&path, &spec).unwrap();
    fs::read(path).unwrap()
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn resolved_for(bytes: &[u8]) -> ResolvedCatalogPull {
    ResolvedCatalogPull {
        requested: "moonshine-tiny:q8".to_string(),
        model_id: "moonshine-tiny".to_string(),
        display_name: "Moonshine Tiny".to_string(),
        quant: "q8_0".to_string(),
        suffix: "q8".to_string(),
        pull: "moonshine-tiny:q8".to_string(),
        filename: "moonshine-tiny-q8_0.oasr".to_string(),
        url: "https://huggingface.co/OpenASR/moonshine-tiny/resolve/0123456789abcdef0123456789abcdef01234567/moonshine-tiny-q8_0.oasr".to_string(),
        mirrors: Vec::new(),
        hf_revision: "0123456789abcdef0123456789abcdef01234567".to_string(),
        sha256: sha256_hex(bytes),
        size_bytes: bytes.len() as u64,
        license: "MIT".to_string(),
        license_url: "https://example.invalid/license".to_string(),
        license_class: crate::LicenseClass::Permissive,
    }
}

#[allow(dead_code)]
fn resolved_with_modelscope_mirror(bytes: &[u8]) -> ResolvedCatalogPull {
    let mut resolved = resolved_for(bytes);
    resolved.mirrors = vec![CatalogMirror {
        source: "modelscope".to_string(),
        url: "https://modelscope.cn/models/openasr/moonshine-tiny/resolve/0123456789abcdef0123456789abcdef01234567/moonshine-tiny-q8_0.oasr".to_string(),
    }];
    resolved
}

fn catalog_for_resolved(resolved: &ResolvedCatalogPull) -> ModelCatalog {
    ModelCatalog {
        schema_version: 1,
        generated_at: "2026-06-08T00:00:00Z".to_string(),
        catalog_url: "fixture".to_string(),
        backends: Vec::new(),
        language_labels: std::collections::BTreeMap::new(),
        models: vec![CatalogModel {
            id: resolved.model_id.clone(),
            kind: CatalogModelKind::AsrModel,
            capability: None,
            experimental: false,
            display_name: resolved.display_name.clone(),
            family: resolved.model_id.clone(),
            aliases: Vec::new(),
            pull_alias: None,
            size: "tiny".to_string(),
            languages: vec!["en".to_string()],
            language_mode: None,
            language_default: None,
            source_langs: Vec::new(),
            target_langs: Vec::new(),
            vendor: Some("OpenASR".to_string()),
            license: resolved.license.clone(),
            license_url: resolved.license_url.clone(),
            license_class: resolved.license_class.clone(),
            hf_repo: "OpenASR/moonshine-tiny".to_string(),
            hf_revision: resolved.hf_revision.clone(),
            public: true,
            min_cli_version: "0.1.0".to_string(),
            min_core_version: None,
            recommended_quant: resolved.quant.clone(),
            pull_recommended: resolved.pull.clone(),
            sort_weight: 0,
            recommended: false,
            upstream_release_date: None,
            emits_punctuation: None,
            prose: None,
            prose_locales: None,
            quants: vec![CatalogQuant {
                quant: resolved.quant.clone(),
                suffix: resolved.suffix.clone(),
                pull: resolved.pull.clone(),
                filename: resolved.filename.clone(),
                url: resolved.url.clone(),
                mirrors: resolved.mirrors.clone(),
                sha256: resolved.sha256.clone(),
                size_bytes: resolved.size_bytes,
                recommended: true,
                perf: None,
            }],
        }],
    }
}

fn capability_pack_catalog_for_resolved(resolved: &ResolvedCatalogPull) -> ModelCatalog {
    let mut catalog = catalog_for_resolved(resolved);
    let model = &mut catalog.models[0];
    model.kind = CatalogModelKind::CapabilityPack;
    model.capability = Some(CatalogCapability {
        feature: CATALOG_FEATURE_SPEAKER_DIARIZATION.to_string(),
        role: CatalogCapabilityRole::SpeakerEmbedder,
    });
    model.family = "wespeaker".to_string();
    model.size = "embedder".to_string();
    catalog
}

fn paths_for(home: &Path, resolved: &ResolvedCatalogPull) -> PullPaths {
    let target = PullTarget::from_resolved(resolved).unwrap();
    pull_paths(home, &target).unwrap()
}

fn write_complete_partial(
    home: &Path,
    resolved: &ResolvedCatalogPull,
    bytes: &[u8],
) -> (PullTarget, PullPaths) {
    let target = PullTarget::from_resolved(resolved).unwrap();
    let paths = pull_paths(home, &target).unwrap();
    ensure_storage_dir_within_root(home, &paths).unwrap();
    fs::write(&paths.partial_path, bytes).unwrap();
    write_partial_meta(
        &paths.partial_meta_path,
        &PartialMeta::for_target(&target, Some("etag-test".to_string()), bytes.len() as u64),
    )
    .unwrap();
    (target, paths)
}

fn assert_no_partial_or_install(paths: &PullPaths) {
    assert!(!paths.partial_path.exists());
    assert!(!paths.partial_meta_path.exists());
    assert!(!paths.final_path.exists());
    assert!(!paths.installed_meta_path.exists());
}

#[cfg(unix)]
fn set_stale_mtime(path: &Path) {
    let stale_seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
        .saturating_sub(LOCK_STALE_AFTER.as_secs() + 60);
    let c_path = CString::new(path.as_os_str().as_bytes()).unwrap();
    let times = [
        libc::timeval {
            tv_sec: stale_seconds as libc::time_t,
            tv_usec: 0,
        },
        libc::timeval {
            tv_sec: stale_seconds as libc::time_t,
            tv_usec: 0,
        },
    ];
    let result = unsafe { libc::utimes(c_path.as_ptr(), times.as_ptr()) };
    assert_eq!(
        result,
        0,
        "utimes failed for {}: {}",
        path.display(),
        io::Error::last_os_error()
    );
}

#[test]
fn capture_redirect_cookies_keeps_name_value_pairs_for_manual_redirects() {
    let mut headers = reqwest::header::HeaderMap::new();
    headers.append(
        reqwest::header::SET_COOKIE,
        reqwest::header::HeaderValue::from_static("acw_tc=first; Path=/; HttpOnly"),
    );
    headers.append(
        reqwest::header::SET_COOKIE,
        reqwest::header::HeaderValue::from_static("csrf_token=token-value; Path=/"),
    );

    let mut jar = vec![RedirectCookie {
        host: "huggingface.co".to_string(),
        cookie: "acw_tc=old".to_string(),
    }];
    capture_redirect_cookies(&headers, "huggingface.co", &mut jar);

    assert_eq!(
        jar.iter().map(|c| c.cookie.as_str()).collect::<Vec<_>>(),
        vec!["acw_tc=first", "csrf_token=token-value"]
    );
    assert!(jar.iter().all(|c| c.host == "huggingface.co"));
}

#[test]
fn redirect_cookies_are_scoped_to_the_setting_host() {
    // A cookie set by huggingface.co must not be replayed to a CDN/other host.
    let mut headers = reqwest::header::HeaderMap::new();
    headers.append(
        reqwest::header::SET_COOKIE,
        reqwest::header::HeaderValue::from_static("session=secret; Path=/"),
    );
    let mut jar = Vec::new();
    capture_redirect_cookies(&headers, "huggingface.co", &mut jar);

    assert_eq!(
        cookies_for_host(&jar, "huggingface.co"),
        vec!["session=secret"]
    );
    assert!(cookies_for_host(&jar, "cdn-lfs.evil.example").is_empty());
}

#[test]
fn hf_token_only_attaches_to_the_huggingface_host() {
    // The optional bearer token authenticates to huggingface.co only; it must
    // never ride a redirect to a CDN, mirror, the first-party worker, or an
    // attacker host.
    assert!(hf_token_allowed_for_host(Some("huggingface.co")));
    assert!(!hf_token_allowed_for_host(Some("cdn-lfs.huggingface.co")));
    assert!(!hf_token_allowed_for_host(Some("hf-mirror.com")));
    assert!(!hf_token_allowed_for_host(Some("modelscope.cn")));
    // The weights worker and the Xet CDN it forwards to are always anonymous.
    assert!(!hf_token_allowed_for_host(Some("weights.openasr.org")));
    assert!(!hf_token_allowed_for_host(Some("us.aws.cdn.hf.co")));
    assert!(!hf_token_allowed_for_host(Some("cdn-lfs.evil.example")));
    assert!(!hf_token_allowed_for_host(None));
}

#[test]
fn hf_token_normalizes_and_drops_empty_values() {
    // A whitespace-only or empty token reads as absent (anonymous); a real token is
    // trimmed. This is the per-var selection used by `hf_token_from_env` across
    // OPENASR_HF_TOKEN / HF_TOKEN / HUGGING_FACE_HUB_TOKEN.
    assert_eq!(normalize_hf_token(None), None);
    assert_eq!(normalize_hf_token(Some("   ".to_string())), None);
    assert_eq!(normalize_hf_token(Some(String::new())), None);
    assert_eq!(
        normalize_hf_token(Some("  hf_abc123  ".to_string())),
        Some("hf_abc123".to_string())
    );
}

#[test]
fn weights_worker_redirect_into_xet_is_followed_verbatim() {
    // The weights.openasr.org worker 302s a /resolve request through to Hugging
    // Face's Xet CDN, which the worker does NOT re-serve. That hop must be followed
    // verbatim (host unchanged) -- rewriting it back onto the worker would 404.
    // Same behavior as the direct huggingface.co source.
    let resolved = resolve_redirect_location(
        "https://weights.openasr.org/OpenASR/moonshine-tiny/resolve/abc/model.oasr",
        "https://us.aws.cdn.hf.co/repos/xx/blob?X-Amz-Signature=deadbeef",
    )
    .expect("xet redirect resolves");
    assert_eq!(
        resolved,
        "https://us.aws.cdn.hf.co/repos/xx/blob?X-Amz-Signature=deadbeef"
    );
}

#[test]
fn mirror_source_redirect_into_us_aws_cdn_is_followed_verbatim() {
    // Under the hf-mirror source, a 302 into the `us.aws.cdn.hf.co` Xet frontend
    // must be followed verbatim too: hf-mirror.com does not proxy Xet CAS paths, so
    // host-swapping it onto the mirror endpoint would 404. (`us.aws.cdn.hf.co` ends
    // with `.hf.co` but not `.huggingface.co`.)
    let resolved = resolve_redirect_location(
        "https://hf-mirror.com/OpenASR/moonshine-tiny/resolve/abc/model.oasr",
        "https://us.aws.cdn.hf.co/repos/xx/blob?X-Amz-Signature=deadbeef",
    )
    .expect("xet redirect resolves");
    assert_eq!(
        resolved,
        "https://us.aws.cdn.hf.co/repos/xx/blob?X-Amz-Signature=deadbeef"
    );
}

#[test]
fn redirect_to_non_https_target_is_rejected() {
    // An https origin redirecting to http:// must not silently downgrade.
    let err = resolve_redirect_location(
        "https://huggingface.co/model.gguf",
        "http://cdn.example/model.gguf",
    )
    .expect_err("http redirect target must be rejected");
    assert!(matches!(err, PullError::NonHttpsUrl { .. }), "got {err:?}");

    // A same-scheme https redirect still resolves.
    let ok = resolve_redirect_location(
        "https://huggingface.co/model.gguf",
        "https://cdn.example/model.gguf",
    )
    .expect("https redirect target resolves");
    assert!(ok.starts_with("https://"));
}

#[test]
fn pull_installs_valid_pack_and_writes_record() {
    let bytes = tiny_pack_bytes();
    let resolved = resolved_for(&bytes);
    let temp = tempfile::tempdir().unwrap();
    let mut client = FakeClient::with_responses(vec![ResponseSpec {
        status: 200,
        body: bytes.clone(),
    }]);
    let mut events = Vec::new();

    let installed = pull_model_pack_with_client(
        &resolved,
        temp.path(),
        &mut client,
        PullOptions::default(),
        |event| events.push(event),
    )
    .unwrap();

    assert_eq!(installed.pull, "moonshine-tiny:q8");
    assert!(installed.path.exists());
    assert!(
        paths_for(temp.path(), &resolved)
            .installed_meta_path
            .exists()
    );
    assert_eq!(list_installed_packs(temp.path()).unwrap().len(), 1);
    assert!(
        events
            .iter()
            .any(|event| matches!(event, PullProgress::Installed { .. }))
    );
}

#[test]
fn install_catalog_model_pack_from_path_requires_signed_catalog_digest_match() {
    let bytes = tiny_pack_bytes();
    let mut resolved = resolved_for(&bytes);
    resolved.sha256 = "b".repeat(64);
    let catalog = catalog_for_resolved(&resolved);
    let temp = tempfile::tempdir().unwrap();
    let source_path = temp.path().join("moonshine-tiny-q8_0.oasr");
    fs::write(&source_path, bytes).unwrap();

    let error = install_catalog_model_pack_from_path(&catalog, &source_path, temp.path(), |_| {})
        .unwrap_err();

    assert!(matches!(
        error,
        PullError::InvalidTarget {
            field: "sha256",
            ..
        }
    ));
    assert!(list_installed_packs(temp.path()).unwrap().is_empty());
}

#[test]
fn install_catalog_model_pack_from_path_reuses_catalog_target_and_marks_local_source() {
    let bytes = tiny_pack_bytes();
    let resolved = resolved_for(&bytes);
    let catalog = catalog_for_resolved(&resolved);
    let temp = tempfile::tempdir().unwrap();
    let source_path = temp.path().join("moonshine-tiny-q8_0.oasr");
    fs::write(&source_path, bytes).unwrap();
    let mut events = Vec::new();

    let installed =
        install_catalog_model_pack_from_path(&catalog, &source_path, temp.path(), |event| {
            events.push(event);
        })
        .unwrap();

    let expected_paths = paths_for(temp.path(), &resolved);
    assert_eq!(installed.pull, resolved.pull);
    assert_eq!(installed.path, expected_paths.final_path);
    assert_eq!(installed.source.as_deref(), Some("local"));
    assert!(installed.path.exists());
    assert_eq!(
        list_installed_packs(temp.path()).unwrap()[0]
            .source
            .as_deref(),
        Some("local")
    );
    assert!(
        events
            .iter()
            .any(|event| matches!(event, PullProgress::Installed { .. }))
    );
}

/// Fail-closed regression for the turbo-pack incident: a pack that carries a
/// full whisper runtime graph except `whisper.decoder.attention.head_count`
/// used to "install successfully" (catalog digest + GGUF preflight only) and
/// only failed the first time the daemon tried to run inference against it.
/// Install must now reject it up front, via the same runtime-contract parser
/// the executor uses, and name the missing key in the error.
#[test]
fn install_catalog_model_pack_from_path_rejects_whisper_pack_missing_decoder_head_count() {
    let temp = tempfile::tempdir().unwrap();
    let mut spec = TinyGgufFixtureSpec::whisper_oasr_v1_encoder_graph_one_layer("moonshine-tiny");
    spec.metadata
        .remove("whisper.decoder.attention.head_count")
        .expect("fixture must set the key this test removes");
    let broken_path = temp.path().join("broken-source.oasr");
    write_tiny_gguf_runtime_source(&broken_path, &spec).unwrap();
    let bytes = fs::read(&broken_path).unwrap();

    let resolved = resolved_for(&bytes);
    let catalog = catalog_for_resolved(&resolved);
    let source_path = temp.path().join("moonshine-tiny-q8_0.oasr");
    fs::write(&source_path, &bytes).unwrap();

    let error = install_catalog_model_pack_from_path(&catalog, &source_path, temp.path(), |_| {})
        .unwrap_err();

    let message = error.to_string();
    assert!(
        message.contains("whisper.decoder.attention.head_count"),
        "error must name the missing key: {message}"
    );
    assert!(
        message.contains("outdated") && message.contains("re-convert"),
        "error must explain the pack needs re-conversion: {message}"
    );
    assert!(matches!(error, PullError::RuntimeValidation { .. }));
    assert!(list_installed_packs(temp.path()).unwrap().is_empty());
    assert!(
        !paths_for(temp.path(), &resolved).final_path.exists(),
        "rejected pack must not be committed into the model store"
    );
}

#[test]
fn capability_pack_stays_pullable_and_importable_by_digest() {
    let bytes = tiny_pack_bytes();
    let mut resolved = resolved_for(&bytes);
    resolved.requested = "wespeaker-voxceleb-resnet34-lm:f32".to_string();
    resolved.model_id = "wespeaker-voxceleb-resnet34-lm".to_string();
    resolved.display_name = "WeSpeaker ResNet34 Speaker Embedder (VoxCeleb)".to_string();
    resolved.quant = "f32".to_string();
    resolved.suffix = "f32".to_string();
    resolved.pull = "wespeaker-voxceleb-resnet34-lm:f32".to_string();
    resolved.filename = "wespeaker-voxceleb-resnet34-lm-f32.oasr".to_string();
    resolved.url = "https://huggingface.co/OpenASR/wespeaker-voxceleb-resnet34-lm/resolve/0123456789abcdef0123456789abcdef01234567/wespeaker-voxceleb-resnet34-lm-f32.oasr".to_string();
    let catalog = capability_pack_catalog_for_resolved(&resolved);

    let from_catalog = resolve_catalog_pull(
        &catalog,
        &CatalogPullRequest {
            reference: "wespeaker-voxceleb-resnet34-lm:f32".to_string(),
            quant: None,
            size: None,
        },
    )
    .unwrap();
    assert_eq!(from_catalog.pull, "wespeaker-voxceleb-resnet34-lm:f32");

    let pull_home = tempfile::tempdir().unwrap();
    let mut client = FakeClient::with_responses(vec![ResponseSpec {
        status: 200,
        body: bytes.clone(),
    }]);
    let pulled = pull_model_pack_with_client(
        &from_catalog,
        pull_home.path(),
        &mut client,
        PullOptions::default(),
        |_| {},
    )
    .unwrap();
    assert_eq!(pulled.pull, "wespeaker-voxceleb-resnet34-lm:f32");

    let import_home = tempfile::tempdir().unwrap();
    let source_path = import_home
        .path()
        .join("wespeaker-voxceleb-resnet34-lm-f32.oasr");
    fs::write(&source_path, bytes).unwrap();
    let imported =
        install_catalog_model_pack_from_path(&catalog, &source_path, import_home.path(), |_| {})
            .unwrap();
    assert_eq!(imported.pull, "wespeaker-voxceleb-resnet34-lm:f32");
    assert_eq!(imported.source.as_deref(), Some("local"));
}

#[test]
fn pull_falls_back_to_next_source_after_sha_mismatch() {
    let bytes = tiny_pack_bytes();
    let resolved = resolved_for(&bytes);
    let temp = tempfile::tempdir().unwrap();
    let mut bad_bytes = bytes.clone();
    bad_bytes[32] ^= 0x01;
    let mut client = FakeClient::with_responses(vec![
        ResponseSpec {
            status: 200,
            body: bad_bytes,
        },
        ResponseSpec {
            status: 200,
            body: bytes,
        },
    ]);

    let installed = pull_model_pack_with_client_sources_and_cancel(
        &resolved,
        temp.path(),
        &mut client,
        PullOptions::default(),
        &[DownloadSource::Hf, DownloadSource::HfMirror],
        None,
        |_| {},
        || false,
        || false,
    )
    .unwrap();

    assert_eq!(installed.pull, "moonshine-tiny:q8");
    assert_eq!(
        client.urls(),
        vec![
            resolved.url.clone(),
            "https://hf-mirror.com/OpenASR/moonshine-tiny/resolve/0123456789abcdef0123456789abcdef01234567/moonshine-tiny-q8_0.oasr".to_string(),
        ]
    );
    let paths = paths_for(temp.path(), &resolved);
    assert!(paths.final_path.exists());
    assert!(!paths.partial_path.exists());
}

#[test]
fn pinned_source_does_not_fallback_after_sha_mismatch() {
    let bytes = tiny_pack_bytes();
    let resolved = resolved_for(&bytes);
    let temp = tempfile::tempdir().unwrap();
    let mut bad_bytes = bytes.clone();
    bad_bytes[32] ^= 0x01;
    let mut client = FakeClient::with_responses(vec![ResponseSpec {
        status: 200,
        body: bad_bytes,
    }]);

    let error = pull_model_pack_with_client_sources_and_cancel(
        &resolved,
        temp.path(),
        &mut client,
        PullOptions::default(),
        &[DownloadSource::Hf],
        None,
        |_| {},
        || false,
        || false,
    )
    .unwrap_err();

    assert!(matches!(error, PullError::ShaMismatch { .. }));
    assert_eq!(client.urls(), vec![resolved.url.clone()]);
    let paths = paths_for(temp.path(), &resolved);
    assert!(!paths.final_path.exists());
    assert!(!paths.partial_path.exists());
}

#[test]
fn pull_falls_back_to_hf_mirror_after_weights_404() {
    // weights.openasr.org only proxies the OpenASR/* org; a file outside that
    // scope 404s there even though it exists on the other sources. The chain
    // must fall through to hf-mirror instead of hard-failing the whole pull.
    let bytes = tiny_pack_bytes();
    let resolved = resolved_for(&bytes);
    let temp = tempfile::tempdir().unwrap();
    let mut client = FakeClient::with_responses(vec![
        ResponseSpec {
            status: 404,
            body: b"not found".to_vec(),
        },
        ResponseSpec {
            status: 200,
            body: bytes,
        },
    ]);

    let installed = pull_model_pack_with_client_sources_and_cancel(
        &resolved,
        temp.path(),
        &mut client,
        PullOptions::default(),
        &[DownloadSource::Weights, DownloadSource::HfMirror],
        None,
        |_| {},
        || false,
        || false,
    )
    .unwrap();

    assert_eq!(installed.pull, "moonshine-tiny:q8");
    assert_eq!(
        client.urls(),
        vec![
            "https://weights.openasr.org/OpenASR/moonshine-tiny/resolve/0123456789abcdef0123456789abcdef01234567/moonshine-tiny-q8_0.oasr".to_string(),
            "https://hf-mirror.com/OpenASR/moonshine-tiny/resolve/0123456789abcdef0123456789abcdef01234567/moonshine-tiny-q8_0.oasr".to_string(),
        ]
    );
    let paths = paths_for(temp.path(), &resolved);
    assert!(paths.final_path.exists());
    assert!(!paths.partial_path.exists());
}

#[test]
fn pull_falls_back_to_next_source_after_403_forbidden() {
    let bytes = tiny_pack_bytes();
    let resolved = resolved_for(&bytes);
    let temp = tempfile::tempdir().unwrap();
    let mut client = FakeClient::with_responses(vec![
        ResponseSpec {
            status: 403,
            body: b"forbidden".to_vec(),
        },
        ResponseSpec {
            status: 200,
            body: bytes,
        },
    ]);

    let installed = pull_model_pack_with_client_sources_and_cancel(
        &resolved,
        temp.path(),
        &mut client,
        PullOptions::default(),
        &[DownloadSource::Hf, DownloadSource::HfMirror],
        None,
        |_| {},
        || false,
        || false,
    )
    .unwrap();

    assert_eq!(installed.pull, "moonshine-tiny:q8");
    assert_eq!(
        client.urls(),
        vec![
            resolved.url.clone(),
            "https://hf-mirror.com/OpenASR/moonshine-tiny/resolve/0123456789abcdef0123456789abcdef01234567/moonshine-tiny-q8_0.oasr".to_string(),
        ]
    );
    let paths = paths_for(temp.path(), &resolved);
    assert!(paths.final_path.exists());
    assert!(!paths.partial_path.exists());
}

#[test]
fn pull_does_not_fallback_after_400_bad_request() {
    // 400 is a malformed request, not a per-source availability gap -- it
    // would recur identically against every source, so the chain must not
    // spend the remaining sources retrying it.
    let bytes = tiny_pack_bytes();
    let resolved = resolved_for(&bytes);
    let temp = tempfile::tempdir().unwrap();
    let mut client = FakeClient::with_responses(vec![ResponseSpec {
        status: 400,
        body: b"bad request".to_vec(),
    }]);

    let error = pull_model_pack_with_client_sources_and_cancel(
        &resolved,
        temp.path(),
        &mut client,
        PullOptions::default(),
        &[DownloadSource::Hf, DownloadSource::HfMirror],
        None,
        |_| {},
        || false,
        || false,
    )
    .unwrap_err();

    assert!(matches!(
        error,
        PullError::UnexpectedStatus { status: 400, .. }
    ));
    assert_eq!(client.urls(), vec![resolved.url.clone()]);
    let paths = paths_for(temp.path(), &resolved);
    assert!(!paths.final_path.exists());
    assert!(!paths.partial_path.exists());
}

#[test]
fn pull_does_not_fallback_after_401_unauthorized() {
    // 401 means the underlying (possibly gated) resource needs credentials
    // this pull does not have; switching mirrors cannot supply the missing
    // bearer token, so the chain must not burn the remaining sources on it.
    let bytes = tiny_pack_bytes();
    let resolved = resolved_for(&bytes);
    let temp = tempfile::tempdir().unwrap();
    let mut client = FakeClient::with_responses(vec![ResponseSpec {
        status: 401,
        body: b"unauthorized".to_vec(),
    }]);

    let error = pull_model_pack_with_client_sources_and_cancel(
        &resolved,
        temp.path(),
        &mut client,
        PullOptions::default(),
        &[DownloadSource::Hf, DownloadSource::HfMirror],
        None,
        |_| {},
        || false,
        || false,
    )
    .unwrap_err();

    assert!(matches!(
        error,
        PullError::UnexpectedStatus { status: 401, .. }
    ));
    assert_eq!(client.urls(), vec![resolved.url.clone()]);
    let paths = paths_for(temp.path(), &resolved);
    assert!(!paths.final_path.exists());
    assert!(!paths.partial_path.exists());
}

#[test]
fn pull_cancel_cleans_partial_download() {
    let bytes = tiny_pack_bytes();
    let resolved = resolved_for(&bytes);
    let temp = tempfile::tempdir().unwrap();
    let mut client = FakeClient::with_responses(vec![ResponseSpec {
        status: 200,
        body: bytes.clone(),
    }]);
    let cancel = Arc::new(AtomicBool::new(false));
    let cancel_on_progress = cancel.clone();

    let error = pull_model_pack_with_client_and_cancel(
        &resolved,
        temp.path(),
        &mut client,
        PullOptions::default(),
        |event| {
            if matches!(event, PullProgress::Downloading { .. }) {
                cancel_on_progress.store(true, Ordering::SeqCst);
            }
        },
        || cancel.load(Ordering::SeqCst),
        || false,
    )
    .unwrap_err();

    assert!(matches!(error, PullError::Canceled { .. }));
    let paths = paths_for(temp.path(), &resolved);
    assert!(!paths.partial_path.exists());
    assert!(!paths.partial_meta_path.exists());
    assert!(!paths.final_path.exists());
}

#[test]
fn pull_pause_preserves_partial_download() {
    let bytes = tiny_pack_bytes();
    let resolved = resolved_for(&bytes);
    let temp = tempfile::tempdir().unwrap();
    let mut client = FakeClient::with_responses(vec![ResponseSpec {
        status: 200,
        body: bytes.clone(),
    }]);
    let pause = Arc::new(AtomicBool::new(false));
    let pause_on_progress = pause.clone();

    let error = pull_model_pack_with_client_and_cancel(
        &resolved,
        temp.path(),
        &mut client,
        PullOptions::default(),
        |event| {
            if matches!(event, PullProgress::Downloading { .. }) {
                pause_on_progress.store(true, Ordering::SeqCst);
            }
        },
        || false,
        || pause.load(Ordering::SeqCst),
    )
    .unwrap_err();

    assert!(matches!(error, PullError::Paused { .. }));
    let paths = paths_for(temp.path(), &resolved);
    assert!(paths.partial_path.exists());
    assert!(paths.partial_meta_path.exists());
    assert!(!paths.final_path.exists());

    let mut resume_client = FakeClient::with_responses(vec![]);
    let installed = pull_model_pack_with_client(
        &resolved,
        temp.path(),
        &mut resume_client,
        PullOptions::default(),
        |_| {},
    )
    .unwrap();
    assert_eq!(installed.pull, "moonshine-tiny:q8");
    assert!(paths.final_path.exists());
}

#[test]
fn pull_cancel_pause_race_cancel_wins_and_cleans_partial_download() {
    let bytes = tiny_pack_bytes();
    let resolved = resolved_for(&bytes);
    let temp = tempfile::tempdir().unwrap();
    let mut client = FakeClient::with_responses(vec![ResponseSpec {
        status: 200,
        body: bytes.clone(),
    }]);
    let cancel = Arc::new(AtomicBool::new(false));
    let pause = Arc::new(AtomicBool::new(false));
    let race_started = Arc::new(AtomicBool::new(false));
    let cancel_on_progress = cancel.clone();
    let pause_on_progress = pause.clone();
    let race_started_on_progress = race_started.clone();

    let error = pull_model_pack_with_client_and_cancel(
        &resolved,
        temp.path(),
        &mut client,
        PullOptions::default(),
        |event| {
            if !matches!(event, PullProgress::Downloading { .. }) {
                return;
            }
            if race_started_on_progress.swap(true, Ordering::SeqCst) {
                return;
            }

            let barrier = Arc::new(Barrier::new(3));
            let cancel_barrier = barrier.clone();
            let cancel_flag = cancel_on_progress.clone();
            let cancel_thread = std::thread::spawn(move || {
                cancel_barrier.wait();
                cancel_flag.store(true, Ordering::SeqCst);
            });
            let pause_barrier = barrier.clone();
            let pause_flag = pause_on_progress.clone();
            let pause_thread = std::thread::spawn(move || {
                pause_barrier.wait();
                pause_flag.store(true, Ordering::SeqCst);
            });
            barrier.wait();
            cancel_thread.join().expect("cancel race thread");
            pause_thread.join().expect("pause race thread");
        },
        || cancel.load(Ordering::SeqCst),
        || pause.load(Ordering::SeqCst),
    )
    .unwrap_err();

    assert!(matches!(error, PullError::Canceled { .. }));
    assert!(race_started.load(Ordering::SeqCst));
    let paths = paths_for(temp.path(), &resolved);
    assert!(!paths.partial_path.exists());
    assert!(!paths.partial_meta_path.exists());
    assert!(!paths.final_path.exists());
}

#[test]
fn pull_resumes_partial_when_server_returns_206() {
    let bytes = tiny_pack_bytes();
    let resolved = resolved_for(&bytes);
    let temp = tempfile::tempdir().unwrap();
    let target = PullTarget::from_resolved(&resolved).unwrap();
    let paths = pull_paths(temp.path(), &target).unwrap();
    fs::create_dir_all(&paths.dir).unwrap();
    let split = bytes.len() / 2;
    fs::write(&paths.partial_path, &bytes[..split]).unwrap();
    write_partial_meta(
        &paths.partial_meta_path,
        &PartialMeta::for_target(&target, Some("etag-test".to_string()), split as u64),
    )
    .unwrap();
    let mut client = FakeClient::with_responses(vec![ResponseSpec {
        status: 206,
        body: bytes[split..].to_vec(),
    }]);

    pull_model_pack_with_client(
        &resolved,
        temp.path(),
        &mut client,
        PullOptions::default(),
        |_| {},
    )
    .unwrap();

    assert_eq!(client.ranges(), vec![Some(split as u64)]);
    assert_eq!(fs::read(paths.final_path).unwrap(), bytes);
}

#[test]
fn pull_keeps_partial_when_meta_url_came_from_another_download_source() {
    let bytes = tiny_pack_bytes();
    let resolved = resolved_for(&bytes);
    let temp = tempfile::tempdir().unwrap();
    let target = PullTarget::from_resolved(&resolved).unwrap();
    let paths = pull_paths(temp.path(), &target).unwrap();
    fs::create_dir_all(&paths.dir).unwrap();
    fs::write(&paths.partial_path, &bytes).unwrap();
    // The partial was downloaded via the mirror; this pull resolves the
    // huggingface.co URL. Same content identity, different transport URL.
    let mirror_target = target.with_url(
        "https://hf-mirror.com/OpenASR/moonshine-tiny/resolve/main/moonshine-tiny-q8_0.oasr"
            .to_string(),
    );
    write_partial_meta(
        &paths.partial_meta_path,
        &PartialMeta::for_target(
            &mirror_target,
            Some("etag-test".to_string()),
            bytes.len() as u64,
        ),
    )
    .unwrap();
    let mut client = FakeClient::with_responses(vec![]);

    pull_model_pack_with_client(
        &resolved,
        temp.path(),
        &mut client,
        PullOptions::default(),
        |_| {},
    )
    .unwrap();

    assert_eq!(client.ranges(), Vec::<Option<u64>>::new());
    assert_eq!(fs::read(paths.final_path).unwrap(), bytes);
}

#[test]
fn pull_restarts_partial_when_server_returns_200() {
    let bytes = tiny_pack_bytes();
    let resolved = resolved_for(&bytes);
    let temp = tempfile::tempdir().unwrap();
    let target = PullTarget::from_resolved(&resolved).unwrap();
    let paths = pull_paths(temp.path(), &target).unwrap();
    fs::create_dir_all(&paths.dir).unwrap();
    fs::write(&paths.partial_path, b"partial").unwrap();
    write_partial_meta(
        &paths.partial_meta_path,
        &PartialMeta::for_target(&target, Some("old".to_string()), 7),
    )
    .unwrap();
    let mut client = FakeClient::with_responses(vec![ResponseSpec {
        status: 200,
        body: bytes.clone(),
    }]);

    pull_model_pack_with_client(
        &resolved,
        temp.path(),
        &mut client,
        PullOptions::default(),
        |_| {},
    )
    .unwrap();

    assert_eq!(client.ranges(), vec![Some(7)]);
    assert_eq!(fs::read(paths.final_path).unwrap(), bytes);
}

#[test]
fn pull_restarts_partial_when_content_range_does_not_match_resume() {
    let bytes = tiny_pack_bytes();
    let resolved = resolved_for(&bytes);
    let temp = tempfile::tempdir().unwrap();
    let target = PullTarget::from_resolved(&resolved).unwrap();
    let paths = pull_paths(temp.path(), &target).unwrap();
    fs::create_dir_all(&paths.dir).unwrap();
    let split = bytes.len() / 2;
    fs::write(&paths.partial_path, &bytes[..split]).unwrap();
    write_partial_meta(
        &paths.partial_meta_path,
        &PartialMeta::for_target(&target, Some("etag-test".to_string()), split as u64),
    )
    .unwrap();
    let mut client = InvalidRangeThenSuccessClient::new(bytes.clone(), split);

    pull_model_pack_with_client(
        &resolved,
        temp.path(),
        &mut client,
        PullOptions::default(),
        |_| {},
    )
    .unwrap();

    assert_eq!(client.ranges(), vec![Some(split as u64), None]);
    assert_eq!(fs::read(paths.final_path).unwrap(), bytes);
}

#[test]
fn pull_discards_partial_when_metadata_does_not_match_target() {
    let bytes = tiny_pack_bytes();
    let resolved = resolved_for(&bytes);
    let temp = tempfile::tempdir().unwrap();
    let target = PullTarget::from_resolved(&resolved).unwrap();
    let paths = pull_paths(temp.path(), &target).unwrap();
    fs::create_dir_all(&paths.dir).unwrap();
    let split = bytes.len() / 2;
    fs::write(&paths.partial_path, &bytes[..split]).unwrap();
    let mut stale_target = target.clone();
    stale_target.sha256 = "0".repeat(64);
    write_partial_meta(
        &paths.partial_meta_path,
        &PartialMeta::for_target(&stale_target, Some("etag-test".to_string()), split as u64),
    )
    .unwrap();
    let mut client = FakeClient::with_responses(vec![ResponseSpec {
        status: 200,
        body: bytes.clone(),
    }]);

    pull_model_pack_with_client(
        &resolved,
        temp.path(),
        &mut client,
        PullOptions::default(),
        |_| {},
    )
    .unwrap();

    assert_eq!(client.ranges(), vec![None]);
    assert_eq!(fs::read(paths.final_path).unwrap(), bytes);
    assert!(!paths.partial_meta_path.exists());
}

#[test]
fn pull_rejects_sha_mismatch_and_removes_partial() {
    let bytes = tiny_pack_bytes();
    let mut resolved = resolved_for(&bytes);
    resolved.sha256 = "0".repeat(64);
    let temp = tempfile::tempdir().unwrap();
    let mut client = FakeClient::with_responses(vec![ResponseSpec {
        status: 200,
        body: bytes,
    }]);
    let error = pull_model_pack_with_client(
        &resolved,
        temp.path(),
        &mut client,
        PullOptions::default(),
        |_| {},
    )
    .unwrap_err();

    assert!(matches!(error, PullError::ShaMismatch { .. }));
    let paths = paths_for(temp.path(), &resolved);
    assert!(!paths.partial_path.exists());
    assert!(!paths.partial_meta_path.exists());
}

#[test]
fn pull_rejects_size_mismatch_and_removes_partial_metadata() {
    let bytes = tiny_pack_bytes();
    let mut resolved = resolved_for(&bytes);
    resolved.size_bytes += 1;
    let temp = tempfile::tempdir().unwrap();
    let (target, paths) = write_complete_partial(temp.path(), &resolved, &bytes);

    let error = verify_partial_and_install(&target, &paths, None, &|| false, |_| {}).unwrap_err();

    assert!(matches!(
        error,
        PullError::SizeMismatch {
            expected,
            actual,
            ..
        } if expected == resolved.size_bytes && actual == bytes.len() as u64
    ));
    assert_no_partial_or_install(&paths);
}

#[test]
fn verify_partial_and_install_removes_stale_segments_meta_on_single_stream_success() {
    let bytes = tiny_pack_bytes();
    let resolved = resolved_for(&bytes);
    let temp = tempfile::tempdir().unwrap();
    let (target, paths) = write_complete_partial(temp.path(), &resolved, &bytes);

    // Simulate a resume that began as a chunked/parallel download (which
    // persists `partial_segments_meta_path`) but finished through this
    // single-stream success path once the remaining bytes dropped below the
    // parallel-eligibility threshold, leaving a stale segments bitmap behind
    // that this success path must also clean up (it previously only removed
    // `partial_meta_path`).
    let meta = SegmentedPartialMeta {
        format: PARALLEL_META_FORMAT.to_string(),
        model_id: target.model_id.clone(),
        quant: target.quant.clone(),
        filename: target.filename.clone(),
        hf_revision: target.hf_revision.clone(),
        sha256: target.sha256.clone(),
        size_bytes: target.size_bytes,
        segment_bytes: bytes.len() as u64,
        etag: Some("etag-a".to_string()),
        segments_done: vec![true],
        updated_at_unix_seconds: 0,
    };
    write_partial_segments_meta(&paths.partial_segments_meta_path, &meta).unwrap();
    assert!(paths.partial_segments_meta_path.exists());

    verify_partial_and_install(&target, &paths, None, &|| false, |_| {}).unwrap();

    assert!(paths.final_path.exists());
    assert!(!paths.partial_meta_path.exists());
    assert!(
        !paths.partial_segments_meta_path.exists(),
        "single-stream success must also clean up a stale segments bitmap left \
         over from an earlier chunked/parallel attempt"
    );
}

#[test]
fn download_response_rejects_fresh_content_length_mismatch_before_reading() {
    let bytes = tiny_pack_bytes();
    let resolved = resolved_for(&bytes);
    let temp = tempfile::tempdir().unwrap();
    let target = PullTarget::from_resolved(&resolved).unwrap();
    let paths = pull_paths(temp.path(), &target).unwrap();
    let actual = resolved.size_bytes - 1;
    let response = DownloadResponse {
        status: 200,
        content_length: Some(actual),
        content_range: None,
        etag: Some("etag-test".to_string()),
        reader: Box::new(PanicOnRead),
    };
    let mut progress = |_| {};

    let error = match download_response(
        &target,
        &paths,
        0,
        response,
        &PullOptions::default(),
        &mut progress,
        &|| false,
        &|| false,
    ) {
        Ok(_) => panic!("content-length mismatch should fail"),
        Err(error) => error,
    };

    assert!(matches!(
        error,
        PullError::SizeMismatch {
            expected,
            actual: observed,
            ..
        } if expected == resolved.size_bytes && observed == actual
    ));
    assert!(!paths.partial_path.exists());
    assert!(!paths.partial_meta_path.exists());
}

#[test]
fn pull_retries_server_error_and_resumes_successfully() {
    let bytes = tiny_pack_bytes();
    let resolved = resolved_for(&bytes);
    let temp = tempfile::tempdir().unwrap();
    let mut client = FakeClient::with_responses(vec![
        ResponseSpec {
            status: 500,
            body: Vec::new(),
        },
        ResponseSpec {
            status: 200,
            body: bytes,
        },
    ]);

    let installed = pull_model_pack_with_client(
        &resolved,
        temp.path(),
        &mut client,
        PullOptions::default(),
        |_| {},
    )
    .unwrap();

    assert_eq!(installed.pull, "moonshine-tiny:q8");
    assert_eq!(client.ranges(), vec![None, None]);
}

#[test]
fn pull_retries_body_read_timeout_and_restarts_safely() {
    let bytes = tiny_pack_bytes();
    let resolved = resolved_for(&bytes);
    let temp = tempfile::tempdir().unwrap();
    let mut client = StalledThenSuccessClient::new(bytes, FirstResponse::Timeout);

    let installed = pull_model_pack_with_client(
        &resolved,
        temp.path(),
        &mut client,
        PullOptions::default(),
        |_| {},
    )
    .unwrap();

    assert_eq!(installed.pull, "moonshine-tiny:q8");
    assert_eq!(client.ranges(), vec![None, None]);
}

#[test]
fn pull_retries_low_speed_body_and_restarts_safely() {
    let bytes = tiny_pack_bytes();
    let resolved = resolved_for(&bytes);
    let temp = tempfile::tempdir().unwrap();
    let mut client = StalledThenSuccessClient::new(bytes, FirstResponse::SingleByte);

    let installed = pull_model_pack_with_client(
        &resolved,
        temp.path(),
        &mut client,
        PullOptions {
            low_speed_timeout: Duration::ZERO,
            low_speed_min_bytes: 2,
            ..PullOptions::default()
        },
        |_| {},
    )
    .unwrap();

    assert_eq!(installed.pull, "moonshine-tiny:q8");
    assert_eq!(client.ranges(), vec![None, None]);
}

#[test]
fn pull_rejects_non_https_url_before_downloading() {
    let bytes = tiny_pack_bytes();
    let mut resolved = resolved_for(&bytes);
    resolved.url = "http://127.0.0.1/model.oasr".to_string();
    let temp = tempfile::tempdir().unwrap();
    let mut client = FakeClient::default();

    let error = pull_model_pack_with_client(
        &resolved,
        temp.path(),
        &mut client,
        PullOptions::default(),
        |_| {},
    )
    .unwrap_err();

    assert!(matches!(error, PullError::NonHttpsUrl { .. }));
    assert!(client.ranges().is_empty());
}

#[test]
fn pull_rejects_path_traversal_target_before_downloading() {
    let bytes = tiny_pack_bytes();
    let mut resolved = resolved_for(&bytes);
    resolved.model_id = "../outside".to_string();
    let temp = tempfile::tempdir().unwrap();
    let mut client = FakeClient::default();

    let error = pull_model_pack_with_client(
        &resolved,
        temp.path(),
        &mut client,
        PullOptions::default(),
        |_| {},
    )
    .unwrap_err();

    assert!(matches!(
        error,
        PullError::InvalidTarget {
            field: "model_id",
            ..
        }
    ));
    assert!(client.ranges().is_empty());
    assert!(!temp.path().join("outside").exists());
}

#[cfg(unix)]
#[test]
fn pull_rejects_symlinked_model_storage_dir_before_downloading() {
    let bytes = tiny_pack_bytes();
    let resolved = resolved_for(&bytes);
    let temp = tempfile::tempdir().unwrap();
    let home = temp.path().join("home");
    let outside = temp.path().join("outside");
    fs::create_dir_all(home.join("models")).unwrap();
    fs::create_dir_all(&outside).unwrap();
    symlink(&outside, home.join("models").join("moonshine-tiny")).unwrap();
    let mut client = FakeClient::default();

    let error = pull_model_pack_with_client(
        &resolved,
        &home,
        &mut client,
        PullOptions::default(),
        |_| {},
    )
    .unwrap_err();

    assert!(matches!(error, PullError::UnsafeStoragePath { .. }));
    assert!(client.ranges().is_empty());
    assert_eq!(fs::read_dir(&outside).unwrap().count(), 0);
}

#[cfg(unix)]
#[test]
fn pull_rejects_symlinked_quant_storage_dir_before_downloading() {
    let bytes = tiny_pack_bytes();
    let resolved = resolved_for(&bytes);
    let temp = tempfile::tempdir().unwrap();
    let home = temp.path().join("home");
    let model_dir = home.join("models").join("moonshine-tiny");
    let outside = temp.path().join("outside");
    fs::create_dir_all(&model_dir).unwrap();
    fs::create_dir_all(&outside).unwrap();
    symlink(&outside, model_dir.join("q8_0")).unwrap();
    let mut client = FakeClient::default();

    let error = pull_model_pack_with_client(
        &resolved,
        &home,
        &mut client,
        PullOptions::default(),
        |_| {},
    )
    .unwrap_err();

    assert!(matches!(error, PullError::UnsafeStoragePath { .. }));
    assert!(client.ranges().is_empty());
    assert_eq!(fs::read_dir(&outside).unwrap().count(), 0);
}

#[test]
fn pull_lock_blocks_second_writer() {
    let bytes = tiny_pack_bytes();
    let resolved = resolved_for(&bytes);
    let temp = tempfile::tempdir().unwrap();
    let paths = paths_for(temp.path(), &resolved);
    fs::create_dir_all(&paths.dir).unwrap();
    fs::write(&paths.lock_path, "pid=1\n").unwrap();
    let mut client = FakeClient::default();

    let error = pull_model_pack_with_client(
        &resolved,
        temp.path(),
        &mut client,
        PullOptions::default(),
        |_| {},
    )
    .unwrap_err();

    assert!(matches!(error, PullError::LockHeld { .. }));
}

#[cfg(unix)]
#[test]
fn pull_lock_recovers_stale_lock() {
    let temp = tempfile::tempdir().unwrap();
    let lock_path = temp.path().join("model.lock");
    fs::write(&lock_path, "pid=1\n").unwrap();
    set_stale_mtime(&lock_path);

    let lock = PullLock::acquire(&lock_path).unwrap();

    assert!(lock_path.exists());
    drop(lock);
    assert!(!lock_path.exists());
}

#[cfg(unix)]
#[test]
fn pull_lock_recovers_dead_pid_lock() {
    let temp = tempfile::tempdir().unwrap();
    let lock_path = temp.path().join("model.lock");
    fs::write(&lock_path, "pid=99999999\n").unwrap();

    let lock = PullLock::acquire(&lock_path).unwrap();

    assert!(lock_path.exists());
    drop(lock);
    assert!(!lock_path.exists());
}

#[cfg(unix)]
#[test]
fn pull_lock_returns_lock_io_when_stale_lock_cannot_be_removed() {
    let temp = tempfile::tempdir().unwrap();
    let lock_path = temp.path().join("model.lock");
    fs::create_dir(&lock_path).unwrap();
    set_stale_mtime(&lock_path);

    let error = match PullLock::acquire(&lock_path) {
        Ok(_) => panic!("directory lock path should not be acquired"),
        Err(error) => error,
    };

    assert!(matches!(error, PullError::LockIo { path, .. } if path == lock_path));
    assert!(lock_path.is_dir());
}

#[test]
fn pull_rejects_corrupt_gguf_before_installing() {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"GGUF");
    bytes.extend_from_slice(&3_u32.to_le_bytes());
    bytes.extend_from_slice(&1_u64.to_le_bytes());
    bytes.extend_from_slice(&(MAX_GGUF_METADATA_ENTRIES + 1).to_le_bytes());
    let resolved = resolved_for(&bytes);
    let temp = tempfile::tempdir().unwrap();
    let mut client = FakeClient::with_responses(vec![ResponseSpec {
        status: 200,
        body: bytes,
    }]);

    let error = pull_model_pack_with_client(
        &resolved,
        temp.path(),
        &mut client,
        PullOptions::default(),
        |_| {},
    )
    .unwrap_err();

    assert!(matches!(error, PullError::GgufPreflight { .. }));
    let paths = paths_for(temp.path(), &resolved);
    assert_no_partial_or_install(&paths);
}

#[test]
fn pull_cancel_during_verify_removes_partial_without_installing() {
    let bytes = tiny_pack_bytes();
    let resolved = resolved_for(&bytes);
    let temp = tempfile::tempdir().unwrap();
    let (target, paths) = write_complete_partial(temp.path(), &resolved, &bytes);

    let error = verify_partial_and_install(
        &target,
        &paths,
        Some(DownloadedPartial {
            bytes_done: bytes.len() as u64,
            sha256: sha256_hex(&bytes),
        }),
        &|| true,
        |_| {},
    )
    .unwrap_err();

    assert!(matches!(error, PullError::Canceled { .. }));
    assert_no_partial_or_install(&paths);
    assert!(list_installed_packs(temp.path()).unwrap().is_empty());
}

#[test]
fn pull_cancel_after_verify_hash_removes_partial_without_installing() {
    let bytes = tiny_pack_bytes();
    let resolved = resolved_for(&bytes);
    let temp = tempfile::tempdir().unwrap();
    let (target, paths) = write_complete_partial(temp.path(), &resolved, &bytes);
    let cancel_calls = Cell::new(0_usize);
    let should_cancel = || {
        let next = cancel_calls.get() + 1;
        cancel_calls.set(next);
        next == 2
    };

    let error =
        verify_partial_and_install(&target, &paths, None, &should_cancel, |_| {}).unwrap_err();

    assert!(matches!(error, PullError::Canceled { .. }));
    assert_eq!(cancel_calls.get(), 2);
    assert_no_partial_or_install(&paths);
    assert!(list_installed_packs(temp.path()).unwrap().is_empty());
}

#[test]
fn pull_cancel_before_rename_removes_partial_without_installing() {
    let bytes = tiny_pack_bytes();
    let resolved = resolved_for(&bytes);
    let temp = tempfile::tempdir().unwrap();
    let (target, paths) = write_complete_partial(temp.path(), &resolved, &bytes);
    let cancel_calls = Cell::new(0_usize);
    let should_cancel = || {
        let next = cancel_calls.get() + 1;
        cancel_calls.set(next);
        next == 3
    };

    let error =
        verify_partial_and_install(&target, &paths, None, &should_cancel, |_| {}).unwrap_err();

    assert!(matches!(error, PullError::Canceled { .. }));
    assert_eq!(cancel_calls.get(), 3);
    assert_no_partial_or_install(&paths);
    assert!(list_installed_packs(temp.path()).unwrap().is_empty());
}

#[test]
fn list_installed_packs_ignores_orphaned_pack_without_record() {
    let bytes = tiny_pack_bytes();
    let resolved = resolved_for(&bytes);
    let temp = tempfile::tempdir().unwrap();
    let target = PullTarget::from_resolved(&resolved).unwrap();
    let paths = pull_paths(temp.path(), &target).unwrap();
    ensure_storage_dir_within_root(temp.path(), &paths).unwrap();
    fs::write(&paths.final_path, &bytes).unwrap();

    assert!(list_installed_packs(temp.path()).unwrap().is_empty());
}

#[test]
fn list_installed_packs_rejects_corrupt_installed_record() {
    let bytes = tiny_pack_bytes();
    let resolved = resolved_for(&bytes);
    let temp = tempfile::tempdir().unwrap();
    let target = PullTarget::from_resolved(&resolved).unwrap();
    let paths = pull_paths(temp.path(), &target).unwrap();
    ensure_storage_dir_within_root(temp.path(), &paths).unwrap();
    fs::write(&paths.final_path, &bytes).unwrap();
    fs::write(&paths.installed_meta_path, b"{").unwrap();

    let error = list_installed_packs(temp.path()).unwrap_err();

    assert!(matches!(error, PullError::ParseMeta { .. }));
}

#[test]
fn list_installed_packs_ignores_truncated_pack_with_record() {
    let bytes = tiny_pack_bytes();
    let resolved = resolved_for(&bytes);
    let temp = tempfile::tempdir().unwrap();
    let target = PullTarget::from_resolved(&resolved).unwrap();
    let paths = pull_paths(temp.path(), &target).unwrap();
    ensure_storage_dir_within_root(temp.path(), &paths).unwrap();
    fs::write(&paths.final_path, &bytes[..bytes.len() - 1]).unwrap();
    write_installed_record(&target, &paths).unwrap();

    assert!(list_installed_packs(temp.path()).unwrap().is_empty());
}

#[test]
fn pull_overwrites_truncated_pack_with_installed_record() {
    let bytes = tiny_pack_bytes();
    let resolved = resolved_for(&bytes);
    let temp = tempfile::tempdir().unwrap();
    let target = PullTarget::from_resolved(&resolved).unwrap();
    let paths = pull_paths(temp.path(), &target).unwrap();
    ensure_storage_dir_within_root(temp.path(), &paths).unwrap();
    fs::write(&paths.final_path, &bytes[..bytes.len() - 1]).unwrap();
    write_installed_record(&target, &paths).unwrap();
    let mut client = FakeClient::with_responses(vec![ResponseSpec {
        status: 200,
        body: bytes.clone(),
    }]);

    let installed = pull_model_pack_with_client(
        &resolved,
        temp.path(),
        &mut client,
        PullOptions::default(),
        |_| {},
    )
    .unwrap();

    assert_eq!(client.ranges(), vec![None]);
    assert_eq!(fs::read(installed.path).unwrap(), bytes);
    assert_eq!(list_installed_packs(temp.path()).unwrap().len(), 1);
}

/// `config.json`'s `models_dir` field must be the single thing that decides
/// where a pack lands and where `list_installed_packs` looks for it -- a
/// redirected home must land the pack entirely outside `<home>/models` and
/// still be found by the same reference.
#[test]
fn config_models_dir_redirects_pull_and_list() {
    let home = tempfile::tempdir().unwrap();
    let redirected = tempfile::tempdir().unwrap();
    crate::config::save_config(
        home.path(),
        &crate::config::OpenAsrConfig {
            models_dir: Some(redirected.path().to_path_buf()),
            ..crate::config::OpenAsrConfig::default()
        },
    )
    .unwrap();

    let bytes = tiny_pack_bytes();
    let resolved = resolved_for(&bytes);
    let mut client = FakeClient::with_responses(vec![ResponseSpec {
        status: 200,
        body: bytes.clone(),
    }]);

    let installed = pull_model_pack_with_client(
        &resolved,
        home.path(),
        &mut client,
        PullOptions::default(),
        |_| {},
    )
    .unwrap();

    assert!(
        installed.path.starts_with(redirected.path()),
        "pack should land under the redirected models_dir, got {}",
        installed.path.display()
    );
    assert!(
        !home.path().join("models").exists(),
        "the default models/ dir under home must stay untouched when models_dir is redirected"
    );

    let packs = list_installed_packs(home.path()).unwrap();
    assert_eq!(packs.len(), 1);
    assert_eq!(packs[0].pull, installed.pull);

    // OPENASR_MODELS_DIR env still wins over the config field.
    let env_redirected = tempfile::tempdir().unwrap();
    // SAFETY: test-only, single-threaded env mutation guarded by serial test
    // execution within this process is not guaranteed by cargo test, but this
    // matches the existing `OPENASR_HOME`-mutating tests elsewhere in this
    // file/crate that accept the same caveat.
    unsafe { std::env::set_var(crate::config::OPENASR_MODELS_DIR_ENV, env_redirected.path()) };
    let env_resolved = list_installed_packs(home.path()).unwrap();
    unsafe { std::env::remove_var(crate::config::OPENASR_MODELS_DIR_ENV) };
    assert!(
        env_resolved.is_empty(),
        "OPENASR_MODELS_DIR must take priority over config.models_dir"
    );
}

#[test]
fn pull_checks_available_space_before_download() {
    let bytes = tiny_pack_bytes();
    let resolved = resolved_for(&bytes);
    let temp = tempfile::tempdir().unwrap();
    let mut client = FakeClient::default();

    let error = pull_model_pack_with_client(
        &resolved,
        temp.path(),
        &mut client,
        PullOptions {
            available_space_override: Some(1),
            ..PullOptions::default()
        },
        |_| {},
    )
    .unwrap_err();

    assert!(matches!(error, PullError::InsufficientSpace { .. }));
    assert!(client.ranges().is_empty());
}

#[cfg(windows)]
#[test]
fn available_space_bytes_probes_a_real_windows_volume() {
    let temp = tempfile::tempdir().unwrap();
    let free = available_space_bytes(temp.path());
    // A live, writable temp volume must report a positive free-byte count, so the
    // pre-download space preflight is a real check on Windows and no longer the
    // `None` no-op that silently let doomed multi-GB pulls start.
    assert!(
        matches!(free, Some(bytes) if bytes > 0),
        "expected Some(>0) free bytes on a live Windows volume, got {free:?}"
    );
}

#[test]
fn remove_model_pack_ignores_installed_record_pointing_outside_home() {
    let bytes = tiny_pack_bytes();
    let resolved = resolved_for(&bytes);
    let temp = tempfile::tempdir().unwrap();
    let home = temp.path().join("home");
    let metadata_dir = home.join("models").join("moonshine-tiny").join("q8_0");
    let victim_dir = temp.path().join("victim");
    let victim_file = victim_dir.join("keep.oasr");
    fs::create_dir_all(&metadata_dir).unwrap();
    fs::create_dir_all(&victim_dir).unwrap();
    fs::write(&victim_file, b"do not delete").unwrap();

    let forged = InstalledPack {
        model_id: resolved.model_id.clone(),
        display_name: resolved.display_name.clone(),
        quant: resolved.quant.clone(),
        suffix: resolved.suffix.clone(),
        pull: resolved.pull.clone(),
        filename: resolved.filename.clone(),
        path: victim_file.clone(),
        url: resolved.url.clone(),
        hf_revision: resolved.hf_revision.clone(),
        sha256: resolved.sha256.clone(),
        size_bytes: resolved.size_bytes,
        installed_at_unix_seconds: 1,
        source: None,
    };
    let json = serde_json::to_string_pretty(&forged).unwrap();
    fs::write(metadata_dir.join("installed.json"), format!("{json}\n")).unwrap();

    let removed = remove_model_pack(&home, "moonshine-tiny:q8").unwrap();

    assert!(removed.is_none());
    assert!(victim_file.exists());
    assert!(victim_dir.exists());
    assert!(list_installed_packs(&home).unwrap().is_empty());
}

#[test]
fn remove_model_pack_deletes_installed_quant() {
    let bytes = tiny_pack_bytes();
    let resolved = resolved_for(&bytes);
    let temp = tempfile::tempdir().unwrap();
    let mut client = FakeClient::with_responses(vec![ResponseSpec {
        status: 200,
        body: bytes,
    }]);
    let installed = pull_model_pack_with_client(
        &resolved,
        temp.path(),
        &mut client,
        PullOptions::default(),
        |_| {},
    )
    .unwrap();

    let removed = remove_model_pack(temp.path(), "moonshine-tiny:q8")
        .unwrap()
        .unwrap();

    assert_eq!(removed.pull, installed.pull);
    assert!(!installed.path.exists());
    assert!(list_installed_packs(temp.path()).unwrap().is_empty());
}

#[test]
fn remove_model_pack_deletes_empty_model_dir() {
    let bytes = tiny_pack_bytes();
    let resolved = resolved_for(&bytes);
    let temp = tempfile::tempdir().unwrap();
    let mut client = FakeClient::with_responses(vec![ResponseSpec {
        status: 200,
        body: bytes,
    }]);
    let installed = pull_model_pack_with_client(
        &resolved,
        temp.path(),
        &mut client,
        PullOptions::default(),
        |_| {},
    )
    .unwrap();
    let model_dir = temp.path().join("models").join(&installed.model_id);
    assert!(
        model_dir.exists(),
        "fixture setup: model dir must exist before removal"
    );

    remove_model_pack(temp.path(), "moonshine-tiny:q8")
        .unwrap()
        .unwrap();

    // Removing the only installed quant must also clean up the now-empty
    // <models>/<model_id>/ directory, not just the <quant>/ subdirectory --
    // otherwise uninstall leaves a stale empty `models/<id>/` behind.
    assert!(
        !model_dir.exists(),
        "empty model dir must be removed once its last quant is uninstalled"
    );
}

#[test]
fn remove_model_pack_keeps_model_dir_when_sibling_quant_remains() {
    let bytes = tiny_pack_bytes();
    let resolved = resolved_for(&bytes);
    let temp = tempfile::tempdir().unwrap();
    let home = temp.path();
    let model_dir = home.join("models").join("moonshine-tiny");

    let mut client = FakeClient::with_responses(vec![ResponseSpec {
        status: 200,
        body: bytes.clone(),
    }]);
    let first =
        pull_model_pack_with_client(&resolved, home, &mut client, PullOptions::default(), |_| {})
            .unwrap();

    // A second quant for the same model, written directly so the test does
    // not need a second distinct catalog/download fixture.
    let second_quant_dir = model_dir.join("q4_k");
    fs::create_dir_all(&second_quant_dir).unwrap();
    let second_path = second_quant_dir.join("moonshine-tiny-q4_k.oasr");
    fs::write(&second_path, &bytes).unwrap();
    let second_pack = InstalledPack {
        model_id: "moonshine-tiny".to_string(),
        display_name: first.display_name.clone(),
        quant: "q4_k".to_string(),
        suffix: "q4".to_string(),
        pull: "moonshine-tiny:q4".to_string(),
        filename: "moonshine-tiny-q4_k.oasr".to_string(),
        path: second_path.clone(),
        url: first.url.clone(),
        hf_revision: first.hf_revision.clone(),
        sha256: sha256_hex(&bytes),
        size_bytes: bytes.len() as u64,
        installed_at_unix_seconds: 1,
        source: None,
    };
    let json = serde_json::to_string_pretty(&second_pack).unwrap();
    fs::write(second_quant_dir.join("installed.json"), format!("{json}\n")).unwrap();

    let removed = remove_model_pack(home, "moonshine-tiny:q8")
        .unwrap()
        .unwrap();
    assert_eq!(removed.pull, first.pull);

    // The sibling q4_k quant is a different, still-installed pack: removing
    // q8 must not touch it or the shared model dir that contains it.
    assert!(
        model_dir.exists(),
        "model dir must survive: a sibling quant is still installed"
    );
    assert!(
        second_path.exists(),
        "sibling quant file must not be touched"
    );
    let remaining = list_installed_packs(home).unwrap();
    assert_eq!(remaining.len(), 1);
    assert_eq!(remaining[0].pull, "moonshine-tiny:q4");
}

#[test]
fn resolve_installed_pack_reference_matches_quant_aliases() {
    let bytes = tiny_pack_bytes();
    let resolved = resolved_for(&bytes);
    let temp = tempfile::tempdir().unwrap();
    let mut client = FakeClient::with_responses(vec![ResponseSpec {
        status: 200,
        body: bytes,
    }]);
    let installed = pull_model_pack_with_client(
        &resolved,
        temp.path(),
        &mut client,
        PullOptions::default(),
        |_| {},
    )
    .unwrap();
    let packs = list_installed_packs(temp.path()).unwrap();

    for reference in ["moonshine-tiny:q8", "moonshine-tiny:q8_0"] {
        let resolved_pack = resolve_installed_pack_reference(&packs, reference)
            .unwrap()
            .unwrap();
        assert_eq!(resolved_pack.pull, installed.pull, "{reference}");
    }
}

#[test]
fn resolve_installed_pack_reference_rejects_invalid_model_refs() {
    for reference in ["moonshine-tiny:", "moonshine-tiny:q8:extra", ":q8"] {
        let error = resolve_installed_pack_reference(&[], reference).unwrap_err();
        assert!(
            error.to_string().contains("Invalid model reference"),
            "{reference}: {error}"
        );
    }
}

#[test]
fn resolve_installed_pack_reference_with_catalog_resolves_series_aliases() {
    let pack = installed_pack("qwen3-asr-0.6b", "q8_0", "q8", "qwen3-asr-0.6b:q8");
    let catalog = installed_pack_alias_catalog();

    for reference in ["qwen", "qwen:q8", "qwen-asr:q8_0", "qwen3-asr"] {
        let resolved_pack = resolve_installed_pack_reference_with_catalog(
            std::slice::from_ref(&pack),
            &catalog,
            reference,
        )
        .unwrap()
        .unwrap();
        assert_eq!(resolved_pack.pull, pack.pull, "{reference}");
    }
}

#[test]
fn resolve_installed_pack_reference_with_catalog_keeps_unknown_aliases_as_not_installed() {
    let catalog = installed_pack_alias_catalog();

    assert!(
        resolve_installed_pack_reference_with_catalog(&[], &catalog, "not-a-model")
            .unwrap()
            .is_none()
    );
}

#[test]
fn remove_model_pack_deletes_installed_quant_by_canonical_quant_alias() {
    let bytes = tiny_pack_bytes();
    let resolved = resolved_for(&bytes);
    let temp = tempfile::tempdir().unwrap();
    let mut client = FakeClient::with_responses(vec![ResponseSpec {
        status: 200,
        body: bytes,
    }]);
    let installed = pull_model_pack_with_client(
        &resolved,
        temp.path(),
        &mut client,
        PullOptions::default(),
        |_| {},
    )
    .unwrap();

    let removed = remove_model_pack(temp.path(), "moonshine-tiny:q8_0")
        .unwrap()
        .unwrap();

    assert_eq!(removed.pull, installed.pull);
    assert!(!installed.path.exists());
    assert!(list_installed_packs(temp.path()).unwrap().is_empty());
}

fn installed_pack(model_id: &str, quant: &str, suffix: &str, pull: &str) -> InstalledPack {
    InstalledPack {
        model_id: model_id.to_string(),
        display_name: model_id.to_string(),
        quant: quant.to_string(),
        suffix: suffix.to_string(),
        pull: pull.to_string(),
        filename: format!("{model_id}-{quant}.oasr"),
        path: Path::new("/tmp").join(format!("{model_id}-{quant}.oasr")),
        url: "https://example.test/model.oasr".to_string(),
        hf_revision: "0123456789abcdef0123456789abcdef01234567".to_string(),
        sha256: "a".repeat(64),
        size_bytes: 1024,
        installed_at_unix_seconds: 1,
        source: None,
    }
}

fn installed_pack_alias_catalog() -> ModelCatalog {
    ModelCatalog {
        schema_version: 1,
        generated_at: "2026-06-04T00:00:00Z".to_string(),
        catalog_url: "fixture".to_string(),
        backends: Vec::new(),
        language_labels: std::collections::BTreeMap::new(),
        models: vec![CatalogModel {
            id: "qwen3-asr-0.6b".to_string(),
            kind: CatalogModelKind::AsrModel,
            capability: None,
            experimental: false,
            display_name: "Qwen3-ASR 0.6B".to_string(),
            family: "qwen".to_string(),
            aliases: vec!["qwen3".to_string(), "qwen3-asr".to_string()],
            pull_alias: Some("qwen3".to_string()),
            size: "0.6b".to_string(),
            languages: vec!["en".to_string(), "zh".to_string()],
            language_mode: None,
            language_default: None,
            source_langs: Vec::new(),
            target_langs: Vec::new(),
            vendor: Some("Qwen".to_string()),
            license: "Apache-2.0".to_string(),
            license_url: "https://example.test/license".to_string(),
            license_class: LicenseClass::Permissive,
            hf_repo: "OpenASR/qwen3-asr-0.6b".to_string(),
            hf_revision: "0123456789abcdef0123456789abcdef01234567".to_string(),
            public: true,
            min_cli_version: "0.1.0".to_string(),
            min_core_version: None,
            recommended_quant: "q8_0".to_string(),
            pull_recommended: "qwen3-asr-0.6b:q8".to_string(),
            sort_weight: 0,
            recommended: false,
            upstream_release_date: None,
            emits_punctuation: None,
            prose: None,
            prose_locales: None,
            quants: vec![CatalogQuant {
                quant: "q8_0".to_string(),
                suffix: "q8".to_string(),
                pull: "qwen3-asr-0.6b:q8".to_string(),
                filename: "qwen3-asr-0.6b-q8_0.oasr".to_string(),
                url: "https://example.test/qwen3-asr-0.6b-q8_0.oasr".to_string(),
                mirrors: Vec::new(),
                sha256: "a".repeat(64),
                size_bytes: 1024,
                recommended: true,
                perf: None,
            }],
        }],
    }
}

#[test]
fn lock_with_live_owner_pid_is_not_treated_as_stale() {
    let dir = tempfile::tempdir().unwrap();
    let lock = dir.path().join("pack.oasr.lock");
    fs::write(&lock, format!("pid={}\n", std::process::id())).unwrap();
    // A lock owned by THIS (live) process must never be reclaimed — doing so would
    // let a second pull stomp an in-progress download.
    assert!(!lock_owner_is_gone(&lock));
    assert!(!lock_is_stale(&lock));
}

#[test]
fn lock_with_dead_owner_pid_is_stale_regardless_of_mtime() {
    // Spawn a process, reap it, then reuse its now-freed pid as the lock owner.
    // A crashed/killed download leaves exactly this: a lock whose owning pid is
    // gone but whose mtime is fresh. The owner-liveness probe must mark it stale
    // so the next pull reclaims it and resumes, instead of erroring with LockHeld
    // until the 6h mtime timeout elapses.
    #[cfg(windows)]
    let mut child = std::process::Command::new("cmd")
        .args(["/C", "exit"])
        .spawn()
        .unwrap();
    #[cfg(not(windows))]
    let mut child = std::process::Command::new("sh")
        .args(["-c", "exit 0"])
        .spawn()
        .unwrap();
    let dead_pid = child.id();
    child.wait().unwrap();

    let dir = tempfile::tempdir().unwrap();
    let lock = dir.path().join("pack.oasr.lock");
    fs::write(&lock, format!("pid={dead_pid}\n")).unwrap();
    assert!(lock_owner_is_gone(&lock));
    assert!(lock_is_stale(&lock));
}

// ---- backend-pack file preflight (PE/ELF/Mach-O/zip magic) ----

fn write_preflight_fixture(name: &str, bytes: &[u8]) -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join(name);
    fs::write(&path, bytes).unwrap();
    (dir, path)
}

/// Minimal valid PE head: "MZ", e_lfanew=0x40, "PE\0\0" at 0x40.
fn minimal_pe_bytes() -> Vec<u8> {
    let mut bytes = vec![0u8; 0x44];
    bytes[0] = b'M';
    bytes[1] = b'Z';
    bytes[0x3C] = 0x40; // e_lfanew (LE)
    bytes[0x40] = b'P';
    bytes[0x41] = b'E';
    bytes
}

#[test]
fn preflight_backend_file_accepts_pe_library() {
    let (_dir, path) = write_preflight_fixture("ggml-cuda.dll", &minimal_pe_bytes());
    preflight_backend_file(&path, BackendFileFormat::NativeLibrary).unwrap();
}

#[test]
fn preflight_backend_file_accepts_elf_library() {
    let mut bytes = vec![0x7F, b'E', b'L', b'F'];
    bytes.extend_from_slice(&[0u8; 60]);
    let (_dir, path) = write_preflight_fixture("libggml-cuda.so", &bytes);
    preflight_backend_file(&path, BackendFileFormat::NativeLibrary).unwrap();
}

#[test]
fn preflight_backend_file_accepts_macho_library() {
    // MH_MAGIC_64 little-endian.
    let mut bytes = vec![0xCF, 0xFA, 0xED, 0xFE];
    bytes.extend_from_slice(&[0u8; 60]);
    let (_dir, path) = write_preflight_fixture("libggml-metal.dylib", &bytes);
    preflight_backend_file(&path, BackendFileFormat::NativeLibrary).unwrap();
}

#[test]
fn preflight_backend_file_accepts_zip_archive() {
    let (_dir, path) = write_preflight_fixture("rocblas-library.zip", b"PK\x03\x04and the rest");
    preflight_backend_file(&path, BackendFileFormat::ZipArchive).unwrap();
}

#[test]
fn preflight_backend_file_rejects_html_error_page_as_library() {
    let (_dir, path) = write_preflight_fixture(
        "ggml-cuda.dll",
        b"<!DOCTYPE html><title>404 Not Found</title>",
    );
    let error = preflight_backend_file(&path, BackendFileFormat::NativeLibrary).unwrap_err();
    assert!(matches!(error, PullError::BackendFilePreflight { .. }));
}

#[test]
fn preflight_backend_file_rejects_library_served_as_archive() {
    let (_dir, path) = write_preflight_fixture("mislabeled.zip", &minimal_pe_bytes());
    let error = preflight_backend_file(&path, BackendFileFormat::ZipArchive).unwrap_err();
    assert!(matches!(error, PullError::BackendFilePreflight { .. }));
}

#[test]
fn preflight_backend_file_rejects_mz_stub_without_pe_signature() {
    // "MZ" present but no "PE\0\0" at e_lfanew — a DOS stub, not a real DLL.
    let mut bytes = vec![0u8; 0x44];
    bytes[0] = b'M';
    bytes[1] = b'Z';
    bytes[0x3C] = 0x40;
    let (_dir, path) = write_preflight_fixture("fake.dll", &bytes);
    let error = preflight_backend_file(&path, BackendFileFormat::NativeLibrary).unwrap_err();
    assert!(matches!(error, PullError::BackendFilePreflight { .. }));
}

// ---- backend-pack install orchestration (download -> verify -> preflight -> extract) ----

fn tensile_zip_bytes() -> Vec<u8> {
    let mut buf = Vec::new();
    {
        let mut writer = zip::ZipWriter::new(Cursor::new(&mut buf));
        writer
            .start_file(
                "Kernels.so-000-gfx1200.hsaco",
                zip::write::FileOptions::default(),
            )
            .unwrap();
        writer.write_all(b"fake tensile kernel object").unwrap();
        writer.finish().unwrap();
    }
    buf
}

fn hip_pack_resolved(plugin: &[u8], archive: &[u8]) -> ResolvedCatalogBackendPull {
    ResolvedCatalogBackendPull {
        backend_id: "hip-radeon".to_string(),
        vendor: CatalogBackendVendor::Hip,
        version: "0.13.1".to_string(),
        display_name: "AMD ROCm".to_string(),
        files: vec![
            CatalogBackendFile {
                filename: "ggml-hip.dll".to_string(),
                url: "https://example.test/ggml-hip.dll".to_string(),
                mirrors: Vec::new(),
                sha256: sha256_hex(plugin),
                size_bytes: plugin.len() as u64,
                role: CatalogBackendFileRole::Plugin,
                extract_subdir: None,
            },
            CatalogBackendFile {
                filename: "rocblas-library.zip".to_string(),
                url: "https://example.test/rocblas-library.zip".to_string(),
                mirrors: Vec::new(),
                sha256: sha256_hex(archive),
                size_bytes: archive.len() as u64,
                role: CatalogBackendFileRole::Archive,
                extract_subdir: Some("rocblas/library".to_string()),
            },
        ],
    }
}

#[test]
fn install_backend_pack_downloads_verifies_and_extracts() {
    let home = tempfile::tempdir().unwrap();
    let plugin = minimal_pe_bytes();
    let archive = tensile_zip_bytes();
    let resolved = hip_pack_resolved(&plugin, &archive);
    let mut client = FakeClient::with_responses(vec![
        ResponseSpec {
            status: 200,
            body: plugin.clone(),
        },
        ResponseSpec {
            status: 200,
            body: archive.clone(),
        },
    ]);

    let installed =
        install_backend_pack_with_client(&resolved, home.path(), &mut client, |_| {}).unwrap();

    let dir = home.path().join("backends").join("hip").join("0.13.1");
    assert_eq!(installed.dir, dir);
    assert_eq!(installed.plugin_filename, "ggml-hip.dll");
    assert!(dir.join("ggml-hip.dll").is_file());
    assert!(dir.join("rocblas-library.zip").is_file());
    // archive extracted into extract_subdir (zip-slip-safe)
    assert!(
        dir.join("rocblas")
            .join("library")
            .join("Kernels.so-000-gfx1200.hsaco")
            .is_file()
    );
    assert!(dir.join("backend.json").is_file());

    // Idempotent: a re-install short-circuits without downloading (an empty
    // response queue would panic in FakeClient::open if it tried).
    let mut empty = FakeClient::with_responses(Vec::new());
    let again =
        install_backend_pack_with_client(&resolved, home.path(), &mut empty, |_| {}).unwrap();
    assert_eq!(again.backend_id, "hip-radeon");
}

#[test]
fn install_backend_pack_rejects_sha_mismatch() {
    let home = tempfile::tempdir().unwrap();
    let plugin = minimal_pe_bytes();
    let mut resolved = hip_pack_resolved(&plugin, &tensile_zip_bytes());
    resolved.files.truncate(1); // plugin only
    resolved.files[0].sha256 = "f".repeat(64); // wrong hash
    let mut client = FakeClient::with_responses(vec![ResponseSpec {
        status: 200,
        body: plugin,
    }]);
    let error =
        install_backend_pack_with_client(&resolved, home.path(), &mut client, |_| {}).unwrap_err();
    assert!(matches!(error, PullError::ShaMismatch { .. }));
}

#[test]
fn install_backend_pack_rejects_unsafe_version_segment() {
    let home = tempfile::tempdir().unwrap();
    let plugin = minimal_pe_bytes();
    let mut resolved = hip_pack_resolved(&plugin, &tensile_zip_bytes());
    resolved.version = "../escape".to_string();
    let mut client = FakeClient::with_responses(Vec::new());
    let error =
        install_backend_pack_with_client(&resolved, home.path(), &mut client, |_| {}).unwrap_err();
    assert!(matches!(
        error,
        PullError::InvalidTarget {
            field: "backend.version",
            ..
        }
    ));
}

/// A reader that yields the first `remaining` bytes of `inner` and then
/// fails with a plain (non-timeout) I/O error, simulating a dropped
/// connection mid-body -- distinct from `TimedOutReader`'s stall, but the
/// same "retryable, should resume" class per `is_retryable_download_error`.
struct DropAfterBytesReader {
    inner: Cursor<Vec<u8>>,
    remaining: usize,
}

impl Read for DropAfterBytesReader {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if self.remaining == 0 {
            return Err(io::Error::new(
                io::ErrorKind::ConnectionReset,
                "simulated mid-stream connection drop",
            ));
        }
        let cap = buf.len().min(self.remaining);
        let read = self.inner.read(&mut buf[..cap])?;
        self.remaining -= read;
        Ok(read)
    }
}

/// First `open()` call drops the connection after `split` bytes; every
/// subsequent call serves the remainder as a proper Range (206) response,
/// so a real resume (not a from-scratch restart) is what makes the transfer
/// finish -- this is what `download_backend_file`'s retry loop must do.
struct BackendMidStreamDropThenResumeClient {
    bytes: Vec<u8>,
    split: usize,
    attempts: usize,
    ranges: Vec<Option<u64>>,
}

impl BackendMidStreamDropThenResumeClient {
    fn new(bytes: Vec<u8>, split: usize) -> Self {
        Self {
            bytes,
            split,
            attempts: 0,
            ranges: Vec::new(),
        }
    }

    fn ranges(&self) -> Vec<Option<u64>> {
        self.ranges.clone()
    }
}

impl DownloadClient for BackendMidStreamDropThenResumeClient {
    fn open(
        &mut self,
        _url: &str,
        range: Option<ByteRange>,
    ) -> Result<DownloadResponse, PullError> {
        let range_start = range.map(|range| range.start);
        self.ranges.push(range_start);
        self.attempts += 1;
        if self.attempts == 1 {
            return Ok(DownloadResponse {
                status: 200,
                content_length: Some(self.bytes.len() as u64),
                content_range: None,
                etag: Some("etag-test".to_string()),
                reader: Box::new(DropAfterBytesReader {
                    inner: Cursor::new(self.bytes.clone()),
                    remaining: self.split,
                }),
            });
        }
        let start = range_start.unwrap_or(0) as usize;
        let total = self.bytes.len() as u64;
        let body = self.bytes[start..].to_vec();
        Ok(DownloadResponse {
            status: if start > 0 { 206 } else { 200 },
            content_length: Some(body.len() as u64),
            content_range: if start > 0 {
                Some(format!("bytes {start}-{}/{total}", total - 1))
            } else {
                None
            },
            etag: Some("etag-test".to_string()),
            reader: Box::new(Cursor::new(body)),
        })
    }
}

#[test]
fn install_backend_pack_retries_stalled_read_and_succeeds() {
    let home = tempfile::tempdir().unwrap();
    let plugin = minimal_pe_bytes();
    let mut resolved = hip_pack_resolved(&plugin, &tensile_zip_bytes());
    resolved.files.truncate(1); // plugin only
    let mut client = StalledThenSuccessClient::new(plugin.clone(), FirstResponse::Timeout);

    let installed =
        install_backend_pack_with_client(&resolved, home.path(), &mut client, |_| {}).unwrap();

    let dir = home.path().join("backends").join("hip").join("0.13.1");
    assert_eq!(installed.dir, dir);
    assert!(dir.join("ggml-hip.dll").is_file());
    assert_eq!(fs::read(dir.join("ggml-hip.dll")).unwrap(), plugin);
}

#[test]
fn install_backend_pack_resumes_after_mid_stream_drop_and_retries() {
    let home = tempfile::tempdir().unwrap();
    // Long enough that a partial prefix is meaningfully smaller than the
    // whole file (the minimal PE fixture is only 0x44 bytes). Starts with
    // the ELF magic so `preflight_backend_file` accepts it as a native
    // library after the (fake) content is written.
    let mut plugin: Vec<u8> = vec![0x7F, b'E', b'L', b'F'];
    plugin.extend((0_u32..2000).map(|value| (value % 251) as u8));
    let mut resolved = hip_pack_resolved(&plugin, &tensile_zip_bytes());
    resolved.files.truncate(1); // plugin only
    resolved.files[0].filename = "libbackend.so".to_string();
    resolved.files[0].sha256 = sha256_hex(&plugin);
    resolved.files[0].size_bytes = plugin.len() as u64;
    let mut client = BackendMidStreamDropThenResumeClient::new(plugin.clone(), 700);

    let installed =
        install_backend_pack_with_client(&resolved, home.path(), &mut client, |_| {}).unwrap();

    let dir = home.path().join("backends").join("hip").join("0.13.1");
    assert_eq!(installed.dir, dir);
    assert_eq!(fs::read(dir.join("libbackend.so")).unwrap(), plugin);
    // Second attempt must have asked for a Range starting at the byte the
    // dropped connection had already delivered -- a from-scratch restart
    // would instead show `[None, None]`.
    assert_eq!(client.ranges(), vec![None, Some(700)]);
    assert!(!dir.join("libbackend.so.partial").exists());
}

#[cfg(windows)]
#[test]
fn windows_in_use_os_errors_classify_as_model_in_use() {
    // ERROR_SHARING_VIOLATION (32) and ERROR_USER_MAPPED_FILE (1224) mean the
    // file can't be replaced because it is open/mapped — treat as "in use".
    assert!(is_file_in_use_error(&io::Error::from_raw_os_error(32)));
    assert!(is_file_in_use_error(&io::Error::from_raw_os_error(1224)));
    // ERROR_FILE_NOT_FOUND (2) and ERROR_ACCESS_DENIED (5, ambiguous) are not.
    assert!(!is_file_in_use_error(&io::Error::from_raw_os_error(2)));
    assert!(!is_file_in_use_error(&io::Error::from_raw_os_error(5)));
    assert!(!is_file_in_use_error(&io::Error::other("x")));
}

#[cfg(windows)]
#[test]
fn remove_existing_final_pack_reports_model_in_use_for_held_handle() {
    use std::os::windows::fs::OpenOptionsExt;
    const FILE_SHARE_READ: u32 = 0x0000_0001;

    let temp = tempfile::tempdir().unwrap();
    let home = temp.path();
    let resolved = resolved_for(&tiny_pack_bytes());
    let paths = paths_for(home, &resolved);
    ensure_storage_dir_within_root(home, &paths).unwrap();
    fs::write(&paths.final_path, b"model").unwrap();

    // Open allowing only read sharing (no FILE_SHARE_DELETE), so Windows rejects
    // the delete with ERROR_SHARING_VIOLATION — the same failure family as a
    // model mmap'd for inference (ERROR_USER_MAPPED_FILE). std's File::open
    // defaults to FILE_SHARE_DELETE, which would let the delete succeed.
    let _held = std::fs::OpenOptions::new()
        .read(true)
        .share_mode(FILE_SHARE_READ)
        .open(&paths.final_path)
        .unwrap();

    let error = remove_existing_final_pack(&paths).unwrap_err();
    assert!(
        matches!(error, PullError::ModelInUse { .. }),
        "expected ModelInUse, got {error:?}"
    );
}

// -- Concurrent chunked-download tests -------------------------------------
//
// These exercise `download_parallel_attempt` and friends via
// `pull_model_pack_with_client_parallel`, using `RangeServerClient` (a
// range-aware mock that serves any byte range from an in-memory buffer,
// independent of request order) plus a small `parallel_segment_bytes_override`
// so multi-segment behavior is exercised against tiny fixtures.

/// Build a probe-client clone (for the caller's primary `client: &mut C`)
/// plus a boxed worker-client factory, both backed by clones of `server`.
/// Every clone shares the same `Arc`-backed state (bytes, ETag sequence,
/// call counter, request log), so assertions against `server` after the
/// pull see everything every worker thread (and the probe) did. Returns a
/// concrete `Box<dyn Fn>` (rather than `-> impl Fn`) purely to sidestep
/// edition-2024 RPIT lifetime-capture rules for a helper that borrows
/// `server` only to clone it.
fn parallel_probe_and_factory(
    server: &RangeServerClient,
) -> (
    RangeServerClient,
    Box<dyn Fn() -> Result<BoxedDownloadClient, PullError>>,
) {
    let probe_client = server.clone();
    let factory_server = server.clone();
    let factory: Box<dyn Fn() -> Result<BoxedDownloadClient, PullError>> =
        Box::new(move || Ok(Box::new(factory_server.clone()) as BoxedDownloadClient));
    (probe_client, factory)
}

#[test]
fn parallel_download_splits_into_segments_and_reassembles_correctly() {
    let bytes = tiny_pack_bytes();
    let resolved = resolved_for(&bytes);
    let temp = tempfile::tempdir().unwrap();
    let segment_bytes = small_segment_bytes(bytes.len(), 5);
    let total_segments = segment_count(bytes.len() as u64, segment_bytes);
    assert!(
        total_segments >= 2,
        "fixture too small to exercise chunking"
    );

    let server = RangeServerClient::new(bytes.clone());
    let (mut probe_client, factory) = parallel_probe_and_factory(&server);
    let parallel = ParallelDownloadConfig {
        connections: 4,
        factory: &*factory,
    };

    let installed = pull_model_pack_with_client_parallel(
        &resolved,
        temp.path(),
        &mut probe_client,
        parallel_test_options(segment_bytes),
        parallel,
        |_| {},
        || false,
        || false,
    )
    .unwrap();

    assert_eq!(installed.pull, "moonshine-tiny:q8");
    let paths = paths_for(temp.path(), &resolved);
    assert_eq!(fs::read(&paths.final_path).unwrap(), bytes);
    assert!(!paths.partial_path.exists());
    assert!(!paths.partial_segments_meta_path.exists());
    assert_eq!(server.call_count(), total_segments);
    // Every request is a genuinely bounded Range (has an explicit `end`),
    // confirming this went through the chunked path, not a bare open-ended
    // sequential fetch that happened to succeed.
    for (_, end) in server.requests() {
        assert!(end.is_some());
    }
}

#[test]
fn parallel_download_falls_back_to_sequential_when_source_ignores_range() {
    let bytes = tiny_pack_bytes();
    let resolved = resolved_for(&bytes);
    let temp = tempfile::tempdir().unwrap();
    let segment_bytes = small_segment_bytes(bytes.len(), 4);

    let server = RangeServerClient::new(bytes.clone()).without_range_support();
    let (mut probe_client, factory) = parallel_probe_and_factory(&server);
    let parallel = ParallelDownloadConfig {
        connections: 4,
        factory: &*factory,
    };

    let installed = pull_model_pack_with_client_parallel(
        &resolved,
        temp.path(),
        &mut probe_client,
        parallel_test_options(segment_bytes),
        parallel,
        |_| {},
        || false,
        || false,
    )
    .unwrap();

    assert_eq!(installed.pull, "moonshine-tiny:q8");
    let paths = paths_for(temp.path(), &resolved);
    assert_eq!(fs::read(&paths.final_path).unwrap(), bytes);
    assert!(!paths.partial_segments_meta_path.exists());
    // One wasted probe (200, ignored) plus one real single-stream fetch.
    assert_eq!(server.call_count(), 2);
}

#[test]
fn parallel_download_restarts_whole_download_when_etag_changes_mid_download() {
    let bytes = tiny_pack_bytes();
    let resolved = resolved_for(&bytes);
    let temp = tempfile::tempdir().unwrap();
    let segment_bytes = small_segment_bytes(bytes.len(), 4);
    let total_segments = segment_count(bytes.len() as u64, segment_bytes);
    assert!(total_segments >= 2, "fixture too small for this ETag test");

    // Call 0 (the synchronous probe) gets "etag-a"; every later call (every
    // worker's first segment fetch) is clamped to the sequence's last entry,
    // "etag-b" -- deterministically regardless of thread scheduling. So
    // attempt 1 always: probes with "etag-a", then every worker sees
    // "etag-b" and fails with `EtagChanged`, wiping the partial. Attempt 2's
    // probe then itself gets "etag-b" (still the last entry) and every
    // following call keeps matching it, so the retry succeeds cleanly.
    let server = RangeServerClient::new(bytes.clone()).with_etag_sequence(&["etag-a", "etag-b"]);
    let (mut probe_client, factory) = parallel_probe_and_factory(&server);
    let parallel = ParallelDownloadConfig {
        connections: 4,
        factory: &*factory,
    };

    let installed = pull_model_pack_with_client_parallel(
        &resolved,
        temp.path(),
        &mut probe_client,
        parallel_test_options(segment_bytes),
        parallel,
        |_| {},
        || false,
        || false,
    )
    .unwrap();

    assert_eq!(installed.pull, "moonshine-tiny:q8");
    let paths = paths_for(temp.path(), &resolved);
    assert_eq!(fs::read(&paths.final_path).unwrap(), bytes);
    assert!(!paths.partial_path.exists());
    assert!(!paths.partial_segments_meta_path.exists());
    // At least one full failed attempt (whose segment fetches all errored)
    // plus one fully successful attempt happened.
    assert!(server.call_count() > total_segments);
}

#[test]
fn parallel_download_resumes_from_segment_bitmap_without_refetching_done_segments() {
    let bytes = tiny_pack_bytes();
    let resolved = resolved_for(&bytes);
    let temp = tempfile::tempdir().unwrap();
    let target = PullTarget::from_resolved(&resolved).unwrap();
    let paths = pull_paths(temp.path(), &target).unwrap();
    ensure_storage_dir_within_root(temp.path(), &paths).unwrap();
    let segment_bytes = small_segment_bytes(bytes.len(), 4);
    let total_segments = segment_count(bytes.len() as u64, segment_bytes);
    assert!(
        total_segments >= 3,
        "fixture too small for this bitmap test"
    );

    // Pre-seed the on-disk state exactly as a prior (interrupted) attempt
    // would leave it after segment 0 completed: a full-size `.partial` file
    // with segment 0's window already correct (the rest is unwritten/zero)
    // and a segment bitmap marking only index 0 done.
    let (seg0_start, seg0_end) = segment_range(0, bytes.len() as u64, segment_bytes);
    let mut partial_content = vec![0_u8; bytes.len()];
    partial_content[seg0_start as usize..=seg0_end as usize]
        .copy_from_slice(&bytes[seg0_start as usize..=seg0_end as usize]);
    fs::write(&paths.partial_path, &partial_content).unwrap();
    let mut segments_done = vec![false; total_segments];
    segments_done[0] = true;
    let meta = SegmentedPartialMeta {
        format: PARALLEL_META_FORMAT.to_string(),
        model_id: target.model_id.clone(),
        quant: target.quant.clone(),
        filename: target.filename.clone(),
        hf_revision: target.hf_revision.clone(),
        sha256: target.sha256.clone(),
        size_bytes: target.size_bytes,
        segment_bytes,
        etag: Some("etag-a".to_string()),
        segments_done,
        updated_at_unix_seconds: 0,
    };
    write_partial_segments_meta(&paths.partial_segments_meta_path, &meta).unwrap();

    let server = RangeServerClient::new(bytes.clone()).with_etag_sequence(&["etag-a"]);
    let (mut probe_client, factory) = parallel_probe_and_factory(&server);
    let parallel = ParallelDownloadConfig {
        connections: 4,
        factory: &*factory,
    };

    let installed = pull_model_pack_with_client_parallel(
        &resolved,
        temp.path(),
        &mut probe_client,
        parallel_test_options(segment_bytes),
        parallel,
        |_| {},
        || false,
        || false,
    )
    .unwrap();

    assert_eq!(installed.pull, "moonshine-tiny:q8");
    assert_eq!(fs::read(&paths.final_path).unwrap(), bytes);
    // Segment 0's byte range is never requested again.
    for (start, _) in server.requests() {
        assert_ne!(
            start, seg0_start,
            "already-completed segment 0 should not be refetched"
        );
    }
    assert_eq!(server.call_count(), total_segments - 1);
}

#[test]
fn parallel_download_cancel_deletes_partial_and_allows_clean_restart() {
    let bytes = tiny_pack_bytes();
    let resolved = resolved_for(&bytes);
    let temp = tempfile::tempdir().unwrap();
    let segment_bytes = small_segment_bytes(bytes.len(), 4);

    let server = RangeServerClient::new(bytes.clone());
    let (mut probe_client, factory) = parallel_probe_and_factory(&server);
    let parallel = ParallelDownloadConfig {
        connections: 4,
        factory: &*factory,
    };
    let cancel = Arc::new(AtomicBool::new(false));
    let cancel_on_progress = cancel.clone();

    let error = pull_model_pack_with_client_parallel(
        &resolved,
        temp.path(),
        &mut probe_client,
        parallel_test_options(segment_bytes),
        parallel,
        move |event| {
            if matches!(event, PullProgress::Downloading { .. }) {
                cancel_on_progress.store(true, Ordering::SeqCst);
            }
        },
        move || cancel.load(Ordering::SeqCst),
        || false,
    )
    .unwrap_err();

    assert!(matches!(error, PullError::Canceled { .. }));
    let paths = paths_for(temp.path(), &resolved);
    assert!(!paths.partial_path.exists());
    assert!(!paths.partial_segments_meta_path.exists());
    assert!(!paths.final_path.exists());

    // A fresh, uncanceled pull afterward succeeds cleanly.
    let server2 = RangeServerClient::new(bytes.clone());
    let (mut probe_client2, factory2) = parallel_probe_and_factory(&server2);
    let parallel2 = ParallelDownloadConfig {
        connections: 4,
        factory: &*factory2,
    };
    let installed = pull_model_pack_with_client_parallel(
        &resolved,
        temp.path(),
        &mut probe_client2,
        parallel_test_options(segment_bytes),
        parallel2,
        |_| {},
        || false,
        || false,
    )
    .unwrap();
    assert_eq!(installed.pull, "moonshine-tiny:q8");
    assert_eq!(fs::read(&paths.final_path).unwrap(), bytes);
}

/// Reader that trips a shared cancel flag on its first `read` and records how
/// many times it is read, so a test can prove the chunked probe segment stops
/// reading the moment the pull is canceled instead of streaming the whole (up
/// to 64 MiB) probe segment first.
struct ProbeCancelReader {
    inner: Cursor<Vec<u8>>,
    reads: Arc<AtomicUsize>,
    cancel: Arc<AtomicBool>,
}

impl Read for ProbeCancelReader {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.reads.fetch_add(1, Ordering::SeqCst);
        self.cancel.store(true, Ordering::SeqCst);
        self.inner.read(buf)
    }
}

/// Serves every bounded Range request with a `ProbeCancelReader`, so the very
/// first byte of the synchronous probe segment cancels the pull.
#[derive(Clone)]
struct ProbeCancelClient {
    bytes: Arc<Vec<u8>>,
    reads: Arc<AtomicUsize>,
    cancel: Arc<AtomicBool>,
}

impl DownloadClient for ProbeCancelClient {
    fn open(
        &mut self,
        _url: &str,
        range: Option<ByteRange>,
    ) -> Result<DownloadResponse, PullError> {
        let total = self.bytes.len() as u64;
        let range = range.expect("parallel probe issues a bounded range");
        let end = range
            .end
            .unwrap_or(total.saturating_sub(1))
            .min(total.saturating_sub(1));
        let start = range.start.min(end);
        let slice = self.bytes[start as usize..=end as usize].to_vec();
        Ok(DownloadResponse {
            status: 206,
            content_length: Some(slice.len() as u64),
            content_range: Some(format!("bytes {start}-{end}/{total}")),
            etag: Some("etag-a".to_string()),
            reader: Box::new(ProbeCancelReader {
                inner: Cursor::new(slice),
                reads: self.reads.clone(),
                cancel: self.cancel.clone(),
            }),
        })
    }
}

#[test]
fn parallel_download_cancel_during_probe_stops_without_reading_whole_segment() {
    // A probe segment larger than one `DOWNLOAD_BUFFER_BYTES` (64 KiB) read:
    // if the probe write ignored cancellation it would read the segment in
    // several chunks before noticing, so `reads > 1` would betray the old
    // "download the whole probe segment first" behavior. Bytes are arbitrary
    // (never verified/installed: the pull is canceled first).
    let segment_bytes = 96 * 1024_u64;
    let bytes = vec![7_u8; (segment_bytes as usize) * 3 + 17];
    let resolved = resolved_for(&bytes);
    assert!(
        parallel_download_eligible(
            &PullTarget::from_resolved(&resolved).unwrap(),
            4,
            segment_bytes,
        ),
        "fixture must exercise the chunked path"
    );
    let temp = tempfile::tempdir().unwrap();

    let cancel = Arc::new(AtomicBool::new(false));
    let reads = Arc::new(AtomicUsize::new(0));
    let mut probe_client = ProbeCancelClient {
        bytes: Arc::new(bytes.clone()),
        reads: reads.clone(),
        cancel: cancel.clone(),
    };
    let factory_client = probe_client.clone();
    let factory: Box<dyn Fn() -> Result<BoxedDownloadClient, PullError>> =
        Box::new(move || Ok(Box::new(factory_client.clone()) as BoxedDownloadClient));
    let parallel = ParallelDownloadConfig {
        connections: 4,
        factory: &*factory,
    };

    let cancel_predicate = cancel.clone();
    let error = pull_model_pack_with_client_parallel(
        &resolved,
        temp.path(),
        &mut probe_client,
        parallel_test_options(segment_bytes),
        parallel,
        |_| {},
        move || cancel_predicate.load(Ordering::SeqCst),
        || false,
    )
    .unwrap_err();

    assert!(matches!(error, PullError::Canceled { .. }));
    // The probe write must abort after the first read that tripped the cancel,
    // never streaming the rest of the 96 KiB probe segment.
    assert_eq!(
        reads.load(Ordering::SeqCst),
        1,
        "probe segment kept reading after cancellation was requested"
    );
    let paths = paths_for(temp.path(), &resolved);
    assert!(!paths.partial_path.exists());
    assert!(!paths.partial_segments_meta_path.exists());
    assert!(!paths.final_path.exists());
}

#[test]
fn parallel_download_sha_mismatch_deletes_partial() {
    let bytes = tiny_pack_bytes();
    let resolved = resolved_for(&bytes);
    let temp = tempfile::tempdir().unwrap();
    let segment_bytes = small_segment_bytes(bytes.len(), 4);
    let mut corrupted = bytes.clone();
    let mid = corrupted.len() / 2;
    corrupted[mid] ^= 0x01;

    // The source serves bit-flipped bytes, but `resolved` is still pinned to
    // the original (correct) sha256/size -- every segment fetch and the
    // per-segment content-range/ETag checks succeed, so only the final
    // full-file re-hash catches this.
    let server = RangeServerClient::new(corrupted);
    let (mut probe_client, factory) = parallel_probe_and_factory(&server);
    let parallel = ParallelDownloadConfig {
        connections: 4,
        factory: &*factory,
    };

    let error = pull_model_pack_with_client_parallel(
        &resolved,
        temp.path(),
        &mut probe_client,
        parallel_test_options(segment_bytes),
        parallel,
        |_| {},
        || false,
        || false,
    )
    .unwrap_err();

    assert!(matches!(error, PullError::ShaMismatch { .. }));
    let paths = paths_for(temp.path(), &resolved);
    assert!(!paths.partial_path.exists());
    assert!(!paths.partial_segments_meta_path.exists());
    assert!(!paths.final_path.exists());
}

/// Regression guard: `reqwest::blocking::ClientBuilder::timeout` defaults to
/// `Some(Duration::from_secs(30))` even when `.timeout()` is never called
/// (see `Timeout::default()` in reqwest's blocking client), and it caps
/// connect + send + the ENTIRE response body read as a single deadline that
/// keeps ticking while the body streams -- not an idle/stall timeout. A
/// prior version of `blocking_client_no_redirect` passed
/// `HTTP_STALL_TIMEOUT` (30s) straight into `.timeout(...)`, so any download
/// whose wall-clock time exceeded 30 seconds -- every real multi-hundred-MB
/// model pack on any non-trivial connection -- was silently killed
/// regardless of active progress, before the low-speed/stall detection ever
/// got a chance to run. This drives a real socket that dribbles the body
/// slowly over > 30 seconds through the exact client constructor
/// `HttpDownloadClient::new` uses, and asserts the transfer still completes.
#[test]
fn download_client_does_not_kill_a_slow_but_steadily_progressing_transfer() {
    let body: Vec<u8> = (0_u8..32).cycle().take(32 * 16).collect();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let server_body = body.clone();
    let server = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        // Drain the request line and headers up to the blank-line terminator;
        // this test never inspects it (a bare GET is all `reqwest` sends).
        let mut request = Vec::new();
        let mut buf = [0_u8; 4096];
        loop {
            let read = stream.read(&mut buf).unwrap();
            request.extend_from_slice(&buf[..read]);
            if read == 0 || request.windows(4).any(|window| window == b"\r\n\r\n") {
                break;
            }
        }
        let header = format!(
            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            server_body.len()
        );
        stream.write_all(header.as_bytes()).unwrap();
        stream.flush().unwrap();
        // 32 chunks of 16 bytes, paced 1 second apart (31 sleeps): a >= 31s
        // transfer, comfortably past the historical 30s bug boundary, while
        // keeping the data volume itself trivial.
        let chunk_size = 16;
        for (index, chunk) in server_body.chunks(chunk_size).enumerate() {
            if index > 0 {
                std::thread::sleep(Duration::from_secs(1));
            }
            stream.write_all(chunk).unwrap();
            stream.flush().unwrap();
        }
    });

    let client = http::blocking_client_no_redirect(HTTP_CONNECT_TIMEOUT).unwrap();
    let started = Instant::now();
    let mut response = client
        .get(format!("http://{addr}/slow-file"))
        .send()
        .unwrap();
    let mut received = Vec::new();
    response.read_to_end(&mut received).unwrap();
    let elapsed = started.elapsed();
    server.join().unwrap();

    assert_eq!(received, body);
    assert!(
        elapsed >= Duration::from_secs(30),
        "expected the transfer to genuinely take >= 30s (was {elapsed:?}); a shorter \
         elapsed time here means this test stopped exercising the historical 30s \
         total-timeout bug"
    );
}
