use std::{
    cell::Cell,
    collections::{HashMap, VecDeque},
    fs,
    io::{self, Cursor, Read, Write},
    net::TcpListener,
    path::{Path, PathBuf},
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

#[test]
fn backend_protected_bytes_count_every_protected_tree() {
    let temp = tempfile::tempdir().unwrap();
    let pack = temp.path().join("pack");
    let shared = temp.path().join("shared");
    fs::create_dir_all(pack.join("nested")).unwrap();
    fs::create_dir_all(&shared).unwrap();
    fs::write(pack.join("plugin.dll"), [0_u8; 11]).unwrap();
    fs::write(pack.join("nested").join("backend.json"), [0_u8; 7]).unwrap();
    fs::write(shared.join("runtime.dll"), [0_u8; 13]).unwrap();

    assert_eq!(protected_backend_roots_bytes([pack, shared]).unwrap(), 31);
}

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
    // This helper backs the Moonshine catalog/pull fixtures, so its proven
    // route must be Moonshine rather than a Whisper-shaped pack carrying a
    // Moonshine model id. This runtime-ready fixture is complete for the
    // install-time family contract and its tiny tensor skeleton keeps the
    // pull tests independent of a downloaded production model.
    let spec = TinyGgufFixtureSpec::moonshine_oasr_v1_runtime_ready("moonshine-tiny");
    write_tiny_gguf_runtime_source(&path, &spec).unwrap();
    fs::read(path).unwrap()
}

#[test]
fn model_pack_preflight_receipt_describes_the_exact_verified_bytes() {
    let bytes = tiny_pack_bytes();
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("receipt-source.oasr");
    fs::write(&path, &bytes).unwrap();

    let receipt = preflight_model_pack_with_receipt(&path).unwrap();

    assert_eq!(receipt.schema, "openasr.model-pack-preflight.v1");
    assert_eq!(receipt.content_id, format!("sha256:{}", sha256_hex(&bytes)));
    assert_eq!(receipt.size_bytes, bytes.len() as u64);
    assert_eq!(receipt.route, "asr");
    assert_eq!(receipt.catalog_family_id, "moonshine");
    assert_eq!(receipt.model_family.as_deref(), Some("moonshine"));
    assert_eq!(receipt.model_architecture, "moonshine-encoder-decoder");
    assert_eq!(receipt.build_commit, None);
}

fn tiny_redimnet_pack_bytes() -> Vec<u8> {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("redimnet2-tiny.oasr");
    let tensor = crate::ggml_runtime::GgufWriteTensor {
        name: "fixture.tensor".to_string(),
        dims: vec![1],
        tensor_type: crate::ggml_runtime::GgufWriteTensorType::F32,
        data: 0.0_f32.to_le_bytes().to_vec(),
    };
    crate::models::oasr_metadata::OasrPackWriter::write(
        &path,
        crate::models::oasr_metadata::PackEnvelope::aux(
            crate::models::aux_pack_registry::REDIMNET2_GGML_ARCHITECTURE_ID,
        ),
        std::collections::BTreeMap::new(),
        &[tensor],
    )
    .unwrap();
    fs::read(path).unwrap()
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn resolved_for(bytes: &[u8]) -> ResolvedCatalogPull {
    ResolvedCatalogPull {
        requested: "moonshine-tiny:q8".to_string(),
        model_id: "moonshine-tiny".to_string(),
        catalog_family_id: "moonshine".to_string(),
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
            family: resolved.catalog_family_id.clone(),
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
            speaker_source: None,
            word_timestamp_source: None,
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
    model.family = "redimnet2".to_string();
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
fn resolve_catalog_model_pack_from_path_exposes_catalog_policy_before_install() {
    let bytes = tiny_pack_bytes();
    let mut resolved = resolved_for(&bytes);
    resolved.license_class = LicenseClass::Noncommercial;
    resolved.license = "CC-BY-NC-4.0".to_string();
    resolved.license_url = "https://example.invalid/license".to_string();
    let catalog = catalog_for_resolved(&resolved);
    let temp = tempfile::tempdir().unwrap();
    let source_path = temp.path().join("moonshine-tiny-q8_0.oasr");
    fs::write(&source_path, bytes).unwrap();

    let observed = resolve_catalog_model_pack_from_path(&catalog, &source_path).unwrap();

    assert_eq!(observed.pull, resolved.pull);
    assert_eq!(observed.license_class, LicenseClass::Noncommercial);
    assert_eq!(observed.license_url, resolved.license_url);
    assert!(
        list_installed_packs(temp.path()).unwrap().is_empty(),
        "license preflight must not install the local pack"
    );
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

#[test]
fn install_catalog_model_pack_rejects_a_catalog_family_mismatch() {
    let bytes = tiny_pack_bytes();
    let mut resolved = resolved_for(&bytes);
    // The bytes prove Moonshine, while the signed catalog target claims a
    // different family. Digest and size alone are insufficient authority for
    // a runtime install; the verified route must bind to the catalog family.
    resolved.catalog_family_id = "whisper".to_string();
    let catalog = catalog_for_resolved(&resolved);
    let temp = tempfile::tempdir().unwrap();
    let source_path = temp.path().join("moonshine-tiny-q8_0.oasr");
    let digest = sha256_hex(&bytes);
    fs::write(&source_path, bytes).unwrap();

    let error = install_catalog_model_pack_from_path(&catalog, &source_path, temp.path(), |_| {})
        .expect_err("catalog family mismatch must fail closed");

    let message = error.to_string();
    assert!(
        message.contains("catalog family") && message.contains("whisper"),
        "error must identify the mismatched catalog family: {message}"
    );
    assert!(matches!(error, PullError::RuntimeValidation { .. }));
    assert!(
        list_installed_packs(temp.path()).unwrap().is_empty(),
        "a mismatched family must not publish an installed reference"
    );
    assert!(
        !object_path_for(&temp.path().join("models"), &digest).exists(),
        "a mismatched family must not publish a content object"
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

    let mut resolved = resolved_for(&bytes);
    resolved.catalog_family_id = "whisper".to_string();
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
    let bytes = tiny_redimnet_pack_bytes();
    let mut resolved = resolved_for(&bytes);
    resolved.requested = "redimnet2-b6-cn:fp16".to_string();
    resolved.model_id = "redimnet2-b6-cn".to_string();
    resolved.display_name = "ReDimNet2-B6 Speaker Embedder".to_string();
    resolved.quant = "fp16".to_string();
    resolved.suffix = "fp16".to_string();
    resolved.pull = "redimnet2-b6-cn:fp16".to_string();
    resolved.filename = "redimnet2-b6-cn-fp16.oasr".to_string();
    resolved.url = "https://huggingface.co/OpenASR/redimnet2-b6-cn/resolve/0123456789abcdef0123456789abcdef01234567/redimnet2-b6-cn-fp16.oasr".to_string();
    let catalog = capability_pack_catalog_for_resolved(&resolved);

    let from_catalog = resolve_catalog_pull(
        &catalog,
        &CatalogPullRequest {
            reference: "redimnet2-b6-cn:fp16".to_string(),
            quant: None,
            size: None,
        },
    )
    .unwrap();
    assert_eq!(from_catalog.pull, "redimnet2-b6-cn:fp16");

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
    assert_eq!(pulled.pull, "redimnet2-b6-cn:fp16");

    let import_home = tempfile::tempdir().unwrap();
    let source_path = import_home.path().join("redimnet2-b6-cn-fp16.oasr");
    fs::write(&source_path, bytes).unwrap();
    let imported =
        install_catalog_model_pack_from_path(&catalog, &source_path, import_home.path(), |_| {})
            .unwrap();
    assert_eq!(imported.pull, "redimnet2-b6-cn:fp16");
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
fn pull_does_not_fallback_after_gguf_preflight_failure() {
    // The downloaded bytes matched the catalog sha256 (every source in the
    // chain serves those exact bytes), so a GGUF preflight failure is a
    // verdict on the pack itself -- retrying the remaining mirrors would
    // just re-download identical content and reach the same verdict. The
    // pull must fail terminally after the first source.
    let garbage = b"this is not a GGUF-backed .oasr pack".to_vec();
    let resolved = resolved_for(&garbage);
    let temp = tempfile::tempdir().unwrap();
    let mut client = FakeClient::with_responses(vec![
        ResponseSpec {
            status: 200,
            body: garbage.clone(),
        },
        ResponseSpec {
            status: 200,
            body: garbage,
        },
    ]);

    let error = pull_model_pack_with_client_sources_and_cancel(
        &resolved,
        temp.path(),
        &mut client,
        PullOptions::default(),
        &[DownloadSource::Hf, DownloadSource::HfMirror],
        None,
        None,
        |_| {},
        || false,
        || false,
    )
    .unwrap_err();

    assert!(matches!(error, PullError::GgufPreflight { .. }));
    assert_eq!(client.urls(), vec![resolved.url.clone()]);
    let paths = paths_for(temp.path(), &resolved);
    assert!(!paths.final_path.exists());
    assert!(!paths.partial_path.exists());
}

#[test]
fn pull_does_not_fallback_after_runtime_validation_failure() {
    // Same terminal-error contract as the GGUF preflight case, for the
    // runtime-contract half of `preflight_model_pack_for_install`: a pack
    // whose GGUF structure is valid but whose runtime metadata is missing
    // `whisper.decoder.attention.head_count` fails validation on every
    // source identically (sha256-identical bytes), so the chain must not
    // burn the remaining mirrors on it.
    let fixture_dir = tempfile::tempdir().unwrap();
    let mut spec = TinyGgufFixtureSpec::whisper_oasr_v1_encoder_graph_one_layer("moonshine-tiny");
    spec.metadata
        .remove("whisper.decoder.attention.head_count")
        .expect("fixture must set the key this test removes");
    let broken_path = fixture_dir.path().join("broken-source.oasr");
    write_tiny_gguf_runtime_source(&broken_path, &spec).unwrap();
    let bytes = fs::read(&broken_path).unwrap();

    let mut resolved = resolved_for(&bytes);
    resolved.catalog_family_id = "whisper".to_string();
    let temp = tempfile::tempdir().unwrap();
    let mut client = FakeClient::with_responses(vec![
        ResponseSpec {
            status: 200,
            body: bytes.clone(),
        },
        ResponseSpec {
            status: 200,
            body: bytes,
        },
    ]);

    let error = pull_model_pack_with_client_sources_and_cancel(
        &resolved,
        temp.path(),
        &mut client,
        PullOptions::default(),
        &[DownloadSource::Hf, DownloadSource::HfMirror],
        None,
        None,
        |_| {},
        || false,
        || false,
    )
    .unwrap_err();

    assert!(matches!(error, PullError::RuntimeValidation { .. }));
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

    let error =
        verify_partial_and_install(&target, &paths, None, None, &|| false, |_| {}).unwrap_err();

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

    verify_partial_and_install(&target, &paths, None, None, &|| false, |_| {}).unwrap();

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

/// Default `PullOptions` low-speed knobs, but with a zero window timeout so
/// `SegmentLowSpeedWindow::observe` judges its window on the very first call
/// regardless of real elapsed time (matches the same trick the whole-file
/// `pull_retries_low_speed_body_and_restarts_safely` test above uses). Safe
/// here specifically because `observe` compares raw per-window *byte counts*,
/// never a computed bytes/sec rate, so there is no division-by-elapsed-time
/// to make unstable.
fn zero_window_options() -> PullOptions {
    PullOptions {
        segment_low_speed_timeout: Duration::ZERO,
        ..PullOptions::default()
    }
}

/// Feeds `samples` into a reference (each one both consulted as history and
/// then recorded), so a test can cheaply establish "this session has already
/// seen these per-window byte counts" before evaluating the sample under
/// test in isolation.
fn reference_with_history(samples: &[u64]) -> SegmentThroughputReference {
    let reference = SegmentThroughputReference::new();
    for sample in samples {
        reference.record(*sample);
    }
    reference
}

#[test]
fn segment_throughput_reference_is_cold_below_the_minimum_sample_count() {
    let reference = SegmentThroughputReference::new();
    assert_eq!(reference.median(), None, "no samples yet: cold start");
    reference.record(1_000_000);
    reference.record(1_000_000);
    assert_eq!(
        reference.median(),
        None,
        "still below SEGMENT_LOW_SPEED_MIN_REFERENCE_SAMPLES"
    );
    reference.record(1_000_000);
    assert_eq!(
        reference.median(),
        Some(1_000_000),
        "the Nth sample must cross the cold-start threshold"
    );
}

#[test]
fn segment_low_speed_window_never_trips_during_cold_start() {
    // Fewer than `SEGMENT_LOW_SPEED_MIN_REFERENCE_SAMPLES` samples exist, so
    // there is no reference yet to be an outlier against -- even a session's
    // very first, objectively tiny window must never be judged low-speed.
    let reference = reference_with_history(&[1_000_000, 1_000_000]);
    let cooldown_slot = Mutex::new(None);
    let options = zero_window_options();
    let mut window = SegmentLowSpeedWindow::new(&options, &reference, &cooldown_slot, false);
    assert!(
        !window.observe(1),
        "cold start (fewer than the minimum reference samples) must never trip"
    );
}

#[test]
fn segment_low_speed_window_trips_on_a_relative_outlier_matching_the_reported_case() {
    // Models the reported real-world failure: the bulk of the download ran
    // fast (here, a 10 MB/window reference -- comparable to a healthy several
    // MB/s connection), then a lone tail segment lands on a degraded
    // connection running at a small fraction of that.
    let reference = reference_with_history(&[10_000_000, 10_000_000, 10_000_000, 10_000_000]);
    let cooldown_slot = Mutex::new(None);
    let options = zero_window_options();
    let mut window = SegmentLowSpeedWindow::new(&options, &reference, &cooldown_slot, false);
    // Well under 15% of the 10 MB reference AND under the absolute floor.
    assert!(
        window.observe(1_000_000),
        "1 MB against a 10 MB reference is both a relative outlier and below \
         the absolute floor -- must trip"
    );
}

#[test]
fn segment_low_speed_window_never_trips_in_a_uniformly_slow_session() {
    // Every window in this session reads about the same (a real, working,
    // if modest ~200 KB/s-class connection): no segment is an outlier
    // relative to its own siblings, so the ratio test alone must never fire,
    // regardless of how small the absolute numbers are. This is the "慢但能成"
    // contract: a uniformly slow network must never be turned into a
    // deterministic failure by this guard.
    let uniform_window_bytes = 200_000_u64;
    let reference = reference_with_history(&[
        uniform_window_bytes,
        uniform_window_bytes,
        uniform_window_bytes,
        uniform_window_bytes,
    ]);
    let cooldown_slot = Mutex::new(None);
    let options = zero_window_options();
    let mut window = SegmentLowSpeedWindow::new(&options, &reference, &cooldown_slot, false);
    assert!(
        !window.observe(uniform_window_bytes),
        "a segment performing exactly like its siblings must never be an outlier"
    );
}

#[test]
fn segment_low_speed_window_absolute_floor_protects_a_merely_modest_speed_in_a_fast_session() {
    // A segment reading ~400 KB/s-equivalent worth of bytes in one window is
    // a real, working connection -- just not this unusually fast session's
    // best. The ratio alone would flag it (well under 15% of a very high
    // reference), but the absolute floor must still protect it from being
    // needlessly abandoned.
    let fast_reference_window_bytes = 150_000_000_u64; // ~10 MB/s-class
    let reference = reference_with_history(&[
        fast_reference_window_bytes,
        fast_reference_window_bytes,
        fast_reference_window_bytes,
    ]);
    let modest_but_real_window_bytes = 6_000_000_u64; // ~400 KB/s over the 15s window
    assert!(
        modest_but_real_window_bytes > SEGMENT_LOW_SPEED_ABSOLUTE_FLOOR_BYTES,
        "fixture must sit above the absolute floor to exercise its protection"
    );
    let cooldown_slot = Mutex::new(None);
    let options = zero_window_options();
    let mut window = SegmentLowSpeedWindow::new(&options, &reference, &cooldown_slot, false);
    assert!(
        !window.observe(modest_but_real_window_bytes),
        "above the absolute floor must never trip, no matter how fast the reference is"
    );
}

#[test]
fn segment_low_speed_window_reflects_the_sessions_historical_median_for_a_lone_tail_segment() {
    // Once every other segment has completed, the reference is entirely this
    // session's *history* (nothing else is in flight to compare against) --
    // exactly the "one straggler segment left" case from the report. This
    // test only asserts that history alone (no live siblings) is sufficient
    // for the guard to keep working, by feeding a fast history and then
    // evaluating a slow lone window against it.
    let reference = reference_with_history(&[8_000_000, 8_000_000, 8_000_000, 8_000_000]);
    let cooldown_slot = Mutex::new(None);
    let options = zero_window_options();
    let mut window = SegmentLowSpeedWindow::new(&options, &reference, &cooldown_slot, false);
    assert!(
        window.observe(90_000),
        "a lone straggler window judged purely against session history must still trip"
    );
}

#[test]
fn segment_low_speed_window_cooldown_suppresses_a_thrashing_retrip() {
    // Hysteresis: once tripped, the same segment index must not immediately
    // re-trip on its very next window even if that window is also still an
    // outlier -- this damps requeue churn for a segment sitting right at the
    // boundary instead of burning through reconnects one window apart.
    let reference = reference_with_history(&[10_000_000, 10_000_000, 10_000_000, 10_000_000]);
    let cooldown_slot = Mutex::new(None);
    let options = zero_window_options();
    let mut first = SegmentLowSpeedWindow::new(&options, &reference, &cooldown_slot, false);
    assert!(first.observe(1_000_000), "first window: a genuine outlier");
    let mut second = SegmentLowSpeedWindow::new(&options, &reference, &cooldown_slot, false);
    assert!(
        !second.observe(1_000_000),
        "still within the cooldown since the first trip: must be suppressed"
    );
}

#[test]
fn segment_low_speed_window_disabled_flag_never_trips_regardless_of_speed() {
    // Once a segment has exhausted its reconnect budget, evaluation is
    // disabled for its final attempt: the worst case must degrade to
    // "let it finish", never a hard failure, no matter how extreme the
    // outlier looks.
    let reference = reference_with_history(&[10_000_000, 10_000_000, 10_000_000, 10_000_000]);
    let cooldown_slot = Mutex::new(None);
    let options = zero_window_options();
    let mut window = SegmentLowSpeedWindow::new(&options, &reference, &cooldown_slot, true);
    assert!(
        !window.observe(1),
        "disabled must never trip, even for a single byte against a 10 MB reference"
    );
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
    fs::write(&paths.lock_path, format!("pid={}\n", std::process::id())).unwrap();
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
    bytes.extend_from_slice(
        &(crate::ggml_runtime::MAX_RUNTIME_GGUF_METADATA_ENTRIES + 1).to_le_bytes(),
    );
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
        None,
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

    let error = verify_partial_and_install(&target, &paths, None, None, &should_cancel, |_| {})
        .unwrap_err();

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

    let error = verify_partial_and_install(&target, &paths, None, None, &should_cancel, |_| {})
        .unwrap_err();

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

    let store = crate::InstalledModelStore::read(temp.path()).unwrap();

    assert!(store.packs().is_empty());
    assert_eq!(store.diagnostics().len(), 1);
    assert_eq!(store.diagnostics()[0].path, paths.installed_meta_path);
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
fn pull_rejects_truncated_immutable_object_without_replacing_it() {
    let bytes = tiny_pack_bytes();
    let resolved = resolved_for(&bytes);
    let temp = tempfile::tempdir().unwrap();
    let target = PullTarget::from_resolved(&resolved).unwrap();
    let paths = pull_paths(temp.path(), &target).unwrap();
    ensure_storage_dir_within_root(temp.path(), &paths).unwrap();
    let corrupt = bytes[..bytes.len() - 1].to_vec();
    fs::write(&paths.final_path, &corrupt).unwrap();
    write_installed_record(&target, &paths).unwrap();
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

    assert!(matches!(error, PullError::ContentStore(_)));
    assert_eq!(fs::read(&paths.final_path).unwrap(), corrupt);
}

/// Land bytes at the object path a pull would download to, sealed or not,
/// exactly the two states `installed_matches` must tell apart.
fn seed_final_object(paths: &PullPaths, bytes: &[u8], read_only: bool) {
    if let Some(parent) = paths.final_path.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    fs::write(&paths.final_path, bytes).unwrap();
    let mut permissions = fs::metadata(&paths.final_path).unwrap().permissions();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        permissions.set_mode(if read_only { 0o444 } else { 0o644 });
    }
    #[cfg(not(unix))]
    permissions.set_readonly(read_only);
    fs::set_permissions(&paths.final_path, permissions).unwrap();
}

/// The digest half of the download-skip verdict for a sealed object must be
/// answered from the anchored path without re-hashing gigabytes. The verifier
/// still reads the bounded GGUF header and family contract before accepting
/// the object. Pinned by construction: bytes that do *not* hash to the catalog
/// digest can only pass the identity half if the digest came from the sealed
/// object path, never from a full-file hash.
#[test]
fn installed_matches_trusts_a_sealed_object_without_rehashing() {
    let bytes = tiny_pack_bytes();
    let mut resolved = resolved_for(&bytes);
    resolved.sha256 = "ab".repeat(32);
    assert_ne!(
        sha256_hex(&bytes),
        resolved.sha256,
        "the fixture must not accidentally hash to the named digest"
    );
    let temp = tempfile::tempdir().unwrap();
    let target = PullTarget::from_resolved(&resolved).unwrap();
    let paths = pull_paths(temp.path(), &target).unwrap();
    seed_final_object(&paths, &bytes, true);

    assert!(installed_matches(&target, &paths).unwrap());
}

#[test]
fn installed_matches_rejects_a_sealed_object_that_fails_the_family_contract() {
    let fixture_dir = tempfile::tempdir().unwrap();
    let mut spec = TinyGgufFixtureSpec::whisper_oasr_v1_encoder_graph_one_layer("broken-installed");
    spec.metadata
        .remove("whisper.decoder.attention.head_count")
        .unwrap();
    let source = fixture_dir.path().join("broken-installed.oasr");
    write_tiny_gguf_runtime_source(&source, &spec).unwrap();
    let bytes = fs::read(source).unwrap();
    let mut resolved = resolved_for(&bytes);
    resolved.catalog_family_id = "whisper".to_string();
    let temp = tempfile::tempdir().unwrap();
    let target = PullTarget::from_resolved(&resolved).unwrap();
    let paths = pull_paths(temp.path(), &target).unwrap();
    seed_final_object(&paths, &bytes, true);

    let error = installed_matches(&target, &paths).unwrap_err();

    assert!(matches!(error, PullError::RuntimeValidation { .. }));
}

/// The fail-closed half: the seal gone, the same mismatched object goes back
/// through a full hash, and because nothing pins its bytes to the catalog
/// digest it must not match.
#[test]
fn installed_matches_unsealed_object_falls_back_to_hashing() {
    let bytes = tiny_pack_bytes();
    let mut resolved = resolved_for(&bytes);
    resolved.sha256 = "cd".repeat(32);
    let temp = tempfile::tempdir().unwrap();
    let target = PullTarget::from_resolved(&resolved).unwrap();
    let paths = pull_paths(temp.path(), &target).unwrap();
    seed_final_object(&paths, &bytes, false);

    assert!(!installed_matches(&target, &paths).unwrap());
}

/// The fallback still accepts: an unsealed object whose bytes really do hash
/// to the catalog digest (a store whose seals a backup restore stripped)
/// matches through the hashing path and skips the download.
#[test]
fn installed_matches_unsealed_object_still_matches_on_honest_hash() {
    let bytes = tiny_pack_bytes();
    let resolved = resolved_for(&bytes);
    let temp = tempfile::tempdir().unwrap();
    let target = PullTarget::from_resolved(&resolved).unwrap();
    let paths = pull_paths(temp.path(), &target).unwrap();
    seed_final_object(&paths, &bytes, false);

    assert!(installed_matches(&target, &paths).unwrap());
}

/// Size is the O(1) first gate: a mismatch is rejected on the stat alone,
/// before either the path digest or a hash is consulted.
#[test]
fn installed_matches_rejects_a_size_mismatch_on_the_stat_alone() {
    let bytes = tiny_pack_bytes();
    let mut resolved = resolved_for(&bytes);
    resolved.size_bytes += 1;
    let temp = tempfile::tempdir().unwrap();
    let target = PullTarget::from_resolved(&resolved).unwrap();
    let paths = pull_paths(temp.path(), &target).unwrap();
    seed_final_object(&paths, &bytes, true);

    assert!(!installed_matches(&target, &paths).unwrap());
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
    let env_resolved = crate::test_process_env::with_test_process_env(
        [(
            crate::config::OPENASR_MODELS_DIR_ENV,
            Some(env_redirected.path().as_os_str().to_os_string()),
        )],
        || list_installed_packs(home.path()).unwrap(),
    );
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

    // A second quant of the same model, published as a ref against the very
    // same object. Deduplication makes this the interesting case: removing one
    // ref must not collect content the surviving ref still names.
    let object = first.path.clone();
    let ref_dir = home.join("models").join("refs").join("moonshine-tiny");
    let second_ref = ref_dir.join("q4_k.json");
    let second_pack = InstalledPack {
        model_id: "moonshine-tiny".to_string(),
        display_name: first.display_name.clone(),
        quant: "q4_k".to_string(),
        suffix: "q4".to_string(),
        pull: "moonshine-tiny:q4".to_string(),
        filename: "moonshine-tiny-q4_k.oasr".to_string(),
        path: object.clone(),
        url: first.url.clone(),
        hf_revision: first.hf_revision.clone(),
        sha256: sha256_hex(&bytes),
        size_bytes: bytes.len() as u64,
        installed_at_unix_seconds: 1,
        source: None,
    };
    let json = serde_json::to_string_pretty(&second_pack).unwrap();
    fs::write(&second_ref, format!("{json}\n")).unwrap();

    let removed = remove_model_pack(home, "moonshine-tiny:q8")
        .unwrap()
        .unwrap();
    assert_eq!(removed.pull, first.pull);

    assert!(second_ref.is_file(), "sibling quant ref must survive");
    assert!(
        ref_dir.exists(),
        "the model's ref directory must survive while a quant remains"
    );
    assert!(
        object.is_file(),
        "shared content must survive while another ref names it"
    );
    assert!(
        !model_dir.exists(),
        "no legacy per-quant tree is created by an install"
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
            speaker_source: None,
            word_timestamp_source: None,
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
    let extracted_tree_sha256 = materialized_tree_sha256(&[InstalledBackendMaterializedFile {
        relative_path: "rocblas/library/Kernels.so-000-gfx1200.hsaco".to_string(),
        sha256: sha256_hex(b"fake tensile kernel object"),
        size_bytes: b"fake tensile kernel object".len() as u64,
    }]);
    ResolvedCatalogBackendPull {
        backend_id: "hip-radeon".to_string(),
        vendor: CatalogBackendVendor::Hip,
        version: "0.13.1".to_string(),
        display_name: "AMD ROCm".to_string(),
        host_abi: crate::backend_distribution::BackendHostAbi::current(),
        targets: vec!["gfx1200".to_string()],
        min_driver_api: Some("7.1.0".to_string()),
        activation: crate::CatalogBackendActivation {
            state: crate::CatalogBackendActivationState::Activated,
            qualification_source_catalog_sha256: Some("1".repeat(64)),
            hardware_evidence_sha256: Some("2".repeat(64)),
            qualified_device_target: Some("gfx1200".to_string()),
            qualified_driver_version: Some("7.1.0".to_string()),
            correctness_matrix_sha256: Some("3".repeat(64)),
            correctness_receipts_sha256: Some("4".repeat(64)),
        },
        files: vec![
            CatalogBackendFile {
                filename: "ggml-hip.dll".to_string(),
                url: "https://example.test/ggml-hip.dll".to_string(),
                mirrors: Vec::new(),
                sha256: sha256_hex(plugin),
                size_bytes: plugin.len() as u64,
                role: CatalogBackendFileRole::Plugin,
                extract_subdir: None,
                extracted_tree_sha256: None,
            },
            CatalogBackendFile {
                filename: "rocblas-library.zip".to_string(),
                url: "https://example.test/rocblas-library.zip".to_string(),
                mirrors: Vec::new(),
                sha256: sha256_hex(archive),
                size_bytes: archive.len() as u64,
                role: CatalogBackendFileRole::Archive,
                extract_subdir: Some("rocblas/library".to_string()),
                extracted_tree_sha256: Some(extracted_tree_sha256),
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

    let dir = backend_pack_install_dir(home.path(), &resolved).unwrap();
    assert_eq!(installed.dir, dir);
    assert_eq!(installed.plugin_filename, "ggml-hip.dll");
    assert!(dir.join("ggml-hip.dll").is_file());
    assert!(!dir.join("rocblas-library.zip").exists());
    assert!(
        backend_content_object_dir(home.path(), &resolved.files[1])
            .join("object.json")
            .is_file()
    );
    // archive extracted into extract_subdir (zip-slip-safe)
    assert!(
        dir.join("rocblas")
            .join("library")
            .join("Kernels.so-000-gfx1200.hsaco")
            .is_file()
    );
    assert!(dir.join("backend.json").is_file());
    read_and_verify_installed_backend(&dir, &resolved).unwrap();

    // Idempotent: a re-install short-circuits without downloading (an empty
    // response queue would panic in FakeClient::open if it tried).
    let mut empty = FakeClient::with_responses(Vec::new());
    let again =
        install_backend_pack_with_client(&resolved, home.path(), &mut empty, |_| {}).unwrap();
    assert_eq!(again.backend_id, "hip-radeon");

    let plan = backend_pack_download_plan(home.path(), &resolved).unwrap();
    assert_eq!(plan.total_bytes, (plugin.len() + archive.len()) as u64);
    assert_eq!(plan.plugin_bytes, plugin.len() as u64);
    assert_eq!(plan.vendor_bytes, archive.len() as u64);
    assert_eq!(plan.required_download_bytes, 0);
    assert_eq!(plan.required_plugin_bytes, 0);
    assert_eq!(plan.required_vendor_bytes, 0);
}

#[cfg(target_os = "windows")]
struct LocalBackendArtifactClient {
    by_url: HashMap<String, PathBuf>,
}

#[cfg(target_os = "windows")]
impl DownloadClient for LocalBackendArtifactClient {
    fn open(&mut self, url: &str, range: Option<ByteRange>) -> Result<DownloadResponse, PullError> {
        let path = self
            .by_url
            .get(url)
            .unwrap_or_else(|| panic!("no local backend artifact mapped for {url}"));
        let mut bytes = fs::read(path).unwrap_or_else(|error| {
            panic!(
                "could not read local backend artifact '{}': {error}",
                path.display()
            )
        });
        let total = bytes.len() as u64;
        let start = range.map_or(0, |range| range.start);
        assert!(start <= total, "range start {start} exceeds {total}");
        if start > 0 {
            bytes.drain(..start as usize);
        }
        let status = if start == 0 { 200 } else { 206 };
        Ok(DownloadResponse {
            status,
            content_length: Some(bytes.len() as u64),
            content_range: (start > 0).then(|| format!("bytes {start}-{}/{total}", total - 1)),
            etag: Some(format!("local-{total}")),
            reader: Box::new(Cursor::new(bytes)),
        })
    }
}

/// Real Windows pack smoke for release rehearsal. This remains ignored unless
/// the caller deliberately supplies a dev-signed catalog, its artifact
/// directory, and an empty isolated OpenASR home. It exercises the same
/// install, full-file verification, extraction, live ABI/target/driver probe,
/// and atomic activation pointer as production without publishing local build
/// artifacts just to make them downloadable over public HTTPS.
#[cfg(target_os = "windows")]
#[test]
#[ignore = "requires a real locally built Windows CUDA backend pack"]
fn windows_real_cuda_backend_pack_installs_and_activates() {
    let catalog_path = PathBuf::from(
        std::env::var("OPENASR_WINDOWS_GPU_PLUGIN_E2E_CATALOG")
            .expect("OPENASR_WINDOWS_GPU_PLUGIN_E2E_CATALOG is required"),
    );
    let artifact_dir = PathBuf::from(
        std::env::var("OPENASR_WINDOWS_GPU_PLUGIN_E2E_ARTIFACT_DIR")
            .expect("OPENASR_WINDOWS_GPU_PLUGIN_E2E_ARTIFACT_DIR is required"),
    );
    let home = PathBuf::from(
        std::env::var("OPENASR_WINDOWS_GPU_PLUGIN_E2E_HOME")
            .expect("OPENASR_WINDOWS_GPU_PLUGIN_E2E_HOME is required"),
    );
    fs::create_dir_all(&home).unwrap();
    assert!(
        fs::read_dir(&home).unwrap().next().is_none(),
        "real backend smoke requires an empty isolated home: {}",
        home.display()
    );

    let signature_path = catalog_path.with_file_name(crate::CATALOG_SIGNATURE_FILE_NAME);
    let signature: crate::CatalogSignatureManifest =
        serde_json::from_str(&fs::read_to_string(&signature_path).unwrap()).unwrap();
    let catalog =
        crate::load_local_catalog_file_with_identity(&catalog_path, &signature.catalog_url, &home)
            .unwrap();
    let resolved = crate::resolve_catalog_backend_pull_for_host(
        &catalog,
        CatalogBackendVendor::Cuda,
        &crate::BackendHostAbi::current(),
    )
    .unwrap();
    assert_eq!(resolved.targets, ["sm_86"]);

    let mut client = LocalBackendArtifactClient {
        by_url: resolved
            .files
            .iter()
            .map(|file| {
                let path = artifact_dir.join(&file.filename);
                assert!(path.is_file(), "missing local artifact: {}", path.display());
                (file.url.clone(), path)
            })
            .collect(),
    };
    let installed =
        install_backend_pack_with_client(&resolved, &home, &mut client, |_| {}).unwrap();
    read_and_verify_installed_backend(&installed.dir, &resolved).unwrap();

    let activated =
        crate::activate_installed_backend_pack_auto(&catalog, &resolved.backend_id, &home).unwrap();
    assert_eq!(activated.vendor, CatalogBackendVendor::Cuda);
    assert_eq!(activated.device_target, "sm_86");
    assert!(!activated.driver_version.is_empty());
    assert_eq!(
        crate::backend_plugin_status(&home)
            .unwrap()
            .activated
            .as_ref(),
        Some(&activated)
    );

    // Prove the durable pointer is consumable by the actual neutral-host
    // runtime, not merely well-formed JSON.  This specifically locks the
    // catalog-source contract: Desktop/CLI may provide catalog bytes and their
    // signed identity as separate values, and the first runtime query must use
    // that same identity rather than silently resolving the embedded/default
    // catalog and losing the freshly activated entry.
    let _runtime_env = crate::test_process_env::TestProcessEnvGuard::new([
        ("OPENASR_HOME", Some(home.as_os_str().to_os_string())),
        (
            crate::OPENASR_CATALOG_FILE_ENV_VAR,
            Some(catalog_path.as_os_str().to_os_string()),
        ),
        (
            crate::OPENASR_CATALOG_IDENTITY_ENV_VAR,
            Some(std::ffi::OsString::from(&signature.catalog_url)),
        ),
        ("OPENASR_CATALOG_URL", None),
    ]);
    assert_eq!(
        crate::ggml_runtime::backend_plugin_activation_status()
            .unwrap()
            .as_deref(),
        Some(resolved.backend_id.as_str())
    );
    assert!(
        crate::ggml_runtime::ggml_available_devices()
            .iter()
            .any(|device| device.name == "CUDA0"),
        "activated CUDA pack must register CUDA0 in the neutral host"
    );
}

#[test]
fn backend_pack_download_plan_accounts_for_fresh_cached_and_partial_bytes() {
    let home = tempfile::tempdir().unwrap();
    let plugin = minimal_pe_bytes();
    let archive = tensile_zip_bytes();
    let first = hip_pack_resolved(&plugin, &archive);

    let fresh = backend_pack_download_plan(home.path(), &first).unwrap();
    assert_eq!(fresh.total_bytes, (plugin.len() + archive.len()) as u64);
    assert_eq!(fresh.plugin_bytes, plugin.len() as u64);
    assert_eq!(fresh.vendor_bytes, archive.len() as u64);
    assert_eq!(fresh.required_download_bytes, fresh.total_bytes);
    assert_eq!(fresh.required_plugin_bytes, fresh.plugin_bytes);
    assert_eq!(fresh.required_vendor_bytes, fresh.vendor_bytes);

    let mut first_client = FakeClient::with_responses(vec![
        ResponseSpec {
            status: 200,
            body: plugin.clone(),
        },
        ResponseSpec {
            status: 200,
            body: archive,
        },
    ]);
    install_backend_pack_with_client(&first, home.path(), &mut first_client, |_| {}).unwrap();

    let mut next = first.clone();
    next.version = "0.13.2".to_string();
    let staging = backend_pack_staging_dir(home.path(), &next).unwrap();
    fs::create_dir_all(&staging).unwrap();
    let plugin_dest = staging.join(&next.files[0].filename);
    let (partial, partial_meta) = backend_partial_paths(&plugin_dest).unwrap();
    let prefix_len = plugin.len() / 2;
    fs::write(&partial, &plugin[..prefix_len]).unwrap();
    write_backend_partial_meta(
        &partial_meta,
        &next.files[0],
        Some("stable-etag".to_string()),
        prefix_len as u64,
    )
    .unwrap();

    let resumed = backend_pack_download_plan(home.path(), &next).unwrap();
    assert_eq!(resumed.total_bytes, fresh.total_bytes);
    assert_eq!(resumed.required_vendor_bytes, 0);
    assert_eq!(
        resumed.required_plugin_bytes,
        plugin.len() as u64 - prefix_len as u64
    );
    assert_eq!(
        resumed.required_download_bytes,
        resumed.required_plugin_bytes
    );
}

#[test]
fn complete_backend_partial_is_verified_and_promoted_without_network() {
    let home = tempfile::tempdir().unwrap();
    let plugin = minimal_pe_bytes();
    let mut resolved = hip_pack_resolved(&plugin, &tensile_zip_bytes());
    resolved.files.truncate(1);
    let staging = backend_pack_staging_dir(home.path(), &resolved).unwrap();
    fs::create_dir_all(&staging).unwrap();
    let dest = staging.join(&resolved.files[0].filename);
    let (partial, partial_meta) = backend_partial_paths(&dest).unwrap();
    fs::write(&partial, &plugin).unwrap();
    write_backend_partial_meta(
        &partial_meta,
        &resolved.files[0],
        Some("stable-etag".to_string()),
        plugin.len() as u64,
    )
    .unwrap();

    let plan = backend_pack_download_plan(home.path(), &resolved).unwrap();
    assert_eq!(plan.required_download_bytes, 0);
    let mut empty = FakeClient::with_responses(Vec::new());
    let installed =
        install_backend_pack_with_client(&resolved, home.path(), &mut empty, |_| {}).unwrap();
    assert_eq!(
        fs::read(installed.dir.join("ggml-hip.dll")).unwrap(),
        plugin
    );
    assert!(!partial.exists());
    assert!(!partial_meta.exists());
}

#[test]
fn corrupted_complete_backend_partial_is_not_counted_as_resumable() {
    let home = tempfile::tempdir().unwrap();
    let plugin = minimal_pe_bytes();
    let mut resolved = hip_pack_resolved(&plugin, &tensile_zip_bytes());
    resolved.files.truncate(1);
    let staging = backend_pack_staging_dir(home.path(), &resolved).unwrap();
    fs::create_dir_all(&staging).unwrap();
    let dest = staging.join(&resolved.files[0].filename);
    let (partial, partial_meta) = backend_partial_paths(&dest).unwrap();
    let mut corrupted = plugin.clone();
    corrupted[0] ^= 0xff;
    fs::write(&partial, corrupted).unwrap();
    write_backend_partial_meta(&partial_meta, &resolved.files[0], None, plugin.len() as u64)
        .unwrap();

    let plan = backend_pack_download_plan(home.path(), &resolved).unwrap();
    assert_eq!(plan.required_download_bytes, plugin.len() as u64);
}

#[test]
fn backend_vendor_content_object_is_reused_across_pack_versions() {
    let home = tempfile::tempdir().unwrap();
    let plugin = minimal_pe_bytes();
    let archive = tensile_zip_bytes();
    let first = hip_pack_resolved(&plugin, &archive);
    let mut first_client = FakeClient::with_responses(vec![
        ResponseSpec {
            status: 200,
            body: plugin.clone(),
        },
        ResponseSpec {
            status: 200,
            body: archive,
        },
    ]);
    install_backend_pack_with_client(&first, home.path(), &mut first_client, |_| {}).unwrap();

    let mut second = first.clone();
    second.version = "0.13.2".to_string();
    let mut second_client = FakeClient::with_responses(vec![ResponseSpec {
        status: 200,
        body: plugin,
    }]);
    let installed =
        install_backend_pack_with_client(&second, home.path(), &mut second_client, |_| {}).unwrap();
    assert_eq!(second_client.urls(), vec![second.files[0].url.clone()]);
    assert!(
        installed
            .dir
            .join("rocblas/library/Kernels.so-000-gfx1200.hsaco")
            .is_file()
    );
    read_and_verify_installed_backend(&installed.dir, &second).unwrap();
}

#[test]
fn installed_backend_full_verification_rejects_tampered_file() {
    let home = tempfile::tempdir().unwrap();
    let plugin = minimal_pe_bytes();
    let mut resolved = hip_pack_resolved(&plugin, &tensile_zip_bytes());
    resolved.files.truncate(1);
    let mut client = FakeClient::with_responses(vec![ResponseSpec {
        status: 200,
        body: plugin,
    }]);
    install_backend_pack_with_client(&resolved, home.path(), &mut client, |_| {}).unwrap();
    let dir = backend_pack_install_dir(home.path(), &resolved).unwrap();
    fs::write(
        dir.join("ggml-hip.dll"),
        minimal_pe_bytes()
            .into_iter()
            .chain([7])
            .collect::<Vec<_>>(),
    )
    .unwrap();

    assert!(matches!(
        read_and_verify_installed_backend(&dir, &resolved),
        Err(PullError::SizeMismatch { .. }) | Err(PullError::ShaMismatch { .. })
    ));
}

#[test]
fn installed_backend_full_verification_rejects_unexpected_file() {
    let home = tempfile::tempdir().unwrap();
    let plugin = minimal_pe_bytes();
    let mut resolved = hip_pack_resolved(&plugin, &tensile_zip_bytes());
    resolved.files.truncate(1);
    let mut client = FakeClient::with_responses(vec![ResponseSpec {
        status: 200,
        body: plugin,
    }]);
    install_backend_pack_with_client(&resolved, home.path(), &mut client, |_| {}).unwrap();
    let dir = backend_pack_install_dir(home.path(), &resolved).unwrap();
    fs::write(dir.join("cublas64_12.dll"), b"planted").unwrap();

    let error = read_and_verify_installed_backend(&dir, &resolved).unwrap_err();
    assert!(matches!(
        error,
        PullError::UnexpectedInstalledBackendFile { .. }
    ));
    assert!(
        error
            .to_string()
            .contains("Unexpected file in installed backend pack")
    );
}

#[test]
fn installed_backend_full_verification_accepts_declared_files_only() {
    let home = tempfile::tempdir().unwrap();
    let plugin = minimal_pe_bytes();
    let archive = tensile_zip_bytes();
    let resolved = hip_pack_resolved(&plugin, &archive);
    let mut client = FakeClient::with_responses(vec![
        ResponseSpec {
            status: 200,
            body: plugin,
        },
        ResponseSpec {
            status: 200,
            body: archive,
        },
    ]);
    let installed =
        install_backend_pack_with_client(&resolved, home.path(), &mut client, |_| {}).unwrap();

    read_and_verify_installed_backend(&installed.dir, &resolved).unwrap();
}

#[test]
fn installed_backend_load_images_skips_tampered_non_dll_archive_payload() {
    let home = tempfile::tempdir().unwrap();
    let plugin = minimal_pe_bytes();
    let archive = tensile_zip_bytes();
    let resolved = hip_pack_resolved(&plugin, &archive);
    let mut client = FakeClient::with_responses(vec![
        ResponseSpec {
            status: 200,
            body: plugin,
        },
        ResponseSpec {
            status: 200,
            body: archive,
        },
    ]);
    let installed =
        install_backend_pack_with_client(&resolved, home.path(), &mut client, |_| {}).unwrap();
    let hsaco = installed
        .dir
        .join("rocblas")
        .join("library")
        .join("Kernels.so-000-gfx1200.hsaco");
    let mut bytes = fs::read(&hsaco).unwrap();
    bytes[0] ^= 0xff;
    fs::write(&hsaco, bytes).unwrap();

    assert!(matches!(
        read_and_verify_installed_backend(&installed.dir, &resolved),
        Err(PullError::ShaMismatch { .. })
    ));
    read_and_verify_installed_backend_for_activation(&installed.dir, &resolved).unwrap();
}

#[test]
fn installed_backend_load_images_rejects_unexpected_dll() {
    let home = tempfile::tempdir().unwrap();
    let plugin = minimal_pe_bytes();
    let archive = tensile_zip_bytes();
    let resolved = hip_pack_resolved(&plugin, &archive);
    let mut client = FakeClient::with_responses(vec![
        ResponseSpec {
            status: 200,
            body: plugin,
        },
        ResponseSpec {
            status: 200,
            body: archive,
        },
    ]);
    let installed =
        install_backend_pack_with_client(&resolved, home.path(), &mut client, |_| {}).unwrap();
    fs::write(installed.dir.join("planted.dll"), b"planted").unwrap();

    let error =
        read_and_verify_installed_backend_for_activation(&installed.dir, &resolved).unwrap_err();
    assert!(matches!(
        error,
        PullError::UnexpectedInstalledBackendFile { .. }
    ));
}

#[test]
fn backend_install_lock_is_os_owned_and_exclusive() {
    let home = tempfile::tempdir().unwrap();
    let path = home.path().join("backend-install.lock");
    let first = BackendInstallLock::acquire(&path).unwrap();
    assert!(matches!(
        BackendInstallLock::acquire(&path),
        Err(PullError::LockHeld { .. })
    ));
    drop(first);
    BackendInstallLock::acquire(&path).unwrap();
}

#[test]
fn backend_store_mutation_lock_serializes_install_activation_and_gc() {
    let home = tempfile::tempdir().unwrap();
    let first = BackendStoreMutationLock::acquire(home.path()).unwrap();
    assert!(matches!(
        BackendStoreMutationLock::acquire(home.path()),
        Err(PullError::LockHeld { .. })
    ));
    drop(first);
    BackendStoreMutationLock::acquire(home.path()).unwrap();
}

#[test]
fn backend_store_gc_retains_requested_pack_and_its_shared_vendor_object() {
    let home = tempfile::tempdir().unwrap();
    let plugin = minimal_pe_bytes();
    let archive = tensile_zip_bytes();
    let first = hip_pack_resolved(&plugin, &archive);
    let mut first_client = FakeClient::with_responses(vec![
        ResponseSpec {
            status: 200,
            body: plugin.clone(),
        },
        ResponseSpec {
            status: 200,
            body: archive,
        },
    ]);
    install_backend_pack_with_client(&first, home.path(), &mut first_client, |_| {}).unwrap();

    let mut second = first.clone();
    second.backend_id = "hip-radeon-next".to_string();
    second.version = "0.13.2".to_string();
    let mut second_client = FakeClient::with_responses(vec![ResponseSpec {
        status: 200,
        body: plugin,
    }]);
    install_backend_pack_with_client(&second, home.path(), &mut second_client, |_| {}).unwrap();

    let first_dir = backend_pack_install_dir(home.path(), &first).unwrap();
    let second_dir = backend_pack_install_dir(home.path(), &second).unwrap();
    let object_dir = backend_content_object_dir(home.path(), &second.files[1]);
    let report = gc_backend_store(
        home.path(),
        [second.backend_id.clone()],
        Some(Duration::ZERO),
    )
    .unwrap();
    assert!(!first_dir.exists());
    assert!(second_dir.is_dir());
    assert!(object_dir.is_dir());
    assert_eq!(report.removed_pack_directories, 1);
    assert_eq!(report.removed_content_objects, 0);

    let report = gc_backend_store(home.path(), Vec::new(), Some(Duration::ZERO)).unwrap();
    assert!(!second_dir.exists());
    assert!(!object_dir.exists());
    assert_eq!(report.removed_pack_directories, 1);
    assert_eq!(report.removed_content_objects, 1);
}

#[test]
fn backend_store_gc_reclaims_stale_staging_tree() {
    let home = tempfile::tempdir().unwrap();
    let staging = home.path().join("backends/.staging/hip/0.13.1/artifact");
    fs::create_dir_all(&staging).unwrap();
    fs::write(staging.join("payload.bin"), b"payload").unwrap();
    let report = gc_backend_store(home.path(), Vec::new(), Some(Duration::ZERO)).unwrap();
    assert_eq!(report.removed_staging_directories, 1);
    assert!(!staging.exists());
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

#[test]
fn backend_pack_download_plan_rejects_unsafe_version_segment() {
    let home = tempfile::tempdir().unwrap();
    let plugin = minimal_pe_bytes();
    let mut resolved = hip_pack_resolved(&plugin, &tensile_zip_bytes());
    resolved.version = "../escape".to_string();
    assert!(matches!(
        backend_pack_download_plan(home.path(), &resolved),
        Err(PullError::InvalidTarget {
            field: "backend.version",
            ..
        })
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

    let dir = backend_pack_install_dir(home.path(), &resolved).unwrap();
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

    let dir = backend_pack_install_dir(home.path(), &resolved).unwrap();
    assert_eq!(installed.dir, dir);
    assert_eq!(fs::read(dir.join("libbackend.so")).unwrap(), plugin);
    // Second attempt must have asked for a Range starting at the byte the
    // dropped connection had already delivered -- a from-scratch restart
    // would instead show `[None, None]`.
    assert_eq!(client.ranges(), vec![None, Some(700)]);
    let partial = dir.join(".libbackend.so.partial");
    assert!(!partial.exists());
    assert!(!dir.join(".libbackend.so.partial.json").exists());
}

#[test]
fn backend_pack_resume_survives_process_boundary_metadata_reload() {
    let home = tempfile::tempdir().unwrap();
    let mut plugin: Vec<u8> = vec![0x7F, b'E', b'L', b'F'];
    plugin.extend((0_u32..2000).map(|value| (value % 251) as u8));
    let mut resolved = hip_pack_resolved(&plugin, &tensile_zip_bytes());
    resolved.files.truncate(1);
    {
        let file = &mut resolved.files[0];
        file.filename = "libbackend.so".to_string();
        file.sha256 = sha256_hex(&plugin);
        file.size_bytes = plugin.len() as u64;
    }

    let dir = backend_pack_install_dir(home.path(), &resolved).unwrap();
    fs::create_dir_all(&dir).unwrap();
    let file = &resolved.files[0];
    let dest = dir.join(&file.filename);
    let partial = dir.join(".libbackend.so.partial");
    let partial_meta = dir.join(".libbackend.so.partial.json");
    let mut first = BackendMidStreamDropThenResumeClient::new(plugin.clone(), 700);
    let mut expected_etag = None;
    assert!(
        download_backend_file_attempt(
            &mut first,
            file,
            &dest,
            &partial,
            &partial_meta,
            &mut expected_etag,
            &mut |_| {},
        )
        .is_err()
    );
    assert_eq!(fs::metadata(&partial).unwrap().len(), 700);
    assert!(partial_meta.is_file());

    // A new client and a fresh in-memory ETag represent the replacement
    // process after NSIS terminates the old one. Resume provenance must come
    // entirely from the stable on-disk metadata.
    let mut replacement = BackendMidStreamDropThenResumeClient {
        bytes: plugin.clone(),
        split: 700,
        attempts: 1,
        ranges: Vec::new(),
    };
    download_backend_file(&mut replacement, file, &dest, &mut |_| {}).unwrap();
    assert_eq!(replacement.ranges(), vec![Some(700)]);
    assert_eq!(fs::read(dest).unwrap(), plugin);
    assert!(!partial.exists());
    assert!(!partial_meta.exists());
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
fn parallel_download_emits_verifying_after_segments_before_hash() {
    // Guards the progress-phase fix: once the chunked path has every segment on
    // disk it rereads and hashes the whole file, which is silent (no byte
    // progress) and can take seconds on a multi-GB pack. The download path must
    // emit `Verifying` before that hash so the UI leaves the "downloading" phase
    // instead of freezing at 100%. Observable proxy: a clean chunked pull emits
    // `Verifying` twice -- once from the chunked completion (pre-hash) and once
    // from `verify_partial_and_install` -- versus once without the fix, and every
    // `Downloading` event precedes the first `Verifying`.
    let bytes = tiny_pack_bytes();
    let resolved = resolved_for(&bytes);
    let temp = tempfile::tempdir().unwrap();
    let segment_bytes = small_segment_bytes(bytes.len(), 5);
    assert!(
        segment_count(bytes.len() as u64, segment_bytes) >= 2,
        "fixture too small to exercise chunking"
    );

    let server = RangeServerClient::new(bytes.clone());
    let (mut probe_client, factory) = parallel_probe_and_factory(&server);
    let parallel = ParallelDownloadConfig {
        connections: 4,
        factory: &*factory,
    };

    let phases = Arc::new(Mutex::new(Vec::<&'static str>::new()));
    let phases_sink = phases.clone();
    let installed = pull_model_pack_with_client_parallel(
        &resolved,
        temp.path(),
        &mut probe_client,
        parallel_test_options(segment_bytes),
        parallel,
        move |event| {
            let name = match event {
                PullProgress::UsingInstalled { .. } => "using_installed",
                PullProgress::DownloadStarted { .. } => "download_started",
                PullProgress::Downloading { .. } => "downloading",
                PullProgress::Verifying { .. } => "verifying",
                PullProgress::Installed { .. } => "installed",
            };
            phases_sink.lock().unwrap().push(name);
        },
        || false,
        || false,
    )
    .unwrap();
    assert_eq!(installed.pull, "moonshine-tiny:q8");

    let phases = phases.lock().unwrap().clone();
    let first_verifying = phases
        .iter()
        .position(|phase| *phase == "verifying")
        .expect("a chunked pull must emit Verifying");
    let last_downloading = phases.iter().rposition(|phase| *phase == "downloading");
    if let Some(last_downloading) = last_downloading {
        assert!(
            last_downloading < first_verifying,
            "every Downloading event must precede the first Verifying: {phases:?}"
        );
    }
    assert_eq!(
        phases.iter().filter(|phase| **phase == "verifying").count(),
        2,
        "chunked download must signal Verifying before its own hash (plus the \
         install-time verify): {phases:?}"
    );
    assert_eq!(
        phases.last().copied(),
        Some("installed"),
        "the terminal event must be Installed: {phases:?}"
    );
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

/// Yields exactly one byte per `read()` call, regardless of the caller's
/// buffer size -- used to guarantee the very first chunk `write_segment_body`
/// observes is smaller than any realistic low-speed floor, without depending
/// on real wall-clock timing (paired with `segment_low_speed_timeout:
/// Duration::ZERO` in the tests below, which makes `SegmentLowSpeedWindow`
/// judge the very first observed chunk instead of waiting out a real window).
struct OneByteAtATimeReader {
    remaining: VecDeque<u8>,
}

impl OneByteAtATimeReader {
    fn new(bytes: Vec<u8>) -> Self {
        Self {
            remaining: bytes.into(),
        }
    }
}

impl Read for OneByteAtATimeReader {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        match self.remaining.pop_front() {
            Some(byte) => {
                buf[0] = byte;
                Ok(1)
            }
            None => Ok(0),
        }
    }
}

/// A `RangeServerClient`-alike whose responses trickle one byte at a time
/// for whichever segment(s) `flaky_start` selects -- tripping a zero-timeout
/// `SegmentLowSpeedWindow` window on the very first chunk of that segment --
/// while every other segment (and, if `always_flaky` is `false`, every
/// attempt after the first on the flaky segment too) is served normally in
/// one read. Models "one connection to this source is bad; a fresh one for
/// the same range is fine" without needing a real slow socket.
///
/// `flaky_start: None` makes *every* segment trickle uniformly instead of
/// singling one out -- used to model a session where the whole network is
/// slow, not just one connection.
#[derive(Clone)]
struct FlakySegmentRangeClient {
    bytes: Arc<Vec<u8>>,
    attempts_by_start: Arc<Mutex<HashMap<u64, usize>>>,
    flaky_start: Option<u64>,
    always_flaky: bool,
}

impl FlakySegmentRangeClient {
    fn new(bytes: Vec<u8>, flaky_start: u64, always_flaky: bool) -> Self {
        Self {
            bytes: Arc::new(bytes),
            attempts_by_start: Arc::new(Mutex::new(HashMap::new())),
            flaky_start: Some(flaky_start),
            always_flaky,
        }
    }

    /// Every segment trickles on every attempt: models a uniformly slow
    /// session (see the type's doc comment).
    fn new_uniformly_slow(bytes: Vec<u8>) -> Self {
        Self {
            bytes: Arc::new(bytes),
            attempts_by_start: Arc::new(Mutex::new(HashMap::new())),
            flaky_start: None,
            always_flaky: true,
        }
    }

    fn attempts_for(&self, start: u64) -> usize {
        self.attempts_by_start
            .lock()
            .unwrap()
            .get(&start)
            .copied()
            .unwrap_or(0)
    }

    fn attempts_on_flaky_segment(&self) -> usize {
        let flaky_start = self
            .flaky_start
            .expect("attempts_on_flaky_segment requires a single flaky_start");
        self.attempts_for(flaky_start)
    }

    fn every_recorded_segment_attempted_exactly_once(&self) -> bool {
        self.attempts_by_start
            .lock()
            .unwrap()
            .values()
            .all(|count| *count == 1)
    }
}

impl DownloadClient for FlakySegmentRangeClient {
    fn open(
        &mut self,
        _url: &str,
        range: Option<ByteRange>,
    ) -> Result<DownloadResponse, PullError> {
        let range = range.expect("the segmented download path always sends a bounded Range");
        let start = range.start;
        let end = range.end.expect("segment fetches always bound the end");
        let total = self.bytes.len() as u64;
        let slice = self.bytes[start as usize..=end as usize].to_vec();
        let attempt = {
            let mut attempts = self.attempts_by_start.lock().unwrap();
            let counter = attempts.entry(start).or_insert(0);
            *counter += 1;
            *counter
        };
        let is_flaky_now = match self.flaky_start {
            None => true,
            Some(flaky_start) => start == flaky_start && (self.always_flaky || attempt == 1),
        };
        let reader: Box<dyn Read> = if is_flaky_now {
            Box::new(OneByteAtATimeReader::new(slice))
        } else {
            Box::new(Cursor::new(slice))
        };
        Ok(DownloadResponse {
            status: 206,
            content_length: Some(end - start + 1),
            content_range: Some(format!("bytes {start}-{end}/{total}")),
            etag: Some("etag-a".to_string()),
            reader,
        })
    }
}

/// Builds the `PullTarget`/`PullPaths` pair `download_parallel_attempt` needs,
/// without going through the outer `download_with_retries` retry loop (whose
/// real `std::thread::sleep` backoff would make a deliberately-exhausted
/// low-speed test slow for no reason -- these tests call the segmented
/// attempt directly so the requeue/cap logic under test is exercised without
/// any unrelated timing).
fn parallel_attempt_paths(home: &Path, resolved: &ResolvedCatalogPull) -> (PullTarget, PullPaths) {
    let target = PullTarget::from_resolved(resolved).unwrap();
    let paths = pull_paths(home, &target).unwrap();
    ensure_storage_dir_within_root(home, &paths).unwrap();
    (target, paths)
}

/// `connections: 1` makes the single worker drain the shared queue strictly
/// in ascending index order (FIFO, one segment at a time) -- these tests
/// need that determinism to control exactly how many reference samples
/// exist by the time a particular segment is evaluated, which a real
/// multi-connection race wouldn't guarantee.
fn single_worker_parallel_config(
    factory: &dyn Fn() -> Result<BoxedDownloadClient, PullError>,
) -> ParallelDownloadConfig<'_> {
    ParallelDownloadConfig {
        connections: 1,
        factory,
    }
}

fn low_speed_test_options(segment_bytes: u64) -> PullOptions {
    // Duration::ZERO: see `zero_window_options` above -- makes every
    // `observe` call judge its window (here, one buffer read) immediately.
    PullOptions {
        segment_low_speed_timeout: Duration::ZERO,
        ..parallel_test_options(segment_bytes)
    }
}

#[test]
fn parallel_download_low_speed_segment_requeues_onto_fresh_connection_and_recovers() {
    let bytes = tiny_pack_bytes();
    let resolved = resolved_for(&bytes);
    let temp = tempfile::tempdir().unwrap();
    let segment_bytes = small_segment_bytes(bytes.len(), 5);
    let total_segments = segment_count(bytes.len() as u64, segment_bytes);
    assert!(
        total_segments >= 5,
        "fixture too small: need the probe plus 3 normal siblings to warm the \
         reference before the flaky (last) segment is evaluated"
    );
    // The *last* index, fetched last by a single sequential worker: by the
    // time it's evaluated, the probe (index 0) plus every other worker
    // segment before it have already recorded their (normal-speed) windows,
    // satisfying SEGMENT_LOW_SPEED_MIN_REFERENCE_SAMPLES -- exactly the
    // reported "lone tail straggler after a fast bulk" scenario.
    let (flaky_start, _) = segment_range(total_segments - 1, bytes.len() as u64, segment_bytes);

    let mut probe_client = FlakySegmentRangeClient::new(bytes.clone(), flaky_start, false);
    let factory_client = probe_client.clone();
    let factory: Box<dyn Fn() -> Result<BoxedDownloadClient, PullError>> =
        Box::new(move || Ok(Box::new(factory_client.clone()) as BoxedDownloadClient));
    let parallel = single_worker_parallel_config(&*factory);
    let options = low_speed_test_options(segment_bytes);

    let progress_events = Arc::new(Mutex::new(Vec::<(u64, u64)>::new()));
    let progress_sink = progress_events.clone();
    let (target, paths) = parallel_attempt_paths(temp.path(), &resolved);

    let outcome = download_parallel_attempt(
        &target,
        &paths,
        &mut probe_client,
        &parallel,
        segment_bytes,
        &options,
        &mut |event| {
            if let PullProgress::Downloading {
                bytes_done,
                bytes_total,
            } = event
            {
                progress_sink
                    .lock()
                    .unwrap()
                    .push((bytes_done, bytes_total));
            }
        },
        &|| false,
        &|| false,
    )
    .unwrap();

    let downloaded = match outcome {
        ParallelAttemptOutcome::Completed(downloaded) => downloaded,
        ParallelAttemptOutcome::RangeNotSupported => {
            panic!("this mock always honors Range with 206")
        }
    };
    assert_eq!(downloaded.bytes_done, bytes.len() as u64);
    assert_eq!(downloaded.sha256, sha256_hex(&bytes));
    assert_eq!(fs::read(&paths.partial_path).unwrap(), bytes);

    // Exactly one abandoned attempt, then one successful retry -- proof the
    // segment was requeued and refetched (on what this mock models as a
    // fresh connection) rather than either silently corrupting the file or
    // hanging on the first bad connection.
    assert_eq!(probe_client.attempts_on_flaky_segment(), 2);

    // The low-speed rollback must keep `bytes_done` from ever being reported
    // past the true total: without it, the abandoned attempt's partial bytes
    // would still be counted once the retry re-downloads the same range from
    // scratch, double-counting them.
    for (bytes_done, bytes_total) in progress_events.lock().unwrap().iter().copied() {
        assert!(
            bytes_done <= bytes_total,
            "progress double-counted: reported {bytes_done} of {bytes_total} total bytes"
        );
    }
}

#[test]
fn parallel_download_low_speed_segment_gives_up_after_max_abandons_and_still_completes() {
    // The worst case after exhausting the reconnect budget must be identical
    // to not having this guard at all: the download still succeeds, just
    // slower for this one segment, never a hard failure.
    let bytes = tiny_pack_bytes();
    let resolved = resolved_for(&bytes);
    let temp = tempfile::tempdir().unwrap();
    let segment_bytes = small_segment_bytes(bytes.len(), 5);
    let total_segments = segment_count(bytes.len() as u64, segment_bytes);
    assert!(
        total_segments >= 5,
        "fixture too small to warm the reference"
    );
    let (flaky_start, _) = segment_range(total_segments - 1, bytes.len() as u64, segment_bytes);

    // `always_flaky: true` -- this segment never recovers, no matter how
    // many fresh connections it gets, exercising the bounded degrade-and-
    // finish path.
    let mut probe_client = FlakySegmentRangeClient::new(bytes.clone(), flaky_start, true);
    let factory_client = probe_client.clone();
    let factory: Box<dyn Fn() -> Result<BoxedDownloadClient, PullError>> =
        Box::new(move || Ok(Box::new(factory_client.clone()) as BoxedDownloadClient));
    let parallel = single_worker_parallel_config(&*factory);
    // Cooldown disabled: this test isolates the abandon-count cap from the
    // separate hysteresis behavior (covered by
    // `segment_low_speed_window_cooldown_suppresses_a_thrashing_retrip`).
    // With the default cooldown active, this mock's near-instant retry would
    // itself fall inside the cooldown window and never re-trip -- which
    // would still finish successfully, but wouldn't be exercising the cap.
    let options = PullOptions {
        segment_low_speed_cooldown: Duration::ZERO,
        ..low_speed_test_options(segment_bytes)
    };
    let (target, paths) = parallel_attempt_paths(temp.path(), &resolved);

    let outcome = download_parallel_attempt(
        &target,
        &paths,
        &mut probe_client,
        &parallel,
        segment_bytes,
        &options,
        &mut |_| {},
        &|| false,
        &|| false,
    )
    .unwrap();

    let downloaded = match outcome {
        ParallelAttemptOutcome::Completed(downloaded) => downloaded,
        ParallelAttemptOutcome::RangeNotSupported => {
            panic!("this mock always honors Range with 206")
        }
    };
    assert_eq!(downloaded.sha256, sha256_hex(&bytes));
    assert_eq!(fs::read(&paths.partial_path).unwrap(), bytes);

    // SEGMENT_MAX_RETRIES abandon-and-requeue rounds, then one final attempt
    // with the guard disabled that's simply allowed to trickle to
    // completion -- never a `PullError` (that variant no longer exists).
    assert_eq!(
        probe_client.attempts_on_flaky_segment(),
        SEGMENT_MAX_RETRIES + 1
    );
}

#[test]
fn parallel_download_cold_start_never_flags_an_early_trickling_segment() {
    // The segment fetched immediately after the probe has, at most, one
    // prior sample (the probe's own) -- below
    // SEGMENT_LOW_SPEED_MIN_REFERENCE_SAMPLES, so it must never be judged
    // low-speed no matter how slowly it trickles. This is the same
    // cold-start contract as the pure unit test above, exercised end to end
    // through the real mock transport.
    let bytes = tiny_pack_bytes();
    let resolved = resolved_for(&bytes);
    let temp = tempfile::tempdir().unwrap();
    let segment_bytes = small_segment_bytes(bytes.len(), 5);
    let total_segments = segment_count(bytes.len() as u64, segment_bytes);
    assert!(total_segments >= 5, "fixture too small");
    let (flaky_start, _) = segment_range(1, bytes.len() as u64, segment_bytes);

    let mut probe_client = FlakySegmentRangeClient::new(bytes.clone(), flaky_start, true);
    let factory_client = probe_client.clone();
    let factory: Box<dyn Fn() -> Result<BoxedDownloadClient, PullError>> =
        Box::new(move || Ok(Box::new(factory_client.clone()) as BoxedDownloadClient));
    let parallel = single_worker_parallel_config(&*factory);
    let options = low_speed_test_options(segment_bytes);
    let (target, paths) = parallel_attempt_paths(temp.path(), &resolved);

    let outcome = download_parallel_attempt(
        &target,
        &paths,
        &mut probe_client,
        &parallel,
        segment_bytes,
        &options,
        &mut |_| {},
        &|| false,
        &|| false,
    )
    .unwrap();

    match outcome {
        ParallelAttemptOutcome::Completed(downloaded) => {
            assert_eq!(downloaded.sha256, sha256_hex(&bytes));
        }
        ParallelAttemptOutcome::RangeNotSupported => {
            panic!("this mock always honors Range with 206")
        }
    }
    // Never requeued: cold start suppressed every evaluation of this segment.
    assert_eq!(probe_client.attempts_on_flaky_segment(), 1);
}

#[test]
fn parallel_download_uniformly_slow_session_completes_without_any_requeue() {
    // The "慢但能成" contract end to end: every segment (including the
    // probe) trickles uniformly, so no segment is ever an outlier relative
    // to its own siblings -- the ratio test alone must never fire, and the
    // whole download must still succeed without a single requeue, exactly
    // like it would have before this guard existed.
    let bytes = tiny_pack_bytes();
    let resolved = resolved_for(&bytes);
    let temp = tempfile::tempdir().unwrap();
    let segment_bytes = small_segment_bytes(bytes.len(), 5);
    let total_segments = segment_count(bytes.len() as u64, segment_bytes);
    assert!(total_segments >= 5, "fixture too small");

    let mut probe_client = FlakySegmentRangeClient::new_uniformly_slow(bytes.clone());
    let factory_client = probe_client.clone();
    let factory: Box<dyn Fn() -> Result<BoxedDownloadClient, PullError>> =
        Box::new(move || Ok(Box::new(factory_client.clone()) as BoxedDownloadClient));
    let parallel = ParallelDownloadConfig {
        connections: 4,
        factory: &*factory,
    };
    let options = low_speed_test_options(segment_bytes);
    let (target, paths) = parallel_attempt_paths(temp.path(), &resolved);

    let outcome = download_parallel_attempt(
        &target,
        &paths,
        &mut probe_client,
        &parallel,
        segment_bytes,
        &options,
        &mut |_| {},
        &|| false,
        &|| false,
    )
    .unwrap();

    match outcome {
        ParallelAttemptOutcome::Completed(downloaded) => {
            assert_eq!(downloaded.sha256, sha256_hex(&bytes));
        }
        ParallelAttemptOutcome::RangeNotSupported => {
            panic!("this mock always honors Range with 206")
        }
    }
    assert!(
        probe_client.every_recorded_segment_attempted_exactly_once(),
        "a uniformly slow session must never trigger a single requeue"
    );
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

// ---------------------------------------------------------------------------
// Legacy-layout migration
//
// The store has one readable layout. These cover the one-way conversion that
// gets an upgrading user there: it must move bytes rather than copy them, must
// leave exactly one copy behind, and must never delete anything it did not
// successfully replace.
// ---------------------------------------------------------------------------

/// Write a complete legacy install at `<models>/<model>/<quant>/`.
///
/// `recorded_sha` lets a test plant the stale digest a real upgrading store can
/// carry, so migration is forced to prove it recomputes rather than trusts it.
fn write_legacy_install(
    home: &Path,
    models_root_dir: &Path,
    model_id: &str,
    quant: &str,
    recorded_sha: Option<&str>,
) -> (PathBuf, Vec<u8>) {
    let _ = home;
    let dir = models_root_dir.join(model_id).join(quant);
    fs::create_dir_all(&dir).unwrap();
    // Per-model bytes, so a fixture with two models exercises two distinct
    // objects rather than silently deduplicating into one.
    let bytes = {
        let temp = tempfile::tempdir().unwrap();
        let source = temp.path().join("tiny.oasr");
        let spec = TinyGgufFixtureSpec::whisper_oasr_v1_encoder_graph_one_layer(model_id)
            .with_whisper_minimal_tokenizer();
        write_tiny_gguf_runtime_source(&source, &spec).unwrap();
        fs::read(source).unwrap()
    };
    let filename = format!("{model_id}-{quant}.oasr");
    let path = dir.join(&filename);
    fs::write(&path, &bytes).unwrap();
    let pack = InstalledPack {
        model_id: model_id.to_string(),
        display_name: model_id.to_string(),
        quant: quant.to_string(),
        suffix: "q8".to_string(),
        pull: format!("{model_id}:q8"),
        filename,
        path: path.clone(),
        url: "https://example.invalid/model.oasr".to_string(),
        hf_revision: "test".to_string(),
        sha256: recorded_sha
            .map(str::to_string)
            .unwrap_or_else(|| sha256_hex(&bytes)),
        size_bytes: bytes.len() as u64,
        installed_at_unix_seconds: 1,
        source: None,
    };
    fs::write(
        dir.join("installed.json"),
        serde_json::to_string_pretty(&pack).unwrap(),
    )
    .unwrap();
    (path, bytes)
}

fn object_path_for(models_root_dir: &Path, digest: &str) -> PathBuf {
    models_root_dir
        .join("objects/sha256")
        .join(digest)
        .join("content")
}

/// Total bytes of regular files under a directory tree.
fn tree_bytes(root: &Path) -> u64 {
    let mut total = 0;
    let Ok(entries) = fs::read_dir(root) else {
        return 0;
    };
    for entry in entries.flatten() {
        let Ok(metadata) = entry.metadata() else {
            continue;
        };
        if metadata.is_dir() {
            total += tree_bytes(&entry.path());
        } else if metadata.is_file() {
            total += metadata.len();
        }
    }
    total
}

#[test]
fn migration_converts_a_legacy_install_and_leaves_exactly_one_copy() {
    let home = tempfile::tempdir().unwrap();
    let models = home.path().join("models");
    let (legacy_path, bytes) =
        write_legacy_install(home.path(), &models, "moonshine-tiny", "q8_0", None);
    let digest = sha256_hex(&bytes);
    let before = tree_bytes(&models);

    let report = migrate_legacy_model_store(home.path()).unwrap();
    assert_eq!(report.migrated, vec!["moonshine-tiny:q8".to_string()]);
    assert!(report.failures.is_empty(), "{:?}", report.failures);

    // The legacy tree is gone in full -- record *and* pack bytes. Leaving the
    // `.oasr` behind is what made a converted store carry two copies of every
    // model.
    assert!(!legacy_path.exists());
    assert!(!models.join("moonshine-tiny").exists());

    let object = object_path_for(&models, &digest);
    assert!(object.is_file());
    assert_eq!(fs::read(&object).unwrap(), bytes);
    assert!(models.join("refs/moonshine-tiny/q8_0.json").is_file());

    // One copy on disk, not two. Converting a store that already held a single
    // copy only swaps `installed.json` for a ref, so the totals differ by
    // metadata; what must never happen is the pack's own bytes appearing twice.
    let after = tree_bytes(&models);
    let pack_len = bytes.len() as u64;
    assert!(
        after >= pack_len && after < before + pack_len,
        "exactly one copy of the pack must survive (before {before}, after {after}, pack {pack_len})"
    );

    let packs = list_installed_packs(home.path()).unwrap();
    assert_eq!(packs.len(), 1);
    assert_eq!(packs[0].pull, "moonshine-tiny:q8");
    assert_eq!(packs[0].path, object);
    assert_eq!(packs[0].sha256, digest);
}

#[test]
fn migration_seals_the_object_it_lands() {
    let home = tempfile::tempdir().unwrap();
    let models = home.path().join("models");
    let (_, bytes) = write_legacy_install(home.path(), &models, "moonshine-tiny", "q8_0", None);

    migrate_legacy_model_store(home.path()).unwrap();

    let object = object_path_for(&models, &sha256_hex(&bytes));
    assert!(
        fs::metadata(&object).unwrap().permissions().readonly(),
        "a migrated object must be sealed like an admitted one"
    );
}

#[test]
fn migration_recomputes_the_digest_instead_of_trusting_the_legacy_record() {
    let home = tempfile::tempdir().unwrap();
    let models = home.path().join("models");
    let stale = "b".repeat(64);
    let (_, bytes) =
        write_legacy_install(home.path(), &models, "moonshine-tiny", "q8_0", Some(&stale));

    migrate_legacy_model_store(home.path()).unwrap();

    let real = sha256_hex(&bytes);
    assert!(object_path_for(&models, &real).is_file());
    assert!(
        !models.join("objects/sha256").join(&stale).exists(),
        "a stale recorded digest must never name an object"
    );
    let packs = list_installed_packs(home.path()).unwrap();
    assert_eq!(packs[0].sha256, real);
}

#[test]
fn migration_reclaims_a_legacy_copy_whose_ref_already_exists() {
    // A previous run published the ref and died before cleanup, so the legacy
    // tree is pure duplication. This is the 4.9G of redundant copies measured on
    // a real upgraded store.
    let home = tempfile::tempdir().unwrap();
    let models = home.path().join("models");
    let (_, bytes) = write_legacy_install(home.path(), &models, "moonshine-tiny", "q8_0", None);
    migrate_legacy_model_store(home.path()).unwrap();

    // Re-create the legacy copy beside the now-authoritative ref.
    let (legacy_path, _) =
        write_legacy_install(home.path(), &models, "moonshine-tiny", "q8_0", None);
    assert!(legacy_path.is_file());
    let before = tree_bytes(&models);

    let report = migrate_legacy_model_store(home.path()).unwrap();
    assert!(
        report.migrated.is_empty(),
        "nothing new is published; the ref already exists"
    );
    assert!(report.reclaimed_bytes >= bytes.len() as u64);
    assert!(!legacy_path.exists());
    assert!(!models.join("moonshine-tiny").exists());
    assert_eq!(tree_bytes(&models), before - report.reclaimed_bytes);

    // The surviving pack is still fully usable.
    let packs = list_installed_packs(home.path()).unwrap();
    assert_eq!(packs.len(), 1);
    assert_eq!(fs::read(&packs[0].path).unwrap(), bytes);
}

#[test]
fn migration_is_idempotent() {
    let home = tempfile::tempdir().unwrap();
    let models = home.path().join("models");
    write_legacy_install(home.path(), &models, "moonshine-tiny", "q8_0", None);

    let first = migrate_legacy_model_store(home.path()).unwrap();
    assert_eq!(first.migrated.len(), 1);
    let settled = tree_bytes(&models);

    let second = migrate_legacy_model_store(home.path()).unwrap();
    assert!(second.is_empty(), "{second:?}");
    assert_eq!(tree_bytes(&models), settled);
    assert_eq!(list_installed_packs(home.path()).unwrap().len(), 1);
}

#[test]
fn migration_happens_in_place_under_a_custom_models_dir() {
    let home = tempfile::tempdir().unwrap();
    let elsewhere = tempfile::tempdir().unwrap();
    fs::write(
        home.path().join("config.json"),
        serde_json::json!({ "models_dir": elsewhere.path() }).to_string(),
    )
    .unwrap();
    let (_, bytes) = write_legacy_install(
        home.path(),
        elsewhere.path(),
        "moonshine-tiny",
        "q8_0",
        None,
    );

    let report = migrate_legacy_model_store(home.path()).unwrap();
    assert_eq!(report.migrated.len(), 1);

    // Converted inside the user's chosen directory, with nothing created in the
    // default location.
    assert!(object_path_for(elsewhere.path(), &sha256_hex(&bytes)).is_file());
    assert!(
        elsewhere
            .path()
            .join("refs/moonshine-tiny/q8_0.json")
            .is_file()
    );
    assert!(
        !home.path().join("models").exists(),
        "migration must never relocate a redirected store to the default root"
    );
    assert_eq!(
        list_installed_packs(home.path()).unwrap()[0].path,
        object_path_for(elsewhere.path(), &sha256_hex(&bytes))
    );
}

#[test]
fn migration_leaves_an_unconvertible_record_and_its_bytes_in_place() {
    let home = tempfile::tempdir().unwrap();
    let models = home.path().join("models");
    let (legacy_path, _) =
        write_legacy_install(home.path(), &models, "moonshine-tiny", "q8_0", None);
    // Truncate the pack so the record no longer matches its file: the reader
    // already refuses this, and migration must refuse it too rather than admit
    // garbage or delete the operator's bytes.
    fs::write(&legacy_path, b"not a pack").unwrap();

    let report = migrate_legacy_model_store(home.path()).unwrap();
    assert!(report.migrated.is_empty());
    assert_eq!(report.failures.len(), 1);
    assert!(legacy_path.is_file(), "failing records keep their bytes");
    assert!(!models.join("refs").exists());
}

#[test]
fn migration_converts_every_quant_of_a_model_independently() {
    let home = tempfile::tempdir().unwrap();
    let models = home.path().join("models");
    write_legacy_install(home.path(), &models, "moonshine-tiny", "q8_0", None);
    let (broken, _) = write_legacy_install(home.path(), &models, "moonshine-tiny", "q4_k", None);
    fs::write(&broken, b"not a pack").unwrap();

    let report = migrate_legacy_model_store(home.path()).unwrap();
    assert_eq!(report.migrated, vec!["moonshine-tiny:q8".to_string()]);
    assert_eq!(report.failures.len(), 1);
    // The healthy quant converted; the broken sibling kept its directory, so the
    // shared model directory survives with only the unconverted quant in it.
    assert!(!models.join("moonshine-tiny/q8_0").exists());
    assert!(models.join("moonshine-tiny/q4_k").is_dir());
    assert_eq!(list_installed_packs(home.path()).unwrap().len(), 1);
}

#[test]
fn startup_migration_is_the_same_operation() {
    let home = tempfile::tempdir().unwrap();
    let models = home.path().join("models");
    write_legacy_install(home.path(), &models, "moonshine-tiny", "q8_0", None);

    let report = migrate_model_store_at_startup(home.path()).unwrap();
    assert_eq!(report.migrated, vec!["moonshine-tiny:q8".to_string()]);
    assert_eq!(list_installed_packs(home.path()).unwrap().len(), 1);
}

#[test]
fn removing_a_migrated_pack_frees_its_object() {
    let home = tempfile::tempdir().unwrap();
    let models = home.path().join("models");
    let (_, bytes) = write_legacy_install(home.path(), &models, "moonshine-tiny", "q8_0", None);
    migrate_legacy_model_store(home.path()).unwrap();
    let occupied = tree_bytes(&models);

    let removed = remove_model_pack(home.path(), "moonshine-tiny:q8")
        .unwrap()
        .expect("pack is installed");
    assert_eq!(removed.pull, "moonshine-tiny:q8");

    // Deleting a model must return the space, not just unlink a few hundred
    // bytes of JSON.
    let remaining = tree_bytes(&models);
    assert!(
        remaining < occupied - (bytes.len() as u64 / 2),
        "removal freed {} of {occupied} bytes",
        occupied - remaining
    );
    assert!(!object_path_for(&models, &sha256_hex(&bytes)).exists());
    assert!(list_installed_packs(home.path()).unwrap().is_empty());
}

/// End-to-end: an upgraded store that carries every leak class at once.
///
/// This mirrors the shape measured on a real machine -- superseded legacy
/// copies, transaction files from a retry loop whose process is long gone, and
/// unreferenced content -- and proves that one migration plus one collection
/// returns the store to exactly its live data, with every ref still resolvable.
#[test]
fn model_store_lifecycle_converts_and_reclaims_a_leaking_store() {
    use crate::model_store_gc::{
        ORPHAN_OBJECT_GRACE, collect_model_store_garbage, model_store_usage, verify_model_store,
    };

    fn write_blob(models: &Path, bytes: &[u8], age: Option<std::time::Duration>) -> String {
        let digest = sha256_hex(bytes);
        let path = object_path_for(models, &digest);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, bytes).unwrap();
        if let Some(age) = age {
            let file = fs::OpenOptions::new().write(true).open(&path).unwrap();
            file.set_times(fs::FileTimes::new().set_modified(std::time::SystemTime::now() - age))
                .unwrap();
        }
        digest
    }

    let home = tempfile::tempdir().unwrap();
    let models = home.path().join("models");

    // Two healthy legacy installs, still in the pre-content-store layout.
    let (_, alpha_bytes) =
        write_legacy_install(home.path(), &models, "moonshine-tiny", "q8_0", None);
    let (_, beta_bytes) = write_legacy_install(home.path(), &models, "whisper-small", "q4_k", None);

    // Transaction files from a retry loop whose process exited: 3 x 2 MiB.
    let staging = models.join("staging");
    fs::create_dir_all(&staging).unwrap();
    let dead = (900_000..999_999)
        .find(|pid| crate::pull::process_is_gone(*pid))
        .unwrap();
    let mut dead_staging_bytes = 0;
    for nonce in 0..3 {
        let path = staging.join(format!("admit-{dead}-{nonce}.tmp"));
        fs::write(&path, vec![9_u8; 2 * 1024 * 1024]).unwrap();
        dead_staging_bytes += 2 * 1024 * 1024_u64;
    }
    // A resumable download that must survive untouched.
    let partial = staging.join(format!("{}-in-flight.oasr.partial", "c".repeat(64)));
    fs::write(&partial, vec![4_u8; 512 * 1024]).unwrap();

    // Unreferenced content: one long past its grace window, one just written.
    let old_orphan = write_blob(
        &models,
        &vec![1_u8; 3 * 1024 * 1024],
        Some(ORPHAN_OBJECT_GRACE * 3),
    );
    let young_orphan = write_blob(&models, &vec![2_u8; 1024 * 1024], None);

    let before = tree_bytes(&models);
    let usage_before = model_store_usage(home.path()).unwrap();
    println!("--- before ---");
    println!("total on disk:        {before} bytes");
    println!(
        "installed (refs):     {} model(s)",
        usage_before.entries.len()
    );
    println!(
        "legacy copies:        {} bytes in {} install(s)",
        usage_before.legacy_copy_bytes, usage_before.legacy_copy_count
    );
    println!(
        "unreferenced objects: {} bytes in {}",
        usage_before.orphan_object_bytes, usage_before.orphan_object_count
    );
    println!(
        "dead staging:         {} bytes in {}",
        usage_before.dead_staging_bytes, usage_before.dead_staging_count
    );
    println!(
        "reclaimable now:      {} bytes",
        usage_before.reclaimable_bytes
    );

    // Nothing is visible yet: the store has one readable layout, and these packs
    // are not in it.
    assert!(list_installed_packs(home.path()).unwrap().is_empty());
    assert_eq!(usage_before.legacy_copy_count, 2);
    assert_eq!(usage_before.dead_staging_bytes, dead_staging_bytes);

    let migration = migrate_model_store_at_startup(home.path()).unwrap();
    println!("--- migration ---");
    println!("migrated:  {:?}", migration.migrated);
    println!("reclaimed: {} bytes", migration.reclaimed_bytes);
    println!("failures:  {:?}", migration.failures);
    assert_eq!(migration.migrated.len(), 2);
    assert!(migration.failures.is_empty());

    let gc = collect_model_store_garbage(home.path()).unwrap();
    println!("--- collection ---");
    println!("removed objects: {}", gc.removed_objects.len());
    println!("removed scratch: {}", gc.removed_staging.len());
    println!("freed:           {} bytes", gc.freed_bytes);
    println!("kept young orphans: {}", gc.retained_young_orphans);

    let after = tree_bytes(&models);
    let usage_after = model_store_usage(home.path()).unwrap();
    println!("--- after ---");
    println!("total on disk:        {after} bytes");
    println!(
        "installed (refs):     {} model(s)",
        usage_after.entries.len()
    );
    println!("legacy copies:        {}", usage_after.legacy_copy_count);
    println!(
        "unreferenced objects: {} bytes in {}",
        usage_after.orphan_object_bytes, usage_after.orphan_object_count
    );
    println!("dead staging:         {}", usage_after.dead_staging_count);
    for entry in &usage_after.entries {
        println!("  {} {} bytes", entry.pull, entry.size_bytes);
    }

    // The aged orphan and every dead transaction file are gone; the young orphan
    // is held by its grace window, and the resumable download is untouched.
    assert_eq!(gc.removed_objects, vec![old_orphan]);
    assert_eq!(gc.removed_staging.len(), 3);
    assert_eq!(gc.retained_young_orphans, 1);
    assert_eq!(gc.freed_bytes, 3 * 1024 * 1024 + dead_staging_bytes);
    assert!(partial.is_file(), "a resumable download must survive GC");
    assert!(object_path_for(&models, &young_orphan).is_file());

    // Both models survived, are served from content-addressed storage, and pass
    // a full re-hash.
    let packs = list_installed_packs(home.path()).unwrap();
    assert_eq!(packs.len(), 2);
    assert_eq!(
        packs
            .iter()
            .map(|pack| pack.pull.as_str())
            .collect::<Vec<_>>(),
        vec!["moonshine-tiny:q8", "whisper-small:q8"]
    );
    for pack in &packs {
        assert!(pack.path.starts_with(models.join("objects/sha256")));
    }
    let verification = verify_model_store(home.path()).unwrap();
    assert!(verification.is_ok(), "{:?}", verification.checked);
    assert_eq!(verification.checked.len(), 2);

    // Live data is intact byte for byte.
    let alpha = packs
        .iter()
        .find(|pack| pack.model_id == "moonshine-tiny")
        .unwrap();
    let beta = packs
        .iter()
        .find(|pack| pack.model_id == "whisper-small")
        .unwrap();
    assert_eq!(fs::read(&alpha.path).unwrap(), alpha_bytes);
    assert_eq!(fs::read(&beta.path).unwrap(), beta_bytes);

    // What is left is the live packs, the young orphan, and the in-flight
    // download -- nothing else.
    let live = alpha_bytes.len() as u64 + beta_bytes.len() as u64;
    let expected_floor = live + 1024 * 1024 + 512 * 1024;
    assert!(
        after >= expected_floor && after < expected_floor + 64 * 1024,
        "after {after} should be live data ({live}) plus the young orphan and \
         the in-flight download, plus only metadata"
    );
    assert!(
        after < before,
        "the store must shrink (before {before}, after {after})"
    );
}
