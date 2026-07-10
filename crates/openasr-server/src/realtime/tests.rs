//! Unit tests for the realtime module. Pure code-motion from `realtime.rs`.

use std::{collections::BTreeMap, fs, sync::OnceLock};

use super::*;
use crate::PairingCredentialState;

fn test_distribution() -> DistributionContext {
    let temp = tempfile::tempdir().unwrap();
    let openasr_home = temp.path().to_path_buf();
    std::mem::forget(temp);
    DistributionContext::new(crate::DistributionRuntime {
        openasr_home: Some(openasr_home),
        catalog_url: None,
    })
}

struct EnvVarGuard {
    key: &'static str,
    previous: Option<String>,
}

impl EnvVarGuard {
    fn set(key: &'static str, value: impl AsRef<std::ffi::OsStr>) -> Self {
        let previous = std::env::var(key).ok();
        unsafe { std::env::set_var(key, value) };
        Self { key, previous }
    }

    fn unset(key: &'static str) -> Self {
        let previous = std::env::var(key).ok();
        unsafe { std::env::remove_var(key) };
        Self { key, previous }
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        match self.previous.take() {
            Some(value) => unsafe { std::env::set_var(self.key, value) },
            None => unsafe { std::env::remove_var(self.key) },
        }
    }
}

async fn speaker_embedder_env_lock() -> tokio::sync::MutexGuard<'static, ()> {
    static LOCK: OnceLock<tokio::sync::Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| tokio::sync::Mutex::new(()))
        .lock()
        .await
}

/// `idle_activity::NATIVE_UNLOAD_GENERATION` is a process-wide counter (it
/// has to be: a real idle-unload evicts every worker thread's resident
/// runtime, not just one). Only the two warm-up/generation tests below
/// mutate it directly (via `bump_native_unload_generation`); this lock keeps
/// those two from racing each other under `cargo test`'s default test-thread
/// parallelism, the same way `speaker_embedder_env_lock` above serializes
/// tests that share process-wide env state.
async fn native_unload_generation_test_lock() -> tokio::sync::MutexGuard<'static, ()> {
    static LOCK: OnceLock<tokio::sync::Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| tokio::sync::Mutex::new(()))
        .lock()
        .await
}

fn test_native_streaming_worker_key(name: &str) -> NativeStreamingWorkerKey {
    NativeStreamingWorkerKey::new(
        PathBuf::from(format!("/test/native-streaming/{name}")),
        openasr_core::NativeAsrHardwareTarget::Cpu,
        None,
    )
}

#[test]
fn native_streaming_worker_key_canonicalizes_existing_pack_paths() {
    let temp = tempfile::tempdir().unwrap();
    let pack_dir = temp.path().join("pack");
    fs::create_dir_all(&pack_dir).unwrap();
    let raw_pack_dir = pack_dir.join("..").join("pack");

    let key_from_raw = NativeStreamingWorkerKey::new(
        raw_pack_dir,
        openasr_core::NativeAsrHardwareTarget::Accelerated,
        Some(4),
    );
    let key_from_canonical = NativeStreamingWorkerKey::new(
        pack_dir.canonicalize().unwrap(),
        openasr_core::NativeAsrHardwareTarget::Accelerated,
        Some(4),
    );

    assert_eq!(key_from_raw, key_from_canonical);
    assert_eq!(
        key_from_raw.model_pack_path,
        pack_dir.canonicalize().unwrap()
    );
}

#[test]
fn partial_prefix_wer_scores_first_partial_against_final_prefix() {
    assert_eq!(
        openasr_core::word_prefix_error_rate("And so.", "And so, my fellow Americans, ask not.")
            .unwrap(),
        0.0
    );
    assert_eq!(
        openasr_core::word_prefix_error_rate("Answer.", "And so, my fellow Americans, ask not.")
            .unwrap(),
        1.0
    );
}

fn remote_compute_headers(token: Option<&str>) -> HeaderMap {
    let mut headers = HeaderMap::new();
    headers.insert(
        crate::REMOTE_COMPUTE_HEADER,
        crate::REMOTE_COMPUTE_CLIENT_VALUE.parse().unwrap(),
    );
    if let Some(token) = token {
        headers.insert(
            axum::http::header::AUTHORIZATION,
            format!("Bearer {token}").parse().unwrap(),
        );
    }
    headers
}

#[test]
fn realtime_remote_compute_history_skip_requires_paired_device_token() {
    let headers = remote_compute_headers(None);

    assert!(
        should_record_history_for_headers(&headers, &ServerAuth::disabled()),
        "remote-compute header alone must not suppress server realtime history"
    );
    assert!(
        should_record_history_for_headers(
            &remote_compute_headers(Some("remote-secret")),
            &ServerAuth::bearer("remote-secret")
        ),
        "static bearer auth is not a paired remote-compute client"
    );
    assert!(
        should_record_history_for_headers(
            &remote_compute_headers(Some("admin-secret")),
            &ServerAuth::pairing("admin-secret")
        ),
        "pairing admin token is not a paired remote-compute client"
    );

    let auth = ServerAuth::pairing("admin-secret");
    let request = auth.create_pairing_request("Test Desktop").unwrap();
    auth.approve_pairing_request(&request.request_id).unwrap();
    let PairingCredentialState::Ready(credential) =
        auth.pairing_credential(&request.request_id).unwrap()
    else {
        panic!("expected approved pairing credential");
    };
    assert!(!should_record_history_for_headers(
        &remote_compute_headers(Some(&credential.bearer_token)),
        &auth
    ));
}

fn frame(seq: u64, start_ms: u64, sample: i16) -> RealtimeAudioFrame {
    RealtimeAudioFrame::new(
        seq,
        start_ms,
        RealtimeAudioFormat::pcm16_mono_16khz(),
        vec![sample; 320],
    )
    .unwrap()
}

fn pcm16_frame_bytes(sample: i16) -> Vec<u8> {
    std::iter::repeat_n(sample.to_le_bytes(), 320)
        .flatten()
        .collect()
}

fn pcm16_samples_to_bytes(samples: &[i16]) -> Vec<u8> {
    samples
        .iter()
        .flat_map(|sample| sample.to_le_bytes())
        .collect()
}

fn required_env_path(name: &str) -> PathBuf {
    let value = std::env::var(name).unwrap_or_else(|_| {
        panic!("{name} must point to a local file for this ignored smoke test")
    });
    let path = PathBuf::from(value);
    assert!(
        path.exists(),
        "{name} path does not exist: {}",
        path.display()
    );
    path
}

fn write_native_streaming_fixture_pack(
    path: &std::path::Path,
    model_id: &str,
    family: &str,
    architecture: &str,
    audio_frontend: &str,
    decode_policy: &str,
    tokenizer: &str,
) {
    let mut metadata = BTreeMap::new();
    metadata.insert("openasr.model.id".to_string(), model_id.to_string());
    metadata.insert(
        openasr_core::models::oasr_metadata::OASR_METADATA_KEY_PACKAGE_VERSION.to_string(),
        openasr_core::models::oasr_metadata::OASR_PACKAGE_VERSION_V1.to_string(),
    );
    metadata.insert(
        openasr_core::models::oasr_metadata::OASR_METADATA_KEY_MODEL_FAMILY.to_string(),
        family.to_string(),
    );
    metadata.insert(
        openasr_core::models::oasr_metadata::OASR_METADATA_KEY_MODEL_ARCHITECTURE.to_string(),
        architecture.to_string(),
    );
    metadata.insert(
        openasr_core::models::oasr_metadata::OASR_METADATA_KEY_AUDIO_FRONTEND.to_string(),
        audio_frontend.to_string(),
    );
    metadata.insert(
        openasr_core::models::oasr_metadata::OASR_METADATA_KEY_DECODE_POLICY.to_string(),
        decode_policy.to_string(),
    );
    metadata.insert(
        openasr_core::GGML_TOKENIZER_ID_KEY.to_string(),
        tokenizer.to_string(),
    );
    let spec = openasr_core::testing::TinyGgufFixtureSpec::new(metadata);
    openasr_core::testing::write_tiny_gguf_runtime_source(path, &spec)
        .expect("write native streaming fixture pack");
}

fn write_xasr_streaming_fixture_pack(path: &std::path::Path, model_id: &str) {
    write_native_streaming_fixture_pack(
        path,
        model_id,
        "xasr-zipformer",
        openasr_core::XASR_ZIPFORMER_GGML_ARCHITECTURE_ID,
        openasr_core::XASR_ZIPFORMER_AUDIO_FRONTEND_ID,
        openasr_core::XASR_ZIPFORMER_DECODE_POLICY_ID,
        openasr_core::XASR_ZIPFORMER_TOKENIZER_ID,
    );
}

fn write_qwen_streaming_fixture_pack(path: &std::path::Path, model_id: &str) {
    write_native_streaming_fixture_pack(
        path,
        model_id,
        openasr_core::QWEN3_ASR_MODEL_FAMILY,
        openasr_core::QWEN3_ASR_GGML_ARCHITECTURE_ID,
        openasr_core::QWEN3_ASR_AUDIO_FRONTEND_ID,
        openasr_core::QWEN3_ASR_DECODE_POLICY_ID,
        openasr_core::QWEN3_ASR_TOKENIZER_ID,
    );
}

fn write_moonshine_streaming_fixture_pack(path: &std::path::Path, model_id: &str) {
    write_native_streaming_fixture_pack(
        path,
        model_id,
        openasr_core::MOONSHINE_MODEL_FAMILY,
        openasr_core::MOONSHINE_GGML_ARCHITECTURE_ID,
        openasr_core::MOONSHINE_AUDIO_FRONTEND_ID,
        openasr_core::MOONSHINE_DECODE_POLICY_ID,
        openasr_core::MOONSHINE_TOKENIZER_ID,
    );
}

fn read_wav_mono_16k_pcm16(path: &std::path::Path) -> Result<Vec<i16>, String> {
    let bytes =
        fs::read(path).map_err(|error| format!("could not read '{}': {error}", path.display()))?;
    if bytes.len() < 44 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(format!("'{}' is not a RIFF/WAVE file", path.display()));
    }

    let mut channels = None;
    let mut sample_rate = None;
    let mut bits_per_sample = None;
    let mut data = None;
    let mut i = 12;
    while i + 8 <= bytes.len() {
        let id = &bytes[i..i + 4];
        let size =
            u32::from_le_bytes([bytes[i + 4], bytes[i + 5], bytes[i + 6], bytes[i + 7]]) as usize;
        let start = i + 8;
        let end = start.saturating_add(size).min(bytes.len());
        if id == b"fmt " && size >= 16 && end <= bytes.len() {
            channels = Some(u16::from_le_bytes([bytes[start + 2], bytes[start + 3]]));
            sample_rate = Some(u32::from_le_bytes([
                bytes[start + 4],
                bytes[start + 5],
                bytes[start + 6],
                bytes[start + 7],
            ]));
            bits_per_sample = Some(u16::from_le_bytes([bytes[start + 14], bytes[start + 15]]));
        } else if id == b"data" && end <= bytes.len() {
            data = Some(&bytes[start..end]);
        }
        i += 8 + size + (size & 1);
    }

    if channels != Some(1) || sample_rate != Some(16_000) || bits_per_sample != Some(16) {
        return Err(format!(
            "'{}' must be 16 kHz mono PCM16 WAV (got channels={channels:?}, sample_rate={sample_rate:?}, bits={bits_per_sample:?})",
            path.display()
        ));
    }
    let data = data.ok_or_else(|| format!("'{}' has no data chunk", path.display()))?;
    Ok(data
        .chunks_exact(2)
        .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
        .collect())
}

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(default)
}

fn env_f64(name: &str, default: f64) -> f64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .unwrap_or(default)
}

fn backend_job_for_test(id: &str) -> BackendJob {
    BackendJob {
        utterance_id: TranscriptUtteranceId(format!("utt_{id}")),
        start_ms: 0,
        end_ms: 20,
        segment_id: TranscriptSegmentId(format!("seg_{id}")),
        model_id: "whisper-large-v3-turbo".to_string(),
        language: None,
        task: None,
        prompt: None,
        phrase_bias: None,
        inference_threads: None,
        execution_target: None,
        word_timestamps: false,
        display_name: "realtime-utterance.wav".to_string(),
        temp_wav: tempfile::NamedTempFile::new().unwrap(),
    }
}

fn work_item_for_test(session_key: &str, id: &str) -> RealtimeBackendWorkItem {
    let (result_sender, _result_receiver) = mpsc::channel(4);
    RealtimeBackendWorkItem {
        session_key: session_key.to_string(),
        job: backend_job_for_test(id),
        result_sender,
        cancelled: Arc::new(AtomicBool::new(false)),
    }
}

fn started_controller(session_id: &str, model_id: &str) -> RealtimeSessionController {
    let mut controller = RealtimeSessionController::new(RealtimeSessionConfig::new(
        session_id,
        model_id,
        timestamp_now(),
    ))
    .unwrap();
    controller
        .lifecycle(RealtimeLifecycleAction::Configure, timestamp_now())
        .unwrap();
    controller
        .lifecycle(RealtimeLifecycleAction::StartAudio, timestamp_now())
        .unwrap();
    controller
}

struct TestServerNativeSession {
    session_id: String,
    next_seq: u64,
}

impl TestServerNativeSession {
    fn new(session_id: impl Into<String>) -> Self {
        Self {
            session_id: session_id.into(),
            next_seq: 1,
        }
    }

    fn transcript(&mut self, event: RealtimeTranscriptEvent) -> Vec<RealtimeEventEnvelope> {
        let event = RealtimeEvent::Transcript(event);
        let envelope = RealtimeEventEnvelope {
            event_type: event.event_type(),
            session_id: RealtimeSessionId(self.session_id.clone()),
            event_id: openasr_core::RealtimeEventId(format!("evt_{:06}", self.next_seq)),
            seq: self.next_seq,
            created_at: timestamp_now(),
            trace_id: None,
            request_id: None,
            event,
        };
        self.next_seq += 1;
        vec![envelope]
    }
}

impl NativeAsrSession for TestServerNativeSession {
    fn session_id(&self) -> &str {
        &self.session_id
    }

    fn push_audio(
        &mut self,
        frame: RealtimeAudioFrame,
    ) -> Result<Vec<RealtimeEventEnvelope>, openasr_core::NativeAsrError> {
        Ok(self.transcript(RealtimeTranscriptEvent::Partial(
            openasr_core::RealtimeTranscriptPartial {
                utterance_id: TranscriptUtteranceId("utt_native_000001".to_string()),
                segment_id: TranscriptSegmentId("seg_native_000001".to_string()),
                revision: frame.seq,
                text: "native partial".to_string(),
                start_ms: frame.start_ms,
                end_ms: frame.end_ms(),
                is_final: false,
                words: Vec::new(),
                language: None,
                speaker: None,
                speaker_label: None,
                speaker_profile_id: None,
            },
        )))
    }

    fn poll_events(&mut self) -> Result<Vec<RealtimeEventEnvelope>, openasr_core::NativeAsrError> {
        Ok(Vec::new())
    }

    fn finish(&mut self) -> Result<Vec<RealtimeEventEnvelope>, openasr_core::NativeAsrError> {
        Ok(self.transcript(RealtimeTranscriptEvent::Final(
            openasr_core::RealtimeTranscriptFinal {
                utterance_id: TranscriptUtteranceId("utt_native_000001".to_string()),
                segment_id: TranscriptSegmentId("seg_native_000001".to_string()),
                revision: 1,
                text: "native final".to_string(),
                start_ms: 0,
                end_ms: 20,
                is_final: true,
                words: Vec::new(),
                language: None,
                speaker: None,
                speaker_label: None,
                speaker_profile_id: None,
            },
        )))
    }

    fn cancel(&mut self) -> Result<Vec<RealtimeEventEnvelope>, openasr_core::NativeAsrError> {
        Ok(Vec::new())
    }
}

/// Native streaming stub whose `push_audio` can hang past the decode watchdog
/// or fail, to exercise the A2 worker failure paths the real packs can't.
enum StubDecodeBehavior {
    Hang(Duration),
    Fail,
    Panic,
}

struct ConfigurableNativeSession {
    session_id: String,
    behavior: StubDecodeBehavior,
}

impl NativeAsrSession for ConfigurableNativeSession {
    fn session_id(&self) -> &str {
        &self.session_id
    }

    fn push_audio(
        &mut self,
        _frame: RealtimeAudioFrame,
    ) -> Result<Vec<RealtimeEventEnvelope>, openasr_core::NativeAsrError> {
        // Runs on the worker thread, so a real sleep here does not block tokio.
        match &self.behavior {
            StubDecodeBehavior::Hang(duration) => {
                std::thread::sleep(*duration);
                Ok(Vec::new())
            }
            StubDecodeBehavior::Fail => Err(openasr_core::NativeAsrError::SessionFailed {
                message: "stub decode failure".to_string(),
            }),
            // Panic on the worker thread: it unwinds and drops the outcome
            // sender, so the WS task's recv() yields None (worker-died path).
            StubDecodeBehavior::Panic => panic!("stub decode panic"),
        }
    }

    fn poll_events(&mut self) -> Result<Vec<RealtimeEventEnvelope>, openasr_core::NativeAsrError> {
        Ok(Vec::new())
    }

    fn finish(&mut self) -> Result<Vec<RealtimeEventEnvelope>, openasr_core::NativeAsrError> {
        Ok(Vec::new())
    }

    fn cancel(&mut self) -> Result<Vec<RealtimeEventEnvelope>, openasr_core::NativeAsrError> {
        Ok(Vec::new())
    }
}

struct SlowPollNativeSession {
    session_id: String,
    poll_sleep: Duration,
    poll_calls: Option<Arc<AtomicUsize>>,
}

impl NativeAsrSession for SlowPollNativeSession {
    fn session_id(&self) -> &str {
        &self.session_id
    }

    fn push_audio(
        &mut self,
        _frame: RealtimeAudioFrame,
    ) -> Result<Vec<RealtimeEventEnvelope>, openasr_core::NativeAsrError> {
        Ok(Vec::new())
    }

    fn poll_events(&mut self) -> Result<Vec<RealtimeEventEnvelope>, openasr_core::NativeAsrError> {
        if let Some(poll_calls) = &self.poll_calls {
            poll_calls.fetch_add(1, Ordering::AcqRel);
        }
        std::thread::sleep(self.poll_sleep);
        Ok(Vec::new())
    }

    fn finish(&mut self) -> Result<Vec<RealtimeEventEnvelope>, openasr_core::NativeAsrError> {
        Ok(Vec::new())
    }

    fn cancel(&mut self) -> Result<Vec<RealtimeEventEnvelope>, openasr_core::NativeAsrError> {
        Ok(Vec::new())
    }
}

struct BlockingPushPollNativeSession {
    session_id: String,
    push_sleep: Duration,
    poll_calls: Arc<AtomicUsize>,
}

impl NativeAsrSession for BlockingPushPollNativeSession {
    fn session_id(&self) -> &str {
        &self.session_id
    }

    fn push_audio(
        &mut self,
        _frame: RealtimeAudioFrame,
    ) -> Result<Vec<RealtimeEventEnvelope>, openasr_core::NativeAsrError> {
        std::thread::sleep(self.push_sleep);
        Ok(Vec::new())
    }

    fn poll_events(&mut self) -> Result<Vec<RealtimeEventEnvelope>, openasr_core::NativeAsrError> {
        self.poll_calls.fetch_add(1, Ordering::AcqRel);
        Ok(Vec::new())
    }

    fn finish(&mut self) -> Result<Vec<RealtimeEventEnvelope>, openasr_core::NativeAsrError> {
        Ok(Vec::new())
    }

    fn cancel(&mut self) -> Result<Vec<RealtimeEventEnvelope>, openasr_core::NativeAsrError> {
        Ok(Vec::new())
    }
}

struct BlockingCancelableNativeSession {
    session_id: String,
    started: std::sync::mpsc::Sender<()>,
    release: Arc<Mutex<std::sync::mpsc::Receiver<()>>>,
}

impl NativeAsrSession for BlockingCancelableNativeSession {
    fn session_id(&self) -> &str {
        &self.session_id
    }

    fn push_audio(
        &mut self,
        _frame: RealtimeAudioFrame,
    ) -> Result<Vec<RealtimeEventEnvelope>, openasr_core::NativeAsrError> {
        self.started.send(()).expect("started send");
        self.release
            .lock()
            .expect("release mutex")
            .recv()
            .expect("release blocked push");
        Ok(Vec::new())
    }

    fn poll_events(&mut self) -> Result<Vec<RealtimeEventEnvelope>, openasr_core::NativeAsrError> {
        Ok(Vec::new())
    }

    fn finish(&mut self) -> Result<Vec<RealtimeEventEnvelope>, openasr_core::NativeAsrError> {
        Ok(Vec::new())
    }

    fn cancel(&mut self) -> Result<Vec<RealtimeEventEnvelope>, openasr_core::NativeAsrError> {
        Ok(Vec::new())
    }
}

struct SlowWarmNativeSession {
    inner: TestServerNativeSession,
    warm_sleep: Duration,
    warm_calls: Arc<AtomicUsize>,
}

impl NativeAsrSession for SlowWarmNativeSession {
    fn session_id(&self) -> &str {
        self.inner.session_id()
    }

    fn warm_up(&mut self) -> Result<(), openasr_core::NativeAsrError> {
        self.warm_calls.fetch_add(1, Ordering::AcqRel);
        std::thread::sleep(self.warm_sleep);
        Ok(())
    }

    fn push_audio(
        &mut self,
        frame: RealtimeAudioFrame,
    ) -> Result<Vec<RealtimeEventEnvelope>, openasr_core::NativeAsrError> {
        self.inner.push_audio(frame)
    }

    fn poll_events(&mut self) -> Result<Vec<RealtimeEventEnvelope>, openasr_core::NativeAsrError> {
        self.inner.poll_events()
    }

    fn finish(&mut self) -> Result<Vec<RealtimeEventEnvelope>, openasr_core::NativeAsrError> {
        self.inner.finish()
    }

    fn cancel(&mut self) -> Result<Vec<RealtimeEventEnvelope>, openasr_core::NativeAsrError> {
        self.inner.cancel()
    }
}

struct MultiFinalizeNativeSession {
    inner: TestServerNativeSession,
    utterance_index: u64,
}

impl MultiFinalizeNativeSession {
    fn utterance_id(&self) -> TranscriptUtteranceId {
        TranscriptUtteranceId(format!("utt_native_{:06}", self.utterance_index))
    }

    fn segment_id(&self) -> TranscriptSegmentId {
        TranscriptSegmentId(format!("seg_native_{:06}", self.utterance_index))
    }
}

impl NativeAsrSession for MultiFinalizeNativeSession {
    fn session_id(&self) -> &str {
        self.inner.session_id()
    }

    fn push_audio(
        &mut self,
        frame: RealtimeAudioFrame,
    ) -> Result<Vec<RealtimeEventEnvelope>, openasr_core::NativeAsrError> {
        Ok(self.inner.transcript(RealtimeTranscriptEvent::Partial(
            openasr_core::RealtimeTranscriptPartial {
                utterance_id: self.utterance_id(),
                segment_id: self.segment_id(),
                revision: frame.seq,
                text: format!("partial {}", self.utterance_index),
                start_ms: frame.start_ms,
                end_ms: frame.end_ms(),
                is_final: false,
                words: Vec::new(),
                language: None,
                speaker: None,
                speaker_label: None,
                speaker_profile_id: None,
            },
        )))
    }

    fn poll_events(&mut self) -> Result<Vec<RealtimeEventEnvelope>, openasr_core::NativeAsrError> {
        Ok(Vec::new())
    }

    fn finalize_utterance(
        &mut self,
    ) -> Result<Vec<RealtimeEventEnvelope>, openasr_core::NativeAsrError> {
        let event = self.inner.transcript(RealtimeTranscriptEvent::Final(
            openasr_core::RealtimeTranscriptFinal {
                utterance_id: self.utterance_id(),
                segment_id: self.segment_id(),
                revision: self.utterance_index.saturating_mul(10),
                text: format!("final {}", self.utterance_index),
                start_ms: 0,
                end_ms: 20,
                is_final: true,
                words: Vec::new(),
                language: None,
                speaker: None,
                speaker_label: None,
                speaker_profile_id: None,
            },
        ));
        self.utterance_index = self.utterance_index.saturating_add(1);
        Ok(event)
    }

    fn finish(&mut self) -> Result<Vec<RealtimeEventEnvelope>, openasr_core::NativeAsrError> {
        self.finalize_utterance()
    }

    fn cancel(&mut self) -> Result<Vec<RealtimeEventEnvelope>, openasr_core::NativeAsrError> {
        self.inner.cancel()
    }
}

struct ThreadRecordingNativeSession {
    session_id: String,
    threads: Arc<Mutex<Vec<std::thread::ThreadId>>>,
}

impl NativeAsrSession for ThreadRecordingNativeSession {
    fn session_id(&self) -> &str {
        &self.session_id
    }

    fn push_audio(
        &mut self,
        _frame: RealtimeAudioFrame,
    ) -> Result<Vec<RealtimeEventEnvelope>, openasr_core::NativeAsrError> {
        self.threads
            .lock()
            .expect("thread recorder mutex poisoned")
            .push(std::thread::current().id());
        Ok(Vec::new())
    }

    fn poll_events(&mut self) -> Result<Vec<RealtimeEventEnvelope>, openasr_core::NativeAsrError> {
        Ok(Vec::new())
    }

    fn finish(&mut self) -> Result<Vec<RealtimeEventEnvelope>, openasr_core::NativeAsrError> {
        Ok(Vec::new())
    }

    fn cancel(&mut self) -> Result<Vec<RealtimeEventEnvelope>, openasr_core::NativeAsrError> {
        Ok(Vec::new())
    }
}

fn first_error_code(events: &[RealtimeEventEnvelope]) -> Option<RealtimeErrorCode> {
    events.iter().find_map(|event| match &event.event {
        RealtimeEvent::Error(error) => Some(error.code),
        _ => None,
    })
}

async fn drain_native_until_backend_failed(session: &mut WsSession) {
    tokio::time::timeout(Duration::from_secs(1), async {
        loop {
            let _ = session.drain_native_streaming_outcomes().await;
            if session.backend_failed {
                return;
            }
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    })
    .await
    .expect("native streaming session should fail within timeout");
}

async fn recv_native_event(
    session: &mut WsSession,
    event_receiver: &mut mpsc::Receiver<RealtimeEventEnvelope>,
) -> RealtimeEventEnvelope {
    tokio::time::timeout(Duration::from_secs(1), async {
        loop {
            session.drain_native_streaming_outcomes().await.unwrap();
            if let Ok(event) = event_receiver.try_recv() {
                return event;
            }
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    })
    .await
    .expect("native streaming session should emit an event within timeout")
}

async fn start_energy_fallback_test_session(
    session: &mut WsSession,
    source_name: &str,
) -> Result<(), ()> {
    let vad = ClientVadConfig {
        engine: Some("energy".to_string()),
        ..Default::default()
    }
    .into_vad_config(DEFAULT_FRAME_DURATION_MS);
    let mut config = RealtimeSessionConfig::new(
        session.session_id.0.clone(),
        "fallback-test-model",
        timestamp_now(),
    );
    config.vad = vad;
    config.buffer = realtime_buffer_config(DEFAULT_FRAME_DURATION_MS, vad).unwrap();
    let mut controller = RealtimeSessionController::new(config).unwrap();
    session.source_name = Some(source_name.to_string());
    session.spawn_backend_worker();
    let created = controller.session_created_event(timestamp_now());
    session.emit_envelope(created).await?;
    let configured = controller
        .lifecycle(RealtimeLifecycleAction::Configure, timestamp_now())
        .unwrap();
    session.emit_envelope(configured).await?;
    let started = controller
        .lifecycle(RealtimeLifecycleAction::StartAudio, timestamp_now())
        .unwrap();
    session.emit_envelope(started).await?;
    session.controller = Some(controller);
    Ok(())
}

async fn start_test_session_with_vad(
    session: &mut WsSession,
    source_name: &str,
    vad: VadConfig,
) -> Result<(), ()> {
    let mut config = RealtimeSessionConfig::new(
        session.session_id.0.clone(),
        "native-vad-test-model",
        timestamp_now(),
    );
    config.vad = vad;
    config.buffer = realtime_buffer_config(DEFAULT_FRAME_DURATION_MS, vad).unwrap();
    let mut controller = RealtimeSessionController::new(config).unwrap();
    session.source_name = Some(source_name.to_string());
    let created = controller.session_created_event(timestamp_now());
    session.emit_envelope(created).await?;
    let configured = controller
        .lifecycle(RealtimeLifecycleAction::Configure, timestamp_now())
        .unwrap();
    session.emit_envelope(configured).await?;
    let started = controller
        .lifecycle(RealtimeLifecycleAction::StartAudio, timestamp_now())
        .unwrap();
    session.emit_envelope(started).await?;
    session.controller = Some(controller);
    Ok(())
}

#[tokio::test]
async fn native_streaming_decode_error_fails_closed() {
    let (event_sender, mut event_receiver) = mpsc::channel(8);
    let mut session = WsSession::new(ServerRuntime::default(), test_distribution(), event_sender);
    session
        .attach_native_streaming_session(
            test_native_streaming_worker_key("decode-error"),
            Box::new(ConfigurableNativeSession {
                session_id: session.session_id.0.clone(),
                behavior: StubDecodeBehavior::Fail,
            }),
        )
        .await
        .unwrap();

    session.handle_binary(&vec![0; 640]).await.unwrap();
    drain_native_until_backend_failed(&mut session).await;
    assert!(session.backend_failed);

    let mut events = Vec::new();
    while let Ok(event) = event_receiver.try_recv() {
        events.push(event);
    }
    assert_eq!(
        first_error_code(&events),
        Some(RealtimeErrorCode::BackendCrashed)
    );
}

#[tokio::test]
async fn native_streaming_decode_timeout_fails_closed() {
    let (event_sender, mut event_receiver) = mpsc::channel(8);
    let mut session = WsSession::new(ServerRuntime::default(), test_distribution(), event_sender);
    // Watchdog far below the stub's hang so the round-trip times out.
    session.native_decode_timeout = Duration::from_millis(20);
    session
        .attach_native_streaming_session(
            test_native_streaming_worker_key("decode-timeout"),
            Box::new(ConfigurableNativeSession {
                session_id: session.session_id.0.clone(),
                behavior: StubDecodeBehavior::Hang(Duration::from_secs(30)),
            }),
        )
        .await
        .unwrap();

    session.handle_binary(&vec![0; 640]).await.unwrap();
    drain_native_until_backend_failed(&mut session).await;
    assert!(session.backend_failed);

    let mut events = Vec::new();
    while let Ok(event) = event_receiver.try_recv() {
        events.push(event);
    }
    assert_eq!(
        first_error_code(&events),
        Some(RealtimeErrorCode::BackendCrashed)
    );
}

#[tokio::test]
async fn native_streaming_decode_worker_death_fails_closed() {
    let (event_sender, mut event_receiver) = mpsc::channel(8);
    let mut session = WsSession::new(ServerRuntime::default(), test_distribution(), event_sender);
    // The worker thread panics mid-decode: it unwinds and drops the outcome
    // sender, so the WS task observes the worker-died (recv == None) path.
    session
        .attach_native_streaming_session(
            test_native_streaming_worker_key("worker-death"),
            Box::new(ConfigurableNativeSession {
                session_id: session.session_id.0.clone(),
                behavior: StubDecodeBehavior::Panic,
            }),
        )
        .await
        .unwrap();

    session.handle_binary(&vec![0; 640]).await.unwrap();
    drain_native_until_backend_failed(&mut session).await;
    assert!(session.backend_failed);

    let mut events = Vec::new();
    while let Ok(event) = event_receiver.try_recv() {
        events.push(event);
    }
    assert_eq!(
        first_error_code(&events),
        Some(RealtimeErrorCode::BackendCrashed)
    );
}

#[tokio::test]
async fn native_streaming_cancel_on_transport_close_detaches_worker() {
    let (event_sender, _event_receiver) = mpsc::channel(8);
    let mut session = WsSession::new(ServerRuntime::default(), test_distribution(), event_sender);
    session
        .attach_native_streaming_session(
            test_native_streaming_worker_key("transport-close"),
            Box::new(TestServerNativeSession::new(session.session_id.0.clone())),
        )
        .await
        .unwrap();

    // transport_closed detaches the blocked-capable worker and drops the
    // session-local handle without waiting for a terminal decode outcome.
    session
        .finish_native_streaming_session(false, true)
        .await
        .unwrap();

    assert!(session.native_streaming.is_none());
    assert!(session.closed);
}

#[tokio::test]
async fn native_streaming_cancel_emits_closed_without_waiting_for_blocked_decode() {
    let (event_sender, mut event_receiver) = mpsc::channel(8);
    let mut session = WsSession::new(ServerRuntime::default(), test_distribution(), event_sender);
    session.controller = Some(started_controller(
        &session.session_id.0,
        "whisper-large-v3-turbo",
    ));
    let (started_sender, started_receiver) = std::sync::mpsc::channel();
    let (release_sender, release_receiver) = std::sync::mpsc::channel();
    session
        .attach_native_streaming_session(
            test_native_streaming_worker_key("cancel-blocked-decode"),
            Box::new(BlockingCancelableNativeSession {
                session_id: session.session_id.0.clone(),
                started: started_sender,
                release: Arc::new(Mutex::new(release_receiver)),
            }),
        )
        .await
        .unwrap();
    session
        .send_native_streaming_command(NativeStreamingCommand::PushAudio(frame(1, 0, 1)))
        .await
        .unwrap();
    started_receiver
        .recv_timeout(Duration::from_secs(1))
        .expect("blocked decode started");

    let started_at = Instant::now();
    assert!(session.cancel("client_cancelled").await.is_err());
    assert!(
        started_at.elapsed() < Duration::from_millis(200),
        "cancel waited for the blocked decode"
    );
    assert!(session.native_streaming.is_none());
    assert!(session.closed);
    let events = collect_events(&mut event_receiver).await;
    assert_eq!(
        first_error_code(&events),
        Some(RealtimeErrorCode::Cancelled)
    );
    assert!(
        events
            .iter()
            .any(|event| event.event_type == "session.closed")
    );

    release_sender.send(()).expect("release blocked decode");
}

#[tokio::test]
async fn native_streaming_worker_reuses_thread_across_sessions_with_same_key() {
    let key = test_native_streaming_worker_key("reuse-thread");
    let threads = Arc::new(Mutex::new(Vec::new()));

    for _ in 0..2 {
        let (event_sender, _event_receiver) = mpsc::channel(8);
        let mut session =
            WsSession::new(ServerRuntime::default(), test_distribution(), event_sender);
        session
            .attach_native_streaming_session(
                key.clone(),
                Box::new(ThreadRecordingNativeSession {
                    session_id: session.session_id.0.clone(),
                    threads: threads.clone(),
                }),
            )
            .await
            .unwrap();

        session.handle_binary(&vec![0; 640]).await.unwrap();
        session
            .finish_native_streaming_session(true, false)
            .await
            .unwrap();
        assert!(session.native_streaming.is_none());
    }

    let threads = threads.lock().expect("thread recorder mutex poisoned");
    assert_eq!(threads.len(), 2);
    assert_eq!(threads[0], threads[1]);
}

#[test]
fn native_streaming_worker_prune_releases_only_idle_entries() {
    let key = test_native_streaming_worker_key("hard-release");
    let handle = native_streaming_worker_for_key(key.clone());
    let far_future = Instant::now() + Duration::from_secs(120);

    let _ = prune_idle_native_streaming_workers(far_future, Duration::from_secs(60));
    {
        let registry = SHARED_NATIVE_STREAMING_WORKERS
            .get()
            .expect("native streaming worker registry should be initialized");
        let workers = registry
            .lock()
            .expect("native streaming worker registry mutex poisoned");
        assert!(
            workers.contains_key(&key),
            "active native streaming worker must not be pruned"
        );
    }

    handle.state.release();
    drop(handle);
    let removed = prune_idle_native_streaming_workers(far_future, Duration::from_secs(60));
    assert!(removed >= 1);
    let registry = SHARED_NATIVE_STREAMING_WORKERS
        .get()
        .expect("native streaming worker registry should be initialized");
    let workers = registry
        .lock()
        .expect("native streaming worker registry mutex poisoned");
    assert!(
        !workers.contains_key(&key),
        "idle native streaming worker should be pruned after the release threshold"
    );
}

#[test]
fn into_vad_config_hangover_is_mode_conditional() {
    let saved = std::env::var("OPENASR_VAD").ok();
    // SAFETY: only this test (within the openasr-server test binary) mutates
    // OPENASR_VAD; assertions are sequential and the original is restored.
    unsafe { std::env::remove_var("OPENASR_VAD") };

    let neural = ClientVadConfig {
        engine: Some("neural".to_string()),
        ..Default::default()
    }
    .into_vad_config(DEFAULT_FRAME_DURATION_MS);
    assert_eq!(neural.mode, VadMode::ExternalProbability);
    assert_eq!(
        neural.speech_start_ms,
        openasr_core::diarize::vad::DEFAULT_NEURAL_SPEECH_START_MS
    );
    assert_eq!(
        neural.speech_stop_ms,
        openasr_core::diarize::vad::SHORT_NEURAL_SPEECH_STOP_MS
    );
    assert_eq!(neural.pre_roll_ms, VadConfig::default().pre_roll_ms);

    let energy = ClientVadConfig {
        engine: Some("energy".to_string()),
        ..Default::default()
    }
    .into_vad_config(DEFAULT_FRAME_DURATION_MS);
    assert_eq!(energy.mode, VadMode::Energy);
    assert_eq!(energy.speech_start_ms, VadConfig::default().speech_start_ms);
    assert_eq!(energy.speech_stop_ms, VadConfig::default().speech_stop_ms);

    // An explicit client value wins in either mode.
    let pinned = ClientVadConfig {
        engine: Some("neural".to_string()),
        speech_stop_ms: Some(123),
        ..Default::default()
    }
    .into_vad_config(DEFAULT_FRAME_DURATION_MS);
    assert_eq!(pinned.speech_stop_ms, 123);

    match saved {
        Some(value) => unsafe { std::env::set_var("OPENASR_VAD", value) },
        None => unsafe { std::env::remove_var("OPENASR_VAD") },
    }
}

#[test]
fn backend_result_timeout_parses_override_and_falls_back_to_default() {
    assert_eq!(
        parse_backend_result_timeout(None),
        DEFAULT_BACKEND_RESULT_TIMEOUT
    );
    assert_eq!(
        parse_backend_result_timeout(Some("")),
        DEFAULT_BACKEND_RESULT_TIMEOUT
    );
    assert_eq!(
        parse_backend_result_timeout(Some("not-a-number")),
        DEFAULT_BACKEND_RESULT_TIMEOUT
    );
    // 0 is rejected (a zero-length watchdog would fire immediately) -> default.
    assert_eq!(
        parse_backend_result_timeout(Some("0")),
        DEFAULT_BACKEND_RESULT_TIMEOUT
    );
    assert_eq!(
        parse_backend_result_timeout(Some("60")),
        Duration::from_secs(60)
    );
    assert_eq!(
        parse_backend_result_timeout(Some("  120  ")),
        Duration::from_secs(120)
    );
}

#[test]
fn realtime_words_from_transcription_maps_seconds_to_milliseconds() {
    let transcription = Transcription {
        text: "hello world".to_string(),
        segments: vec![openasr_core::Segment {
            start: 0.0,
            end: 1.0,
            text: "hello world".to_string(),
            speaker: None,
            speaker_label: None,
            speaker_profile_id: None,
            words: vec![
                WordTimestamp {
                    word: "hello".to_string(),
                    start: 0.12,
                    end: 0.345,
                    confidence: None,
                },
                WordTimestamp {
                    word: "world".to_string(),
                    start: 0.345,
                    end: 0.9,
                    confidence: None,
                },
                WordTimestamp {
                    word: "clamped".to_string(),
                    start: 1.0,
                    end: 0.5,
                    confidence: None,
                },
            ],
        }],
        longform: None,
        language: None,
    };

    let words = realtime_words_from_transcription(&transcription);

    assert_eq!(
        words,
        vec![
            RealtimeTranscriptWord {
                word: "hello".to_string(),
                start_ms: 120,
                end_ms: 345,
                confidence: None,
            },
            RealtimeTranscriptWord {
                word: "world".to_string(),
                start_ms: 345,
                end_ms: 900,
                confidence: None,
            },
            RealtimeTranscriptWord {
                word: "clamped".to_string(),
                start_ms: 1000,
                end_ms: 1000,
                confidence: None,
            },
        ]
    );
}

#[test]
fn shared_backend_scheduler_keeps_session_fifo_while_coalescing_sessions() {
    let mut pending_by_session = HashMap::new();
    let mut active_sessions = HashSet::new();
    handle_realtime_backend_worker_message(
        RealtimeBackendWorkerMessage::Job(work_item_for_test("s1", "1a")),
        &mut pending_by_session,
        &mut active_sessions,
    );
    handle_realtime_backend_worker_message(
        RealtimeBackendWorkerMessage::Job(work_item_for_test("s1", "1b")),
        &mut pending_by_session,
        &mut active_sessions,
    );
    handle_realtime_backend_worker_message(
        RealtimeBackendWorkerMessage::Job(work_item_for_test("s2", "2a")),
        &mut pending_by_session,
        &mut active_sessions,
    );

    let mut ready =
        take_ready_realtime_backend_items(&mut pending_by_session, &mut active_sessions);
    ready.sort_by(|left, right| left.session_key.cmp(&right.session_key));
    assert_eq!(ready.len(), 2);
    assert_eq!(ready[0].session_key, "s1");
    assert_eq!(ready[0].job.utterance_id.0, "utt_1a");
    assert_eq!(ready[1].session_key, "s2");
    assert_eq!(ready[1].job.utterance_id.0, "utt_2a");
    assert!(active_sessions.contains("s1"));
    assert!(active_sessions.contains("s2"));
    assert_eq!(
        pending_by_session
            .get("s1")
            .expect("second s1 item remains queued")
            .len(),
        1
    );

    handle_realtime_backend_worker_message(
        RealtimeBackendWorkerMessage::Completed {
            session_key: "s1".to_string(),
        },
        &mut pending_by_session,
        &mut active_sessions,
    );
    let ready = take_ready_realtime_backend_items(&mut pending_by_session, &mut active_sessions);
    assert_eq!(ready.len(), 1);
    assert_eq!(ready[0].session_key, "s1");
    assert_eq!(ready[0].job.utterance_id.0, "utt_1b");
}

#[test]
fn websocket_writer_uses_explicit_close_frames_by_termination() {
    match ws_close(1000) {
        Message::Close(Some(frame)) => {
            assert_eq!(frame.code, 1000);
            assert_eq!(frame.reason.as_str(), "openasr_session_closed");
        }
        other => panic!("expected explicit close frame, got {other:?}"),
    }
    match ws_close(1011) {
        Message::Close(Some(frame)) => {
            assert_eq!(frame.code, 1011);
            assert_eq!(frame.reason.as_str(), "openasr_session_error");
        }
        other => panic!("expected explicit close frame, got {other:?}"),
    }
    assert_eq!(
        ws_close_code_for_error(RealtimeErrorCode::StartupConfigError),
        1008
    );
    assert_eq!(
        ws_close_code_for_error(RealtimeErrorCode::UnsupportedAudioFormat),
        1003
    );
    assert_eq!(
        ws_close_code_for_error(RealtimeErrorCode::BackendCrashed),
        1011
    );
    assert_eq!(ws_close_code_for_error(RealtimeErrorCode::Cancelled), 1000);
}

#[tokio::test]
async fn unsupported_legacy_stop_and_flush_controls_fail_closed() {
    for message_type in ["audio.input.stop", "transcript.flush"] {
        let (event_sender, mut event_receiver) = mpsc::channel(2);
        let mut session =
            WsSession::new(ServerRuntime::default(), test_distribution(), event_sender);
        let message = format!(r#"{{"type":"{message_type}"}}"#);

        assert!(session.handle_text(&message).await.is_err());
        let event = event_receiver
            .try_recv()
            .expect("unsupported control emits an error");
        assert_eq!(event.event_type, "error");
        match event.event {
            RealtimeEvent::Error(RealtimeErrorEvent {
                code: RealtimeErrorCode::StartupConfigError,
                message,
                recoverable: false,
            }) => {
                assert!(message.contains("Unsupported realtime control message schema"));
                assert!(message.contains(message_type));
            }
            other => panic!("expected startup_config_error event, got {other:?}"),
        }
    }
}

#[tokio::test]
async fn native_streaming_session_receives_binary_frames_without_file_fallback_worker() {
    let (event_sender, mut event_receiver) = mpsc::channel(8);
    let mut session = WsSession::new(ServerRuntime::default(), test_distribution(), event_sender);
    session
        .attach_native_streaming_session(
            test_native_streaming_worker_key("binary-frames"),
            Box::new(TestServerNativeSession::new(session.session_id.0.clone())),
        )
        .await
        .unwrap();

    session.handle_binary(&vec![0; 640]).await.unwrap();

    assert!(session.controller.is_none());
    assert!(session.backend_jobs.is_none());
    assert_eq!(session.pending_backend_jobs, 0);
    let event = recv_native_event(&mut session, &mut event_receiver).await;
    assert_eq!(event.event_type, "transcript.partial");
    match event.event {
        RealtimeEvent::Transcript(RealtimeTranscriptEvent::Partial(partial)) => {
            assert_eq!(partial.text, "native partial");
            assert_eq!(partial.start_ms, 0);
            assert_eq!(partial.end_ms, 20);
        }
        other => panic!("expected transcript.partial, got {other:?}"),
    }
}

#[tokio::test]
async fn native_streaming_slow_poll_does_not_block_audio_ingest() {
    let (event_sender, _event_receiver) = mpsc::channel(8);
    let mut session = WsSession::new(ServerRuntime::default(), test_distribution(), event_sender);
    session
        .attach_native_streaming_session(
            test_native_streaming_worker_key("slow-poll-ingest"),
            Box::new(SlowPollNativeSession {
                session_id: session.session_id.0.clone(),
                poll_sleep: Duration::from_millis(200),
                poll_calls: None,
            }),
        )
        .await
        .unwrap();

    session.native_had_speech_since_last_poll = true;
    session.poll_native_streaming().await.unwrap();
    assert!(session.native_poll_outstanding > 0);

    tokio::time::timeout(
        Duration::from_millis(30),
        session.handle_binary(&vec![0; 640]),
    )
    .await
    .expect("audio ingest must not wait for the slow Poll")
    .unwrap();
    assert_eq!(session.next_frame_seq, 2);

    tokio::time::sleep(Duration::from_millis(220)).await;
    session.drain_native_streaming_outcomes().await.unwrap();
    session.finish("client_closed", true).await.unwrap();
}

#[tokio::test]
async fn native_streaming_warm_up_does_not_block_audio_ingest() {
    let (event_sender, mut event_receiver) = mpsc::channel(8);
    let mut session = WsSession::new(ServerRuntime::default(), test_distribution(), event_sender);
    let warm_calls = Arc::new(AtomicUsize::new(0));
    session
        .attach_native_streaming_session(
            test_native_streaming_worker_key("slow-warm-ingest"),
            Box::new(SlowWarmNativeSession {
                inner: TestServerNativeSession::new(session.session_id.0.clone()),
                warm_sleep: Duration::from_millis(200),
                warm_calls: Arc::clone(&warm_calls),
            }),
        )
        .await
        .unwrap();

    let started = Instant::now();
    session
        .send_native_streaming_command(NativeStreamingCommand::Warm)
        .await
        .unwrap();
    assert!(
        started.elapsed() < Duration::from_millis(30),
        "Warm must be queued asynchronously, not awaited inline"
    );

    tokio::time::timeout(
        Duration::from_millis(30),
        session.handle_binary(&vec![0; 640]),
    )
    .await
    .expect("audio ingest must not wait for the slow Warm")
    .unwrap();
    assert_eq!(session.next_frame_seq, 2);

    let event = recv_native_event(&mut session, &mut event_receiver).await;
    assert_eq!(event.event_type, "transcript.partial");
    assert_eq!(
        warm_calls.load(Ordering::Acquire),
        1,
        "Warm must actually run (paying the cold build before first speech); \
         audio queued behind it is processed right after"
    );
    session.finish("client_closed", true).await.unwrap();
}

#[tokio::test]
async fn native_streaming_warm_up_runs_immediately_and_once_per_worker() {
    // Two Warm commands with no idle-unload between them must still collapse
    // to one real warm-up; take the shared generation lock so a concurrently
    // running idle-unload-generation test cannot bump the process-wide
    // counter mid-window and make this one flake (see
    // `native_unload_generation_test_lock`'s doc comment).
    let _generation_lock = native_unload_generation_test_lock().await;
    let (event_sender, _event_receiver) = mpsc::channel(8);
    let mut session = WsSession::new(ServerRuntime::default(), test_distribution(), event_sender);
    let warm_calls = Arc::new(AtomicUsize::new(0));
    session
        .attach_native_streaming_session(
            test_native_streaming_worker_key("warm-once"),
            Box::new(SlowWarmNativeSession {
                inner: TestServerNativeSession::new(session.session_id.0.clone()),
                warm_sleep: Duration::from_millis(1),
                warm_calls: Arc::clone(&warm_calls),
            }),
        )
        .await
        .unwrap();

    // Warm runs immediately (no idle grace) so the cold build is paid before
    // the first real decode; a second Warm on the same worker thread is a
    // no-op.
    session
        .send_native_streaming_command(NativeStreamingCommand::Warm)
        .await
        .unwrap();
    session
        .send_native_streaming_command(NativeStreamingCommand::Warm)
        .await
        .unwrap();
    tokio::time::sleep(Duration::from_millis(200)).await;
    session.drain_native_streaming_outcomes().await.unwrap();
    assert_eq!(warm_calls.load(Ordering::Acquire), 1);
    session.finish("client_closed", true).await.unwrap();
}

#[tokio::test]
async fn boot_native_warmup_runs_in_background_without_blocking_a_concurrent_task() {
    // Exercises `attach_and_run_boot_warmup` (the session-agnostic half of
    // `spawn_boot_native_warmup`, which `serve_with_launch_options` fires
    // right after bind) against an artificially slow fake session, in place
    // of a real (and much harder to slow down predictably in a test) model
    // pack. This is the property that actually matters for
    // "/health must not wait on warm-up": whatever spawns this must get
    // control back immediately, and anything else the runtime schedules
    // concurrently must not be starved by the slow warm-up.
    let warm_calls = Arc::new(AtomicUsize::new(0));
    let session = Box::new(SlowWarmNativeSession {
        inner: TestServerNativeSession::new("boot-warmup-nonblocking"),
        warm_sleep: Duration::from_millis(300),
        warm_calls: Arc::clone(&warm_calls),
    });
    let key = test_native_streaming_worker_key("boot-warmup-nonblocking");

    let spawn_started = Instant::now();
    let warmup_handle = tokio::spawn(attach_and_run_boot_warmup(key, session));
    assert!(
        spawn_started.elapsed() < Duration::from_millis(100),
        "spawning the boot warm-up must not itself block"
    );

    // A concurrent tokio task must be free to run to completion while the
    // slow warm-up is still sleeping (on its own dedicated worker OS thread,
    // not the tokio runtime) -- standing in for `/health` staying responsive.
    tokio::time::timeout(Duration::from_millis(100), async { 1 + 1 })
        .await
        .expect("a concurrent tokio task must not be starved by the slow warm-up");

    warmup_handle
        .await
        .expect("boot warmup task must not panic");
    assert_eq!(
        warm_calls.load(Ordering::Acquire),
        1,
        "the slow warm-up must actually have run to completion"
    );
}

#[tokio::test]
async fn health_answers_immediately_while_boot_warmup_is_artificially_slow() {
    use tower::ServiceExt;

    // The literal /health acceptance: with the boot warm-up artificially
    // slowed (injected slow mock), a real GET /health through the real router
    // must still answer immediately -- warm-up must never sit anywhere on the
    // health path.
    let warm_calls = Arc::new(AtomicUsize::new(0));
    let session = Box::new(SlowWarmNativeSession {
        inner: TestServerNativeSession::new("health-vs-slow-warmup"),
        warm_sleep: Duration::from_millis(500),
        warm_calls: Arc::clone(&warm_calls),
    });
    let key = test_native_streaming_worker_key("health-vs-slow-warmup");
    let warmup_started = Instant::now();
    let warmup_handle = tokio::spawn(attach_and_run_boot_warmup(key, session));

    let app = crate::app_with_runtime(ServerRuntime::default());
    let response = tokio::time::timeout(
        Duration::from_millis(200),
        app.oneshot(
            axum::http::Request::builder()
                .uri("/health")
                .body(axum::body::Body::empty())
                .expect("build health request"),
        ),
    )
    .await
    .expect("/health must answer while warm-up is still running, not after it")
    .expect("/health request must succeed");
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    assert!(
        warmup_started.elapsed() < Duration::from_millis(500),
        "/health answered only after the 500ms warm-up window had fully \
         elapsed -- this test then proved nothing about ordering"
    );

    warmup_handle
        .await
        .expect("boot warmup task must not panic");
    assert_eq!(warm_calls.load(Ordering::Acquire), 1);
}

#[tokio::test]
async fn boot_native_warmup_leaves_the_worker_thread_warm_for_the_next_real_attach() {
    // Confirms warm-up dedup: a boot warm-up and a subsequent
    // real WS attach on the SAME worker key share the one thread-local
    // `WARMED_AT_GENERATION` gate (`warm_up_native_streaming_session_once`)
    // -- the real session's own `Warm` command must be a no-op, not a second
    // cold build, as long as no idle-unload has happened in between (see
    // `native_streaming_warm_up_rewarms_after_idle_unload_bumps_the_generation`
    // for that case). Takes the shared generation lock for the same reason as
    // `native_streaming_warm_up_runs_immediately_and_once_per_worker`: this
    // spans two attaches expecting one warm-up, so a concurrent generation
    // bump from another test would otherwise flake it.
    let _generation_lock = native_unload_generation_test_lock().await;
    let warm_calls = Arc::new(AtomicUsize::new(0));
    let key = test_native_streaming_worker_key("boot-warmup-reuse");

    let boot_session = Box::new(SlowWarmNativeSession {
        inner: TestServerNativeSession::new("boot-warmup-reuse-boot"),
        warm_sleep: Duration::from_millis(50),
        warm_calls: Arc::clone(&warm_calls),
    });
    attach_and_run_boot_warmup(key.clone(), boot_session).await;
    assert_eq!(warm_calls.load(Ordering::Acquire), 1);

    let (event_sender, _event_receiver) = mpsc::channel(8);
    let mut real_session =
        WsSession::new(ServerRuntime::default(), test_distribution(), event_sender);
    real_session
        .attach_native_streaming_session(
            key,
            Box::new(SlowWarmNativeSession {
                inner: TestServerNativeSession::new(real_session.session_id.0.clone()),
                warm_sleep: Duration::from_millis(50),
                warm_calls: Arc::clone(&warm_calls),
            }),
        )
        .await
        .unwrap();
    real_session
        .send_native_streaming_command(NativeStreamingCommand::Warm)
        .await
        .unwrap();
    tokio::time::sleep(Duration::from_millis(120)).await;
    real_session
        .drain_native_streaming_outcomes()
        .await
        .unwrap();

    assert_eq!(
        warm_calls.load(Ordering::Acquire),
        1,
        "the real session's own Warm must be a no-op on the already-warmed \
         worker thread -- warm_up() must not run a second time"
    );
    real_session.finish("client_closed", true).await.unwrap();
}

#[tokio::test]
async fn native_streaming_warm_up_stays_once_across_reattach_without_an_idle_unload() {
    // Companion to the generation-bump regression test below: two separate
    // attaches on the SAME worker key, with no idle-unload in between, must
    // still share the one warm-up -- the generation-keyed gate must not
    // regress the plain reuse case `boot_native_warmup_leaves_the_worker_\
    // thread_warm_for_the_next_real_attach` already covers for the
    // boot-warmup/real-attach pairing.
    let _generation_lock = native_unload_generation_test_lock().await;
    let warm_calls = Arc::new(AtomicUsize::new(0));
    let key = test_native_streaming_worker_key("warm-once-across-reattach-no-unload");

    let (event_sender, _event_receiver) = mpsc::channel(8);
    let mut first_session =
        WsSession::new(ServerRuntime::default(), test_distribution(), event_sender);
    first_session
        .attach_native_streaming_session(
            key.clone(),
            Box::new(SlowWarmNativeSession {
                inner: TestServerNativeSession::new(first_session.session_id.0.clone()),
                warm_sleep: Duration::from_millis(1),
                warm_calls: Arc::clone(&warm_calls),
            }),
        )
        .await
        .unwrap();
    first_session
        .send_native_streaming_command(NativeStreamingCommand::Warm)
        .await
        .unwrap();
    tokio::time::sleep(Duration::from_millis(50)).await;
    first_session
        .drain_native_streaming_outcomes()
        .await
        .unwrap();
    assert_eq!(warm_calls.load(Ordering::Acquire), 1);
    first_session.finish("client_closed", true).await.unwrap();

    let (event_sender, _event_receiver) = mpsc::channel(8);
    let mut second_session =
        WsSession::new(ServerRuntime::default(), test_distribution(), event_sender);
    second_session
        .attach_native_streaming_session(
            key,
            Box::new(SlowWarmNativeSession {
                inner: TestServerNativeSession::new(second_session.session_id.0.clone()),
                warm_sleep: Duration::from_millis(1),
                warm_calls: Arc::clone(&warm_calls),
            }),
        )
        .await
        .unwrap();
    second_session
        .send_native_streaming_command(NativeStreamingCommand::Warm)
        .await
        .unwrap();
    tokio::time::sleep(Duration::from_millis(50)).await;
    second_session
        .drain_native_streaming_outcomes()
        .await
        .unwrap();
    assert_eq!(
        warm_calls.load(Ordering::Acquire),
        1,
        "no idle-unload happened between the two attaches, so the second \
         attach's Warm must still be a no-op on the reused worker thread"
    );
    second_session.finish("client_closed", true).await.unwrap();
}

#[tokio::test]
async fn native_streaming_warm_up_rewarms_after_idle_unload_bumps_the_generation() {
    // Regression test for the WARMED/idle-unload race: an opt-in
    // `idle_unload` policy can evict the resident runtime well before the
    // decode worker OS thread's own (much longer) hard-release threshold, so
    // the worker thread stays alive with a stale "already warmed" bit while
    // the runtime it warmed is gone. Simulates that by bumping the process-
    // wide unload generation directly (what the real `idle_unload` reaper
    // does right after calling `unload_idle_native_model_runtime_caches`)
    // between two attaches on the same worker key, and asserts the second
    // attach's `Warm` actually re-runs `warm_up()` instead of reading the
    // stale flag.
    let _generation_lock = native_unload_generation_test_lock().await;
    let warm_calls = Arc::new(AtomicUsize::new(0));
    let key = test_native_streaming_worker_key("rewarm-after-idle-unload-generation-bump");

    let (event_sender, _event_receiver) = mpsc::channel(8);
    let mut first_session =
        WsSession::new(ServerRuntime::default(), test_distribution(), event_sender);
    first_session
        .attach_native_streaming_session(
            key.clone(),
            Box::new(SlowWarmNativeSession {
                inner: TestServerNativeSession::new(first_session.session_id.0.clone()),
                warm_sleep: Duration::from_millis(1),
                warm_calls: Arc::clone(&warm_calls),
            }),
        )
        .await
        .unwrap();
    first_session
        .send_native_streaming_command(NativeStreamingCommand::Warm)
        .await
        .unwrap();
    tokio::time::sleep(Duration::from_millis(50)).await;
    first_session
        .drain_native_streaming_outcomes()
        .await
        .unwrap();
    assert_eq!(warm_calls.load(Ordering::Acquire), 1);
    first_session.finish("client_closed", true).await.unwrap();

    // Simulate an `idle_unload` firing between the two attaches: the worker
    // OS thread for `key` is still alive (attach/detach never tears it
    // down on its own), but the resident runtime it warmed is now gone.
    crate::idle_activity::bump_native_unload_generation();

    let (event_sender, _event_receiver) = mpsc::channel(8);
    let mut second_session =
        WsSession::new(ServerRuntime::default(), test_distribution(), event_sender);
    second_session
        .attach_native_streaming_session(
            key,
            Box::new(SlowWarmNativeSession {
                inner: TestServerNativeSession::new(second_session.session_id.0.clone()),
                warm_sleep: Duration::from_millis(1),
                warm_calls: Arc::clone(&warm_calls),
            }),
        )
        .await
        .unwrap();
    second_session
        .send_native_streaming_command(NativeStreamingCommand::Warm)
        .await
        .unwrap();
    tokio::time::sleep(Duration::from_millis(50)).await;
    second_session
        .drain_native_streaming_outcomes()
        .await
        .unwrap();
    assert_eq!(
        warm_calls.load(Ordering::Acquire),
        2,
        "a generation bump between attaches must force a real re-warm, not \
         reuse a stale thread-local WARMED_AT_GENERATION flag from before \
         the idle-unload evicted the resident runtime"
    );
    second_session.finish("client_closed", true).await.unwrap();
}

#[tokio::test]
async fn attached_native_streaming_session_keeps_the_global_activity_tracker_non_idle() {
    // Integration counterpart of the isolated tracker-logic unit tests in
    // `idle_activity.rs`: proves the real attach/release call sites in
    // `native_worker.rs` (`native_streaming_worker_for_key` /
    // `spawn_native_streaming_worker`) actually drive the process-wide
    // tracker the `idle_unload` reaper reads. Only asserts the "never idle
    // while active" direction against the real (process-wide, shared with
    // every other test in this crate) tracker -- the only direction that
    // stays deterministic under test parallelism, and exactly the safety
    // property that matters: an active session must never be raced by an
    // idle-triggered unload.
    let (event_sender, _event_receiver) = mpsc::channel(8);
    let mut session = WsSession::new(ServerRuntime::default(), test_distribution(), event_sender);
    session
        .attach_native_streaming_session(
            test_native_streaming_worker_key("activity-tracker-stays-non-idle"),
            Box::new(TestServerNativeSession::new(session.session_id.0.clone())),
        )
        .await
        .unwrap();

    assert!(
        !crate::idle_activity::native_activity_is_idle_for(Instant::now(), Duration::ZERO),
        "a live attach must never read idle, even against a zero threshold"
    );

    session.finish("client_closed", true).await.unwrap();
}

#[tokio::test]
async fn native_streaming_finalize_keeps_session_open_for_next_utterance() {
    let (event_sender, mut event_receiver) = mpsc::channel(8);
    let mut session = WsSession::new(ServerRuntime::default(), test_distribution(), event_sender);
    session
        .attach_native_streaming_session(
            test_native_streaming_worker_key("multi-finalize"),
            Box::new(MultiFinalizeNativeSession {
                inner: TestServerNativeSession::new(session.session_id.0.clone()),
                utterance_index: 1,
            }),
        )
        .await
        .unwrap();

    session.handle_binary(&vec![0; 640]).await.unwrap();
    let first_partial = recv_native_event(&mut session, &mut event_receiver).await;
    match first_partial.event {
        RealtimeEvent::Transcript(RealtimeTranscriptEvent::Partial(partial)) => {
            assert_eq!(partial.utterance_id.0, "utt_native_000001");
            assert_eq!(partial.text, "partial 1");
        }
        other => panic!("expected first partial, got {other:?}"),
    }

    let (_, first_final) = session
        .native_streaming_command(NativeStreamingCommand::Finalize)
        .await
        .unwrap();
    assert_eq!(first_final.len(), 1);
    match &first_final[0].event {
        RealtimeEvent::Transcript(RealtimeTranscriptEvent::Final(final_)) => {
            assert_eq!(final_.utterance_id.0, "utt_native_000001");
            assert_eq!(final_.text, "final 1");
        }
        other => panic!("expected first final, got {other:?}"),
    }
    assert!(session.native_streaming.is_some());
    assert!(!session.closed);

    session.handle_binary(&vec![0; 640]).await.unwrap();
    let second_partial = recv_native_event(&mut session, &mut event_receiver).await;
    match second_partial.event {
        RealtimeEvent::Transcript(RealtimeTranscriptEvent::Partial(partial)) => {
            assert_eq!(partial.utterance_id.0, "utt_native_000002");
            assert_eq!(partial.text, "partial 2");
        }
        other => panic!("expected second partial, got {other:?}"),
    }

    let (_, second_final) = session
        .native_streaming_command(NativeStreamingCommand::Finalize)
        .await
        .unwrap();
    assert_eq!(second_final.len(), 1);
    match &second_final[0].event {
        RealtimeEvent::Transcript(RealtimeTranscriptEvent::Final(final_)) => {
            assert_eq!(final_.utterance_id.0, "utt_native_000002");
            assert_eq!(final_.text, "final 2");
        }
        other => panic!("expected second final, got {other:?}"),
    }
    assert!(session.native_streaming.is_some());
    session.finish("client_closed", true).await.unwrap();
}

#[tokio::test]
async fn native_streaming_silence_does_not_queue_poll() {
    let (event_sender, _event_receiver) = mpsc::channel(8);
    let mut session = WsSession::new(ServerRuntime::default(), test_distribution(), event_sender);
    let vad = VadConfig {
        mode: VadMode::Energy,
        energy_threshold: 0.02,
        ..VadConfig::default()
    };
    start_test_session_with_vad(&mut session, "Live", vad)
        .await
        .unwrap();
    session
        .attach_native_streaming_session(
            test_native_streaming_worker_key("silence-no-poll"),
            Box::new(SlowPollNativeSession {
                session_id: session.session_id.0.clone(),
                poll_sleep: Duration::from_millis(200),
                poll_calls: None,
            }),
        )
        .await
        .unwrap();

    session.handle_binary(&vec![0; 640]).await.unwrap();
    assert!(!session.native_had_speech_since_last_poll);
    session.poll_native_streaming().await.unwrap();
    assert_eq!(session.native_poll_outstanding, 0);
    assert_eq!(session.native_poll_outstanding, 0);
    session.finish("client_closed", true).await.unwrap();
}

#[tokio::test]
async fn native_streaming_poll_is_single_flight_and_preserves_latest_speech() {
    let (event_sender, _event_receiver) = mpsc::channel(8);
    let mut session = WsSession::new(ServerRuntime::default(), test_distribution(), event_sender);
    session
        .attach_native_streaming_session(
            test_native_streaming_worker_key("poll-single-flight"),
            Box::new(SlowPollNativeSession {
                session_id: session.session_id.0.clone(),
                poll_sleep: Duration::from_millis(200),
                poll_calls: None,
            }),
        )
        .await
        .unwrap();

    session.native_had_speech_since_last_poll = true;
    session.poll_native_streaming().await.unwrap();
    assert_eq!(session.native_poll_outstanding, 1);
    assert!(session.native_poll_outstanding > 0);

    session.native_had_speech_since_last_poll = true;
    session.poll_native_streaming().await.unwrap();
    assert_eq!(
        session.native_poll_outstanding, 1,
        "second Poll must not queue behind an in-flight heavy decode"
    );
    assert!(
        session.native_had_speech_since_last_poll,
        "latest speech should remain pending for the next tick"
    );

    tokio::time::sleep(Duration::from_millis(220)).await;
    session.drain_native_streaming_outcomes().await.unwrap();
    assert_eq!(session.native_poll_outstanding, 0);
    assert_eq!(session.native_poll_outstanding, 0);
    session.finish("client_closed", true).await.unwrap();
}

#[tokio::test]
async fn native_streaming_skips_queued_poll_when_finalize_is_pending() {
    let (event_sender, _event_receiver) = mpsc::channel(8);
    let mut session = WsSession::new(ServerRuntime::default(), test_distribution(), event_sender);
    let poll_calls = Arc::new(AtomicUsize::new(0));
    session
        .attach_native_streaming_session(
            test_native_streaming_worker_key("skip-poll-before-finalize"),
            Box::new(BlockingPushPollNativeSession {
                session_id: session.session_id.0.clone(),
                push_sleep: Duration::from_millis(150),
                poll_calls: Arc::clone(&poll_calls),
            }),
        )
        .await
        .unwrap();

    session
        .send_native_streaming_command(NativeStreamingCommand::PushAudio(frame(0, 0, 0)))
        .await
        .unwrap();
    session.native_had_speech_since_last_poll = true;
    session.poll_native_streaming().await.unwrap();
    assert_eq!(session.native_poll_outstanding, 1);

    session
        .native_streaming_command(NativeStreamingCommand::Finalize)
        .await
        .unwrap();
    assert_eq!(
        poll_calls.load(Ordering::Acquire),
        0,
        "queued Poll must be skipped once Finalize is pending"
    );
    assert_eq!(session.native_poll_outstanding, 0);
    assert_eq!(session.native_poll_outstanding, 0);
    session.finish("client_closed", true).await.unwrap();
}

#[tokio::test]
async fn native_streaming_poll_uses_raw_speech_before_vad_start_debounce() {
    let (event_sender, mut event_receiver) = mpsc::channel(16);
    let mut session = WsSession::new(ServerRuntime::default(), test_distribution(), event_sender);
    let vad = VadConfig {
        mode: VadMode::Energy,
        speech_start_ms: 1_000,
        energy_threshold: 0.02,
        ..VadConfig::default()
    };
    start_test_session_with_vad(&mut session, "Live", vad)
        .await
        .unwrap();
    session
        .attach_native_streaming_session(
            test_native_streaming_worker_key("raw-speech-gate"),
            Box::new(SlowPollNativeSession {
                session_id: session.session_id.0.clone(),
                poll_sleep: Duration::from_millis(50),
                poll_calls: None,
            }),
        )
        .await
        .unwrap();

    session
        .handle_binary(&pcm16_frame_bytes(16_000))
        .await
        .unwrap();
    let mut event_types = Vec::new();
    while let Ok(event) = event_receiver.try_recv() {
        event_types.push(event.event_type);
    }
    assert!(
        !event_types.contains(&"vad.speech_started"),
        "one raw speech-positive frame must not satisfy speech_start debounce"
    );
    assert!(session.native_had_speech_since_last_poll);

    session.poll_native_streaming().await.unwrap();
    assert!(
        session.native_poll_outstanding > 0,
        "raw speech must gate the first Poll before vad.speech_started"
    );
    assert!(!session.native_had_speech_since_last_poll);

    tokio::time::sleep(Duration::from_millis(70)).await;
    session.drain_native_streaming_outcomes().await.unwrap();
    session.finish("client_closed", true).await.unwrap();
}

#[tokio::test]
#[ignore = "requires OPENASR_NATIVE_STREAMING_SMOKE_PACK and OPENASR_NATIVE_STREAMING_SMOKE_WAV"]
async fn native_realtime_server_smoke_with_real_qwen_pack() {
    let pack_path = required_env_path("OPENASR_NATIVE_STREAMING_SMOKE_PACK");
    let wav_path = required_env_path("OPENASR_NATIVE_STREAMING_SMOKE_WAV");
    let max_ms = std::env::var("OPENASR_NATIVE_STREAMING_SMOKE_MAX_MS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(5_000);
    let poll_ms = std::env::var("OPENASR_NATIVE_STREAMING_SMOKE_POLL_MS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(NATIVE_STREAMING_POLL_INTERVAL.as_millis() as u64);
    let max_first_partial_end_ms = env_u64(
        "OPENASR_NATIVE_STREAMING_SMOKE_MAX_FIRST_PARTIAL_END_MS",
        1_200,
    );
    let max_first_partial_wall_ms = env_u64(
        "OPENASR_NATIVE_STREAMING_SMOKE_MAX_FIRST_PARTIAL_WALL_MS",
        120_000,
    );
    let max_final_wall_ms = env_u64("OPENASR_NATIVE_STREAMING_SMOKE_MAX_FINAL_WALL_MS", 120_000);
    let max_first_partial_prefix_wer = env_f64(
        "OPENASR_NATIVE_STREAMING_SMOKE_MAX_FIRST_PARTIAL_PREFIX_WER",
        0.0,
    );
    let max_session_start_ms =
        env_u64("OPENASR_NATIVE_STREAMING_SMOKE_MAX_SESSION_START_MS", 1_000);
    let pre_audio_idle_ms = env_u64("OPENASR_NATIVE_STREAMING_SMOKE_PRE_AUDIO_IDLE_MS", 0);
    let frame_pace_ms = env_u64("OPENASR_NATIVE_STREAMING_SMOKE_FRAME_PACE_MS", 0);
    let expected_final_text = std::env::var("OPENASR_NATIVE_STREAMING_SMOKE_EXPECTED_FINAL")
        .ok()
        .filter(|value| !value.trim().is_empty());
    let samples = read_wav_mono_16k_pcm16(&wav_path).unwrap();
    let sample_count = samples
        .len()
        .min((max_ms as usize).saturating_mul(16).max(320));
    let (event_sender, mut event_receiver) = mpsc::channel(512);
    let runtime = ServerRuntime {
        backend: openasr_core::BackendKind::Native,
        ffmpeg_bin: None,
        ffmpeg_bin_explicit: false,
        model_pack_path: Some(pack_path),
    };
    let mut session = WsSession::new(runtime, test_distribution(), event_sender);
    session.native_decode_timeout = Duration::from_secs(180);
    let session_start_started = Instant::now();
    session
        .handle_text(
            r#"{"type":"session.start","session":{"model":"qwen3-asr-0.6b","source_name":"Live","partial_results":true,"vad":{"engine":"energy","speech_start_ms":40,"speech_stop_ms":240,"pre_roll_ms":320,"energy_threshold":0.001}}}"#,
        )
        .await
        .unwrap();
    let session_start_ms = session_start_started.elapsed().as_millis() as u64;
    assert!(
        session_start_ms <= max_session_start_ms,
        "session.start took {session_start_ms}ms, above {max_session_start_ms}ms; warm-up must stay asynchronous"
    );

    let frame_samples = 320;
    let poll_every_frames = poll_ms.div_ceil(20).max(1) as usize;
    let mut events = Vec::new();
    let pre_audio_idle_started = Instant::now();
    if pre_audio_idle_ms > 0 {
        let deadline = pre_audio_idle_started + Duration::from_millis(pre_audio_idle_ms);
        loop {
            session.drain_native_streaming_outcomes().await.unwrap();
            while let Ok(event) = event_receiver.try_recv() {
                events.push(event);
            }
            let warm_pending = session
                .native_command_watchdogs
                .iter()
                .any(|(kind, _)| *kind == NativeStreamingCommandKind::Warm);
            if !warm_pending || Instant::now() >= deadline {
                break;
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            tokio::time::sleep(remaining.min(Duration::from_millis(20))).await;
        }
        session.drain_native_streaming_outcomes().await.unwrap();
        while let Ok(event) = event_receiver.try_recv() {
            events.push(event);
        }
    }
    let pre_audio_waited_ms = pre_audio_idle_started.elapsed().as_millis() as u64;
    let warm_pending_after_pre_audio_idle = session
        .native_command_watchdogs
        .iter()
        .any(|(kind, _)| *kind == NativeStreamingCommandKind::Warm);
    let audio_started_at = Instant::now();
    let mut first_partial_wall_ms = None;
    let mut final_wall_ms = None;
    let drain_forwarded_events =
        |receiver: &mut mpsc::Receiver<RealtimeEventEnvelope>,
         events: &mut Vec<RealtimeEventEnvelope>,
         first_partial_wall_ms: &mut Option<u64>,
         final_wall_ms: &mut Option<u64>| {
            while let Ok(event) = receiver.try_recv() {
                if first_partial_wall_ms.is_none()
                    && matches!(
                        &event.event,
                        RealtimeEvent::Transcript(RealtimeTranscriptEvent::Partial(_))
                    )
                {
                    *first_partial_wall_ms = Some(audio_started_at.elapsed().as_millis() as u64);
                }
                if final_wall_ms.is_none()
                    && matches!(
                        &event.event,
                        RealtimeEvent::Transcript(RealtimeTranscriptEvent::Final(_))
                    )
                {
                    *final_wall_ms = Some(audio_started_at.elapsed().as_millis() as u64);
                }
                events.push(event);
            }
        };
    for (index, chunk) in samples[..sample_count].chunks(frame_samples).enumerate() {
        let mut frame = chunk.to_vec();
        if frame.len() < frame_samples {
            frame.resize(frame_samples, 0);
        }
        session
            .handle_binary(&pcm16_samples_to_bytes(&frame))
            .await
            .unwrap();
        if (index + 1) % poll_every_frames == 0 {
            session.poll_native_streaming().await.unwrap();
        }
        session.drain_native_streaming_outcomes().await.unwrap();
        drain_forwarded_events(
            &mut event_receiver,
            &mut events,
            &mut first_partial_wall_ms,
            &mut final_wall_ms,
        );
        if frame_pace_ms > 0 {
            tokio::time::sleep(Duration::from_millis(frame_pace_ms)).await;
        }
    }

    for index in 0..30 {
        session.handle_binary(&vec![0; 640]).await.unwrap();
        if (index + 1) % poll_every_frames == 0 {
            session.poll_native_streaming().await.unwrap();
        }
        session.drain_native_streaming_outcomes().await.unwrap();
        drain_forwarded_events(
            &mut event_receiver,
            &mut events,
            &mut first_partial_wall_ms,
            &mut final_wall_ms,
        );
        if frame_pace_ms > 0 {
            tokio::time::sleep(Duration::from_millis(frame_pace_ms)).await;
        }
    }

    tokio::time::timeout(Duration::from_secs(180), async {
        loop {
            session.drain_native_streaming_outcomes().await.unwrap();
            drain_forwarded_events(
                &mut event_receiver,
                &mut events,
                &mut first_partial_wall_ms,
                &mut final_wall_ms,
            );
            if events
                .iter()
                .any(|event| event.event_type == "transcript.final")
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .expect("real qwen server smoke should finalize");

    let partials = events
        .iter()
        .filter_map(|event| match &event.event {
            RealtimeEvent::Transcript(RealtimeTranscriptEvent::Partial(partial)) => Some(partial),
            _ => None,
        })
        .collect::<Vec<_>>();
    let final_event = events
        .iter()
        .find_map(|event| match &event.event {
            RealtimeEvent::Transcript(RealtimeTranscriptEvent::Final(final_)) => Some(final_),
            _ => None,
        })
        .expect("server smoke must emit a final transcript");
    assert!(
        !partials.is_empty(),
        "server smoke must emit at least one native partial"
    );
    assert!(
        partials[0].end_ms <= max_first_partial_end_ms,
        "first partial ended at {}ms, above {}ms; text={:?}",
        partials[0].end_ms,
        max_first_partial_end_ms,
        partials[0].text
    );
    let first_partial_wall_ms =
        first_partial_wall_ms.expect("server smoke must record first partial wall latency");
    assert!(
        first_partial_wall_ms <= max_first_partial_wall_ms,
        "first partial wall latency was {first_partial_wall_ms}ms, above {max_first_partial_wall_ms}ms; text={:?}",
        partials[0].text
    );
    let final_wall_ms = final_wall_ms.expect("server smoke must record final wall latency");
    assert!(
        final_wall_ms <= max_final_wall_ms,
        "final wall latency was {final_wall_ms}ms, above {max_final_wall_ms}ms; text={:?}",
        final_event.text
    );
    assert!(
        events
            .iter()
            .any(|event| event.event_type == "vad.speech_stopped"),
        "server smoke must finalize from VAD speech_stop"
    );
    assert!(!final_event.text.trim().is_empty());
    if let Some(expected) = expected_final_text.as_deref() {
        assert_eq!(
            openasr_core::normalize_text(&final_event.text),
            openasr_core::normalize_text(expected),
            "native qwen server smoke final drifted"
        );
    }
    let prefix_reference = expected_final_text.as_deref().unwrap_or(&final_event.text);
    let first_partial_prefix_wer =
        openasr_core::word_prefix_error_rate(&partials[0].text, prefix_reference)
            .expect("first partial and final prefix must be non-empty");
    assert!(
        first_partial_prefix_wer <= max_first_partial_prefix_wer,
        "first partial prefix WER {first_partial_prefix_wer:.3} exceeded {max_first_partial_prefix_wer:.3}; first_partial={:?}; reference={:?}",
        partials[0].text,
        prefix_reference
    );
    eprintln!(
        "native server smoke: session_start_ms={}, pre_audio_waited_ms={}, frame_pace_ms={}, warm_pending_after_pre_audio_idle={}, partials={}, first_partial_end_ms={}, first_partial_wall_ms={}, final_wall_ms={}, first_partial_prefix_wer={:.3}, first_partial_text={}, final_text={}",
        session_start_ms,
        pre_audio_waited_ms,
        frame_pace_ms,
        warm_pending_after_pre_audio_idle,
        partials.len(),
        partials[0].end_ms,
        first_partial_wall_ms,
        final_wall_ms,
        first_partial_prefix_wer,
        partials[0].text.trim(),
        final_event.text.trim()
    );
    let finals = events
        .iter()
        .filter_map(|event| match &event.event {
            RealtimeEvent::Transcript(RealtimeTranscriptEvent::Final(final_)) => Some(final_),
            _ => None,
        })
        .collect::<Vec<_>>();
    eprintln!("  TOTAL segment finals = {}", finals.len());
    for (idx, final_) in finals.iter().enumerate() {
        eprintln!(
            "  final[{idx}] end_ms={} text={}",
            final_.end_ms,
            final_.text.trim()
        );
    }
    for (idx, partial) in partials.iter().enumerate() {
        eprintln!(
            "  partial[{idx}] end_ms={} text={}",
            partial.end_ms,
            partial.text.trim()
        );
    }
    if let Some(last) = partials.last() {
        let last_wer =
            openasr_core::word_prefix_error_rate(&last.text, prefix_reference).unwrap_or(1.0);
        eprintln!(
            "  LAST partial prefix WER vs final = {last_wer:.3}; last_partial_text={}",
            last.text.trim()
        );
    }
    session.finish("client_closed", true).await.unwrap();
}

#[tokio::test]
async fn native_streaming_finish_forwards_final_and_records_history() {
    let temp = tempfile::tempdir().unwrap();
    let openasr_home = temp.path().join("home");
    let distribution = DistributionContext::new(crate::DistributionRuntime {
        openasr_home: Some(openasr_home.clone()),
        catalog_url: None,
    });
    std::fs::create_dir_all(&openasr_home).unwrap();
    // History recording is governed by history_retention alone; auto_save
    // stays false to lock in that it does not gate history.
    std::fs::write(
        openasr_home.join("config.json"),
        serde_json::json!({
            "preferences": { "auto_save": false, "history_retention": "last5" }
        })
        .to_string(),
    )
    .unwrap();
    let (event_sender, mut event_receiver) = mpsc::channel(8);
    let mut session = WsSession::new(ServerRuntime::default(), distribution, event_sender);
    session.controller = Some(started_controller(
        "rt_native_finish",
        "whisper-large-v3-turbo",
    ));
    session
        .attach_native_streaming_session(
            test_native_streaming_worker_key("finish-final"),
            Box::new(TestServerNativeSession::new(session.session_id.0.clone())),
        )
        .await
        .unwrap();

    session
        .finish_native_streaming_session(true, false)
        .await
        .unwrap();

    assert!(session.controller.is_some());
    assert!(session.native_streaming.is_none());
    assert!(session.closed);

    let event = event_receiver
        .try_recv()
        .expect("native streaming finish emits a final transcript event");
    assert_eq!(event.event_type, "transcript.final");
    match event.event {
        RealtimeEvent::Transcript(RealtimeTranscriptEvent::Final(final_event)) => {
            assert_eq!(final_event.text, "native final");
            assert_eq!(final_event.start_ms, 0);
            assert_eq!(final_event.end_ms, 20);
        }
        other => panic!("expected transcript.final, got {other:?}"),
    }
    assert!(event_receiver.try_recv().is_err());

    let history = DaemonHistoryStore::open(&openasr_home)
        .list()
        .expect("history list");
    assert_eq!(history.len(), 1);
    let record = &history[0];
    assert_eq!(record.kind, DaemonHistoryKind::Live);
    assert_eq!(record.model, "whisper-large-v3-turbo");
    assert_eq!(record.preview, "native final");
    assert_eq!(record.duration_seconds, Some(0.02));
    let detail = DaemonHistoryStore::open(&openasr_home)
        .get(&record.id)
        .expect("history detail")
        .expect("history detail exists");
    assert_eq!(detail.text, "native final");
}

#[tokio::test]
async fn websocket_session_emits_capabilities_before_start_with_monotonic_sequence() {
    let (event_sender, mut event_receiver) = mpsc::channel(8);
    let mut session = WsSession::new(ServerRuntime::default(), test_distribution(), event_sender);

    session.emit_capabilities().await.unwrap();
    session
        .handle_text(r#"{"type":"session.start","session":{"model":"whisper-large-v3-turbo"}}"#)
        .await
        .unwrap();

    let mut events = Vec::new();
    while let Ok(event) = event_receiver.try_recv() {
        events.push(event);
    }

    assert_eq!(
        events
            .iter()
            .map(|event| event.event_type)
            .collect::<Vec<_>>(),
        vec![
            "session.capabilities",
            "session.created",
            "session.configured",
            "audio.input.started"
        ]
    );
    assert_eq!(
        events.iter().map(|event| event.seq).collect::<Vec<_>>(),
        vec![1, 2, 3, 4]
    );
    match &events[0].event {
        RealtimeEvent::Lifecycle(RealtimeLifecycleEvent::SessionCapabilities(event)) => {
            assert!(event.capabilities.supports_realtime_sessions);
            assert!(!event.capabilities.translation.supported);
            assert!(!event.capabilities.translation.installed);
            // Default runtime (mock backend, no model pack) is
            // file-per-utterance fallback, never frame-sync.
            assert!(!event.capabilities.frame_sync_partials);
            assert_eq!(
                event.capabilities.translation.reason,
                Some("translation_pack_missing")
            );
            assert_eq!(event.frame_duration_ms, DEFAULT_FRAME_DURATION_MS);
            assert_eq!(event.frame_byte_len, 640);
            assert_eq!(event.max_message_bytes, MAX_WS_MESSAGE_BYTES);
        }
        other => panic!("expected session.capabilities event, got {other:?}"),
    }
}

#[tokio::test]
async fn session_capabilities_event_reports_frame_sync_only_for_xasr_zipformer() {
    let temp = tempfile::tempdir().unwrap();

    let xasr_path = temp.path().join("xasr-zipformer-capability-test.oasr");
    write_xasr_streaming_fixture_pack(&xasr_path, "xasr-zipformer-capability-test");
    let xasr_runtime = ServerRuntime {
        backend: openasr_core::BackendKind::Native,
        ffmpeg_bin: None,
        ffmpeg_bin_explicit: false,
        model_pack_path: Some(xasr_path),
    };
    let (xasr_event_sender, mut xasr_event_receiver) = mpsc::channel(8);
    let mut xasr_session = WsSession::new(xasr_runtime, test_distribution(), xasr_event_sender);
    xasr_session.emit_capabilities().await.unwrap();
    match xasr_event_receiver.recv().await.unwrap().event {
        RealtimeEvent::Lifecycle(RealtimeLifecycleEvent::SessionCapabilities(event)) => {
            assert!(event.capabilities.is_true_streaming);
            assert!(event.capabilities.frame_sync_partials);
        }
        other => panic!("expected session.capabilities event, got {other:?}"),
    }

    let qwen_path = temp.path().join("qwen-capability-test.oasr");
    write_qwen_streaming_fixture_pack(&qwen_path, "qwen-capability-test");
    let qwen_runtime = ServerRuntime {
        backend: openasr_core::BackendKind::Native,
        ffmpeg_bin: None,
        ffmpeg_bin_explicit: false,
        model_pack_path: Some(qwen_path),
    };
    let (qwen_event_sender, mut qwen_event_receiver) = mpsc::channel(8);
    let mut qwen_session = WsSession::new(qwen_runtime, test_distribution(), qwen_event_sender);
    qwen_session.emit_capabilities().await.unwrap();
    match qwen_event_receiver.recv().await.unwrap().event {
        RealtimeEvent::Lifecycle(RealtimeLifecycleEvent::SessionCapabilities(event)) => {
            // Qwen also runs a native true-streaming session, but through the
            // buffered re-decode driver -- it must not claim frame-sync partials.
            assert!(event.capabilities.is_true_streaming);
            assert!(!event.capabilities.frame_sync_partials);
        }
        other => panic!("expected session.capabilities event, got {other:?}"),
    }
}

#[test]
fn wav_writer_sets_header_and_data() {
    let mut bytes = Vec::new();
    write_pcm16_mono_16khz_wav(&mut bytes, &[1, -2]).unwrap();
    assert_eq!(&bytes[0..4], b"RIFF");
    assert_eq!(&bytes[8..12], b"WAVE");
    assert_eq!(u16::from_le_bytes([bytes[20], bytes[21]]), 1);
    assert_eq!(u16::from_le_bytes([bytes[22], bytes[23]]), 1);
    assert_eq!(
        u32::from_le_bytes([bytes[24], bytes[25], bytes[26], bytes[27]]),
        16_000
    );
    assert_eq!(u16::from_le_bytes([bytes[34], bytes[35]]), 16);
    assert_eq!(
        u32::from_le_bytes([bytes[40], bytes[41], bytes[42], bytes[43]]),
        4
    );
    assert_eq!(i16::from_le_bytes([bytes[44], bytes[45]]), 1);
    assert_eq!(i16::from_le_bytes([bytes[46], bytes[47]]), -2);
}

#[test]
fn temp_wav_is_removed_after_drop() {
    let utterance = BufferedUtterance {
        utterance_id: TranscriptUtteranceId("utt_1".to_string()),
        start_ms: 0,
        end_ms: 20,
        frames: vec![frame(1, 0, 1000)],
        reason: RealtimeUtteranceEndReason::VadStop,
    };
    let file = write_temp_utterance_wav(&utterance).unwrap();
    let path = file.path().to_path_buf();
    assert!(path.exists());
    drop(file);
    assert!(!path.exists());
}

#[test]
fn fallback_diarization_samples_trim_vad_preroll_and_hangover() {
    let utterance = BufferedUtterance {
        utterance_id: TranscriptUtteranceId("utt_1".to_string()),
        start_ms: 40,
        end_ms: 80,
        frames: vec![
            frame(1, 0, 1000),
            frame(2, 20, 2000),
            frame(3, 40, 3000),
            frame(4, 60, 4000),
            frame(5, 80, 5000),
        ],
        reason: RealtimeUtteranceEndReason::VadStop,
    };

    let samples = utterance_speech_samples_f32(&utterance);

    assert_eq!(samples.len(), 640);
    assert!(
        samples[..320]
            .iter()
            .all(|sample| (*sample - pcm16_sample_to_f32(3000)).abs() < f32::EPSILON)
    );
    assert!(
        samples[320..]
            .iter()
            .all(|sample| (*sample - pcm16_sample_to_f32(4000)).abs() < f32::EPSILON)
    );
}

#[tokio::test]
async fn finish_discards_later_backend_finals_after_backend_error() {
    let (event_sender, mut event_receiver) = mpsc::channel(8);
    let mut session = WsSession::new(ServerRuntime::default(), test_distribution(), event_sender);
    let mut controller = RealtimeSessionController::new(RealtimeSessionConfig::new(
        "test_session",
        "whisper-large-v3-turbo",
        timestamp_now(),
    ))
    .unwrap();
    controller
        .lifecycle(RealtimeLifecycleAction::Configure, timestamp_now())
        .unwrap();
    controller
        .lifecycle(RealtimeLifecycleAction::StartAudio, timestamp_now())
        .unwrap();
    session.controller = Some(controller);
    session.pending_backend_jobs = 2;

    let (result_sender, result_receiver) = mpsc::channel(2);
    session.backend_results = Some(result_receiver);
    result_sender
        .send(BackendResult::Error("backend failed".to_string()))
        .await
        .unwrap();
    result_sender
        .send(BackendResult::Final(BackendSuccess {
            utterance_id: TranscriptUtteranceId("utt_2".to_string()),
            start_ms: 0,
            end_ms: 20,
            segment_id: TranscriptSegmentId("seg_2".to_string()),
            text: "must not be emitted".to_string(),
            language: None,
            words: Vec::new(),
        }))
        .await
        .unwrap();
    drop(result_sender);

    assert!(session.finish("client_closed", true).await.is_err());
    assert_eq!(session.pending_backend_jobs, 0);
    assert!(session.backend_cancelled.load(Ordering::Relaxed));

    let mut event_types = Vec::new();
    while let Ok(event) = event_receiver.try_recv() {
        event_types.push(event.event_type);
    }
    assert_eq!(
        event_types,
        vec!["error", "audio.input.stopped", "session.closed"]
    );
}

#[tokio::test]
async fn finish_remembers_backend_error_seen_before_shutdown() {
    let (event_sender, mut event_receiver) = mpsc::channel(8);
    let mut session = WsSession::new(ServerRuntime::default(), test_distribution(), event_sender);
    let mut controller = RealtimeSessionController::new(RealtimeSessionConfig::new(
        "test_session",
        "whisper-large-v3-turbo",
        timestamp_now(),
    ))
    .unwrap();
    controller
        .lifecycle(RealtimeLifecycleAction::Configure, timestamp_now())
        .unwrap();
    controller
        .lifecycle(RealtimeLifecycleAction::StartAudio, timestamp_now())
        .unwrap();
    session.controller = Some(controller);
    session.pending_backend_jobs = 1;

    assert!(
        session
            .apply_backend_result(BackendResult::Error("backend failed".to_string()))
            .await
            .is_err()
    );
    assert!(session.backend_failed);
    assert!(session.backend_cancelled.load(Ordering::Relaxed));
    session.pending_backend_jobs = 1;
    session.carry = vec![0];

    let (result_sender, result_receiver) = mpsc::channel(1);
    session.backend_results = Some(result_receiver);
    result_sender
        .send(BackendResult::Final(BackendSuccess {
            utterance_id: TranscriptUtteranceId("utt_2".to_string()),
            start_ms: 0,
            end_ms: 20,
            segment_id: TranscriptSegmentId("seg_2".to_string()),
            text: "must not be emitted".to_string(),
            language: None,
            words: Vec::new(),
        }))
        .await
        .unwrap();
    drop(result_sender);

    assert!(session.finish("transport_closed", true).await.is_err());
    let mut event_types = Vec::new();
    while let Ok(event) = event_receiver.try_recv() {
        event_types.push(event.event_type);
    }
    assert_eq!(
        event_types,
        vec!["error", "audio.input.stopped", "session.closed"]
    );
}

#[tokio::test]
async fn finish_transport_closed_cancels_pending_backend_jobs_without_waiting() {
    let (event_sender, _event_receiver) = mpsc::channel(8);
    let mut session = WsSession::new(ServerRuntime::default(), test_distribution(), event_sender);
    let mut controller = RealtimeSessionController::new(RealtimeSessionConfig::new(
        "test_session",
        "whisper-large-v3-turbo",
        timestamp_now(),
    ))
    .unwrap();
    controller
        .lifecycle(RealtimeLifecycleAction::Configure, timestamp_now())
        .unwrap();
    controller
        .lifecycle(RealtimeLifecycleAction::StartAudio, timestamp_now())
        .unwrap();
    session.controller = Some(controller);
    session.spawn_backend_worker();
    session.pending_backend_jobs = 1;

    tokio::time::timeout(
        Duration::from_millis(100),
        session.finish("transport_closed", true),
    )
    .await
    .expect("transport close should not wait for backend results")
    .unwrap();
    assert_eq!(session.pending_backend_jobs, 0);
    assert!(session.backend_cancelled.load(Ordering::Relaxed));
    assert!(session.backend_jobs.is_none());
}

#[tokio::test]
async fn session_start_rejects_realtime_hotwords_instead_of_ignoring_them() {
    let (event_sender, mut event_receiver) = mpsc::channel(8);
    let mut session = WsSession::new(ServerRuntime::default(), test_distribution(), event_sender);

    assert!(
        session
            .start_session(StartSession {
                model: Some("whisper-large-v3-turbo".to_string()),
                hotwords: Some(vec!["OpenASR".to_string()]),
                ..StartSession::default()
            })
            .await
            .is_err()
    );

    let event = event_receiver.recv().await.unwrap();
    assert_eq!(event.event_type, "error");
    assert!(matches!(
        event.event,
        RealtimeEvent::Error(RealtimeErrorEvent {
            code: RealtimeErrorCode::StartupConfigError,
            ..
        })
    ));
}

#[tokio::test]
async fn session_start_rejects_xasr_hotwords_from_active_native_capabilities() {
    let temp = tempfile::tempdir().unwrap();
    let model_id = "xasr-zipformer-test";
    let pack_path = temp.path().join("xasr-zipformer-test.oasr");
    write_xasr_streaming_fixture_pack(&pack_path, model_id);
    let runtime = ServerRuntime {
        backend: openasr_core::BackendKind::Native,
        ffmpeg_bin: None,
        ffmpeg_bin_explicit: false,
        model_pack_path: Some(pack_path),
    };
    let (event_sender, mut event_receiver) = mpsc::channel(8);
    let mut session = WsSession::new(runtime, test_distribution(), event_sender);

    assert!(
        session
            .start_session(StartSession {
                model: Some(model_id.to_string()),
                hotwords: Some(vec!["OpenASR".to_string()]),
                partial_results: Some(true),
                ..StartSession::default()
            })
            .await
            .is_err()
    );

    let event = event_receiver.recv().await.unwrap();
    assert_eq!(event.event_type, "error");
    match event.event {
        RealtimeEvent::Error(RealtimeErrorEvent {
            code: RealtimeErrorCode::StartupConfigError,
            message,
            recoverable: false,
        }) => {
            assert!(message.contains("xasr-zipformer"), "{message}");
            assert!(message.contains("silently ignoring hotwords"), "{message}");
        }
        other => panic!("expected startup_config_error event, got {other:?}"),
    }
}

#[tokio::test]
async fn session_start_accepts_hotwords_for_supporting_native_model() {
    let temp = tempfile::tempdir().unwrap();
    let model_id = "moonshine-hotword-test";
    let pack_path = temp.path().join("moonshine-hotword-test.oasr");
    write_moonshine_streaming_fixture_pack(&pack_path, model_id);
    let runtime = ServerRuntime {
        backend: openasr_core::BackendKind::Native,
        ffmpeg_bin: None,
        ffmpeg_bin_explicit: false,
        model_pack_path: Some(pack_path),
    };
    let (event_sender, _event_receiver) = mpsc::channel(8);
    let mut session = WsSession::new(runtime, test_distribution(), event_sender);

    session
        .start_session(StartSession {
            model: Some(model_id.to_string()),
            phrase_bias: Some(ClientPhraseBias {
                phrases: vec!["OpenASR".to_string()],
                boost: Some(3.0),
            }),
            partial_results: Some(true),
            ..StartSession::default()
        })
        .await
        .expect("moonshine phrase bias should pass session.start capability gate");

    let phrase_bias = session.phrase_bias.as_ref().expect("phrase bias retained");
    assert_eq!(phrase_bias.entries()[0].phrase(), "OpenASR");
    assert_eq!(phrase_bias.entries()[0].boost(), 3.0);
    assert!(session.native_streaming.is_some());
    let _ = session.finish("test_complete", true).await;
}

#[tokio::test]
async fn native_streaming_configured_event_preserves_diarize_request() {
    let _env_lock = speaker_embedder_env_lock().await;
    let temp = tempfile::tempdir().unwrap();
    let model_id = "qwen3-asr-0.6b";
    let pack_path = temp.path().join("qwen3-asr-0.6b.oasr");
    write_qwen_streaming_fixture_pack(&pack_path, model_id);
    let wespeaker = temp.path().join("wespeaker.oasr");
    std::fs::write(&wespeaker, b"GGUF\x00\x00\x00\x00").unwrap();
    let _wespeaker = EnvVarGuard::set("OPENASR_WESPEAKER_PACK", &wespeaker);
    let runtime = ServerRuntime {
        backend: openasr_core::BackendKind::Native,
        ffmpeg_bin: None,
        ffmpeg_bin_explicit: false,
        model_pack_path: Some(pack_path),
    };
    let (event_sender, mut event_receiver) = mpsc::channel(8);
    let mut session = WsSession::new(runtime, test_distribution(), event_sender);
    static EMBEDDER: FixedSpeakerEmbedder = FixedSpeakerEmbedder;
    session.test_streaming_diarizer_embedder = Some(&EMBEDDER);

    let result = session
        .start_session(StartSession {
            model: Some(model_id.to_string()),
            partial_results: Some(true),
            word_timestamps: Some(true),
            diarize: Some(true),
            ..StartSession::default()
        })
        .await;
    result.expect("session.start should preserve accepted diarize option");
    assert!(session.streaming_diarizer.is_some());
    assert!(session.native_speaker_change_detector.is_some());

    let mut configured_envelope = None;
    while let Ok(envelope) = event_receiver.try_recv() {
        if let RealtimeEvent::Lifecycle(RealtimeLifecycleEvent::SessionConfigured(_)) =
            &envelope.event
        {
            configured_envelope = Some(envelope);
            break;
        }
    }
    let configured_envelope = configured_envelope.expect("session.configured event");
    assert_eq!(configured_envelope.event_type, "session.configured");
    let RealtimeEvent::Lifecycle(RealtimeLifecycleEvent::SessionConfigured(configured)) =
        configured_envelope.event
    else {
        panic!("expected session.configured lifecycle event");
    };
    assert!(configured.partial_results);
    assert!(configured.word_timestamps);
    assert!(
        configured.diarize,
        "native true-streaming session.configured must reflect accepted diarize=true"
    );

    let _ = session.finish("test_complete", true).await;
}

#[tokio::test]
async fn session_start_rejects_diarize_without_embedder_pack() {
    let _env_lock = speaker_embedder_env_lock().await;
    let temp = tempfile::tempdir().unwrap();
    // Hermetic: realtime diarization availability probes the installed
    // WeSpeaker pack, so pin the lookup to an empty home.
    let _wespeaker = EnvVarGuard::unset("OPENASR_WESPEAKER_PACK");
    let _home = EnvVarGuard::set("OPENASR_HOME", temp.path());
    let (event_sender, mut event_receiver) = mpsc::channel(8);
    let mut session = WsSession::new(ServerRuntime::default(), test_distribution(), event_sender);

    assert!(
        session
            .start_session(StartSession {
                model: Some("whisper-large-v3-turbo".to_string()),
                diarize: Some(true),
                ..StartSession::default()
            })
            .await
            .is_err()
    );

    let event = event_receiver.recv().await.unwrap();
    assert_eq!(event.event_type, "error");
    match &event.event {
        RealtimeEvent::Error(RealtimeErrorEvent { code, message, .. }) => {
            assert_eq!(*code, RealtimeErrorCode::StartupConfigError);
            assert!(message.contains("speaker-embedder pack"));
        }
        other => panic!("expected startup config error, got {other:?}"),
    }
}

fn translation_options_enabled() -> ClientTranslationOptions {
    ClientTranslationOptions {
        enabled: Some(true),
        target_lang: Some("en".to_string()),
        model: Some("hymt2-1.8b".to_string()),
        mode: Some("clause_retranslation".to_string()),
        provisional: Some(true),
    }
}

fn fake_translation_worker(
    f: impl Fn(TranslationRequest) -> Result<TranslationWorkerOutput, TranslationQueueError>
    + Send
    + Sync
    + 'static,
) -> TranslationWorkerHook {
    Arc::new(f)
}

fn translation_transcript_envelope(event: RealtimeTranscriptEvent) -> RealtimeEventEnvelope {
    TestServerNativeSession::new("translation_source")
        .transcript(event)
        .remove(0)
}

fn transcript_partial_text(text: &str, revision: u64, end_ms: u64) -> RealtimeTranscriptEvent {
    RealtimeTranscriptEvent::Partial(openasr_core::RealtimeTranscriptPartial {
        utterance_id: TranscriptUtteranceId("utt_translation_000001".to_string()),
        segment_id: TranscriptSegmentId("seg_translation_000001".to_string()),
        revision,
        text: text.to_string(),
        start_ms: 0,
        end_ms,
        is_final: false,
        words: Vec::new(),
        language: Some("zh".to_string()),
        speaker: None,
        speaker_label: None,
        speaker_profile_id: None,
    })
}

fn transcript_final_text(text: &str, revision: u64, end_ms: u64) -> RealtimeTranscriptEvent {
    RealtimeTranscriptEvent::Final(openasr_core::RealtimeTranscriptFinal {
        utterance_id: TranscriptUtteranceId("utt_translation_000001".to_string()),
        segment_id: TranscriptSegmentId("seg_translation_000001".to_string()),
        revision,
        text: text.to_string(),
        start_ms: 0,
        end_ms,
        is_final: true,
        words: Vec::new(),
        language: Some("zh".to_string()),
        speaker: None,
        speaker_label: None,
        speaker_profile_id: None,
    })
}

fn transcript_revision_text(text: &str, revision: u64, end_ms: u64) -> RealtimeTranscriptEvent {
    RealtimeTranscriptEvent::Revision(openasr_core::RealtimeTranscriptRevision {
        utterance_id: TranscriptUtteranceId("utt_translation_000001".to_string()),
        segment_id: TranscriptSegmentId("seg_translation_000001".to_string()),
        revises_event_id: None,
        revision,
        text: text.to_string(),
        start_ms: 0,
        end_ms,
        is_final: true,
        reason: openasr_core::TRANSCRIPT_REVISION_REASON_POST_FINAL_CORRECTION.to_string(),
        words: Vec::new(),
        language: Some("zh".to_string()),
        speaker: None,
        speaker_label: None,
        speaker_profile_id: None,
    })
}

async fn collect_events(
    receiver: &mut mpsc::Receiver<RealtimeEventEnvelope>,
) -> Vec<RealtimeEventEnvelope> {
    let mut events = Vec::new();
    while let Ok(event) = receiver.try_recv() {
        events.push(event);
    }
    events
}

/// Regression for the live zh-en stall (recording 1781267241): clause
/// boundaries landing next to ASCII-run spaces made the segmenter misread
/// every later partial as a revision, so translation output stopped after the
/// first clauses while the worker kept burning CPU on doomed requests.
#[tokio::test]
async fn mixed_script_clause_boundaries_do_not_stall_translation() {
    let (event_sender, mut event_receiver) = mpsc::channel(64);
    let mut session = WsSession::new(ServerRuntime::default(), test_distribution(), event_sender);
    session.test_translation_worker = Some(fake_translation_worker(|request| {
        Ok(TranslationWorkerOutput {
            text: format!("en:{}", request.source_text),
            timings: openasr_core::TranslationTimings::default(),
        })
    }));
    session
        .start_session(StartSession {
            model: Some("whisper-large-v3-turbo".to_string()),
            language: Some("zh".to_string()),
            translation: Some(translation_options_enabled()),
            ..StartSession::default()
        })
        .await
        .expect("start with translation");

    // The comma boundary leaves a remainder that starts with the space before
    // "codex" — every later partial must still be read as a pure append.
    let mut text = String::from("我们现在来看一下这个东西好吗， codex 是一个工具确实不错。");
    let mut revision = 1;
    for tail in [
        "我们在 collect 里面怎么做一个项目的管理。",
        "呃这期视频呢无论是对于这种开发者，",
    ] {
        text.push_str(tail);
        revision += 1;
        session
            .emit_envelope_with_translation(translation_transcript_envelope(
                transcript_partial_text(&text, revision, revision * 480),
            ))
            .await
            .expect("emit growing partial");
    }
    session
        .drain_translation_until_idle()
        .await
        .expect("drain translation");

    let events = collect_events(&mut event_receiver).await;
    let finals = events
        .iter()
        .filter_map(|event| match &event.event {
            RealtimeEvent::Translation(RealtimeTranslationEvent::Final(event)) => {
                Some(event.text.as_str())
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    assert!(
        finals
            .iter()
            .any(|text| text.contains("collect 里面怎么做一个项目的管理。")),
        "clauses after the mixed-script boundary must keep translating; got {finals:?}"
    );
    assert!(
        finals
            .iter()
            .any(|text| text.contains("呃这期视频呢无论是对于这种开发者，")),
        "late clauses must keep translating; got {finals:?}"
    );
    assert!(
        !events.iter().any(|event| matches!(
            &event.event,
            RealtimeEvent::Translation(RealtimeTranslationEvent::Tombstone(_))
        )),
        "pure appends must not retire clauses"
    );
}

#[tokio::test]
async fn punctuation_only_clause_is_not_sent_to_the_translator() {
    let (event_sender, mut event_receiver) = mpsc::channel(64);
    let mut session = WsSession::new(ServerRuntime::default(), test_distribution(), event_sender);
    session.test_translation_worker = Some(fake_translation_worker(|request| {
        Ok(TranslationWorkerOutput {
            text: format!("en:{}", request.source_text),
            timings: openasr_core::TranslationTimings::default(),
        })
    }));
    session
        .start_session(StartSession {
            model: Some("whisper-large-v3-turbo".to_string()),
            language: Some("zh".to_string()),
            translation: Some(translation_options_enabled()),
            ..StartSession::default()
        })
        .await
        .expect("start with translation");

    // The ASR final consumes the words; the next partial leaves a lone "。"
    // as its own finalized clause, which must not reach the worker.
    session
        .emit_envelope_with_translation(translation_transcript_envelope(transcript_final_text(
            "这是没问题的",
            1,
            500,
        )))
        .await
        .expect("emit final");
    session
        .emit_envelope_with_translation(translation_transcript_envelope(transcript_partial_text(
            "这是没问题的。后面我们继续讲，",
            2,
            900,
        )))
        .await
        .expect("emit partial with leftover punctuation");
    session
        .drain_translation_until_idle()
        .await
        .expect("drain translation");

    let events = collect_events(&mut event_receiver).await;
    let finals = events
        .iter()
        .filter_map(|event| match &event.event {
            RealtimeEvent::Translation(RealtimeTranslationEvent::Final(event)) => {
                Some(event.text.as_str())
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    assert!(
        !finals.contains(&"en:。"),
        "punctuation-only clause must be skipped; got {finals:?}"
    );
    assert!(
        finals.contains(&"en:后面我们继续讲，"),
        "the real clause after the punctuation must still translate; got {finals:?}"
    );
}

#[tokio::test]
async fn session_start_rejects_translation_without_hymt2_pack() {
    let (event_sender, mut event_receiver) = mpsc::channel(8);
    let mut session = WsSession::new(ServerRuntime::default(), test_distribution(), event_sender);

    let result = session
        .start_session(StartSession {
            model: Some("whisper-large-v3-turbo".to_string()),
            language: Some("zh".to_string()),
            translation: Some(translation_options_enabled()),
            ..StartSession::default()
        })
        .await;

    assert!(result.is_err());
    let events = collect_events(&mut event_receiver).await;
    assert_eq!(
        first_error_code(&events),
        Some(RealtimeErrorCode::StartupConfigError)
    );
    let message = events
        .iter()
        .find_map(|event| match &event.event {
            RealtimeEvent::Error(error) => Some(error.message.as_str()),
            _ => None,
        })
        .expect("translation startup error");
    assert!(message.contains("translation_pack_missing"));
}

#[tokio::test]
async fn translation_async_init_does_not_block_session_start_and_announces_ready() {
    let (event_sender, mut event_receiver) = mpsc::channel(32);
    let mut session = WsSession::new(ServerRuntime::default(), test_distribution(), event_sender);
    let (release_init_sender, release_init_receiver) = std::sync::mpsc::channel::<()>();
    let release_init_receiver = Arc::new(Mutex::new(release_init_receiver));
    session.test_translation_worker = Some(fake_translation_worker(|request| {
        Ok(TranslationWorkerOutput {
            text: format!("en:{}", request.source_text),
            timings: openasr_core::TranslationTimings::default(),
        })
    }));
    session.test_translation_worker_init = Some(Arc::new(move || {
        release_init_receiver
            .lock()
            .expect("init release mutex")
            .recv()
            .map_err(|_| TranslationQueueError::Worker {
                reason: "init release channel closed".to_string(),
            })
    }));

    // session.start must be accepted while the translation model is still
    // loading: this is the cold-load-off-critical-path contract.
    session
        .start_session(StartSession {
            model: Some("whisper-large-v3-turbo".to_string()),
            language: Some("zh".to_string()),
            translation: Some(translation_options_enabled()),
            ..StartSession::default()
        })
        .await
        .expect("start while translation worker is still initializing");
    let startup_events = collect_events(&mut event_receiver).await;
    assert!(
        startup_events
            .iter()
            .any(|event| event.event_type == "session.created")
    );
    assert!(
        !startup_events
            .iter()
            .any(|event| event.event_type == "translation.status"),
        "ready must not be announced before the worker finished loading"
    );

    // Source transcripts arriving during the load are buffered, not dropped.
    session
        .emit_envelope_with_translation(translation_transcript_envelope(transcript_final_text(
            "我们需要保持流式路径很快。",
            1,
            500,
        )))
        .await
        .expect("emit transcript final during load");

    release_init_sender.send(()).expect("release init");
    session
        .drain_translation_until_idle()
        .await
        .expect("drain translation after init");

    let events = collect_events(&mut event_receiver).await;
    let status_events = events
        .iter()
        .filter_map(|event| match &event.event {
            RealtimeEvent::Translation(RealtimeTranslationEvent::Status(status)) => Some(status),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(status_events.len(), 1, "exactly one ready announcement");
    assert_eq!(status_events[0].state, "ready");
    assert_eq!(status_events[0].model, HYMT2_TRANSLATION_MODEL_ID);
    assert_eq!(status_events[0].target_lang, "en");
    let translation_final = events
        .iter()
        .find_map(|event| match &event.event {
            RealtimeEvent::Translation(RealtimeTranslationEvent::Final(event)) => Some(event),
            _ => None,
        })
        .expect("buffered source translated after init");
    assert_eq!(translation_final.text, "en:我们需要保持流式路径很快。");

    // The announcement is one-shot.
    session
        .drain_translation_outputs()
        .await
        .expect("idle drain");
    let extra_events = collect_events(&mut event_receiver).await;
    assert!(
        !extra_events
            .iter()
            .any(|event| event.event_type == "translation.status")
    );
}

#[tokio::test]
async fn translation_async_init_failure_fails_session_via_error_event() {
    let (event_sender, mut event_receiver) = mpsc::channel(32);
    let mut session = WsSession::new(ServerRuntime::default(), test_distribution(), event_sender);
    session.test_translation_worker = Some(fake_translation_worker(|request| {
        Ok(TranslationWorkerOutput {
            text: request.source_text,
            timings: openasr_core::TranslationTimings::default(),
        })
    }));
    session.test_translation_worker_init = Some(Arc::new(|| {
        Err(TranslationQueueError::Worker {
            reason: "Realtime translation Hy-MT2 runtime could not be loaded: 模拟加载失败"
                .to_string(),
        })
    }));

    session
        .start_session(StartSession {
            model: Some("whisper-large-v3-turbo".to_string()),
            language: Some("zh".to_string()),
            translation: Some(translation_options_enabled()),
            ..StartSession::default()
        })
        .await
        .expect("start is accepted before the load failure is known");
    let _ = collect_events(&mut event_receiver).await;

    // The failure must surface as a session-fatal error on the next drain.
    let deadline = std::time::Instant::now() + Duration::from_secs(2);
    loop {
        if session.drain_translation_outputs().await.is_err() {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "translation init failure never surfaced"
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }

    let events = collect_events(&mut event_receiver).await;
    assert_eq!(
        first_error_code(&events),
        Some(RealtimeErrorCode::BackendCrashed)
    );
    let message = events
        .iter()
        .find_map(|event| match &event.event {
            RealtimeEvent::Error(error) => Some(error.message.as_str()),
            _ => None,
        })
        .expect("translation load failure error");
    assert!(message.contains("Hy-MT2 runtime could not be loaded"));
    assert!(
        !events
            .iter()
            .any(|event| event.event_type == "translation.status"),
        "a failed load must never claim readiness"
    );
}

#[tokio::test]
async fn session_configured_reports_translation_truthfully() {
    let (event_sender, mut event_receiver) = mpsc::channel(8);
    let mut session = WsSession::new(ServerRuntime::default(), test_distribution(), event_sender);
    session.test_translation_worker = Some(fake_translation_worker(|request| {
        Ok(TranslationWorkerOutput {
            text: request.source_text,
            timings: openasr_core::TranslationTimings::default(),
        })
    }));

    session
        .start_session(StartSession {
            model: Some("whisper-large-v3-turbo".to_string()),
            language: Some("zh".to_string()),
            translation: Some(translation_options_enabled()),
            ..StartSession::default()
        })
        .await
        .expect("start with fake translation");

    assert!(session.translation.is_some());
    let events = collect_events(&mut event_receiver).await;
    let configured = events
        .iter()
        .find_map(|event| match &event.event {
            RealtimeEvent::Lifecycle(RealtimeLifecycleEvent::SessionConfigured(configured)) => {
                Some(configured)
            }
            _ => None,
        })
        .expect("session.configured");
    assert!(configured.translation.enabled);
    assert_eq!(configured.translation.target_lang.as_deref(), Some("en"));
    assert_eq!(
        configured.translation.model.as_deref(),
        Some(HYMT2_TRANSLATION_MODEL_ID)
    );
}

#[tokio::test]
async fn translation_final_uses_session_sequencer_and_versions() {
    let (event_sender, mut event_receiver) = mpsc::channel(16);
    let mut session = WsSession::new(ServerRuntime::default(), test_distribution(), event_sender);
    session.test_translation_worker = Some(fake_translation_worker(|request| {
        Ok(TranslationWorkerOutput {
            text: format!("en:{}:{}", request.source_version, request.source_text),
            timings: openasr_core::TranslationTimings::default(),
        })
    }));
    session
        .start_session(StartSession {
            model: Some("whisper-large-v3-turbo".to_string()),
            language: Some("zh".to_string()),
            translation: Some(translation_options_enabled()),
            ..StartSession::default()
        })
        .await
        .expect("start with fake translation");
    let _ = collect_events(&mut event_receiver).await;

    session
        .emit_envelope_with_translation(translation_transcript_envelope(transcript_final_text(
            "我们需要保持流式路径很快。",
            1,
            500,
        )))
        .await
        .expect("emit transcript final");
    session
        .drain_translation_until_idle()
        .await
        .expect("drain translation");

    let events = collect_events(&mut event_receiver).await;
    assert_eq!(
        events
            .iter()
            .map(|event| event.event_type)
            .collect::<Vec<_>>(),
        vec!["transcript.final", "translation.final"]
    );
    assert_eq!(events[0].seq + 1, events[1].seq);
    match &events[1].event {
        RealtimeEvent::Translation(RealtimeTranslationEvent::Final(event)) => {
            assert_eq!(event.clause_id, "c-1");
            assert_eq!(event.source_segment_id, "seg_translation_000001");
            assert_eq!(event.source_version, 1);
            assert_eq!(event.translation_version, 1);
            assert_eq!(event.target_lang, "en");
            assert!(event.is_final);
            assert_eq!(event.model, HYMT2_TRANSLATION_MODEL_ID);
        }
        other => panic!("expected translation.final, got {other:?}"),
    }
}

#[tokio::test]
async fn translation_latest_only_drops_stale_provisional_outputs() {
    let (event_sender, mut event_receiver) = mpsc::channel(32);
    let mut session = WsSession::new(ServerRuntime::default(), test_distribution(), event_sender);
    let (started_sender, started_receiver) = std::sync::mpsc::channel();
    let (release_sender, release_receiver) = std::sync::mpsc::channel();
    let release_receiver = Arc::new(Mutex::new(release_receiver));
    session.test_translation_worker = Some(fake_translation_worker(move |request| {
        started_sender
            .send(request.source_version)
            .expect("started send");
        if request.source_version == 1 {
            release_receiver
                .lock()
                .expect("release mutex")
                .recv()
                .expect("release first translation");
        }
        Ok(TranslationWorkerOutput {
            text: format!("en-v{}", request.source_version),
            timings: openasr_core::TranslationTimings::default(),
        })
    }));
    session
        .start_session(StartSession {
            model: Some("whisper-large-v3-turbo".to_string()),
            language: Some("zh".to_string()),
            translation: Some(translation_options_enabled()),
            ..StartSession::default()
        })
        .await
        .expect("start with fake translation");
    let _ = collect_events(&mut event_receiver).await;

    for (revision, text, end_ms) in [(1, "我们需要", 0), (2, "我们需要", 200)] {
        session
            .emit_envelope_with_translation(translation_transcript_envelope(
                transcript_partial_text(text, revision, end_ms),
            ))
            .await
            .expect("emit partial");
    }
    // Wait until the worker has TAKEN v1 (and is blocked inside it) before
    // growing the source: otherwise the v2 provisional can replace v1 in the
    // pending slot and the worker would never see v1 at all.
    assert_eq!(
        started_receiver
            .recv_timeout(Duration::from_secs(1))
            .expect("first translation started"),
        1
    );
    for (revision, text, end_ms) in [
        (3, "我们需要保持流式路径", 300),
        (4, "我们需要保持流式路径", 500),
    ] {
        session
            .emit_envelope_with_translation(translation_transcript_envelope(
                transcript_partial_text(text, revision, end_ms),
            ))
            .await
            .expect("emit partial");
    }
    release_sender.send(()).expect("release stale worker");
    session
        .drain_translation_until_idle()
        .await
        .expect("drain translations");

    let events = collect_events(&mut event_receiver).await;
    let translations = events
        .iter()
        .filter_map(|event| match &event.event {
            RealtimeEvent::Translation(RealtimeTranslationEvent::Partial(partial)) => Some(partial),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(translations.len(), 1);
    assert_eq!(translations[0].source_version, 2);
    assert_eq!(translations[0].translation_version, 2);
    assert_eq!(translations[0].text, "en-v2");
}

#[tokio::test]
async fn translation_reemits_revised_source_with_replacement_metadata() {
    let (event_sender, mut event_receiver) = mpsc::channel(32);
    let mut session = WsSession::new(ServerRuntime::default(), test_distribution(), event_sender);
    session.test_translation_worker = Some(fake_translation_worker(|request| {
        Ok(TranslationWorkerOutput {
            text: format!("en:{}", request.source_text),
            timings: openasr_core::TranslationTimings::default(),
        })
    }));
    session
        .start_session(StartSession {
            model: Some("whisper-large-v3-turbo".to_string()),
            language: Some("zh".to_string()),
            translation: Some(translation_options_enabled()),
            ..StartSession::default()
        })
        .await
        .expect("start with fake translation");
    let _ = collect_events(&mut event_receiver).await;

    session
        .emit_envelope_with_translation(translation_transcript_envelope(transcript_final_text(
            "我们需要保持流式路径很快。",
            1,
            500,
        )))
        .await
        .expect("emit final");
    session.drain_translation_until_idle().await.unwrap();
    session
        .emit_envelope_with_translation(translation_transcript_envelope(transcript_revision_text(
            "我们必须保持流式路径很快。",
            2,
            700,
        )))
        .await
        .expect("emit revision");
    session.drain_translation_until_idle().await.unwrap();

    let events = collect_events(&mut event_receiver).await;
    let translations = events
        .iter()
        .filter_map(|event| match &event.event {
            RealtimeEvent::Translation(RealtimeTranslationEvent::Final(final_event)) => {
                Some(final_event)
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        translations
            .iter()
            .map(|event| event.clause_id.as_str())
            .collect::<Vec<_>>(),
        vec!["c-1", "c-2"]
    );
    assert_eq!(translations[0].replaces_clause_id, None);
    assert_eq!(translations[0].revises_clause_id, None);
    assert_eq!(translations[1].replaces_clause_id.as_deref(), Some("c-1"));
    assert_eq!(translations[1].revises_clause_id.as_deref(), Some("c-1"));
}

#[tokio::test]
async fn translation_revision_drops_in_flight_retired_final() {
    let (event_sender, mut event_receiver) = mpsc::channel(32);
    let mut session = WsSession::new(ServerRuntime::default(), test_distribution(), event_sender);
    let (started_sender, started_receiver) = std::sync::mpsc::channel();
    let (release_sender, release_receiver) = std::sync::mpsc::channel();
    let release_receiver = Arc::new(Mutex::new(release_receiver));
    session.test_translation_worker = Some(fake_translation_worker(move |request| {
        started_sender
            .send(request.clause_id)
            .expect("started send");
        if request.clause_id == ClauseId::new(1) {
            release_receiver
                .lock()
                .expect("release mutex")
                .recv()
                .expect("release retired final");
        }
        Ok(TranslationWorkerOutput {
            text: format!("en:{}", request.source_text),
            timings: openasr_core::TranslationTimings::default(),
        })
    }));
    session
        .start_session(StartSession {
            model: Some("whisper-large-v3-turbo".to_string()),
            language: Some("zh".to_string()),
            translation: Some(translation_options_enabled()),
            ..StartSession::default()
        })
        .await
        .expect("start with fake translation");
    let _ = collect_events(&mut event_receiver).await;

    session
        .emit_envelope_with_translation(translation_transcript_envelope(transcript_final_text(
            "我们需要保持流式路径很快。",
            1,
            500,
        )))
        .await
        .expect("emit final");
    assert_eq!(
        started_receiver
            .recv_timeout(Duration::from_secs(1))
            .expect("first final started"),
        ClauseId::new(1)
    );
    session
        .emit_envelope_with_translation(translation_transcript_envelope(transcript_revision_text(
            "我们必须保持流式路径很快。",
            2,
            700,
        )))
        .await
        .expect("emit revision");
    release_sender.send(()).expect("release retired final");
    session.drain_translation_until_idle().await.unwrap();

    let events = collect_events(&mut event_receiver).await;
    let translations = events
        .iter()
        .filter_map(|event| match &event.event {
            RealtimeEvent::Translation(RealtimeTranslationEvent::Final(final_event)) => {
                Some(final_event)
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(translations.len(), 1);
    assert_eq!(translations[0].clause_id, "c-2");
    assert_eq!(translations[0].replaces_clause_id.as_deref(), Some("c-1"));
    assert_eq!(translations[0].revises_clause_id.as_deref(), Some("c-1"));
    assert!(
        !translations.iter().any(|event| event.clause_id == "c-1"),
        "retired c-1 final must not emit after the revision"
    );
}

#[tokio::test]
async fn translation_revision_tombstones_clause_removed_without_replacement() {
    let (event_sender, mut event_receiver) = mpsc::channel(32);
    let mut session = WsSession::new(ServerRuntime::default(), test_distribution(), event_sender);
    session.test_translation_worker = Some(fake_translation_worker(|request| {
        Ok(TranslationWorkerOutput {
            text: format!("en:{}", request.source_text),
            timings: openasr_core::TranslationTimings::default(),
        })
    }));
    session
        .start_session(StartSession {
            model: Some("whisper-large-v3-turbo".to_string()),
            language: Some("zh".to_string()),
            translation: Some(translation_options_enabled()),
            ..StartSession::default()
        })
        .await
        .expect("start with fake translation");
    let _ = collect_events(&mut event_receiver).await;

    session
        .emit_envelope_with_translation(translation_transcript_envelope(transcript_final_text(
            "我们需要保持流式路径很快。",
            1,
            500,
        )))
        .await
        .expect("emit final");
    session.drain_translation_until_idle().await.unwrap();
    let _ = collect_events(&mut event_receiver).await;

    session
        .emit_envelope_with_translation(translation_transcript_envelope(transcript_revision_text(
            "", 2, 700,
        )))
        .await
        .expect("emit revision deleting text");

    let events = collect_events(&mut event_receiver).await;
    let tombstone = events
        .iter()
        .find_map(|event| match &event.event {
            RealtimeEvent::Translation(RealtimeTranslationEvent::Tombstone(tombstone)) => {
                Some(tombstone)
            }
            _ => None,
        })
        .expect("translation tombstone");
    assert_eq!(tombstone.clause_id, "c-1");
    assert_eq!(tombstone.source_segment_id, "seg_translation_000001");
    assert_eq!(tombstone.reason, "source_clause_retired");
    assert_eq!(tombstone.target_lang, "en");
    assert_eq!(tombstone.model, HYMT2_TRANSLATION_MODEL_ID);
}

#[tokio::test]
async fn slow_translation_does_not_block_asr_event_emission() {
    let (event_sender, mut event_receiver) = mpsc::channel(32);
    let mut session = WsSession::new(ServerRuntime::default(), test_distribution(), event_sender);
    let (release_sender, release_receiver) = std::sync::mpsc::channel();
    let release_receiver = Arc::new(Mutex::new(release_receiver));
    session.test_translation_worker = Some(fake_translation_worker(move |request| {
        release_receiver
            .lock()
            .expect("release mutex")
            .recv()
            .expect("release slow translation");
        Ok(TranslationWorkerOutput {
            text: request.source_text,
            timings: openasr_core::TranslationTimings::default(),
        })
    }));
    session
        .start_session(StartSession {
            model: Some("whisper-large-v3-turbo".to_string()),
            language: Some("zh".to_string()),
            translation: Some(translation_options_enabled()),
            ..StartSession::default()
        })
        .await
        .expect("start with fake translation");
    let _ = collect_events(&mut event_receiver).await;

    session
        .emit_envelope_with_translation(translation_transcript_envelope(transcript_final_text(
            "我们需要保持流式路径很快。",
            1,
            500,
        )))
        .await
        .expect("emit final while translation blocks");
    let events = collect_events(&mut event_receiver).await;
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].event_type, "transcript.final");

    release_sender.send(()).expect("release slow translation");
    session.drain_translation_until_idle().await.unwrap();
    let events = collect_events(&mut event_receiver).await;
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].event_type, "translation.final");
    assert!(events[0].seq > 1);
}

#[tokio::test]
async fn cancel_with_blocked_translation_worker_emits_closed_without_waiting() {
    let (event_sender, mut event_receiver) = mpsc::channel(32);
    let mut session = WsSession::new(ServerRuntime::default(), test_distribution(), event_sender);
    let (started_sender, started_receiver) = std::sync::mpsc::channel();
    let (release_sender, release_receiver) = std::sync::mpsc::channel();
    let release_receiver = Arc::new(Mutex::new(release_receiver));
    session.test_translation_worker = Some(fake_translation_worker(move |request| {
        started_sender
            .send(request.clause_id)
            .expect("started send");
        release_receiver
            .lock()
            .expect("release mutex")
            .recv()
            .expect("release blocked translation");
        Ok(TranslationWorkerOutput {
            text: request.source_text,
            timings: openasr_core::TranslationTimings::default(),
        })
    }));
    session
        .start_session(StartSession {
            model: Some("whisper-large-v3-turbo".to_string()),
            language: Some("zh".to_string()),
            translation: Some(translation_options_enabled()),
            ..StartSession::default()
        })
        .await
        .expect("start with fake translation");
    let _ = collect_events(&mut event_receiver).await;

    session
        .emit_envelope_with_translation(translation_transcript_envelope(transcript_final_text(
            "我们需要保持流式路径很快。",
            1,
            500,
        )))
        .await
        .expect("emit final while translation blocks");
    started_receiver
        .recv_timeout(Duration::from_secs(1))
        .expect("translation started");

    let started_at = Instant::now();
    assert!(session.cancel("client_cancelled").await.is_err());
    assert!(
        started_at.elapsed() < Duration::from_millis(200),
        "cancel waited for the blocked translation worker"
    );
    let events = collect_events(&mut event_receiver).await;
    assert_eq!(
        first_error_code(&events),
        Some(RealtimeErrorCode::Cancelled)
    );
    assert!(
        events
            .iter()
            .any(|event| event.event_type == "session.closed")
    );

    release_sender.send(()).expect("release translation");
}

#[tokio::test]
async fn session_start_rejects_diarize_when_embedder_pack_fails_to_load() {
    let _env_lock = speaker_embedder_env_lock().await;
    let temp = tempfile::tempdir().unwrap();
    // A resolvable pack that is not loadable must reject the session instead
    // of silently degrading to anonymous transcripts.
    let wespeaker = temp.path().join("wespeaker.oasr");
    std::fs::write(&wespeaker, b"GGUF\x00\x00\x00\x00garbage").unwrap();
    let _wespeaker = EnvVarGuard::set("OPENASR_WESPEAKER_PACK", &wespeaker);
    let (event_sender, mut event_receiver) = mpsc::channel(8);
    let mut session = WsSession::new(ServerRuntime::default(), test_distribution(), event_sender);

    let result = session
        .start_session(StartSession {
            model: Some("whisper-large-v3-turbo".to_string()),
            diarize: Some(true),
            ..StartSession::default()
        })
        .await;
    assert!(result.is_err());
    assert!(session.streaming_diarizer.is_none());

    let event = event_receiver.recv().await.unwrap();
    assert_eq!(event.event_type, "error");
    match &event.event {
        RealtimeEvent::Error(RealtimeErrorEvent { code, message, .. }) => {
            assert_eq!(*code, RealtimeErrorCode::StartupConfigError);
            assert!(message.contains("could not be loaded"));
        }
        other => panic!("expected startup config error, got {other:?}"),
    }
}

#[tokio::test]
async fn session_start_without_diarize_keeps_sessions_anonymous() {
    let (event_sender, mut event_receiver) = mpsc::channel(8);
    let mut session = WsSession::new(ServerRuntime::default(), test_distribution(), event_sender);

    session
        .start_session(StartSession {
            model: Some("whisper-large-v3-turbo".to_string()),
            ..StartSession::default()
        })
        .await
        .unwrap();

    assert!(session.streaming_diarizer.is_none());
    let mut saw_configured = false;
    while let Ok(event) = event_receiver.try_recv() {
        if let RealtimeEvent::Lifecycle(RealtimeLifecycleEvent::SessionConfigured(configured)) =
            &event.event
        {
            assert!(!configured.diarize);
            saw_configured = true;
        }
    }
    assert!(saw_configured);
}

#[tokio::test]
async fn session_start_uses_request_inference_threads() {
    let (event_sender, _event_receiver) = mpsc::channel(8);
    let mut session = WsSession::new(ServerRuntime::default(), test_distribution(), event_sender);

    session
        .start_session(StartSession {
            model: Some("whisper-large-v3-turbo".to_string()),
            inference_threads: Some(6),
            ..StartSession::default()
        })
        .await
        .unwrap();

    assert_eq!(session.inference_threads, Some(6));
}

#[tokio::test]
async fn session_start_uses_request_execution_target() {
    let (event_sender, _event_receiver) = mpsc::channel(8);
    let mut session = WsSession::new(ServerRuntime::default(), test_distribution(), event_sender);

    session
        .start_session(StartSession {
            model: Some("whisper-large-v3-turbo".to_string()),
            execution_target: Some(openasr_core::ExecutionTarget::Cpu),
            ..StartSession::default()
        })
        .await
        .unwrap();

    assert_eq!(
        session.execution_target,
        Some(openasr_core::ExecutionTarget::Cpu)
    );
}

#[tokio::test]
async fn session_start_rejects_invalid_inference_threads() {
    let (event_sender, mut event_receiver) = mpsc::channel(8);
    let mut session = WsSession::new(ServerRuntime::default(), test_distribution(), event_sender);

    assert!(
        session
            .start_session(StartSession {
                model: Some("whisper-large-v3-turbo".to_string()),
                inference_threads: Some(0),
                ..StartSession::default()
            })
            .await
            .is_err()
    );

    let event = event_receiver.recv().await.unwrap();
    assert_eq!(event.event_type, "error");
    match event.event {
        RealtimeEvent::Error(RealtimeErrorEvent {
            code: RealtimeErrorCode::StartupConfigError,
            message,
            recoverable: false,
        }) => {
            assert!(message.contains("inference_threads must be between 1 and 256"));
        }
        other => panic!("expected startup_config_error event, got {other:?}"),
    }
}

#[test]
fn true_streaming_sessions_use_native_for_live_and_dictation() {
    let capabilities = RealtimeBackendCapabilities::true_streaming_local();

    assert!(should_use_native_streaming_session(
        Some(DICTATION_SOURCE_NAME),
        capabilities
    ));
    assert!(should_use_native_streaming_session(
        Some("Live"),
        capabilities
    ));
    assert!(should_use_native_streaming_session(None, capabilities));
}

#[test]
fn live_native_sessions_enable_effective_partials() {
    let capabilities = RealtimeBackendCapabilities::true_streaming_local();

    assert!(effective_session_partial_results(
        true,
        capabilities,
        should_use_native_streaming_session(Some(DICTATION_SOURCE_NAME), capabilities)
    ));
    assert!(effective_session_partial_results(
        true,
        capabilities,
        should_use_native_streaming_session(Some("Live"), capabilities)
    ));
}

#[tokio::test]
async fn dictation_finish_transcribes_low_energy_audio_without_vad_start() {
    let (event_sender, mut event_receiver) = mpsc::channel(64);
    let mut session = WsSession::new(ServerRuntime::default(), test_distribution(), event_sender);
    start_energy_fallback_test_session(&mut session, DICTATION_SOURCE_NAME)
        .await
        .unwrap();

    for seq in 1..=50 {
        session
            .process_frame(frame(seq, (seq - 1) * 20, 200))
            .await
            .unwrap();
    }
    session
        .apply_backend_result(BackendResult::Final(BackendSuccess {
            utterance_id: TranscriptUtteranceId("utt_dictation_000001".to_string()),
            start_ms: 0,
            end_ms: 1_000,
            segment_id: TranscriptSegmentId("seg_dictation_000001".to_string()),
            text: "dictation fallback final".to_string(),
            language: None,
            words: Vec::new(),
        }))
        .await
        .unwrap();
    session.finish("client_closed", true).await.unwrap();

    let mut event_types = Vec::new();
    while let Ok(event) = event_receiver.try_recv() {
        event_types.push(event.event_type);
    }
    assert!(!event_types.contains(&"vad.speech_started"));
    assert!(event_types.contains(&"transcript.final"));
    assert!(event_types.contains(&"session.closed"));
}

#[tokio::test]
async fn live_finish_does_not_force_transcribe_low_energy_audio_without_vad_start() {
    let (event_sender, mut event_receiver) = mpsc::channel(64);
    let mut session = WsSession::new(ServerRuntime::default(), test_distribution(), event_sender);
    start_energy_fallback_test_session(&mut session, "Live")
        .await
        .unwrap();

    for seq in 1..=50 {
        session
            .process_frame(frame(seq, (seq - 1) * 20, 200))
            .await
            .unwrap();
    }
    session.finish("client_closed", true).await.unwrap();

    let mut event_types = Vec::new();
    while let Ok(event) = event_receiver.try_recv() {
        event_types.push(event.event_type);
    }
    assert!(!event_types.contains(&"vad.speech_started"));
    assert!(!event_types.contains(&"transcript.final"));
    assert!(event_types.contains(&"session.closed"));
}

#[tokio::test]
async fn dictation_finish_does_not_force_transcribe_silence() {
    let (event_sender, mut event_receiver) = mpsc::channel(64);
    let mut session = WsSession::new(ServerRuntime::default(), test_distribution(), event_sender);
    start_energy_fallback_test_session(&mut session, DICTATION_SOURCE_NAME)
        .await
        .unwrap();

    for seq in 1..=50 {
        session
            .process_frame(frame(seq, (seq - 1) * 20, 0))
            .await
            .unwrap();
    }
    session.finish("client_closed", true).await.unwrap();

    let mut event_types = Vec::new();
    while let Ok(event) = event_receiver.try_recv() {
        event_types.push(event.event_type);
    }
    assert!(!event_types.contains(&"vad.speech_started"));
    assert!(!event_types.contains(&"transcript.final"));
    assert!(event_types.contains(&"session.closed"));
}

#[tokio::test]
async fn finish_records_completed_websocket_session_history() {
    let temp = tempfile::tempdir().unwrap();
    let distribution = DistributionContext::new(crate::DistributionRuntime {
        openasr_home: Some(temp.path().to_path_buf()),
        catalog_url: None,
    });
    // auto_save only controls transcript-file exports; history recording is
    // governed by history_retention alone, so auto_save=false must still record.
    std::fs::write(
        temp.path().join("config.json"),
        serde_json::json!({
            "preferences": { "auto_save": false, "history_retention": "last5" }
        })
        .to_string(),
    )
    .unwrap();
    let (event_sender, _event_receiver) = mpsc::channel(8);
    let mut session = WsSession::new(ServerRuntime::default(), distribution, event_sender);
    let mut controller = RealtimeSessionController::new(RealtimeSessionConfig::new(
        "test_session",
        "whisper-large-v3-turbo",
        timestamp_now(),
    ))
    .unwrap();
    controller
        .lifecycle(RealtimeLifecycleAction::Configure, timestamp_now())
        .unwrap();
    controller
        .lifecycle(RealtimeLifecycleAction::StartAudio, timestamp_now())
        .unwrap();
    session.controller = Some(controller);
    session.source_name = Some("Dictation".to_string());
    session.history_text = vec!["hello".to_string(), "world".to_string()];
    session.history_duration_ms = 1_240;

    session.finish("client_closed", true).await.unwrap();

    let store = DaemonHistoryStore::open(temp.path());
    let entries = store.list().unwrap();
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].kind, DaemonHistoryKind::Live);
    assert_eq!(entries[0].source_name.as_deref(), Some("Dictation"));
    assert!((entries[0].duration_seconds.unwrap() - 1.24).abs() < f32::EPSILON);
    assert_eq!(entries[0].output_format, Some(ResponseFormat::Text));
    assert_eq!(entries[0].diarization_active, Some(false));
    assert_eq!(
        entries[0].provenance,
        Some(DaemonHistoryProvenance::Recorded)
    );
    let detail = store.get(&entries[0].id).unwrap().unwrap();
    assert_eq!(detail.text, "hello\nworld");
    assert_eq!(detail.entry.output_format, Some(ResponseFormat::Text));
    assert_eq!(detail.entry.diarization_active, Some(false));
    assert_eq!(
        detail.entry.provenance,
        Some(DaemonHistoryProvenance::Recorded)
    );
}

#[tokio::test]
async fn finish_skips_websocket_session_history_when_retention_off() {
    let temp = tempfile::tempdir().unwrap();
    let distribution = DistributionContext::new(crate::DistributionRuntime {
        openasr_home: Some(temp.path().to_path_buf()),
        catalog_url: None,
    });
    // Even with auto_save enabled, "off" retention must skip the write:
    // history_retention is the only history switch.
    std::fs::write(
        temp.path().join("config.json"),
        serde_json::json!({
            "preferences": { "auto_save": true, "history_retention": "off" }
        })
        .to_string(),
    )
    .unwrap();
    let (event_sender, _event_receiver) = mpsc::channel(8);
    let mut session = WsSession::new(ServerRuntime::default(), distribution, event_sender);
    let mut controller = RealtimeSessionController::new(RealtimeSessionConfig::new(
        "test_session",
        "whisper-large-v3-turbo",
        timestamp_now(),
    ))
    .unwrap();
    controller
        .lifecycle(RealtimeLifecycleAction::Configure, timestamp_now())
        .unwrap();
    controller
        .lifecycle(RealtimeLifecycleAction::StartAudio, timestamp_now())
        .unwrap();
    session.controller = Some(controller);
    session.source_name = Some("Dictation".to_string());
    session.history_text = vec!["hello".to_string()];
    session.history_duration_ms = 500;

    session.finish("client_closed", true).await.unwrap();

    let store = DaemonHistoryStore::open(temp.path());
    assert!(store.list().unwrap().is_empty());
}

#[tokio::test]
async fn remote_compute_websocket_session_does_not_record_server_history() {
    let temp = tempfile::tempdir().unwrap();
    let distribution = DistributionContext::new(crate::DistributionRuntime {
        openasr_home: Some(temp.path().to_path_buf()),
        catalog_url: None,
    });
    let (event_sender, _event_receiver) = mpsc::channel(8);
    let mut session =
        WsSession::new_with_history(ServerRuntime::default(), distribution, event_sender, false);
    let mut controller = RealtimeSessionController::new(RealtimeSessionConfig::new(
        "test_session",
        "whisper-large-v3-turbo",
        timestamp_now(),
    ))
    .unwrap();
    controller
        .lifecycle(RealtimeLifecycleAction::Configure, timestamp_now())
        .unwrap();
    controller
        .lifecycle(RealtimeLifecycleAction::StartAudio, timestamp_now())
        .unwrap();
    session.controller = Some(controller);
    session.source_name = Some("Dictation".to_string());
    session.history_text = vec!["remote".to_string(), "client".to_string()];
    session.history_duration_ms = 1_240;

    session.finish("client_closed", true).await.unwrap();

    let store = DaemonHistoryStore::open(temp.path());
    let entries = store.list().unwrap();
    assert!(entries.is_empty());
}

fn native_transcript_final_envelope(utterance: &str, seq: u64) -> RealtimeEventEnvelope {
    native_transcript_final_envelope_with_text(utterance, seq, "native final")
}

fn native_transcript_final_envelope_with_text(
    utterance: &str,
    seq: u64,
    text: &str,
) -> RealtimeEventEnvelope {
    let event = RealtimeEvent::Transcript(RealtimeTranscriptEvent::Final(
        openasr_core::RealtimeTranscriptFinal {
            utterance_id: TranscriptUtteranceId(utterance.to_string()),
            segment_id: TranscriptSegmentId(format!("{utterance}_seg_000001")),
            revision: 1,
            text: text.to_string(),
            start_ms: 0,
            end_ms: 100,
            is_final: true,
            words: Vec::new(),
            language: None,
            speaker: None,
            speaker_label: None,
            speaker_profile_id: None,
        },
    ));
    RealtimeEventEnvelope {
        event_type: event.event_type(),
        session_id: RealtimeSessionId("rt_test".to_string()),
        event_id: openasr_core::RealtimeEventId(format!("evt_{seq:06}")),
        seq,
        created_at: timestamp_now(),
        trace_id: None,
        request_id: None,
        event,
    }
}

fn envelope_speaker(envelope: &RealtimeEventEnvelope) -> Option<String> {
    match &envelope.event {
        RealtimeEvent::Transcript(RealtimeTranscriptEvent::Final(event)) => event.speaker.clone(),
        RealtimeEvent::Transcript(RealtimeTranscriptEvent::Partial(event)) => event.speaker.clone(),
        _ => None,
    }
}

fn envelope_speaker_label(envelope: &RealtimeEventEnvelope) -> Option<String> {
    match &envelope.event {
        RealtimeEvent::Transcript(RealtimeTranscriptEvent::Final(event)) => {
            event.speaker_label.clone()
        }
        RealtimeEvent::Transcript(RealtimeTranscriptEvent::Partial(event)) => {
            event.speaker_label.clone()
        }
        _ => None,
    }
}

fn envelope_speaker_profile_id(envelope: &RealtimeEventEnvelope) -> Option<String> {
    match &envelope.event {
        RealtimeEvent::Transcript(RealtimeTranscriptEvent::Final(event)) => {
            event.speaker_profile_id.clone()
        }
        RealtimeEvent::Transcript(RealtimeTranscriptEvent::Partial(event)) => {
            event.speaker_profile_id.clone()
        }
        _ => None,
    }
}

fn matched_assignment() -> openasr_core::diarize::enrollment::SpeakerDisplayAssignment {
    openasr_core::diarize::enrollment::SpeakerDisplayAssignment {
        speaker_id: openasr_core::diarize::contract::SpeakerId(0),
        speaker: "Alice".to_string(),
        speaker_label: "SPEAKER_00".to_string(),
        speaker_profile_id: Some("vp_aaaaaaaaaaaaaaaa".to_string()),
    }
}

fn resolved_native_speaker_slot(
    assignment: Option<openasr_core::diarize::enrollment::SpeakerDisplayAssignment>,
) -> NativePendingSpeakerSlot {
    NativePendingSpeakerSlot::Resolved(assignment)
}

#[tokio::test]
async fn fallback_backend_result_emits_matched_profile_identity() {
    let (event_sender, mut event_receiver) = mpsc::channel(8);
    let mut session = WsSession::new(ServerRuntime::default(), test_distribution(), event_sender);
    session.controller = Some(started_controller(
        "rt_fallback_identity",
        "whisper-large-v3-turbo",
    ));
    session.pending_utterance_speakers.insert(
        TranscriptUtteranceId("utt_match".to_string()),
        matched_assignment(),
    );

    session
        .apply_backend_result(BackendResult::Final(BackendSuccess {
            utterance_id: TranscriptUtteranceId("utt_match".to_string()),
            start_ms: 0,
            end_ms: 1_000,
            segment_id: TranscriptSegmentId("utt_match_seg_000001".to_string()),
            text: "hello".to_string(),
            language: None,
            words: Vec::new(),
        }))
        .await
        .unwrap();

    let event = event_receiver.try_recv().expect("transcript final");
    assert_eq!(event.event_type, "transcript.final");
    assert_eq!(envelope_speaker(&event), Some("Alice".to_string()));
    assert_eq!(
        envelope_speaker_label(&event),
        Some("SPEAKER_00".to_string())
    );
    assert_eq!(
        envelope_speaker_profile_id(&event),
        Some("vp_aaaaaaaaaaaaaaaa".to_string())
    );
}

// Native true-streaming diarization labels bind to terminal transcripts in
// finalize order; a forced split's terminal segment (label not computed yet)
// stays unlabelled and must not consume another utterance's label.
#[tokio::test]
async fn native_speaker_labels_bind_in_finalize_order() {
    let (event_sender, _event_receiver) = mpsc::channel(8);
    let mut session = WsSession::new(ServerRuntime::default(), test_distribution(), event_sender);

    // Split-terminal segment before utt_1's finalize: queue empty, no label.
    let mut early = native_transcript_final_envelope("utt_1", 1);
    session
        .stamp_native_transcript_speaker(NativeStreamingCommandKind::SplitUtterance, &mut early)
        .await;
    assert_eq!(envelope_speaker(&early), None);

    // Finalize queued utt_1's label; the next terminal transcript binds it.
    session
        .pending_native_speaker_labels
        .push_back(resolved_native_speaker_slot(Some(
            openasr_core::diarize::enrollment::SpeakerDisplayAssignment::anonymous(
                openasr_core::diarize::contract::SpeakerId(0),
            ),
        )));
    let mut terminal = native_transcript_final_envelope("utt_1", 2);
    session
        .stamp_native_transcript_speaker(NativeStreamingCommandKind::Finalize, &mut terminal)
        .await;
    assert_eq!(envelope_speaker(&terminal), Some("SPEAKER_00".to_string()));
    assert_eq!(envelope_speaker_label(&terminal), None);
    assert_eq!(envelope_speaker_profile_id(&terminal), None);

    // Later events of the bound utterance (post-final revisions) reuse it.
    let mut replay = native_transcript_final_envelope("utt_1", 3);
    session
        .stamp_native_transcript_speaker(NativeStreamingCommandKind::Poll, &mut replay)
        .await;
    assert_eq!(envelope_speaker(&replay), Some("SPEAKER_00".to_string()));

    // The next utterance pops its own label, not a stale one.
    session
        .pending_native_speaker_labels
        .push_back(resolved_native_speaker_slot(Some(
            openasr_core::diarize::enrollment::SpeakerDisplayAssignment::anonymous(
                openasr_core::diarize::contract::SpeakerId(1),
            ),
        )));
    let mut second = native_transcript_final_envelope("utt_2", 4);
    session
        .stamp_native_transcript_speaker(NativeStreamingCommandKind::Finalize, &mut second)
        .await;
    assert_eq!(envelope_speaker(&second), Some("SPEAKER_01".to_string()));
}

#[tokio::test]
async fn native_split_terminal_does_not_steal_queued_finalize_label() {
    let (event_sender, _event_receiver) = mpsc::channel(8);
    let mut session = WsSession::new(ServerRuntime::default(), test_distribution(), event_sender);
    session
        .pending_native_speaker_labels
        .push_back(resolved_native_speaker_slot(Some(
            openasr_core::diarize::enrollment::SpeakerDisplayAssignment::anonymous(
                openasr_core::diarize::contract::SpeakerId(0),
            ),
        )));

    let mut split_terminal = native_transcript_final_envelope("utt_split", 1);
    session
        .stamp_native_transcript_speaker(
            NativeStreamingCommandKind::SplitUtterance,
            &mut split_terminal,
        )
        .await;
    assert_eq!(envelope_speaker(&split_terminal), None);
    assert_eq!(session.pending_native_speaker_labels.len(), 1);
    assert!(
        !session
            .native_speaker_by_utterance
            .contains_key(&TranscriptUtteranceId("utt_split".to_string()))
    );

    let mut finalize_terminal = native_transcript_final_envelope("utt_final", 2);
    session
        .stamp_native_transcript_speaker(
            NativeStreamingCommandKind::Finalize,
            &mut finalize_terminal,
        )
        .await;
    assert_eq!(
        envelope_speaker(&finalize_terminal),
        Some("SPEAKER_00".to_string())
    );
    assert!(session.pending_native_speaker_labels.is_empty());
}

#[tokio::test]
async fn native_speaker_change_split_binds_split_label() {
    let (event_sender, _event_receiver) = mpsc::channel(8);
    let mut session = WsSession::new(ServerRuntime::default(), test_distribution(), event_sender);
    session
        .pending_native_split_speaker_slots
        .push_back(resolved_native_speaker_slot(Some(
            openasr_core::diarize::enrollment::SpeakerDisplayAssignment::anonymous(
                openasr_core::diarize::contract::SpeakerId(1),
            ),
        )));

    let mut split_terminal = native_transcript_final_envelope("utt_split", 1);
    session
        .stamp_native_transcript_speaker(
            NativeStreamingCommandKind::SplitUtterance,
            &mut split_terminal,
        )
        .await;

    assert_eq!(
        envelope_speaker(&split_terminal),
        Some("SPEAKER_01".to_string())
    );
    assert!(session.pending_native_split_speaker_slots.is_empty());
    assert_eq!(
        session
            .native_speaker_by_utterance
            .get(&TranscriptUtteranceId("utt_split".to_string()))
            .and_then(|assignment| assignment.as_ref())
            .map(|assignment| assignment.speaker.as_str()),
        Some("SPEAKER_01")
    );
}

#[tokio::test]
async fn native_split_slots_bind_interleaved_max_and_speaker_change_outcomes() {
    let (event_sender, mut event_receiver) = mpsc::channel(8);
    let mut session = WsSession::new(ServerRuntime::default(), test_distribution(), event_sender);
    session
        .pending_native_split_speaker_slots
        .push_back(resolved_native_speaker_slot(None));
    session
        .pending_native_split_speaker_slots
        .push_back(resolved_native_speaker_slot(Some(
            openasr_core::diarize::enrollment::SpeakerDisplayAssignment::anonymous(
                openasr_core::diarize::contract::SpeakerId(1),
            ),
        )));

    session
        .forward_native_streaming_events(
            NativeStreamingCommandKind::SplitUtterance,
            vec![native_transcript_final_envelope("utt_max_split", 1)],
        )
        .await
        .unwrap();
    let max_split = event_receiver.try_recv().expect("max split terminal");
    assert_eq!(envelope_speaker(&max_split), None);
    assert_eq!(session.pending_native_split_speaker_slots.len(), 1);
    assert!(
        matches!(
            session
                .native_speaker_by_utterance
                .get(&TranscriptUtteranceId("utt_max_split".to_string())),
            Some(None)
        ),
        "the unlabelled max split must consume exactly its own slot"
    );

    session
        .forward_native_streaming_events(
            NativeStreamingCommandKind::SplitUtterance,
            vec![native_transcript_final_envelope("utt_speaker_change", 2)],
        )
        .await
        .unwrap();
    let speaker_change = event_receiver
        .try_recv()
        .expect("speaker-change split terminal");
    assert_eq!(
        envelope_speaker(&speaker_change),
        Some("SPEAKER_01".to_string())
    );
    assert!(session.pending_native_split_speaker_slots.is_empty());
}

#[tokio::test]
async fn native_speaker_label_stamping_carries_matched_profile_identity() {
    let (event_sender, _event_receiver) = mpsc::channel(8);
    let mut session = WsSession::new(ServerRuntime::default(), test_distribution(), event_sender);
    session
        .pending_native_speaker_labels
        .push_back(resolved_native_speaker_slot(Some(matched_assignment())));

    let mut terminal = native_transcript_final_envelope("utt_match", 1);
    session
        .stamp_native_transcript_speaker(NativeStreamingCommandKind::Finalize, &mut terminal)
        .await;

    assert_eq!(envelope_speaker(&terminal), Some("Alice".to_string()));
    assert_eq!(
        envelope_speaker_label(&terminal),
        Some("SPEAKER_00".to_string())
    );
    assert_eq!(
        envelope_speaker_profile_id(&terminal),
        Some("vp_aaaaaaaaaaaaaaaa".to_string())
    );
}

struct FixedSpeakerEmbedder;

impl openasr_core::diarize::embed::SpeakerEmbedder for FixedSpeakerEmbedder {
    fn embed(
        &self,
        _samples: &[f32],
        _sample_rate_hz: u32,
    ) -> Result<
        openasr_core::diarize::contract::SpeakerEmbedding,
        openasr_core::diarize::embed::EmbedError,
    > {
        Ok(openasr_core::diarize::contract::SpeakerEmbedding::l2_normalized(vec![1.0, 0.0]))
    }

    fn embedding_dim(&self) -> usize {
        2
    }
}

struct PolaritySpeakerEmbedder;

impl openasr_core::diarize::embed::SpeakerEmbedder for PolaritySpeakerEmbedder {
    fn embed(
        &self,
        samples: &[f32],
        _sample_rate_hz: u32,
    ) -> Result<
        openasr_core::diarize::contract::SpeakerEmbedding,
        openasr_core::diarize::embed::EmbedError,
    > {
        let embedding = if samples.first().copied().unwrap_or_default() >= 0.0 {
            vec![1.0, 0.0]
        } else {
            vec![0.0, 1.0]
        };
        Ok(openasr_core::diarize::contract::SpeakerEmbedding::l2_normalized(embedding))
    }

    fn embedding_dim(&self) -> usize {
        2
    }
}

struct ThreeSpeakerEmbedder;

impl openasr_core::diarize::embed::SpeakerEmbedder for ThreeSpeakerEmbedder {
    fn embed(
        &self,
        samples: &[f32],
        _sample_rate_hz: u32,
    ) -> Result<
        openasr_core::diarize::contract::SpeakerEmbedding,
        openasr_core::diarize::embed::EmbedError,
    > {
        let first = samples.first().copied().unwrap_or_default();
        let embedding = if first > 0.5 {
            vec![0.0, 0.0, 1.0]
        } else if first < 0.0 {
            vec![0.0, 1.0, 0.0]
        } else {
            vec![1.0, 0.0, 0.0]
        };
        Ok(openasr_core::diarize::contract::SpeakerEmbedding::l2_normalized(embedding))
    }

    fn embedding_dim(&self) -> usize {
        3
    }
}

// A stop mid-speech never reaches the VAD SpeechStopped path, so the session
// finish must diarize the retained in-flight audio itself: queueing from the
// retained samples labels the Finish-induced terminal transcript.
#[tokio::test]
async fn finish_mid_speech_labels_inflight_utterance_from_retained_samples() {
    let (event_sender, _event_receiver) = mpsc::channel(8);
    let mut session = WsSession::new(ServerRuntime::default(), test_distribution(), event_sender);
    static EMBEDDER: FixedSpeakerEmbedder = FixedSpeakerEmbedder;
    session.streaming_diarizer =
        Some(openasr_core::diarize::streaming::StreamingDiarizer::with_embedder(&EMBEDDER, 16_000));
    session.native_diarize_samples = vec![0.1; 16_000 * 3];

    session.queue_native_speaker_label().await;

    assert!(session.native_diarize_samples.is_empty());
    let mut terminal = native_transcript_final_envelope("utt_1", 1);
    session
        .stamp_native_transcript_speaker(NativeStreamingCommandKind::Finish, &mut terminal)
        .await;
    assert_eq!(envelope_speaker(&terminal), Some("SPEAKER_00".to_string()));
}

#[tokio::test]
async fn finish_empty_terminal_transcript_does_not_learn_or_stamp_speaker() {
    let (event_sender, _event_receiver) = mpsc::channel(8);
    let mut session = WsSession::new(ServerRuntime::default(), test_distribution(), event_sender);
    static EMBEDDER: ThreeSpeakerEmbedder = ThreeSpeakerEmbedder;
    session.streaming_diarizer =
        Some(openasr_core::diarize::streaming::StreamingDiarizer::with_embedder(&EMBEDDER, 16_000));

    let diarizer = session.streaming_diarizer.as_mut().expect("diarizer");
    let first = diarizer
        .assign_with_path(
            &vec![0.1; 16_000 * 3],
            16_000,
            openasr_core::diarize::streaming::StreamingDiarizePath::Native,
        )
        .expect("first speaker");
    let second = diarizer
        .assign_with_path(
            &vec![-0.1; 16_000 * 3],
            16_000,
            openasr_core::diarize::streaming::StreamingDiarizePath::Native,
        )
        .expect("second speaker");
    assert_eq!(first.speaker, "SPEAKER_00");
    assert_eq!(second.speaker, "SPEAKER_01");
    assert_eq!(diarizer.registry().speaker_count(), 2);

    session.native_diarize_samples = vec![0.7; 16_000 * 3];

    session.queue_native_speaker_label().await;
    assert!(session.native_diarize_samples.is_empty());
    assert_eq!(session.pending_native_speaker_labels.len(), 1);
    assert_eq!(
        session
            .streaming_diarizer
            .as_ref()
            .expect("diarizer kept")
            .registry()
            .speaker_count(),
        2,
        "queueing close-time samples must not learn SPEAKER_02 before transcript text is known"
    );

    let mut terminal = native_transcript_final_envelope_with_text("utt_empty", 1, "");
    session
        .stamp_native_transcript_speaker(NativeStreamingCommandKind::Finish, &mut terminal)
        .await;

    assert_eq!(envelope_speaker(&terminal), None);
    assert!(session.pending_native_speaker_labels.is_empty());
    assert!(
        !session
            .native_speaker_by_utterance
            .contains_key(&TranscriptUtteranceId("utt_empty".to_string()))
    );
    assert_eq!(
        session
            .streaming_diarizer
            .as_ref()
            .expect("diarizer kept")
            .registry()
            .speaker_count(),
        2
    );
}

#[tokio::test]
async fn native_max_utterance_boundary_resets_diarization_for_later_speaker_change() {
    let (event_sender, _event_receiver) = mpsc::channel(8);
    let mut session = WsSession::new(ServerRuntime::default(), test_distribution(), event_sender);
    static EMBEDDER: PolaritySpeakerEmbedder = PolaritySpeakerEmbedder;
    session.streaming_diarizer =
        Some(openasr_core::diarize::streaming::StreamingDiarizer::with_embedder(&EMBEDDER, 16_000));
    session.native_speaker_change_detector = Some(
        openasr_core::diarize::streaming::StreamingSpeakerChangeDetector::with_embedder(
            &EMBEDDER, 16_000,
        ),
    );

    session.native_diarize_samples = vec![0.1; 16_000 * 30];
    assert!(
        !session
            .maybe_split_native_on_speaker_change()
            .await
            .unwrap(),
        "same-speaker speech at the retention cap should only advance the detector"
    );

    session
        .queue_native_max_utterance_split_speaker_slot()
        .await;

    assert!(session.native_diarize_samples.is_empty());
    assert_eq!(session.pending_native_split_speaker_slots.len(), 1);
    assert!(matches!(
        session.pending_native_split_speaker_slots.front(),
        Some(NativePendingSpeakerSlot::DeferredSamples(_))
    ));

    let mut post_boundary = vec![0.1; 16_000 * 5 / 2];
    post_boundary.extend(vec![-0.1; 16_000 * 5 / 2]);
    session.native_diarize_samples = post_boundary;

    assert!(
        session
            .maybe_split_native_on_speaker_change()
            .await
            .unwrap(),
        "detector must continue analyzing after the max-duration boundary"
    );
    assert_eq!(
        session.pending_native_split_speaker_slots.len(),
        2,
        "speaker-change split queues behind the prior max-duration split"
    );
    assert_eq!(session.native_diarize_samples.len(), 16_000 * 5 / 2);
}

// ---------------------------------------------------------------------------
// Retroactive speaker attribution (speakerless sentence finals + change-split
// word reattribution).
// ---------------------------------------------------------------------------

fn native_final_envelope_with(
    utterance: &str,
    segment: &str,
    seq: u64,
    revision: u64,
    text: &str,
    start_ms: u64,
    end_ms: u64,
    words: Vec<RealtimeTranscriptWord>,
) -> RealtimeEventEnvelope {
    let event = RealtimeEvent::Transcript(RealtimeTranscriptEvent::Final(
        openasr_core::RealtimeTranscriptFinal {
            utterance_id: TranscriptUtteranceId(utterance.to_string()),
            segment_id: TranscriptSegmentId(segment.to_string()),
            revision,
            text: text.to_string(),
            start_ms,
            end_ms,
            is_final: true,
            words,
            language: None,
            speaker: None,
            speaker_label: None,
            speaker_profile_id: None,
        },
    ));
    RealtimeEventEnvelope {
        event_type: event.event_type(),
        session_id: RealtimeSessionId("rt_test".to_string()),
        event_id: openasr_core::RealtimeEventId(format!("evt_{seq:06}")),
        seq,
        created_at: timestamp_now(),
        trace_id: None,
        request_id: None,
        event,
    }
}

fn rt_word(word: &str, start_ms: u64, end_ms: u64) -> RealtimeTranscriptWord {
    RealtimeTranscriptWord {
        word: word.to_string(),
        start_ms,
        end_ms,
        confidence: None,
    }
}

fn revision_event(envelope: &RealtimeEventEnvelope) -> &openasr_core::RealtimeTranscriptRevision {
    match &envelope.event {
        RealtimeEvent::Transcript(RealtimeTranscriptEvent::Revision(event)) => event,
        other => panic!("expected transcript.revision, got {other:?}"),
    }
}

fn final_event(envelope: &RealtimeEventEnvelope) -> &openasr_core::RealtimeTranscriptFinal {
    match &envelope.event {
        RealtimeEvent::Transcript(RealtimeTranscriptEvent::Final(event)) => event,
        other => panic!("expected transcript.final, got {other:?}"),
    }
}

// A mid-utterance sentence-cut final goes to the client before the
// utterance's label binds (labels bind on the terminal transcript). Once the
// label binds, the speakerless line must be revised retroactively with the
// speaker attached, referencing the client-visible event id it revises.
#[tokio::test]
async fn speakerless_sentence_final_is_revised_when_the_label_binds() {
    let (event_sender, mut event_receiver) = mpsc::channel(16);
    let mut session = WsSession::new(ServerRuntime::default(), test_distribution(), event_sender);
    static EMBEDDER: FixedSpeakerEmbedder = FixedSpeakerEmbedder;
    session.streaming_diarizer =
        Some(openasr_core::diarize::streaming::StreamingDiarizer::with_embedder(&EMBEDDER, 16_000));

    // Sentence cut emitted from a partial Poll: no label exists yet.
    session
        .forward_native_streaming_events(
            NativeStreamingCommandKind::Poll,
            vec![native_final_envelope_with(
                "utt_1",
                "seg_a",
                1,
                5,
                "第一句。",
                0,
                2_000,
                Vec::new(),
            )],
        )
        .await
        .unwrap();
    let sentence = event_receiver.try_recv().expect("sentence final");
    assert_eq!(sentence.event_type, "transcript.final");
    assert_eq!(envelope_speaker(&sentence), None);
    assert_eq!(session.native_speakerless_finals.len(), 1);

    // Terminal final binds the utterance label.
    session
        .pending_native_speaker_labels
        .push_back(resolved_native_speaker_slot(Some(
            openasr_core::diarize::enrollment::SpeakerDisplayAssignment::anonymous(
                openasr_core::diarize::contract::SpeakerId(0),
            ),
        )));
    session
        .forward_native_streaming_events(
            NativeStreamingCommandKind::Finalize,
            vec![native_final_envelope_with(
                "utt_1",
                "seg_b",
                2,
                7,
                "第二句。",
                2_000,
                4_000,
                Vec::new(),
            )],
        )
        .await
        .unwrap();

    let terminal = event_receiver.try_recv().expect("terminal final");
    assert_eq!(terminal.event_type, "transcript.final");
    assert_eq!(envelope_speaker(&terminal), Some("SPEAKER_00".to_string()));

    let revision = event_receiver.try_recv().expect("retroactive revision");
    assert_eq!(revision.event_type, "transcript.revision");
    let revision = revision_event(&revision);
    assert_eq!(revision.segment_id.0, "seg_a");
    assert_eq!(revision.text, "第一句。");
    assert_eq!(revision.revision, 6, "one past the original final");
    assert!(revision.is_final);
    assert_eq!(revision.speaker.as_deref(), Some("SPEAKER_00"));
    assert_eq!(
        revision.revises_event_id.as_ref().map(|id| id.0.as_str()),
        Some(sentence.event_id.0.as_str()),
        "must reference the client-visible id of the original final"
    );
    assert!(session.native_speakerless_finals.is_empty());
}

// A speakerless final whose utterance resolves UNLABELLED must be dropped
// without a revision; records of finished utterances must not leak across
// utterances.
#[tokio::test]
async fn speakerless_final_of_unlabelled_utterance_is_dropped() {
    let (event_sender, mut event_receiver) = mpsc::channel(16);
    let mut session = WsSession::new(ServerRuntime::default(), test_distribution(), event_sender);
    static EMBEDDER: FixedSpeakerEmbedder = FixedSpeakerEmbedder;
    session.streaming_diarizer =
        Some(openasr_core::diarize::streaming::StreamingDiarizer::with_embedder(&EMBEDDER, 16_000));

    session
        .forward_native_streaming_events(
            NativeStreamingCommandKind::Poll,
            vec![native_final_envelope_with(
                "utt_1",
                "seg_a",
                1,
                5,
                "第一句。",
                0,
                2_000,
                Vec::new(),
            )],
        )
        .await
        .unwrap();
    let _ = event_receiver.try_recv().expect("sentence final");

    // Terminal binds an explicit None (unlabelled short/low-confidence).
    session
        .pending_native_speaker_labels
        .push_back(resolved_native_speaker_slot(None));
    session
        .forward_native_streaming_events(
            NativeStreamingCommandKind::Finalize,
            vec![native_final_envelope_with(
                "utt_1",
                "seg_b",
                2,
                7,
                "第二句。",
                2_000,
                4_000,
                Vec::new(),
            )],
        )
        .await
        .unwrap();
    let terminal = event_receiver.try_recv().expect("terminal final");
    assert_eq!(envelope_speaker(&terminal), None);
    assert!(
        event_receiver.try_recv().is_err(),
        "no retroactive revision for an unlabelled utterance"
    );
    assert!(session.native_speakerless_finals.is_empty());
}

// Speaker-change split with word timestamps: the trailing words after the
// estimated change point are carved off the OLD speaker's terminal final
// (trim revision) and re-emitted as their own segment, which is relabelled
// once the NEXT utterance's speaker binds.
#[tokio::test]
async fn speaker_change_split_reattributes_trailing_words_to_the_new_speaker() {
    let (event_sender, mut event_receiver) = mpsc::channel(16);
    let mut session = WsSession::new(ServerRuntime::default(), test_distribution(), event_sender);
    static EMBEDDER: FixedSpeakerEmbedder = FixedSpeakerEmbedder;
    session.streaming_diarizer =
        Some(openasr_core::diarize::streaming::StreamingDiarizer::with_embedder(&EMBEDDER, 16_000));

    // OLD speaker's label for the split's "before" audio, and the change
    // point estimate queued by the speaker-change split.
    session
        .pending_native_split_speaker_slots
        .push_back(resolved_native_speaker_slot(Some(
            openasr_core::diarize::enrollment::SpeakerDisplayAssignment::anonymous(
                openasr_core::diarize::contract::SpeakerId(0),
            ),
        )));
    session
        .pending_native_split_change_points
        .push_back(Some(25_500));

    let words = vec![
        rt_word("还特意给出了具体的过程", 22_000, 25_400),
        rt_word("那现在", 25_700, 26_500),
        rt_word("又回到了我", 26_500, 27_600),
    ];
    session
        .forward_native_streaming_events(
            NativeStreamingCommandKind::SplitUtterance,
            vec![native_final_envelope_with(
                "utt_old",
                "seg_x",
                1,
                9,
                "还特意给出了具体的过程。那现在又回到了我。",
                22_000,
                27_600,
                words,
            )],
        )
        .await
        .unwrap();

    let original = event_receiver.try_recv().expect("split terminal final");
    assert_eq!(original.event_type, "transcript.final");
    assert_eq!(envelope_speaker(&original), Some("SPEAKER_00".to_string()));

    let trimmed = event_receiver.try_recv().expect("trim revision");
    assert_eq!(trimmed.event_type, "transcript.revision");
    let trimmed = revision_event(&trimmed);
    assert_eq!(trimmed.segment_id.0, "seg_x");
    assert_eq!(trimmed.text, "还特意给出了具体的过程。");
    assert_eq!(trimmed.end_ms, 25_400);
    assert_eq!(trimmed.revision, 10);
    assert_eq!(trimmed.speaker.as_deref(), Some("SPEAKER_00"));
    assert_eq!(
        trimmed.revises_event_id.as_ref().map(|id| id.0.as_str()),
        Some(original.event_id.0.as_str())
    );

    let moved = event_receiver.try_recv().expect("moved tail final");
    assert_eq!(moved.event_type, "transcript.final");
    let moved_final = final_event(&moved);
    assert_eq!(moved_final.segment_id.0, "seg_x_sw");
    assert_eq!(moved_final.text, "那现在又回到了我。");
    assert_eq!(moved_final.start_ms, 25_700);
    assert_eq!(moved_final.speaker, None, "new speaker not known yet");
    assert_eq!(session.pending_split_tail_relabels.len(), 1);

    // The NEXT utterance's terminal binds the NEW speaker; the moved tail is
    // relabelled with it.
    session
        .pending_native_speaker_labels
        .push_back(resolved_native_speaker_slot(Some(
            openasr_core::diarize::enrollment::SpeakerDisplayAssignment::anonymous(
                openasr_core::diarize::contract::SpeakerId(1),
            ),
        )));
    session
        .forward_native_streaming_events(
            NativeStreamingCommandKind::Finalize,
            vec![native_final_envelope_with(
                "utt_new",
                "seg_y",
                2,
                3,
                "你能听出来这个声音是我吗？",
                27_600,
                31_000,
                Vec::new(),
            )],
        )
        .await
        .unwrap();
    let new_terminal = event_receiver.try_recv().expect("new utterance terminal");
    assert_eq!(
        envelope_speaker(&new_terminal),
        Some("SPEAKER_01".to_string())
    );
    let relabel = event_receiver.try_recv().expect("tail relabel revision");
    assert_eq!(relabel.event_type, "transcript.revision");
    let relabel = revision_event(&relabel);
    assert_eq!(relabel.segment_id.0, "seg_x_sw");
    assert_eq!(relabel.text, "那现在又回到了我。");
    assert_eq!(relabel.speaker.as_deref(), Some("SPEAKER_01"));
    assert_eq!(
        relabel.revises_event_id.as_ref().map(|id| id.0.as_str()),
        Some(moved.event_id.0.as_str())
    );
    assert!(session.pending_split_tail_relabels.is_empty());
}

// Families without realtime word timestamps cannot carve the text faithfully:
// the change point must be consumed without any reattribution (current
// behavior preserved).
#[tokio::test]
async fn speaker_change_split_without_words_falls_back_to_no_reattribution() {
    let (event_sender, mut event_receiver) = mpsc::channel(16);
    let mut session = WsSession::new(ServerRuntime::default(), test_distribution(), event_sender);
    static EMBEDDER: FixedSpeakerEmbedder = FixedSpeakerEmbedder;
    session.streaming_diarizer =
        Some(openasr_core::diarize::streaming::StreamingDiarizer::with_embedder(&EMBEDDER, 16_000));
    session
        .pending_native_split_speaker_slots
        .push_back(resolved_native_speaker_slot(Some(
            openasr_core::diarize::enrollment::SpeakerDisplayAssignment::anonymous(
                openasr_core::diarize::contract::SpeakerId(0),
            ),
        )));
    session
        .pending_native_split_change_points
        .push_back(Some(25_500));

    session
        .forward_native_streaming_events(
            NativeStreamingCommandKind::SplitUtterance,
            vec![native_final_envelope_with(
                "utt_old",
                "seg_x",
                1,
                9,
                "还特意给出了具体的过程。那现在又回到了我。",
                22_000,
                27_600,
                Vec::new(),
            )],
        )
        .await
        .unwrap();

    let original = event_receiver.try_recv().expect("split terminal final");
    assert_eq!(original.event_type, "transcript.final");
    assert!(
        event_receiver.try_recv().is_err(),
        "no synthetic events without word anchors"
    );
    assert!(session.pending_native_split_change_points.is_empty());
    assert!(session.pending_split_tail_relabels.is_empty());
}

#[test]
fn diarize_sample_spans_map_split_samples_to_stream_time() {
    // Three 320-sample frames retained at 1 000/1 020/1 040 ms.
    let spans = vec![(0usize, 1_000u64), (320, 1_020), (640, 1_040)];
    assert_eq!(diarize_sample_abs_ms(&spans, 0), Some(1_000));
    assert_eq!(diarize_sample_abs_ms(&spans, 160), Some(1_010));
    assert_eq!(diarize_sample_abs_ms(&spans, 320), Some(1_020));
    assert_eq!(diarize_sample_abs_ms(&spans, 800), Some(1_050));
    assert_eq!(diarize_sample_abs_ms(&[], 100), None);

    // Rebase after carving 480 samples off the front: the straddled anchor
    // becomes the new head at its mid-frame time.
    let rebased = rebase_diarize_sample_spans(spans, 480);
    assert_eq!(rebased, vec![(0, 1_030), (160, 1_040)]);
    assert_eq!(diarize_sample_abs_ms(&rebased, 0), Some(1_030));
}
