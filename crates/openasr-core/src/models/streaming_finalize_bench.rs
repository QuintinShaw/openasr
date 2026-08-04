//! Host-local measurement harness for realtime session finalize latency
//! (`session.close -> terminal final`). Ignored by default: it needs real
//! `.oasr` packs and audio under the developer's `~/.openasr` / repo `tmp`.
//!
//! It faithfully replays the server's single-threaded worker ordering: push
//! 20 ms frames, poll on the family cadence, then a 200 ms silence "grace tail"
//! (frames that arrive after the user releases), then `finish()`, timing only
//! the finalize decode. WS transport is sub-millisecond locally, so the
//! `finish()` wall time is the server-side close->final cost we care about.

#![cfg(test)]

use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::arch::OpenAsrArchitectureRegistry;
use crate::models::ggml_asr_executor::GgmlAsrStreamingSessionRequest;
use crate::models::pack_verifier::{PackCandidate, PackRoute, PackVerifier};
use crate::models::runtime_selection_metadata::selection_metadata_from_gguf;
use crate::{
    GgmlAsrBackendPreference, GgmlAsrExecutionOptions, NativeAsrSession, NativeAsrSessionContext,
    NativeAsrStreamingSessionConfig, RealtimeAudioFormat, RealtimeAudioFrame, RealtimeEvent,
    RealtimeEventEnvelope, RealtimeTranscriptEvent,
};

const FRAME_MS: u64 = 20;
const GRACE_TAIL_MS: u64 = 200;

fn manifest_relative(path: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(path)
}

fn home_pack(rel: &str) -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    let path = PathBuf::from(home).join(".openasr/models").join(rel);
    path.exists().then_some(path)
}

fn load_wav(path: &Path) -> Vec<f32> {
    crate::api::audio_io::load_wav_16khz_mono_f32_v0(
        path,
        "finalize-latency-bench",
        "finalize-latency-bench",
    )
    .expect("wav should load as 16k mono f32")
}

fn frame_from_samples(seq: u64, start_ms: u64, samples: &[f32]) -> RealtimeAudioFrame {
    // Pad a short trailing chunk up to a full 20 ms frame (320 samples): the
    // realtime frame contract only accepts 10/20/30 ms durations. Batch and
    // streaming pad identically, so parity is preserved.
    let mut pcm: Vec<i16> = samples
        .iter()
        .map(|s| (s.clamp(-1.0, 1.0) * 32767.0).round() as i16)
        .collect();
    pcm.resize(320, 0);
    RealtimeAudioFrame::new(seq, start_ms, RealtimeAudioFormat::pcm16_mono_16khz(), pcm)
        .expect("frame")
}

fn terminal_final_text(events: &[RealtimeEventEnvelope]) -> Option<String> {
    let mut text = None;
    for envelope in events {
        match &envelope.event {
            RealtimeEvent::Transcript(RealtimeTranscriptEvent::Final(final_)) => {
                text = Some(final_.text.clone());
            }
            RealtimeEvent::Transcript(RealtimeTranscriptEvent::Revision(revision))
                if revision.is_final =>
            {
                text = Some(revision.text.clone());
            }
            _ => {}
        }
    }
    text
}

fn build_request(pack: &Path) -> GgmlAsrStreamingSessionRequest {
    // Verify and open the source exactly once. The request retains this
    // proof (including its open source generation) through dispatch; the
    // benchmark must not reconstruct a family from a path or re-run a second
    // metadata/tensor-index scan before starting the session.
    let verified_pack = PackVerifier
        .verify_candidate(PackCandidate::new(pack.to_path_buf()))
        .expect("benchmark runtime must pass PackVerifier");
    let PackRoute::Asr {
        model_architecture, ..
    } = verified_pack.route()
    else {
        panic!("benchmark source is not an ASR pack: {}", pack.display());
    };
    let model_architecture = *model_architecture;
    let architecture = OpenAsrArchitectureRegistry::with_builtins()
        .find_by_model_architecture(model_architecture)
        .unwrap_or_else(|| {
            panic!(
                "verified benchmark architecture is absent from the canonical inventory: {model_architecture}"
            )
        });
    let selection_metadata = selection_metadata_from_gguf(verified_pack.preflight().metadata());
    let selected_family = OpenAsrArchitectureRegistry::with_builtins()
        .select_ggml_adapter_from_gguf_metadata_v1(&selection_metadata)
        .expect("verified ASR metadata must select one builtin family")
        .ggml_family_adapter_descriptor();
    assert_eq!(
        selected_family.model_architecture, model_architecture,
        "PackVerifier route and metadata family selection disagree"
    );
    assert_eq!(
        architecture.identity.model_architecture, selected_family.model_architecture,
        "architecture registry and adapter projection disagree"
    );
    GgmlAsrStreamingSessionRequest {
        execution_services:
            crate::models::native_execution_services::test_native_execution_services(),
        decoder_state: crate::models::ggml_asr_executor::GgmlAsrDecoderState::NoPersistentState,
        verified_pack,
        selected_family,
        request_options: GgmlAsrExecutionOptions::default(),
        configured_diarize: false,
        backend_preference: GgmlAsrBackendPreference::CpuOnly,
        resolved_runtime: crate::ggml_runtime::ResolvedFamilyRuntimeInput::resolve(
            (GgmlAsrBackendPreference::CpuOnly).request_backend_override(),
            crate::ggml_runtime::AutoGpuPolicy::AllBackends,
        ),
        final_text_processor: None,
        session_context: NativeAsrSessionContext::new("rt_finalize_bench"),
        session_config: NativeAsrStreamingSessionConfig::new()
            .with_partial_results(true)
            .into(),
    }
}

fn start_session(pack: &Path, partial_results: bool) -> Box<dyn NativeAsrSession> {
    let mut request = build_request(pack);
    request.session_config = NativeAsrStreamingSessionConfig::new()
        .with_partial_results(partial_results)
        .into();
    request
        .execution_services
        .streaming_dispatch()
        .start_streaming_session(&request)
        .expect("streaming session should start")
}

/// Offline batch reference: a partials-OFF session decodes the full buffer
/// once at finish (`decode_full_buffer`), which is byte-identical to the
/// offline `execute()` path — the parity contract's "batch" side. It sees the
/// SAME audio the streaming run does (speech + 200 ms grace tail), so the tail
/// cannot be the source of a spurious mismatch.
fn batch_reference(pack: &Path, samples: &[f32]) -> String {
    let mut session = start_session(pack, false);
    session.warm_up().expect("warm up");
    let mut seq = 0;
    for chunk in samples.chunks(320) {
        let start_ms = seq * FRAME_MS;
        let _ = session
            .push_audio(frame_from_samples(seq, start_ms, chunk))
            .expect("push");
        seq += 1;
    }
    for _ in 0..(GRACE_TAIL_MS / FRAME_MS) {
        let start_ms = seq * FRAME_MS;
        let _ = session
            .push_audio(frame_from_samples(seq, start_ms, &[0.0f32; 320]))
            .expect("push tail");
        seq += 1;
    }
    let events = session.finish().expect("finish");
    terminal_final_text(&events).unwrap_or_default()
}

struct RunOutcome {
    finalize_ms: u128,
    final_text: String,
}

/// Replay one utterance with partials on + a 200 ms grace tail, timing finish().
///
/// `poll_after_tail` models the difference that decides the reuse hit:
/// - `false` (realistic dictation stop): the last partial poll runs during
///   speech, then the grace-tail frames arrive with no poll, then close. The
///   buffer grew after the last whole-buffer decode -> finalize MISSES the
///   reuse cache and does a full re-decode.
/// - `true` (reuse ceiling): a poll runs after the whole tail-inclusive buffer
///   is present, so finalize HITS the cache and reuses that decode.
fn run_utterance(
    pack: &Path,
    samples: &[f32],
    poll_every_ms: u64,
    poll_after_tail: bool,
) -> RunOutcome {
    let mut session = start_session(pack, true);
    session.warm_up().expect("warm up");
    let mut seq = 0u64;
    let mut last_poll_ms = 0u64;
    for chunk in samples.chunks(320) {
        let start_ms = seq * FRAME_MS;
        let _ = session
            .push_audio(frame_from_samples(seq, start_ms, chunk))
            .expect("push");
        seq += 1;
        if start_ms.saturating_sub(last_poll_ms) >= poll_every_ms {
            let _ = session.poll_events().expect("poll");
            last_poll_ms = start_ms;
        }
    }
    // 200 ms grace tail of near-silence arriving after the user released. These
    // frames land after the last partial poll, so the buffer grows before close.
    for _ in 0..(GRACE_TAIL_MS / FRAME_MS) {
        let start_ms = seq * FRAME_MS;
        let _ = session
            .push_audio(frame_from_samples(seq, start_ms, &[0.0f32; 320]))
            .expect("push tail");
        seq += 1;
    }
    if poll_after_tail {
        let _ = session.poll_events().expect("poll after tail");
    }
    let started = Instant::now();
    let events = session.finish().expect("finish");
    let finalize_ms = started.elapsed().as_millis();
    RunOutcome {
        finalize_ms,
        final_text: terminal_final_text(&events).unwrap_or_default(),
    }
}

fn summarize(label: &str, mut samples_ms: Vec<u128>) {
    samples_ms.sort_unstable();
    let n = samples_ms.len();
    let median = samples_ms[n / 2];
    let min = samples_ms[0];
    let max = samples_ms[n - 1];
    let p90 = samples_ms[(n * 9 / 10).min(n - 1)];
    eprintln!(
        "[finalize-latency] {label}: n={n} median={median}ms min={min}ms p90={p90}ms max={max}ms all={samples_ms:?}"
    );
}

fn run_family_bench(descriptor_id: &str, pack: &Path, wav: &Path, iters: usize) {
    let mut samples = load_wav(wav);
    // Cap to a realistic short-dictation length (~4 s); the measured finalize
    // floor is per-short-utterance, and it keeps autoregressive qwen bounded.
    const MAX_SAMPLES: usize = 4 * 16_000;
    samples.truncate(MAX_SAMPLES);
    let audio_ms = samples.len() as u64 * 1000 / 16_000;
    eprintln!(
        "[finalize-latency] {descriptor_id}: pack={} audio={audio_ms}ms",
        pack.display()
    );
    let batch = batch_reference(pack, &samples);
    eprintln!("[finalize-latency] {descriptor_id}: batch_reference={batch:?}");

    // Realistic dictation stop: grace tail arrives after the last poll -> reuse
    // MISSES, finalize does a full re-decode (parity with batch either way).
    let mut realistic_ms = Vec::with_capacity(iters);
    // Reuse ceiling: a poll covers the whole tail-inclusive buffer before close
    // -> finalize HITS the reuse cache and skips the redundant decode.
    let mut ceiling_ms = Vec::with_capacity(iters);
    for i in 0..iters {
        let realistic = run_utterance(pack, &samples, 300, false);
        assert_eq!(
            realistic.final_text, batch,
            "iter {i}: realistic terminal final must equal offline batch (parity)"
        );
        realistic_ms.push(realistic.finalize_ms);

        let ceiling = run_utterance(pack, &samples, 300, true);
        assert_eq!(
            ceiling.final_text, batch,
            "iter {i}: reuse-path terminal final must equal offline batch (parity)"
        );
        ceiling_ms.push(ceiling.finalize_ms);
    }
    summarize(
        &format!("{descriptor_id} realistic-tail (reuse miss)"),
        realistic_ms,
    );
    summarize(
        &format!("{descriptor_id} poll-after-tail (reuse hit)"),
        ceiling_ms,
    );
}

#[test]
#[ignore = "host-local: needs sensevoice-small q8_0 pack + tmp audio"]
fn finalize_latency_sensevoice_q8() {
    let Some(pack) = home_pack("sensevoice-small/q8_0/sensevoice-small-q8_0.oasr") else {
        panic!("sensevoice q8_0 pack absent");
    };
    let wav = manifest_relative("../../tmp/audio/sensevoice/zh.wav");
    run_family_bench("sensevoice", &pack, &wav, 12);
}

#[test]
#[ignore = "host-local: needs qwen3-asr-0.6b q4_k pack + jfk fixture"]
fn finalize_latency_qwen_q4() {
    let Some(pack) = home_pack("qwen3-asr-0.6b/q4_k/qwen3-asr-0.6b-q4_k.oasr") else {
        panic!("qwen q4_k pack absent");
    };
    let wav = manifest_relative("../../fixtures/jfk.wav");
    run_family_bench("qwen", &pack, &wav, 12);
}

#[test]
#[ignore = "host-local: needs dolphin-cn-dialect-small fp16 pack + tmp audio"]
fn finalize_latency_dolphin_fp16() {
    // Dolphin routes through the seq2seq incremental driver too, so it shares
    // the whole-buffer finalize reuse; verify byte-identical parity + latency.
    let Some(pack) = home_pack("dolphin-cn-dialect-small/fp16/dolphin-cn-dialect-small-fp16.oasr")
    else {
        panic!("dolphin fp16 pack absent");
    };
    let wav = manifest_relative("../../tmp/audio/sensevoice/zh.wav");
    run_family_bench("dolphin", &pack, &wav, 12);
}
