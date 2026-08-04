use crate::models::frame_sync_streaming_driver::IncrementalAudioDecoder;
use crate::models::ggml_asr_executor::{GgmlAsrExecutionError, GgmlAsrStreamingSessionRequest};
use crate::models::graph_runtime_config::install_request_inference_threads_override;

use super::frontend::{
    XASR_FINAL_FLUSH_TAIL_PAD_SAMPLES, XASR_N_MELS, XasrFbankFeatures, XasrFbankFrontend,
    clean_frame_count_for_samples, earliest_sample_needed_for_frame,
    samples_needed_for_clean_frame_count, total_frame_count_for_samples,
};
use super::runtime::{
    HopDecodeOutcome, XasrChunkedDecodeState, XasrRuntimeActor, XasrZipformerPreparedRuntime,
};
use super::tokenizer::XasrStreamingDetokenizer;

const XASR_STREAMING_BASELINE_LEFT_CONTEXT_TOKENS: usize = 16;

pub(crate) struct XasrIncrementalDecoder {
    executor_id: &'static str,
    adapter_id: &'static str,
    request: GgmlAsrStreamingSessionRequest,
    runtime: XasrRuntimeActor,
    decode_state: Option<XasrChunkedDecodeState>,
    audio: Vec<f32>,
    /// Samples drained from the front of `audio`; all sample/frame indices
    /// below stay absolute against the full stream.
    dropped_samples: usize,
    frontend: XasrFbankFrontend,
    /// Cached fbank rows for frames already free of right-edge reflection;
    /// those rows never change as audio grows, so each push only pays for
    /// newly clean frames instead of recomputing the whole buffer (O(n^2)).
    features: XasrFbankFeatures,
    /// Feature rows drained from the front of `features`; together with the
    /// audio drain a session holds O(1) memory however long an utterance runs.
    dropped_frames: usize,
    /// Exact streaming detokenizer state; `decoded_tokens` counts how many of
    /// `decode_state.emitted` have been fed, so each delta only detokenizes
    /// the NEW tokens instead of re-decoding the whole utterance history.
    detokenizer: XasrStreamingDetokenizer,
    decoded_tokens: usize,
}

impl XasrIncrementalDecoder {
    pub(super) fn new(
        request: &GgmlAsrStreamingSessionRequest,
        executor_id: &'static str,
        adapter_id: &'static str,
        runtime: XasrRuntimeActor,
    ) -> Result<Self, GgmlAsrExecutionError> {
        let decode_state = runtime
            .call_mut(|runtime| runtime.new_decode_state())
            .map_err(|error| {
                GgmlAsrExecutionError::executor_failed(executor_id, adapter_id, error.to_string())
            })?;
        Ok(Self {
            executor_id,
            adapter_id,
            request: request.clone(),
            runtime,
            decode_state: Some(decode_state),
            audio: Vec::new(),
            dropped_samples: 0,
            frontend: XasrFbankFrontend::new(),
            features: XasrFbankFeatures {
                data: Vec::new(),
                n_frames: 0,
                n_mels: XASR_N_MELS,
            },
            dropped_frames: 0,
            detokenizer: XasrStreamingDetokenizer::default(),
            decoded_tokens: 0,
        })
    }

    fn failed(&self, reason: impl Into<String>) -> GgmlAsrExecutionError {
        GgmlAsrExecutionError::executor_failed(self.executor_id, self.adapter_id, reason)
    }

    fn decode_state(&self) -> Result<&XasrChunkedDecodeState, GgmlAsrExecutionError> {
        self.decode_state
            .as_ref()
            .ok_or_else(|| self.failed("xasr runtime actor terminated"))
    }

    fn decode_state_mut(&mut self) -> Result<&mut XasrChunkedDecodeState, GgmlAsrExecutionError> {
        let executor_id = self.executor_id;
        let adapter_id = self.adapter_id;
        self.decode_state.as_mut().ok_or_else(|| {
            GgmlAsrExecutionError::executor_failed(
                executor_id,
                adapter_id,
                "xasr runtime actor terminated",
            )
        })
    }

    /// Moves the session-local decode state and feature window through the
    /// command channel while the thread-affine runtime stays on its owner
    /// actor. Both values are restored before an ordinary model error is
    /// returned. A transport/panic failure is terminal for this actor, so the
    /// session intentionally remains without a decode state and fails closed.
    fn call_decode<O, F>(&mut self, operation: F) -> Result<O, GgmlAsrExecutionError>
    where
        O: Send + 'static,
        F: FnOnce(
                &mut XasrZipformerPreparedRuntime,
                &mut XasrChunkedDecodeState,
                &XasrFbankFeatures,
            ) -> Result<O, String>
            + Send
            + 'static,
    {
        let mut state = self
            .decode_state
            .take()
            .ok_or_else(|| self.failed("xasr runtime actor terminated"))?;
        let features = std::mem::replace(
            &mut self.features,
            XasrFbankFeatures {
                data: Vec::new(),
                n_frames: 0,
                n_mels: XASR_N_MELS,
            },
        );
        let response = self.runtime.call_mut(move |runtime| {
            let result = operation(runtime, &mut state, &features);
            (state, features, result)
        });
        let (state, features, result) = response.map_err(|error| self.failed(error.to_string()))?;
        self.decode_state = Some(state);
        self.features = features;
        result.map_err(|error| self.failed(error))
    }

    fn decode_available_chunks_on_actor(
        &mut self,
        final_flush: bool,
    ) -> Result<usize, GgmlAsrExecutionError> {
        self.call_decode(move |runtime, state, features| {
            runtime.decode_available_chunks(state, features, final_flush)
        })
    }

    fn decode_next_chunk_on_actor(
        &mut self,
        final_flush: bool,
    ) -> Result<HopDecodeOutcome, GgmlAsrExecutionError> {
        self.call_decode(move |runtime, state, features| {
            runtime.decode_next_chunk(state, features, final_flush)
        })
    }

    /// Extends the feature cache up to `target_total_frames` (an absolute
    /// frame count against the full stream).
    fn extend_feature_rows(
        &mut self,
        target_total_frames: usize,
    ) -> Result<(), GgmlAsrExecutionError> {
        let cached_total = self.dropped_frames + self.features.n_frames;
        if target_total_frames <= cached_total {
            return Ok(());
        }
        let rows = self
            .frontend
            .features_for_frame_range_from(
                &self.audio,
                self.dropped_samples,
                cached_total,
                target_total_frames,
            )
            .map_err(|error| self.failed(error.to_string()))?;
        self.features.data.extend_from_slice(&rows);
        self.features.n_frames = target_total_frames - self.dropped_frames;
        Ok(())
    }

    /// Drops feature rows the chunk loop consumed and audio samples no future
    /// fbank frame can read, keeping per-session memory constant. Draining is
    /// amortized: it only compacts once a meaningful prefix is dead.
    fn drain_consumed_prefix(&mut self) {
        const DRAIN_SLACK_FRAMES: usize = 96;
        const DRAIN_SLACK_SAMPLES: usize = 16 * 1024;
        let consumed = self
            .decode_state
            .as_ref()
            .map_or(0, XasrChunkedDecodeState::consumed_feature_frames);
        if consumed >= DRAIN_SLACK_FRAMES {
            self.features.data.drain(..consumed * self.features.n_mels);
            self.features.n_frames -= consumed;
            if let Some(state) = &mut self.decode_state {
                state.rebase_feature_frames(consumed);
            }
            self.dropped_frames += consumed;
        }
        let next_frame = self.dropped_frames + self.features.n_frames;
        let keep_from = earliest_sample_needed_for_frame(next_frame);
        if keep_from > self.dropped_samples {
            let dead = (keep_from - self.dropped_samples).min(self.audio.len());
            if dead >= DRAIN_SLACK_SAMPLES {
                self.audio.drain(..dead);
                self.dropped_samples += dead;
            }
        }
    }

    fn process_available_chunks(
        &mut self,
        final_flush: bool,
    ) -> Result<String, GgmlAsrExecutionError> {
        if self.audio.is_empty() {
            return Ok(String::new());
        }
        let total_samples = self.dropped_samples + self.audio.len();
        let target_total_frames = if final_flush {
            total_frame_count_for_samples(total_samples)
        } else {
            clean_frame_count_for_samples(total_samples)
        };
        if target_total_frames == 0 {
            return Ok(String::new());
        }
        self.extend_feature_rows(target_total_frames)?;
        let new_tokens = self.decode_available_chunks_on_actor(final_flush)?;
        self.drain_consumed_prefix();
        if new_tokens == 0 {
            return Ok(String::new());
        }
        self.text_delta()
    }

    fn text_delta(&mut self) -> Result<String, GgmlAsrExecutionError> {
        let emitted = self.decode_state()?.emitted_token_ids();
        let emitted_len = emitted.len();
        let token_ids = emitted[self.decoded_tokens..].to_vec();
        let stable_len = self.detokenizer.text().len();
        let mut detokenizer = std::mem::take(&mut self.detokenizer);
        let response = self.runtime.call_mut(move |runtime| {
            let result = token_ids
                .iter()
                .try_for_each(|&id| detokenizer.push_token(runtime.tokenizer(), id));
            (detokenizer, result)
        });
        let (detokenizer, result) = response.map_err(|error| self.failed(error.to_string()))?;
        self.detokenizer = detokenizer;
        result.map_err(|error| self.failed(error))?;
        self.decoded_tokens = emitted_len;
        Ok(self.detokenizer.text()[stable_len..].to_string())
    }

    fn rebase_decode_baseline(&mut self) {
        let decoded_tokens = self.decoded_tokens;
        let dropped = self
            .decode_state_mut()
            .expect("live decoder must retain decode state")
            .rebase_decoded_emitted_history(
                decoded_tokens,
                XASR_STREAMING_BASELINE_LEFT_CONTEXT_TOKENS,
            );
        self.decoded_tokens -= dropped;
        self.detokenizer.rebase_preserving_boundary_context();
        debug_assert_eq!(
            self.decoded_tokens,
            self.decode_state
                .as_ref()
                .map_or(0, XasrChunkedDecodeState::emitted_history_len)
        );
    }

    /// Adaptive early-exit predicate for the final-flush pad loop in
    /// [`IncrementalAudioDecoder::finish`]: true once the model has settled on
    /// the end of the utterance. The `#[cfg(test)]` escape hatch forces the loop
    /// to run every pad hop so a golden test can byte-compare "early exit"
    /// against "pad the full 0.8 s" on the same decoder.
    fn finish_endpoint_reached(&self, new_tokens_this_hop: usize) -> bool {
        #[cfg(test)]
        if FORCE_FULL_FLUSH.with(std::cell::Cell::get) {
            return false;
        }
        xasr_finish_endpoint_reached(new_tokens_this_hop, self.detokenizer.text())
    }
}

/// The tail text already ends a sentence: its last visible character is one of
/// the terminal punctuation marks the streaming zipformer emits only after
/// seeing trailing silence. Matches the batch golden's punctuation set in this
/// file.
fn xasr_tail_is_sentence_terminal(tail_text: &str) -> bool {
    tail_text
        .trim_end()
        .ends_with(['.', '?', '!', '\u{3002}', '\u{ff1f}', '\u{ff01}'])
}

/// Whether the final-flush pad loop can safely stop after this hop. BOTH
/// conditions are required: the tail must already carry sentence-terminal
/// punctuation (the sole reason the flush pads with silence at all) AND this
/// hop must have produced no new non-blank token (the padded frames
/// greedy-decoded to all blanks, i.e. the model answered the silence with
/// blanks). Only then are the remaining pad hops -- more of the same trailing
/// silence -- safe to skip. Requiring terminal punctuation keeps early exit
/// from ever dropping it; requiring a zero-emission hop keeps it from cutting
/// off a word the model is still emitting.
fn xasr_finish_endpoint_reached(new_tokens_this_hop: usize, tail_text: &str) -> bool {
    new_tokens_this_hop == 0 && xasr_tail_is_sentence_terminal(tail_text)
}

#[cfg(test)]
thread_local! {
    /// Test-only override: when set, [`XasrIncrementalDecoder::finish_endpoint_reached`]
    /// never fires, so `finish` pads the full 0.8 s and decodes every hop --
    /// the byte-for-byte baseline the early-exit golden compares against.
    static FORCE_FULL_FLUSH: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

impl IncrementalAudioDecoder for XasrIncrementalDecoder {
    fn accept_samples(&mut self, samples: &[f32]) -> Result<String, GgmlAsrExecutionError> {
        let _execution_scope =
            crate::models::native_execution_services::install_native_execution_services(
                self.request.execution_services.as_ref(),
            );
        if samples.iter().any(|value| !value.is_finite()) {
            return Err(self.failed("xasr streaming requires finite audio samples"));
        }
        self.audio.extend_from_slice(samples);
        let _thread_override = install_request_inference_threads_override(
            self.request.request_options.inference_threads,
        );
        self.process_available_chunks(false)
    }

    fn finish(&mut self) -> Result<String, GgmlAsrExecutionError> {
        let _execution_scope =
            crate::models::native_execution_services::install_native_execution_services(
                self.request.execution_services.as_ref(),
            );
        let _thread_override = install_request_inference_threads_override(
            self.request.request_options.inference_threads,
        );
        if self.audio.is_empty() {
            return self.process_available_chunks(true);
        }
        // Final flush: append the tail padding so the model sees the trailing
        // silence it needs to emit the last sentence's terminal punctuation
        // (mirrors the batch path in the prepared runtime). Appending the
        // whole 0.8 s up front -- rather than growing it hop by hop -- keeps
        // every fbank row byte-identical to a full-pad flush, so the adaptive
        // early exit below can only ever skip trailing hops, never change the
        // rows the retained hops decode. The session driver guarantees finish()
        // runs at most once.
        self.audio.extend(std::iter::repeat_n(
            0.0f32,
            XASR_FINAL_FLUSH_TAIL_PAD_SAMPLES,
        ));
        let total_samples = self.dropped_samples + self.audio.len();
        let target_total_frames = total_frame_count_for_samples(total_samples);
        if target_total_frames == 0 {
            return Ok(String::new());
        }
        self.extend_feature_rows(target_total_frames)?;

        // Decode the flush hops one at a time and stop as soon as the model has
        // settled on the end of the utterance: the tail already carries terminal
        // punctuation AND this hop produced no new non-blank token (the padded
        // frames greedy-decoded to all blanks). The real audio's last word is
        // never at risk -- it lives in the hops before any padding and is always
        // decoded. The 0.8 s pad stays the hard upper bound: if the heuristic
        // never fires (e.g. a clause that ends without terminal punctuation)
        // every hop runs, exactly like the pad-all-then-flush path, so this can
        // only skip work, never change a retained hop's output.
        let mut delta = String::new();
        loop {
            let outcome = self.decode_next_chunk_on_actor(true)?;
            if !outcome.processed {
                break;
            }
            delta.push_str(&self.text_delta()?);
            if self.finish_endpoint_reached(outcome.new_tokens) {
                break;
            }
        }
        self.drain_consumed_prefix();
        Ok(delta)
    }

    fn reset(&mut self) {
        self.audio.clear();
        self.dropped_samples = 0;
        self.features.data.clear();
        self.features.n_frames = 0;
        self.dropped_frames = 0;
        self.detokenizer.reset();
        self.decoded_tokens = 0;
        self.decode_state = self
            .runtime
            .call_mut(|runtime| runtime.new_decode_state())
            .ok();
    }

    fn rebase_after_soft_split(&mut self) -> Result<(), GgmlAsrExecutionError> {
        self.rebase_decode_baseline();
        Ok(())
    }

    /// Runs one real encoder chunk over silence so the lazily built GGML
    /// runner/weight-arena residency (`encoder_graph_runner_init`, ~300ms on
    /// CPU/Metal alike) lands here instead of on the first real audio a user
    /// speaks. Feeds exactly the first-chunk threshold
    /// (`first_chunk_input_frames`, 61 clean fbank frames = 9880 samples for
    /// the shipped decode_chunk_len=48 pack) through the same
    /// `accept_samples` -> `process_available_chunks` path real audio takes,
    /// so the warmed shape (frames/dim/valid_left_context) exactly matches
    /// what the real first chunk will request -- `full_encoder_reuse` then
    /// hits its cached session instead of rebuilding it too.
    ///
    /// `self.reset()` afterwards is the exact same reset `reset_utterance`
    /// uses in production (VAD segment restarts): it clears every field this
    /// warm-up touched (audio/features/detokenizer/decoded_tokens) and
    /// rebuilds `decode_state` via `runtime.new_decode_state()`, so the
    /// silence never leaks into the accumulated text, cache, or timestamps of
    /// the session's first real utterance. It deliberately does NOT touch
    /// `self.runtime`'s lazily initialized GGML runners/weight arenas --
    /// those are process/runtime-lifetime residency, not per-utterance state,
    /// and staying warm across the reset is the entire point.
    fn warm_up(&mut self) -> Result<(), GgmlAsrExecutionError> {
        let target_frames = self
            .runtime
            .call_mut(|runtime| runtime.first_chunk_input_frames())
            .map_err(|error| self.failed(error.to_string()))?
            .map_err(|error| self.failed(error))?;
        let silence = vec![0.0f32; samples_needed_for_clean_frame_count(target_frames)];
        self.accept_samples(&silence)?;
        self.reset();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ggml_runtime::{
        build_runtime_tensor_reader_from_preflight, load_runtime_source_metadata_and_tensor_index,
    };
    use crate::models::xasr_zipformer::executor::transcribe_xasr_zipformer_pcm;

    #[test]
    fn endpoint_requires_both_terminal_punctuation_and_a_silent_hop() {
        // Fires: terminal punctuation is present and the hop added nothing.
        assert!(xasr_finish_endpoint_reached(0, "hello world."));
        assert!(xasr_finish_endpoint_reached(0, "hello world.  "));
        assert!(xasr_finish_endpoint_reached(0, "\u{4f60}\u{597d}\u{3002}"));
        // Negative: the hop was silent but the tail is not sentence-final, so
        // the model may still owe the closing punctuation -- keep padding.
        assert!(!xasr_finish_endpoint_reached(0, "hello world"));
        assert!(!xasr_finish_endpoint_reached(0, "hello,"));
        assert!(!xasr_finish_endpoint_reached(0, ""));
        // Negative: punctuation is there but the hop still emitted a token, so
        // the model has not settled -- never cut off an in-flight emission.
        assert!(!xasr_finish_endpoint_reached(1, "hello world."));
        assert!(!xasr_finish_endpoint_reached(3, "hello world."));
    }

    #[test]
    fn tail_terminal_check_matches_the_batch_golden_punctuation_set() {
        for terminal in ['.', '?', '!', '\u{3002}', '\u{ff1f}', '\u{ff01}'] {
            assert!(
                xasr_tail_is_sentence_terminal(&format!("done{terminal}")),
                "'{terminal}' must count as sentence-terminal"
            );
            // Trailing whitespace must not hide the terminal mark.
            assert!(xasr_tail_is_sentence_terminal(&format!(
                "done{terminal} \t"
            )));
        }
        for non_terminal in [',', ';', ':', '\u{ff0c}', '-'] {
            assert!(
                !xasr_tail_is_sentence_terminal(&format!("clause{non_terminal}")),
                "'{non_terminal}' must not count as sentence-terminal"
            );
        }
    }

    #[test]
    #[ignore = "host-local: requires the X-ASR q8_0 pack under tmp/xasr-test/out"]
    fn xasr_accelerated_request_engages_gpu_and_matches_cpu_text() {
        use crate::ggml_runtime::{RequestBackendPreference, ResolvedFamilyRuntimeInput};

        let pack = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tmp/xasr-test/out/xasr-zh-en-onnx-q8_0.oasr");
        if !pack.exists() {
            eprintln!("skipping: xasr q8_0 pack absent at {}", pack.display());
            return;
        }
        let wav = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/jfk.wav")
            .canonicalize()
            .expect("sample wav fixture path must exist");
        let samples = crate::api::audio_io::load_wav_16khz_mono_f32_v0(
            wav,
            "xasr accelerated parity test",
            "xasr accelerated parity test",
        )
        .expect("sample wav should load");
        let preflight =
            load_runtime_source_metadata_and_tensor_index(&pack).expect("runtime preflight");
        let reader = build_runtime_tensor_reader_from_preflight(&preflight).expect("reader");
        let metadata = preflight.metadata.as_ref();

        // The encoder gate keys off the explicit request preference: CpuOnly
        // must resolve a CPU config, Accelerated must keep the GPU-class
        // backend.
        let policy = crate::arch::family_auto_gpu_policy_for_model_architecture(
            crate::XASR_ZIPFORMER_GGML_ARCHITECTURE_ID,
        );
        let (cpu_text, cpu_elapsed) = {
            let resolved = ResolvedFamilyRuntimeInput::resolve(
                Some(RequestBackendPreference::CpuOnly),
                policy,
            )
            .backend();
            assert_eq!(
                super::super::graph_config::xasr_zipformer_encoder_graph_config(resolved).backend,
                crate::ggml_runtime::GgmlCpuGraphBackend::Cpu
            );
            let started = std::time::Instant::now();
            let text =
                transcribe_xasr_zipformer_pcm(&reader, metadata, &samples, None, false, resolved)
                    .expect("cpu xasr")
                    .text;
            (text, started.elapsed())
        };

        let (gpu_text, gpu_elapsed) = {
            let resolved = ResolvedFamilyRuntimeInput::resolve(
                Some(RequestBackendPreference::Accelerated),
                policy,
            )
            .backend();
            assert!(
                resolved.is_gpu_class(),
                "accelerated request must keep the GPU-class backend, got {resolved:?}"
            );
            let started = std::time::Instant::now();
            let text =
                transcribe_xasr_zipformer_pcm(&reader, metadata, &samples, None, false, resolved)
                    .expect("gpu xasr")
                    .text;
            (text, started.elapsed())
        };

        eprintln!(
            "xasr accelerated parity: cpu={cpu_elapsed:?} gpu={gpu_elapsed:?} text={cpu_text:?}"
        );
        assert!(!cpu_text.trim().is_empty());
        assert_eq!(cpu_text, gpu_text, "GPU and CPU transcripts must match");
    }

    #[test]
    #[ignore = "host-local: requires the X-ASR q8_0 pack under tmp/xasr-test/out"]
    fn xasr_incremental_streaming_matches_batch_on_real_speech() {
        let pack = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tmp/xasr-test/out/xasr-zh-en-onnx-q8_0.oasr");
        if !pack.exists() {
            eprintln!("skipping: xasr q8_0 pack absent at {}", pack.display());
            return;
        }
        let wav = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/jfk.wav")
            .canonicalize()
            .expect("sample wav fixture path must exist");
        let samples = crate::api::audio_io::load_wav_16khz_mono_f32_v0(
            wav,
            "xasr streaming parity test",
            "xasr streaming parity test",
        )
        .expect("sample wav should load");
        let preflight =
            load_runtime_source_metadata_and_tensor_index(&pack).expect("runtime preflight");
        let reader = build_runtime_tensor_reader_from_preflight(&preflight).expect("reader");
        let metadata = preflight.metadata.as_ref();
        let resolved_runtime = crate::ggml_runtime::ResolvedFamilyRuntimeInput::resolve(
            Some(crate::ggml_runtime::RequestBackendPreference::CpuOnly),
            crate::arch::family_auto_gpu_policy_for_model_architecture(
                crate::XASR_ZIPFORMER_GGML_ARCHITECTURE_ID,
            ),
        );
        let batch = transcribe_xasr_zipformer_pcm(
            &reader,
            metadata,
            &samples,
            None,
            false,
            resolved_runtime.backend(),
        )
        .expect("batch xasr")
        .text;
        let request = GgmlAsrStreamingSessionRequest {
            execution_services:
                crate::models::native_execution_services::test_native_execution_services(),
            decoder_state: crate::models::ggml_asr_executor::GgmlAsrDecoderState::NoPersistentState,
            verified_pack: crate::models::runtime_preflight::verified_pack_from_preflight_for_test(
                preflight.clone(),
                crate::XASR_ZIPFORMER_GGML_ARCHITECTURE_ID,
            ),
            selected_family: crate::arch::builtin_adapter_descriptor(
                crate::arch::XASR_ZIPFORMER_GGML_ARCHITECTURE_ID,
            ),
            request_options: crate::GgmlAsrExecutionOptions::default(),
            configured_diarize: false,
            backend_preference: crate::GgmlAsrBackendPreference::CpuOnly,
            resolved_runtime,
            final_text_processor: None,
            session_context: crate::NativeAsrSessionContext::new("rt_xasr_streaming_match"),
            session_config: crate::NativeAsrStreamingSessionConfig::new()
                .with_partial_results(true)
                .into(),
        };
        let runtime_pool = super::super::runtime::new_runtime_actor_pool();
        let mut decoder = XasrIncrementalDecoder::new(
            &request,
            crate::arch::XASR_ZIPFORMER_STREAMING_EXECUTOR_COMPONENT_ID,
            crate::XASR_ZIPFORMER_GGML_ADAPTER_ID,
            super::super::runtime::checkout_prepared_runtime(
                &runtime_pool,
                &preflight,
                resolved_runtime.backend(),
            )
            .expect("streaming runtime"),
        )
        .expect("streaming decoder");
        let mut streaming = String::new();
        for chunk in samples.chunks(320) {
            streaming.push_str(&decoder.accept_samples(chunk).expect("stream chunk"));
        }
        streaming.push_str(&decoder.finish().expect("stream finish"));
        eprintln!("xasr real-speech streaming==batch text={streaming:?}");
        assert!(
            !batch.trim().is_empty(),
            "batch transcript must be non-empty for a meaningful parity check"
        );
        assert_eq!(streaming, batch);
        // Punctuation fidelity: the final-flush tail padding gives the model
        // the trailing silence it needs to emit the terminal punctuation of
        // the last sentence. Without the padding this clip decodes without
        // its closing period.
        assert!(
            batch
                .trim_end()
                .ends_with(['.', '?', '!', '\u{3002}', '\u{ff1f}', '\u{ff01}']),
            "batch transcript must keep the model's terminal punctuation: {batch:?}"
        );
        // Prefix draining must have kept the session buffers bounded: the
        // 5.5s sample is ~88k samples / ~555 feature rows, of which only a
        // small working tail may remain resident.
        assert!(
            decoder.dropped_samples > 0 && decoder.audio.len() < 40_000,
            "audio prefix was not drained: dropped={} resident={}",
            decoder.dropped_samples,
            decoder.audio.len()
        );
        assert!(
            decoder.dropped_frames > 0 && decoder.features.n_frames < 256,
            "feature prefix was not drained: dropped={} resident={}",
            decoder.dropped_frames,
            decoder.features.n_frames
        );
    }

    #[test]
    #[ignore = "host-local: requires the X-ASR q8_0 pack under tmp/xasr-test/out"]
    fn early_exit_finish_matches_full_pad_finish_byte_for_byte() {
        let pack = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tmp/xasr-test/out/xasr-zh-en-onnx-q8_0.oasr");
        if !pack.exists() {
            eprintln!("skipping: xasr q8_0 pack absent at {}", pack.display());
            return;
        }
        let preflight =
            load_runtime_source_metadata_and_tensor_index(&pack).expect("runtime preflight");
        let resolved_runtime = crate::ggml_runtime::ResolvedFamilyRuntimeInput::resolve(
            Some(crate::ggml_runtime::RequestBackendPreference::CpuOnly),
            crate::arch::family_auto_gpu_policy_for_model_architecture(
                crate::XASR_ZIPFORMER_GGML_ARCHITECTURE_ID,
            ),
        );
        let mut request = xasr_streaming_request();
        request.verified_pack =
            crate::models::runtime_preflight::verified_pack_from_preflight_for_test(
                preflight.clone(),
                crate::XASR_ZIPFORMER_GGML_ARCHITECTURE_ID,
            );
        request.resolved_runtime = resolved_runtime;
        let runtime_pool = super::super::runtime::new_runtime_actor_pool();

        // Streams `wav` in small chunks then flushes, returning the final-flush
        // delta. With `force_full` the endpoint heuristic is disabled, so
        // finish pads the full 0.8 s and decodes every hop -- the baseline the
        // early-exit path must reproduce byte-for-byte.
        let finish_delta = |wav: &str, force_full: bool| -> String {
            let samples = crate::api::audio_io::load_wav_16khz_mono_f32_v0(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../fixtures")
                    .join(wav)
                    .canonicalize()
                    .expect("fixture wav path must exist"),
                "xasr early-exit parity test",
                "xasr early-exit parity test",
            )
            .expect("sample wav should load");
            let runtime = super::super::runtime::checkout_prepared_runtime(
                &runtime_pool,
                &preflight,
                resolved_runtime.backend(),
            )
            .expect("streaming runtime");
            let mut decoder = XasrIncrementalDecoder::new(
                &request,
                crate::arch::XASR_ZIPFORMER_STREAMING_EXECUTOR_COMPONENT_ID,
                crate::XASR_ZIPFORMER_GGML_ADAPTER_ID,
                runtime,
            )
            .expect("streaming decoder");
            for chunk in samples.chunks(320) {
                decoder.accept_samples(chunk).expect("stream chunk");
            }
            FORCE_FULL_FLUSH.with(|f| f.set(force_full));
            let delta = decoder.finish().expect("stream finish");
            FORCE_FULL_FLUSH.with(|f| f.set(false));
            delta
        };

        // en, zh-en mixed, and a short zh clip: representative of the punctuation
        // the adaptive early exit keys off. Any divergence means the heuristic
        // dropped or added a token relative to padding the full 0.8 s.
        for wav in ["jfk.wav", "en_zh_mixed.wav", "zh_sample.wav"] {
            let early = finish_delta(wav, false);
            let full = finish_delta(wav, true);
            eprintln!("xasr early-exit vs full-pad {wav}: early_finish={early:?}");
            assert_eq!(
                early, full,
                "early-exit finish must byte-match full-pad finish for {wav}"
            );
        }
    }

    fn xasr_streaming_request() -> GgmlAsrStreamingSessionRequest {
        let runtime_source_preflight =
            crate::models::runtime_preflight::leaked_tiny_runtime_source_preflight();
        GgmlAsrStreamingSessionRequest {
            execution_services:
                crate::models::native_execution_services::test_native_execution_services(),
            decoder_state: crate::models::ggml_asr_executor::GgmlAsrDecoderState::NoPersistentState,
            verified_pack: crate::models::runtime_preflight::verified_pack_from_preflight_for_test(
                runtime_source_preflight,
                crate::XASR_ZIPFORMER_GGML_ARCHITECTURE_ID,
            ),
            selected_family: crate::arch::builtin_adapter_descriptor(
                crate::arch::XASR_ZIPFORMER_GGML_ARCHITECTURE_ID,
            ),
            request_options: crate::GgmlAsrExecutionOptions::default(),
            configured_diarize: false,
            backend_preference: crate::GgmlAsrBackendPreference::CpuOnly,
            resolved_runtime: crate::ggml_runtime::ResolvedFamilyRuntimeInput::resolve(
                (crate::GgmlAsrBackendPreference::CpuOnly).request_backend_override(),
                crate::ggml_runtime::AutoGpuPolicy::AllBackends,
            ),
            final_text_processor: None,
            session_context: crate::NativeAsrSessionContext::new("rt_xasr_streaming_warmup"),
            session_config: crate::NativeAsrStreamingSessionConfig::new()
                .with_partial_results(true)
                .into(),
        }
    }

    #[test]
    #[ignore = "host-local: requires the X-ASR q8_0 pack under tmp/xasr-test/out"]
    fn warm_up_initializes_the_encoder_runner_and_resets_decoder_state() {
        let pack = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tmp/xasr-test/out/xasr-zh-en-onnx-q8_0.oasr");
        if !pack.exists() {
            eprintln!("skipping: xasr q8_0 pack absent at {}", pack.display());
            return;
        }
        let preflight =
            load_runtime_source_metadata_and_tensor_index(&pack).expect("runtime preflight");
        let mut request = xasr_streaming_request();
        request.verified_pack =
            crate::models::runtime_preflight::verified_pack_from_preflight_for_test(
                preflight.clone(),
                crate::XASR_ZIPFORMER_GGML_ARCHITECTURE_ID,
            );
        let runtime_pool = super::super::runtime::new_runtime_actor_pool();
        let runtime = super::super::runtime::checkout_prepared_runtime(
            &runtime_pool,
            &preflight,
            request.resolved_runtime.backend(),
        )
        .expect("streaming runtime");
        let mut decoder = XasrIncrementalDecoder::new(
            &request,
            crate::arch::XASR_ZIPFORMER_STREAMING_EXECUTOR_COMPONENT_ID,
            crate::XASR_ZIPFORMER_GGML_ADAPTER_ID,
            runtime,
        )
        .expect("streaming decoder");

        assert!(
            !decoder
                .runtime
                .call_mut(|runtime| runtime.encoder_runner_is_initialized())
                .expect("inspect cold runner"),
            "runner must be cold before warm_up"
        );
        let started = std::time::Instant::now();
        decoder
            .warm_up()
            .expect("warm up should decode a real chunk");
        let warm_up_elapsed = started.elapsed();
        eprintln!("xasr streaming warm_up elapsed={warm_up_elapsed:?}");

        // The expensive lazy runner/weight-arena init must have already
        // happened -- the first real accept_samples call therefore cannot
        // pay it again.
        assert!(
            decoder
                .runtime
                .call_mut(|runtime| runtime.encoder_runner_is_initialized())
                .expect("inspect warm runner"),
            "warm_up must force the encoder_graph_runner_init lazy init"
        );
        // Warm-up's silence must not leak: every field `reset` clears must be
        // back to exactly its fresh-decoder value.
        assert!(decoder.audio.is_empty(), "audio buffer must be empty");
        assert_eq!(decoder.dropped_samples, 0);
        assert_eq!(decoder.features.n_frames, 0, "feature cache must be empty");
        assert_eq!(decoder.dropped_frames, 0);
        assert_eq!(decoder.decoded_tokens, 0);
        assert!(
            decoder.detokenizer.text().is_empty(),
            "detokenizer state must be empty"
        );

        // A second warm_up must be a cheap no-op relative to the first (the
        // runner stays resident): generous bound just guards against a
        // regression that silently re-pays the init.
        let second_started = std::time::Instant::now();
        decoder
            .warm_up()
            .expect("second warm up should also succeed");
        let second_elapsed = second_started.elapsed();
        eprintln!("xasr streaming second warm_up elapsed={second_elapsed:?}");
        assert!(
            second_elapsed < warm_up_elapsed,
            "second warm_up ({second_elapsed:?}) should be faster than the cold first \
             one ({warm_up_elapsed:?}) now that the runner is resident"
        );
    }

    #[test]
    #[ignore = "host-local: requires the X-ASR q8_0 pack under tmp/xasr-test/out"]
    fn warm_up_does_not_change_subsequent_transcription_of_real_speech() {
        let pack = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tmp/xasr-test/out/xasr-zh-en-onnx-q8_0.oasr");
        if !pack.exists() {
            eprintln!("skipping: xasr q8_0 pack absent at {}", pack.display());
            return;
        }
        let wav = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/jfk.wav")
            .canonicalize()
            .expect("sample wav fixture path must exist");
        let samples = crate::api::audio_io::load_wav_16khz_mono_f32_v0(
            wav,
            "xasr streaming warm up parity test",
            "xasr streaming warm up parity test",
        )
        .expect("sample wav should load");

        let preflight =
            load_runtime_source_metadata_and_tensor_index(&pack).expect("runtime preflight");
        let mut request = xasr_streaming_request();
        request.verified_pack =
            crate::models::runtime_preflight::verified_pack_from_preflight_for_test(
                preflight.clone(),
                crate::XASR_ZIPFORMER_GGML_ARCHITECTURE_ID,
            );
        let runtime_pool = super::super::runtime::new_runtime_actor_pool();

        let transcribe = |warm_up: bool| -> String {
            let runtime = super::super::runtime::checkout_prepared_runtime(
                &runtime_pool,
                &preflight,
                request.resolved_runtime.backend(),
            )
            .expect("streaming runtime");
            let mut decoder = XasrIncrementalDecoder::new(
                &request,
                crate::arch::XASR_ZIPFORMER_STREAMING_EXECUTOR_COMPONENT_ID,
                crate::XASR_ZIPFORMER_GGML_ADAPTER_ID,
                runtime,
            )
            .expect("streaming decoder");
            if warm_up {
                decoder.warm_up().expect("warm up before real audio");
            }
            let mut text = String::new();
            for chunk in samples.chunks(320) {
                text.push_str(&decoder.accept_samples(chunk).expect("stream chunk"));
            }
            text.push_str(&decoder.finish().expect("stream finish"));
            text
        };

        let without_warm_up = transcribe(false);
        let with_warm_up = transcribe(true);

        assert!(!without_warm_up.trim().is_empty());
        // Golden: warm-up's silence must be fully invisible to the very next
        // utterance -- byte-for-byte, not just "close enough".
        assert_eq!(
            with_warm_up, without_warm_up,
            "warm_up must not change the transcript of the real audio that follows it"
        );
    }
}
