use std::collections::HashMap;
use std::ops::{Deref, DerefMut};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use crate::ggml_runtime::{GgmlCpuGraphBackend, GgufMetadata, GgufTensorDataReader};
use crate::models::thread_local_runtime_cache::canonical_runtime_cache_path;

use super::decoder::XasrDecoder;
use super::encoder_graph::{
    XasrEncoderChunkState, XasrEncoderFeatureInput, XasrZipformerEncoderGraph,
};
use super::encoder_weights::load_xasr_encoder_weights;
use super::frontend::{XasrFbankFeatures, XasrFbankFrontend};
use super::graph_config::xasr_zipformer_encoder_graph_config;
use super::greedy::{
    DEFAULT_MAX_SYMBOLS_PER_FRAME, XasrGreedyDecodeResult, greedy_decode_frames_incremental,
};
use super::joiner::XasrJoiner;
use super::runtime_contract::{
    XasrZipformerExecutionMetadata, parse_xasr_zipformer_execution_metadata,
};
use super::tokenizer::XasrZipformerTokenizer;
use super::weights::{load_xasr_decoder_weights, load_xasr_joiner_weights};

const XASR_ZIPFORMER_STREAMING_WARMUP_FRAMES: usize = 13;
const XASR_PROFILE_ENV: &str = "OPENASR_XASR_PROFILE";
const MAX_IDLE_RUNTIMES_PER_KEY: usize = 2;

/// Pool key: pack path + the backend the runtime's prepared encoder graph was
/// built for. CPU and Metal runtimes must never conflate — a checkout for an
/// accelerated session must not receive a CPU-frozen runtime (or vice versa).
type RuntimePoolKey = (PathBuf, GgmlCpuGraphBackend);
type RuntimePool = HashMap<RuntimePoolKey, Vec<SendableRuntime>>;

static XASR_PROCESS_RUNTIME_POOL: OnceLock<Mutex<RuntimePool>> = OnceLock::new();

#[derive(Debug)]
pub(super) struct XasrZipformerPreparedRuntime {
    metadata: XasrZipformerExecutionMetadata,
    tokenizer: XasrZipformerTokenizer,
    encoder: XasrZipformerEncoderGraph,
    decoder: XasrDecoder,
    joiner: XasrJoiner,
}

#[derive(Debug)]
pub(super) struct XasrChunkedDecodeState {
    feature_cursor: usize,
    first_chunk: bool,
    encoder_state: Option<XasrEncoderChunkState>,
    context: Vec<u32>,
    emitted: Vec<u32>,
    /// Absolute encoder frame of each emission, parallel to `emitted`.
    emitted_frames: Vec<usize>,
    /// Joiner softmax probability of each emission, parallel to `emitted`.
    emitted_probabilities: Vec<f32>,
    encoder_frames: usize,
    chunk_index: usize,
}

impl XasrChunkedDecodeState {
    fn new(context: Vec<u32>) -> Self {
        Self {
            feature_cursor: 0,
            first_chunk: true,
            encoder_state: None,
            context,
            emitted: Vec::new(),
            emitted_frames: Vec::new(),
            emitted_probabilities: Vec::new(),
            encoder_frames: 0,
            chunk_index: 0,
        }
    }

    pub(super) fn reset_for_runtime(&mut self, runtime: &XasrZipformerPreparedRuntime) {
        *self = runtime.new_decode_state();
    }

    pub(super) fn emitted_token_ids(&self) -> &[u32] {
        &self.emitted
    }

    pub(super) fn emitted_history_len(&self) -> usize {
        self.emitted.len()
    }

    /// Drops already-returned emission history while retaining a token-level
    /// left-context suffix. The caller supplies how many leading entries are
    /// stable/decoded; entries after that point are never dropped.
    pub(super) fn rebase_decoded_emitted_history(
        &mut self,
        decoded_tokens: usize,
        retain_tokens: usize,
    ) -> usize {
        let stable_tokens = decoded_tokens.min(self.emitted.len());
        let retained_stable_tokens = stable_tokens.min(retain_tokens);
        let drop_tokens = stable_tokens - retained_stable_tokens;
        if drop_tokens == 0 {
            return 0;
        }
        self.emitted.drain(..drop_tokens);
        self.emitted_frames.drain(..drop_tokens);
        self.emitted_probabilities.drain(..drop_tokens);
        debug_assert_eq!(self.emitted.len(), self.emitted_frames.len());
        debug_assert_eq!(self.emitted.len(), self.emitted_probabilities.len());
        drop_tokens
    }

    /// Feature frames the chunk loop has fully consumed (it never re-reads
    /// rows before the cursor), i.e. how many leading rows the caller may
    /// drain from its feature cache.
    pub(super) fn consumed_feature_frames(&self) -> usize {
        self.feature_cursor
    }

    /// Shifts the cursor after the caller drained `dropped_frames` leading
    /// rows from the feature cache the cursor indexes into.
    pub(super) fn rebase_feature_frames(&mut self, dropped_frames: usize) {
        debug_assert!(dropped_frames <= self.feature_cursor);
        self.feature_cursor = self.feature_cursor.saturating_sub(dropped_frames);
    }
}

#[derive(Debug)]
pub(super) struct SendableRuntime(XasrZipformerPreparedRuntime);

// SAFETY: the prepared runtime owns CPU-only GGML graph handles and is moved as
// an exclusive value. Streaming sessions only access it through `&mut`, while
// per-session encoder cache state lives in `XasrChunkedDecodeState`, not inside
// the pooled runtime.
unsafe impl Send for SendableRuntime {}

pub(super) struct PooledRuntime {
    key: RuntimePoolKey,
    runtime: Option<SendableRuntime>,
}

impl Deref for PooledRuntime {
    type Target = XasrZipformerPreparedRuntime;

    fn deref(&self) -> &Self::Target {
        &self
            .runtime
            .as_ref()
            .expect("pooled xasr runtime must be present")
            .0
    }
}

impl DerefMut for PooledRuntime {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self
            .runtime
            .as_mut()
            .expect("pooled xasr runtime must be present")
            .0
    }
}

impl Drop for PooledRuntime {
    fn drop(&mut self) {
        let Some(runtime) = self.runtime.take() else {
            return;
        };
        let Ok(mut pool) = runtime_pool().lock() else {
            return;
        };
        let idle = pool.entry(self.key.clone()).or_default();
        if idle.len() < MAX_IDLE_RUNTIMES_PER_KEY {
            idle.push(runtime);
        }
    }
}

pub(super) fn checkout_prepared_runtime(pack_path: &Path) -> Result<PooledRuntime, String> {
    let backend = xasr_zipformer_encoder_graph_config().backend;
    let key = (canonical_runtime_cache_path(pack_path), backend);
    if let Some(runtime) = runtime_pool()
        .lock()
        .map_err(|_| "xasr runtime pool lock poisoned".to_string())?
        .get_mut(&key)
        .and_then(Vec::pop)
    {
        return Ok(PooledRuntime {
            key,
            runtime: Some(runtime),
        });
    }

    let runtime = XasrZipformerPreparedRuntime::load(pack_path)?;
    Ok(PooledRuntime {
        key,
        runtime: Some(SendableRuntime(runtime)),
    })
}

fn runtime_pool() -> &'static Mutex<RuntimePool> {
    XASR_PROCESS_RUNTIME_POOL.get_or_init(|| Mutex::new(HashMap::new()))
}

impl XasrZipformerPreparedRuntime {
    pub(super) fn load(pack_path: &Path) -> Result<Self, String> {
        let profile = xasr_profile_start();
        let reader = GgufTensorDataReader::from_path(pack_path).map_err(|e| e.to_string())?;
        let gguf_metadata =
            crate::ggml_runtime::read_gguf_metadata(pack_path).map_err(|e| e.to_string())?;
        let runtime = Self::from_reader_metadata(&reader, &gguf_metadata)?;
        xasr_profile_log(
            "runtime_load",
            profile,
            format_args!("pack={}", pack_path.display()),
        );
        Ok(runtime)
    }

    pub(super) fn from_reader_metadata(
        reader: &GgufTensorDataReader,
        gguf_metadata: &GgufMetadata,
    ) -> Result<Self, String> {
        let metadata =
            parse_xasr_zipformer_execution_metadata(gguf_metadata).map_err(|e| e.to_string())?;
        let tokenizer = XasrZipformerTokenizer::from_metadata(gguf_metadata, metadata.blank_id)?;
        let encoder_weights =
            load_xasr_encoder_weights(reader, &metadata).map_err(|e| e.to_string())?;
        let decoder_weights =
            load_xasr_decoder_weights(reader, &metadata).map_err(|e| e.to_string())?;
        let joiner_weights =
            load_xasr_joiner_weights(reader, &metadata).map_err(|e| e.to_string())?;
        let encoder = XasrZipformerEncoderGraph::new_ggml_cpu_full_encoder(
            metadata.clone(),
            encoder_weights,
            xasr_zipformer_encoder_graph_config(),
        )
        .map_err(|e| e.to_string())?;
        Ok(Self {
            decoder: XasrDecoder::new(
                decoder_weights,
                metadata.decoder_context_size,
                metadata.blank_id,
            ),
            joiner: XasrJoiner::new(joiner_weights),
            metadata,
            tokenizer,
            encoder,
        })
    }

    pub(super) fn transcribe(&mut self, samples: &[f32]) -> Result<XasrGreedyDecodeResult, String> {
        let total_profile = xasr_profile_start();
        let fbank_profile = xasr_profile_start();
        let frontend = XasrFbankFrontend::new();
        let features = frontend
            .features_from_samples(samples)
            .map_err(|e| e.to_string())?;
        xasr_profile_log(
            "fbank",
            fbank_profile,
            format_args!("samples={} frames={}", samples.len(), features.n_frames),
        );

        let mut state = self.new_decode_state();
        self.decode_available_chunks(&mut state, &features, true)?;
        let text = self.decode_text(state.emitted_token_ids())?;
        xasr_profile_log(
            "decode_total",
            total_profile,
            format_args!(
                "chunks={} encoder_frames={}",
                state.chunk_index, state.encoder_frames
            ),
        );
        Ok(XasrGreedyDecodeResult {
            token_ids: state.emitted,
            emit_frames: state.emitted_frames,
            emit_probabilities: state.emitted_probabilities,
            encoder_frames: state.encoder_frames,
            text,
        })
    }

    pub(super) fn new_decode_state(&self) -> XasrChunkedDecodeState {
        XasrChunkedDecodeState::new(self.decoder.initial_context())
    }

    pub(super) fn decode_available_chunks(
        &mut self,
        state: &mut XasrChunkedDecodeState,
        features: &XasrFbankFeatures,
        final_flush: bool,
    ) -> Result<usize, String> {
        let chunk_hop = self.metadata.decode_chunk_len;
        let chunk_input_frames = chunk_hop
            .checked_add(XASR_ZIPFORMER_STREAMING_WARMUP_FRAMES)
            .ok_or_else(|| "xasr chunk frame count overflows".to_string())?;
        let mut new_tokens = 0usize;
        let mut greedy_elapsed = Duration::ZERO;
        let mut processed_chunks = 0usize;

        loop {
            if state.feature_cursor >= features.n_frames {
                break;
            }
            let remaining = features.n_frames - state.feature_cursor;
            if !state.first_chunk && remaining <= XASR_ZIPFORMER_STREAMING_WARMUP_FRAMES {
                break;
            }
            if !final_flush {
                let end_frame = state
                    .feature_cursor
                    .checked_add(chunk_input_frames)
                    .ok_or_else(|| "xasr chunk end frame overflows".to_string())?;
                if end_frame > features.n_frames {
                    break;
                }
            }

            let real_chunk_frames = if final_flush {
                remaining.min(chunk_input_frames)
            } else {
                chunk_input_frames
            };
            let input = XasrEncoderFeatureInput::new(
                chunk_input_frames,
                features.n_mels,
                feature_chunk_rows(
                    features,
                    state.feature_cursor,
                    real_chunk_frames,
                    chunk_input_frames,
                )?,
            )
            .map_err(|e| e.to_string())?;
            let chunk_profile = xasr_profile_start();
            let chunk = self
                .encoder
                .encode_streaming_chunk_from_features(&input, state.encoder_state.as_ref())
                .map_err(|e| e.to_string())?;
            xasr_profile_log(
                "encoder_chunk",
                chunk_profile,
                format_args!(
                    "chunk={} cursor={} real_frames={} padded_frames={} output_frames={}",
                    state.chunk_index,
                    state.feature_cursor,
                    real_chunk_frames,
                    chunk_input_frames,
                    chunk.output.frames
                ),
            );

            // The chunk's emissions index encoder frames from the offset the
            // stream had before this chunk's output was appended.
            let chunk_frame_offset = state.encoder_frames;
            state.encoder_frames = state
                .encoder_frames
                .checked_add(chunk.output.frames)
                .ok_or_else(|| "xasr encoder frame count overflows".to_string())?;
            let greedy_profile = xasr_profile_start();
            let emitted = greedy_decode_frames_incremental(
                &chunk.output.rows,
                chunk.output.frames,
                self.metadata.encoder_output_dim(),
                &self.decoder,
                &self.joiner,
                self.metadata.blank_id,
                DEFAULT_MAX_SYMBOLS_PER_FRAME,
                &mut state.context,
                &mut state.emitted,
                &mut state.emitted_frames,
                &mut state.emitted_probabilities,
                chunk_frame_offset,
            )?;
            if let Some(started_at) = greedy_profile {
                greedy_elapsed += started_at.elapsed();
            }
            new_tokens = new_tokens
                .checked_add(emitted)
                .ok_or_else(|| "xasr emitted token count overflows".to_string())?;
            state.encoder_state = Some(chunk.state);
            let advance = chunk_hop.min(remaining);
            state.feature_cursor = state
                .feature_cursor
                .checked_add(advance)
                .ok_or_else(|| "xasr chunk cursor overflows".to_string())?;
            state.first_chunk = false;
            state.chunk_index = state
                .chunk_index
                .checked_add(1)
                .ok_or_else(|| "xasr chunk index overflows".to_string())?;
            processed_chunks = processed_chunks
                .checked_add(1)
                .ok_or_else(|| "xasr processed chunk count overflows".to_string())?;
        }

        if processed_chunks > 0 {
            xasr_profile_log_duration(
                "greedy",
                greedy_elapsed,
                format_args!("chunks={processed_chunks} new_tokens={new_tokens}"),
            );
        }
        Ok(new_tokens)
    }

    pub(super) fn decode_text(&self, token_ids: &[u32]) -> Result<String, String> {
        self.tokenizer.decode(token_ids)
    }

    pub(super) fn tokenizer(&self) -> &XasrZipformerTokenizer {
        &self.tokenizer
    }
}

fn feature_chunk_rows(
    features: &XasrFbankFeatures,
    start_frame: usize,
    real_frames: usize,
    padded_frames: usize,
) -> Result<Vec<f32>, String> {
    if features.n_mels == 0 {
        return Err("xasr feature dimension must be non-zero".to_string());
    }
    if real_frames == 0 || real_frames > padded_frames {
        return Err(format!(
            "xasr invalid chunk shape real_frames={real_frames}, padded_frames={padded_frames}"
        ));
    }
    let expected = features
        .n_frames
        .checked_mul(features.n_mels)
        .ok_or_else(|| "xasr feature shape overflows".to_string())?;
    if features.data.len() != expected {
        return Err(format!(
            "xasr feature data has {} values, expected {expected}",
            features.data.len()
        ));
    }
    let end_frame = start_frame
        .checked_add(real_frames)
        .ok_or_else(|| "xasr chunk end frame overflows".to_string())?;
    if end_frame > features.n_frames {
        return Err(format!(
            "xasr chunk end frame {end_frame} exceeds feature frames {}",
            features.n_frames
        ));
    }

    let mut rows = Vec::with_capacity(
        padded_frames
            .checked_mul(features.n_mels)
            .ok_or_else(|| "xasr padded feature chunk shape overflows".to_string())?,
    );
    for frame_offset in 0..padded_frames {
        let source_frame = if frame_offset < real_frames {
            start_frame + frame_offset
        } else {
            end_frame - 1
        };
        let start = source_frame
            .checked_mul(features.n_mels)
            .ok_or_else(|| "xasr chunk source start overflows".to_string())?;
        let end = start
            .checked_add(features.n_mels)
            .ok_or_else(|| "xasr chunk source end overflows".to_string())?;
        rows.extend_from_slice(&features.data[start..end]);
    }
    Ok(rows)
}

fn xasr_profile_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_truthy(XASR_PROFILE_ENV))
}

fn env_truthy(name: &str) -> bool {
    std::env::var_os(name)
        .and_then(|value| value.into_string().ok())
        .is_some_and(|value| {
            let value = value.trim();
            !value.is_empty()
                && !value.eq_ignore_ascii_case("0")
                && !value.eq_ignore_ascii_case("false")
        })
}

fn xasr_profile_start() -> Option<Instant> {
    xasr_profile_enabled().then(Instant::now)
}

fn xasr_profile_log(stage: &str, started_at: Option<Instant>, detail: std::fmt::Arguments<'_>) {
    if let Some(started_at) = started_at {
        xasr_profile_log_duration(stage, started_at.elapsed(), detail);
    }
}

fn xasr_profile_log_duration(stage: &str, elapsed: Duration, detail: std::fmt::Arguments<'_>) {
    if xasr_profile_enabled() {
        eprintln!(
            "openasr_xasr_profile stage={stage} elapsed_ms={:.3} {detail}",
            elapsed.as_secs_f64() * 1000.0
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feature_chunk_rows_pads_tail_with_last_frame() {
        let features = XasrFbankFeatures {
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            n_frames: 3,
            n_mels: 2,
        };

        let rows = feature_chunk_rows(&features, 1, 2, 4).expect("chunk rows");

        assert_eq!(rows, vec![3.0, 4.0, 5.0, 6.0, 5.0, 6.0, 5.0, 6.0]);
    }

    #[test]
    fn decoded_emitted_history_rebase_stays_bounded_across_many_soft_splits() {
        const CAP: usize = 8;
        let mut state = XasrChunkedDecodeState::new(vec![0, 0]);

        for split in 0..100usize {
            for offset in 0..23usize {
                let token = 1 + ((split + offset) % 7) as u32;
                state.emitted.push(token);
                state.emitted_frames.push(split * 100 + offset);
                state.emitted_probabilities.push(0.9);
            }
            let mut decoded_tokens = state.emitted_history_len();

            let dropped = state.rebase_decoded_emitted_history(decoded_tokens, CAP);
            decoded_tokens -= dropped;

            assert!(
                state.emitted_history_len() <= CAP,
                "split {split} kept {} tokens above cap {CAP}",
                state.emitted_history_len()
            );
            assert_eq!(decoded_tokens, state.emitted_history_len());
            assert_eq!(state.emitted_frames.len(), state.emitted_history_len());
            assert_eq!(
                state.emitted_probabilities.len(),
                state.emitted_history_len()
            );
        }
    }
}
