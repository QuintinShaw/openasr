//! Granite Speech incremental KV-cache decode session.
//!
//! The one-shot `decoder_graph::prefill_logits*` path recomputes the *entire*
//! token prefix from scratch on every decode step (see `decode_executor`'s
//! historical doc): step N re-runs a full 40-layer forward over
//! `prompt ++ generated[..N]`, so a full transcription is `O(n^2)` in decoded
//! length -- for the 2B Granite dense decoder that is ~430x realtime, which
//! makes full-length WER / peer-gate runs infeasible.
//!
//! This session removes that quadratic by giving Granite the same incremental
//! KV cache every other autoregressive family here already has (qwen's
//! `Qwen3AsrLayerKvCacheState`, firered-llm, cohere, ...): the prompt is
//! prefilled once, each layer's post-RoPE K/V is persisted, and every
//! subsequent step computes Q/K/V for **only the new token**, appends its K/V
//! to the cache, and attends the single new query against the full cached
//! history. Per-step compute drops from `O(prefix)` to `O(1)` projection/MLP
//! plus an `O(prefix)` attention dot-product -- total decode `O(n)` (plus a
//! small `O(n^2)` host-side K/V copy that is orders of magnitude cheaper than
//! the projections/MLP it replaces).
//!
//! Bit-exactness (the hard requirement): every op here is byte-for-byte the one
//! the one-shot recompute runs. Prefill and decode share
//! `decoder_graph::granite_pre_attention` / `granite_post_attention` verbatim,
//! so a cached K/V equals the K/V a full recompute would produce at that
//! position (a causal decoder's position-`j` representation is independent of
//! any later token, and this runs CPU-only, where ggml `mul_mat` computes each
//! output element via a fixed-order `vec_dot` regardless of batch width). The
//! cache stores **f32** (never f16) so no rounding is introduced. The only
//! attention difference is prefill's additive causal mask (masked keys underflow
//! to exactly `0.0` in `soft_max_ext`) versus decode attending a history that
//! simply omits those never-contributing keys -- the surviving softmax terms,
//! their max, and their sum are identical, so the last-position logits match to
//! the bit. This is proven in-repo by
//! `granite_incremental_decode_matches_full_recompute_bit_exact`.
//!
//! Weights are held for the session's whole lifetime -- either uploaded once
//! into a persistent f32 `GraniteDecoderWeightArena` (the `new` path, used by
//! the synthetic bit-exact test and any host-`HashMap` provider) or bound
//! zero-copy, keep-quantized, from the mmap'd `.oasr` pack via
//! `GraniteDecoderLoadedWeights` (the `new_keep_quantized` path the runtime
//! executor uses, so a 2B decoder stays ~its packed size resident instead of a
//! ~8 GB f32 dequant + upload). Only the tiny per-step inputs (one embedding,
//! one position, the K/V history views) live in the reset-per-step graph
//! context.
//!
//! ## Metal reuse path (device-resident KV + build-once decode graph)
//!
//! The description above is the CPU / scheduler-on path: it keeps the K/V in
//! host `Vec<f32>` buffers and rebuilds the 40-layer step graph every token,
//! re-uploading the whole history each step. On the single-backend GPU path
//! (`reusable_decode_graph_supported_for_runner`: gpu-class backend + scheduler
//! off) that host round-trip is the dominant decode cost, so this module also
//! provides the resident path taken there:
//!
//! - The K/V lives in a device-resident fixed-span arena (`resident_kv`,
//!   `[head_dim, resident_capacity, kv_heads]` f32 per layer). `prefill` seeds
//!   rows `0..n_tokens` in place from the prefill graph (`set_rows`), never
//!   copying K/V to the host; each `decode_step` writes only the one new row.
//! - A single persistent single-token decode graph is built once
//!   (`build_reusable_decode_graph`) and re-run per step -- no 40-layer rebuild.
//!   It reads the full fixed span under an externally uploaded additive `-inf`
//!   tail mask, so its shape is constant across steps.
//!
//! Both mirror the firered-aed / qwen resident-decode machinery while keeping
//! Granite's forked attention numerics. This path is NOT bit-identical to the
//! CPU reference (ggml's Metal reduction order differs from the CPU
//! `vec_dot` -- by design); the bit-exact gate
//! `granite_incremental_decode_matches_full_recompute_bit_exact` covers the CPU
//! path it does not touch, and end-to-end transcript equality validates the
//! Metal path.

#![allow(dead_code)]

use crate::ggml_runtime::{
    GgmlCpuGraphBackend, GgmlCpuGraphConfig, GgmlCpuGraphError, GgmlCpuGraphRunner, GgmlCpuTensor,
    GgmlKvElementType, GgmlLoadedWeightContext, GgmlPersistentGraphSession, GgmlRopeExtParams,
    GgmlRuntimeSource,
};
use crate::nn::decoder::{
    LlmKvCacheSpec, LlmResidentKvArena, allocate_zeroed_llm_resident_kv_arena,
    build_causal_mask_f16_bits, build_fixed_kv_attention_mask_bits, last_token_hidden_view,
    reusable_decode_graph_supported_for_runner,
};

use super::decoder_graph::{
    GraniteDecoderLoadedWeights, GraniteDecoderWeightArena, GraniteDecoderWeights,
    GraniteSpeechDecoderConfig, GraniteSpeechDecoderError, GraniteSpeechDecoderWeightProvider,
    embed_token_row, granite_post_attention, granite_pre_attention, linear, rms_norm,
    weight_in_major,
};

/// Headroom (in tokens) reserved above the prompt length when sizing the
/// resident KV arena's fixed span, so the whole greedy decode
/// (`GRANITE_SPEECH_MAX_GENERATED_TOKENS` new tokens after the prompt) fits
/// without ever overrunning the arena. Kept in sync with the executor's
/// `max_generated_tokens`; a larger value only reserves more (masked-until-used)
/// span, never changes results.
const GRANITE_RESIDENT_DECODE_HEADROOM: usize = 256;

/// f32 resident KV element type for the Metal reuse path. f32 (not f16) keeps
/// the cached K/V rounding-free, so the only numerical difference from the CPU
/// growing-KV reference is ggml's backend reduction order (by design; the
/// bit-exact gate covers the CPU path). Both host + resident are f32.
const GRANITE_RESIDENT_KV_SPEC: LlmKvCacheSpec = LlmKvCacheSpec {
    host: GgmlKvElementType::F32,
    resident: GgmlKvElementType::F32,
};

/// ggml metadata-context budget for both the session runner and the reused
/// persistent decode graph. Holds tensor structs + names, not compute buffers.
const GRANITE_DECODE_GRAPH_CONTEXT_BYTES: usize = 256 * 1024 * 1024;

fn decoder_graph_config(backend: GgmlCpuGraphBackend) -> GgmlCpuGraphConfig {
    GgmlCpuGraphConfig {
        context_bytes: GRANITE_DECODE_GRAPH_CONTEXT_BYTES,
        graph_size: 32768,
        n_threads: GgmlCpuGraphConfig::resolve_runtime_thread_count_for(
            backend,
            crate::ggml_runtime::GgmlCpuGraphThreadingWorkload::EncoderPrelude,
        ),
        backend,
        // Scheduler off on the single-backend GPU path so the in-place
        // resident-KV reuse graph is legal. CPU keeps the scheduler and the
        // growing-KV host decode path.
        use_scheduler: !backend.is_gpu_class(),
    }
}

const fn resident_flash_attention_enabled(backend: GgmlCpuGraphBackend) -> bool {
    // The resident path has a measured, transcript-equivalent win on Metal.
    // Keep the generic CUDA/HIP/Vulkan lane on its established naive path
    // until each backend's wide-prefill flash kernel has its own correctness
    // and performance evidence.
    matches!(backend, GgmlCpuGraphBackend::Metal)
}

/// Additive self-attention mask for the fixed-span reuse graph: `0.0` for every
/// key column `<= position` (the prompt + generated-so-far + the just-written
/// new token) and `f32::MIN` for the never-yet-written / future tail. Added
/// after the `attention_multiplier` scale in `soft_max_ext`, exactly as the
/// growing path's causal mask, so masked columns underflow to `0.0` and the
/// surviving softmax terms match the growing-KV attention over `position + 1`
/// keys.
fn fixed_span_tail_mask(max_positions: usize, position: usize) -> Vec<f32> {
    let mut mask = vec![0.0f32; max_positions];
    for slot in mask.iter_mut().skip(position + 1) {
        *slot = f32::MIN;
    }
    mask
}

fn map_ggml(stage: &'static str) -> impl Fn(GgmlCpuGraphError) -> GraniteSpeechDecoderError + Copy {
    move |source| GraniteSpeechDecoderError::Ggml { stage, source }
}

/// A forward pass' outputs: the (last-position for prefill / only-position for a
/// step) logits row, plus each layer's tapped `(K, V)` in `[head_dim, tokens,
/// kv_heads]` (kv-head-major) layout for the KV cache.
type ForwardGraphOutput = (Vec<f32>, Vec<(Vec<f32>, Vec<f32>)>);

/// A prefilled, persistent Granite decoder ready to emit one incremental
/// single-token step at a time. Construct with [`new`](Self::new), seed with
/// [`prefill`](Self::prefill), then call [`decode_step`](Self::decode_step) once
/// per generated token.
pub(crate) struct GraniteSpeechDecodeSession {
    /// Build-once/re-run single-token decode graph for the Metal reuse path
    /// (`None` until the first reused step builds it, and on CPU / scheduler-on
    /// runners where the growing-KV host path stays authoritative). Declared
    /// FIRST so it drops first: its persistent graph holds raw pointers into
    /// `runner`, `weights`/`_loaded`, and the `resident_kv` arena, all declared
    /// below and therefore dropped after it.
    reuse: Option<GraniteReusableDecodeGraph>,
    config: GraniteSpeechDecoderConfig,
    runner: GgmlCpuGraphRunner,
    weights: GraniteDecoderWeights,
    /// Kept alive so the keep-quantized `weights`' zero-copy handles (raw
    /// pointers into this context's mmap-backed backend buffer) stay valid for
    /// the session's lifetime. `None` on the f32-arena path (the arena owns its
    /// own storage inside `weights`). Declared after `weights` so `weights`
    /// drops first.
    ///
    /// The session holds NO borrow of the weight provider: `new`/`new_keep_quantized`
    /// consult it transiently (arena upload / not at all), and `decode_step`
    /// takes the provider as a call argument. That keeps the whole session
    /// owned (no lifetime), so a keep-quantized instance -- runner plus the
    /// mmap'd loaded context plus its zero-copy bound weights -- can live in
    /// the cross-request resident cache (`executor::GraniteSpeechPreparedRuntime`).
    _loaded: Option<GgmlLoadedWeightContext>,
    /// `k_history[layer][kv_head]` is that head's `[seq, head_dim]` (row-major)
    /// key rows, appended token by token; concatenating the `kv_heads` inner
    /// buffers yields the `[head_dim, seq, kv_heads]` (kv-head-major) layout the
    /// attention `mul_mat` consumes.
    k_history: Vec<Vec<Vec<f32>>>,
    v_history: Vec<Vec<Vec<f32>>>,
    seq_len: usize,
    prefilled: bool,
    /// Device-resident, per-layer fixed-span `[head_dim, resident_capacity,
    /// kv_heads]` f32 K/V arena for the Metal reuse path. Seeded once per
    /// request by `prefill` (rows `0..n_tokens` via `set_rows`) and extended one
    /// row per `decode_step`, so the K/V history never round-trips to the host.
    /// `None` on CPU / scheduler-on runners (which keep `k_history`/`v_history`).
    /// Declared after `reuse` so the reuse graph (which points into this arena)
    /// drops first.
    resident_kv: Option<LlmResidentKvArena>,
    /// Allocated column count (`max_positions`) of every `resident_kv` tensor.
    /// The fixed span the reuse graph attends over and bakes into its topology;
    /// grown (never shrunk) when a later request's `prompt + headroom` exceeds
    /// it. `0` before the first resident allocation.
    resident_capacity: usize,
}

/// Build-once/re-run persistent single-token Granite decode graph plus its
/// per-step runtime inputs. The op sequence is exactly one `n_tokens == 1` step
/// of the growing-KV path, except the new token's K/V is written into the
/// resident arena via `set_rows` on a runtime `row_index` input and
/// self-attention reads the full fixed `resident_capacity` span under an
/// externally uploaded additive `-inf` tail mask, so the graph shape stays
/// constant across steps. The owning [`GraniteSpeechDecodeSession`] declares
/// this object before its resident arena and runner, so `session` drops before
/// every allocation its graph tensors point into.
struct GraniteReusableDecodeGraph {
    session: GgmlPersistentGraphSession,
    /// The fixed self-KV span this persistent graph was built for; a mismatch
    /// against the session's current `resident_capacity` forces a rebuild.
    max_positions: usize,
    use_flash_attention: bool,
    embed: GgmlCpuTensor<'static>,
    row_index: GgmlCpuTensor<'static>,
    position: GgmlCpuTensor<'static>,
    mask: GgmlCpuTensor<'static>,
    logits: GgmlCpuTensor<'static>,
}

impl GraniteSpeechDecodeSession {
    /// Build the runner and upload every decoder weight once. No prefill yet.
    /// `provider` is consulted transiently here (to fill the f32 arena) and is
    /// not retained; `decode_step` takes it again per call for the token embed.
    pub(crate) fn new(
        config: GraniteSpeechDecoderConfig,
        provider: &dyn GraniteSpeechDecoderWeightProvider,
        backend: GgmlCpuGraphBackend,
    ) -> Result<Self, GraniteSpeechDecoderError> {
        let graph_config = decoder_graph_config(backend);
        let runner =
            GgmlCpuGraphRunner::new(graph_config).map_err(map_ggml("session_runner_init"))?;
        let weights = GraniteDecoderWeightArena::load(&runner, &config, provider)?;
        Ok(Self::assemble(
            config,
            runner,
            GraniteDecoderWeights::Arena(weights),
            None,
        ))
    }

    /// Keep-quantized session: bind every decoder weight zero-copy from `source`'s
    /// mmap'd `.oasr` pack (native q8_0/q4_k/f16/f32) instead of dequantizing the
    /// whole 2-B decoder to an f32 host copy + arena upload. The projection/norm/
    /// lm_head weights come from the pack; the token-embedding rows are supplied
    /// by the `provider` passed to `decode_step`. The loaded weight context is
    /// loaded context and this session's runner use the same thread-cached
    /// backend/device, and both are held for the session's whole lifetime.
    pub(crate) fn new_keep_quantized(
        config: GraniteSpeechDecoderConfig,
        source: &GgmlRuntimeSource,
        backend: GgmlCpuGraphBackend,
    ) -> Result<Self, GraniteSpeechDecoderError> {
        let graph_config = decoder_graph_config(backend);
        let runner =
            GgmlCpuGraphRunner::new(graph_config).map_err(map_ggml("session_runner_init"))?;
        let loaded = runner
            .load_gguf_weight_context(source)
            .map_err(map_ggml("session_load_gguf_weight_context"))?;
        let weights = GraniteDecoderLoadedWeights::load(&loaded, &config)?;
        Ok(Self::assemble(
            config,
            runner,
            GraniteDecoderWeights::Loaded(weights),
            Some(loaded),
        ))
    }

    fn assemble(
        config: GraniteSpeechDecoderConfig,
        runner: GgmlCpuGraphRunner,
        weights: GraniteDecoderWeights,
        loaded: Option<GgmlLoadedWeightContext>,
    ) -> Self {
        let num_layers = config.num_layers;
        Self {
            reuse: None,
            config,
            runner,
            weights,
            _loaded: loaded,
            k_history: vec![Vec::new(); num_layers],
            v_history: vec![Vec::new(); num_layers],
            seq_len: 0,
            prefilled: false,
            resident_kv: None,
            resident_capacity: 0,
        }
    }

    /// Whether this session's runner can run the in-place resident-KV reuse path
    /// (single-backend GPU, scheduler off). CPU / scheduler-on runners fall back
    /// to the growing-KV host path, which the bit-exact gate covers.
    fn reuse_supported(&self) -> bool {
        reusable_decode_graph_supported_for_runner(&self.runner)
    }

    /// Reset this decode's request-visible state before the prepared runtime
    /// re-enters the cross-request cache. On the CPU / scheduler path the grown
    /// host K/V vectors are released. On the GPU reuse path the fixed-capacity
    /// resident K/V arena and persistent graph deliberately remain allocated;
    /// the next prefill overwrites every newly visible prompt row, each decode
    /// step overwrites its own row, and the fixed-span mask keeps all remaining
    /// stale rows invisible. In both cases clearing `seq_len` and `prefilled`
    /// makes a new prefill mandatory before another step can execute.
    pub(crate) fn release_session_scoped_buffers(&mut self) {
        let num_layers = self.config.num_layers;
        self.k_history = vec![Vec::new(); num_layers];
        self.v_history = vec![Vec::new(); num_layers];
        self.seq_len = 0;
        self.prefilled = false;
    }

    pub(crate) fn is_prefilled(&self) -> bool {
        self.prefilled
    }

    /// Number of tokens (prompt + generated) whose K/V is currently cached.
    pub(crate) fn cached_positions(&self) -> usize {
        self.seq_len
    }

    /// Prefill the whole prompt once: run a single causal forward over
    /// `embeddings` (`[n_tokens, hidden_size]`, row-major, pre-`embedding_multiplier`),
    /// persist every layer's post-RoPE K/V, and return the logits row for the
    /// token immediately following the prompt (i.e. the first generated token's
    /// distribution). Same op sequence as `decoder_graph::prefill_logits_from_embeddings`.
    pub(crate) fn prefill(
        &mut self,
        embeddings: &[f32],
        n_tokens: usize,
    ) -> Result<Vec<f32>, GraniteSpeechDecoderError> {
        if self.prefilled {
            return Err(GraniteSpeechDecoderError::Shape {
                reason: "granite decode session already prefilled".to_string(),
            });
        }
        if n_tokens == 0 {
            return Err(GraniteSpeechDecoderError::Shape {
                reason: "prefill n_tokens must be non-zero".to_string(),
            });
        }
        if embeddings.len() != n_tokens * self.config.hidden_size {
            return Err(GraniteSpeechDecoderError::Shape {
                reason: format!(
                    "prefill embeddings has {} values, expected {n_tokens}x{}",
                    embeddings.len(),
                    self.config.hidden_size
                ),
            });
        }

        if self.reuse_supported() {
            // Metal reuse path: seed the resident KV arena directly from the
            // prefill graph (rows `0..n_tokens` via `set_rows`), no host copy.
            let required = n_tokens
                .checked_add(GRANITE_RESIDENT_DECODE_HEADROOM)
                .ok_or_else(|| GraniteSpeechDecoderError::Shape {
                    reason: "granite resident KV span overflow".to_string(),
                })?;
            self.ensure_resident_arena(required)?;
            let last_logits = run_prefill_graph_seeding_resident(
                &mut self.runner,
                &self.weights,
                &self.config,
                self.resident_kv
                    .as_ref()
                    .expect("resident arena allocated above"),
                embeddings,
                n_tokens,
            )?;
            self.seq_len = n_tokens;
            self.prefilled = true;
            return Ok(last_logits);
        }

        let (last_logits, per_layer_kv) = run_prefill_graph(
            &mut self.runner,
            &self.weights,
            &self.config,
            embeddings,
            n_tokens,
        )?;

        let head_dim = self.config.head_dim;
        let kv_heads = self.config.num_kv_heads;
        for (layer, (k_tap, v_tap)) in per_layer_kv.into_iter().enumerate() {
            // k_tap / v_tap are `[head_dim, n_tokens, kv_heads]` (kv-head-major):
            // split each kv_head's contiguous `[n_tokens, head_dim]` block into
            // its own history buffer.
            let mut k_heads = Vec::with_capacity(kv_heads);
            let mut v_heads = Vec::with_capacity(kv_heads);
            let block = n_tokens * head_dim;
            for h in 0..kv_heads {
                k_heads.push(k_tap[h * block..(h + 1) * block].to_vec());
                v_heads.push(v_tap[h * block..(h + 1) * block].to_vec());
            }
            self.k_history[layer] = k_heads;
            self.v_history[layer] = v_heads;
        }
        self.seq_len = n_tokens;
        self.prefilled = true;
        Ok(last_logits)
    }

    /// Run one incremental decode step for `new_token_id` (the position it
    /// occupies is the current cache length), append its K/V, and return the
    /// logits row for the NEXT token.
    pub(crate) fn decode_step(
        &mut self,
        new_token_id: u32,
        provider: &dyn GraniteSpeechDecoderWeightProvider,
    ) -> Result<Vec<f32>, GraniteSpeechDecoderError> {
        if !self.prefilled {
            return Err(GraniteSpeechDecoderError::Shape {
                reason: "granite decode session must be prefilled before decode_step".to_string(),
            });
        }

        if self.reuse_supported() {
            // Metal reuse path: one persistent single-token step against the
            // resident KV arena (write the new row via `set_rows`, attend the
            // fixed span). No graph rebuild, no host K/V round-trip.
            return self.decode_step_reused(new_token_id, provider);
        }

        let embed_row = embed_token_row(&self.config, provider, new_token_id)?.to_vec();

        let head_dim = self.config.head_dim;
        let kv_heads = self.config.num_kv_heads;
        let seq_len = self.seq_len;
        // Flatten each layer's per-kv-head history into one contiguous
        // `[head_dim, seq_len, kv_heads]` (kv-head-major) buffer for upload.
        let k_hist_bufs: Vec<Vec<f32>> = self
            .k_history
            .iter()
            .map(|layer| flatten_history(layer, seq_len, head_dim, kv_heads))
            .collect();
        let v_hist_bufs: Vec<Vec<f32>> = self
            .v_history
            .iter()
            .map(|layer| flatten_history(layer, seq_len, head_dim, kv_heads))
            .collect();

        let (logits, per_layer_kv) = run_decode_step_graph(
            &mut self.runner,
            &self.weights,
            &self.config,
            &embed_row,
            seq_len,
            &k_hist_bufs,
            &v_hist_bufs,
        )?;

        // Append this token's K/V (`[head_dim, 1, kv_heads]` = kv-head-major)
        // to each head's history buffer.
        for (layer, (k_new, v_new)) in per_layer_kv.into_iter().enumerate() {
            for h in 0..kv_heads {
                self.k_history[layer][h]
                    .extend_from_slice(&k_new[h * head_dim..(h + 1) * head_dim]);
                self.v_history[layer][h]
                    .extend_from_slice(&v_new[h * head_dim..(h + 1) * head_dim]);
            }
        }
        self.seq_len += 1;
        Ok(logits)
    }

    /// Ensure the resident KV arena has at least `required` columns of fixed
    /// span, (re)allocating (and zero-filling) it when absent or too small.
    /// Growing invalidates the reuse graph (whose baked-in tensors point into
    /// the old arena buffer), so drop it; the next reused step rebuilds against
    /// the new arena. Never shrinks -- a smaller later request reuses the wider
    /// span (its tail simply stays masked).
    fn ensure_resident_arena(&mut self, required: usize) -> Result<(), GraniteSpeechDecoderError> {
        if self.resident_kv.is_some() && self.resident_capacity >= required {
            return Ok(());
        }
        // Drop the reuse graph before freeing the old arena it points into.
        self.reuse = None;
        self.resident_kv = None;
        let num_layers = self.config.num_layers;
        // Two KV tensors per layer plus a small fixed spare for the arena's
        // metadata pool (tensor structs + names).
        let context_bytes = GgmlCpuGraphConfig::metadata_context_bytes(num_layers * 2 + 16);
        let arena = allocate_zeroed_llm_resident_kv_arena(
            &self.runner,
            context_bytes,
            num_layers,
            self.config.head_dim,
            required,
            self.config.num_kv_heads,
            1,
            "granite_resident_kv",
            GRANITE_RESIDENT_KV_SPEC,
        )
        .map_err(map_ggml("resident_kv_arena_alloc"))?;
        self.resident_kv = Some(arena);
        self.resident_capacity = required;
        Ok(())
    }

    /// One incremental single-token step on the Metal reuse path: build the
    /// persistent graph on first use (or after a capacity change / poison),
    /// refresh the per-step token / row-index / position / tail-mask inputs, and
    /// re-run -- no graph construction, no reallocation, and the new K/V is
    /// written in place into the resident arena (`set_rows`) instead of round-
    /// tripping to the host. The new token occupies row `self.seq_len`.
    fn decode_step_reused(
        &mut self,
        new_token_id: u32,
        provider: &dyn GraniteSpeechDecoderWeightProvider,
    ) -> Result<Vec<f32>, GraniteSpeechDecoderError> {
        let position = self.seq_len;
        let max_positions = self.resident_capacity;
        if position >= max_positions {
            return Err(GraniteSpeechDecoderError::Shape {
                reason: format!(
                    "granite decode position {position} exceeds resident KV span {max_positions}"
                ),
            });
        }
        let embed_row = embed_token_row(&self.config, provider, new_token_id)?.to_vec();

        let needs_build = self
            .reuse
            .as_ref()
            .map(|reuse| reuse.session.is_poisoned() || reuse.max_positions != max_positions)
            .unwrap_or(true);
        if needs_build {
            self.build_reusable_decode_graph()?;
        }

        let vocab_size = self.config.vocab_size;
        let position_i32 =
            i32::try_from(position).map_err(|_| GraniteSpeechDecoderError::Shape {
                reason: format!("granite decode position {position} does not fit i32"),
            })?;
        let use_flash_attention = self
            .reuse
            .as_ref()
            .expect("granite reusable decode graph built above")
            .use_flash_attention;
        let mask_values =
            (!use_flash_attention).then(|| fixed_span_tail_mask(max_positions, position));
        let mask_bits = if use_flash_attention {
            Some(
                build_fixed_kv_attention_mask_bits(max_positions, position + 1)
                    .map_err(map_ggml("reuse_build_flash_mask"))?,
            )
        } else {
            None
        };

        let reuse = self
            .reuse
            .as_mut()
            .expect("granite reusable decode graph built above");
        let embed = reuse.embed;
        let row_index = reuse.row_index;
        let position_tensor = reuse.position;
        let mask = reuse.mask;
        let logits = reuse.logits;
        let graph = reuse.session.builder();

        graph
            .set_f32_slice(embed, &embed_row, "granite_reuse_embed")
            .map_err(map_ggml("reuse_upload_embed"))?;
        graph
            .set_i32_slice(row_index, &[position_i32], "granite_reuse_row")
            .map_err(map_ggml("reuse_upload_row"))?;
        graph
            .set_i32_slice(position_tensor, &[position_i32], "granite_reuse_position")
            .map_err(map_ggml("reuse_upload_position"))?;
        if let Some(mask_bits) = mask_bits {
            graph
                .set_f16_bits_slice(mask, &mask_bits, "granite_reuse_mask")
                .map_err(map_ggml("reuse_upload_flash_mask"))?;
        } else {
            graph
                .set_f32_slice(
                    mask,
                    mask_values
                        .as_deref()
                        .expect("naive Granite reuse mask values"),
                    "granite_reuse_mask",
                )
                .map_err(map_ggml("reuse_upload_mask"))?;
        }
        let output = graph
            .compute_output_f32(logits, vocab_size)
            .map_err(map_ggml("reuse_compute"))?;
        self.seq_len = position + 1;
        Ok(output)
    }

    /// Build the persistent single-token decode graph against the current
    /// resident arena and `resident_capacity` span. Mirrors firered-aed's
    /// `build_reusable_decode_graph`, but keeps Granite's forked
    /// `granite_pre_attention` / `granite_post_attention` numerics (all four
    /// Granite scalars: `attention_multiplier`, `residual_multiplier`,
    /// `embedding_multiplier`, `logits_scaling`). The new token's K/V is written
    /// into the arena via `set_rows(arena, k/v, row_index)`; self-attention then
    /// reads the whole fixed span, with the per-step additive `-inf` tail mask
    /// zeroing every not-yet-written (or masked-future) column so the result is
    /// numerically the growing-KV attention over exactly `position + 1` keys.
    fn build_reusable_decode_graph(&mut self) -> Result<(), GraniteSpeechDecoderError> {
        let config = self.config;
        let head_dim = config.head_dim;
        let hidden_size = config.hidden_size;
        let max_positions = self.resident_capacity;
        let use_flash_attention = resident_flash_attention_enabled(self.runner.backend_kind());
        let resident_kv = self
            .resident_kv
            .as_ref()
            .expect("resident arena present before building reuse graph");
        // Fixed RoPE params: `qwen_neox` uses `ext_factor = 0`, so `n_ctx_orig`
        // never enters the rotation -- baking `max_positions` here is bit-for-bit
        // identical to the growing path's per-step `seq_len + 1`.
        let rope = GgmlRopeExtParams::qwen_neox(head_dim, max_positions, config.rope_theta)
            .map_err(map_ggml("reuse_rope_params"))?;

        // Snapshot the resident arena's graph tensors as `'static` for the
        // persistent session (the arena outlives the reuse graph by field order).
        let kv_tensors: Vec<(GgmlCpuTensor<'static>, GgmlCpuTensor<'static>)> =
            resident_kv.graph_tensors();

        // Same metadata-context budget the session runner itself uses (see
        // `GRANITE_DECODE_GRAPH_CONTEXT_BYTES`): ample for the single-token
        // graph's ~20-op-per-layer tensor structs.
        let mut session = self
            .runner
            .start_persistent_graph_session(GRANITE_DECODE_GRAPH_CONTEXT_BYTES)
            .map_err(map_ggml("reuse_session_start"))?;
        let graph = session.builder();

        let embed = graph
            .new_tensor_2d_f32(hidden_size, 1, "granite_reuse_embed")
            .map_err(map_ggml("reuse_embed_alloc"))?;
        let row_index = graph
            .new_tensor_1d_i32(1, "granite_reuse_row")
            .map_err(map_ggml("reuse_row_alloc"))?;
        let position = graph
            .new_tensor_1d_i32(1, "granite_reuse_position")
            .map_err(map_ggml("reuse_position_alloc"))?;
        let mask = if use_flash_attention {
            graph
                .new_tensor_2d_f16(max_positions, 1, "granite_reuse_mask")
                .map_err(map_ggml("reuse_flash_mask_alloc"))?
        } else {
            graph
                .new_tensor_2d_f32(max_positions, 1, "granite_reuse_mask")
                .map_err(map_ggml("reuse_mask_alloc"))?
        };
        graph
            .set_input(embed)
            .map_err(map_ggml("reuse_embed_input"))?;
        graph
            .set_input(row_index)
            .map_err(map_ggml("reuse_row_input"))?;
        graph
            .set_input(position)
            .map_err(map_ggml("reuse_position_input"))?;
        graph
            .set_input(mask)
            .map_err(map_ggml("reuse_mask_input"))?;

        let mut hidden = graph
            .scale(embed, config.embedding_multiplier)
            .map_err(map_ggml("reuse_embed_scale"))?;

        for (index, (arena_k, arena_v)) in kv_tensors.iter().copied().enumerate() {
            let layer_weights = self.weights.layer_weights(index);
            let pre =
                granite_pre_attention(graph, hidden, position, &layer_weights, &config, 1, rope)?;
            // Write this token's K/V into row `row_index` of the resident arena;
            // the returned handles are the full `[head_dim, max_positions,
            // kv_heads]` span (with the new row now live) that attention reads.
            let k_full = graph
                .set_rows(arena_k, pre.k_perm, row_index)
                .map_err(map_ggml("reuse_k_set_rows"))?;
            let v_full = graph
                .set_rows(arena_v, pre.v_perm, row_index)
                .map_err(map_ggml("reuse_v_set_rows"))?;
            let attended = if use_flash_attention {
                graph
                    .flash_attn_ext(
                        pre.q_perm,
                        k_full,
                        v_full,
                        Some(mask),
                        config.attention_multiplier,
                        0.0,
                        0.0,
                    )
                    .map_err(map_ggml("reuse_flash_attn"))?
            } else {
                let scores = graph
                    .mul_mat(k_full, pre.q_perm)
                    .map_err(map_ggml("reuse_scores"))?;
                let probs = graph
                    .soft_max_ext(scores, Some(mask), config.attention_multiplier, 0.0)
                    .map_err(map_ggml("reuse_softmax"))?;
                let v_t = graph
                    .cont(
                        graph
                            .transpose(v_full)
                            .map_err(map_ggml("reuse_v_transpose"))?,
                    )
                    .map_err(map_ggml("reuse_v_cont"))?;
                graph
                    .mul_mat(v_t, probs)
                    .map_err(map_ggml("reuse_attended"))?
            };
            hidden = granite_post_attention(
                graph,
                hidden,
                attended,
                &layer_weights,
                &config,
                1,
                use_flash_attention,
            )?;
        }

        let hidden_out = rms_norm(
            graph,
            hidden,
            config.rms_norm_eps,
            self.weights.final_norm_weight(),
        )?;
        let lm_head_w = weight_in_major(
            graph,
            self.weights.lm_head_weight(),
            config.hidden_size,
            config.vocab_size,
            "reuse_lm_head_reshape",
        )?;
        let logits_raw = linear(graph, lm_head_w, hidden_out, "reuse_lm_head")?;
        let logits = graph
            .scale(logits_raw, 1.0 / config.logits_scaling)
            .map_err(map_ggml("reuse_logits_scale"))?;
        graph
            .set_output(logits)
            .map_err(map_ggml("reuse_set_output"))?;
        graph
            .prepare_outputs_for_upload(&[logits])
            .map_err(map_ggml("reuse_prepare_outputs"))?;

        self.reuse = Some(GraniteReusableDecodeGraph {
            session,
            max_positions,
            use_flash_attention,
            embed,
            row_index,
            position,
            mask,
            logits,
        });
        Ok(())
    }
}

/// Concatenate a layer's `kv_heads` history buffers (each `[seq_len, head_dim]`,
/// row-major) into one `[head_dim, seq_len, kv_heads]` kv-head-major buffer.
fn flatten_history(
    layer: &[Vec<f32>],
    seq_len: usize,
    head_dim: usize,
    kv_heads: usize,
) -> Vec<f32> {
    let mut out = Vec::with_capacity(seq_len * head_dim * kv_heads);
    for head in layer.iter().take(kv_heads) {
        out.extend_from_slice(head);
    }
    out
}

/// One-shot causal prefill that ALSO taps every layer's post-RoPE K/V. Returns
/// the last-position logits row plus, per layer, the `[head_dim, n_tokens,
/// kv_heads]` K and V tensors.
fn run_prefill_graph(
    runner: &mut GgmlCpuGraphRunner,
    weights: &GraniteDecoderWeights,
    config: &GraniteSpeechDecoderConfig,
    embeddings: &[f32],
    n_tokens: usize,
) -> Result<ForwardGraphOutput, GraniteSpeechDecoderError> {
    let head_dim = config.head_dim;
    let kv_heads = config.num_kv_heads;
    let hidden_size = config.hidden_size;
    let vocab_size = config.vocab_size;
    let kv_width = kv_heads * head_dim;

    let mut graph = runner.start_graph();

    let embed_tensor = graph
        .new_tensor_2d_f32(hidden_size, n_tokens, "granite_session_prefill_embeds")
        .map_err(map_ggml("session_prefill_input_alloc"))?;
    let positions = graph
        .new_tensor_1d_i32(n_tokens, "granite_session_prefill_positions")
        .map_err(map_ggml("session_prefill_positions_alloc"))?;
    let mask = graph
        .new_tensor_2d_f32(n_tokens, n_tokens, "granite_session_prefill_mask")
        .map_err(map_ggml("session_prefill_mask_alloc"))?;

    let mut hidden = graph
        .scale(embed_tensor, config.embedding_multiplier)
        .map_err(map_ggml("session_prefill_embed_scale"))?;
    let rope = GgmlRopeExtParams::qwen_neox(head_dim, n_tokens, config.rope_theta)
        .map_err(map_ggml("session_prefill_rope_params"))?;

    let mut kv_taps = Vec::with_capacity(config.num_layers);
    for index in 0..config.num_layers {
        let layer_weights = weights.layer_weights(index);
        let pre = granite_pre_attention(
            &mut graph,
            hidden,
            positions,
            &layer_weights,
            config,
            n_tokens,
            rope,
        )?;
        kv_taps.push((pre.k_perm, pre.v_perm));

        let scores = graph
            .mul_mat(pre.k_perm, pre.q_perm)
            .map_err(map_ggml("session_prefill_scores"))?;
        let probs = graph
            .soft_max_ext(scores, Some(mask), config.attention_multiplier, 0.0)
            .map_err(map_ggml("session_prefill_softmax"))?;
        let v_t = graph
            .cont(
                graph
                    .transpose(pre.v_perm)
                    .map_err(map_ggml("session_prefill_v_transpose"))?,
            )
            .map_err(map_ggml("session_prefill_v_cont"))?;
        let attended = graph
            .mul_mat(v_t, probs)
            .map_err(map_ggml("session_prefill_attended"))?;
        hidden = granite_post_attention(
            &mut graph,
            hidden,
            attended,
            &layer_weights,
            config,
            n_tokens,
            false,
        )?;
    }

    let hidden_out = rms_norm(
        &graph,
        hidden,
        config.rms_norm_eps,
        weights.final_norm_weight(),
    )?;
    let lm_head_w = weight_in_major(
        &graph,
        weights.lm_head_weight(),
        config.hidden_size,
        config.vocab_size,
        "lm_head_reshape",
    )?;
    let logits_input = last_token_hidden_view(&graph, hidden_out, hidden_size, n_tokens)
        .map_err(map_ggml("session_prefill_last_hidden"))?;
    let logits_raw = linear(&graph, lm_head_w, logits_input, "lm_head")?;
    let logits = graph
        .scale(logits_raw, 1.0 / config.logits_scaling)
        .map_err(map_ggml("session_prefill_logits_scale"))?;

    graph
        .set_output(logits)
        .map_err(map_ggml("session_prefill_set_output_logits"))?;
    for (k_tap, v_tap) in &kv_taps {
        graph
            .set_output(*k_tap)
            .map_err(map_ggml("session_prefill_set_output_k"))?;
        graph
            .set_output(*v_tap)
            .map_err(map_ggml("session_prefill_set_output_v"))?;
    }
    graph
        .set_input(embed_tensor)
        .map_err(map_ggml("session_prefill_mark_input_embeds"))?;
    graph
        .set_input(positions)
        .map_err(map_ggml("session_prefill_mark_input_positions"))?;
    graph
        .set_input(mask)
        .map_err(map_ggml("session_prefill_mark_input_mask"))?;

    let mut outputs: Vec<_> = Vec::with_capacity(1 + kv_taps.len() * 2);
    outputs.push(logits);
    for (k_tap, v_tap) in &kv_taps {
        outputs.push(*k_tap);
        outputs.push(*v_tap);
    }
    graph
        .prepare_outputs_for_upload(&outputs)
        .map_err(map_ggml("session_prefill_prepare_outputs"))?;

    graph
        .set_f32_slice(embed_tensor, embeddings, "granite_session_prefill_embeds")
        .map_err(map_ggml("session_prefill_upload_embeds"))?;
    let position_ids: Vec<i32> = (0..n_tokens as i32).collect();
    graph
        .set_i32_slice(
            positions,
            &position_ids,
            "granite_session_prefill_positions",
        )
        .map_err(map_ggml("session_prefill_upload_positions"))?;
    let mask_values = super::decoder_graph::causal_mask(n_tokens);
    graph
        .set_f32_slice(mask, &mask_values, "granite_session_prefill_mask")
        .map_err(map_ggml("session_prefill_upload_mask"))?;

    let mut request: Vec<_> = Vec::with_capacity(1 + kv_taps.len() * 2);
    request.push((logits, vocab_size));
    for (k_tap, v_tap) in &kv_taps {
        request.push((*k_tap, n_tokens * kv_width));
        request.push((*v_tap, n_tokens * kv_width));
    }
    let results = graph
        .compute_outputs_f32(&request)
        .map_err(map_ggml("session_prefill_compute"))?;

    let mut iter = results.into_iter();
    let last_logits = iter.next().expect("prefill logits tap");

    let mut per_layer_kv = Vec::with_capacity(config.num_layers);
    for _ in 0..config.num_layers {
        let k = iter.next().expect("prefill k tap");
        let v = iter.next().expect("prefill v tap");
        per_layer_kv.push((k, v));
    }
    Ok((last_logits, per_layer_kv))
}

/// One-shot causal prefill that ALSO seeds the device-resident KV arena
/// (`set_rows` writing rows `0..n_tokens` of every layer's K/V in place), used
/// on the Metal reuse path. Same batched causal forward as `run_prefill_graph`
/// (byte-for-byte the same attention op sequence), but the per-layer K/V is
/// written straight into the resident arena instead of tapped back to the host,
/// so prefill also avoids the O(n) K/V readback. Returns only the last-position
/// logits row.
fn run_prefill_graph_seeding_resident(
    runner: &mut GgmlCpuGraphRunner,
    weights: &GraniteDecoderWeights,
    config: &GraniteSpeechDecoderConfig,
    resident_kv: &LlmResidentKvArena,
    embeddings: &[f32],
    n_tokens: usize,
) -> Result<Vec<f32>, GraniteSpeechDecoderError> {
    let head_dim = config.head_dim;
    let hidden_size = config.hidden_size;
    let vocab_size = config.vocab_size;
    let use_flash_attention = resident_flash_attention_enabled(runner.backend_kind());

    let mut graph = runner.start_graph();

    let embed_tensor = graph
        .new_tensor_2d_f32(hidden_size, n_tokens, "granite_seed_prefill_embeds")
        .map_err(map_ggml("seed_prefill_input_alloc"))?;
    let positions = graph
        .new_tensor_1d_i32(n_tokens, "granite_seed_prefill_positions")
        .map_err(map_ggml("seed_prefill_positions_alloc"))?;
    let mask = if use_flash_attention {
        graph
            .new_tensor_2d_f16(n_tokens, n_tokens, "granite_seed_prefill_mask")
            .map_err(map_ggml("seed_prefill_flash_mask_alloc"))?
    } else {
        graph
            .new_tensor_2d_f32(n_tokens, n_tokens, "granite_seed_prefill_mask")
            .map_err(map_ggml("seed_prefill_mask_alloc"))?
    };
    let seed_indices = graph
        .new_tensor_1d_i32(n_tokens, "granite_seed_prefill_rows")
        .map_err(map_ggml("seed_prefill_rows_alloc"))?;

    let kv_tensors = resident_kv.graph_tensors();

    let mut hidden = graph
        .scale(embed_tensor, config.embedding_multiplier)
        .map_err(map_ggml("seed_prefill_embed_scale"))?;
    let rope = GgmlRopeExtParams::qwen_neox(head_dim, n_tokens, config.rope_theta)
        .map_err(map_ggml("seed_prefill_rope_params"))?;

    for (index, (arena_k, arena_v)) in kv_tensors.iter().copied().enumerate() {
        let layer_weights = weights.layer_weights(index);
        let pre = granite_pre_attention(
            &mut graph,
            hidden,
            positions,
            &layer_weights,
            config,
            n_tokens,
            rope,
        )?;
        // Seed rows 0..n_tokens of the resident arena (side effects), then run
        // the ordinary batched causal attention over this prompt's own K/V.
        let k_seed = graph
            .set_rows(arena_k, pre.k_perm, seed_indices)
            .map_err(map_ggml("seed_prefill_k_set_rows"))?;
        graph
            .add_side_effect_root(k_seed)
            .map_err(map_ggml("seed_prefill_k_root"))?;
        let v_seed = graph
            .set_rows(arena_v, pre.v_perm, seed_indices)
            .map_err(map_ggml("seed_prefill_v_set_rows"))?;
        graph
            .add_side_effect_root(v_seed)
            .map_err(map_ggml("seed_prefill_v_root"))?;

        let attended = if use_flash_attention {
            graph
                .flash_attn_ext(
                    pre.q_perm,
                    pre.k_perm,
                    pre.v_perm,
                    Some(mask),
                    config.attention_multiplier,
                    0.0,
                    0.0,
                )
                .map_err(map_ggml("seed_prefill_flash_attn"))?
        } else {
            let scores = graph
                .mul_mat(pre.k_perm, pre.q_perm)
                .map_err(map_ggml("seed_prefill_scores"))?;
            let probs = graph
                .soft_max_ext(scores, Some(mask), config.attention_multiplier, 0.0)
                .map_err(map_ggml("seed_prefill_softmax"))?;
            let v_t = graph
                .cont(
                    graph
                        .transpose(pre.v_perm)
                        .map_err(map_ggml("seed_prefill_v_transpose"))?,
                )
                .map_err(map_ggml("seed_prefill_v_cont"))?;
            graph
                .mul_mat(v_t, probs)
                .map_err(map_ggml("seed_prefill_attended"))?
        };
        hidden = granite_post_attention(
            &mut graph,
            hidden,
            attended,
            &layer_weights,
            config,
            n_tokens,
            use_flash_attention,
        )?;
    }

    let hidden_out = rms_norm(
        &graph,
        hidden,
        config.rms_norm_eps,
        weights.final_norm_weight(),
    )?;
    let lm_head_w = weight_in_major(
        &graph,
        weights.lm_head_weight(),
        config.hidden_size,
        config.vocab_size,
        "seed_lm_head_reshape",
    )?;
    let logits_input = last_token_hidden_view(&graph, hidden_out, hidden_size, n_tokens)
        .map_err(map_ggml("seed_prefill_last_hidden"))?;
    let logits_raw = linear(&graph, lm_head_w, logits_input, "seed_lm_head")?;
    let logits = graph
        .scale(logits_raw, 1.0 / config.logits_scaling)
        .map_err(map_ggml("seed_prefill_logits_scale"))?;

    graph
        .set_output(logits)
        .map_err(map_ggml("seed_prefill_set_output_logits"))?;
    graph
        .set_input(embed_tensor)
        .map_err(map_ggml("seed_prefill_mark_input_embeds"))?;
    graph
        .set_input(positions)
        .map_err(map_ggml("seed_prefill_mark_input_positions"))?;
    graph
        .set_input(mask)
        .map_err(map_ggml("seed_prefill_mark_input_mask"))?;
    graph
        .set_input(seed_indices)
        .map_err(map_ggml("seed_prefill_mark_input_rows"))?;

    graph
        .prepare_outputs_for_upload(&[logits])
        .map_err(map_ggml("seed_prefill_prepare_outputs"))?;
    graph
        .set_f32_slice(embed_tensor, embeddings, "granite_seed_prefill_embeds")
        .map_err(map_ggml("seed_prefill_upload_embeds"))?;
    let position_ids: Vec<i32> = (0..n_tokens as i32).collect();
    graph
        .set_i32_slice(positions, &position_ids, "granite_seed_prefill_positions")
        .map_err(map_ggml("seed_prefill_upload_positions"))?;
    if use_flash_attention {
        let mask_bits = build_causal_mask_f16_bits(
            n_tokens,
            "granite_seed_prefill_flash_mask",
            |stage, source| GraniteSpeechDecoderError::Ggml { stage, source },
        )?;
        graph
            .set_f16_bits_slice(mask, &mask_bits, "granite_seed_prefill_mask")
            .map_err(map_ggml("seed_prefill_upload_flash_mask"))?;
    } else {
        let mask_values = super::decoder_graph::causal_mask(n_tokens);
        graph
            .set_f32_slice(mask, &mask_values, "granite_seed_prefill_mask")
            .map_err(map_ggml("seed_prefill_upload_mask"))?;
    }
    // Rows 0..n_tokens: the prompt's K/V seeds the arena's leading span.
    graph
        .set_i32_slice(seed_indices, &position_ids, "granite_seed_prefill_rows")
        .map_err(map_ggml("seed_prefill_upload_rows"))?;
    let logits_output = graph
        .compute_output_f32(logits, vocab_size)
        .map_err(map_ggml("seed_prefill_compute"))?;
    Ok(logits_output)
}

/// One incremental single-token step. `k_hist_bufs[layer]` / `v_hist_bufs[layer]`
/// are the `[head_dim, seq_len, kv_heads]` cached history for that layer. Returns
/// the next-token logits row plus, per layer, this token's `[head_dim, 1,
/// kv_heads]` K and V (to append to the cache).
#[allow(clippy::too_many_arguments)]
fn run_decode_step_graph(
    runner: &mut GgmlCpuGraphRunner,
    weights: &GraniteDecoderWeights,
    config: &GraniteSpeechDecoderConfig,
    embed_row: &[f32],
    seq_len: usize,
    k_hist_bufs: &[Vec<f32>],
    v_hist_bufs: &[Vec<f32>],
) -> Result<ForwardGraphOutput, GraniteSpeechDecoderError> {
    let head_dim = config.head_dim;
    let kv_heads = config.num_kv_heads;
    let hidden_size = config.hidden_size;
    let vocab_size = config.vocab_size;
    let kv_width = kv_heads * head_dim;
    let new_position = seq_len; // 0-based position of the new token.

    let mut graph = runner.start_graph();

    let embed_tensor = graph
        .new_tensor_2d_f32(hidden_size, 1, "granite_session_step_embed")
        .map_err(map_ggml("session_step_input_alloc"))?;
    let positions = graph
        .new_tensor_1d_i32(1, "granite_session_step_position")
        .map_err(map_ggml("session_step_position_alloc"))?;

    // Per-layer K/V history input tensors (`[head_dim, seq_len, kv_heads]`).
    let mut k_hist_tensors = Vec::with_capacity(config.num_layers);
    let mut v_hist_tensors = Vec::with_capacity(config.num_layers);
    for _ in 0..config.num_layers {
        k_hist_tensors.push(
            graph
                .new_tensor_3d_f32(head_dim, seq_len, kv_heads, "granite_session_step_k_hist")
                .map_err(map_ggml("session_step_k_hist_alloc"))?,
        );
        v_hist_tensors.push(
            graph
                .new_tensor_3d_f32(head_dim, seq_len, kv_heads, "granite_session_step_v_hist")
                .map_err(map_ggml("session_step_v_hist_alloc"))?,
        );
    }

    let mut hidden = graph
        .scale(embed_tensor, config.embedding_multiplier)
        .map_err(map_ggml("session_step_embed_scale"))?;
    let rope = GgmlRopeExtParams::qwen_neox(head_dim, seq_len + 1, config.rope_theta)
        .map_err(map_ggml("session_step_rope_params"))?;

    let mut kv_taps = Vec::with_capacity(config.num_layers);
    for index in 0..config.num_layers {
        let layer_weights = weights.layer_weights(index);
        let pre = granite_pre_attention(
            &mut graph,
            hidden,
            positions,
            &layer_weights,
            config,
            1,
            rope,
        )?;
        kv_taps.push((pre.k_perm, pre.v_perm));

        // Attend the single new query against `history ++ new`.
        let k_full = graph
            .concat(k_hist_tensors[index], pre.k_perm, 1)
            .map_err(map_ggml("session_step_k_concat"))?;
        let v_full = graph
            .concat(v_hist_tensors[index], pre.v_perm, 1)
            .map_err(map_ggml("session_step_v_concat"))?;
        let scores = graph
            .mul_mat(k_full, pre.q_perm)
            .map_err(map_ggml("session_step_scores"))?;
        // No mask: every cached key precedes the new query, so all are valid
        // (prefill's masked keys would contribute exactly 0.0 and are simply
        // absent here -- bit-identical, see module doc).
        let probs = graph
            .soft_max_ext(scores, None, config.attention_multiplier, 0.0)
            .map_err(map_ggml("session_step_softmax"))?;
        let v_t = graph
            .cont(
                graph
                    .transpose(v_full)
                    .map_err(map_ggml("session_step_v_transpose"))?,
            )
            .map_err(map_ggml("session_step_v_cont"))?;
        let attended = graph
            .mul_mat(v_t, probs)
            .map_err(map_ggml("session_step_attended"))?;
        hidden = granite_post_attention(
            &mut graph,
            hidden,
            attended,
            &layer_weights,
            config,
            1,
            false,
        )?;
    }

    let hidden_out = rms_norm(
        &graph,
        hidden,
        config.rms_norm_eps,
        weights.final_norm_weight(),
    )?;
    let lm_head_w = weight_in_major(
        &graph,
        weights.lm_head_weight(),
        config.hidden_size,
        config.vocab_size,
        "lm_head_reshape",
    )?;
    let logits_raw = linear(&graph, lm_head_w, hidden_out, "lm_head")?;
    let logits = graph
        .scale(logits_raw, 1.0 / config.logits_scaling)
        .map_err(map_ggml("session_step_logits_scale"))?;

    graph
        .set_output(logits)
        .map_err(map_ggml("session_step_set_output_logits"))?;
    for (k_tap, v_tap) in &kv_taps {
        graph
            .set_output(*k_tap)
            .map_err(map_ggml("session_step_set_output_k"))?;
        graph
            .set_output(*v_tap)
            .map_err(map_ggml("session_step_set_output_v"))?;
    }
    graph
        .set_input(embed_tensor)
        .map_err(map_ggml("session_step_mark_input_embed"))?;
    graph
        .set_input(positions)
        .map_err(map_ggml("session_step_mark_input_position"))?;
    for index in 0..config.num_layers {
        graph
            .set_input(k_hist_tensors[index])
            .map_err(map_ggml("session_step_mark_input_k_hist"))?;
        graph
            .set_input(v_hist_tensors[index])
            .map_err(map_ggml("session_step_mark_input_v_hist"))?;
    }

    let mut outputs: Vec<_> = Vec::with_capacity(1 + kv_taps.len() * 2);
    outputs.push(logits);
    for (k_tap, v_tap) in &kv_taps {
        outputs.push(*k_tap);
        outputs.push(*v_tap);
    }
    graph
        .prepare_outputs_for_upload(&outputs)
        .map_err(map_ggml("session_step_prepare_outputs"))?;

    graph
        .set_f32_slice(embed_tensor, embed_row, "granite_session_step_embed")
        .map_err(map_ggml("session_step_upload_embed"))?;
    graph
        .set_i32_slice(
            positions,
            &[new_position as i32],
            "granite_session_step_position",
        )
        .map_err(map_ggml("session_step_upload_position"))?;
    for index in 0..config.num_layers {
        graph
            .set_f32_slice(
                k_hist_tensors[index],
                &k_hist_bufs[index],
                "granite_session_step_k_hist",
            )
            .map_err(map_ggml("session_step_upload_k_hist"))?;
        graph
            .set_f32_slice(
                v_hist_tensors[index],
                &v_hist_bufs[index],
                "granite_session_step_v_hist",
            )
            .map_err(map_ggml("session_step_upload_v_hist"))?;
    }

    let mut request: Vec<_> = Vec::with_capacity(1 + kv_taps.len() * 2);
    request.push((logits, vocab_size));
    for (k_tap, v_tap) in &kv_taps {
        request.push((*k_tap, kv_width));
        request.push((*v_tap, kv_width));
    }
    let results = graph
        .compute_outputs_f32(&request)
        .map_err(map_ggml("session_step_compute"))?;

    let mut iter = results.into_iter();
    let logits_row = iter.next().expect("step logits tap");
    let mut per_layer_kv = Vec::with_capacity(config.num_layers);
    for _ in 0..config.num_layers {
        let k = iter.next().expect("step k tap");
        let v = iter.next().expect("step v tap");
        per_layer_kv.push((k, v));
    }
    Ok((logits_row, per_layer_kv))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::ggml_runtime::GgmlCpuGraphBackend;
    use crate::models::granite_speech::decoder_graph::prefill_logits;

    /// Deterministic pseudo-random f32 generator (xorshift64*, no `rand` dep),
    /// values scaled into `[-amp, amp)` so a two-layer forward stays finite.
    fn deterministic_weights(seed: u64, len: usize, amp: f32) -> Vec<f32> {
        let mut state = seed ^ 0x9E37_79B9_7F4A_7C15;
        let mut out = Vec::with_capacity(len);
        for _ in 0..len {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let unit = ((state >> 40) as u32 & 0x00FF_FFFF) as f32 / 16_777_216.0;
            out.push((unit * 2.0 - 1.0) * amp);
        }
        out
    }

    /// A tiny Granite decoder config exercising every scaling scalar plus GQA
    /// (4 query / 2 KV heads) and an even (RoPE-NEOX) head dim.
    fn tiny_config() -> GraniteSpeechDecoderConfig {
        GraniteSpeechDecoderConfig {
            hidden_size: 32,
            num_layers: 2,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 8,
            intermediate_size: 64,
            vocab_size: 48,
            rms_norm_eps: 1.0e-5,
            rope_theta: 10000.0,
            attention_multiplier: 0.0078125,
            embedding_multiplier: 12.0,
            residual_multiplier: 0.22,
            logits_scaling: 8.0,
        }
    }

    fn build_tiny_weights(config: &GraniteSpeechDecoderConfig) -> HashMap<String, Vec<f32>> {
        let d = config.hidden_size;
        let q_width = config.num_heads * config.head_dim;
        let kv_width = config.num_kv_heads * config.head_dim;
        let inter = config.intermediate_size;
        let mut weights = HashMap::new();
        let mut seed = 1u64;
        let next = |len: usize, amp: f32, seed: &mut u64| {
            *seed = seed.wrapping_add(0x1000);
            deterministic_weights(*seed, len, amp)
        };
        for layer in 0..config.num_layers {
            let p = |suffix: &str| format!("language_model.model.layers.{layer}.{suffix}");
            // Norm weights near 1.0 (RMSNorm scale); projections small.
            weights.insert(
                p("input_layernorm.weight"),
                next(d, 0.05, &mut seed).iter().map(|x| 1.0 + x).collect(),
            );
            weights.insert(
                p("self_attn.q_proj.weight"),
                next(d * q_width, 0.05, &mut seed),
            );
            weights.insert(
                p("self_attn.k_proj.weight"),
                next(d * kv_width, 0.05, &mut seed),
            );
            weights.insert(
                p("self_attn.v_proj.weight"),
                next(d * kv_width, 0.05, &mut seed),
            );
            weights.insert(
                p("self_attn.o_proj.weight"),
                next(q_width * d, 0.05, &mut seed),
            );
            weights.insert(
                p("post_attention_layernorm.weight"),
                next(d, 0.05, &mut seed).iter().map(|x| 1.0 + x).collect(),
            );
            weights.insert(p("mlp.gate_proj.weight"), next(d * inter, 0.05, &mut seed));
            weights.insert(p("mlp.up_proj.weight"), next(d * inter, 0.05, &mut seed));
            weights.insert(p("mlp.down_proj.weight"), next(inter * d, 0.05, &mut seed));
        }
        weights.insert(
            "language_model.model.norm.weight".to_string(),
            next(d, 0.05, &mut seed).iter().map(|x| 1.0 + x).collect(),
        );
        weights.insert(
            "language_model.lm_head.weight".to_string(),
            next(d * config.vocab_size, 0.05, &mut seed),
        );
        weights.insert(
            "language_model.model.embed_tokens.weight".to_string(),
            next(config.vocab_size * d, 0.1, &mut seed),
        );
        weights
    }

    fn argmax(logits: &[f32]) -> u32 {
        logits
            .iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |(best_i, best_v), (i, &v)| {
                if v > best_v { (i, v) } else { (best_i, best_v) }
            })
            .0 as u32
    }

    /// Full-recompute reference last-position logits over `token_ids`, using the
    /// one-shot `prefill_logits` (the path this session replaces).
    fn recompute_last_logits(
        config: &GraniteSpeechDecoderConfig,
        weights: &HashMap<String, Vec<f32>>,
        token_ids: &[u32],
    ) -> Vec<f32> {
        let out = prefill_logits(config, weights, token_ids, GgmlCpuGraphBackend::Cpu)
            .expect("full recompute prefill");
        let last_start = (out.n_tokens - 1) * out.vocab_size;
        out.logits[last_start..last_start + out.vocab_size].to_vec()
    }

    fn assert_bit_identical(step: usize, incremental: &[f32], recompute: &[f32]) {
        assert_eq!(
            incremental.len(),
            recompute.len(),
            "step {step}: logits width mismatch"
        );
        for (i, (a, b)) in incremental.iter().zip(recompute.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "step {step}: logit[{i}] differs (incremental {a} vs recompute {b})"
            );
        }
    }

    #[test]
    fn fixed_span_tail_mask_exposes_only_cached_prefix() {
        assert_eq!(
            fixed_span_tail_mask(4, 0),
            vec![0.0, f32::MIN, f32::MIN, f32::MIN]
        );
        assert_eq!(fixed_span_tail_mask(4, 2), vec![0.0, 0.0, 0.0, f32::MIN]);
        assert_eq!(fixed_span_tail_mask(4, 3), vec![0.0; 4]);
    }

    /// The load-bearing correctness gate: the incremental KV-cache session must
    /// reproduce the one-shot full recompute's logits BIT-FOR-BIT at every step
    /// (which also forces identical greedy token choices). Runs entirely on
    /// synthetic weights, so it needs no external checkpoint and runs in CI.
    #[test]
    fn granite_incremental_decode_matches_full_recompute_bit_exact() {
        let config = tiny_config();
        let weights = build_tiny_weights(&config);
        let prompt: Vec<u32> = vec![1, 7, 3, 42, 5, 9];

        // Prompt embeddings (raw, pre-`embedding_multiplier`) for the session.
        let mut prompt_embeddings = Vec::with_capacity(prompt.len() * config.hidden_size);
        for &id in &prompt {
            prompt_embeddings
                .extend_from_slice(embed_token_row(&config, &weights, id).expect("embed prompt"));
        }

        let mut session =
            GraniteSpeechDecodeSession::new(config, &weights, GgmlCpuGraphBackend::Cpu)
                .expect("session");

        // Step 0: prefill logits vs full recompute over the prompt alone.
        let inc0 = session
            .prefill(&prompt_embeddings, prompt.len())
            .expect("prefill");
        let ref0 = recompute_last_logits(&config, &weights, &prompt);
        assert_bit_identical(0, &inc0, &ref0);
        assert_eq!(session.cached_positions(), prompt.len());

        // Greedy-decode a handful of steps, comparing each step's incremental
        // logits against a fresh full recompute over prompt ++ generated.
        let mut generated: Vec<u32> = Vec::new();
        let mut next_logits = inc0;
        for step in 1..=8usize {
            let token = argmax(&next_logits);
            generated.push(token);

            let inc = session
                .decode_step(token, &weights)
                .expect("incremental decode step");

            let mut sequence = prompt.clone();
            sequence.extend_from_slice(&generated);
            let reference = recompute_last_logits(&config, &weights, &sequence);

            assert_bit_identical(step, &inc, &reference);
            assert_eq!(session.cached_positions(), prompt.len() + generated.len());
            next_logits = inc;
        }
    }

    /// Not a gate -- a manual demonstration that the incremental session is
    /// `O(1)` per step (flat decode time as the prefix grows) while the old
    /// recompute-the-whole-prefix path is `O(prefix)` per step (i.e. `O(n^2)`
    /// over a full decode). Uses a mid-sized synthetic config so ggml compute,
    /// not graph construction, dominates. Run with:
    /// `cargo test -p openasr-core --lib granite_incremental_decode_is_linear_not_quadratic -- --ignored --nocapture`.
    #[test]
    #[ignore = "perf demonstration (synthetic weights), not a correctness gate"]
    fn granite_incremental_decode_is_linear_not_quadratic() {
        use std::time::Instant;

        let config = GraniteSpeechDecoderConfig {
            hidden_size: 1024,
            num_layers: 6,
            num_heads: 16,
            num_kv_heads: 4,
            head_dim: 64,
            intermediate_size: 2816,
            vocab_size: 2048,
            rms_norm_eps: 1.0e-5,
            rope_theta: 10000.0,
            attention_multiplier: 0.0078125,
            embedding_multiplier: 12.0,
            residual_multiplier: 0.22,
            logits_scaling: 8.0,
        };
        let weights = build_tiny_weights(&config);
        let prompt: Vec<u32> = (0..32u32).collect();

        let mut prompt_embeddings = Vec::with_capacity(prompt.len() * config.hidden_size);
        for &id in &prompt {
            prompt_embeddings
                .extend_from_slice(embed_token_row(&config, &weights, id).expect("embed prompt"));
        }

        // Incremental: prefill once, then time individual steps as the cache grows.
        let mut session =
            GraniteSpeechDecodeSession::new(config, &weights, GgmlCpuGraphBackend::Cpu)
                .expect("session");
        let mut logits = session
            .prefill(&prompt_embeddings, prompt.len())
            .expect("prefill");
        let mut incremental_first = None;
        let mut incremental_last = None;
        let mut incremental_total = std::time::Duration::ZERO;
        let steps = 96usize;
        for step in 0..steps {
            let token = argmax(&logits) % config.vocab_size as u32;
            let start = Instant::now();
            logits = session
                .decode_step(token, &weights)
                .expect("incremental step");
            let elapsed = start.elapsed();
            incremental_total += elapsed;
            if step == 0 {
                incremental_first = Some(elapsed);
            }
            if step == steps - 1 {
                incremental_last = Some(elapsed);
            }
        }

        // Recompute: time a full-prefix forward at growing prefix lengths (the
        // work the old executor did on EVERY step).
        let mut recompute_samples = Vec::new();
        for extra in [0usize, 32, 64, 96] {
            let mut sequence = prompt.clone();
            sequence.extend((0..extra as u32).map(|i| i % config.vocab_size as u32));
            let start = Instant::now();
            let _ = prefill_logits(&config, &weights, &sequence, GgmlCpuGraphBackend::Cpu)
                .expect("recompute");
            recompute_samples.push((sequence.len(), start.elapsed()));
        }

        let inc_first = incremental_first.unwrap();
        let inc_last = incremental_last.unwrap();
        println!("== granite incremental-vs-recompute scaling ==");
        println!(
            "incremental: {steps} steps, first-step {inc_first:?}, last-step (prefix {}) {inc_last:?}, avg {:?}",
            prompt.len() + steps - 1,
            incremental_total / steps as u32
        );
        for (len, dur) in &recompute_samples {
            println!("recompute full forward: prefix {len:>4} tokens -> {dur:?}");
        }
        // The incremental last step attends a ~4x longer prefix than the first
        // yet stays within a small constant factor (projections/MLP are O(1));
        // the recompute grows roughly linearly with prefix length.
        println!(
            "incremental last/first ratio: {:.2}x (flat == O(1) per step)",
            inc_last.as_secs_f64() / inc_first.as_secs_f64().max(1e-9)
        );
    }
}
