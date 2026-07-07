//! Dolphin native deep-biasing hotword context module (`context_module.*`).
//!
//! Reference: upstream `dolphin/hotword.py` (`HotwordEncoder`/`_BLSTM`) +
//! `dolphin/model.py`'s `decode()` insertion point (arXiv:2305.12493, deep
//! contextual biasing). Two stages, mirroring the upstream module exactly:
//!
//!  1. **Context extractor** ([`encode_hotword_context_embeddings`]): each hotword
//!     phrase (as a char-token id sequence, plus a mandatory leading "no-bias"
//!     `[0]` entry the upstream always prepends) runs through a 2-layer
//!     bidirectional LSTM over its own char embeddings; the last forward/backward
//!     hidden+cell states concat to a 3072-dim vector, then `context_encoder`
//!     (`Linear(3072, 768)` + `LayerNorm(768)`) projects it to one 768-dim context
//!     row. This stage runs in plain Rust, not a ggml graph: ggml has no LSTM
//!     primitive, one hotword list is a handful of short sequences (not a
//!     per-frame op), and the codebase already has pure-Rust recurrent/hand-written
//!     numeric forward-pass precedent (e.g. `diarize::vad::firered_stream`'s
//!     causal DFSMN VAD) for exactly this shape of workload. The 3072-dim
//!     recurrent weights and the embedding table are never quantized for the same
//!     reason convs/embeddings elsewhere aren't: no established block-quant
//!     precedent for recurrent/lookup operands in this codebase, and the
//!     workload is tiny so there is nothing to gain from shrinking them.
//!  2. **Biasing fusion** ([`apply_hotword_deep_biasing`]): a small ggml graph --
//!     plain (non-rel-pos) multi-head cross-attention (`biasing_layer`, query =
//!     encoder frames, key/value = the context rows) + `combiner` linear +
//!     residual + `norm_aft_combiner` LayerNorm. This *is* a per-frame op (query
//!     length = the full utterance), so it runs in-graph and reuses the encoder's
//!     attention head reshape/merge primitives; `biasing_layer`/`combiner`
//!     `.weight` matrices are ordinary rank-2 matmul operands and follow the
//!     family's normal keep-quantized policy (bound native at their pack type).
//!
//! Upstream applies this fusion to the encoder output that feeds `attention`
//! beam search / `attention_rescoring`'s decoder, but computes CTC log-probs
//! (and thus the CTC prefix-beam n-best) from the *unbiased* encoder output --
//! `model.py`'s `decode()` calls `self.ctc_logprobs(encoder_out, ...)` before
//! `apply_deep_biasing` replaces `encoder_out`. The Rust wiring in
//! [`super::executor`]/[`super::joint_decode`] mirrors that split exactly.
//!
//! Upstream's `deep_biasing_score` (a per-request scalar multiplier on the
//! attention fusion term) has no per-word semantics -- it is one global scale on
//! the whole fused delta, not a per-phrase strength. This module fixes it to the
//! upstream default `1.0` and does not read `PhraseBiasEntry::boost()`: there is
//! no trained per-phrase weight in this mechanism, so inventing one from the
//! generic logit-boost `boost` field would fabricate semantics upstream doesn't
//! have. The phrase *list* is the only signal this family's phrase bias honors.

#![allow(dead_code)]

use crate::ggml_runtime::{
    GgmlCpuGraphBackend, GgmlCpuGraphBuilder, GgmlCpuGraphConfig, GgmlCpuGraphError,
    GgmlCpuGraphRunner, GgmlCpuTensor,
};
use crate::nn::attn::{
    AttentionHeadLayout, AttentionReshapeSteps, AttentionValueMergeSteps,
    STANDARD_HEAD_PERMUTE_AXES, attention_context_from_probs,
    reshape_projection_to_attention_heads,
};
use crate::nn::norm::{AffineLayerNormSteps, apply_affine_layer_norm};

use super::encoder_graph::DolphinWeightProvider;

/// `context_module.*` embedding/hidden width (== the encoder `d_model`).
pub(crate) const HOTWORD_EMBED_DIM: usize = 768;
/// `_BLSTM`'s `nn.LSTM(..., num_layers=2, bidirectional=True)`.
const HOTWORD_LSTM_LAYERS: usize = 2;
/// `context_encoder`'s input width: `cat(last_h_bwd, last_h_fwd, last_c_bwd, last_c_fwd)`.
const HOTWORD_CONTEXT_STATE_DIM: usize = HOTWORD_EMBED_DIM * 4;
/// `HotwordEncoder(attention_heads=4, ...)` (`train.yaml`'s `context_module_conf`
/// and the `transcribe.py` hardcoded fallback both use 4).
const HOTWORD_ATTENTION_HEADS: usize = 4;
const HOTWORD_ATTENTION_HEAD_DIM: usize = HOTWORD_EMBED_DIM / HOTWORD_ATTENTION_HEADS;
/// LayerNorm epsilon (WeNet/PyTorch default, matches the encoder's).
const HOTWORD_LAYER_NORM_EPS: f32 = 1e-5;
/// `prepare_hotword_tensor`'s mandatory leading "no-bias" entry: a length-1
/// sequence of token id 0, prepended before every real hotword phrase so the
/// biasing attention always has a null option to attend to.
const HOTWORD_NO_BIAS_TOKEN_ID: u32 = 0;

#[derive(Debug, thiserror::Error)]
pub(crate) enum DolphinHotwordError {
    #[error("dolphin hotword context missing weight tensor '{name}'")]
    MissingWeight { name: String },
    #[error("dolphin hotword context weight '{name}' has {actual} values, expected {expected}")]
    WeightLen {
        name: String,
        expected: usize,
        actual: usize,
    },
    #[error("dolphin hotword context shape error: {reason}")]
    Shape { reason: String },
    #[error("dolphin hotword biasing GGML backend failed at {stage}: {source}")]
    Ggml {
        stage: &'static str,
        source: GgmlCpuGraphError,
    },
}

fn ggml_err(stage: &'static str) -> impl Fn(GgmlCpuGraphError) -> DolphinHotwordError + Copy {
    move |source| DolphinHotwordError::Ggml { stage, source }
}

fn fetch<'p>(
    provider: &'p dyn DolphinWeightProvider,
    name: &str,
    expected: usize,
) -> Result<&'p [f32], DolphinHotwordError> {
    let data = provider
        .tensor(name)
        .ok_or_else(|| DolphinHotwordError::MissingWeight {
            name: name.to_string(),
        })?;
    if data.len() != expected {
        return Err(DolphinHotwordError::WeightLen {
            name: name.to_string(),
            expected,
            actual: data.len(),
        });
    }
    Ok(data)
}

// --- Stage 1: pure-Rust BiLSTM context extractor + context_encoder ---------

/// One direction's `nn.LSTM` weights for one layer (PyTorch layout: `w_ih`/`w_hh`
/// rows ordered as the four gates `[i, f, g, o]`, each `hidden`-wide).
struct LstmDirectionWeights<'p> {
    w_ih: &'p [f32],
    w_hh: &'p [f32],
    b_ih: &'p [f32],
    b_hh: &'p [f32],
}

fn fetch_lstm_direction<'p>(
    provider: &'p dyn DolphinWeightProvider,
    layer: usize,
    reverse: bool,
    input_size: usize,
    hidden: usize,
) -> Result<LstmDirectionWeights<'p>, DolphinHotwordError> {
    let suffix = if reverse { "_reverse" } else { "" };
    let p =
        |field: &str| format!("context_module.context_extractor.sen_rnn.{field}_l{layer}{suffix}");
    Ok(LstmDirectionWeights {
        w_ih: fetch(provider, &p("weight_ih"), 4 * hidden * input_size)?,
        w_hh: fetch(provider, &p("weight_hh"), 4 * hidden * hidden)?,
        b_ih: fetch(provider, &p("bias_ih"), 4 * hidden)?,
        b_hh: fetch(provider, &p("bias_hh"), 4 * hidden)?,
    })
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// One `LSTMCell` step: `gates = W_ih @ x + b_ih + W_hh @ h_prev + b_hh`, split
/// into the four `hidden`-wide gates in PyTorch order `[i, f, g, o]`.
fn lstm_cell_step(
    x: &[f32],
    h_prev: &[f32],
    c_prev: &[f32],
    weights: &LstmDirectionWeights,
    hidden: usize,
) -> (Vec<f32>, Vec<f32>) {
    let input_size = x.len();
    let mut gates = vec![0.0f32; 4 * hidden];
    for (row, gate) in gates.iter_mut().enumerate() {
        let wi = &weights.w_ih[row * input_size..(row + 1) * input_size];
        let wh = &weights.w_hh[row * hidden..(row + 1) * hidden];
        let mut acc = weights.b_ih[row] + weights.b_hh[row];
        for (k, &xv) in x.iter().enumerate() {
            acc += wi[k] * xv;
        }
        for (k, &hv) in h_prev.iter().enumerate() {
            acc += wh[k] * hv;
        }
        *gate = acc;
    }
    let mut h = vec![0.0f32; hidden];
    let mut c = vec![0.0f32; hidden];
    for n in 0..hidden {
        let i = sigmoid(gates[n]);
        let f = sigmoid(gates[hidden + n]);
        let g = gates[2 * hidden + n].tanh();
        let o = sigmoid(gates[3 * hidden + n]);
        let cn = f * c_prev[n] + i * g;
        c[n] = cn;
        h[n] = o * cn.tanh();
    }
    (h, c)
}

/// Run one direction of one LSTM layer over `inputs` (`[t][input_size]`, time
/// order). Returns `(last_h, last_c, outputs_in_time_order)`; the "last" state is
/// whichever end of the sequence that direction finishes on (t=T-1 forward,
/// t=0 backward) -- exactly PyTorch's per-direction final `h_n`/`c_n`.
fn lstm_layer_forward(
    inputs: &[Vec<f32>],
    weights: &LstmDirectionWeights,
    hidden: usize,
    reverse: bool,
) -> (Vec<f32>, Vec<f32>, Vec<Vec<f32>>) {
    let t = inputs.len();
    let mut h = vec![0.0f32; hidden];
    let mut c = vec![0.0f32; hidden];
    let mut outputs = vec![Vec::new(); t];
    let order: Box<dyn Iterator<Item = usize>> = if reverse {
        Box::new((0..t).rev())
    } else {
        Box::new(0..t)
    };
    for idx in order {
        let (nh, nc) = lstm_cell_step(&inputs[idx], &h, &c, weights, hidden);
        h = nh;
        c = nc;
        outputs[idx] = h.clone();
    }
    (h, c, outputs)
}

fn embed_row(embed_table: &[f32], dim: usize, token_id: u32) -> Vec<f32> {
    let row = token_id as usize * dim;
    embed_table[row..row + dim].to_vec()
}

/// The upstream `_BLSTM.forward` state for one hotword phrase: embed each char
/// token, run the 2-layer bidirectional LSTM, and concat
/// `[last_h_bwd, last_h_fwd, last_c_bwd, last_c_fwd]` (PyTorch's `h_n`/`c_n`
/// layout puts the last layer's backward direction at index `-1`, forward at
/// `-2`).
fn bilstm_context_state(
    provider: &dyn DolphinWeightProvider,
    embed_table: &[f32],
    token_ids: &[u32],
) -> Result<[f32; HOTWORD_CONTEXT_STATE_DIM], DolphinHotwordError> {
    let d = HOTWORD_EMBED_DIM;
    let layer0_fwd = fetch_lstm_direction(provider, 0, false, d, d)?;
    let layer0_bwd = fetch_lstm_direction(provider, 0, true, d, d)?;
    let layer1_fwd = fetch_lstm_direction(provider, 1, false, 2 * d, d)?;
    let layer1_bwd = fetch_lstm_direction(provider, 1, true, 2 * d, d)?;

    let embedded: Vec<Vec<f32>> = token_ids
        .iter()
        .map(|&id| embed_row(embed_table, d, id))
        .collect();

    let (_, _, fwd0_seq) = lstm_layer_forward(&embedded, &layer0_fwd, d, false);
    let (_, _, bwd0_seq) = lstm_layer_forward(&embedded, &layer0_bwd, d, true);
    let layer1_input: Vec<Vec<f32>> = fwd0_seq
        .iter()
        .zip(bwd0_seq.iter())
        .map(|(f, b)| {
            let mut row = Vec::with_capacity(2 * d);
            row.extend_from_slice(f);
            row.extend_from_slice(b);
            row
        })
        .collect();

    let (fwd1_h, fwd1_c, _) = lstm_layer_forward(&layer1_input, &layer1_fwd, d, false);
    let (bwd1_h, bwd1_c, _) = lstm_layer_forward(&layer1_input, &layer1_bwd, d, true);

    let mut state = [0.0f32; HOTWORD_CONTEXT_STATE_DIM];
    state[0..d].copy_from_slice(&bwd1_h);
    state[d..2 * d].copy_from_slice(&fwd1_h);
    state[2 * d..3 * d].copy_from_slice(&bwd1_c);
    state[3 * d..4 * d].copy_from_slice(&fwd1_c);
    Ok(state)
}

/// `context_encoder`: `Linear(3072, 768)` + `LayerNorm(768)`, applied
/// per-row (batch-independent over the trailing 768/3072 dim, exactly like
/// PyTorch's `nn.Sequential(Linear, LayerNorm)` over a `(rows, 3072)` batch).
fn context_encoder_project(
    state: &[f32; HOTWORD_CONTEXT_STATE_DIM],
    linear_w: &[f32],
    linear_b: &[f32],
    ln_w: &[f32],
    ln_b: &[f32],
) -> [f32; HOTWORD_EMBED_DIM] {
    let d_out = HOTWORD_EMBED_DIM;
    let d_in = HOTWORD_CONTEXT_STATE_DIM;
    let mut projected = [0.0f32; HOTWORD_EMBED_DIM];
    for (o, slot) in projected.iter_mut().enumerate() {
        let w = &linear_w[o * d_in..(o + 1) * d_in];
        let mut acc = linear_b[o];
        for (k, &x) in state.iter().enumerate() {
            acc += w[k] * x;
        }
        *slot = acc;
    }
    let mean = projected.iter().sum::<f32>() / d_out as f32;
    let var = projected.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / d_out as f32;
    let inv_std = 1.0 / (var + HOTWORD_LAYER_NORM_EPS).sqrt();
    let mut out = [0.0f32; HOTWORD_EMBED_DIM];
    for i in 0..d_out {
        out[i] = (projected[i] - mean) * inv_std * ln_w[i] + ln_b[i];
    }
    out
}

/// Build the `[rows, 768]` context embedding matrix (row-major, 768 innermost)
/// for `hotword_token_ids` (each entry is one phrase's char-token ids). Prepends
/// the upstream's mandatory `[0]` no-bias entry, so `rows == hotword_token_ids.len() + 1`.
/// Every phrase must be non-empty (the no-bias entry is supplied internally).
pub(crate) fn encode_hotword_context_embeddings(
    provider: &dyn DolphinWeightProvider,
    hotword_token_ids: &[Vec<u32>],
) -> Result<Vec<f32>, DolphinHotwordError> {
    let d = HOTWORD_EMBED_DIM;
    let vocab_embed = provider
        .tensor("context_module.context_extractor.word_embedding.weight")
        .ok_or_else(|| DolphinHotwordError::MissingWeight {
            name: "context_module.context_extractor.word_embedding.weight".to_string(),
        })?;
    if !vocab_embed.len().is_multiple_of(d) {
        return Err(DolphinHotwordError::Shape {
            reason: format!(
                "word_embedding.weight has {} values, not a multiple of {d}",
                vocab_embed.len()
            ),
        });
    }
    let vocab_size = vocab_embed.len() / d;

    let linear_w = fetch(
        provider,
        "context_module.context_encoder.0.weight",
        d * HOTWORD_CONTEXT_STATE_DIM,
    )?;
    let linear_b = fetch(provider, "context_module.context_encoder.0.bias", d)?;
    let ln_w = fetch(provider, "context_module.context_encoder.1.weight", d)?;
    let ln_b = fetch(provider, "context_module.context_encoder.1.bias", d)?;

    let mut phrases: Vec<&[u32]> = Vec::with_capacity(hotword_token_ids.len() + 1);
    let no_bias = [HOTWORD_NO_BIAS_TOKEN_ID];
    phrases.push(&no_bias);
    for tokens in hotword_token_ids {
        if tokens.is_empty() {
            return Err(DolphinHotwordError::Shape {
                reason: "hotword phrase produced zero tokens".to_string(),
            });
        }
        for &id in tokens {
            if id as usize >= vocab_size {
                return Err(DolphinHotwordError::Shape {
                    reason: format!("hotword token id {id} is out of vocab range {vocab_size}"),
                });
            }
        }
        phrases.push(tokens.as_slice());
    }

    let mut out = Vec::with_capacity(phrases.len() * d);
    for tokens in &phrases {
        let state = bilstm_context_state(provider, vocab_embed, tokens)?;
        let row = context_encoder_project(&state, linear_w, linear_b, ln_w, ln_b);
        out.extend_from_slice(&row);
    }
    Ok(out)
}

// --- Phrase -> char-token-id tokenization -----------------------------------

/// Char-level tokenizer for hotword phrases: upstream always splits hotwords
/// per-char (`transcribe.py`: "For hotwords, always use char-level splitting --
/// BPE doesn't work well for Chinese hotwords") and maps each char through the
/// vocab, falling back to `<unk>` for a char with no exact vocab entry (mirrors
/// `CharTokenizer.tokens2ids`, which silently drops a char if even `<unk>` is
/// absent from the vocab -- this pack's vocab always carries `<unk>` per the
/// importer's fail-closed dialect-vocab guard, so that branch is unreachable in
/// practice but the fallback keeps this total rather than partial).
pub(crate) fn tokenize_hotword_phrase(vocab: &[String], phrase: &str) -> Vec<u32> {
    let unk_id = vocab.iter().position(|token| token == "<unk>");
    let char_id = |ch: char| -> Option<u32> {
        let mut buf = [0u8; 4];
        let s = ch.encode_utf8(&mut buf);
        vocab.iter().position(|token| token == s).map(|i| i as u32)
    };
    phrase
        .chars()
        .filter_map(|ch| char_id(ch).or_else(|| unk_id.map(|id| id as u32)))
        .collect()
}

// --- Stage 2: ggml biasing fusion -------------------------------------------

/// A rank-2 `.weight` matmul operand, bound at its pack-native ggml type
/// (quantized/f16) when the provider keeps it that way, else f32 -- the same
/// policy `encoder_graph`/`joint_decode` use for every other family matmul
/// weight.
enum PendingWeightUpload<'p> {
    Native { bytes: &'p [u8], ggml_type: i32 },
    F32(&'p [f32]),
}

fn bind_matmul_weight<'a, 'p>(
    graph: &GgmlCpuGraphBuilder<'a>,
    provider: &'p dyn DolphinWeightProvider,
    name: &str,
    ne0: usize,
    ne1: usize,
) -> Result<(GgmlCpuTensor<'a>, PendingWeightUpload<'p>), DolphinHotwordError> {
    if let Some(native) = provider.native_weight(name) {
        let tensor = graph
            .new_matmul_weight_2d_typed(ne0, ne1, native.ggml_type, "dolphin_hotword_weight")
            .map_err(ggml_err("weight_alloc_native"))?;
        return Ok((
            tensor,
            PendingWeightUpload::Native {
                bytes: native.bytes,
                ggml_type: native.ggml_type,
            },
        ));
    }
    let data = fetch(provider, name, ne0 * ne1)?;
    let tensor = graph
        .new_tensor_2d_f32(ne0, ne1, "dolphin_hotword_weight")
        .map_err(ggml_err("weight_alloc"))?;
    Ok((tensor, PendingWeightUpload::F32(data)))
}

fn upload_matmul_weight<'a>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    tensor: GgmlCpuTensor<'a>,
    upload: PendingWeightUpload<'_>,
) -> Result<(), DolphinHotwordError> {
    match upload {
        PendingWeightUpload::Native { bytes, ggml_type } => graph
            .set_matmul_weight_bytes(tensor, bytes, ggml_type, "dolphin_hotword_weight")
            .map_err(ggml_err("upload_weight_native")),
        PendingWeightUpload::F32(data) => graph
            .set_f32_slice(tensor, data, "dolphin_hotword_weight")
            .map_err(ggml_err("upload_weight")),
    }
}

fn bind_bias_1d<'a, 'p>(
    graph: &GgmlCpuGraphBuilder<'a>,
    provider: &'p dyn DolphinWeightProvider,
    name: &str,
    len: usize,
) -> Result<(GgmlCpuTensor<'a>, &'p [f32]), DolphinHotwordError> {
    let data = fetch(provider, name, len)?;
    let tensor = graph
        .new_tensor_1d_f32(len, "dolphin_hotword_bias")
        .map_err(ggml_err("bias_alloc"))?;
    Ok((tensor, data))
}

fn linear<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    weight: GgmlCpuTensor<'a>,
    input: GgmlCpuTensor<'a>,
    bias: GgmlCpuTensor<'a>,
    stage: &'static str,
) -> Result<GgmlCpuTensor<'a>, DolphinHotwordError> {
    let projected = graph.mul_mat(weight, input).map_err(ggml_err(stage))?;
    graph.add(projected, bias).map_err(ggml_err(stage))
}

/// Apply the upstream `HotwordEncoder.forward` fusion: plain (non-rel-pos)
/// multi-head cross-attention (query = `encoder_out` frames, key/value = the
/// `context_emb` rows) + `combiner` linear + residual + `norm_aft_combiner`
/// LayerNorm. `deep_biasing_score` is fixed to the upstream default `1.0`
/// (see the module docs on why there is no per-phrase weight to apply here).
///
/// `encoder_out` is `[frames, d_model]` frame-major (matches the encoder graph's
/// output layout); `context_emb` is `[rows, d_model]` row-major (the output of
/// [`encode_hotword_context_embeddings`]). Returns the fused `[frames, d_model]`
/// encoder output, the same layout, for the decoder's `attention_rescoring`
/// input -- CTC keeps using the unbiased `encoder_out` upstream also feeds it
/// (see the module docs).
pub(crate) fn apply_hotword_deep_biasing(
    provider: &dyn DolphinWeightProvider,
    encoder_out: &[f32],
    frames: usize,
    context_emb: &[f32],
    backend: GgmlCpuGraphBackend,
) -> Result<Vec<f32>, DolphinHotwordError> {
    let d = HOTWORD_EMBED_DIM;
    if encoder_out.len() != frames * d {
        return Err(DolphinHotwordError::Shape {
            reason: format!(
                "encoder_out has {} values, expected {frames}x{d}",
                encoder_out.len()
            ),
        });
    }
    if context_emb.is_empty() || !context_emb.len().is_multiple_of(d) {
        return Err(DolphinHotwordError::Shape {
            reason: format!(
                "context_emb has {} values, not a positive multiple of {d}",
                context_emb.len()
            ),
        });
    }
    let rows = context_emb.len() / d;

    let graph_config = GgmlCpuGraphConfig {
        context_bytes: 64 * 1024 * 1024,
        graph_size: 2048,
        n_threads: GgmlCpuGraphConfig::resolve_runtime_thread_count_for(
            backend,
            crate::ggml_runtime::GgmlCpuGraphThreadingWorkload::Default,
        ),
        backend,
        use_scheduler: backend.is_gpu_class(),
    };
    let mut runner = GgmlCpuGraphRunner::new(graph_config).map_err(ggml_err("runner_init"))?;
    let mut graph = runner.start_graph();

    // Phase A: declare every weight tensor before the first buffer alloc.
    let (q_w, q_w_up) = bind_matmul_weight(
        &graph,
        provider,
        "context_module.biasing_layer.linear_q.weight",
        d,
        d,
    )?;
    let (q_b, q_b_data) = bind_bias_1d(
        &graph,
        provider,
        "context_module.biasing_layer.linear_q.bias",
        d,
    )?;
    let (k_w, k_w_up) = bind_matmul_weight(
        &graph,
        provider,
        "context_module.biasing_layer.linear_k.weight",
        d,
        d,
    )?;
    let (k_b, k_b_data) = bind_bias_1d(
        &graph,
        provider,
        "context_module.biasing_layer.linear_k.bias",
        d,
    )?;
    let (v_w, v_w_up) = bind_matmul_weight(
        &graph,
        provider,
        "context_module.biasing_layer.linear_v.weight",
        d,
        d,
    )?;
    let (v_b, v_b_data) = bind_bias_1d(
        &graph,
        provider,
        "context_module.biasing_layer.linear_v.bias",
        d,
    )?;
    let (out_w, out_w_up) = bind_matmul_weight(
        &graph,
        provider,
        "context_module.biasing_layer.linear_out.weight",
        d,
        d,
    )?;
    let (out_b, out_b_data) = bind_bias_1d(
        &graph,
        provider,
        "context_module.biasing_layer.linear_out.bias",
        d,
    )?;
    let (combiner_w, combiner_w_up) =
        bind_matmul_weight(&graph, provider, "context_module.combiner.weight", d, d)?;
    let (combiner_b, combiner_b_data) =
        bind_bias_1d(&graph, provider, "context_module.combiner.bias", d)?;
    let (norm_w, norm_w_data) = bind_bias_1d(
        &graph,
        provider,
        "context_module.norm_aft_combiner.weight",
        d,
    )?;
    let (norm_b, norm_b_data) =
        bind_bias_1d(&graph, provider, "context_module.norm_aft_combiner.bias", d)?;

    let encoder_tensor = graph
        .new_tensor_2d_f32(d, frames, "dolphin_hotword_encoder_out")
        .map_err(ggml_err("encoder_alloc"))?;
    let context_tensor = graph
        .new_tensor_2d_f32(d, rows, "dolphin_hotword_context_emb")
        .map_err(ggml_err("context_alloc"))?;

    // Phase B: forward graph.
    let q = linear(&graph, q_w, encoder_tensor, q_b, "hotword_q")?;
    let k = linear(&graph, k_w, context_tensor, k_b, "hotword_k")?;
    let v = linear(&graph, v_w, context_tensor, v_b, "hotword_v")?;

    let layout_q = AttentionHeadLayout {
        head_dim: HOTWORD_ATTENTION_HEAD_DIM,
        attention_heads: HOTWORD_ATTENTION_HEADS,
        sequence_len: frames,
    };
    let layout_kv = AttentionHeadLayout {
        head_dim: HOTWORD_ATTENTION_HEAD_DIM,
        attention_heads: HOTWORD_ATTENTION_HEADS,
        sequence_len: rows,
    };
    let reshape_steps = AttentionReshapeSteps {
        reshape: "hotword_attn_reshape",
        permute: "hotword_attn_permute",
        cont: "hotword_attn_cont",
    };
    let map_err = |s, source| DolphinHotwordError::Ggml { stage: s, source };
    let q_heads = reshape_projection_to_attention_heads(
        &graph,
        q,
        layout_q,
        STANDARD_HEAD_PERMUTE_AXES,
        false,
        reshape_steps,
        map_err,
    )?;
    let k_heads = reshape_projection_to_attention_heads(
        &graph,
        k,
        layout_kv,
        STANDARD_HEAD_PERMUTE_AXES,
        false,
        reshape_steps,
        map_err,
    )?;
    let v_heads = reshape_projection_to_attention_heads(
        &graph,
        v,
        layout_kv,
        STANDARD_HEAD_PERMUTE_AXES,
        true,
        reshape_steps,
        map_err,
    )?;

    let k_heads_cont = graph.cont(k_heads).map_err(ggml_err("hotword_k_cont"))?;
    let scores = graph
        .mul_mat(k_heads_cont, q_heads)
        .map_err(ggml_err("hotword_scores"))?;
    let scores = graph
        .scale(scores, 1.0 / (HOTWORD_ATTENTION_HEAD_DIM as f32).sqrt())
        .map_err(ggml_err("hotword_scale"))?;
    let scores = graph
        .soft_max(scores)
        .map_err(ggml_err("hotword_softmax"))?;

    let context = attention_context_from_probs(
        &graph,
        v_heads,
        scores,
        layout_q,
        AttentionValueMergeSteps {
            value_permute: "hotword_v_t",
            value_cont: "hotword_v_t",
            context_mul: "hotword_ctx",
            context_merge_permute: "hotword_merge",
            context_merge_cont: "hotword_merge",
            context_merge_reshape: "hotword_merge",
        },
        map_err,
    )?;
    let attn_out = linear(&graph, out_w, context, out_b, "hotword_out")?;
    let combined = linear(&graph, combiner_w, attn_out, combiner_b, "hotword_combiner")?;
    let fused = graph
        .add(encoder_tensor, combined)
        .map_err(ggml_err("hotword_residual"))?;
    let output = apply_affine_layer_norm(
        &graph,
        fused,
        HOTWORD_LAYER_NORM_EPS,
        norm_w,
        norm_b,
        AffineLayerNormSteps {
            norm: "hotword_norm",
            scale: "hotword_norm",
            bias: "hotword_norm",
        },
        |s, source| DolphinHotwordError::Ggml { stage: s, source },
    )?;
    graph.set_output(output).map_err(ggml_err("set_output"))?;

    // Phase C: upload inputs + weights, then compute.
    graph
        .set_f32_slice(encoder_tensor, encoder_out, "dolphin_hotword_encoder_out")
        .map_err(ggml_err("upload_encoder"))?;
    graph
        .set_f32_slice(context_tensor, context_emb, "dolphin_hotword_context_emb")
        .map_err(ggml_err("upload_context"))?;
    upload_matmul_weight(&mut graph, q_w, q_w_up)?;
    graph
        .set_f32_slice(q_b, q_b_data, "dolphin_hotword_bias")
        .map_err(ggml_err("upload_bias"))?;
    upload_matmul_weight(&mut graph, k_w, k_w_up)?;
    graph
        .set_f32_slice(k_b, k_b_data, "dolphin_hotword_bias")
        .map_err(ggml_err("upload_bias"))?;
    upload_matmul_weight(&mut graph, v_w, v_w_up)?;
    graph
        .set_f32_slice(v_b, v_b_data, "dolphin_hotword_bias")
        .map_err(ggml_err("upload_bias"))?;
    upload_matmul_weight(&mut graph, out_w, out_w_up)?;
    graph
        .set_f32_slice(out_b, out_b_data, "dolphin_hotword_bias")
        .map_err(ggml_err("upload_bias"))?;
    upload_matmul_weight(&mut graph, combiner_w, combiner_w_up)?;
    graph
        .set_f32_slice(combiner_b, combiner_b_data, "dolphin_hotword_bias")
        .map_err(ggml_err("upload_bias"))?;
    graph
        .set_f32_slice(norm_w, norm_w_data, "dolphin_hotword_bias")
        .map_err(ggml_err("upload_bias"))?;
    graph
        .set_f32_slice(norm_b, norm_b_data, "dolphin_hotword_bias")
        .map_err(ggml_err("upload_bias"))?;

    graph
        .compute_output_f32(output, frames * d)
        .map_err(ggml_err("compute"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// Deterministic tiny weight fixture (`d_model=768` fixed by the pack
    /// contract; this test only exercises shapes/wiring, not real numerics).
    fn fixture_provider(vocab: usize) -> HashMap<String, Vec<f32>> {
        let d = HOTWORD_EMBED_DIM;
        let mut m = HashMap::new();
        let mut fill = |name: &str, len: usize, seed: f32| {
            m.insert(
                name.to_string(),
                (0..len)
                    .map(|i| ((i as f32 + seed) * 0.001).sin())
                    .collect(),
            );
        };
        fill(
            "context_module.context_extractor.word_embedding.weight",
            vocab * d,
            1.0,
        );
        for layer in 0..HOTWORD_LSTM_LAYERS {
            let input_size = if layer == 0 { d } else { 2 * d };
            for reverse in [false, true] {
                let suffix = if reverse { "_reverse" } else { "" };
                fill(
                    &format!("context_module.context_extractor.sen_rnn.weight_ih_l{layer}{suffix}"),
                    4 * d * input_size,
                    2.0,
                );
                fill(
                    &format!("context_module.context_extractor.sen_rnn.weight_hh_l{layer}{suffix}"),
                    4 * d * d,
                    3.0,
                );
                fill(
                    &format!("context_module.context_extractor.sen_rnn.bias_ih_l{layer}{suffix}"),
                    4 * d,
                    4.0,
                );
                fill(
                    &format!("context_module.context_extractor.sen_rnn.bias_hh_l{layer}{suffix}"),
                    4 * d,
                    5.0,
                );
            }
        }
        fill(
            "context_module.context_encoder.0.weight",
            d * HOTWORD_CONTEXT_STATE_DIM,
            6.0,
        );
        fill("context_module.context_encoder.0.bias", d, 7.0);
        fill("context_module.context_encoder.1.weight", d, 1000.0);
        fill("context_module.context_encoder.1.bias", d, 0.0);
        for name in [
            "context_module.biasing_layer.linear_q.weight",
            "context_module.biasing_layer.linear_k.weight",
            "context_module.biasing_layer.linear_v.weight",
            "context_module.biasing_layer.linear_out.weight",
            "context_module.combiner.weight",
        ] {
            fill(name, d * d, 8.0);
        }
        for name in [
            "context_module.biasing_layer.linear_q.bias",
            "context_module.biasing_layer.linear_k.bias",
            "context_module.biasing_layer.linear_v.bias",
            "context_module.biasing_layer.linear_out.bias",
            "context_module.combiner.bias",
        ] {
            fill(name, d, 9.0);
        }
        m.insert(
            "context_module.norm_aft_combiner.weight".to_string(),
            vec![1.0; d],
        );
        m.insert(
            "context_module.norm_aft_combiner.bias".to_string(),
            vec![0.0; d],
        );
        m
    }

    #[test]
    fn tokenize_hotword_phrase_maps_known_chars_and_falls_back_to_unk() {
        let vocab: Vec<String> = ["<blank>", "<unk>", "学", "校"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(tokenize_hotword_phrase(&vocab, "学校"), vec![2, 3]);
        // An unknown char (not in the tiny fixture vocab) falls back to <unk>.
        assert_eq!(tokenize_hotword_phrase(&vocab, "学河"), vec![2, 1]);
    }

    #[test]
    fn context_embeddings_prepend_the_no_bias_row_and_shape_correctly() {
        let provider = fixture_provider(64);
        let phrases = vec![vec![3u32, 4, 5], vec![10u32]];
        let emb = encode_hotword_context_embeddings(&provider, &phrases).expect("embeddings");
        assert_eq!(emb.len(), (phrases.len() + 1) * HOTWORD_EMBED_DIM);
    }

    #[test]
    fn context_embeddings_reject_out_of_vocab_token() {
        let provider = fixture_provider(4);
        let phrases = vec![vec![100u32]];
        let error = encode_hotword_context_embeddings(&provider, &phrases).unwrap_err();
        assert!(matches!(error, DolphinHotwordError::Shape { .. }));
    }

    #[test]
    fn context_embeddings_reject_empty_phrase() {
        let provider = fixture_provider(64);
        let phrases = vec![Vec::new()];
        let error = encode_hotword_context_embeddings(&provider, &phrases).unwrap_err();
        assert!(matches!(error, DolphinHotwordError::Shape { .. }));
    }

    #[test]
    fn biasing_fusion_preserves_frame_shape() {
        let provider = fixture_provider(64);
        let frames = 5;
        let d = HOTWORD_EMBED_DIM;
        let encoder_out: Vec<f32> = (0..frames * d).map(|i| (i as f32 * 0.01).sin()).collect();
        let phrases = vec![vec![3u32, 4, 5]];
        let context_emb =
            encode_hotword_context_embeddings(&provider, &phrases).expect("embeddings");
        let fused = apply_hotword_deep_biasing(
            &provider,
            &encoder_out,
            frames,
            &context_emb,
            GgmlCpuGraphBackend::Cpu,
        )
        .expect("fusion");
        assert_eq!(fused.len(), frames * d);
        // A real (non-degenerate) fusion changes the frames from the raw
        // encoder_out (attention + combiner + norm are not the identity map).
        assert_ne!(fused, encoder_out);
    }
}
