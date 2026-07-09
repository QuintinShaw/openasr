//! Dolphin `small.cn` CTC/attention joint decode (WeNet `attention_rescoring`).
//!
//! The model's intended inference: the CTC head does a prefix-beam search over the
//! encoder frames to produce n-best content hypotheses, then the parity-verified
//! Transformer decoder rescores each hypothesis (teacher-forced over the canonical
//! `<sos><lang><region><asr><notimestamp>` prompt + the hypothesis tokens), and the
//! best combined score wins. This is `attention_rescoring`, not attention-only
//! greedy.
//!
//! Everything family-specific stays here on top of the shared decoder graph
//! ([`decode_prompt_logits`]) and the encoder's [`DolphinWeightProvider`]. The CTC
//! projection reuses the same CPU ggml graph runner as the encoder/decoder so the
//! head stays fast in debug builds.

#![allow(dead_code)]

use std::collections::HashMap;

use crate::ggml_runtime::{
    GgmlCpuGraphBackend, GgmlCpuGraphConfig, GgmlCpuGraphError, GgmlCpuGraphRunner,
};

use super::decoder_graph::{
    DolphinDecoderConfig, DolphinDecoderError, DolphinDecoderRescoreRuntime,
};
use super::encoder_graph::DolphinWeightProvider;
use crate::models::spm_decoder::{SpmDecoderConfig, decode_spm_pieces};

#[derive(Debug, thiserror::Error)]
pub(crate) enum DolphinJointDecodeError {
    #[error("dolphin joint decode shape error: {reason}")]
    Shape { reason: String },
    #[error("dolphin joint decode missing weight tensor '{name}'")]
    MissingWeight { name: String },
    #[error("dolphin joint decode weight '{name}' has {actual} values, expected {expected}")]
    WeightLen {
        name: String,
        expected: usize,
        actual: usize,
    },
    #[error("dolphin ctc head GGML backend failed at {stage}: {source}")]
    Ggml {
        stage: &'static str,
        source: GgmlCpuGraphError,
    },
    #[error(transparent)]
    Decoder(#[from] DolphinDecoderError),
    #[error("dolphin joint decode produced no hypotheses")]
    NoHypotheses,
}

/// Decode-time knobs (not baked in the pack; fixed to the WeNet reference decode).
#[derive(Debug, Clone)]
pub(crate) struct DolphinJointDecodeConfig {
    /// CTC prefix-beam width (WeNet default 10).
    pub beam_size: usize,
    /// Rescoring combination weight: `combined = attention + ctc_weight * ctc`.
    /// The reference `attention_rescoring` decode uses `0.0` (attention-only
    /// selection over the CTC n-best); the model's `0.3` is the *training* loss
    /// weight, not this decode-time knob.
    pub ctc_weight: f32,
    /// Canonical decode prompt `<sos><lang><region><asr><notimestamp>` that the
    /// decoder is conditioned on (baked as `dolphin.prompt.prefix_token_ids`).
    pub prompt_prefix: Vec<u32>,
    pub eos_token_id: u32,
    pub blank_token_id: u32,
}

/// One rescored hypothesis, kept for diagnostics/reporting.
#[derive(Debug, Clone)]
pub(crate) struct DolphinScoredHypothesis {
    pub token_ids: Vec<u32>,
    pub ctc_score: f32,
    pub attention_score: f32,
    pub combined_score: f32,
}

#[derive(Debug, Clone)]
pub(crate) struct DolphinJointDecodeResult {
    /// Content tokens of the winning (rescored) hypothesis.
    pub best_token_ids: Vec<u32>,
    /// CTC greedy content tokens (pre-rescoring), for diagnostics.
    pub ctc_greedy_token_ids: Vec<u32>,
    /// The full rescored n-best, ranked by combined score (best first).
    pub scored_nbest: Vec<DolphinScoredHypothesis>,
}

/// Run the full CTC/attention joint decode on a single encoder output.
///
/// `encoder_out` is the frame-major `[frames, d_model]` encoder output (d_model
/// innermost), the layout the encoder graph emits and the decoder graph consumes.
/// The CTC head (prefix-beam n-best generation) always reads `encoder_out`;
/// `rescoring_encoder_out` is what the Transformer decoder rescores each
/// hypothesis against -- ordinarily the same buffer, but the hotword deep-biasing
/// fusion (`models::dolphin::hotword_context`) feeds a *different*, biased buffer
/// there while leaving CTC's `encoder_out` unbiased, mirroring upstream's
/// `decode()` (which computes `ctc_logprobs` before `apply_deep_biasing` replaces
/// `encoder_out`).
pub(crate) fn joint_decode(
    decoder_config: &DolphinDecoderConfig,
    provider: &dyn DolphinWeightProvider,
    encoder_out: &[f32],
    rescoring_encoder_out: &[f32],
    frames: usize,
    decode_config: &DolphinJointDecodeConfig,
    backend: GgmlCpuGraphBackend,
) -> Result<DolphinJointDecodeResult, DolphinJointDecodeError> {
    let vocab = decoder_config.vocab_size;
    let d_model = decoder_config.d_model;
    if frames == 0 || encoder_out.len() != frames * d_model {
        return Err(DolphinJointDecodeError::Shape {
            reason: format!(
                "encoder_out has {} values, expected {frames}x{d_model}",
                encoder_out.len()
            ),
        });
    }
    if rescoring_encoder_out.len() != frames * d_model {
        return Err(DolphinJointDecodeError::Shape {
            reason: format!(
                "rescoring_encoder_out has {} values, expected {frames}x{d_model}",
                rescoring_encoder_out.len()
            ),
        });
    }
    let ctc_log_probs =
        compute_ctc_log_probs(provider, encoder_out, frames, d_model, vocab, backend)?;
    let blank = decode_config.blank_token_id as usize;
    if blank >= vocab {
        return Err(DolphinJointDecodeError::Shape {
            reason: format!("blank id {blank} out of vocab range {vocab}"),
        });
    }
    let ctc_greedy_token_ids = ctc_greedy_search(&ctc_log_probs, frames, vocab, blank);
    let nbest = ctc_prefix_beam_search(
        &ctc_log_probs,
        frames,
        vocab,
        blank,
        decode_config.beam_size.max(1),
    );
    if nbest.is_empty() {
        return Err(DolphinJointDecodeError::NoHypotheses);
    }

    let scored_nbest = attention_rescore(
        decoder_config,
        provider,
        rescoring_encoder_out,
        frames,
        decode_config,
        &nbest,
        backend,
    )?;
    let best_token_ids = scored_nbest
        .first()
        .map(|hyp| hyp.token_ids.clone())
        .ok_or(DolphinJointDecodeError::NoHypotheses)?;
    Ok(DolphinJointDecodeResult {
        best_token_ids,
        ctc_greedy_token_ids,
        scored_nbest,
    })
}

// --- CTC head projection ---------------------------------------------------

/// `ctc.ctc_lo(encoder_out)` -> `log_softmax`, returned row-major `[frames, vocab]`
/// (vocab innermost). The linear runs in the CPU ggml graph (like the encoder);
/// the softmax is a cheap Rust pass.
fn compute_ctc_log_probs(
    provider: &dyn DolphinWeightProvider,
    encoder_out: &[f32],
    frames: usize,
    d_model: usize,
    vocab: usize,
    backend: GgmlCpuGraphBackend,
) -> Result<Vec<f32>, DolphinJointDecodeError> {
    let native_weight = provider.native_weight("ctc.ctc_lo.weight");
    let weight = if native_weight.is_some() {
        None
    } else {
        Some(fetch(provider, "ctc.ctc_lo.weight", vocab * d_model)?)
    };
    let bias = fetch(provider, "ctc.ctc_lo.bias", vocab)?;

    let graph_config = GgmlCpuGraphConfig {
        context_bytes: 256 * 1024 * 1024,
        graph_size: 2048,
        n_threads: GgmlCpuGraphConfig::resolve_runtime_thread_count_for(
            backend,
            crate::ggml_runtime::GgmlCpuGraphThreadingWorkload::Default,
        ),
        backend,
        // See the matching comment in encoder_graph.rs: unconditionally
        // enabling the gallocr scheduler only bounds memory footprint, never
        // the CTC head's computed output.
        use_scheduler: true,
    };
    let ggml = |stage: &'static str| move |source| DolphinJointDecodeError::Ggml { stage, source };
    let mut runner = GgmlCpuGraphRunner::new(graph_config).map_err(ggml("runner_init"))?;
    let mut graph = runner.start_graph();

    // Weight `[vocab, d_model]` binds as ggml `[ne0=d_model, ne1=vocab]` so
    // `mul_mat(weight, enc)` projects each frame to the vocab logits. When the
    // provider keeps it quantized/f16, it binds at the stored ggml type and the
    // raw block bytes are uploaded verbatim (stays quantized in the buffer);
    // otherwise it binds f32.
    let weight_tensor = match native_weight {
        Some(native) => graph
            .new_matmul_weight_2d_typed(d_model, vocab, native.ggml_type, "dolphin_ctc_weight")
            .map_err(ggml("weight_alloc_native"))?,
        None => graph
            .new_tensor_2d_f32(d_model, vocab, "dolphin_ctc_weight")
            .map_err(ggml("weight_alloc"))?,
    };
    let bias_tensor = graph
        .new_tensor_1d_f32(vocab, "dolphin_ctc_bias")
        .map_err(ggml("bias_alloc"))?;
    let encoder_tensor = graph
        .new_tensor_2d_f32(d_model, frames, "dolphin_ctc_encoder_out")
        .map_err(ggml("encoder_alloc"))?;

    let logits = graph
        .mul_mat(weight_tensor, encoder_tensor)
        .map_err(ggml("ctc_mul_mat"))?;
    let logits = graph
        .add(logits, bias_tensor)
        .map_err(ggml("ctc_bias_add"))?;
    graph.set_output(logits).map_err(ggml("set_output"))?;
    // Every tensor this graph uploads to (rather than computes) must be
    // flagged `set_input`: it is a fresh leaf tensor in this per-call graph
    // with no buffer yet, so without the flag the scheduler's
    // backend-assignment pass has no rule to place it on and aborts.
    graph
        .set_input(weight_tensor)
        .map_err(ggml("mark_input(weight)"))?;
    graph
        .set_input(bias_tensor)
        .map_err(ggml("mark_input(bias)"))?;
    graph
        .set_input(encoder_tensor)
        .map_err(ggml("mark_input(encoder_out)"))?;
    // Allocate the forward graph through the scheduler's gallocr for
    // liveness-based buffer reuse before uploading inputs (mirrors the
    // encoder/decoder graphs).
    graph
        .prepare_outputs_for_upload(&[logits])
        .map_err(ggml("prepare_outputs"))?;

    match (native_weight, weight) {
        (Some(native), _) => graph
            .set_matmul_weight_bytes(
                weight_tensor,
                native.bytes,
                native.ggml_type,
                "dolphin_ctc_weight",
            )
            .map_err(ggml("upload_weight_native"))?,
        (None, Some(weight)) => graph
            .set_f32_slice(weight_tensor, weight, "dolphin_ctc_weight")
            .map_err(ggml("upload_weight"))?,
        (None, None) => unreachable!("ctc weight is fetched f32 when not native"),
    }
    graph
        .set_f32_slice(bias_tensor, bias, "dolphin_ctc_bias")
        .map_err(ggml("upload_bias"))?;
    graph
        .set_f32_slice(encoder_tensor, encoder_out, "dolphin_ctc_encoder_out")
        .map_err(ggml("upload_encoder"))?;

    let mut logits = graph
        .compute_output_f32(logits, frames * vocab)
        .map_err(ggml("compute"))?;

    // In-place log_softmax over each frame's vocab row.
    for row in logits.chunks_exact_mut(vocab) {
        log_softmax_in_place(row);
    }
    Ok(logits)
}

fn fetch<'p>(
    provider: &'p dyn DolphinWeightProvider,
    name: &str,
    expected: usize,
) -> Result<&'p [f32], DolphinJointDecodeError> {
    let data = provider
        .tensor(name)
        .ok_or_else(|| DolphinJointDecodeError::MissingWeight {
            name: name.to_string(),
        })?;
    if data.len() != expected {
        return Err(DolphinJointDecodeError::WeightLen {
            name: name.to_string(),
            expected,
            actual: data.len(),
        });
    }
    Ok(data)
}

fn log_softmax_in_place(row: &mut [f32]) {
    let max = row.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
    let mut sum = 0.0f64;
    for &v in row.iter() {
        sum += ((v - max) as f64).exp();
    }
    let log_sum = sum.ln() as f32;
    for v in row.iter_mut() {
        *v = *v - max - log_sum;
    }
}

// --- CTC greedy (diagnostics) ----------------------------------------------

/// Best-path CTC collapse: argmax per frame, drop blanks and consecutive repeats.
fn ctc_greedy_search(log_probs: &[f32], frames: usize, vocab: usize, blank: usize) -> Vec<u32> {
    let mut out = Vec::new();
    let mut prev: Option<usize> = None;
    for t in 0..frames {
        let row = &log_probs[t * vocab..(t + 1) * vocab];
        let argmax = row
            .iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &x)| {
                if x > bv { (i, x) } else { (bi, bv) }
            })
            .0;
        if argmax != blank && Some(argmax) != prev {
            out.push(argmax as u32);
        }
        prev = Some(argmax);
    }
    out
}

// --- CTC prefix beam search ------------------------------------------------

const NEG_INF: f64 = f64::NEG_INFINITY;

/// Stable log-sum-exp of two log-domain values.
fn log_add(a: f64, b: f64) -> f64 {
    if a == NEG_INF {
        return b;
    }
    if b == NEG_INF {
        return a;
    }
    let (hi, lo) = if a > b { (a, b) } else { (b, a) };
    hi + (1.0 + (lo - hi).exp()).ln()
}

/// WeNet CTC prefix-beam search over `[frames, vocab]` log-probs. Returns up to
/// `beam_size` `(content_tokens, ctc_log_score)` hypotheses ranked best-first.
///
/// Each surviving prefix carries `(pb, pnb)` -- the log-prob of reaching it with a
/// trailing blank vs. a trailing non-blank -- so repeated-label collapse is exact.
fn ctc_prefix_beam_search(
    log_probs: &[f32],
    frames: usize,
    vocab: usize,
    blank: usize,
    beam_size: usize,
) -> Vec<(Vec<u32>, f32)> {
    // (prefix, (log_pb, log_pnb)). Seed: empty prefix reachable only via blank.
    let mut cur: Vec<(Vec<u32>, (f64, f64))> = vec![(Vec::new(), (0.0, NEG_INF))];

    for t in 0..frames {
        let row = &log_probs[t * vocab..(t + 1) * vocab];
        let top = top_k_indices(row, beam_size);
        let mut next: HashMap<Vec<u32>, (f64, f64)> = HashMap::new();

        for &s in &top {
            let ps = row[s] as f64;
            for (prefix, (pb, pnb)) in &cur {
                if s == blank {
                    let entry = next.entry(prefix.clone()).or_insert((NEG_INF, NEG_INF));
                    entry.0 = log_add(entry.0, log_add(pb + ps, pnb + ps));
                } else if prefix.last() == Some(&(s as u32)) {
                    // Same label as the prefix tail: either a repeat (updates the
                    // non-blank prob of the same prefix) or a genuine new token
                    // after a blank (extends the prefix).
                    let entry = next.entry(prefix.clone()).or_insert((NEG_INF, NEG_INF));
                    entry.1 = log_add(entry.1, pnb + ps);
                    let mut extended = prefix.clone();
                    extended.push(s as u32);
                    let ext = next.entry(extended).or_insert((NEG_INF, NEG_INF));
                    ext.1 = log_add(ext.1, pb + ps);
                } else {
                    let mut extended = prefix.clone();
                    extended.push(s as u32);
                    let ext = next.entry(extended).or_insert((NEG_INF, NEG_INF));
                    ext.1 = log_add(ext.1, log_add(pb + ps, pnb + ps));
                }
            }
        }

        let mut items: Vec<(Vec<u32>, (f64, f64))> = next.into_iter().collect();
        items.sort_by(|a, b| {
            let sa = log_add(a.1.0, a.1.1);
            let sb = log_add(b.1.0, b.1.1);
            // Score desc; deterministic prefix tie-break so ties don't depend on
            // HashMap iteration order.
            sb.partial_cmp(&sa)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        items.truncate(beam_size);
        cur = items;
    }

    cur.into_iter()
        .map(|(prefix, (pb, pnb))| (prefix, log_add(pb, pnb) as f32))
        .collect()
}

/// Indices of the `k` largest values in `row` (unordered), `k` clamped to the row.
fn top_k_indices(row: &[f32], k: usize) -> Vec<usize> {
    let k = k.min(row.len());
    if k == 0 {
        return Vec::new();
    }
    let mut idx: Vec<usize> = (0..row.len()).collect();
    idx.select_nth_unstable_by(k - 1, |&a, &b| {
        row[b]
            .partial_cmp(&row[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    idx.truncate(k);
    idx
}

// --- Attention rescoring ---------------------------------------------------

/// Rescore each CTC hypothesis with the Transformer decoder and rank by the
/// combined `attention + ctc_weight * ctc` score.
///
/// The decoder is teacher-forced over `prompt_prefix ++ hyp_tokens`. Because the
/// prompt is identical across hypotheses (and the decoder is causal), the prompt-
/// position log-probs are a per-hypothesis constant and are excluded; the score is
/// the sum of `log P(hyp[k] | prompt, hyp[<k])` plus `log P(eos | prompt, hyp)`.
fn attention_rescore(
    decoder_config: &DolphinDecoderConfig,
    provider: &dyn DolphinWeightProvider,
    encoder_out: &[f32],
    frames: usize,
    decode_config: &DolphinJointDecodeConfig,
    nbest: &[(Vec<u32>, f32)],
    backend: GgmlCpuGraphBackend,
) -> Result<Vec<DolphinScoredHypothesis>, DolphinJointDecodeError> {
    let prompt = &decode_config.prompt_prefix;
    let prompt_len = prompt.len();
    if prompt_len == 0 {
        return Err(DolphinJointDecodeError::Shape {
            reason: "prompt prefix must be non-empty".to_string(),
        });
    }
    let vocab = decoder_config.vocab_size;
    let eos = decode_config.eos_token_id as usize;

    // Build-once/run-many (P5): every hypothesis below teacher-forces the same
    // decoder weights over the same encoder_out, differing only in the token
    // sequence, so the ~200 decoder weight tensors + the encoder memory are
    // loaded into the runtime's persistent arena exactly once here rather than
    // rebuilt and re-uploaded per hypothesis (up to DOLPHIN_BEAM_SIZE times).
    let mut runtime =
        DolphinDecoderRescoreRuntime::new(decoder_config, provider, encoder_out, frames, backend)?;

    let mut scored = Vec::with_capacity(nbest.len());
    for (tokens, ctc_score) in nbest {
        let attention_score = if tokens.is_empty() {
            // Empty hypothesis: score is just log P(eos | prompt).
            let logits = runtime.decode_prompt_logits(prompt)?;
            let mut row = logits.last_token_logits().to_vec();
            log_softmax_in_place(&mut row);
            row[eos]
        } else {
            let mut sequence = Vec::with_capacity(prompt_len + tokens.len());
            sequence.extend_from_slice(prompt);
            sequence.extend_from_slice(tokens);
            let logits = runtime.decode_prompt_logits(&sequence)?;
            score_hypothesis(&logits.logits, vocab, prompt_len, tokens, eos)
        };
        let combined = attention_score + decode_config.ctc_weight * *ctc_score;
        scored.push(DolphinScoredHypothesis {
            token_ids: tokens.clone(),
            ctc_score: *ctc_score,
            attention_score,
            combined_score: combined,
        });
    }

    scored.sort_by(|a, b| {
        b.combined_score
            .partial_cmp(&a.combined_score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.token_ids.cmp(&b.token_ids))
    });
    Ok(scored)
}

/// Sum the decoder log-probs of `tokens` (+ trailing eos) from the teacher-forced
/// logits over `prompt ++ tokens`. Row `prompt_len - 1 + k` predicts `tokens[k]`;
/// row `prompt_len - 1 + tokens.len()` predicts eos.
fn score_hypothesis(
    logits: &[f32],
    vocab: usize,
    prompt_len: usize,
    tokens: &[u32],
    eos: usize,
) -> f32 {
    let mut score = 0.0f32;
    let base = prompt_len - 1;
    for (k, &token) in tokens.iter().enumerate() {
        let row_start = (base + k) * vocab;
        let mut row = logits[row_start..row_start + vocab].to_vec();
        log_softmax_in_place(&mut row);
        score += row[token as usize];
    }
    let eos_row_start = (base + tokens.len()) * vocab;
    let mut eos_row = logits[eos_row_start..eos_row_start + vocab].to_vec();
    log_softmax_in_place(&mut eos_row);
    score + eos_row[eos]
}

// --- detokenize ------------------------------------------------------------

/// Join Dolphin content tokens into text, dropping the special `<...>` marker
/// tokens (prompt/task/blank/unk). Every Dolphin `units.txt` vocab observed so
/// far (cn-dialect small/base, and the multilingual small/base's SentencePiece
/// vocab) is a MIXED unigram vocab: single CJK characters are their own token
/// (no separator needed, so plain concatenation is correct for them), while
/// Latin-script words -- English code-switch tokens in the cn-dialect vocabs,
/// and ordinary word pieces in the multilingual vocabs -- carry a leading
/// SentencePiece `\u{2581}` ("▁") word-start marker. A token starting with `▁`
/// therefore opens a new word (emit a preceding space, dropped at the very
/// start of the transcript) with the marker itself stripped; every other
/// token (a bare CJK char, or a continuation piece with no marker) is
/// concatenated directly, exactly as before.
pub(crate) fn detokenize_char_tokens(token_ids: &[u32], tokens: &[String]) -> String {
    let pieces = token_ids
        .iter()
        .filter_map(|&id| tokens.get(id as usize).map(String::as_str));
    decode_spm_pieces(pieces, SpmDecoderConfig::MIXED_UNIGRAM_LEADING_MARKER)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two frames, vocab 3 (blank=0). Frame 0 favors token 1, frame 1 favors
    /// token 2 -> greedy "1 2".
    #[test]
    fn ctc_greedy_collapses_blanks_and_repeats() {
        // rows are already log-domain-ish; magnitude only needs the argmax right.
        let log_probs = vec![
            -5.0, -0.1, -3.0, // frame0 -> 1
            -5.0, -0.1, -3.0, // frame1 -> 1 (repeat, collapsed)
            -0.1, -5.0, -3.0, // frame2 -> blank
            -5.0, -3.0, -0.1, // frame3 -> 2
        ];
        let out = ctc_greedy_search(&log_probs, 4, 3, 0);
        assert_eq!(out, vec![1, 2]);
    }

    #[test]
    fn prefix_beam_recovers_single_token() {
        // Strongly peaked on token 1 for two frames -> "1".
        let log_probs = vec![
            -10.0, -0.001, -10.0, //
            -10.0, -0.001, -10.0, //
        ];
        let nbest = ctc_prefix_beam_search(&log_probs, 2, 3, 0, 4);
        assert_eq!(nbest[0].0, vec![1]);
    }

    #[test]
    fn top_k_indices_picks_largest() {
        let row = vec![0.1f32, 5.0, -2.0, 3.0, 4.5];
        let mut got = top_k_indices(&row, 3);
        got.sort_unstable();
        assert_eq!(got, vec![1, 3, 4]);
    }

    #[test]
    fn log_add_matches_reference() {
        let got = log_add(0.0, 0.0);
        assert!((got - (2.0f64).ln()).abs() < 1e-12);
        assert_eq!(log_add(NEG_INF, -1.0), -1.0);
        assert_eq!(log_add(-1.0, NEG_INF), -1.0);
    }

    #[test]
    fn detokenize_drops_special_markers() {
        let tokens: Vec<String> = ["<blank>", "<sos>", "学", "校", "<eos>"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(detokenize_char_tokens(&[1, 2, 3, 4], &tokens), "学校");
    }
}
