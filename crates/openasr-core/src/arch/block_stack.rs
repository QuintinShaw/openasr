//! Data-driven block-stack descriptors (P4 "new model = data").
//!
//! These types declare, as pure `&'static` data on each architecture, *which*
//! shared `nn/` layer block a stage's layers are built from, *how many* layers
//! there are (by hparam key), and *where* the per-layer weights' tensor names
//! live. A per-orchestration-shape composer (added in a later stage) walks this
//! descriptor to build the layer stack instead of a hand-coded `for layer` loop,
//! mirroring `llama.cpp`'s per-architecture `build_graph`.
//!
//! S1 scope: types + data + fail-closed validation only. Nothing builds graphs
//! from these yet — the composer that consumes them lands in S2+. Keeping the
//! data and its validation in the tree first means the descriptors are exercised
//! (and kept honest) by startup validation before any code path depends on them.

/// Which shared `nn/` layer block one stage's layers are built from.
///
/// Each variant names exactly one `nn::encoder`/`nn::decoder` block whose op
/// sequence the composer must reproduce bit-identically.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OpenAsrBlockKind {
    /// `nn::encoder::transformer_layer` — masked scaled-dot self-attention +
    /// GeLU FFN (qwen audio encoder).
    TransformerEncoderLayer,
    /// `nn::encoder::conformer_block` — macaron FFN + relative-position
    /// self-attention + depthwise conv (cohere encoder).
    ConformerBlock,
    /// `nn::wav2vec2::wav2vec2_post_norm_encoder_layer` — post-norm
    /// (`do_stable_layer_norm=False`) self-attention + GeLU FFN, full
    /// bidirectional attention, no rel-pos (wav2vec2 base/hubert/data2vec).
    Wav2Vec2PostNormEncoderLayer,
    /// `nn::encoder::sanm_fsmn_encoder_layer` — SenseVoice/Paraformer SAN-M
    /// block: multi-head self-attention whose attention context is summed with a
    /// DFSMN memory branch (a depthwise conv1d over the value/context sequence,
    /// expressible via im2col like the conformer depthwise conv), followed by a
    /// position-wise FFN. Non-autoregressive CTC encoder (`Ctc` shape).
    SanMFsmnEncoderLayer,
    /// `nn::decoder::seq2seq_layer` — self-attention KV + cross-attention +
    /// ReLU FFN (cohere/whisper decoder).
    Seq2SeqDecoderLayer,
    /// `nn::decoder::llm_layer` — RMSNorm + (fused/split) QKV + QK-norm + RoPE +
    /// GQA self-attention + SwiGLU FFN (qwen LLM decoder).
    LlmDecoderLayer,
}

/// The high-level decode orchestration a composer drives. The per-architecture
/// glue (frontend, splicing, cross-KV precompute, decode loop) lives behind a
/// `ShapeOrchestrator` keyed by this — a *new* model on an existing shape is
/// expected to be data + maybe a new block, never a new orchestrator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OpenAsrOrchestrationShape {
    /// Audio-conditioned causal LLM decoder (qwen): audio encoder → prompt
    /// splice → autoregressive LLM decode loop, no cross-attention.
    LlmDecoder,
    /// Seq2seq encoder-decoder with cross-attention (cohere/whisper): encoder →
    /// precomputed cross-KV → cross-attending decode loop.
    Seq2SeqEncoderDecoder,
    /// Non-autoregressive CTC (parakeet/conformer-CTC): encoder → CTC head →
    /// greedy frame-argmax + blank-collapse. There is NO decoder stage, NO KV
    /// cache, NO cross-attention, NO autoregressive loop — so its `block_stack`
    /// has `decoder_stage: None` and it does not use the seq2seq decode loop.
    Ctc,
}

/// One layer-stack stage (encoder or decoder): which block, how many layers, and
/// the tensor-name scope its per-layer weights are bound from.
///
/// `layer_count_hparam` is **per-stage** — encoder and decoder each carry their
/// own (qwen reads `audio.n_layers` for the encoder and `llm.n_layers` for the
/// decoder; cohere reads encoder vs decoder layer counts) — never one global
/// layer count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct OpenAsrStageDescriptor {
    /// The `nn/` block every layer in this stage is built from.
    pub block_kind: OpenAsrBlockKind,
    /// The hparam key whose value is this stage's layer count. MUST be present
    /// in the architecture's `hparam_schema`.
    pub layer_count_hparam: &'static str,
    /// The P5 tensor-name scope (block prefix, e.g. `"blk"`, `"enc.blk"`) the
    /// composer binds this stage's per-layer weights from.
    pub tensor_name_scope: &'static str,
}

/// The encoder/decoder block stack for an architecture, plus the orchestration
/// shape whose glue hosts it.
///
/// `encoder_stage` is `None` for shapes with no separate encoder stack.
/// `decoder_stage` is `None` for the non-autoregressive `Ctc` shape (encoder +
/// CTC head only) and `Some` for every decoding shape (LlmDecoder, Seq2Seq);
/// `validate_block_stack` fails closed on the wrong presence per shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct OpenAsrBlockStackDescriptor {
    pub orchestration_shape: OpenAsrOrchestrationShape,
    pub encoder_stage: Option<OpenAsrStageDescriptor>,
    pub decoder_stage: Option<OpenAsrStageDescriptor>,
}

impl OpenAsrBlockStackDescriptor {
    /// Every present stage in declaration order (encoder first when present,
    /// then decoder when present), for validation and composition walks. A `Ctc`
    /// stack yields just its encoder stage.
    pub(crate) fn stages(&self) -> impl Iterator<Item = &OpenAsrStageDescriptor> {
        self.encoder_stage.iter().chain(self.decoder_stage.iter())
    }
}
