//! Shared FastConformer encoder graph: arena alloc/upload/bind plumbing, the
//! dw-striding subsampling prelude, and the conformer layer loop, carried
//! over byte-for-byte from `parakeet_ctc::encoder_graph` /
//! `parakeet_tdt::encoder_graph` (which built the identical graph shape).
//!
//! What is intentionally NOT here: the tail (CTC head vs. joint encoder
//! projection matmul + output), the family's mel/output struct names, and
//! the family's own error type. [`FastConformerEncoderCore::build`] takes
//! the tail as a pair of declare/upload closures so the "declare every arena
//! tensor, THEN upload every value" ordering the arena requires (the first
//! `set_*_slice` call freezes further `new_tensor_*` allocation) stays
//! intact across the family-specific tail tensors too.

use std::path::Path;

use crate::ggml_runtime::{
    ArenaAllocError, GgmlCpuGraphBuilder, GgmlCpuGraphConfig, GgmlCpuGraphError,
    GgmlCpuGraphRunner, GgmlCpuTensor, GgmlLoadedWeightContext, GgmlStaticTensor,
    GgmlStaticTensorArena, WeightSlot, alloc_static_f16 as arena_alloc_static_f16,
    alloc_static_f32 as arena_alloc_static_f32, bind_loaded as arena_bind_loaded,
    upload_static_f16 as arena_upload_static_f16, upload_static_f32 as arena_upload_static_f32,
};
use crate::nn::conv::{
    Conv2dParams, ConvActivation, ConvBlockSteps, apply_conv_2d_bias_activation,
    apply_conv_2d_depthwise_bias_activation, reshape_bias_4d,
};
use crate::nn::encoder::{
    ConformerBlockConfig, ConformerBlockWeights, build_relative_positional_encoding,
    conformer_block,
};
use crate::nn::half::f32_to_f16_bits;

use super::FastConformerGraphError;
use super::weights::{FastConformerLayerWeights, NamedTensor};

const ENCODER_LAYER_NORM_EPSILON: f32 = 1.0e-5;
const CONFORMER_MACARON_SCALE: f32 = 0.5;
const SUBSAMPLING_KERNEL: usize = 3;
const SUBSAMPLING_STRIDE: usize = 2;
const SUBSAMPLING_PADDING: usize = 1;

pub(crate) fn conv_out_dim(input: usize) -> usize {
    (input + 2 * SUBSAMPLING_PADDING - SUBSAMPLING_KERNEL) / SUBSAMPLING_STRIDE + 1
}

fn bf<E: FastConformerGraphError>(step: &'static str) -> impl Fn(GgmlCpuGraphError) -> E {
    move |source| E::graph_build_failed(step, source)
}

/// Bind a 2-D linear zero-copy from the mmap'd pack (`loaded`) by its on-disk
/// name. FAILS CLOSED if the loaded context is absent or the tensor is
/// missing: the host f32 `values` for bound weights are dropped at load, so
/// there is no arena fallback -- uploading an empty buffer would silently
/// corrupt the graph.
pub(crate) fn bind_loaded<E: FastConformerGraphError>(
    loaded: Option<&GgmlLoadedWeightContext>,
    name: &str,
) -> Result<WeightSlot, E> {
    arena_bind_loaded(loaded, name)
        .map(WeightSlot::Loaded)
        .map_err(E::shape)
}

/// Allocate a static arena tensor matching a host weight's stored dims (the
/// same layout the importer wrote, so `conformer_block` consumes it
/// identically).
pub(crate) fn alloc_static<E: FastConformerGraphError>(
    arena: &GgmlStaticTensorArena,
    weight: &NamedTensor,
    step: &'static str,
) -> Result<GgmlStaticTensor, E> {
    arena_alloc_static_f32(arena, &weight.dims, weight.values.len(), step, true).map_err(
        |e| match e {
            ArenaAllocError::Graph(source) => E::graph_build_failed(step, source),
            ArenaAllocError::UnsupportedRank(dims) => E::shape(format!(
                "tensor '{}' has unsupported rank {:?}",
                weight.name, dims
            )),
        },
    )
}

pub(crate) fn upload_static<E: FastConformerGraphError>(
    arena: &mut GgmlStaticTensorArena,
    tensor: GgmlStaticTensor,
    weight: &NamedTensor,
    step: &'static str,
) -> Result<(), E> {
    arena_upload_static_f32(arena, tensor, &weight.values, step).map_err(bf(step))
}

/// Allocate an f16 arena tensor for a depthwise conv kernel (ggml
/// `conv_2d_dw` requires an f16 kernel; regular `conv_2d` accepts f32).
pub(crate) fn alloc_static_f16<E: FastConformerGraphError>(
    arena: &GgmlStaticTensorArena,
    weight: &NamedTensor,
    step: &'static str,
) -> Result<GgmlStaticTensor, E> {
    arena_alloc_static_f16(arena, &weight.dims, step, true).map_err(|e| match e {
        ArenaAllocError::Graph(source) => E::graph_build_failed(step, source),
        ArenaAllocError::UnsupportedRank(dims) => {
            E::shape(format!("f16 depthwise '{}' rank {:?}", weight.name, dims))
        }
    })
}

pub(crate) fn upload_static_f16<E: FastConformerGraphError>(
    arena: &mut GgmlStaticTensorArena,
    tensor: GgmlStaticTensor,
    weight: &NamedTensor,
    step: &'static str,
) -> Result<(), E> {
    arena_upload_static_f16(arena, tensor, &weight.values, step, f32_to_f16_bits).map_err(bf(step))
}

pub(crate) fn upload_graph_f32<'a, E: FastConformerGraphError>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    tensor: GgmlCpuTensor<'a>,
    values: &[f32],
    step: &'static str,
) -> Result<(), E> {
    graph.set_f32_slice(tensor, values, step).map_err(bf(step))
}

fn find_sub<'a, E: FastConformerGraphError>(
    subsampling: &'a [NamedTensor],
    name: &str,
) -> Result<&'a NamedTensor, E> {
    subsampling
        .iter()
        .find(|t| t.name == name)
        .ok_or_else(|| E::shape(format!("missing subsampling tensor '{name}'")))
}

/// Per-layer handles for the conformer block weights. The 2-D linears
/// (`ff*.{up,down}`, `attn.{q,k,v,out,pos}`, `conv.pw{1,2}`) are `WeightSlot`
/// (bound zero-copy from the pack when a runtime path is supplied); norms,
/// biases, and the BN-folded depthwise conv stay plain arena tensors.
pub(crate) struct LayerArena {
    ff1_norm_weight: GgmlStaticTensor,
    ff1_norm_bias: GgmlStaticTensor,
    ff1_up_weight: WeightSlot,
    ff1_up_bias: GgmlStaticTensor,
    ff1_down_weight: WeightSlot,
    ff1_down_bias: GgmlStaticTensor,
    attn_norm_weight: GgmlStaticTensor,
    attn_norm_bias: GgmlStaticTensor,
    attn_q_weight: WeightSlot,
    attn_q_bias: GgmlStaticTensor,
    attn_k_weight: WeightSlot,
    attn_k_bias: GgmlStaticTensor,
    attn_v_weight: WeightSlot,
    attn_v_bias: GgmlStaticTensor,
    attn_out_weight: WeightSlot,
    attn_out_bias: GgmlStaticTensor,
    attn_pos_weight: WeightSlot,
    attn_pos_bias_u: GgmlStaticTensor,
    attn_pos_bias_v: GgmlStaticTensor,
    conv_norm_weight: GgmlStaticTensor,
    conv_norm_bias: GgmlStaticTensor,
    conv_pw1_weight: WeightSlot,
    conv_pw1_bias: GgmlStaticTensor,
    conv_dw_weight: GgmlStaticTensor,
    conv_dw_bias: GgmlStaticTensor,
    conv_pw2_weight: WeightSlot,
    conv_pw2_bias: GgmlStaticTensor,
    ff2_norm_weight: GgmlStaticTensor,
    ff2_norm_bias: GgmlStaticTensor,
    ff2_up_weight: WeightSlot,
    ff2_up_bias: GgmlStaticTensor,
    ff2_down_weight: WeightSlot,
    ff2_down_bias: GgmlStaticTensor,
    out_norm_weight: GgmlStaticTensor,
    out_norm_bias: GgmlStaticTensor,
}

pub(crate) struct SubsamplingArena {
    conv0_w: GgmlStaticTensor,
    conv0_b: GgmlStaticTensor,
    conv2_w: GgmlStaticTensor,
    conv2_b: GgmlStaticTensor,
    conv3_w: GgmlStaticTensor,
    conv3_b: GgmlStaticTensor,
    conv5_w: GgmlStaticTensor,
    conv5_b: GgmlStaticTensor,
    conv6_w: GgmlStaticTensor,
    conv6_b: GgmlStaticTensor,
    linear_w: WeightSlot,
    linear_b: GgmlStaticTensor,
    conv6_channels: usize,
}

/// Allocate one conformer layer's arena tensors + bind its 2-D linears
/// zero-copy from the mmap'd pack. Fails closed (`bind_loaded`) when a bound
/// weight is absent -- its host payload was dropped at load.
pub(crate) fn alloc_layer<E: FastConformerGraphError>(
    arena: &GgmlStaticTensorArena,
    loaded: Option<&GgmlLoadedWeightContext>,
    layer: &FastConformerLayerWeights,
) -> Result<LayerArena, E> {
    Ok(LayerArena {
        ff1_norm_weight: alloc_static(arena, &layer.ff1_norm_weight, "ff1_norm_w")?,
        ff1_norm_bias: alloc_static(arena, &layer.ff1_norm_bias, "ff1_norm_b")?,
        ff1_up_weight: bind_loaded(loaded, &layer.ff1_up_weight.name)?,
        ff1_up_bias: alloc_static(arena, &layer.ff1_up_bias, "ff1_up_b")?,
        ff1_down_weight: bind_loaded(loaded, &layer.ff1_down_weight.name)?,
        ff1_down_bias: alloc_static(arena, &layer.ff1_down_bias, "ff1_down_b")?,
        attn_norm_weight: alloc_static(arena, &layer.attn_norm_weight, "attn_norm_w")?,
        attn_norm_bias: alloc_static(arena, &layer.attn_norm_bias, "attn_norm_b")?,
        attn_q_weight: bind_loaded(loaded, &layer.attn_q_weight.name)?,
        attn_q_bias: alloc_static(arena, &layer.attn_q_bias, "attn_q_b")?,
        attn_k_weight: bind_loaded(loaded, &layer.attn_k_weight.name)?,
        attn_k_bias: alloc_static(arena, &layer.attn_k_bias, "attn_k_b")?,
        attn_v_weight: bind_loaded(loaded, &layer.attn_v_weight.name)?,
        attn_v_bias: alloc_static(arena, &layer.attn_v_bias, "attn_v_b")?,
        attn_out_weight: bind_loaded(loaded, &layer.attn_out_weight.name)?,
        attn_out_bias: alloc_static(arena, &layer.attn_out_bias, "attn_out_b")?,
        attn_pos_weight: bind_loaded(loaded, &layer.attn_pos_weight.name)?,
        attn_pos_bias_u: alloc_static(arena, &layer.attn_pos_bias_u, "attn_pos_u")?,
        attn_pos_bias_v: alloc_static(arena, &layer.attn_pos_bias_v, "attn_pos_v")?,
        conv_norm_weight: alloc_static(arena, &layer.conv_norm_weight, "conv_norm_w")?,
        conv_norm_bias: alloc_static(arena, &layer.conv_norm_bias, "conv_norm_b")?,
        conv_pw1_weight: bind_loaded(loaded, &layer.conv_pw1_weight.name)?,
        conv_pw1_bias: alloc_static(arena, &layer.conv_pw1_bias, "conv_pw1_b")?,
        conv_dw_weight: alloc_static_f16(arena, &layer.conv_dw_weight, "conv_dw_w")?,
        conv_dw_bias: alloc_static(arena, &layer.conv_dw_bias, "conv_dw_b")?,
        conv_pw2_weight: bind_loaded(loaded, &layer.conv_pw2_weight.name)?,
        conv_pw2_bias: alloc_static(arena, &layer.conv_pw2_bias, "conv_pw2_b")?,
        ff2_norm_weight: alloc_static(arena, &layer.ff2_norm_weight, "ff2_norm_w")?,
        ff2_norm_bias: alloc_static(arena, &layer.ff2_norm_bias, "ff2_norm_b")?,
        ff2_up_weight: bind_loaded(loaded, &layer.ff2_up_weight.name)?,
        ff2_up_bias: alloc_static(arena, &layer.ff2_up_bias, "ff2_up_b")?,
        ff2_down_weight: bind_loaded(loaded, &layer.ff2_down_weight.name)?,
        ff2_down_bias: alloc_static(arena, &layer.ff2_down_bias, "ff2_down_b")?,
        out_norm_weight: alloc_static(arena, &layer.out_norm_weight, "out_norm_w")?,
        out_norm_bias: alloc_static(arena, &layer.out_norm_bias, "out_norm_b")?,
    })
}

/// Upload one layer's ARENA tensors (norms, biases, BN-folded depthwise
/// conv). The 2-D linears are bound zero-copy in `alloc_layer`, so they are
/// absent here (their host f32 `values` were dropped at load).
pub(crate) fn upload_layer<E: FastConformerGraphError>(
    arena: &mut GgmlStaticTensorArena,
    layer: &FastConformerLayerWeights,
    h: &LayerArena,
) -> Result<(), E> {
    upload_static_f16(arena, h.conv_dw_weight, &layer.conv_dw_weight, "conv_dw_w")?;
    let pairs: [(GgmlStaticTensor, &NamedTensor); 23] = [
        (h.ff1_norm_weight, &layer.ff1_norm_weight),
        (h.ff1_norm_bias, &layer.ff1_norm_bias),
        (h.ff1_up_bias, &layer.ff1_up_bias),
        (h.ff1_down_bias, &layer.ff1_down_bias),
        (h.attn_norm_weight, &layer.attn_norm_weight),
        (h.attn_norm_bias, &layer.attn_norm_bias),
        (h.attn_q_bias, &layer.attn_q_bias),
        (h.attn_k_bias, &layer.attn_k_bias),
        (h.attn_v_bias, &layer.attn_v_bias),
        (h.attn_out_bias, &layer.attn_out_bias),
        (h.attn_pos_bias_u, &layer.attn_pos_bias_u),
        (h.attn_pos_bias_v, &layer.attn_pos_bias_v),
        (h.conv_norm_weight, &layer.conv_norm_weight),
        (h.conv_norm_bias, &layer.conv_norm_bias),
        (h.conv_pw1_bias, &layer.conv_pw1_bias),
        (h.conv_dw_bias, &layer.conv_dw_bias),
        (h.conv_pw2_bias, &layer.conv_pw2_bias),
        (h.ff2_norm_weight, &layer.ff2_norm_weight),
        (h.ff2_norm_bias, &layer.ff2_norm_bias),
        (h.ff2_up_bias, &layer.ff2_up_bias),
        (h.ff2_down_bias, &layer.ff2_down_bias),
        (h.out_norm_weight, &layer.out_norm_weight),
        (h.out_norm_bias, &layer.out_norm_bias),
    ];
    for (tensor, weight) in pairs {
        upload_static(arena, tensor, weight, "layer_weight")?;
    }
    Ok(())
}

pub(crate) fn conformer_weights<'a>(
    arena: &'a GgmlStaticTensorArena,
    h: &LayerArena,
) -> ConformerBlockWeights<'a> {
    let g = |t: GgmlStaticTensor| arena.graph_tensor(t);
    // Bound 2-D linears resolve to their mmap'd leaf via the `WeightSlot`
    // handle; everything else is a plain arena tensor.
    let b = |slot: WeightSlot| slot.graph(arena);
    ConformerBlockWeights {
        ff1_norm_weight: g(h.ff1_norm_weight),
        ff1_norm_bias: g(h.ff1_norm_bias),
        ff1_up_weight: b(h.ff1_up_weight),
        ff1_up_bias: g(h.ff1_up_bias),
        ff1_down_weight: b(h.ff1_down_weight),
        ff1_down_bias: g(h.ff1_down_bias),
        attn_norm_weight: g(h.attn_norm_weight),
        attn_norm_bias: g(h.attn_norm_bias),
        attn_q_weight: b(h.attn_q_weight),
        attn_q_bias: g(h.attn_q_bias),
        attn_k_weight: b(h.attn_k_weight),
        attn_k_bias: g(h.attn_k_bias),
        attn_v_weight: b(h.attn_v_weight),
        attn_v_bias: g(h.attn_v_bias),
        attn_out_weight: b(h.attn_out_weight),
        attn_out_bias: g(h.attn_out_bias),
        attn_pos_weight: b(h.attn_pos_weight),
        attn_pos_bias_u: g(h.attn_pos_bias_u),
        attn_pos_bias_v: g(h.attn_pos_bias_v),
        conv_norm_weight: g(h.conv_norm_weight),
        conv_norm_bias: g(h.conv_norm_bias),
        conv_pw1_weight: b(h.conv_pw1_weight),
        conv_pw1_bias: g(h.conv_pw1_bias),
        conv_dw_weight: g(h.conv_dw_weight),
        conv_dw_bias: g(h.conv_dw_bias),
        conv_pw2_weight: b(h.conv_pw2_weight),
        conv_pw2_bias: g(h.conv_pw2_bias),
        ff2_norm_weight: g(h.ff2_norm_weight),
        ff2_norm_bias: g(h.ff2_norm_bias),
        ff2_up_weight: b(h.ff2_up_weight),
        ff2_up_bias: g(h.ff2_up_bias),
        ff2_down_weight: b(h.ff2_down_weight),
        ff2_down_bias: g(h.ff2_down_bias),
        out_norm_weight: g(h.out_norm_weight),
        out_norm_bias: g(h.out_norm_bias),
    }
}

/// The shared FastConformer encoder residency: the graph runner, the
/// mmap-backed loaded-weight context the bound slots alias, the static
/// tensor arena, and every subsampling/conformer-layer handle. A family's
/// own `EncoderGraph` embeds this plus its own tail handles (CTC head or
/// joint encoder projection).
pub(crate) struct FastConformerEncoderCore {
    pub(crate) runner: GgmlCpuGraphRunner,
    // Owns the mmap-backed buffer the `Loaded` weight slots above alias.
    // Rust drops struct fields in declaration order (first-declared drops
    // first), so declaring it here does NOT make it outlive `arena`;
    // soundness relies on neither `arena` nor `runner` dereferencing weight
    // memory on drop. Never read directly -- it exists to keep the mapping
    // alive.
    #[allow(dead_code)]
    pub(crate) loaded_weights: Option<GgmlLoadedWeightContext>,
    pub(crate) arena: GgmlStaticTensorArena,
    pub(crate) sub: SubsamplingArena,
    pub(crate) layers: Vec<LayerArena>,
}

impl FastConformerEncoderCore {
    /// Build the runner + arena + every subsampling/conformer-layer tensor,
    /// threading the family's own tail (CTC head weight+bias, or joint
    /// encoder projection weight+bias) through as a declare/upload closure
    /// pair so it participates in the arena's single "declare everything,
    /// then upload everything" pass (the first upload call freezes further
    /// tensor allocation).
    pub(crate) fn build<E, T>(
        mut graph_config: GgmlCpuGraphConfig,
        context_bytes: usize,
        runtime_path: Option<&Path>,
        subsampling: &[NamedTensor],
        layers: &[FastConformerLayerWeights],
        declare_tail: impl FnOnce(
            &GgmlStaticTensorArena,
            Option<&GgmlLoadedWeightContext>,
        ) -> Result<T, E>,
        upload_tail: impl FnOnce(&mut GgmlStaticTensorArena, &T) -> Result<(), E>,
    ) -> Result<(Self, T), E>
    where
        E: FastConformerGraphError,
    {
        graph_config.context_bytes = context_bytes;
        // FastConformer-XL builds more graph nodes than the default 4096-node
        // cap, tripping `GGML_ASSERT(cgraph->n_nodes < cgraph->size)`. Size
        // the cgraph to the actual (data-driven) layer count with generous
        // per-layer headroom. This is capacity only -- the built graph and
        // its op order are unchanged, so a model within the default cap
        // stays byte-for-byte identical.
        graph_config.graph_size = graph_config.graph_size.max(layers.len() * 256 + 2048);
        let runner = GgmlCpuGraphRunner::new(graph_config)
            .map_err(|source| E::graph_build_failed("runner_init", source))?;
        // Bind the 2-D linears zero-copy from the mmap'd pack (no f32
        // dequantize-to-host, no arena upload). Fails closed below if the
        // load failed but a bindable weight's host payload was dropped.
        let loaded_weights =
            runtime_path.and_then(|path| runner.load_gguf_weight_context(path).ok());
        let loaded = loaded_weights.as_ref();
        let mut arena = runner
            .start_static_tensor_arena(context_bytes)
            .map_err(|source| E::graph_build_failed("static_tensor_arena", source))?;

        // ----- declare (allocate) all arena tensors first (first upload freezes) -----
        let s = |n: &str| find_sub::<E>(subsampling, n);
        let conv0_w_t = alloc_static(&arena, s("enc.sub.layers.0.weight")?, "sub0_w")?;
        let conv0_b_t = alloc_static(&arena, s("enc.sub.layers.0.bias")?, "sub0_b")?;
        let conv2_w_t = alloc_static_f16(&arena, s("enc.sub.layers.2.weight")?, "sub2_w")?;
        let conv2_b_t = alloc_static(&arena, s("enc.sub.layers.2.bias")?, "sub2_b")?;
        let conv3_w_t = alloc_static(&arena, s("enc.sub.layers.3.weight")?, "sub3_w")?;
        let conv3_b_t = alloc_static(&arena, s("enc.sub.layers.3.bias")?, "sub3_b")?;
        let conv5_w_t = alloc_static_f16(&arena, s("enc.sub.layers.5.weight")?, "sub5_w")?;
        let conv5_b_t = alloc_static(&arena, s("enc.sub.layers.5.bias")?, "sub5_b")?;
        let conv6_w_t = alloc_static(&arena, s("enc.sub.layers.6.weight")?, "sub6_w")?;
        let conv6_b_t = alloc_static(&arena, s("enc.sub.layers.6.bias")?, "sub6_b")?;
        let linear_w_slot = bind_loaded(loaded, "enc.sub.linear.weight")?;
        let linear_b_t = alloc_static(&arena, s("enc.sub.linear.bias")?, "sub_lin_b")?;
        let conv6_channels = s("enc.sub.layers.6.bias")?.values.len();

        let mut layer_arenas = Vec::with_capacity(layers.len());
        for layer in layers {
            layer_arenas.push(alloc_layer(&arena, loaded, layer)?);
        }

        let tail = declare_tail(&arena, loaded)?;

        // ----- upload all values (arena now freezes on first set) -----
        upload_static(
            &mut arena,
            conv0_w_t,
            s("enc.sub.layers.0.weight")?,
            "sub0_w",
        )?;
        upload_static(&mut arena, conv0_b_t, s("enc.sub.layers.0.bias")?, "sub0_b")?;
        upload_static_f16(
            &mut arena,
            conv2_w_t,
            s("enc.sub.layers.2.weight")?,
            "sub2_w",
        )?;
        upload_static(&mut arena, conv2_b_t, s("enc.sub.layers.2.bias")?, "sub2_b")?;
        upload_static(
            &mut arena,
            conv3_w_t,
            s("enc.sub.layers.3.weight")?,
            "sub3_w",
        )?;
        upload_static(&mut arena, conv3_b_t, s("enc.sub.layers.3.bias")?, "sub3_b")?;
        upload_static_f16(
            &mut arena,
            conv5_w_t,
            s("enc.sub.layers.5.weight")?,
            "sub5_w",
        )?;
        upload_static(&mut arena, conv5_b_t, s("enc.sub.layers.5.bias")?, "sub5_b")?;
        upload_static(
            &mut arena,
            conv6_w_t,
            s("enc.sub.layers.6.weight")?,
            "sub6_w",
        )?;
        upload_static(&mut arena, conv6_b_t, s("enc.sub.layers.6.bias")?, "sub6_b")?;
        // `enc.sub.linear.weight` is bound zero-copy; only its bias is arena-uploaded.
        upload_static(
            &mut arena,
            linear_b_t,
            s("enc.sub.linear.bias")?,
            "sub_lin_b",
        )?;
        for (layer, handles) in layers.iter().zip(&layer_arenas) {
            upload_layer(&mut arena, layer, handles)?;
        }
        upload_tail(&mut arena, &tail)?;

        Ok((
            Self {
                runner,
                loaded_weights,
                arena,
                sub: SubsamplingArena {
                    conv0_w: conv0_w_t,
                    conv0_b: conv0_b_t,
                    conv2_w: conv2_w_t,
                    conv2_b: conv2_b_t,
                    conv3_w: conv3_w_t,
                    conv3_b: conv3_b_t,
                    conv5_w: conv5_w_t,
                    conv5_b: conv5_b_t,
                    conv6_w: conv6_w_t,
                    conv6_b: conv6_b_t,
                    linear_w: linear_w_slot,
                    linear_b: linear_b_t,
                    conv6_channels,
                },
                layers: layer_arenas,
            },
            tail,
        ))
    }
}

/// Scalar knobs the shared subsampling + conformer stack needs, mapped from
/// each family's own execution-metadata struct. `scale_input` is the one
/// checkpoint-dependent knob: parakeet-ctc's HF conversion always scales
/// (`true`); parakeet-tdt-0.6b-v3's does not (metadata-driven).
#[derive(Debug, Clone, Copy)]
pub(crate) struct FastConformerStackConfig {
    pub hidden_size: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub conv_kernel: usize,
    pub subsampling_channels: usize,
    pub scale_input: bool,
}

/// Output of the shared subsampling + conformer stack: the last block's
/// output tensor (`state`, pre-tail), the mel/positional input tensors (for
/// the caller to upload after building its own tail), and the frame count.
pub(crate) struct FastConformerStackOutput<'a> {
    pub state: GgmlCpuTensor<'a>,
    pub mel_t: GgmlCpuTensor<'a>,
    pub pos_t: GgmlCpuTensor<'a>,
    pub positional: Vec<f32>,
    pub subsampled_frames: usize,
}

/// Build the dw-striding subsampling prelude (verbatim cohere FastConformer
/// clone: conv0+ReLU, dw conv2, pw conv3+ReLU, dw conv5, pw conv6+ReLU) +
/// linear + optional `scale_input` + the conformer layer loop (the shared
/// `nn::encoder::conformer_block`). The caller builds its own tail from
/// `state`, then must call `graph.set_output`, `prepare_outputs_for_upload`,
/// and upload `mel_t`/`pos_t` (via [`upload_graph_f32`] with
/// `output.positional`) before computing -- mirroring the ordering the two
/// families' `encode()` used before this was shared.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_conformer_stack<'a, E: FastConformerGraphError>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    arena: &'a GgmlStaticTensorArena,
    sub: &SubsamplingArena,
    layers: &[LayerArena],
    config: FastConformerStackConfig,
    n_mels: usize,
    n_frames: usize,
    mel_tensor_name: &'static str,
    pos_tensor_name: &'static str,
) -> Result<FastConformerStackOutput<'a>, E> {
    let d_model = config.hidden_size;
    let subsampled_frames = conv_out_dim(conv_out_dim(conv_out_dim(n_frames)));
    let subsampled_freq = conv_out_dim(conv_out_dim(conv_out_dim(n_mels)));
    let positional = build_relative_positional_encoding(d_model, subsampled_frames, || {
        E::shape("relative positional encoding shape overflow".to_string())
    })?;

    let mel_t = graph
        .new_tensor_2d_f32(n_mels, n_frames, mel_tensor_name)
        .map_err(bf("new_mel"))?;
    let pos_t = graph
        .new_tensor_2d_f32(d_model, positional.len() / d_model, pos_tensor_name)
        .map_err(bf("new_pos"))?;
    graph.set_input(mel_t).map_err(bf("set_input_mel"))?;
    graph.set_input(pos_t).map_err(bf("set_input_pos"))?;

    let conv_map = |step, source| E::graph_build_failed(step, source);
    let stride2 = Conv2dParams {
        stride_x: 2,
        stride_y: 2,
        padding_x: 1,
        padding_y: 1,
        dilation_x: 1,
        dilation_y: 1,
    };
    let pointwise = Conv2dParams {
        stride_x: 1,
        stride_y: 1,
        padding_x: 0,
        padding_y: 0,
        dilation_x: 1,
        dilation_y: 1,
    };
    let bias4d = |g: &_, t: GgmlStaticTensor, len: usize, step| {
        reshape_bias_4d(g, arena.graph_tensor(t), len, step, conv_map)
    };

    // ----- dw-striding subsampling (verbatim cohere FastConformer prelude) -----
    let mut state_4d = graph
        .reshape_4d(mel_t, n_mels, n_frames, 1, 1)
        .map_err(bf("reshape_mel_4d"))?;
    let conv0_b = bias4d(
        &*graph,
        sub.conv0_b,
        config.subsampling_channels,
        "sub0_bias4d",
    )?;
    state_4d = apply_conv_2d_bias_activation(
        &*graph,
        arena.graph_tensor(sub.conv0_w),
        state_4d,
        conv0_b,
        stride2,
        ConvActivation::Relu,
        ConvBlockSteps {
            conv: "conv0",
            bias: "conv0_bias",
            activation: "conv0_relu",
        },
        conv_map,
    )?;
    let conv2_b = bias4d(
        &*graph,
        sub.conv2_b,
        config.subsampling_channels,
        "sub2_bias4d",
    )?;
    state_4d = apply_conv_2d_depthwise_bias_activation(
        &*graph,
        arena.graph_tensor(sub.conv2_w),
        state_4d,
        conv2_b,
        stride2,
        None,
        ConvBlockSteps {
            conv: "conv2_dw",
            bias: "conv2_bias",
            activation: "conv2_noact",
        },
        conv_map,
    )?;
    let conv3_b = bias4d(
        &*graph,
        sub.conv3_b,
        config.subsampling_channels,
        "sub3_bias4d",
    )?;
    state_4d = apply_conv_2d_bias_activation(
        &*graph,
        arena.graph_tensor(sub.conv3_w),
        state_4d,
        conv3_b,
        pointwise,
        ConvActivation::Relu,
        ConvBlockSteps {
            conv: "conv3_pw",
            bias: "conv3_bias",
            activation: "conv3_relu",
        },
        conv_map,
    )?;
    let conv5_b = bias4d(
        &*graph,
        sub.conv5_b,
        config.subsampling_channels,
        "sub5_bias4d",
    )?;
    state_4d = apply_conv_2d_depthwise_bias_activation(
        &*graph,
        arena.graph_tensor(sub.conv5_w),
        state_4d,
        conv5_b,
        stride2,
        None,
        ConvBlockSteps {
            conv: "conv5_dw",
            bias: "conv5_bias",
            activation: "conv5_noact",
        },
        conv_map,
    )?;
    let conv6_b = bias4d(
        &*graph,
        sub.conv6_b,
        config.subsampling_channels,
        "sub6_bias4d",
    )?;
    state_4d = apply_conv_2d_bias_activation(
        &*graph,
        arena.graph_tensor(sub.conv6_w),
        state_4d,
        conv6_b,
        pointwise,
        ConvActivation::Relu,
        ConvBlockSteps {
            conv: "conv6_pw",
            bias: "conv6_bias",
            activation: "conv6_relu",
        },
        conv_map,
    )?;

    // flatten [channels, freq] per frame -> [channels*freq, frames] -> linear -> d_model.
    let flattened = sub
        .conv6_channels
        .checked_mul(subsampled_freq)
        .ok_or_else(|| E::shape("flatten overflow".into()))?;
    let mut state = graph
        .permute(state_4d, 0, 2, 1, 3)
        .map_err(bf("permute_flatten"))?;
    state = graph.cont(state).map_err(bf("cont_flatten"))?;
    state = graph
        .reshape_2d(state, flattened, subsampled_frames)
        .map_err(bf("reshape_flatten"))?;
    state = graph
        .mul_mat(sub.linear_w.graph(arena), state)
        .map_err(bf("sub_linear"))?;
    state = graph
        .add(state, arena.graph_tensor(sub.linear_b))
        .map_err(bf("sub_linear_bias"))?;
    // scale_input: x *= sqrt(d_model) (NeMo FastConformer), metadata-driven.
    if config.scale_input {
        state = graph
            .scale(state, (d_model as f32).sqrt())
            .map_err(bf("scale_input"))?;
    }

    // ----- conformer layers (shared nn/ block) -----
    let element = std::mem::size_of::<f32>();
    let frame = subsampled_frames;
    let block_config = ConformerBlockConfig {
        d_model,
        attention_heads: config.n_heads,
        head_dim: config.head_dim,
        frame_count: frame,
        conv_kernel: config.conv_kernel,
        layer_norm_epsilon: ENCODER_LAYER_NORM_EPSILON,
        macaron_scale: CONFORMER_MACARON_SCALE,
        rel_shift_nb1: (2 * frame - 2) * element,
        rel_shift_nb2: (2 * frame - 1) * frame * element,
        rel_shift_offset: (frame - 1) * element,
    };
    let pos_enc = pos_t;
    for handles in layers {
        let weights = conformer_weights(arena, handles);
        let block = conformer_block(graph, state, pos_enc, block_config, weights, conv_map)?;
        state = block.output;
    }

    Ok(FastConformerStackOutput {
        state,
        mel_t,
        pos_t,
        positional,
        subsampled_frames,
    })
}
