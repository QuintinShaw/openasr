//! ReDimNet2-B6 backbone ggml graph: `stem -> stage0..5 -> fin_wght1d -> head
//! -> fin_to2d -> ASTP pool -> BN -> linear -> 192-d embedding`.
//!
//! Mirrors `models::dolphin::encoder_graph`'s pattern (arena weight upload,
//! `start_graph`/`set_input`/`compute_output(s)_f32`), fed from the f32 `.oasr`
//! pack via `diarize::embed::weights::Weights::from_oasr`. See
//! `docs/design/redimnet2-b6-embedder.md` and `HANDOFF.md` for the staged
//! bring-up plan and golden anchors this module's tests pin against.

use crate::ggml_runtime::GgmlCpuGraphRunner;
use crate::ggml_runtime::{
    GgmlCpuGraphBackend, GgmlCpuGraphConfig, GgmlCpuGraphError, GgmlCpuTensor, GgmlStaticTensor,
    GgmlStaticTensorArena,
};

use super::super::weights::{Weights, WeightsError};
use super::config::{self, StageConfig};
use super::ops;

#[derive(Debug, thiserror::Error)]
pub(crate) enum RedimNetBackboneError {
    #[error("redimnet backbone weight error: {0}")]
    Weights(#[from] WeightsError),
    #[error("redimnet backbone shape error: {reason}")]
    Shape { reason: String },
    #[error("redimnet backbone ggml error: {0}")]
    Ggml(#[from] GgmlCpuGraphError),
}

fn shape_err(reason: impl Into<String>) -> RedimNetBackboneError {
    RedimNetBackboneError::Shape {
        reason: reason.into(),
    }
}

/// Pending weight upload: `(arena handle, owned f32 data)`. Owned (not
/// borrowed) so both pack-sourced tensors and host-precomputed BatchNorm
/// affines (`ops::batchnorm_affine`) share one upload path.
struct Pending {
    handle: GgmlStaticTensor,
    data: Vec<f32>,
}

/// Loads every backbone weight into the arena, on demand, by GGUF tensor name
/// (verbatim `backbone.*`/`pool.*`/`bn.*`/`linear.*`, matching
/// `tooling/redimnet2/convert_redimnet2.py`'s `remap_tensor`). Two allocation
/// phases like `dolphin::encoder_graph::WeightBuilder`: every tensor is
/// allocated first (arena freezes after its first upload), then everything is
/// uploaded once.
struct WBuilder<'p> {
    weights: &'p Weights,
    pending: Vec<Pending>,
}

impl<'p> WBuilder<'p> {
    fn new(weights: &'p Weights) -> Self {
        Self {
            weights,
            pending: Vec::new(),
        }
    }

    fn shape(&self, name: &str) -> Result<Vec<usize>, RedimNetBackboneError> {
        Ok(self.weights.shape(name)?.to_vec())
    }

    /// Fetch a tensor's flat f32 data and assert its pack shape (ne-order)
    /// against the caller's expectation before it is ever bound to a
    /// graph-tensor shape -- a mismatch here is a converter/name-formula bug,
    /// and must fail closed instead of silently reinterpreting bytes.
    fn fetch(&self, name: &str, expect_ne: &[usize]) -> Result<Vec<f32>, RedimNetBackboneError> {
        let shape = self.weights.shape(name)?;
        if shape != expect_ne {
            return Err(shape_err(format!(
                "tensor '{name}' has pack shape {shape:?}, expected ne {expect_ne:?}"
            )));
        }
        Ok(self.weights.get(name)?.to_vec())
    }

    /// Like [`Self::fetch`] but only checks the flat element count, not the
    /// exact `ne` rank: the GGUF writer trims *trailing* (highest-index, in
    /// `ne` order) size-1 dims, so `weigth1d`'s `(1,N,CF,1)` param reads back
    /// as rank-2 `[1,CF]` when `N==1` (both trailing dims collapse) or rank-3
    /// `[1,CF,N]` otherwise (only the final `1` collapses) -- the underlying
    /// flat byte order is unaffected either way, which is all
    /// `weigth1d_softmax_host` depends on.
    fn fetch_flat(&self, name: &str, expect_len: usize) -> Result<Vec<f32>, RedimNetBackboneError> {
        let data = self.weights.get(name)?;
        if data.len() != expect_len {
            return Err(shape_err(format!(
                "tensor '{name}' has {} elements, expected {expect_len}",
                data.len()
            )));
        }
        Ok(data.to_vec())
    }

    fn tensor_1d<'a>(
        &mut self,
        arena: &GgmlStaticTensorArena,
        name: &str,
        len: usize,
    ) -> Result<GgmlCpuTensor<'a>, RedimNetBackboneError> {
        let data = self.fetch(name, &[len])?;
        let handle = arena.new_tensor_1d_f32(len, "redimnet_weight")?;
        self.pending.push(Pending { handle, data });
        Ok(arena.graph_tensor(handle))
    }

    fn tensor_2d<'a>(
        &mut self,
        arena: &GgmlStaticTensorArena,
        name: &str,
        ne0: usize,
        ne1: usize,
    ) -> Result<GgmlCpuTensor<'a>, RedimNetBackboneError> {
        let data = self.fetch(name, &[ne0, ne1])?;
        let handle = arena.new_tensor_2d_f32(ne0, ne1, "redimnet_weight")?;
        self.pending.push(Pending { handle, data });
        Ok(arena.graph_tensor(handle))
    }

    fn tensor_3d<'a>(
        &mut self,
        arena: &GgmlStaticTensorArena,
        name: &str,
        ne0: usize,
        ne1: usize,
        ne2: usize,
    ) -> Result<GgmlCpuTensor<'a>, RedimNetBackboneError> {
        let data = self.fetch(name, &[ne0, ne1, ne2])?;
        let handle = arena.new_tensor_3d_f32(ne0, ne1, ne2, "redimnet_weight")?;
        self.pending.push(Pending { handle, data });
        Ok(arena.graph_tensor(handle))
    }

    fn tensor_4d<'a>(
        &mut self,
        arena: &GgmlStaticTensorArena,
        name: &str,
        ne0: usize,
        ne1: usize,
        ne2: usize,
        ne3: usize,
    ) -> Result<GgmlCpuTensor<'a>, RedimNetBackboneError> {
        let data = self.fetch(name, &[ne0, ne1, ne2, ne3])?;
        let handle = arena.new_tensor_4d_f32(ne0, ne1, ne2, ne3, "redimnet_weight")?;
        self.pending.push(Pending { handle, data });
        Ok(arena.graph_tensor(handle))
    }

    /// Reads a down-conv (or any grouped/full) `Conv2d` kernel at whatever
    /// shape the pack actually stores (`ne=[kw,kh,cin_per_group,cout]`),
    /// self-checking only the total element count via the fetched length, and
    /// returns both the tensor and its ne dims (the caller derives `groups =
    /// cin_running / cin_per_group` from `dims[2]`).
    fn tensor_4d_any<'a>(
        &mut self,
        arena: &GgmlStaticTensorArena,
        name: &str,
    ) -> Result<(GgmlCpuTensor<'a>, [usize; 4]), RedimNetBackboneError> {
        let shape = self.shape(name)?;
        if shape.len() != 4 {
            return Err(shape_err(format!(
                "tensor '{name}' has rank {}, expected 4",
                shape.len()
            )));
        }
        let dims = [shape[0], shape[1], shape[2], shape[3]];
        let data = self.weights.get(name)?.to_vec();
        let handle =
            arena.new_tensor_4d_f32(dims[0], dims[1], dims[2], dims[3], "redimnet_weight")?;
        self.pending.push(Pending { handle, data });
        Ok((arena.graph_tensor(handle), dims))
    }

    /// Precomputes a BatchNorm's eval-mode affine on the host from the four
    /// pack tensors (`{prefix}.weight/.bias/.running_mean/.running_var`) and
    /// uploads the two resulting `ne=[channels]` (`scale`,`shift`) tensors.
    fn batchnorm_affine<'a>(
        &mut self,
        arena: &GgmlStaticTensorArena,
        prefix: &str,
        channels: usize,
        eps: f32,
    ) -> Result<(GgmlCpuTensor<'a>, GgmlCpuTensor<'a>), RedimNetBackboneError> {
        let gamma = self.fetch(&format!("{prefix}.weight"), &[channels])?;
        let beta = self.fetch(&format!("{prefix}.bias"), &[channels])?;
        let mean = self.fetch(&format!("{prefix}.running_mean"), &[channels])?;
        let var = self.fetch(&format!("{prefix}.running_var"), &[channels])?;
        let (scale, shift) = ops::batchnorm_affine(&gamma, &beta, &mean, &var, eps);
        let scale_handle = arena.new_tensor_1d_f32(channels, "redimnet_bn_scale")?;
        let shift_handle = arena.new_tensor_1d_f32(channels, "redimnet_bn_shift")?;
        self.pending.push(Pending {
            handle: scale_handle,
            data: scale,
        });
        self.pending.push(Pending {
            handle: shift_handle,
            data: shift,
        });
        Ok((
            arena.graph_tensor(scale_handle),
            arena.graph_tensor(shift_handle),
        ))
    }

    /// A single f32 scalar leaf, broadcastable against anything (used for the
    /// ASTP epsilon/clamp constant -- see `ops::astp_pool`).
    fn scalar<'a>(
        &mut self,
        arena: &GgmlStaticTensorArena,
        value: f32,
    ) -> Result<GgmlCpuTensor<'a>, RedimNetBackboneError> {
        let handle = arena.new_tensor_2d_f32(1, 1, "redimnet_scalar")?;
        self.pending.push(Pending {
            handle,
            data: vec![value],
        });
        Ok(arena.graph_tensor(handle))
    }

    fn upload(&self, arena: &mut GgmlStaticTensorArena) -> Result<(), RedimNetBackboneError> {
        for p in &self.pending {
            arena.set_f32_slice(p.handle, &p.data, "redimnet_weight")?;
        }
        Ok(())
    }
}

/// Weight handles for one `ResBasicBlock` (`ConvBlock2d`).
struct ResBlockW<'a> {
    w: ops::ResBasicBlockWeights<'a>,
}

fn load_resblock<'a>(
    b: &mut WBuilder<'_>,
    arena: &GgmlStaticTensorArena,
    prefix: &str,
    channels: usize,
) -> Result<ResBlockW<'a>, RedimNetBackboneError> {
    let conv1 = b.tensor_4d(arena, &format!("{prefix}.conv1.weight"), 3, 3, 1, channels)?;
    let conv1pw_w = b.tensor_4d(
        arena,
        &format!("{prefix}.conv1pw.weight"),
        1,
        1,
        channels,
        channels,
    )?;
    let conv1pw_b = b.tensor_1d(arena, &format!("{prefix}.conv1pw.bias"), channels)?;
    let (bn1_scale, bn1_shift) =
        b.batchnorm_affine(arena, &format!("{prefix}.bn1"), channels, 1e-5)?;
    let conv2 = b.tensor_4d(arena, &format!("{prefix}.conv2.weight"), 3, 3, 1, channels)?;
    let conv2pw_w = b.tensor_4d(
        arena,
        &format!("{prefix}.conv2pw.weight"),
        1,
        1,
        channels,
        channels,
    )?;
    let conv2pw_b = b.tensor_1d(arena, &format!("{prefix}.conv2pw.bias"), channels)?;
    let (bn2_scale, bn2_shift) =
        b.batchnorm_affine(arena, &format!("{prefix}.bn2"), channels, 1e-5)?;
    Ok(ResBlockW {
        w: ops::ResBasicBlockWeights {
            conv1,
            conv1pw_w,
            conv1pw_b,
            bn1_scale,
            bn1_shift,
            conv2,
            conv2pw_w,
            conv2pw_b,
            bn2_scale,
            bn2_shift,
        },
    })
}

/// Weight handles for one `TimeContextBlock1d` (`red_dim_conv` + 4x
/// `ConvNeXtLikeBlock` + `TransformerEncoderLayer` + `exp_dim_conv`).
struct TcmW<'a> {
    red_dim_conv_w: GgmlCpuTensor<'a>,
    red_dim_conv_b: GgmlCpuTensor<'a>,
    red_dim_ln_w: GgmlCpuTensor<'a>,
    red_dim_ln_b: GgmlCpuTensor<'a>,
    dw: [DwBlockW<'a>; 4],
    attn: ops::TransformerEncoderLayerWeights<'a>,
    exp_dim_conv_w: GgmlCpuTensor<'a>,
    exp_dim_conv_b: GgmlCpuTensor<'a>,
}

struct DwBlockW<'a> {
    kernel: GgmlCpuTensor<'a>,
    bias: GgmlCpuTensor<'a>,
    bn_scale: GgmlCpuTensor<'a>,
    bn_shift: GgmlCpuTensor<'a>,
    pwconv_w: GgmlCpuTensor<'a>,
    pwconv_b: GgmlCpuTensor<'a>,
}

fn load_tcm<'a>(
    b: &mut WBuilder<'_>,
    arena: &GgmlStaticTensorArena,
    prefix: &str,
    hidden: usize,
) -> Result<TcmW<'a>, RedimNetBackboneError> {
    let cf = config::AGG_CHANNELS;
    let red_dim_conv_w = b.tensor_3d(
        arena,
        &format!("{prefix}.red_dim_conv.0.weight"),
        1,
        cf,
        hidden,
    )?;
    let red_dim_conv_b = b.tensor_1d(arena, &format!("{prefix}.red_dim_conv.0.bias"), hidden)?;
    let red_dim_ln_w = b.tensor_1d(arena, &format!("{prefix}.red_dim_conv.1.weight"), hidden)?;
    let red_dim_ln_b = b.tensor_1d(arena, &format!("{prefix}.red_dim_conv.1.bias"), hidden)?;

    let kernels = config::TCM_DWCONV_KERNELS;
    let mut dw: Vec<DwBlockW<'a>> = Vec::with_capacity(4);
    for (idx, &k) in kernels.iter().enumerate() {
        let p = format!("{prefix}.tcm.{idx}");
        // `nn.Conv1d(hC,hC,k,groups=hC)` weight is torch `(hC,1,k)` -> ggml
        // `ne=[k,1,hC]` (rank-3, native Conv1d layout; unlike the 2D depthwise
        // convs there is no redundant leading `1` axis to make this rank-4 in
        // the pack). `depthwise_conv1d_ct` reshapes it to `[k,1,1,hC]` inline.
        let kernel = b.tensor_3d(arena, &format!("{p}.dwconvs.0.weight"), k, 1, hidden)?;
        let bias = b.tensor_1d(arena, &format!("{p}.dwconvs.0.bias"), hidden)?;
        let (bn_scale, bn_shift) = b.batchnorm_affine(arena, &format!("{p}.norm"), hidden, 1e-5)?;
        let pwconv_w = b.tensor_3d(arena, &format!("{p}.pwconv1.weight"), 1, hidden, hidden)?;
        let pwconv_b = b.tensor_1d(arena, &format!("{p}.pwconv1.bias"), hidden)?;
        dw.push(DwBlockW {
            kernel,
            bias,
            bn_scale,
            bn_shift,
            pwconv_w,
            pwconv_b,
        });
    }
    let dw: [DwBlockW<'a>; 4] = dw.try_into().ok().expect("4 dwconv kernels");

    let ap = format!("{prefix}.tcm.4.attention");
    let attn = ops::TransformerEncoderLayerWeights {
        q_w: b.tensor_2d(arena, &format!("{ap}.q_proj.weight"), hidden, hidden)?,
        q_b: b.tensor_1d(arena, &format!("{ap}.q_proj.bias"), hidden)?,
        k_w: b.tensor_2d(arena, &format!("{ap}.k_proj.weight"), hidden, hidden)?,
        k_b: b.tensor_1d(arena, &format!("{ap}.k_proj.bias"), hidden)?,
        v_w: b.tensor_2d(arena, &format!("{ap}.v_proj.weight"), hidden, hidden)?,
        v_b: b.tensor_1d(arena, &format!("{ap}.v_proj.bias"), hidden)?,
        out_w: b.tensor_2d(arena, &format!("{ap}.out_proj.weight"), hidden, hidden)?,
        out_b: b.tensor_1d(arena, &format!("{ap}.out_proj.bias"), hidden)?,
        layer_norm_w: b.tensor_1d(arena, &format!("{prefix}.tcm.4.layer_norm.weight"), hidden)?,
        layer_norm_b: b.tensor_1d(arena, &format!("{prefix}.tcm.4.layer_norm.bias"), hidden)?,
        ff_intermediate_w: b.tensor_2d(
            arena,
            &format!("{prefix}.tcm.4.feed_forward.intermediate_dense.weight"),
            hidden,
            hidden,
        )?,
        ff_intermediate_b: b.tensor_1d(
            arena,
            &format!("{prefix}.tcm.4.feed_forward.intermediate_dense.bias"),
            hidden,
        )?,
        ff_output_w: b.tensor_2d(
            arena,
            &format!("{prefix}.tcm.4.feed_forward.output_dense.weight"),
            hidden,
            hidden,
        )?,
        ff_output_b: b.tensor_1d(
            arena,
            &format!("{prefix}.tcm.4.feed_forward.output_dense.bias"),
            hidden,
        )?,
        final_layer_norm_w: b.tensor_1d(
            arena,
            &format!("{prefix}.tcm.4.final_layer_norm.weight"),
            hidden,
        )?,
        final_layer_norm_b: b.tensor_1d(
            arena,
            &format!("{prefix}.tcm.4.final_layer_norm.bias"),
            hidden,
        )?,
    };

    let exp_dim_conv_w = b.tensor_3d(
        arena,
        &format!("{prefix}.exp_dim_conv.weight"),
        1,
        hidden,
        cf,
    )?;
    let exp_dim_conv_b = b.tensor_1d(arena, &format!("{prefix}.exp_dim_conv.bias"), cf)?;

    Ok(TcmW {
        red_dim_conv_w,
        red_dim_conv_b,
        red_dim_ln_w,
        red_dim_ln_b,
        dw,
        attn,
        exp_dim_conv_w,
        exp_dim_conv_b,
    })
}

fn run_tcm<'a>(
    graph: &mut crate::ggml_runtime::GgmlCpuGraphBuilder<'a>,
    x_tc_boundary: GgmlCpuTensor<'a>,
    t: usize,
    hidden: usize,
    w: &TcmW<'a>,
) -> Result<GgmlCpuTensor<'a>, RedimNetBackboneError> {
    // Transpose the [T,CF] boundary tensor to the channel-inner internal
    // convention (ne=[CF,T]) once, run the whole TCM there, transpose back.
    let x_ct = graph.cont(graph.transpose(x_tc_boundary)?)?;
    let skip = x_ct;

    let red = ops::linear_ct(
        graph,
        reshape_kernel_3d_to_2d(graph, w.red_dim_conv_w, config::AGG_CHANNELS, hidden)?,
        w.red_dim_conv_b,
        x_ct,
    )?;
    let mut hstate = ops::layernorm_ct(graph, red, w.red_dim_ln_w, w.red_dim_ln_b, 1e-6)?;

    for (dwb, &k) in w.dw.iter().zip(config::TCM_DWCONV_KERNELS.iter()) {
        hstate = ops::convnext_dw_block_ct(
            graph,
            hstate,
            dwb.kernel,
            dwb.bias,
            dwb.bn_scale,
            dwb.bn_shift,
            reshape_kernel_3d_to_2d(graph, dwb.pwconv_w, hidden, hidden)?,
            dwb.pwconv_b,
            hidden,
            t,
            k,
        )?;
    }

    hstate = ops::transformer_encoder_layer_ct(graph, hstate, hidden, t, 4, 1e-6, &w.attn)?;

    let exp = ops::linear_ct(
        graph,
        reshape_kernel_3d_to_2d(graph, w.exp_dim_conv_w, hidden, config::AGG_CHANNELS)?,
        w.exp_dim_conv_b,
        hstate,
    )?;
    let out_ct = graph.add(skip, exp)?;
    let out_tc = graph.cont(graph.transpose(out_ct)?)?;
    Ok(out_tc)
}

/// `nn.Conv1d(cin,cout,1).weight` is torch shape `(cout,cin,1)`; reversed ggml
/// `ne=[1,cin,cout]`. `mul_mat` needs a plain 2D `[cin,cout]` weight (the
/// trailing size-1 kernel axis is a no-op), so this drops it via `reshape_2d`
/// (data-preserving: the source is already contiguous with that exact byte
/// order since `ne0=1`).
fn reshape_kernel_3d_to_2d<'a>(
    graph: &crate::ggml_runtime::GgmlCpuGraphBuilder<'a>,
    kernel_3d: GgmlCpuTensor<'a>,
    cin: usize,
    cout: usize,
) -> Result<GgmlCpuTensor<'a>, RedimNetBackboneError> {
    Ok(graph.reshape_2d(kernel_3d, cin, cout)?)
}

/// Weight handles for one backbone stage.
struct StageW<'a> {
    agg_w: Vec<GgmlCpuTensor<'a>>, // per-channel softmax weights, one per prior feature map.
    downconv_kernel: GgmlCpuTensor<'a>,
    downconv_bias: GgmlCpuTensor<'a>,
    downconv_dims: [usize; 4], // [kw,kh,cin_per_group,cout_total]
    blocks: Vec<ResBlockW<'a>>,
    squeeze: Option<(
        GgmlCpuTensor<'a>,
        GgmlCpuTensor<'a>,
        GgmlCpuTensor<'a>,
        GgmlCpuTensor<'a>,
    )>, // (1x1 w,b, bn_scale,bn_shift), only when conv_exp!=1
    tcm: TcmW<'a>,
    gnorm_w: GgmlCpuTensor<'a>,
    gnorm_b: GgmlCpuTensor<'a>,
}

fn load_stage<'a>(
    b: &mut WBuilder<'_>,
    arena: &GgmlStaticTensorArena,
    stage_idx: usize,
    cfg: &StageConfig,
    c_in: usize,
) -> Result<StageW<'a>, RedimNetBackboneError> {
    let prefix = format!("backbone.stage{stage_idx}");
    let n_feats = stage_idx + 1; // stem + stages processed so far.
    let cf = config::AGG_CHANNELS;
    let raw_w = b.fetch_flat(&format!("{prefix}.0.w"), n_feats * cf)?;
    let softmax = ops::weigth1d_softmax_host(&raw_w, n_feats, cf);
    let mut agg_w = Vec::with_capacity(n_feats);
    for (i, w) in softmax.into_iter().enumerate() {
        let handle = arena.new_tensor_2d_f32(1, cf, "redimnet_agg_w")?;
        b.pending.push(Pending { handle, data: w });
        agg_w.push(arena.graph_tensor(handle));
        let _ = i;
    }

    let (downconv_kernel, downconv_dims) = b.tensor_4d_any(arena, &format!("{prefix}.2.weight"))?;
    let downconv_bias = b.tensor_1d(arena, &format!("{prefix}.2.bias"), cfg.block_channels)?;
    if downconv_dims[3] != cfg.block_channels {
        return Err(shape_err(format!(
            "stage{stage_idx} downconv cout {} != expected block_channels {}",
            downconv_dims[3], cfg.block_channels
        )));
    }
    if !c_in.is_multiple_of(downconv_dims[2]) {
        return Err(shape_err(format!(
            "stage{stage_idx} downconv cin_per_group {} does not divide c_in {}",
            downconv_dims[2], c_in
        )));
    }

    let mut blocks = Vec::with_capacity(cfg.num_blocks);
    for i in 0..cfg.num_blocks {
        let block_idx = 3 + i;
        blocks.push(load_resblock(
            b,
            arena,
            &format!("{prefix}.{block_idx}.conv_block"),
            cfg.block_channels,
        )?);
    }

    let after_blocks_idx = 3 + cfg.num_blocks;
    let has_squeeze = (cfg.conv_exp - 1.0).abs() > 1e-6;
    let (squeeze, to1d_idx) = if has_squeeze {
        let seq_idx = after_blocks_idx;
        let w = b.tensor_4d(
            arena,
            &format!("{prefix}.{seq_idx}.0.weight"),
            1,
            1,
            cfg.block_channels,
            cfg.c_out_2d,
        )?;
        let bias = b.tensor_1d(arena, &format!("{prefix}.{seq_idx}.0.bias"), cfg.c_out_2d)?;
        let (scale, shift) =
            b.batchnorm_affine(arena, &format!("{prefix}.{seq_idx}.1"), cfg.c_out_2d, 1e-6)?;
        (Some((w, bias, scale, shift)), seq_idx + 1)
    } else {
        (None, after_blocks_idx)
    };
    let tcm_idx = to1d_idx + 1;
    let gnorm_idx = tcm_idx + 2;

    let tcm = load_tcm(b, arena, &format!("{prefix}.{tcm_idx}"), cfg.tcm_hidden)?;
    let gnorm_w = b.tensor_1d(arena, &format!("{prefix}.{gnorm_idx}.weight"), cf)?;
    let gnorm_b = b.tensor_1d(arena, &format!("{prefix}.{gnorm_idx}.bias"), cf)?;

    Ok(StageW {
        agg_w,
        downconv_kernel,
        downconv_bias,
        downconv_dims,
        blocks,
        squeeze,
        tcm,
        gnorm_w,
        gnorm_b,
    })
}

/// Runs one backbone stage: `agg1d -> to2d -> downconv -> N x ResBasicBlock ->
/// [1x1conv+BN if conv_exp!=1] -> to1d -> TCM -> Upsample(stt) -> GroupNorm`.
/// `prior_outputs` are all `ne=[t_full,CF]` (canonical, full time resolution);
/// returns the stage's own `ne=[t_full,CF]` output (a new entry to append to
/// `outputs_1d`).
#[allow(clippy::too_many_arguments)]
fn run_stage<'a>(
    graph: &mut crate::ggml_runtime::GgmlCpuGraphBuilder<'a>,
    prior_outputs: &[GgmlCpuTensor<'a>],
    t_full: usize,
    c_in: usize,
    f_in: usize,
    cfg: &StageConfig,
    w: &StageW<'a>,
) -> Result<GgmlCpuTensor<'a>, RedimNetBackboneError> {
    let cf = config::AGG_CHANNELS;
    let agg = ops::weigth1d_apply(graph, prior_outputs, &w.agg_w)?;
    let x2d = ops::to2d(graph, agg, c_in, f_in, t_full)?;

    let kw = w.downconv_dims[0];
    let kh = w.downconv_dims[1];
    let cin_per_group = w.downconv_dims[2];
    let groups = c_in / cin_per_group;
    let downconv = ops::grouped_conv2d(
        graph,
        w.downconv_kernel,
        w.downconv_bias,
        x2d,
        kw,
        kh,
        t_full,
        f_in,
        c_in,
        cfg.block_channels,
        groups,
        kw,
        kh,
    )?;
    if !f_in.is_multiple_of(kh) || !t_full.is_multiple_of(kw) {
        return Err(shape_err("downconv stride does not evenly divide t/f"));
    }
    let t_reduced = t_full / kw;
    let f_out = f_in / kh;

    let mut x = downconv;
    for block in &w.blocks {
        x = ops::resbasic_block_2d(graph, x, cfg.block_channels, &block.w)?;
    }

    if let Some((sw, sb, sscale, sshift)) = w.squeeze {
        x = ops::conv1x1_2d(graph, sw, sb, x, cfg.c_out_2d)?;
        x = ops::apply_channel_affine_2d(graph, x, cfg.c_out_2d, sscale, sshift)?;
    }

    let x1d = ops::to1d(graph, x, cfg.c_out_2d, f_out, t_reduced)?;
    let after_tcm = run_tcm(graph, x1d, t_reduced, cfg.tcm_hidden, &w.tcm)?;
    let stt = t_full / t_reduced;
    let upsampled = ops::nearest_upsample_time(graph, after_tcm, stt, t_reduced, cf)?;
    Ok(ops::group_norm_1d(
        graph,
        upsampled,
        t_full,
        cf,
        w.gnorm_w,
        w.gnorm_b,
        config::AGG_GNORM_GROUPS,
        1e-5,
    )?)
}

/// Every stem + stage weight, plus the top-level `fin_wght1d`/`head`/`pool`/
/// `bn`/`linear` weights, held for the duration of one forward call.
pub(crate) struct RedimNetBackboneWeights<'a> {
    stem_conv_w: GgmlCpuTensor<'a>,
    stem_conv_b: GgmlCpuTensor<'a>,
    stem_ln_w: GgmlCpuTensor<'a>,
    stem_ln_b: GgmlCpuTensor<'a>,
    stem_gnorm_w: GgmlCpuTensor<'a>,
    stem_gnorm_b: GgmlCpuTensor<'a>,
    stages: Vec<StageW<'a>>,
    fin_wght1d_w: Vec<GgmlCpuTensor<'a>>,
    head_w: GgmlCpuTensor<'a>,
    head_b: GgmlCpuTensor<'a>,
    pool_linear1_w: GgmlCpuTensor<'a>,
    pool_linear1_b: GgmlCpuTensor<'a>,
    pool_linear2_w: GgmlCpuTensor<'a>,
    pool_linear2_b: GgmlCpuTensor<'a>,
    bn_scale: GgmlCpuTensor<'a>,
    bn_shift: GgmlCpuTensor<'a>,
    linear_w: GgmlCpuTensor<'a>,
    linear_b: GgmlCpuTensor<'a>,
    eps_1e7: GgmlCpuTensor<'a>,
}

/// Loads the entire backbone (stem + 6 stages + top-level heads) into the
/// arena. Call once per `runner`, then `upload`, then build the per-call
/// forward graph.
fn load_weights<'a>(
    b: &mut WBuilder<'_>,
    arena: &GgmlStaticTensorArena,
) -> Result<RedimNetBackboneWeights<'a>, RedimNetBackboneError> {
    let c = config::C;
    let f = config::F;
    let cf = config::AGG_CHANNELS;

    let stem_conv_w = b.tensor_4d(arena, "backbone.stem.0.weight", 3, 3, 1, c)?;
    let stem_conv_b = b.tensor_1d(arena, "backbone.stem.0.bias", c)?;
    let stem_ln_w = b.tensor_1d(arena, "backbone.stem.1.weight", c)?;
    let stem_ln_b = b.tensor_1d(arena, "backbone.stem.1.bias", c)?;
    let stem_gnorm_w = b.tensor_1d(arena, "backbone.stem_gnorm.weight", cf)?;
    let stem_gnorm_b = b.tensor_1d(arena, "backbone.stem_gnorm.bias", cf)?;

    let mut stages = Vec::with_capacity(config::STAGES.len());
    let mut c_running = c;
    let mut f_running = f;
    for (i, cfg) in config::STAGES.iter().enumerate() {
        stages.push(load_stage(b, arena, i, cfg, c_running)?);
        c_running = cfg.c_out_2d;
        f_running = cfg.f_out;
    }
    let _ = f_running;

    let n_fin = config::FIN_WGHT1D_N;
    let raw_fin = b.fetch_flat("backbone.fin_wght1d.w", n_fin * cf)?;
    let softmax = ops::weigth1d_softmax_host(&raw_fin, n_fin, cf);
    let mut fin_wght1d_w = Vec::with_capacity(n_fin);
    for w in softmax {
        let handle = arena.new_tensor_2d_f32(1, cf, "redimnet_fin_w")?;
        b.pending.push(Pending { handle, data: w });
        fin_wght1d_w.push(arena.graph_tensor(handle));
    }

    let final_c = config::STAGES.last().expect("6 stages").c_out_2d;
    let head_w = b.tensor_4d(
        arena,
        "backbone.head.weight",
        1,
        1,
        final_c,
        config::OUT_CHANNELS,
    )?;
    let head_b = b.tensor_1d(arena, "backbone.head.bias", config::OUT_CHANNELS)?;

    let pool_in = config::PRE_POOL_CHANNELS * 3;
    let pool_bottleneck = 128;
    let pool_linear1_w = b.tensor_3d(arena, "pool.linear1.weight", 1, pool_in, pool_bottleneck)?;
    let pool_linear1_b = b.tensor_1d(arena, "pool.linear1.bias", pool_bottleneck)?;
    let pool_linear2_w = b.tensor_3d(
        arena,
        "pool.linear2.weight",
        1,
        pool_bottleneck,
        config::PRE_POOL_CHANNELS,
    )?;
    let pool_linear2_b = b.tensor_1d(arena, "pool.linear2.bias", config::PRE_POOL_CHANNELS)?;

    let (bn_scale, bn_shift) = b.batchnorm_affine(arena, "bn", config::POOL_OUT_DIM, 1e-5)?;
    let linear_w = b.tensor_2d(
        arena,
        "linear.weight",
        config::POOL_OUT_DIM,
        config::EMBED_DIM,
    )?;
    let linear_b = b.tensor_1d(arena, "linear.bias", config::EMBED_DIM)?;

    let eps_1e7 = b.scalar(arena, 1e-7)?;

    Ok(RedimNetBackboneWeights {
        stem_conv_w,
        stem_conv_b,
        stem_ln_w,
        stem_ln_b,
        stem_gnorm_w,
        stem_gnorm_b,
        stages,
        fin_wght1d_w,
        head_w,
        head_b,
        pool_linear1_w,
        pool_linear1_b,
        pool_linear2_w,
        pool_linear2_b,
        bn_scale,
        bn_shift,
        linear_w,
        linear_b,
        eps_1e7,
    })
}

/// `stem`: `Conv2d(1,64,3,'same') -> LayerNorm(64,channels_first) -> to1d ->
/// GroupNorm(64,4608)`. `spec` is the front end's `ne=[t,f]` output (`t`
/// innermost, matches `RedimNetFrontend::forward`'s row-major `[mel,frame]`
/// layout reversed); `t` must already be truncated to a multiple of
/// `config::TIME_STRIDE`.
pub(crate) fn run_stem<'a>(
    graph: &crate::ggml_runtime::GgmlCpuGraphBuilder<'a>,
    spec: GgmlCpuTensor<'a>,
    t: usize,
    w: &RedimNetBackboneWeights<'a>,
) -> Result<GgmlCpuTensor<'a>, RedimNetBackboneError> {
    let f = config::F;
    let c = config::C;
    let spec4d = graph.reshape_4d(spec, t, f, 1, 1)?;
    let conv = graph.conv_2d(w.stem_conv_w, spec4d, 1, 1, 1, 1, 1, 1)?;
    let conv_b4d = graph.reshape_4d(w.stem_conv_b, 1, 1, c, 1)?;
    let conv = graph.add(conv, conv_b4d)?;
    let normed = ops::layernorm_channels_first_2d(graph, conv, w.stem_ln_w, w.stem_ln_b, 1e-6)?;
    let x1d = ops::to1d(graph, normed, c, f, t)?;
    Ok(ops::group_norm_1d(
        graph,
        x1d,
        t,
        config::AGG_CHANNELS,
        w.stem_gnorm_w,
        w.stem_gnorm_b,
        config::AGG_GNORM_GROUPS,
        1e-5,
    )?)
}

/// Full backbone forward: `spec -> stem -> stage0..5 -> fin_wght1d -> head ->
/// fin_to2d -> flatten -> ASTP -> BN -> linear -> L2-normalize`. Returns every
/// tap named in `HANDOFF.md`/`B6_STRUCTURE_SPEC.md` so parity tests can pin
/// each stage independently; production callers only need `embedding`.
pub(crate) struct RedimNetForwardTaps<'a> {
    pub outputs_1d: Vec<GgmlCpuTensor<'a>>, // [stem, stage0..5] (8 with fin appended below)
    pub fin_wght1d: GgmlCpuTensor<'a>,
    pub backbone_2d: GgmlCpuTensor<'a>,
    pub pre_pool_flat: GgmlCpuTensor<'a>,
    pub post_pool: GgmlCpuTensor<'a>,
    pub post_bn: GgmlCpuTensor<'a>,
    pub embedding: GgmlCpuTensor<'a>,
    pub t_full: usize,
}

pub(crate) fn forward<'a>(
    graph: &mut crate::ggml_runtime::GgmlCpuGraphBuilder<'a>,
    spec: GgmlCpuTensor<'a>,
    t_raw: usize,
    w: &RedimNetBackboneWeights<'a>,
) -> Result<RedimNetForwardTaps<'a>, RedimNetBackboneError> {
    let t_full = (t_raw / config::TIME_STRIDE) * config::TIME_STRIDE;
    if t_full == 0 {
        return Err(shape_err("input too short for TIME_STRIDE truncation"));
    }
    let cf = config::AGG_CHANNELS;

    // `spec` is `ne=[t_raw,F]` (`T` innermost); truncating to a `TIME_STRIDE`
    // multiple keeps the first `t_full` time steps of *every* mel row, which
    // (since `T` is `ne0`, the per-row stride is `t_raw` elements, not
    // `t_full`) is a genuinely strided view, not a plain byte-range slice.
    let spec = if t_full == t_raw {
        spec
    } else {
        let elem = std::mem::size_of::<f32>();
        let truncated = graph.view_2d(spec, t_full, config::F, t_raw * elem, 0)?;
        graph.cont(truncated)?
    };

    let stem_out = run_stem(graph, spec, t_full, w)?;
    let mut outputs_1d: Vec<GgmlCpuTensor<'a>> = vec![stem_out];

    let mut c_running = config::C;
    let mut f_running = config::F;
    for (cfg, stage_w) in config::STAGES.iter().zip(w.stages.iter()) {
        let out = run_stage(
            graph,
            &outputs_1d,
            t_full,
            c_running,
            f_running,
            cfg,
            stage_w,
        )?;
        outputs_1d.push(out);
        c_running = cfg.c_out_2d;
        f_running = cfg.f_out;
    }

    let fin = ops::weigth1d_apply(graph, &outputs_1d, &w.fin_wght1d_w)?;

    let final_c = config::STAGES.last().expect("6 stages").c_out_2d;
    let final_f = config::F / config::FREQ_STRIDE;
    let fin_2d = ops::to2d(graph, fin, final_c, final_f, t_full)?;
    let head_out = ops::conv1x1_2d(graph, w.head_w, w.head_b, fin_2d, config::OUT_CHANNELS)?;

    let pre_pool_flat =
        ops::flatten_backbone_output(graph, head_out, config::OUT_CHANNELS, final_f, t_full)?;

    let post_pool = ops::astp_pool(
        graph,
        pre_pool_flat,
        t_full,
        config::PRE_POOL_CHANNELS,
        w.eps_1e7,
        reshape_kernel_3d_to_2d(graph, w.pool_linear1_w, config::PRE_POOL_CHANNELS * 3, 128)?,
        w.pool_linear1_b,
        reshape_kernel_3d_to_2d(graph, w.pool_linear2_w, 128, config::PRE_POOL_CHANNELS)?,
        w.pool_linear2_b,
    )?;

    let post_bn = ops::apply_channel_affine_ct(graph, post_pool, w.bn_scale, w.bn_shift)?;
    let embedding = ops::linear_ct(graph, w.linear_w, w.linear_b, post_bn)?;

    let _ = cf;
    Ok(RedimNetForwardTaps {
        outputs_1d,
        fin_wght1d: fin,
        backbone_2d: head_out,
        pre_pool_flat,
        post_pool,
        post_bn,
        embedding,
        t_full,
    })
}

/// A ready-to-use graph runner config for the backbone (generous `graph_size`:
/// the grouped-down-conv split/concat and 24 `ResBasicBlock`s across 6 stages
/// add up to a large node count). Production tuning (thread count, backend
/// selection) lands with the `SpeakerEmbedder` impl -- see `HANDOFF.md`.
pub(crate) fn runner_config() -> GgmlCpuGraphConfig {
    let graph_size = 1usize << 18;
    GgmlCpuGraphConfig {
        context_bytes: GgmlCpuGraphConfig::metadata_context_bytes(graph_size),
        graph_size,
        n_threads: None,
        backend: GgmlCpuGraphBackend::Cpu,
        use_scheduler: true,
    }
}

pub(crate) fn arena_context_bytes() -> usize {
    // Comfortably covers every weight tensor's metadata (not data, which lives
    // in the backend buffer); 1<<16 tensors is far more than the backbone's
    // actual count (a few thousand across 6 stages + stem + heads).
    GgmlCpuGraphConfig::metadata_context_bytes(1usize << 16)
}

/// Runtime entry point for the `SpeakerEmbedder` trait impl
/// (`super::super::RedimNet2Embedder`): owns the pack's parsed weights and
/// runs the full backbone forward on demand.
///
/// Mirrors the ASR families' pack-to-graph convention (e.g.
/// `models::dolphin::executor::encode_dolphin_encoder_from_pack`): the parsed
/// `Weights` are held across calls (avoids re-reading/re-parsing the `.oasr`
/// file from disk on every embed), but the ggml runner/arena/graph are
/// rebuilt fresh per call -- the same shape as every `#[ignore]`d parity test
/// in this module's `run_forward`. Caching the arena/graph across calls is a
/// later perf optimization (HANDOFF.md plan item 5), not attempted here.
pub(crate) struct RedimNet2Model {
    weights: Weights,
}

impl RedimNet2Model {
    pub(crate) fn from_oasr(path: &std::path::Path) -> Result<Self, RedimNetBackboneError> {
        let weights = Weights::from_oasr(path)?;
        Ok(Self { weights })
    }

    pub(crate) fn embedding_dim(&self) -> usize {
        config::EMBED_DIM
    }

    /// Runs `stem -> ... -> linear` on `feats` (the front end's
    /// `[mel*frames+frame]` flat buffer, matching `RedimNetFrontend::forward`'s
    /// output layout verbatim) and returns the raw (pre-L2-normalize) 192-d
    /// embedding. Callers needing a normalized embedding (the `SpeakerEmbedder`
    /// trait contract) normalize on top, same as `WeSpeakerEmbedder`.
    pub(crate) fn forward(
        &self,
        feats: &[f32],
        frames: usize,
    ) -> Result<Vec<f32>, RedimNetBackboneError> {
        if frames < config::TIME_STRIDE {
            return Err(shape_err(format!(
                "redimnet backbone needs at least {} frames (TIME_STRIDE) to produce any output, got {frames}",
                config::TIME_STRIDE
            )));
        }
        if feats.len() != frames * config::F {
            return Err(shape_err(format!(
                "redimnet backbone expected {} spec values for {frames} frames at {} mel bins, got {}",
                frames * config::F,
                config::F,
                feats.len()
            )));
        }

        let mut runner = GgmlCpuGraphRunner::new(runner_config())?;
        let arena = runner.start_static_tensor_arena(arena_context_bytes())?;
        let mut builder = WBuilder::new(&self.weights);
        let w = load_weights(&mut builder, &arena)?;
        let mut arena = arena;
        builder.upload(&mut arena)?;

        let mut graph = runner.start_graph();
        let spec = graph.new_tensor_2d_f32(frames, config::F, "redimnet_spec_input")?;
        let taps = forward(&mut graph, spec, frames, &w)?;
        graph.set_input(spec)?;
        graph.set_output(taps.embedding)?;
        graph.prepare_outputs_for_upload(&[taps.embedding])?;
        graph.set_f32_slice(spec, feats, "redimnet_spec_input")?;
        let embedding = graph.compute_output_f32(taps.embedding, config::EMBED_DIM)?;
        Ok(embedding)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};

    /// Synthetic (no pack/arena needed) check of `to1d`'s `ne` derivation
    /// (risk #1): builds a tiny `ne=[T=2,F=3,C=2,N=1]` tensor with
    /// `x[c,f,t] = c*100+f*10+t`, runs `to1d`, and verifies the output against
    /// the hand-derived formula `cf = f*C+c` directly (independent of the
    /// pack/golden dumps, to localize whether a full-pipeline mismatch is this
    /// reshape or something else).
    #[test]
    fn to1d_matches_hand_derived_frequency_major_formula() {
        let (t, f, c) = (2usize, 3usize, 2usize);
        let mut data = vec![0.0f32; t * f * c];
        for cc in 0..c {
            for ff in 0..f {
                for tt in 0..t {
                    let idx = tt + ff * t + cc * t * f; // ne=[T,F,C] flat order.
                    data[idx] = (cc * 100 + ff * 10 + tt) as f32;
                }
            }
        }
        let mut runner = GgmlCpuGraphRunner::new(runner_config()).expect("runner");
        let mut graph = runner.start_graph();
        let x = graph
            .new_tensor_4d_f32(t, f, c, 1, "to1d_test_input")
            .expect("input tensor");
        let out = ops::to1d(&graph, x, c, f, t).expect("to1d");
        graph.set_input(x).expect("set_input");
        graph.set_output(out).expect("set_output");
        graph
            .prepare_outputs_for_upload(&[out])
            .expect("prepare_outputs");
        graph
            .set_f32_slice(x, &data, "to1d_test_input")
            .expect("upload");
        let result = graph.compute_output_f32(out, t * f * c).expect("compute");

        for cf in 0..(c * f) {
            let cc = cf % c;
            let ff = cf / c;
            for tt in 0..t {
                let expected = (cc * 100 + ff * 10 + tt) as f32;
                let actual = result[tt + cf * t];
                assert_eq!(
                    actual, expected,
                    "to1d[t={tt},cf={cf} (c={cc},f={ff})]: got {actual}, want {expected}"
                );
            }
        }
    }

    fn spike_root() -> PathBuf {
        match crate::testing::external_test_fixture_path(
            "OPENASR_REDIMNET_SPIKE_ROOT",
            "ReDimNet parity fixture directory",
        ) {
            Ok(path) => path,
            Err(skip) => {
                eprintln!("skipping: {skip}");
                PathBuf::new()
            }
        }
    }

    fn f32_pack_path() -> PathBuf {
        spike_root().join("redimnet2-b6-f32.oasr")
    }

    fn stage_dump_dir() -> PathBuf {
        spike_root().join("stage_dump_b6_jfk")
    }

    fn embeddings_dir() -> PathBuf {
        spike_root().join("embeddings_b6")
    }

    /// Plain C-order f32 `.npy` loader (no fortran-order handling -- none of
    /// the backbone golden dumps are fortran-order, unlike the frontend's mel
    /// matrix). Returns `(shape, flat row-major data)`.
    fn load_npy_f32(path: &Path) -> (Vec<usize>, Vec<f32>) {
        let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
        assert_eq!(&bytes[..6], b"\x93NUMPY", "npy magic");
        let major = bytes[6];
        let header_len = if major == 1 {
            u16::from_le_bytes(bytes[8..10].try_into().unwrap()) as usize
        } else {
            u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize
        };
        let header_start = if major == 1 { 10 } else { 12 };
        let header = std::str::from_utf8(&bytes[header_start..header_start + header_len]).unwrap();
        assert!(header.contains("'<f4'"), "expected <f4 npy, got {header}");
        assert!(
            !header.contains("'fortran_order': True"),
            "unexpected fortran-order npy: {path:?}"
        );
        let shape_start = header.find("'shape':").expect("shape key");
        let paren = header[shape_start..].find('(').unwrap() + shape_start;
        let close = header[paren..].find(')').unwrap() + paren;
        let shape: Vec<usize> = header[paren + 1..close]
            .split(',')
            .filter_map(|s| s.trim().parse::<usize>().ok())
            .collect();
        let data_start = header_start + header_len;
        let values: Vec<f32> = bytes[data_start..]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        (shape, values)
    }

    /// `(max abs, mean abs)` diff -- cross-implementation fp32 parity is not
    /// bit-exact (different reduction order), so gate a wide max and a tight
    /// mean, matching the frontend/firered-aed convention.
    fn diff(actual: &[f32], expected: &[f32]) -> (f32, f32) {
        assert_eq!(actual.len(), expected.len(), "length mismatch");
        let mut max = 0.0f32;
        let mut sum = 0.0f64;
        for (a, e) in actual.iter().zip(expected.iter()) {
            let d = (a - e).abs();
            max = max.max(d);
            sum += d as f64;
        }
        (max, (sum / actual.len().max(1) as f64) as f32)
    }

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
        let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        let nb: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        (dot / (na * nb)) as f32
    }

    /// Builds a runner + arena + loaded weights against the f32 pack fixture,
    /// runs the full forward for `sample`'s dumped `00_spec_output.npy`, and
    /// returns every tap alongside the golden values needed to check it.
    /// `#[ignore]`d callers gate individual taps; see `full_pipeline_...` for
    /// the end-to-end run.
    fn run_forward(sample: &str) -> (Vec<Vec<f32>>, Vec<(&'static str, usize)>) {
        let weights = Weights::from_oasr(&f32_pack_path()).expect("load f32 pack");
        let mut runner = GgmlCpuGraphRunner::new(runner_config()).expect("runner");
        let arena = runner
            .start_static_tensor_arena(arena_context_bytes())
            .expect("arena");
        let mut builder = WBuilder::new(&weights);
        let w = load_weights(&mut builder, &arena).expect("load_weights");
        let mut arena = arena;
        builder.upload(&mut arena).expect("upload weights");

        // `frontend_dump/{sample}_04_cmn_output.npy` is the front end's final
        // output (log-mel + CMN), i.e. exactly the backbone's `spec` input --
        // the same tensor `stage_dump_b6_jfk/00_spec_output.npy` holds for
        // jfk specifically. Using the frontend dump (present for all three
        // fixture samples) lets every sample run through the same path,
        // rather than only jfk having a `00_spec_output.npy`.
        let (spec_shape, spec_data) = load_npy_f32(
            &spike_root()
                .join("frontend_dump")
                .join(format!("{sample}_04_cmn_output.npy")),
        );
        let (f_dim, t_raw) = (spec_shape[0], spec_shape[1]);
        assert_eq!(f_dim, config::F, "spec front-end mel bins");

        let mut graph = runner.start_graph();
        let spec = graph
            .new_tensor_2d_f32(t_raw, f_dim, "redimnet_spec_input")
            .expect("spec tensor");
        let taps = forward(&mut graph, spec, t_raw, &w).expect("forward");

        let cf = config::AGG_CHANNELS;
        let t_full = taps.t_full;
        let final_f = config::F / config::FREQ_STRIDE;
        let mut all_taps: Vec<GgmlCpuTensor> = taps.outputs_1d.clone();
        all_taps.push(taps.fin_wght1d);
        all_taps.push(taps.backbone_2d);
        all_taps.push(taps.pre_pool_flat);
        all_taps.push(taps.post_pool);
        all_taps.push(taps.post_bn);
        all_taps.push(taps.embedding);

        graph
            .set_input(spec)
            .and_then(|_| {
                for t in &all_taps {
                    graph.set_output(*t)?;
                }
                Ok(())
            })
            .expect("mark input/outputs");
        graph
            .prepare_outputs_for_upload(&all_taps)
            .expect("prepare_outputs");
        graph
            .set_f32_slice(spec, &spec_data, "redimnet_spec_input")
            .expect("upload spec");

        let mut specs: Vec<(GgmlCpuTensor, usize)> =
            taps.outputs_1d.iter().map(|t| (*t, t_full * cf)).collect();
        specs.push((taps.fin_wght1d, t_full * cf));
        specs.push((taps.backbone_2d, t_full * config::OUT_CHANNELS * final_f));
        specs.push((taps.pre_pool_flat, t_full * config::PRE_POOL_CHANNELS));
        specs.push((taps.post_pool, config::POOL_OUT_DIM));
        specs.push((taps.post_bn, config::POOL_OUT_DIM));
        specs.push((taps.embedding, config::EMBED_DIM));

        let names: Vec<(&'static str, usize)> = vec![
            ("01_outputs_1d(stem)", t_full * cf),
            ("02_outputs_1d(stage0)", t_full * cf),
            ("03_outputs_1d(stage1)", t_full * cf),
            ("04_outputs_1d(stage2)", t_full * cf),
            ("05_outputs_1d(stage3)", t_full * cf),
            ("06_outputs_1d(stage4)", t_full * cf),
            ("07_outputs_1d(stage5)", t_full * cf),
            ("08_outputs_1d(fin_wght1d)", t_full * cf),
            (
                "99_backbone_2d_output",
                t_full * config::OUT_CHANNELS * final_f,
            ),
            ("a0_pre_pool_flattened", t_full * config::PRE_POOL_CHANNELS),
            ("a1_post_pool", config::POOL_OUT_DIM),
            ("a2_post_bn", config::POOL_OUT_DIM),
            ("a3_final_embedding", config::EMBED_DIM),
        ];

        let results = graph.compute_outputs_f32(&specs).expect("compute");
        (results, names)
    }

    /// Stem parity: pins `01_outputs_1d` (`Conv2d -> LayerNorm -> to1d ->
    /// GroupNorm`). Tolerance: wide max / tight mean (cumulative f32,
    /// cross-implementation, per the frontend/firered-aed convention).
    #[test]
    #[ignore = "requires local redimnet2-spike assets under tmp/ (not committed)"]
    fn stem_parity_jfk() {
        if !f32_pack_path().exists() {
            eprintln!("skip: {:?} not present", f32_pack_path());
            return;
        }
        let (results, names) = run_forward("jfk");
        let (_, expected) = load_npy_f32(&stage_dump_dir().join("01_outputs_1d.npy"));
        let (max, mean) = diff(&results[0], &expected);
        println!("stem {}: max {max:.3e} mean {mean:.3e}", names[0].0);
        assert!(max < 1e-1, "stem max diff too large: {max:.3e}");
        assert!(mean < 1e-3, "stem mean diff too large: {mean:.3e}");
    }

    /// Stage parity: pins `02..07_outputs_1d` (stage0..5), each depending on
    /// every previously-verified stage via `weigth1d` aggregation, so a bug in
    /// an earlier stage surfaces here as a failure at that stage's index, not
    /// beyond.
    #[test]
    #[ignore = "requires local redimnet2-spike assets under tmp/ (not committed)"]
    fn stage_parity_jfk() {
        if !f32_pack_path().exists() {
            eprintln!("skip: {:?} not present", f32_pack_path());
            return;
        }
        let (results, _names) = run_forward("jfk");
        for (i, stage_file) in [
            "02_outputs_1d",
            "03_outputs_1d",
            "04_outputs_1d",
            "05_outputs_1d",
            "06_outputs_1d",
            "07_outputs_1d",
        ]
        .iter()
        .enumerate()
        {
            let (_, expected) = load_npy_f32(&stage_dump_dir().join(format!("{stage_file}.npy")));
            let (max, mean) = diff(&results[1 + i], &expected);
            println!("stage{i} ({stage_file}): max {max:.3e} mean {mean:.3e}");
            assert!(max < 2e-1, "stage{i} max diff too large: {max:.3e}");
            assert!(mean < 5e-3, "stage{i} mean diff too large: {mean:.3e}");
        }
    }

    /// `fin_wght1d` (7-way softmax aggregation) + `head`/`fin_to2d` (`99_
    /// backbone_2d_output`, `a0_pre_pool_flattened`).
    #[test]
    #[ignore = "requires local redimnet2-spike assets under tmp/ (not committed)"]
    fn fin_and_head_parity_jfk() {
        if !f32_pack_path().exists() {
            eprintln!("skip: {:?} not present", f32_pack_path());
            return;
        }
        let (results, _names) = run_forward("jfk");
        let (_, fin_expected) = load_npy_f32(&stage_dump_dir().join("08_outputs_1d.npy"));
        let (max, mean) = diff(&results[7], &fin_expected);
        println!("fin_wght1d: max {max:.3e} mean {mean:.3e}");
        assert!(max < 2e-1 && mean < 5e-3, "fin_wght1d diverged");

        let (_, backbone2d_expected) =
            load_npy_f32(&stage_dump_dir().join("99_backbone_2d_output.npy"));
        let (max, mean) = diff(&results[8], &backbone2d_expected);
        println!("backbone_2d: max {max:.3e} mean {mean:.3e}");
        assert!(max < 2e-1 && mean < 5e-3, "backbone_2d diverged");

        let (_, flat_expected) = load_npy_f32(&stage_dump_dir().join("a0_pre_pool_flattened.npy"));
        let (max, mean) = diff(&results[9], &flat_expected);
        println!("pre_pool_flat: max {max:.3e} mean {mean:.3e}");

        assert!(max < 2e-1 && mean < 5e-3, "pre_pool_flat diverged");
    }

    /// ASTP pool -> BN -> linear (`a1`/`a2`/`a3`), then the end-to-end cosine
    /// gate against the golden embeddings for all three fixture samples.
    #[test]
    #[ignore = "requires local redimnet2-spike assets under tmp/ (not committed)"]
    fn full_pipeline_cosine_gate() {
        if !f32_pack_path().exists() {
            eprintln!("skip: {:?} not present", f32_pack_path());
            return;
        }
        let (results, _names) = run_forward("jfk");
        let (_, pool_expected) = load_npy_f32(&stage_dump_dir().join("a1_post_pool.npy"));
        let (max, mean) = diff(&results[10], &pool_expected);
        println!("post_pool: max {max:.3e} mean {mean:.3e}");
        assert!(max < 2e-1 && mean < 5e-3, "post_pool diverged");

        let (_, bn_expected) = load_npy_f32(&stage_dump_dir().join("a2_post_bn.npy"));
        let (max, mean) = diff(&results[11], &bn_expected);
        println!("post_bn: max {max:.3e} mean {mean:.3e}");
        assert!(max < 2e-1 && mean < 5e-3, "post_bn diverged");

        let (_, emb_expected) = load_npy_f32(&stage_dump_dir().join("a3_final_embedding.npy"));
        let (max, mean) = diff(&results[12], &emb_expected);
        println!("embedding: max {max:.3e} mean {mean:.3e}");
        assert!(max < 2e-1 && mean < 5e-3, "embedding diverged");

        for sample in ["jfk", "zh_sample", "en_zh_mixed"] {
            let (results, _names) = run_forward(sample);
            let (_, golden) = load_npy_f32(&embeddings_dir().join(format!("{sample}.npy")));
            let cos = cosine(&results[12], &golden);
            println!("{sample}: cosine vs golden embedding = {cos:.6}");
            assert!(cos > 0.9999, "{sample} cosine too low: {cos}");
        }
    }
}
