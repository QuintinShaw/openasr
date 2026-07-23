//! Shared ggml building blocks for the ReDimNet2-B6 backbone graph.
//!
//! Every op here is a thin composition of the existing `GgmlCpuGraphBuilder`
//! primitives (no new ggml infrastructure) plus the shared `nn::attn`/`nn::norm`/
//! `nn::ffn` helpers already used by other families. Two tensor conventions are
//! used throughout the backbone (see `docs/design/redimnet2-b6-embedder.md`
//! risk #1):
//!
//!   * **2D** (`stem`/`ConvBlock2d`/down-conv): ggml `ne = [T, F, C, N]`, forced
//!     by `conv_2d`'s own `[W, H, Cin, N]` convention (torch `(N,C,F,T)` reversed,
//!     time is `W`, freq is `H` -- see the stem's torch input `(bs,1,F,T)`).
//!   * **1D interchange** (`outputs_1d`, `weigth1d`, `to1d`/`to2d` boundary,
//!     `GroupNorm`): ggml `ne = [T, CF]` (`T` innermost). This is chosen so it
//!     matches the golden `.npy` dumps' flat memory order byte-for-byte (numpy
//!     `(CF, T)` row-major has `T` fastest), so parity taps need zero transpose.
//!   * **1D internal-only** (inside a `TimeContextBlock1d`/ASTP): transiently
//!     transposed to ggml `ne = [C, T]` (channel innermost, matching every other
//!     family's convention) so the shared `nn::attn`/mul_mat-based linear helpers
//!     apply unchanged. Callers transpose in at the block's entry and back out at
//!     its exit; nothing outside the block ever sees this layout.

use crate::ggml_runtime::{GgmlCpuGraphBuilder, GgmlCpuGraphError, GgmlCpuTensor};
use crate::nn::attn::{
    AttentionHeadLayout, AttentionReshapeSteps, AttentionValueMergeSteps,
    STANDARD_HEAD_PERMUTE_AXES, attention_context_from_probs,
    reshape_projection_to_attention_heads,
};
use crate::nn::ffn::{
    FeedForwardActivation, FeedForwardResidualSteps, apply_feed_forward_residual,
};
use crate::nn::norm::{AffineLayerNormSteps, apply_affine_layer_norm};

pub(super) type OpResult<'a> = Result<GgmlCpuTensor<'a>, GgmlCpuGraphError>;

/// Identity error mapper kept only so every op below reads uniformly with the
/// other families' `ggml_err(stage)` call-site pattern (`.map_err(m)`);
/// `GgmlCpuGraphError` (this crate's shared ggml error type, unlike the
/// per-family wrapper enums) needs no stage tagging of its own.
fn map_err(_stage: &'static str) -> impl Fn(GgmlCpuGraphError) -> GgmlCpuGraphError + Copy {
    |source| source
}

/// `to1d`: `(bs,c,f,t) -> permute(0,2,1,3) -> reshape(bs,c*f,t)` (torch), i.e. a
/// frequency-major flatten (`cf = f*C + c`). Input ggml `ne=[T,F,C,1]`, output
/// `ne=[T, C*F]`.
///
/// Derivation: torch's `permute(0,2,1,3)` swaps dims 1,2 (`c`,`f`); the merge
/// keeps `f` slower/outer and `c` faster/inner in the flattened axis (`cf =
/// f*C+c`). In ggml that means, starting from `[T,F,C,N]` (`ne0=T,ne1=F,ne2=C,
/// ne3=N`), swap `ne1`<->`ne2` (source axis1(F)->dest2, axis2(C)->dest1) via
/// `permute(x,0,2,1,3)` to get `[T,C,F,N]`, `cont`, then merge `(ne1=C,ne2=F)`
/// into one axis of size `C*F` (merging keeps the lower axis, `C`, as the fast
/// component -- exactly `cf=f*C+c`).
pub(super) fn to1d<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    x2d: GgmlCpuTensor<'a>,
    c: usize,
    f: usize,
    t: usize,
) -> OpResult<'a> {
    let m = map_err("to1d");
    // Defensively `cont()` the input before taking any view of it (`permute`
    // is itself a view op): if the caller's `x2d` is *also* independently
    // marked `set_output` (e.g. a parity tap read back later, like
    // `backbone_2d`/`head_out` feeding this exact `to1d` call for
    // `pre_pool_flat`), the backend scheduler's gallocr can otherwise recycle
    // its buffer once the view is consumed, corrupting every later read of
    // the ORIGINAL (tap) tensor even though it is still needed. Root-caused
    // via `debug_stem_conv_channel_norms`/`fin_and_head_parity_jfk`: the
    // divergence only ever showed up on tensors that were both (a) marked as
    // an output and (b) fed as a *view* source into further computation.
    let x2d = graph.cont(x2d).map_err(m)?;
    let permuted = graph.permute(x2d, 0, 2, 1, 3).map_err(m)?;
    let cont = graph.cont(permuted).map_err(m)?;
    let reshaped = graph.reshape_2d(cont, t, c * f).map_err(m)?;
    // `reshape_2d` is a view (shares `cont`'s buffer); `to1d`'s output is
    // often a long-lived value (an `outputs_1d` entry, or `pre_pool_flat`,
    // read back much later), and the backend scheduler's gallocr does not
    // protect a view's underlying buffer purely from the view being marked
    // `set_output` -- see `group_norm_1d`'s identical fix for the full
    // root-cause writeup. `cont` here materializes an independent buffer.
    graph.cont(reshaped).map_err(m)
}

/// `to2d`: inverse of [`to1d`]. Input `ne=[T,C*F]`, output `ne=[T,F,C,1]`.
pub(super) fn to2d<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    x1d: GgmlCpuTensor<'a>,
    c: usize,
    f: usize,
    t: usize,
) -> OpResult<'a> {
    let m = map_err("to2d");
    // See `to1d`'s identical defensive `cont()`: `x1d` may itself be an
    // independently `set_output`-marked tap (e.g. `fin_wght1d`), and
    // `reshape_4d` is a view op.
    let x1d = graph.cont(x1d).map_err(m)?;
    // Split `cf` into `(c fast/inner, f slow/outer)`, matching to1d's merge.
    let split = graph.reshape_4d(x1d, t, c, f, 1).map_err(m)?;
    let permuted = graph.permute(split, 0, 2, 1, 3).map_err(m)?;
    graph.cont(permuted).map_err(m)
}

/// The backbone's *final* pre-pool flatten (`ReDimNet2Wrap.forward`: `bs, C,
/// F, T = out.size(); out = out.reshape(bs, C*F, T)`). This is a **plain**
/// torch reshape, NOT `to1d()` -- a subtly different merge order than every
/// other 2D<->1D reshape in this backbone, and worth spelling out explicitly
/// rather than reusing `to1d` (an earlier version of this code did exactly
/// that and silently flattened in the wrong order, see
/// `fin_and_head_parity_jfk`'s diagnostic history).
///
/// Torch merges adjacent dims `(C,F)` with `C` (the earlier/outer dim) slower
/// and `F` (the later/inner dim) faster: `cf_torch = c*F + f`. Input ggml
/// `ne=[T,F,C,N]` (`F` already sits at `ne1`, `C` at `ne2` -- no permute
/// needed, unlike `to1d`/`to2d` which swap them): merging `ne1=F` (fast) and
/// `ne2=C` (slow) directly gives exactly `f + c*F = cf_torch`.
pub(super) fn flatten_backbone_output<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    x2d: GgmlCpuTensor<'a>,
    c: usize,
    f: usize,
    t: usize,
) -> OpResult<'a> {
    let m = map_err("flatten_backbone_output");
    // Defensive `cont()` for the same reason as `to1d`/`to2d`: `x2d` here is
    // `head_out`, itself an independently `set_output`-marked tap
    // (`backbone_2d`).
    let x2d = graph.cont(x2d).map_err(m)?;
    let merged = graph.reshape_3d(x2d, t, f * c, 1).map_err(m)?;
    let flat = graph.reshape_2d(merged, t, f * c).map_err(m)?;
    graph.cont(flat).map_err(m)
}

/// Channels-first LayerNorm on a 2D tensor (`ne=[T,F,C,N]`, normalize per
/// `(t,f)` position over `C`). `C` sits on `ne2`, so this permutes it to `ne0`
/// (`norm` reduces over `ne0`), normalizes, applies the affine, and permutes
/// back. `weight`/`bias` are `ne=[C]`.
pub(super) fn layernorm_channels_first_2d<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    x2d: GgmlCpuTensor<'a>,
    weight: GgmlCpuTensor<'a>,
    bias: GgmlCpuTensor<'a>,
    eps: f32,
) -> OpResult<'a> {
    let m = map_err("ln_channels_first_2d");
    // [T,F,C,N] -> [C,T,F,N]: source axis0(T)->1, axis1(F)->2, axis2(C)->0.
    let permuted = graph.permute(x2d, 1, 2, 0, 3).map_err(m)?;
    let cfirst = graph.cont(permuted).map_err(m)?;
    let normed = apply_affine_layer_norm(
        graph,
        cfirst,
        eps,
        weight,
        bias,
        AffineLayerNormSteps {
            norm: "ln2d_norm",
            scale: "ln2d_scale",
            bias: "ln2d_bias",
        },
        |_s, source| source,
    )?;
    // [C,T,F,N] -> [T,F,C,N]: source axis0(C)->2, axis1(T)->0, axis2(F)->1.
    let back = graph.permute(normed, 2, 0, 1, 3).map_err(m)?;
    graph.cont(back).map_err(m)
}

/// GroupNorm over the 1D interchange tensor (`ne=[T,CF]`, channel axis is `CF`,
/// split into `n_groups`). ggml's `group_norm` expects channel on `ne2` with
/// `ne0*ne1` as the reduced "spatial" size; reinterpreting `[T,CF]` (contiguous,
/// `T` fastest) as `[1,T,CF,1]` is a data-preserving reshape (dummy `ne0=1`
/// spatial axis, `T` the real spatial axis, `CF` on `ne2`) that gives exactly
/// `torch.GroupNorm`'s per-group reduction over `(channels_per_group * T)`.
/// `weight`/`bias` are `ne=[CF]`.
pub(super) fn group_norm_1d<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    x_tc: GgmlCpuTensor<'a>,
    t: usize,
    cf: usize,
    weight: GgmlCpuTensor<'a>,
    bias: GgmlCpuTensor<'a>,
    n_groups: usize,
    eps: f32,
) -> OpResult<'a> {
    let m = map_err("group_norm_1d");
    let x4d = graph.reshape_4d(x_tc, 1, t, cf, 1).map_err(m)?;
    let normed = graph.group_norm(x4d, n_groups, eps).map_err(m)?;
    let weight4d = graph.reshape_4d(weight, 1, 1, cf, 1).map_err(m)?;
    let bias4d = graph.reshape_4d(bias, 1, 1, cf, 1).map_err(m)?;
    let scaled = graph.mul(normed, weight4d).map_err(m)?;
    let biased = graph.add(scaled, bias4d).map_err(m)?;
    let reshaped = graph.reshape_2d(biased, t, cf).map_err(m)?;
    // `reshape_2d` returns a *view* (no independent allocation, shares
    // `biased`'s buffer). This tensor becomes a long-lived `outputs_1d` entry
    // (read back much later by `weigth1d`/`fin_wght1d`, and as a parity tap),
    // and the backend scheduler's gallocr does not protect a view node's
    // underlying buffer purely from the view itself being marked `set_output`
    // -- it can still recycle `biased`'s allocation once the view's own
    // last-recorded consumer is done, silently corrupting every later read of
    // this "stage output" (root-caused via `debug_stem_conv_channel_norms`:
    // the raw ops chain matched the golden dump exactly in isolation, but
    // reading the same tensor back after the graph grew past it did not).
    // `cont` materializes a genuinely independent buffer, which is safe to
    // keep alive for the rest of the graph's lifetime.
    graph.cont(reshaped).map_err(m)
}

/// Precomputed per-channel affine (`scale`,`shift`) for an eval-mode BatchNorm:
/// `y = gamma*(x-mean)/sqrt(var+eps) + beta = x*scale + shift`. Computed on the
/// host (not the graph) since the four BN tensors are static pack weights, not
/// request data.
pub(super) fn batchnorm_affine(
    gamma: &[f32],
    beta: &[f32],
    running_mean: &[f32],
    running_var: &[f32],
    eps: f32,
) -> (Vec<f32>, Vec<f32>) {
    let n = gamma.len();
    let mut scale = vec![0.0f32; n];
    let mut shift = vec![0.0f32; n];
    for i in 0..n {
        let s = gamma[i] / (running_var[i] + eps).sqrt();
        scale[i] = s;
        shift[i] = beta[i] - running_mean[i] * s;
    }
    (scale, shift)
}

/// Apply a precomputed per-channel affine to a 2D tensor (`ne=[T,F,C,N]`,
/// `scale`/`shift` are `ne=[C]`, broadcast via a `[1,1,C,1]` reshape).
pub(super) fn apply_channel_affine_2d<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    x2d: GgmlCpuTensor<'a>,
    c: usize,
    scale: GgmlCpuTensor<'a>,
    shift: GgmlCpuTensor<'a>,
) -> OpResult<'a> {
    let m = map_err("channel_affine_2d");
    let scale4d = graph.reshape_4d(scale, 1, 1, c, 1).map_err(m)?;
    let shift4d = graph.reshape_4d(shift, 1, 1, c, 1).map_err(m)?;
    let scaled = graph.mul(x2d, scale4d).map_err(m)?;
    graph.add(scaled, shift4d).map_err(m)
}

/// Apply a precomputed per-channel affine to a channel-inner 1D tensor
/// (`ne=[C,T]`); `scale`/`shift` are `ne=[C]` and broadcast directly (no
/// reshape needed: `C` is already `ne0`).
pub(super) fn apply_channel_affine_ct<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    x_ct: GgmlCpuTensor<'a>,
    scale: GgmlCpuTensor<'a>,
    shift: GgmlCpuTensor<'a>,
) -> OpResult<'a> {
    let m = map_err("channel_affine_ct");
    let scaled = graph.mul(x_ct, scale).map_err(m)?;
    graph.add(scaled, shift).map_err(m)
}

/// A 1x1 `Conv2d` (groups=1): a per-`(t,f)` linear map over the channel axis
/// (`ne2`). Implemented directly via `conv_2d` (no permute needed: ggml's
/// standard conv handles the full, non-grouped 1x1 case natively).
pub(super) fn conv1x1_2d<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    kernel: GgmlCpuTensor<'a>,
    bias: GgmlCpuTensor<'a>,
    x2d: GgmlCpuTensor<'a>,
    c_out: usize,
) -> OpResult<'a> {
    let m = map_err("conv1x1_2d");
    let conv = graph.conv_2d(kernel, x2d, 1, 1, 0, 0, 1, 1).map_err(m)?;
    let bias4d = graph.reshape_4d(bias, 1, 1, c_out, 1).map_err(m)?;
    graph.add(conv, bias4d).map_err(m)
}

/// A depthwise 3x3 `Conv2d` (`padding='same'`=1, `stride=1`, `groups=C`).
/// Kernel `ne=[3,3,1,C]`.
pub(super) fn depthwise_conv3x3_2d<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    kernel: GgmlCpuTensor<'a>,
    x2d: GgmlCpuTensor<'a>,
) -> OpResult<'a> {
    graph
        .depthwise_conv_2d(kernel, x2d, 1, 1, 1, 1, 1, 1)
        .map_err(map_err("dwconv3x3_2d"))
}

/// A `ResBasicBlock` (`ConvBlock2d`, `block_type="basic_resnet"`, `Gdiv=1`
/// so both inner convs are fully depthwise, `use_fwSE=False`): `conv1
/// (depthwise 3x3) -> conv1pw (1x1) -> relu -> bn1 -> conv2 (depthwise 3x3) ->
/// conv2pw (1x1) -> bn2 -> += residual -> relu`. `inc == outc` always in this
/// backbone (checkpoint never has a `downsample` branch), so the residual is
/// the identity input.
pub(super) struct ResBasicBlockWeights<'a> {
    pub conv1: GgmlCpuTensor<'a>,
    pub conv1pw_w: GgmlCpuTensor<'a>,
    pub conv1pw_b: GgmlCpuTensor<'a>,
    pub bn1_scale: GgmlCpuTensor<'a>,
    pub bn1_shift: GgmlCpuTensor<'a>,
    pub conv2: GgmlCpuTensor<'a>,
    pub conv2pw_w: GgmlCpuTensor<'a>,
    pub conv2pw_b: GgmlCpuTensor<'a>,
    pub bn2_scale: GgmlCpuTensor<'a>,
    pub bn2_shift: GgmlCpuTensor<'a>,
}

pub(super) fn resbasic_block_2d<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    x2d: GgmlCpuTensor<'a>,
    c: usize,
    w: &ResBasicBlockWeights<'a>,
) -> OpResult<'a> {
    let m = map_err("resbasic_block_2d");
    let out = depthwise_conv3x3_2d(graph, w.conv1, x2d)?;
    let out = conv1x1_2d(graph, w.conv1pw_w, w.conv1pw_b, out, c)?;
    let out = graph.relu(out).map_err(m)?;
    let out = apply_channel_affine_2d(graph, out, c, w.bn1_scale, w.bn1_shift)?;
    let out = depthwise_conv3x3_2d(graph, w.conv2, out)?;
    let out = conv1x1_2d(graph, w.conv2pw_w, w.conv2pw_b, out, c)?;
    let out = apply_channel_affine_2d(graph, out, c, w.bn2_scale, w.bn2_shift)?;
    let out = graph.add(out, x2d).map_err(m)?;
    graph.relu(out).map_err(m)
}

/// A grouped `Conv2d` with an arbitrary `groups` divisor (`compress_tconvs`
/// down-conv: `groups=gcd(c_in, block_channels)`, so neither the plain
/// (`groups=1`) nor the depthwise (`groups=c_in`) fast path applies in
/// general). Splits `data` along the channel axis (`ne2`) and `kernel` along
/// the output-channel axis (`ne3`) into `groups` contiguous slices, runs a
/// plain `conv_2d` per group, and concatenates the results back along `ne2`.
///
/// `kernel` is `ne=[kw,kh,cin_per_group,cout_total]` (contiguous, straight
/// from the arena upload); `data` is `ne=[t,f,cin_total,1]`. Both a channel
/// slice of `data` (a contiguous run of `cin_per_group` full `(t,f)` planes)
/// and a slice of `kernel` (a contiguous run of `cout_per_group` "cout"
/// blocks, each `kw*kh*cin_per_group` elements) are plain contiguous
/// sub-ranges of their parent tensors, so `view_4d` with the parent's own
/// natural strides (no gaps) is exact -- no `cont` needed before `conv_2d`
/// (which only requires the *kernel* contiguous, and the *data* to be f32,
/// both satisfied by a contiguous view).
#[allow(clippy::too_many_arguments)]
pub(super) fn grouped_conv2d<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    kernel: GgmlCpuTensor<'a>,
    bias: GgmlCpuTensor<'a>,
    x2d: GgmlCpuTensor<'a>,
    kw: usize,
    kh: usize,
    t: usize,
    f: usize,
    cin_total: usize,
    cout_total: usize,
    groups: usize,
    stride0: usize,
    stride1: usize,
) -> OpResult<'a> {
    let m = map_err("grouped_conv2d");
    let cin_per_group = cin_total / groups;
    let cout_per_group = cout_total / groups;
    let elem = std::mem::size_of::<f32>();

    let data_plane_bytes = t * f * elem; // one input channel's (t,f) plane.
    let data_group_nb2 = data_plane_bytes; // default stride for ne2 in the slice.
    let kernel_plane_elems = kw * kh * cin_per_group; // one "cout" block.
    let kernel_group_bytes = kernel_plane_elems * elem;

    let mut acc: Option<GgmlCpuTensor<'a>> = None;
    for g in 0..groups {
        let data_slice = graph
            .view_4d(
                x2d,
                t,
                f,
                cin_per_group,
                1,
                t * elem,
                data_group_nb2,
                data_group_nb2 * cin_per_group,
                g * cin_per_group * data_plane_bytes,
            )
            .map_err(m)?;
        let kernel_slice = graph
            .view_4d(
                kernel,
                kw,
                kh,
                cin_per_group,
                cout_per_group,
                kw * elem,
                kw * kh * elem,
                kernel_group_bytes,
                g * cout_per_group * kernel_group_bytes,
            )
            .map_err(m)?;
        let out = graph
            .conv_2d(kernel_slice, data_slice, stride0, stride1, 0, 0, 1, 1)
            .map_err(m)?;
        acc = Some(match acc {
            None => out,
            Some(prev) => graph.concat(prev, out, 2).map_err(m)?,
        });
    }
    let conv = acc.expect("groups >= 1");
    let bias4d = graph.reshape_4d(bias, 1, 1, cout_total, 1).map_err(m)?;
    graph.add(conv, bias4d).map_err(m)
}

/// A depthwise `Conv1d` over the channel-inner internal convention (`ne=
/// [C,T]`): mirrors `models::dolphin::encoder_graph`'s `depthwise_conv1d`
/// (transpose to bring time to the spatial axis, run the fused
/// `conv_2d_dw_direct` op, permute/cont back). `kernel` is `ne=[k,1,1,C]`,
/// `bias` is `ne=[C]`. `padding` is symmetric (`k/2` for the odd kernels used
/// here).
#[allow(clippy::too_many_arguments)]
pub(super) fn depthwise_conv1d_ct<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    x_ct: GgmlCpuTensor<'a>,
    kernel: GgmlCpuTensor<'a>,
    bias: GgmlCpuTensor<'a>,
    channels: usize,
    t: usize,
    k: usize,
    padding: usize,
) -> OpResult<'a> {
    let m = map_err("dwconv1d_ct");
    // `kernel` is the pack's native `nn.Conv1d(C,C,k,groups=C)` weight, torch
    // `(C,1,k)` -> ggml `ne=[k,1,C]` (rank-3, no redundant leading `1` axis
    // like the 2D depthwise kernels have); insert it before the fused
    // `conv_2d_dw_direct` op, which wants `[k,1,1,C]`.
    let kernel4d = graph.reshape_4d(kernel, k, 1, 1, channels).map_err(m)?;
    let transposed = graph.transpose(x_ct).map_err(m)?;
    let transposed = graph.cont(transposed).map_err(m)?;
    let as_4d = graph.reshape_4d(transposed, t, 1, channels, 1).map_err(m)?;
    let conv = graph
        .depthwise_conv_2d(kernel4d, as_4d, 1, 1, padding, 0, 1, 1)
        .map_err(m)?;
    let conv = graph.permute(conv, 1, 2, 0, 3).map_err(m)?;
    let conv = graph.cont(conv).map_err(m)?;
    graph.add(conv, bias).map_err(m)
}

/// A 1x1 `Conv1d` (`= Linear`) over the channel-inner convention (`ne=[C,T]`):
/// `mul_mat(weight, x) + bias`. `weight` is `ne=[cin,cout]`, `bias` is
/// `ne=[cout]`.
pub(super) fn linear_ct<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    weight: GgmlCpuTensor<'a>,
    bias: GgmlCpuTensor<'a>,
    x_ct: GgmlCpuTensor<'a>,
) -> OpResult<'a> {
    let m = map_err("linear_ct");
    let projected = graph.mul_mat(weight, x_ct).map_err(m)?;
    graph.add(projected, bias).map_err(m)
}

/// Channels-first LayerNorm on the channel-inner convention (`ne=[C,T]`): `C`
/// is already `ne0`, so this is a direct `norm` + affine (no permute needed).
pub(super) fn layernorm_ct<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    x_ct: GgmlCpuTensor<'a>,
    weight: GgmlCpuTensor<'a>,
    bias: GgmlCpuTensor<'a>,
    eps: f32,
) -> OpResult<'a> {
    apply_affine_layer_norm(
        graph,
        x_ct,
        eps,
        weight,
        bias,
        AffineLayerNormSteps {
            norm: "ln_ct_norm",
            scale: "ln_ct_scale",
            bias: "ln_ct_bias",
        },
        |_s, source| source,
    )
}

/// A `ConvNeXtLikeBlock(dim=1, kernel_sizes=[k], Gdiv=1, norm='bn',
/// norm_placement='mid', activation='gelu')` step inside a TCM: `skip=x;
/// x=dwconv(x); x=bn(x); x=gelu_erf(x); x=pwconv1(x); return skip+x`. Torch's
/// `nn.GELU()` default is the exact erf-based GELU (not the tanh
/// approximation), hence `gelu_erf` here (vs the transformer FFN's tanh-approx
/// `gelu_new`, see `transformer_encoder_layer_ct`).
#[allow(clippy::too_many_arguments)]
pub(super) fn convnext_dw_block_ct<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    x_ct: GgmlCpuTensor<'a>,
    dwconv_kernel: GgmlCpuTensor<'a>,
    dwconv_bias: GgmlCpuTensor<'a>,
    bn_scale: GgmlCpuTensor<'a>,
    bn_shift: GgmlCpuTensor<'a>,
    pwconv_w: GgmlCpuTensor<'a>,
    pwconv_b: GgmlCpuTensor<'a>,
    channels: usize,
    t: usize,
    kernel_size: usize,
) -> OpResult<'a> {
    let m = map_err("convnext_dw_block_ct");
    let padding = kernel_size / 2;
    let out = depthwise_conv1d_ct(
        graph,
        x_ct,
        dwconv_kernel,
        dwconv_bias,
        channels,
        t,
        kernel_size,
        padding,
    )?;
    let out = apply_channel_affine_ct(graph, out, bn_scale, bn_shift)?;
    let out = graph.gelu_erf(out).map_err(m)?;
    let out = linear_ct(graph, pwconv_w, pwconv_b, out)?;
    graph.add(out, x_ct).map_err(m)
}

/// Weight handles for one `TransformerEncoderLayer` (`MultiHeadAttention` +
/// post-norm + `FeedForward` + post-norm), the `att` step of a TCM's `conv+att`
/// pipeline. Standard (non-relative) multi-head self-attention, 4 heads,
/// `n_mlp == hidden` (per `build_internal_1d_tcm_block`'s `conv+att` call).
pub(super) struct TransformerEncoderLayerWeights<'a> {
    pub q_w: GgmlCpuTensor<'a>,
    pub q_b: GgmlCpuTensor<'a>,
    pub k_w: GgmlCpuTensor<'a>,
    pub k_b: GgmlCpuTensor<'a>,
    pub v_w: GgmlCpuTensor<'a>,
    pub v_b: GgmlCpuTensor<'a>,
    pub out_w: GgmlCpuTensor<'a>,
    pub out_b: GgmlCpuTensor<'a>,
    pub layer_norm_w: GgmlCpuTensor<'a>,
    pub layer_norm_b: GgmlCpuTensor<'a>,
    pub ff_intermediate_w: GgmlCpuTensor<'a>,
    pub ff_intermediate_b: GgmlCpuTensor<'a>,
    pub ff_output_w: GgmlCpuTensor<'a>,
    pub ff_output_b: GgmlCpuTensor<'a>,
    pub final_layer_norm_w: GgmlCpuTensor<'a>,
    pub final_layer_norm_b: GgmlCpuTensor<'a>,
}

/// `TransformerEncoderLayer.forward` (post-norm, `channel_last=False` in torch
/// terms -- but our `ne0=channel` convention already IS what that permute
/// dance produces, so no permute is needed here):
/// ```text
/// x1  = x + MHA(x)
/// x1n = layer_norm(x1)
/// x2  = x1n + FFN(x1n)          # FeedForward(hidden_act='gelu_new')
/// out = final_layer_norm(x2)
/// ```
pub(super) fn transformer_encoder_layer_ct<'a>(
    graph: &mut GgmlCpuGraphBuilder<'a>,
    x_ct: GgmlCpuTensor<'a>,
    hidden: usize,
    t: usize,
    heads: usize,
    eps: f32,
    w: &TransformerEncoderLayerWeights<'a>,
) -> OpResult<'a> {
    let m = map_err("transformer_encoder_layer_ct");
    let head_dim = hidden / heads;

    let q = linear_ct(graph, w.q_w, w.q_b, x_ct)?;
    let k = linear_ct(graph, w.k_w, w.k_b, x_ct)?;
    let v = linear_ct(graph, w.v_w, w.v_b, x_ct)?;

    let layout = AttentionHeadLayout {
        head_dim,
        attention_heads: heads,
        sequence_len: t,
    };
    let steps = AttentionReshapeSteps {
        reshape: "mha_reshape",
        permute: "mha_permute",
        cont: "mha_cont",
    };
    let map_e = |_s: &'static str, source: GgmlCpuGraphError| source;
    let q_h = reshape_projection_to_attention_heads(
        graph,
        q,
        layout,
        STANDARD_HEAD_PERMUTE_AXES,
        false,
        steps,
        map_e,
    )?;
    let k_h = reshape_projection_to_attention_heads(
        graph,
        k,
        layout,
        STANDARD_HEAD_PERMUTE_AXES,
        false,
        steps,
        map_e,
    )?;
    let v_h = reshape_projection_to_attention_heads(
        graph,
        v,
        layout,
        STANDARD_HEAD_PERMUTE_AXES,
        true,
        steps,
        map_e,
    )?;

    let k_h_cont = graph.cont(k_h).map_err(m)?;
    let scores = graph.mul_mat(k_h_cont, q_h).map_err(m)?;
    let scores = graph
        .scale(scores, 1.0 / (head_dim as f32).sqrt())
        .map_err(m)?;
    let scores = graph.soft_max(scores).map_err(m)?;

    let context = attention_context_from_probs(
        graph,
        v_h,
        scores,
        layout,
        AttentionValueMergeSteps {
            value_permute: "mha_v_t",
            value_cont: "mha_v_t",
            context_mul: "mha_ctx",
            context_merge_permute: "mha_merge",
            context_merge_cont: "mha_merge",
            context_merge_reshape: "mha_merge",
        },
        map_e,
    )?;
    let attn_out = linear_ct(graph, w.out_w, w.out_b, context)?;

    let x1 = graph.add(x_ct, attn_out).map_err(m)?;
    let x1n = apply_affine_layer_norm(
        graph,
        x1,
        eps,
        w.layer_norm_w,
        w.layer_norm_b,
        AffineLayerNormSteps {
            norm: "mha_ln_norm",
            scale: "mha_ln_scale",
            bias: "mha_ln_bias",
        },
        |_s, source| source,
    )?;
    let x2 = apply_feed_forward_residual(
        graph,
        x1n,
        x1n,
        FeedForwardActivation::Gelu, // gelu_new (tanh-approx), matches ggml_gelu.
        None,
        FeedForwardResidualSteps {
            activation: "mha_ffn_act",
            scale: None,
            residual: "mha_ffn_residual",
        },
        |g, v| linear_ct(g, w.ff_intermediate_w, w.ff_intermediate_b, v),
        |g, v| linear_ct(g, w.ff_output_w, w.ff_output_b, v),
        |_s, source| source,
    )?;
    apply_affine_layer_norm(
        graph,
        x2,
        eps,
        w.final_layer_norm_w,
        w.final_layer_norm_b,
        AffineLayerNormSteps {
            norm: "mha_final_ln_norm",
            scale: "mha_final_ln_scale",
            bias: "mha_final_ln_bias",
        },
        |_s, source| source,
    )
}

/// `nn.Upsample(scale_factor=stt, mode='nearest')` along the time axis of a
/// canonical 1D tensor (`ne=[T_reduced,CF]`): `out[t,cf] = in[t/stt,cf]`
/// (each input step repeated `stt` times *contiguously*, not tiled). Insert a
/// new fastest axis of size 1 (`[1,T_reduced,CF,1]`, data-preserving reshape),
/// broadcast it to size `stt` via `repeat_4d` (tiling a size-1 axis is a
/// constant fill, i.e. exactly a broadcast copy), then merge `(stt,
/// T_reduced)` back into one axis with `stt` fast/inner -- giving `t_full =
/// t_reduced*stt + r` with `r` (`0..stt`) innermost, the nearest-repeat order.
pub(super) fn nearest_upsample_time<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    x_tc: GgmlCpuTensor<'a>,
    stt: usize,
    t_reduced: usize,
    cf: usize,
) -> OpResult<'a> {
    let m = map_err("nearest_upsample_time");
    if stt == 1 {
        return Ok(x_tc);
    }
    let x4d = graph.reshape_4d(x_tc, 1, t_reduced, cf, 1).map_err(m)?;
    let rep = graph.repeat_4d(x4d, stt, t_reduced, cf, 1).map_err(m)?;
    let reshaped = graph.reshape_2d(rep, stt * t_reduced, cf).map_err(m)?;
    graph.cont(reshaped).map_err(m)
}

/// `weigth1d(N,C=AGG_CHANNELS)`: a learned softmax-weighted sum of `N`
/// `ne=[T,CF]` feature maps, `w = softmax(param, dim=N)` (per-channel, NOT a
/// single scalar per map -- `fm_weigthing_type="NC"`). The softmax is static
/// (no request data), so it is computed on the host from the raw pack weight
/// and each `w[:,i]` slice is uploaded as its own `ne=[1,CF]` broadcast
/// tensor; the graph then does `sum_i xs[i] * w_i`.
pub(super) fn weigth1d_softmax_host(raw_w: &[f32], n: usize, cf: usize) -> Vec<Vec<f32>> {
    // raw_w is the pack's `*.w` tensor, logical torch shape (1,N,CF,1) ->
    // flat row-major (n-major, cf-minor): raw_w[i*cf + c].
    let mut out = vec![vec![0.0f32; cf]; n];
    for c in 0..cf {
        let mut max = f32::NEG_INFINITY;
        for i in 0..n {
            max = max.max(raw_w[i * cf + c]);
        }
        let mut sum = 0.0f64;
        let mut exps = vec![0.0f32; n];
        for i in 0..n {
            let e = (raw_w[i * cf + c] - max).exp();
            exps[i] = e;
            sum += e as f64;
        }
        for i in 0..n {
            out[i][c] = (exps[i] as f64 / sum) as f32;
        }
    }
    out
}

pub(super) fn weigth1d_apply<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    xs: &[GgmlCpuTensor<'a>],
    weights: &[GgmlCpuTensor<'a>],
) -> OpResult<'a> {
    let m = map_err("weigth1d_apply");
    assert_eq!(xs.len(), weights.len());
    let mut acc: Option<GgmlCpuTensor<'a>> = None;
    for (x, w) in xs.iter().zip(weights.iter()) {
        let weighted = graph.mul(*x, *w).map_err(m)?;
        acc = Some(match acc {
            None => weighted,
            Some(prev) => graph.add(prev, weighted).map_err(m)?,
        });
    }
    Ok(acc.expect("weigth1d_apply needs at least one input"))
}

/// `ASTP` pooling (`global_context_att=True`): `x` is `ne=[T,CF]` (canonical
/// 1D). Internally transposes to `ne=[CF,T]` (channel-inner, matching
/// `linear_ct`) since the attention scoring conv1d needs channel-innermost
/// mul_mat.
///
/// ```text
/// context_mean, context_std = mean_T(x), std_T(x)   # per-channel, broadcast over T
/// x_in   = cat([x, context_mean, context_std], channel)   # CF -> 3*CF
/// alpha  = softmax(linear2(tanh(linear1(x_in))), dim=T)
/// mean   = sum_T(alpha * x)
/// var    = sum_T(alpha * x^2) - mean^2
/// std    = sqrt(clamp(var, 1e-7))
/// return cat([mean, std])                            # ne=[2*CF]
/// ```
/// `eps_scalar` is a `ne=[1,1]` leaf holding a single constant value (`1e-7`),
/// broadcast against any `[CF,1]`/`[CF,T]` tensor via `ggml_can_repeat` (every
/// dim of a `[1,1]` tensor divides anything). There is no scalar-add op in
/// this crate's ggml surface, so a same-shape-broadcastable constant leaf is
/// the standard way to add/clamp a compile-time epsilon; the backbone builder
/// uploads it once into the static weight arena (see `backbone.rs`).
#[allow(clippy::too_many_arguments)]
pub(super) fn astp_pool<'a>(
    graph: &GgmlCpuGraphBuilder<'a>,
    x_tc: GgmlCpuTensor<'a>,
    t: usize,
    cf: usize,
    eps_1e7: GgmlCpuTensor<'a>,
    linear1_w: GgmlCpuTensor<'a>,
    linear1_b: GgmlCpuTensor<'a>,
    linear2_w: GgmlCpuTensor<'a>,
    linear2_b: GgmlCpuTensor<'a>,
) -> OpResult<'a> {
    let m = map_err("astp_pool");
    // See `to1d`'s identical defensive `cont()`: `x_tc` (`pre_pool_flat`) may
    // itself be an independently `set_output`-marked tap, and `transpose` is
    // a view op.
    let x_tc = graph.cont(x_tc).map_err(m)?;
    let x_ct = graph.cont(graph.transpose(x_tc).map_err(m)?).map_err(m)?; // [CF,T]

    // Per-channel mean/std over T: transpose to bring T to ne0 so
    // sum_rows/mean_rows (which reduce ne0) act over time.
    let x_time_major = graph.cont(graph.transpose(x_ct).map_err(m)?).map_err(m)?; // [T,CF]
    let mean_row = graph.mean_rows(x_time_major).map_err(m)?; // [1,CF]
    let sq = graph.sqr(x_time_major).map_err(m)?;
    let mean_sq_row = graph.mean_rows(sq).map_err(m)?; // [1,CF]
    let mean_ct = graph
        .cont(graph.transpose(mean_row).map_err(m)?)
        .map_err(m)?; // [CF,1]
    let mean_sq_ct = graph
        .cont(graph.transpose(mean_sq_row).map_err(m)?)
        .map_err(m)?; // [CF,1]
    let mean_sq_of_mean = graph.sqr(mean_ct).map_err(m)?;
    let var_ct = graph.sub(mean_sq_ct, mean_sq_of_mean).map_err(m)?;
    let var_eps_ct = graph.add(var_ct, eps_1e7).map_err(m)?; // context_std: +eps (no clamp).
    let std_ct = graph.sqrt(var_eps_ct).map_err(m)?;

    // Broadcast [CF,1] over T via repeat to [CF,T].
    let mean_bc = graph.repeat_4d(mean_ct, cf, t, 1, 1).map_err(m)?;
    let std_bc = graph.repeat_4d(std_ct, cf, t, 1, 1).map_err(m)?;

    let cat1 = graph.concat(x_ct, mean_bc, 0).map_err(m)?;
    let x_in = graph.concat(cat1, std_bc, 0).map_err(m)?; // [3*CF, T]

    let hidden = linear_ct(graph, linear1_w, linear1_b, x_in)?;
    let hidden = graph.tanh(hidden).map_err(m)?;
    let scores = linear_ct(graph, linear2_w, linear2_b, hidden)?; // [CF,T]

    // softmax over T (dim=2 in torch, i.e. per-channel softmax across time):
    // transpose so T is ne0, softmax (reduces ne0), transpose back.
    let scores_tmajor = graph.cont(graph.transpose(scores).map_err(m)?).map_err(m)?; // [T,CF]
    let alpha_tmajor = graph.soft_max(scores_tmajor).map_err(m)?;
    let alpha = graph
        .cont(graph.transpose(alpha_tmajor).map_err(m)?)
        .map_err(m)?; // [CF,T]

    let weighted = graph.mul(alpha, x_ct).map_err(m)?;
    let weighted_tmajor = graph
        .cont(graph.transpose(weighted).map_err(m)?)
        .map_err(m)?;
    let mean_pooled_row = graph.sum_rows(weighted_tmajor).map_err(m)?; // [1,CF]
    let mean_pooled = graph
        .cont(graph.transpose(mean_pooled_row).map_err(m)?)
        .map_err(m)?; // [CF,1]

    let x_sq = graph.sqr(x_ct).map_err(m)?;
    let weighted_sq = graph.mul(alpha, x_sq).map_err(m)?;
    let weighted_sq_tmajor = graph
        .cont(graph.transpose(weighted_sq).map_err(m)?)
        .map_err(m)?;
    let mean_sq_pooled_row = graph.sum_rows(weighted_sq_tmajor).map_err(m)?;
    let mean_sq_pooled = graph
        .cont(graph.transpose(mean_sq_pooled_row).map_err(m)?)
        .map_err(m)?; // [CF,1]

    let mean_pooled_sq = graph.sqr(mean_pooled).map_err(m)?;
    let var_pooled = graph.sub(mean_sq_pooled, mean_pooled_sq).map_err(m)?;
    // torch: `var.clamp(min=1e-7)` == `relu(var - eps) + eps`.
    let var_shifted = graph.sub(var_pooled, eps_1e7).map_err(m)?;
    let var_shifted = graph.relu(var_shifted).map_err(m)?;
    let var_clamped = graph.add(var_shifted, eps_1e7).map_err(m)?;
    let std_pooled = graph.sqrt(var_clamped).map_err(m)?; // [CF,1]

    let mean_flat = graph.reshape_1d(mean_pooled, cf).map_err(m)?;
    let std_flat = graph.reshape_1d(std_pooled, cf).map_err(m)?;
    graph.concat(mean_flat, std_flat, 0).map_err(m)
}
