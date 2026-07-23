//! Structural constants for the ReDimNet2-B6 backbone.
//!
//! Every dimension is hard-coded from the B6 checkpoint (see
//! `docs/design/redimnet2-b6-embedder.md` and `B6_STRUCTURE_SPEC.md`). B6 is
//! *not* a uniform scale-up of B3, so per-stage values (block counts, conv
//! expansion, attention hidden dims) are listed explicitly and must not be
//! derived from a single global constant.
//!
//! The ReDimNet2 backbone is a UNet-style "dimension reshaping" net. The
//! invariant that makes the shapes tractable: every stage's *output* is
//! flattened back to a `C*F`-channel 1D feature map (`AGG_CHANNELS = 4608`) at
//! the full time resolution `T` before aggregation, regardless of the 2D
//! channel counts used inside the stage. `fin_wght1d` then learns a softmax
//! weighted sum over the stem output + all six stage outputs.

/// Base channel count `C` (stem output channels).
pub(crate) const C: usize = 64;
/// Frequency bins `F` fed by the front end (== `n_mels`).
pub(crate) const F: usize = 72;
/// Flattened aggregation channel count `C*F` used by every `to1d` output,
/// `weigth1d`, `red_dim_conv`, and `stem_gnorm`. A backbone-wide constant, NOT a
/// per-stage value.
pub(crate) const AGG_CHANNELS: usize = C * F; // 4608
/// Final embedding dimensionality.
pub(crate) const EMBED_DIM: usize = 192;
/// `out_channels` of the `head` 1x1 conv after `fin_to2d`.
pub(crate) const OUT_CHANNELS: usize = 224;
/// Cumulative frequency stride across all stages (`F` -> `F/8 = 9`).
pub(crate) const FREQ_STRIDE: usize = 8;
/// Cumulative time stride (`max` accumulated `st`); the input time axis is
/// truncated to a multiple of this before the stem.
pub(crate) const TIME_STRIDE: usize = 4;
/// `agg_gnorm` group count (== `C`); GroupNorm over `AGG_CHANNELS`.
pub(crate) const AGG_GNORM_GROUPS: usize = C; // 64

/// Depthwise temporal-conv kernel sizes inside each `TimeContextBlock1d`'s TCM
/// (`dwconvs.{0..3}`), shared across stages (from the checkpoint: 7,19,31,59).
pub(crate) const TCM_DWCONV_KERNELS: [usize; 4] = [7, 19, 31, 59];

/// Per-stage structural parameters.
#[derive(Debug, Clone, Copy)]
pub(crate) struct StageConfig {
    /// Frequency stride `sf` for this stage's strided down-conv.
    pub sf: usize,
    /// Time stride `st` for this stage's strided down-conv.
    pub st: usize,
    /// Number of `ConvBlock2d` (basic ResNet) blocks.
    pub num_blocks: usize,
    /// Conv-expansion ratio for the 2D branch (`c*conv_exp` intermediate
    /// channels). May be < 1 for stages 4/5, which narrow the branch.
    pub conv_exp: f32,
    /// `att_block_red`: `TimeContextBlock1d` hidden dim = `AGG_CHANNELS / red`.
    pub att_block_red: usize,
    /// 2D channel count `c` *after* this stage's `c = sf*c` update (running).
    pub c_out_2d: usize,
    /// Frequency bins `f` after this stage (`f = f / sf`).
    pub f_out: usize,
    /// Intermediate 2D channel count in the blocks: `round(c_out_2d * conv_exp)`.
    pub block_channels: usize,
    /// TCM (attention) hidden dim = `AGG_CHANNELS / att_block_red`.
    pub tcm_hidden: usize,
}

/// The six B6 stages, fully resolved. Running `c` starts at `C=64`, `f` at
/// `F=72`; each stage does `c = sf*c`, `f = f/sf`.
///
/// | stage | sf,st | blocks | conv_exp | red | c_2d | f | block_ch | tcm_hidden |
/// |-------|-------|--------|----------|-----|------|---|----------|------------|
/// | 0     | 1,1   | 3      | 3.0      | 64  | 64   | 72| 192      | 72         |
/// | 1     | 2,1   | 4      | 2.0      | 64  | 128  | 36| 256      | 72         |
/// | 2     | 1,2   | 5      | 2.0      | 48  | 128  | 36| 256      | 96         |
/// | 3     | 2,1   | 5      | 1.0      | 48  | 256  | 18| 256      | 96         |
/// | 4     | 1,2   | 4      | 0.75     | 32  | 256  | 18| 192      | 144        |
/// | 5     | 2,1   | 3      | 0.5      | 24  | 512  | 9 | 256      | 192        |
///
/// (`block_channels` and `c_2d` cross-checked against the checkpoint tensor
/// shapes, e.g. `stage1.2.weight (256,1,2,1)` -> down-conv out 256, and
/// `stage5.head`/`head.weight (224,512,1,1)` -> final 2D channels 512.)
pub(crate) const STAGES: [StageConfig; 6] = [
    StageConfig {
        sf: 1,
        st: 1,
        num_blocks: 3,
        conv_exp: 3.0,
        att_block_red: 64,
        c_out_2d: 64,
        f_out: 72,
        block_channels: 192,
        tcm_hidden: 72,
    },
    StageConfig {
        sf: 2,
        st: 1,
        num_blocks: 4,
        conv_exp: 2.0,
        att_block_red: 64,
        c_out_2d: 128,
        f_out: 36,
        block_channels: 256,
        tcm_hidden: 72,
    },
    StageConfig {
        sf: 1,
        st: 2,
        num_blocks: 5,
        conv_exp: 2.0,
        att_block_red: 48,
        c_out_2d: 128,
        f_out: 36,
        block_channels: 256,
        tcm_hidden: 96,
    },
    StageConfig {
        sf: 2,
        st: 1,
        num_blocks: 5,
        conv_exp: 1.0,
        att_block_red: 48,
        c_out_2d: 256,
        f_out: 18,
        block_channels: 256,
        tcm_hidden: 96,
    },
    StageConfig {
        sf: 1,
        st: 2,
        num_blocks: 4,
        conv_exp: 0.75,
        att_block_red: 32,
        c_out_2d: 256,
        f_out: 18,
        block_channels: 192,
        tcm_hidden: 144,
    },
    StageConfig {
        sf: 2,
        st: 1,
        num_blocks: 3,
        conv_exp: 0.5,
        att_block_red: 24,
        c_out_2d: 512,
        f_out: 9,
        block_channels: 256,
        tcm_hidden: 192,
    },
];

/// Number of feature maps fed to `fin_wght1d`: stem + 6 stage outputs = 7.
pub(crate) const FIN_WGHT1D_N: usize = 1 + STAGES.len();

/// Final 2D shape after `head`: `(OUT_CHANNELS, F/FREQ_STRIDE, T)` = `(224, 9, T)`,
/// flattened to `(2016, T)` before ASTP pooling.
pub(crate) const PRE_POOL_CHANNELS: usize = OUT_CHANNELS * (F / FREQ_STRIDE); // 2016
/// ASTP output dim with `global_context_att`: `2 * PRE_POOL_CHANNELS`.
pub(crate) const POOL_OUT_DIM: usize = 2 * PRE_POOL_CHANNELS; // 4032

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stage_running_dims_are_self_consistent() {
        let mut c = C;
        let mut f = F;
        for (i, s) in STAGES.iter().enumerate() {
            c *= s.sf;
            assert_eq!(f % s.sf, 0, "stage{i}: f {f} not divisible by sf {}", s.sf);
            f /= s.sf;
            assert_eq!(c, s.c_out_2d, "stage{i} c mismatch");
            assert_eq!(f, s.f_out, "stage{i} f mismatch");
            let block_ch = (s.c_out_2d as f32 * s.conv_exp).round() as usize;
            assert_eq!(
                block_ch, s.block_channels,
                "stage{i} block_channels mismatch"
            );
            assert_eq!(
                AGG_CHANNELS / s.att_block_red,
                s.tcm_hidden,
                "stage{i} tcm_hidden"
            );
        }
        assert_eq!(c, 512, "final 2D channels feed head (512->224)");
        assert_eq!(f, F / FREQ_STRIDE, "final freq bins = 9");
    }

    #[test]
    fn aggregate_constants() {
        assert_eq!(AGG_CHANNELS, 4608);
        assert_eq!(PRE_POOL_CHANNELS, 2016);
        assert_eq!(POOL_OUT_DIM, 4032);
        assert_eq!(FIN_WGHT1D_N, 7);
    }
}
