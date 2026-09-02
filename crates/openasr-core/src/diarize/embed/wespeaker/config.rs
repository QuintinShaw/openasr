//! WeSpeaker ResNet size table.
//!
//! Four depths share one ggml builder. Stage-1 ships ResNet34 weights; the
//! remaining rows exist so 152/221/293 do not grow a copied graph.

use crate::ggml_runtime::GgufMetadata;

pub(crate) const ARCHITECTURE_ID: &str = "wespeaker-resnet";
pub(crate) const N_MELS: usize = 80;
pub(crate) const M_CHANNELS: usize = 32;
pub(crate) const EMBED_DIM: usize = 256;
pub(crate) const STAGE_STRIDES: [usize; 4] = [1, 2, 2, 2];
pub(crate) const BN_EPS: f32 = 1e-5;
pub(crate) const TSTP_EPS: f32 = 1e-7;
pub(crate) const FREQ_STRIDE: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BlockKind {
    Basic,
    Bottleneck,
}

impl BlockKind {
    pub(crate) const fn expansion(self) -> usize {
        match self {
            Self::Basic => 1,
            Self::Bottleneck => 4,
        }
    }

    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Basic => "basic",
            Self::Bottleneck => "bottleneck",
        }
    }

    fn parse(value: &str) -> Result<Self, String> {
        match value.trim() {
            "basic" => Ok(Self::Basic),
            "bottleneck" => Ok(Self::Bottleneck),
            other => Err(format!("unsupported wespeaker.block_kind '{other}'")),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ResNetConfig {
    pub depth: u32,
    pub block_kind: BlockKind,
    pub num_blocks: [usize; 4],
    /// `_make_layer` `planes` argument (before Bottleneck expansion).
    pub stage_planes: [usize; 4],
}

impl ResNetConfig {
    pub(crate) const fn expansion(self) -> usize {
        self.block_kind.expansion()
    }

    pub(crate) const fn stem_channels(self) -> usize {
        M_CHANNELS
    }

    pub(crate) const fn stage_out_channels(self, stage: usize) -> usize {
        self.stage_planes[stage] * self.expansion()
    }

    pub(crate) const fn last_channels(self) -> usize {
        self.stage_out_channels(3)
    }

    pub(crate) const fn tstp_in(self) -> usize {
        (N_MELS / FREQ_STRIDE) * self.last_channels()
    }

    pub(crate) const fn tstp_out(self) -> usize {
        self.tstp_in() * 2
    }

    pub(crate) const fn shortcut_required(
        self,
        in_channels: usize,
        stage: usize,
        stride: usize,
    ) -> bool {
        stride != 1 || in_channels != self.stage_out_channels(stage)
    }
}

/// ResNet34: BasicBlock [3,4,6,3], channels 32/64/128/256, TSTP 2560/5120.
pub(crate) const RESNET34: ResNetConfig = ResNetConfig {
    depth: 34,
    block_kind: BlockKind::Basic,
    num_blocks: [3, 4, 6, 3],
    stage_planes: [32, 64, 128, 256],
};

/// ResNet152: Bottleneck [3,8,36,3], stage outs 128/256/512/1024.
pub(crate) const RESNET152: ResNetConfig = ResNetConfig {
    depth: 152,
    block_kind: BlockKind::Bottleneck,
    num_blocks: [3, 8, 36, 3],
    stage_planes: [32, 64, 128, 256],
};

/// ResNet221: Bottleneck [6,16,48,3].
pub(crate) const RESNET221: ResNetConfig = ResNetConfig {
    depth: 221,
    block_kind: BlockKind::Bottleneck,
    num_blocks: [6, 16, 48, 3],
    stage_planes: [32, 64, 128, 256],
};

/// ResNet293: Bottleneck [10,20,64,3].
pub(crate) const RESNET293: ResNetConfig = ResNetConfig {
    depth: 293,
    block_kind: BlockKind::Bottleneck,
    num_blocks: [10, 20, 64, 3],
    stage_planes: [32, 64, 128, 256],
};

pub(crate) const RESNET_CONFIGS: [ResNetConfig; 4] = [RESNET34, RESNET152, RESNET221, RESNET293];

pub(crate) fn config_for_depth(depth: u32) -> Result<ResNetConfig, String> {
    RESNET_CONFIGS
        .iter()
        .copied()
        .find(|config| config.depth == depth)
        .ok_or_else(|| format!("unsupported wespeaker.depth {depth}"))
}

pub(crate) fn config_from_metadata(metadata: &GgufMetadata) -> Result<ResNetConfig, String> {
    let architecture = metadata
        .get_string(crate::arch::GENERAL_ARCHITECTURE_KEY)
        .map(str::trim)
        .ok_or_else(|| "missing general.architecture".to_string())?;
    if architecture != ARCHITECTURE_ID {
        return Err(format!(
            "pack architecture is '{architecture}', expected '{ARCHITECTURE_ID}'"
        ));
    }
    let depth = metadata
        .get_u32("wespeaker.depth")
        .ok_or_else(|| "missing wespeaker.depth".to_string())?;
    let config = config_for_depth(depth)?;
    let kind = BlockKind::parse(
        metadata
            .get_string("wespeaker.block_kind")
            .ok_or_else(|| "missing wespeaker.block_kind".to_string())?,
    )?;
    if kind != config.block_kind {
        return Err(format!(
            "wespeaker.block_kind '{}' does not match depth {depth} ({})",
            kind.as_str(),
            config.block_kind.as_str()
        ));
    }
    let num_blocks_raw = metadata
        .get_string("wespeaker.num_blocks")
        .ok_or_else(|| "missing wespeaker.num_blocks".to_string())?;
    let parsed: Vec<usize> = serde_json::from_str(num_blocks_raw)
        .map_err(|error| format!("wespeaker.num_blocks is not JSON: {error}"))?;
    if parsed.as_slice() != config.num_blocks {
        return Err(format!(
            "wespeaker.num_blocks {parsed:?} does not match depth {depth} ({:?})",
            config.num_blocks
        ));
    }
    let embed_dim = metadata
        .get_u32("wespeaker.embed_dim")
        .ok_or_else(|| "missing wespeaker.embed_dim".to_string())? as usize;
    if embed_dim != EMBED_DIM {
        return Err(format!(
            "wespeaker.embed_dim is {embed_dim}, expected {EMBED_DIM}"
        ));
    }
    let n_mels = metadata
        .get_u32("wespeaker.n_mels")
        .ok_or_else(|| "missing wespeaker.n_mels".to_string())? as usize;
    if n_mels != N_MELS {
        return Err(format!("wespeaker.n_mels is {n_mels}, expected {N_MELS}"));
    }
    let m_channels = metadata
        .get_u32("wespeaker.m_channels")
        .ok_or_else(|| "missing wespeaker.m_channels".to_string())? as usize;
    if m_channels != M_CHANNELS {
        return Err(format!(
            "wespeaker.m_channels is {m_channels}, expected {M_CHANNELS}"
        ));
    }
    let pooling = metadata
        .get_string("wespeaker.pooling")
        .unwrap_or("TSTP")
        .trim();
    if pooling != "TSTP" {
        return Err(format!("wespeaker.pooling is '{pooling}', expected TSTP"));
    }
    if metadata
        .get_bool("wespeaker.two_emb_layer")
        .unwrap_or(false)
    {
        return Err("wespeaker.two_emb_layer=true is not supported".to_string());
    }
    Ok(config)
}

pub(crate) fn post_stride_time_len(input_frames: usize) -> usize {
    let mut frames = input_frames;
    for stride in STAGE_STRIDES {
        if stride > 1 {
            frames = conv_same_padding_stride_len(frames, stride);
        }
    }
    frames
}

fn conv_same_padding_stride_len(input: usize, stride: usize) -> usize {
    if input == 0 {
        0
    } else {
        input.div_ceil(stride)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resnet34_table_matches_official_topology() {
        assert_eq!(RESNET34.block_kind, BlockKind::Basic);
        assert_eq!(RESNET34.num_blocks, [3, 4, 6, 3]);
        assert_eq!(RESNET34.stage_out_channels(0), 32);
        assert_eq!(RESNET34.stage_out_channels(3), 256);
        assert_eq!(RESNET34.tstp_in(), 2560);
        assert_eq!(RESNET34.tstp_out(), 5120);
        assert!(!RESNET34.shortcut_required(32, 0, 1));
        assert!(RESNET34.shortcut_required(32, 1, 2));
    }

    #[test]
    fn bottleneck_table_uses_expansion_4() {
        for config in [RESNET152, RESNET221, RESNET293] {
            assert_eq!(config.block_kind, BlockKind::Bottleneck);
            assert_eq!(config.expansion(), 4);
            assert_eq!(config.stage_out_channels(0), 128);
            assert_eq!(config.stage_out_channels(3), 1024);
            assert_eq!(config.tstp_in(), 10240);
            assert_eq!(config.tstp_out(), 20480);
            assert!(config.shortcut_required(32, 0, 1));
        }
        assert_eq!(RESNET152.num_blocks, [3, 8, 36, 3]);
        assert_eq!(RESNET221.num_blocks, [6, 16, 48, 3]);
        assert_eq!(RESNET293.num_blocks, [10, 20, 64, 3]);
    }

    #[test]
    fn post_stride_time_len_requires_two_frames_for_tstp() {
        assert_eq!(post_stride_time_len(0), 0);
        assert_eq!(post_stride_time_len(8), 1);
        assert_eq!(post_stride_time_len(9), 2);
    }
}
