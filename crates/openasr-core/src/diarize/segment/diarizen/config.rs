use serde::Deserialize;

use crate::GgufMetadata;

use super::DiariZenSegmenterError;

pub const ARCHITECTURE_ID: &str = "diarizen-wavlm-conformer-segmentation";
pub(super) const FAMILY_ID: &str = "diarizen-segmentation";
pub(super) const MODEL_ID: &str = "diarizen-base-s80";
pub(super) const UPSTREAM_MODEL_ID: &str = "BUT-FIT/diarizen-wavlm-base-s80-md";
pub(super) const PINNED_REVISION: &str = "a9857fc34908197fb5336d9d0562f291834a04b2";
pub(super) const TENSOR_SCHEMA: &str = "compact-v1";

pub(super) const SAMPLE_RATE_HZ: u32 = 16_000;
pub(super) const WINDOW_SAMPLES: usize = 16 * SAMPLE_RATE_HZ as usize;
pub(super) const WINDOW_STEP_SAMPLES: usize = 16 * 1_600;
pub(super) const FRAME_STRIDE_SAMPLES: usize = 320;
pub(super) const LOCAL_SPEAKERS: usize = 4;
pub(super) const MAX_SIMULTANEOUS_SPEAKERS: usize = 2;
pub(super) const POWERSET_CLASSES: usize = 11;
pub(super) const MEDIAN_FILTER_FRAMES: usize = 11;

pub(super) const HIDDEN_SIZE: usize = 768;
pub(super) const TOTAL_HEADS: usize = 12;
pub(super) const HEAD_DIM: usize = 64;
pub(super) const CONFORMER_DIM: usize = 256;
pub(super) const CONFORMER_HEADS: usize = 4;
pub(super) const CONFORMER_FFN_DIM: usize = 1024;
pub(super) const CONFORMER_LAYERS: usize = 4;
pub(super) const CONFORMER_KERNEL: usize = 31;
pub(super) const RELATIVE_POSITION_BUCKETS: usize = 320;
pub(super) const RELATIVE_POSITION_MAX_DISTANCE: usize = 800;

pub(super) const CONV_CHANNELS: [usize; 7] = [90, 161, 173, 181, 351, 155, 137];
pub(super) const CONV_KERNELS: [usize; 7] = [10, 3, 3, 3, 3, 2, 2];
pub(super) const CONV_STRIDES: [usize; 7] = [5, 2, 2, 2, 2, 2, 2];
pub(super) const REMAINING_HEADS: [&[usize]; 12] = [
    &[1, 6],
    &[5, 7, 8],
    &[0, 3, 9],
    &[0, 1, 4, 8, 11],
    &[6, 8],
    &[0],
    &[7, 8, 10, 11],
    &[0, 1, 4, 8],
    &[],
    &[],
    &[4, 7],
    &[5],
];
pub(super) const FFN_DIMS: [usize; 12] =
    [666, 660, 649, 1080, 237, 299, 437, 573, 53, 80, 211, 334];

#[derive(Debug, Deserialize, PartialEq, Eq)]
struct WavLmConfig {
    conv_channels: Vec<usize>,
    conv_kernels: Vec<usize>,
    conv_strides: Vec<usize>,
    remaining_heads: Vec<Vec<usize>>,
    ffn_dims: Vec<usize>,
    hidden_size: usize,
    total_heads: usize,
    head_dim: usize,
    relative_position_buckets: usize,
    relative_position_max_distance: usize,
}

fn require_string<'a>(
    metadata: &'a GgufMetadata,
    key: &'static str,
) -> Result<&'a str, DiariZenSegmenterError> {
    metadata
        .get_string(key)
        .ok_or(DiariZenSegmenterError::MissingMetadata { key })
}

fn require_u32(metadata: &GgufMetadata, key: &'static str) -> Result<u32, DiariZenSegmenterError> {
    metadata
        .get_u32(key)
        .ok_or(DiariZenSegmenterError::MissingMetadata { key })
}

fn expect_string(
    metadata: &GgufMetadata,
    key: &'static str,
    expected: &'static str,
) -> Result<(), DiariZenSegmenterError> {
    let actual = require_string(metadata, key)?;
    if actual != expected {
        return Err(DiariZenSegmenterError::MetadataMismatch {
            key,
            expected: expected.to_string(),
            actual: actual.to_string(),
        });
    }
    Ok(())
}

fn expect_u32(
    metadata: &GgufMetadata,
    key: &'static str,
    expected: u32,
) -> Result<(), DiariZenSegmenterError> {
    let actual = require_u32(metadata, key)?;
    if actual != expected {
        return Err(DiariZenSegmenterError::MetadataMismatch {
            key,
            expected: expected.to_string(),
            actual: actual.to_string(),
        });
    }
    Ok(())
}

pub(super) fn validate_metadata(metadata: &GgufMetadata) -> Result<(), DiariZenSegmenterError> {
    expect_string(metadata, "general.architecture", ARCHITECTURE_ID)?;
    expect_string(metadata, "openasr.model.family", FAMILY_ID)?;
    expect_string(metadata, "openasr.model.architecture", ARCHITECTURE_ID)?;
    expect_string(metadata, "openasr.model.id", MODEL_ID)?;
    expect_string(metadata, "diarizen.upstream_model_id", UPSTREAM_MODEL_ID)?;
    expect_string(metadata, "diarizen.upstream_revision", PINNED_REVISION)?;
    expect_string(metadata, "diarizen.tensor_schema", TENSOR_SCHEMA)?;
    expect_u32(metadata, "diarizen.sample_rate", SAMPLE_RATE_HZ)?;
    expect_u32(metadata, "diarizen.window_samples", WINDOW_SAMPLES as u32)?;
    expect_u32(
        metadata,
        "diarizen.window_step_samples",
        WINDOW_STEP_SAMPLES as u32,
    )?;
    expect_u32(
        metadata,
        "diarizen.output_frame_stride_samples",
        FRAME_STRIDE_SAMPLES as u32,
    )?;
    expect_u32(metadata, "diarizen.local_speakers", LOCAL_SPEAKERS as u32)?;
    expect_u32(
        metadata,
        "diarizen.max_simultaneous_speakers",
        MAX_SIMULTANEOUS_SPEAKERS as u32,
    )?;
    expect_u32(
        metadata,
        "diarizen.powerset_classes",
        POWERSET_CLASSES as u32,
    )?;
    expect_u32(
        metadata,
        "diarizen.median_filter_frames",
        MEDIAN_FILTER_FRAMES as u32,
    )?;

    let raw = require_string(metadata, "diarizen.wavlm_config_json")?;
    let actual: WavLmConfig = serde_json::from_str(raw).map_err(|source| {
        DiariZenSegmenterError::InvalidMetadataJson {
            key: "diarizen.wavlm_config_json",
            source,
        }
    })?;
    let expected = WavLmConfig {
        conv_channels: CONV_CHANNELS.to_vec(),
        conv_kernels: CONV_KERNELS.to_vec(),
        conv_strides: CONV_STRIDES.to_vec(),
        remaining_heads: REMAINING_HEADS.iter().map(|heads| heads.to_vec()).collect(),
        ffn_dims: FFN_DIMS.to_vec(),
        hidden_size: HIDDEN_SIZE,
        total_heads: TOTAL_HEADS,
        head_dim: HEAD_DIM,
        relative_position_buckets: RELATIVE_POSITION_BUCKETS,
        relative_position_max_distance: RELATIVE_POSITION_MAX_DISTANCE,
    };
    if actual != expected {
        return Err(DiariZenSegmenterError::MetadataMismatch {
            key: "diarizen.wavlm_config_json",
            expected: format!("{expected:?}"),
            actual: format!("{actual:?}"),
        });
    }
    Ok(())
}

pub(super) fn output_frames(samples: usize) -> usize {
    CONV_KERNELS
        .iter()
        .zip(CONV_STRIDES)
        .try_fold(samples, |frames, (&kernel, stride)| {
            (frames >= kernel).then_some((frames - kernel) / stride + 1)
        })
        .unwrap_or(0)
}
