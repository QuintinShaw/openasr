//! Pure-Rust forward pass of pyannote/WeSpeaker ResNet34 (VoxCeleb, 256-dim).
//!
//! Reference: `pyannote.audio.models.embedding.WeSpeakerResNet34` from
//! pyannote.audio 3.1.1, using WeSpeaker's ResNet34 backbone:
//!
//! ```text
//! features [T,80] -> conv1 + BN + ReLU
//!   -> BasicBlock stages [3,4,6,3] with channels [32,64,128,256]
//!   -> flatten [256,10,T/8] as [2560,T/8]
//!   -> temporal stats pool (mean + unbiased std) -> [5120]
//!   -> seg_1 linear -> [256]
//! ```
//!
//! The public pyannote wrapper returns this `seg_1` output directly
//! (`two_emb_layer=False`).

use std::collections::BTreeSet;
use std::time::Instant;

use super::ops::{conv1d, conv2d_batchnorm, stats_pool};
use super::weights::{Weights, WeightsError};

const BN_EPS: f32 = 1e-5;
const N_MELS: usize = 80;
const EMBEDDING_DIM: usize = 256;
const STATS_POOL_DIM: usize = 5120;
const STAGE_BLOCKS: [usize; 4] = [3, 4, 6, 3];
const STAGE_CHANNELS: [usize; 4] = [32, 64, 128, 256];
const STAGE_STRIDES: [usize; 4] = [1, 2, 2, 2];

struct ExpectedTensor {
    name: String,
    shapes: Vec<Vec<usize>>,
}

pub(crate) struct WeSpeakerResNet34Model {
    w: Weights,
    embedding_dim: usize,
}

impl WeSpeakerResNet34Model {
    pub(crate) fn from_safetensors(bytes: &[u8]) -> Result<Self, WeightsError> {
        let w = Weights::from_safetensors(bytes)?;
        Self::from_weights(w)
    }

    pub(crate) fn from_oasr(path: &std::path::Path) -> Result<Self, WeightsError> {
        let w = Weights::from_oasr(path)?;
        Self::from_weights(w)
    }

    fn from_weights(w: Weights) -> Result<Self, WeightsError> {
        validate_wespeaker_schema(&w)?;
        Ok(Self {
            w,
            embedding_dim: EMBEDDING_DIM,
        })
    }

    pub(crate) fn embedding_dim(&self) -> usize {
        self.embedding_dim
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

    /// Run the network on CMN-normalized fbank features (`[t, 80]` row-major)
    /// and return the raw, un-normalized embedding.
    pub(crate) fn forward(&self, features: &[f32], t: usize) -> Result<Vec<f32>, WeightsError> {
        let profile = profile_enabled();
        let total_started = profile.then(Instant::now);
        let post_stride_t = Self::post_stride_time_len(t);
        if post_stride_t < 2 {
            return Err(WeightsError::InvalidInput(format!(
                "WeSpeaker ResNet34 post-stride time length is {post_stride_t}; need at least 2 for torch std(correction=1) stats_pool parity"
            )));
        }
        if features.len() != t * N_MELS {
            return Err(WeightsError::InvalidInput(format!(
                "WeSpeaker ResNet34 expected {} fbank values for {t} frames, got {}",
                t * N_MELS,
                features.len()
            )));
        }
        let mut img = vec![0.0f32; N_MELS * t];
        for ti in 0..t {
            for f in 0..N_MELS {
                img[f * t + ti] = features[ti * N_MELS + f];
            }
        }

        let stage_started = profile.then(Instant::now);
        let (mut x, mut h, mut w) =
            self.conv2d_bn_relu(&img, 1, N_MELS, t, "resnet.conv1", "resnet.bn1", 1, 1, 3)?;
        log_profile(profile, "conv1", stage_started, x.len());
        let mut channels = STAGE_CHANNELS[0];

        for (stage_idx, (&blocks, &out_channels)) in
            STAGE_BLOCKS.iter().zip(STAGE_CHANNELS.iter()).enumerate()
        {
            for block_idx in 0..blocks {
                let block_started = profile.then(Instant::now);
                let stride = if block_idx == 0 {
                    STAGE_STRIDES[stage_idx]
                } else {
                    1
                };
                let prefix = format!("resnet.layer{}.{}", stage_idx + 1, block_idx);
                let (next, next_h, next_w) =
                    self.basic_block(&x, channels, h, w, out_channels, stride, &prefix)?;
                x = next;
                h = next_h;
                w = next_w;
                channels = out_channels;
                if profile {
                    log_profile(
                        true,
                        &format!("layer{}.{}", stage_idx + 1, block_idx),
                        block_started,
                        x.len(),
                    );
                }
            }
        }

        // TSTP flattens `[channel, freq, time]` into `[channel * freq, time]`.
        let stats_channels = channels * h;
        debug_assert_eq!(x.len(), stats_channels * w);
        let pool_started = profile.then(Instant::now);
        let pooled = stats_pool(&x, stats_channels, w);
        log_profile(profile, "stats_pool", pool_started, pooled.len());

        let seg_started = profile.then(Instant::now);
        let weight = self.w.get("resnet.seg_1.weight")?;
        let bias = self.w.get("resnet.seg_1.bias")?;
        let (embedding, len) = conv1d(
            &pooled,
            pooled.len(),
            1,
            weight,
            Some(bias),
            self.embedding_dim,
            1,
            1,
            0,
            1,
        );
        debug_assert_eq!(len, 1);
        log_profile(profile, "seg_1", seg_started, embedding.len());
        log_profile(profile, "total", total_started, embedding.len());
        Ok(embedding)
    }

    #[allow(clippy::too_many_arguments)]
    fn basic_block(
        &self,
        x: &[f32],
        c_in: usize,
        h: usize,
        w: usize,
        c_out: usize,
        stride: usize,
        prefix: &str,
    ) -> Result<(Vec<f32>, usize, usize), WeightsError> {
        let (out, h1, w1) = self.conv2d_bn_relu(
            x,
            c_in,
            h,
            w,
            &format!("{prefix}.conv1"),
            &format!("{prefix}.bn1"),
            stride,
            stride,
            3,
        )?;
        let (mut out, h2, w2) = self.conv2d_bn(
            &out,
            c_out,
            h1,
            w1,
            &format!("{prefix}.conv2"),
            &format!("{prefix}.bn2"),
            1,
            1,
            3,
        )?;

        let shortcut_conv = format!("{prefix}.shortcut.0");
        let shortcut = if self.w.contains(&format!("{shortcut_conv}.weight")) {
            let (sc, _h, _w) = self.conv2d_bn_raw(
                x,
                c_in,
                h,
                w,
                &shortcut_conv,
                &format!("{prefix}.shortcut.1"),
                stride,
                stride,
                1,
                0,
                false,
            )?;
            sc
        } else {
            x.to_vec()
        };
        debug_assert_eq!(out.len(), shortcut.len());
        for (o, s) in out.iter_mut().zip(shortcut.iter()) {
            *o = (*o + *s).max(0.0);
        }
        Ok((out, h2, w2))
    }

    #[allow(clippy::too_many_arguments)]
    fn conv2d_bn_relu(
        &self,
        x: &[f32],
        c_in: usize,
        h: usize,
        w: usize,
        conv_prefix: &str,
        bn_prefix: &str,
        stride_h: usize,
        stride_w: usize,
        k: usize,
    ) -> Result<(Vec<f32>, usize, usize), WeightsError> {
        let pad = k / 2;
        self.conv2d_bn_raw(
            x,
            c_in,
            h,
            w,
            conv_prefix,
            bn_prefix,
            stride_h,
            stride_w,
            k,
            pad,
            true,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn conv2d_bn(
        &self,
        x: &[f32],
        c_in: usize,
        h: usize,
        w: usize,
        conv_prefix: &str,
        bn_prefix: &str,
        stride_h: usize,
        stride_w: usize,
        k: usize,
    ) -> Result<(Vec<f32>, usize, usize), WeightsError> {
        let pad = k / 2;
        self.conv2d_bn_raw(
            x,
            c_in,
            h,
            w,
            conv_prefix,
            bn_prefix,
            stride_h,
            stride_w,
            k,
            pad,
            false,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn conv2d_bn_raw(
        &self,
        x: &[f32],
        c_in: usize,
        h: usize,
        w: usize,
        conv_prefix: &str,
        bn_prefix: &str,
        stride_h: usize,
        stride_w: usize,
        k: usize,
        pad: usize,
        relu: bool,
    ) -> Result<(Vec<f32>, usize, usize), WeightsError> {
        let weight = self.w.get(&format!("{conv_prefix}.weight"))?;
        let c_out = self.w.shape(&format!("{conv_prefix}.weight"))?[0];
        let gamma = self.w.get(&format!("{bn_prefix}.weight"))?;
        let beta = self.w.get(&format!("{bn_prefix}.bias"))?;
        let mean = self.w.get(&format!("{bn_prefix}.running_mean"))?;
        let var = self.w.get(&format!("{bn_prefix}.running_var"))?;
        Ok(conv2d_batchnorm(
            x, c_in, h, w, weight, None, c_out, k, k, stride_h, stride_w, pad, pad, gamma, beta,
            mean, var, BN_EPS, relu,
        ))
    }
}

fn profile_enabled() -> bool {
    std::env::var("OPENASR_WESPEAKER_PROFILE")
        .map(|value| {
            let value = value.trim();
            !value.is_empty() && value != "0" && !value.eq_ignore_ascii_case("false")
        })
        .unwrap_or(false)
}

fn log_profile(enabled: bool, label: &str, started: Option<Instant>, values: usize) {
    if enabled && let Some(started) = started {
        eprintln!(
            "openasr_wespeaker_profile stage={label} elapsed_ms={:.3} values={values}",
            started.elapsed().as_secs_f64() * 1000.0
        );
    }
}

fn conv_same_padding_stride_len(input: usize, stride: usize) -> usize {
    if input == 0 {
        0
    } else {
        input.div_ceil(stride)
    }
}

fn validate_wespeaker_schema(w: &Weights) -> Result<(), WeightsError> {
    let expected = expected_wespeaker_schema();
    let expected_names: BTreeSet<&str> =
        expected.iter().map(|tensor| tensor.name.as_str()).collect();
    for tensor in &expected {
        let got = w.shape(&tensor.name)?;
        if !tensor.shapes.iter().any(|want| got == want.as_slice()) {
            return Err(WeightsError::ShapeMismatch {
                name: tensor.name.clone(),
                got: got.to_vec(),
                want: tensor.shapes[0].clone(),
            });
        }
    }
    for name in w.names() {
        if !expected_names.contains(name) {
            return Err(WeightsError::Unexpected(name.to_string()));
        }
    }
    Ok(())
}

fn expected_wespeaker_schema() -> Vec<ExpectedTensor> {
    let mut schema = Vec::with_capacity(182);
    push_tensor(&mut schema, "resnet.conv1.weight", &[32, 1, 3, 3]);
    push_batchnorm(&mut schema, "resnet.bn1", 32);

    let mut in_channels = STAGE_CHANNELS[0];
    for (stage_idx, (&blocks, &out_channels)) in
        STAGE_BLOCKS.iter().zip(STAGE_CHANNELS.iter()).enumerate()
    {
        for block_idx in 0..blocks {
            let stride = if block_idx == 0 {
                STAGE_STRIDES[stage_idx]
            } else {
                1
            };
            let prefix = format!("resnet.layer{}.{}", stage_idx + 1, block_idx);
            push_tensor(
                &mut schema,
                &format!("{prefix}.conv1.weight"),
                &[out_channels, in_channels, 3, 3],
            );
            push_batchnorm(&mut schema, &format!("{prefix}.bn1"), out_channels);
            push_tensor(
                &mut schema,
                &format!("{prefix}.conv2.weight"),
                &[out_channels, out_channels, 3, 3],
            );
            push_batchnorm(&mut schema, &format!("{prefix}.bn2"), out_channels);
            if stride != 1 || in_channels != out_channels {
                push_tensor_with_alternates(
                    &mut schema,
                    &format!("{prefix}.shortcut.0.weight"),
                    &[
                        vec![out_channels, in_channels, 1, 1],
                        // GGUF/ggml reports trailing singleton dimensions as a
                        // lower-rank tensor on readback; the flat 1x1 kernel
                        // payload is identical and still schema-checked by name
                        // and element count.
                        vec![out_channels, in_channels],
                    ],
                );
                push_batchnorm(&mut schema, &format!("{prefix}.shortcut.1"), out_channels);
            }
            in_channels = out_channels;
        }
    }
    push_tensor(
        &mut schema,
        "resnet.seg_1.weight",
        &[EMBEDDING_DIM, STATS_POOL_DIM],
    );
    push_tensor(&mut schema, "resnet.seg_1.bias", &[EMBEDDING_DIM]);
    debug_assert_eq!(schema.len(), 182);
    schema
}

fn push_batchnorm(schema: &mut Vec<ExpectedTensor>, prefix: &str, channels: usize) {
    for suffix in ["weight", "bias", "running_mean", "running_var"] {
        push_tensor(schema, &format!("{prefix}.{suffix}"), &[channels]);
    }
}

fn push_tensor(schema: &mut Vec<ExpectedTensor>, name: &str, shape: &[usize]) {
    push_tensor_with_alternates(schema, name, &[shape.to_vec()]);
}

fn push_tensor_with_alternates(
    schema: &mut Vec<ExpectedTensor>,
    name: &str,
    shapes: &[Vec<usize>],
) {
    schema.push(ExpectedTensor {
        name: name.to_string(),
        shapes: shapes.to_vec(),
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn safetensors_with_shapes(shapes: &[(&str, &[usize])]) -> Vec<u8> {
        let mut header = serde_json::Map::new();
        let mut offset = 0usize;
        for (name, shape) in shapes {
            let floats = shape.iter().product::<usize>();
            let end = offset + floats * 4;
            header.insert(
                (*name).to_string(),
                serde_json::json!({
                    "dtype": "F32",
                    "shape": shape,
                    "data_offsets": [offset, end],
                }),
            );
            offset = end;
        }
        let header_bytes = serde_json::Value::Object(header).to_string().into_bytes();
        let mut bytes = Vec::with_capacity(8 + header_bytes.len() + offset);
        bytes.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&header_bytes);
        bytes.resize(bytes.len() + offset, 0);
        bytes
    }

    #[test]
    fn schema_rejects_pack_that_only_has_old_seg1_gate_tensor() {
        let bytes = safetensors_with_shapes(&[
            ("resnet.seg_1.weight", &[EMBEDDING_DIM, STATS_POOL_DIM]),
            ("resnet.seg_1.bias", &[EMBEDDING_DIM]),
        ]);
        let error = match WeSpeakerResNet34Model::from_safetensors(&bytes) {
            Ok(_) => panic!("seg_1-only pack must not load"),
            Err(error) => error,
        };
        assert!(
            error
                .to_string()
                .contains("missing tensor 'resnet.conv1.weight'"),
            "{error}"
        );
    }

    #[test]
    fn post_stride_time_len_requires_two_frames_for_stats_pool() {
        assert_eq!(WeSpeakerResNet34Model::post_stride_time_len(0), 0);
        assert_eq!(WeSpeakerResNet34Model::post_stride_time_len(8), 1);
        assert_eq!(WeSpeakerResNet34Model::post_stride_time_len(9), 2);
    }
}
