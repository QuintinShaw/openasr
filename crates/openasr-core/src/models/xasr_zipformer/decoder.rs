//! Stateless RNN-T predictor for X-ASR.

use super::weights::XasrDecoderWeights;

#[derive(Debug, Clone)]
pub(crate) struct XasrDecoder {
    weights: XasrDecoderWeights,
    context_size: usize,
    blank_id: u32,
}

impl XasrDecoder {
    pub(crate) fn new(weights: XasrDecoderWeights, context_size: usize, blank_id: u32) -> Self {
        Self {
            weights,
            context_size,
            blank_id,
        }
    }

    pub(crate) fn initial_context(&self) -> Vec<u32> {
        vec![self.blank_id; self.context_size]
    }

    pub(crate) fn decode_context(&self, context: &[u32]) -> Result<Vec<f32>, String> {
        if context.len() != self.context_size {
            return Err(format!(
                "xasr decoder expected context_size {}, got {}",
                self.context_size,
                context.len()
            ));
        }
        let dim = self.weights.embedding.input_dim;
        let vocab = self.weights.embedding.output_dim;
        let mut embedded = vec![0.0_f32; dim * self.context_size];
        for (time, &token_id) in context.iter().enumerate() {
            let token = token_id as usize;
            if token >= vocab {
                return Err(format!(
                    "xasr decoder token id {token_id} out of range for vocab {vocab}"
                ));
            }
            let row = &self.weights.embedding.values[token * dim..(token + 1) * dim];
            for channel in 0..dim {
                embedded[channel * self.context_size + time] = row[channel];
            }
        }
        let mut output = grouped_context_conv1d(
            &embedded,
            dim,
            self.context_size,
            &self.weights.conv_weight.values,
            self.weights.groups,
        )?;
        for value in &mut output {
            *value = value.max(0.0);
        }
        Ok(output)
    }
}

fn grouped_context_conv1d(
    embedded_channel_major: &[f32],
    channels: usize,
    context_size: usize,
    weight_output_major: &[f32],
    groups: usize,
) -> Result<Vec<f32>, String> {
    if groups == 0 || !channels.is_multiple_of(groups) {
        return Err(format!(
            "xasr decoder channels {channels} must be divisible by groups {groups}"
        ));
    }
    let in_per_group = channels / groups;
    let out_per_group = channels / groups;
    let expected_input = channels
        .checked_mul(context_size)
        .ok_or_else(|| "xasr decoder input shape overflow".to_string())?;
    if embedded_channel_major.len() != expected_input {
        return Err(format!(
            "xasr decoder embedded input has {} values, expected {expected_input}",
            embedded_channel_major.len()
        ));
    }
    let expected_weight = channels
        .checked_mul(in_per_group)
        .and_then(|value| value.checked_mul(context_size))
        .ok_or_else(|| "xasr decoder conv weight shape overflow".to_string())?;
    if weight_output_major.len() != expected_weight {
        return Err(format!(
            "xasr decoder conv has {} values, expected {expected_weight}",
            weight_output_major.len()
        ));
    }
    let mut output = vec![0.0_f32; channels];
    for (out_channel, output_value) in output.iter_mut().enumerate().take(channels) {
        let group = out_channel / out_per_group;
        let input_base = group * in_per_group;
        let weight_base = out_channel * in_per_group * context_size;
        let mut sum = 0.0_f32;
        for in_group_channel in 0..in_per_group {
            let input_channel = input_base + in_group_channel;
            for t in 0..context_size {
                let input = embedded_channel_major[input_channel * context_size + t];
                let weight = weight_output_major[weight_base + in_group_channel * context_size + t];
                sum += input * weight;
            }
        }
        *output_value = sum;
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ggml_runtime::{GgufTensorDataReader, read_gguf_metadata};
    use crate::models::xasr_zipformer::runtime_contract::parse_xasr_zipformer_execution_metadata;
    use crate::models::xasr_zipformer::weights::{NamedTensor, StoredLinear, XasrDecoderWeights};
    use std::path::Path;

    #[test]
    fn decoder_uses_blank_initial_context() {
        let decoder = XasrDecoder::new(weights(), 2, 0);
        assert_eq!(decoder.initial_context(), vec![0, 0]);
    }

    #[test]
    fn grouped_context_conv_matches_manual_reference() {
        let input = vec![
            1.0, 2.0, // ch0
            3.0, 4.0, // ch1
            5.0, 6.0, // ch2
            7.0, 8.0, // ch3
        ];
        let weights = vec![
            1.0, 0.0, 0.0, 1.0, // out0 sees ch0/ch1
            0.5, 0.5, 0.5, 0.5, // out1 sees ch0/ch1
            1.0, 1.0, 0.0, 0.0, // out2 sees ch2/ch3
            0.0, 0.0, 1.0, 1.0, // out3 sees ch2/ch3
        ];
        let output = grouped_context_conv1d(&input, 4, 2, &weights, 2).unwrap();
        assert_eq!(output, vec![5.0, 5.0, 11.0, 15.0]);
    }

    #[test]
    fn decode_context_gathers_embeddings_in_time_order() {
        let decoder = XasrDecoder::new(weights(), 2, 0);
        let output = decoder.decode_context(&[1, 2]).unwrap();
        assert_eq!(output, vec![10.0, 100.0, 30.0, 300.0]);
    }

    #[test]
    fn decoder_blank_context_matches_onnx_when_pack_present() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tmp/xasr-test/out");
        let onnx_pack = root.join("xasr-zh-en-onnx-fp16.oasr");
        let pack = if onnx_pack.exists() {
            onnx_pack
        } else {
            root.join("xasr-zh-en-fp16.oasr")
        };
        if !pack.exists() {
            eprintln!("skipping: xasr fp16 pack absent at {}", pack.display());
            return;
        }
        let reader = GgufTensorDataReader::from_path(&pack).expect("reader");
        let metadata = read_gguf_metadata(&pack).expect("metadata");
        let metadata = parse_xasr_zipformer_execution_metadata(&metadata).expect("xasr metadata");
        let weights =
            crate::models::xasr_zipformer::weights::load_xasr_decoder_weights(&reader, &metadata)
                .expect("decoder weights");
        let joiner_weights =
            crate::models::xasr_zipformer::weights::load_xasr_joiner_weights(&reader, &metadata)
                .expect("joiner weights");
        let decoder = XasrDecoder::new(weights, metadata.decoder_context_size, metadata.blank_id);
        let pre_projected = decoder
            .decode_context(&decoder.initial_context())
            .expect("decode blank context");
        let output = joiner_weights
            .decoder_proj_weight
            .apply(&pre_projected, Some(&joiner_weights.decoder_proj_bias))
            .expect("decoder projection");
        let expected = [
            0.017111897_f32,
            -0.12831049,
            -0.56158185,
            0.5297903,
            0.27330607,
            0.3115773,
            0.17948422,
            -0.45026723,
        ];
        assert_eq!(output.len(), 512);
        for (index, (&got, &want)) in output.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() <= 2.0e-4,
                "decoder output[{index}] got {got}, want {want}"
            );
        }
    }

    fn weights() -> XasrDecoderWeights {
        XasrDecoderWeights {
            embedding: StoredLinear {
                name: "decoder.EMB.weight".to_string(),
                input_dim: 4,
                output_dim: 3,
                values: vec![
                    0.0, 0.0, 0.0, 0.0, // blank
                    10.0, 20.0, 30.0, 40.0, // token 1
                    100.0, 200.0, 300.0, 400.0, // token 2
                ],
                native: None,
            },
            conv_weight: NamedTensor {
                name: "decoder.conv.weight".to_string(),
                dims: vec![2, 2, 4],
                values: vec![
                    1.0, 0.0, 0.0, 0.0, // out0 group0 ch0/ch1
                    0.0, 1.0, 0.0, 0.0, // out1 group0
                    1.0, 0.0, 0.0, 0.0, // out2 group1 ch2/ch3
                    0.0, 1.0, 0.0, 0.0, // out3 group1
                ],
            },
            groups: 2,
        }
    }
}
