use std::collections::{BTreeMap, BTreeSet};

use crate::ggml_runtime::{GgufTensorDataReader, GgufTensorIndex};

use super::DiariZenSegmenterError;
use super::config::{
    CONFORMER_DIM, CONFORMER_FFN_DIM, CONFORMER_KERNEL, CONFORMER_LAYERS, CONV_CHANNELS,
    CONV_KERNELS, FFN_DIMS, HEAD_DIM, HIDDEN_SIZE, LAYER_REPRESENTATIONS, LOCAL_SPEAKERS,
    POWERSET_CLASSES, RELATIVE_POSITION_BUCKETS, REMAINING_HEADS, TOTAL_HEADS,
};

pub(super) const EXPECTED_TENSOR_COUNT: usize = 594;
const GGUF_MAX_TENSOR_NAME_BYTES: usize = 63;

pub(super) fn runtime_tensor_name(upstream_name: &str) -> String {
    const PREFIXES: [(&str, &str); 4] = [
        ("wavlm_model.feature_extractor.", "dz.fe."),
        ("wavlm_model.encoder.feature_projection.", "dz.fp."),
        ("wavlm_model.encoder.transformer.", "dz.tr."),
        ("conformer.conformer_layer.", "dz.cf."),
    ];
    let name = PREFIXES
        .iter()
        .find_map(|(upstream, runtime)| {
            upstream_name
                .strip_prefix(upstream)
                .map(|suffix| format!("{runtime}{suffix}"))
        })
        .unwrap_or_else(|| upstream_name.to_string());
    debug_assert!(name.len() <= GGUF_MAX_TENSOR_NAME_BYTES);
    name
}

fn insert(tensors: &mut BTreeMap<String, Vec<u64>>, name: impl Into<String>, dims: &[usize]) {
    let name = name.into();
    tensors.insert(
        runtime_tensor_name(&name),
        dims.iter().map(|value| *value as u64).collect(),
    );
}

fn expected_tensors() -> BTreeMap<String, Vec<u64>> {
    let mut tensors = BTreeMap::new();

    insert(&mut tensors, "classifier.bias", &[POWERSET_CLASSES]);
    insert(
        &mut tensors,
        "classifier.weight",
        &[CONFORMER_DIM, POWERSET_CLASSES],
    );
    insert(&mut tensors, "lnorm.bias", &[CONFORMER_DIM]);
    insert(&mut tensors, "lnorm.weight", &[CONFORMER_DIM]);
    insert(&mut tensors, "proj.bias", &[CONFORMER_DIM]);
    insert(&mut tensors, "proj.weight", &[HIDDEN_SIZE, CONFORMER_DIM]);
    insert(&mut tensors, "weight_sum.weight", &[LAYER_REPRESENTATIONS]);

    let mut in_channels = 1;
    for (index, (&out_channels, &kernel)) in
        CONV_CHANNELS.iter().zip(CONV_KERNELS.iter()).enumerate()
    {
        insert(
            &mut tensors,
            format!("wavlm_model.feature_extractor.conv_layers.{index}.conv.weight"),
            &[kernel, in_channels, out_channels],
        );
        for suffix in ["bias", "weight"] {
            insert(
                &mut tensors,
                format!("wavlm_model.feature_extractor.conv_layers.{index}.layer_norm.{suffix}"),
                &[out_channels],
            );
        }
        in_channels = out_channels;
    }
    insert(
        &mut tensors,
        "wavlm_model.feature_extractor.dummy_weight",
        &[CONV_CHANNELS[6]],
    );
    for suffix in ["bias", "weight"] {
        insert(
            &mut tensors,
            format!("wavlm_model.encoder.feature_projection.layer_norm.{suffix}"),
            &[CONV_CHANNELS[6]],
        );
    }
    insert(
        &mut tensors,
        "wavlm_model.encoder.feature_projection.projection.bias",
        &[HIDDEN_SIZE],
    );
    insert(
        &mut tensors,
        "wavlm_model.encoder.feature_projection.projection.weight",
        &[CONV_CHANNELS[6], HIDDEN_SIZE],
    );
    for suffix in ["bias", "weight"] {
        insert(
            &mut tensors,
            format!("wavlm_model.encoder.transformer.layer_norm.{suffix}"),
            &[HIDDEN_SIZE],
        );
    }
    insert(
        &mut tensors,
        "wavlm_model.encoder.transformer.pos_conv_embed.conv.bias",
        &[HIDDEN_SIZE],
    );
    insert(
        &mut tensors,
        "wavlm_model.encoder.transformer.pos_conv_embed.conv.weight",
        &[128, HIDDEN_SIZE / 16, HIDDEN_SIZE],
    );

    for (index, (&heads, &ffn_dim)) in REMAINING_HEADS.iter().zip(FFN_DIMS.iter()).enumerate() {
        let prefix = format!("wavlm_model.encoder.transformer.layers.{index}");
        for norm in ["layer_norm", "final_layer_norm"] {
            for suffix in ["bias", "weight"] {
                insert(
                    &mut tensors,
                    format!("{prefix}.{norm}.{suffix}"),
                    &[HIDDEN_SIZE],
                );
            }
        }
        insert(
            &mut tensors,
            format!("{prefix}.feed_forward.intermediate_dense.bias"),
            &[ffn_dim],
        );
        insert(
            &mut tensors,
            format!("{prefix}.feed_forward.intermediate_dense.weight"),
            &[HIDDEN_SIZE, ffn_dim],
        );
        insert(
            &mut tensors,
            format!("{prefix}.feed_forward.output_dense.bias"),
            &[HIDDEN_SIZE],
        );
        insert(
            &mut tensors,
            format!("{prefix}.feed_forward.output_dense.weight"),
            &[ffn_dim, HIDDEN_SIZE],
        );

        if !heads.is_empty() {
            let attention = format!("{prefix}.attention");
            let projection = heads.len() * HEAD_DIM;
            for name in ["q_proj", "k_proj", "v_proj"] {
                insert(
                    &mut tensors,
                    format!("{attention}.{name}.bias"),
                    &[projection],
                );
                insert(
                    &mut tensors,
                    format!("{attention}.{name}.weight"),
                    &[HIDDEN_SIZE, projection],
                );
            }
            insert(
                &mut tensors,
                format!("{attention}.out_proj.bias"),
                &[HIDDEN_SIZE],
            );
            insert(
                &mut tensors,
                format!("{attention}.out_proj.weight"),
                &[projection, HIDDEN_SIZE],
            );
            insert(
                &mut tensors,
                format!("{attention}.gru_rel_pos_const"),
                &[1, 1, TOTAL_HEADS],
            );
            insert(
                &mut tensors,
                format!("{attention}.gru_rel_pos_linear.bias"),
                &[8],
            );
            insert(
                &mut tensors,
                format!("{attention}.gru_rel_pos_linear.weight"),
                &[HEAD_DIM, 8],
            );
            if index == 0 {
                insert(
                    &mut tensors,
                    format!("{attention}.rel_attn_embed.weight"),
                    &[TOTAL_HEADS, RELATIVE_POSITION_BUCKETS],
                );
            }
        }
    }

    for index in 0..CONFORMER_LAYERS {
        let prefix = format!("conformer.conformer_layer.{index}");
        for ffn in ["ffn1", "ffn2"] {
            for suffix in ["bias", "weight"] {
                insert(
                    &mut tensors,
                    format!("{prefix}.{ffn}.ln_norm.{suffix}"),
                    &[CONFORMER_DIM],
                );
            }
            insert(
                &mut tensors,
                format!("{prefix}.{ffn}.w_1.bias"),
                &[CONFORMER_FFN_DIM],
            );
            insert(
                &mut tensors,
                format!("{prefix}.{ffn}.w_1.weight"),
                &[CONFORMER_DIM, CONFORMER_FFN_DIM],
            );
            insert(
                &mut tensors,
                format!("{prefix}.{ffn}.w_2.bias"),
                &[CONFORMER_DIM],
            );
            insert(
                &mut tensors,
                format!("{prefix}.{ffn}.w_2.weight"),
                &[CONFORMER_FFN_DIM, CONFORMER_DIM],
            );
        }
        for suffix in ["bias", "weight"] {
            insert(
                &mut tensors,
                format!("{prefix}.mha.ln_norm.{suffix}"),
                &[CONFORMER_DIM],
            );
            insert(
                &mut tensors,
                format!("{prefix}.conv.ln_norm.{suffix}"),
                &[CONFORMER_DIM],
            );
            insert(
                &mut tensors,
                format!("{prefix}.ln_norm.{suffix}"),
                &[CONFORMER_DIM],
            );
        }
        for projection in ["linearQ", "linearK", "linearV", "linearO"] {
            insert(
                &mut tensors,
                format!("{prefix}.mha.mha.{projection}.bias"),
                &[CONFORMER_DIM],
            );
            insert(
                &mut tensors,
                format!("{prefix}.mha.mha.{projection}.weight"),
                &[CONFORMER_DIM, CONFORMER_DIM],
            );
        }
        insert(
            &mut tensors,
            format!("{prefix}.conv.pointwise_conv1.bias"),
            &[2 * CONFORMER_DIM],
        );
        insert(
            &mut tensors,
            format!("{prefix}.conv.pointwise_conv1.weight"),
            &[1, CONFORMER_DIM, 2 * CONFORMER_DIM],
        );
        insert(
            &mut tensors,
            format!("{prefix}.conv.depthwise_conv.bias"),
            &[CONFORMER_DIM],
        );
        insert(
            &mut tensors,
            format!("{prefix}.conv.depthwise_conv.weight"),
            &[CONFORMER_KERNEL, 1, CONFORMER_DIM],
        );
        for field in ["bias", "running_mean", "running_var", "weight"] {
            insert(
                &mut tensors,
                format!("{prefix}.conv.bn_norm.{field}"),
                &[CONFORMER_DIM],
            );
        }
        insert(
            &mut tensors,
            format!("{prefix}.conv.pointwise_conv2.bias"),
            &[CONFORMER_DIM],
        );
        insert(
            &mut tensors,
            format!("{prefix}.conv.pointwise_conv2.weight"),
            &[1, CONFORMER_DIM, CONFORMER_DIM],
        );
    }

    debug_assert_eq!(LOCAL_SPEAKERS, 4);
    debug_assert_eq!(tensors.len(), EXPECTED_TENSOR_COUNT);
    tensors
}

pub(super) fn validate_tensor_contract(
    index: &GgufTensorIndex,
) -> Result<(), DiariZenSegmenterError> {
    let expected = expected_tensors();
    let actual_names: BTreeSet<_> = index
        .tensors()
        .iter()
        .map(|tensor| tensor.name.as_str())
        .collect();
    let expected_names: BTreeSet<_> = expected.keys().map(String::as_str).collect();

    if actual_names != expected_names {
        let missing = expected_names
            .difference(&actual_names)
            .copied()
            .collect::<Vec<_>>();
        let unexpected = actual_names
            .difference(&expected_names)
            .copied()
            .collect::<Vec<_>>();
        return Err(DiariZenSegmenterError::TensorSetMismatch {
            expected: EXPECTED_TENSOR_COUNT,
            actual: index.tensors().len(),
            missing: missing.join(", "),
            unexpected: unexpected.join(", "),
        });
    }

    for (name, expected_dims) in expected {
        let tensor = index
            .get(&name)
            .ok_or_else(|| DiariZenSegmenterError::MissingTensor(name.clone()))?;
        if tensor.dims != expected_dims {
            return Err(DiariZenSegmenterError::TensorShapeMismatch {
                name,
                expected: expected_dims,
                actual: tensor.dims.clone(),
            });
        }
        if !matches!(tensor.type_name.as_str(), "f32" | "f16") {
            return Err(DiariZenSegmenterError::UnsupportedTensorType {
                name: tensor.name.clone(),
                tensor_type: tensor.type_name.clone(),
            });
        }
    }
    Ok(())
}

pub(super) fn read_tensor_f32(
    reader: &GgufTensorDataReader,
    name: &str,
) -> Result<Vec<f32>, DiariZenSegmenterError> {
    let tensor = reader
        .tensor_index()
        .get(name)
        .ok_or_else(|| DiariZenSegmenterError::MissingTensor(name.to_string()))?;
    reader
        .host_tensor_f32_copy_dequantized_by_name(name, &tensor.dims)
        .map_err(|source| DiariZenSegmenterError::PackRead(source.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compact_schema_covers_the_full_pinned_tensor_contract() {
        let tensors = expected_tensors();
        assert_eq!(tensors.len(), EXPECTED_TENSOR_COUNT);
        let longest = tensors.keys().map(|name| name.len()).max().unwrap_or(0);
        assert!(longest <= GGUF_MAX_TENSOR_NAME_BYTES, "longest={longest}");
        assert_eq!(
            runtime_tensor_name(
                "wavlm_model.encoder.transformer.layers.0.feed_forward.intermediate_dense.weight"
            ),
            "dz.tr.layers.0.feed_forward.intermediate_dense.weight"
        );
        assert_eq!(
            runtime_tensor_name("conformer.conformer_layer.0.conv.depthwise_conv.weight"),
            "dz.cf.0.conv.depthwise_conv.weight"
        );
    }
}
