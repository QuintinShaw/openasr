use thiserror::Error;

use crate::{GgufTensorDataReadError, GgufTensorDataReader};

use super::ggml_tensor_binding::{
    WhisperGgufEncoderLayerTensorBindings, WhisperGgufTensorBinding, WhisperGgufTensorBindings,
    WhisperGgufTensorSlot,
};

const GGML_TYPE_F32: i32 = 0;
const GGML_TYPE_F16: i32 = 1;
const GGML_TYPE_Q8_0: i32 = 8;
const GGML_TYPE_Q4_K: i32 = 12;

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum WhisperMaterializedTensorPayload {
    F32(Vec<f32>),
    F16Bits(Vec<u16>),
    Quantized { ggml_type: i32, bytes: Vec<u8> },
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct WhisperMaterializedTensor {
    pub slot: WhisperGgufTensorSlot,
    pub tensor_name: String,
    pub dims: Vec<u64>,
    pub num_elements: usize,
    pub payload: WhisperMaterializedTensorPayload,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct WhisperEncoderPreludeWeightBundle {
    pub conv1_weight: WhisperMaterializedTensor,
    pub conv1_bias: WhisperMaterializedTensor,
    pub conv2_weight: WhisperMaterializedTensor,
    pub conv2_bias: WhisperMaterializedTensor,
    pub positional_embedding: WhisperMaterializedTensor,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct WhisperEncoderLayerWeightBundle {
    pub layer_idx: usize,
    pub self_attn_layer_norm_weight: WhisperMaterializedTensor,
    pub self_attn_layer_norm_bias: WhisperMaterializedTensor,
    pub self_attn_q_weight: WhisperMaterializedTensor,
    pub self_attn_q_bias: WhisperMaterializedTensor,
    pub self_attn_k_weight: WhisperMaterializedTensor,
    pub self_attn_v_weight: WhisperMaterializedTensor,
    pub self_attn_v_bias: WhisperMaterializedTensor,
    pub self_attn_out_weight: WhisperMaterializedTensor,
    pub self_attn_out_bias: WhisperMaterializedTensor,
    pub mlp_norm_weight: WhisperMaterializedTensor,
    pub mlp_norm_bias: WhisperMaterializedTensor,
    pub fc1_weight: WhisperMaterializedTensor,
    pub fc1_bias: WhisperMaterializedTensor,
    pub fc2_weight: WhisperMaterializedTensor,
    pub fc2_bias: WhisperMaterializedTensor,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct WhisperEncoderFinalNormWeightBundle {
    pub weight: WhisperMaterializedTensor,
    pub bias: WhisperMaterializedTensor,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct WhisperEncoderWeightBundle {
    pub prelude: WhisperEncoderPreludeWeightBundle,
    pub layers: Vec<WhisperEncoderLayerWeightBundle>,
    pub final_norm: WhisperEncoderFinalNormWeightBundle,
}

impl WhisperEncoderWeightBundle {
    pub(crate) fn materialized_tensor_count(&self) -> usize {
        let prelude_count = 5;
        let layer_count = self.layers.len().saturating_mul(15);
        let final_norm_count = 2;
        prelude_count + layer_count + final_norm_count
    }
}

#[derive(Debug, Error)]
pub(crate) enum WhisperEncoderWeightMaterializationError {
    #[error("whisper encoder binding invariant failed: {reason}")]
    BindingInvariant { reason: String },
    #[error(
        "whisper encoder tensor '{tensor_name}' for slot '{slot}' changed type between binding and materialization: expected={expected_type} ({expected_type_name}), actual={actual_type} ({actual_type_name})"
    )]
    BindingTypeMismatch {
        slot: String,
        tensor_name: String,
        expected_type: i32,
        expected_type_name: String,
        actual_type: i32,
        actual_type_name: String,
    },
    #[error(
        "whisper encoder tensor '{tensor_name}' for slot '{slot}' changed shape between binding and materialization: expected={expected_shape:?}, actual={actual_shape:?}"
    )]
    BindingShapeMismatch {
        slot: String,
        tensor_name: String,
        expected_shape: Vec<u64>,
        actual_shape: Vec<u64>,
    },
    #[error(
        "whisper encoder tensor '{tensor_name}' for slot '{slot}' has unsupported type {ggml_type} ({type_name}); only f32/f16/q8_0/q4_K are supported"
    )]
    UnsupportedTensorType {
        slot: String,
        tensor_name: String,
        ggml_type: i32,
        type_name: String,
    },
    #[error(
        "whisper encoder tensor materialization read failed for slot '{slot}' ('{tensor_name}'): {source}"
    )]
    TensorRead {
        slot: String,
        tensor_name: String,
        #[source]
        source: Box<GgufTensorDataReadError>,
    },
}

pub(crate) fn materialize_whisper_encoder_weight_bundle(
    bindings: &WhisperGgufTensorBindings,
    reader: &GgufTensorDataReader,
) -> Result<WhisperEncoderWeightBundle, WhisperEncoderWeightMaterializationError> {
    let encoder = bindings.encoder();
    if encoder.layers.len() != encoder.n_audio_layer {
        return Err(WhisperEncoderWeightMaterializationError::BindingInvariant {
            reason: format!(
                "encoder layer count mismatch: expected {}, got {}",
                encoder.n_audio_layer,
                encoder.layers.len()
            ),
        });
    }
    let prelude = WhisperEncoderPreludeWeightBundle {
        conv1_weight: materialize_binding(&encoder.prelude.conv1_weight, reader)?,
        conv1_bias: materialize_binding(&encoder.prelude.conv1_bias, reader)?,
        conv2_weight: materialize_binding(&encoder.prelude.conv2_weight, reader)?,
        conv2_bias: materialize_binding(&encoder.prelude.conv2_bias, reader)?,
        positional_embedding: materialize_binding(&encoder.prelude.positional_embedding, reader)?,
    };

    let mut layers = Vec::with_capacity(encoder.layers.len());
    for layer in &encoder.layers {
        layers.push(materialize_layer(layer, reader)?);
    }

    let final_norm = WhisperEncoderFinalNormWeightBundle {
        weight: materialize_binding(&encoder.final_layer_norm_weight, reader)?,
        bias: materialize_binding(&encoder.final_layer_norm_bias, reader)?,
    };
    Ok(WhisperEncoderWeightBundle {
        prelude,
        layers,
        final_norm,
    })
}

fn materialize_layer(
    layer: &WhisperGgufEncoderLayerTensorBindings,
    reader: &GgufTensorDataReader,
) -> Result<WhisperEncoderLayerWeightBundle, WhisperEncoderWeightMaterializationError> {
    Ok(WhisperEncoderLayerWeightBundle {
        layer_idx: layer.layer_idx,
        self_attn_layer_norm_weight: materialize_binding(
            &layer.self_attn_layer_norm_weight,
            reader,
        )?,
        self_attn_layer_norm_bias: materialize_binding(&layer.self_attn_layer_norm_bias, reader)?,
        self_attn_q_weight: materialize_binding(&layer.self_attn_q_weight, reader)?,
        self_attn_q_bias: materialize_binding(&layer.self_attn_q_bias, reader)?,
        self_attn_k_weight: materialize_binding(&layer.self_attn_k_weight, reader)?,
        self_attn_v_weight: materialize_binding(&layer.self_attn_v_weight, reader)?,
        self_attn_v_bias: materialize_binding(&layer.self_attn_v_bias, reader)?,
        self_attn_out_weight: materialize_binding(&layer.self_attn_out_weight, reader)?,
        self_attn_out_bias: materialize_binding(&layer.self_attn_out_bias, reader)?,
        mlp_norm_weight: materialize_binding(&layer.mlp_norm_weight, reader)?,
        mlp_norm_bias: materialize_binding(&layer.mlp_norm_bias, reader)?,
        fc1_weight: materialize_binding(&layer.fc1_weight, reader)?,
        fc1_bias: materialize_binding(&layer.fc1_bias, reader)?,
        fc2_weight: materialize_binding(&layer.fc2_weight, reader)?,
        fc2_bias: materialize_binding(&layer.fc2_bias, reader)?,
    })
}

fn materialize_binding(
    binding: &WhisperGgufTensorBinding,
    reader: &GgufTensorDataReader,
) -> Result<WhisperMaterializedTensor, WhisperEncoderWeightMaterializationError> {
    let slot = binding.slot.label();
    let tensor_name = binding.resolved_name.clone();
    if let Some(reader_metadata) = reader.tensor_index().get(&tensor_name) {
        if reader_metadata.ggml_type != binding.metadata.ggml_type {
            return Err(
                WhisperEncoderWeightMaterializationError::BindingTypeMismatch {
                    slot,
                    tensor_name,
                    expected_type: binding.metadata.ggml_type,
                    expected_type_name: binding.metadata.type_name.clone(),
                    actual_type: reader_metadata.ggml_type,
                    actual_type_name: reader_metadata.type_name.clone(),
                },
            );
        }
        if reader_metadata.dims != binding.metadata.dims {
            return Err(
                WhisperEncoderWeightMaterializationError::BindingShapeMismatch {
                    slot,
                    tensor_name,
                    expected_shape: binding.metadata.dims.clone(),
                    actual_shape: reader_metadata.dims.clone(),
                },
            );
        }
    }

    let num_elements = binding
        .metadata
        .num_elements()
        .ok_or_else(
            || WhisperEncoderWeightMaterializationError::BindingInvariant {
                reason: format!(
                    "tensor '{}' element count overflow for dims {:?}",
                    binding.resolved_name, binding.metadata.dims
                ),
            },
        )
        .and_then(|value| {
            usize::try_from(value).map_err(|_| {
                WhisperEncoderWeightMaterializationError::BindingInvariant {
                    reason: format!(
                        "tensor '{}' element count {value} does not fit usize",
                        binding.resolved_name
                    ),
                }
            })
        })?;

    let payload = match binding.metadata.ggml_type {
        GGML_TYPE_F32 => reader
            .host_tensor_f32_copy_by_name(&binding.resolved_name, &binding.metadata.dims)
            .map(WhisperMaterializedTensorPayload::F32),
        GGML_TYPE_F16 => reader
            .host_tensor_f16_bits_copy_by_name(&binding.resolved_name, &binding.metadata.dims)
            .map(WhisperMaterializedTensorPayload::F16Bits),
        GGML_TYPE_Q8_0 | GGML_TYPE_Q4_K => reader
            .host_tensor_bytes_copy_by_name(&binding.resolved_name)
            .map(|bytes| WhisperMaterializedTensorPayload::Quantized {
                ggml_type: binding.metadata.ggml_type,
                bytes,
            }),
        _ => {
            return Err(
                WhisperEncoderWeightMaterializationError::UnsupportedTensorType {
                    slot,
                    tensor_name,
                    ggml_type: binding.metadata.ggml_type,
                    type_name: binding.metadata.type_name.clone(),
                },
            );
        }
    }
    .map_err(
        |source| WhisperEncoderWeightMaterializationError::TensorRead {
            slot,
            tensor_name,
            source: Box::new(source),
        },
    )?;

    Ok(WhisperMaterializedTensor {
        slot: binding.slot.clone(),
        tensor_name: binding.resolved_name.clone(),
        dims: binding.metadata.dims.clone(),
        num_elements,
        payload,
    })
}

#[cfg(test)]
mod tests {
    use tempfile::NamedTempFile;

    use super::*;
    use crate::testing::{TinyGgufFixtureSpec, write_tiny_gguf_runtime_source};
    use crate::{GgufTensorDataReader, read_gguf_tensor_index};

    #[test]
    fn tiny_valid_bundle_materializes_successfully() {
        let spec = TinyGgufFixtureSpec::whisper_oasr_v1_encoder_graph_one_layer("whisper-fixture");
        let file = write_spec(&spec);
        let bindings = bind_for_test(file.path(), &one_layer_context());
        let reader = GgufTensorDataReader::from_path(file.path()).expect("create tensor reader");

        let bundle =
            materialize_whisper_encoder_weight_bundle(&bindings, &reader).expect("materialize");

        assert_eq!(bundle.layers.len(), 1);
        assert_eq!(bundle.materialized_tensor_count(), 22);
        match &bundle.prelude.conv1_weight.payload {
            WhisperMaterializedTensorPayload::F32(values) => assert_eq!(values.len(), 96),
            payload => panic!("unexpected payload kind: {payload:?}"),
        }
    }

    #[test]
    fn missing_tensor_fails_with_read_error() {
        let base_spec =
            TinyGgufFixtureSpec::whisper_oasr_v1_encoder_graph_one_layer("whisper-fixture");
        let binding_file = write_spec(&base_spec);
        let bindings = bind_for_test(binding_file.path(), &one_layer_context());
        let reader_file = write_spec(
            &base_spec.with_whisper_missing_required_tensor("model.encoder.layers.0.fc2.weight"),
        );
        let reader =
            GgufTensorDataReader::from_path(reader_file.path()).expect("create tensor reader");

        let error = materialize_whisper_encoder_weight_bundle(&bindings, &reader)
            .expect_err("missing tensor must fail closed");
        assert!(matches!(
            error,
            WhisperEncoderWeightMaterializationError::TensorRead {
                source,
                ..
            } if matches!(source.as_ref(), GgufTensorDataReadError::TensorNotFound { .. })
        ));
    }

    #[test]
    fn shape_mismatch_fails_with_binding_shape_error() {
        let base_spec =
            TinyGgufFixtureSpec::whisper_oasr_v1_encoder_graph_one_layer("whisper-fixture");
        let binding_file = write_spec(&base_spec);
        let bindings = bind_for_test(binding_file.path(), &one_layer_context());
        let reader_file = write_spec(&base_spec.with_whisper_required_tensor_shape_mismatch(
            "model.encoder.layers.0.fc2.weight",
            [8_u64, 8],
        ));
        let reader =
            GgufTensorDataReader::from_path(reader_file.path()).expect("create tensor reader");

        let error = materialize_whisper_encoder_weight_bundle(&bindings, &reader)
            .expect_err("shape mismatch must fail closed");
        assert!(matches!(
            error,
            WhisperEncoderWeightMaterializationError::BindingShapeMismatch { .. }
        ));
    }

    #[test]
    fn type_mismatch_fails_with_binding_type_error() {
        let base_spec =
            TinyGgufFixtureSpec::whisper_oasr_v1_encoder_graph_one_layer("whisper-fixture");
        let binding_file = write_spec(&base_spec);
        let bindings = bind_for_test(binding_file.path(), &one_layer_context());
        let reader_file =
            write_spec(&base_spec.with_tensor_f16("model.encoder.layers.0.fc2.weight"));
        let reader =
            GgufTensorDataReader::from_path(reader_file.path()).expect("create tensor reader");

        let error = materialize_whisper_encoder_weight_bundle(&bindings, &reader)
            .expect_err("type mismatch must fail closed");
        assert!(matches!(
            error,
            WhisperEncoderWeightMaterializationError::BindingTypeMismatch { .. }
        ));
    }

    #[test]
    fn unsupported_tensor_type_fails_closed() {
        let spec = TinyGgufFixtureSpec::whisper_oasr_v1_encoder_graph_one_layer("whisper-fixture")
            .with_tensor_type("model.encoder.layers.0.self_attn.q_proj.weight", 26);
        let file = write_spec(&spec);
        let bindings = bind_for_test(file.path(), &one_layer_context());
        let reader = GgufTensorDataReader::from_path(file.path()).expect("create tensor reader");

        let error = materialize_whisper_encoder_weight_bundle(&bindings, &reader)
            .expect_err("unsupported type must fail closed");
        assert!(matches!(
            error,
            WhisperEncoderWeightMaterializationError::UnsupportedTensorType { .. }
        ));
    }

    fn bind_for_test(
        path: &std::path::Path,
        context: &super::super::ggml_tensor_binding::WhisperGgufTensorBindingContext,
    ) -> WhisperGgufTensorBindings {
        let index = read_gguf_tensor_index(path).expect("read tensor index");
        super::super::ggml_tensor_binding::bind_whisper_gguf_tensors(context, &index)
            .expect("bind whisper tensors")
    }

    fn one_layer_context() -> super::super::ggml_tensor_binding::WhisperGgufTensorBindingContext {
        super::super::ggml_tensor_binding::WhisperGgufTensorBindingContext {
            n_audio_layer: 1,
            n_audio_state: 8,
            n_audio_head: 4,
            n_mels: 4,
            n_audio_ctx: 128,
            n_text_layer: 1,
            n_text_state: 8,
            n_text_head: 4,
            n_text_ctx: 128,
            n_vocab: 64,
        }
    }

    fn write_spec(spec: &TinyGgufFixtureSpec) -> NamedTempFile {
        let file = NamedTempFile::new().expect("temp file");
        write_tiny_gguf_runtime_source(file.path(), spec).expect("write gguf fixture");
        file
    }
}
