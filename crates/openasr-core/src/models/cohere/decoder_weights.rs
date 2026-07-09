use crate::ggml_runtime::GgufTensorDataReader;

use super::runtime_contract::CohereTranscribeExecutionMetadata;
use super::tensor_names::{
    DEC_EMB_LN_BIAS, DEC_EMB_LN_WEIGHT, DEC_EMB_WEIGHT, DEC_HEAD_BIAS, DEC_HEAD_WEIGHT,
    DEC_OUT_LN_BIAS, DEC_OUT_LN_WEIGHT, DEC_POS_WEIGHT, decoder_layer_tensor_names,
};
use super::weights::{
    CohereMatrixWeight, CohereVectorWeight, CohereWeightLoadError, load_embedding_weight,
    load_embedding_weight_for_runtime, load_matrix_weight, load_matrix_weight_for_runtime,
    load_vector_weight, load_vector_weight_for_runtime,
};

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(crate) struct CohereDecoderLayerWeights {
    pub attn_ln_weight: CohereVectorWeight,
    pub attn_ln_bias: CohereVectorWeight,
    pub attn_q_weight: CohereMatrixWeight,
    pub attn_q_bias: CohereVectorWeight,
    pub attn_k_weight: CohereMatrixWeight,
    pub attn_k_bias: CohereVectorWeight,
    pub attn_v_weight: CohereMatrixWeight,
    pub attn_v_bias: CohereVectorWeight,
    pub attn_o_weight: CohereMatrixWeight,
    pub attn_o_bias: CohereVectorWeight,
    pub cross_ln_weight: CohereVectorWeight,
    pub cross_ln_bias: CohereVectorWeight,
    pub cross_q_weight: CohereMatrixWeight,
    pub cross_q_bias: CohereVectorWeight,
    pub cross_k_weight: CohereMatrixWeight,
    pub cross_k_bias: CohereVectorWeight,
    pub cross_v_weight: CohereMatrixWeight,
    pub cross_v_bias: CohereVectorWeight,
    pub cross_o_weight: CohereMatrixWeight,
    pub cross_o_bias: CohereVectorWeight,
    pub ffn_ln_weight: CohereVectorWeight,
    pub ffn_ln_bias: CohereVectorWeight,
    pub ffn_up_weight: CohereMatrixWeight,
    pub ffn_up_bias: CohereVectorWeight,
    pub ffn_down_weight: CohereMatrixWeight,
    pub ffn_down_bias: CohereVectorWeight,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(crate) struct CohereTranscribeDecoderWeights {
    pub token_embedding: CohereMatrixWeight,
    pub positional_embedding: CohereMatrixWeight,
    pub emb_ln_weight: CohereVectorWeight,
    pub emb_ln_bias: CohereVectorWeight,
    pub out_ln_weight: CohereVectorWeight,
    pub out_ln_bias: CohereVectorWeight,
    pub output_head_weight: CohereMatrixWeight,
    pub output_head_bias: CohereVectorWeight,
    pub layers: Vec<CohereDecoderLayerWeights>,
}

pub(crate) type CohereDecoderWeightsError = CohereWeightLoadError;

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn load_cohere_transcribe_decoder_weights_from_reader(
    reader: &GgufTensorDataReader,
    metadata: CohereTranscribeExecutionMetadata,
) -> Result<CohereTranscribeDecoderWeights, CohereDecoderWeightsError> {
    load_cohere_transcribe_decoder_weights_impl(reader, metadata, false)
}

pub(crate) fn load_cohere_transcribe_decoder_weights_for_runtime_from_reader(
    reader: &GgufTensorDataReader,
    metadata: CohereTranscribeExecutionMetadata,
) -> Result<CohereTranscribeDecoderWeights, CohereDecoderWeightsError> {
    load_cohere_transcribe_decoder_weights_impl(reader, metadata, true)
}

fn load_cohere_transcribe_decoder_weights_impl(
    reader: &GgufTensorDataReader,
    metadata: CohereTranscribeExecutionMetadata,
    runtime_ready: bool,
) -> Result<CohereTranscribeDecoderWeights, CohereDecoderWeightsError> {
    let d_model = metadata.decoder_d_model;
    let vocab_size = metadata.vocab_size;
    let max_ctx = metadata.decoder_max_context;
    let ffn_dim = metadata.decoder_ffn_dim;
    let matrix_loader = |tensor_name: String, rows: usize, cols: usize, allow_lazy: bool| {
        if runtime_ready && allow_lazy {
            load_matrix_weight_for_runtime(reader, &tensor_name, rows, cols)
        } else {
            load_matrix_weight(reader, &tensor_name, rows, cols)
        }
    };
    let vector_loader = |tensor_name: &str, len: usize| {
        if runtime_ready {
            load_vector_weight_for_runtime(reader, tensor_name, len)
        } else {
            load_vector_weight(reader, tensor_name, len)
        }
    };

    let token_embedding = if runtime_ready {
        load_embedding_weight_for_runtime(reader, DEC_EMB_WEIGHT, vocab_size, d_model)?
    } else {
        load_embedding_weight(reader, DEC_EMB_WEIGHT, vocab_size, d_model)?
    };
    let positional_embedding = if runtime_ready {
        load_embedding_weight_for_runtime(reader, DEC_POS_WEIGHT, max_ctx, d_model)?
    } else {
        load_embedding_weight(reader, DEC_POS_WEIGHT, max_ctx, d_model)?
    };
    let emb_ln_weight = vector_loader(DEC_EMB_LN_WEIGHT, d_model)?;
    let emb_ln_bias = vector_loader(DEC_EMB_LN_BIAS, d_model)?;
    let out_ln_weight = vector_loader(DEC_OUT_LN_WEIGHT, d_model)?;
    let out_ln_bias = vector_loader(DEC_OUT_LN_BIAS, d_model)?;
    let output_head_weight = matrix_loader(DEC_HEAD_WEIGHT.to_string(), vocab_size, d_model, true)?;
    let output_head_bias = vector_loader(DEC_HEAD_BIAS, vocab_size)?;

    let mut layers = Vec::with_capacity(metadata.decoder_layers);
    for layer_idx in 0..metadata.decoder_layers {
        let names = decoder_layer_tensor_names(layer_idx);
        layers.push(CohereDecoderLayerWeights {
            attn_ln_weight: vector_loader(&names.attn_ln_weight, d_model)?,
            attn_ln_bias: vector_loader(&names.attn_ln_bias, d_model)?,
            attn_q_weight: matrix_loader(names.attn_q_weight, d_model, d_model, true)?,
            attn_q_bias: vector_loader(&names.attn_q_bias, d_model)?,
            attn_k_weight: matrix_loader(names.attn_k_weight, d_model, d_model, true)?,
            attn_k_bias: vector_loader(&names.attn_k_bias, d_model)?,
            attn_v_weight: matrix_loader(names.attn_v_weight, d_model, d_model, true)?,
            attn_v_bias: vector_loader(&names.attn_v_bias, d_model)?,
            attn_o_weight: matrix_loader(names.attn_o_weight, d_model, d_model, true)?,
            attn_o_bias: vector_loader(&names.attn_o_bias, d_model)?,
            cross_ln_weight: vector_loader(&names.cross_ln_weight, d_model)?,
            cross_ln_bias: vector_loader(&names.cross_ln_bias, d_model)?,
            cross_q_weight: matrix_loader(names.cross_q_weight, d_model, d_model, true)?,
            cross_q_bias: vector_loader(&names.cross_q_bias, d_model)?,
            cross_k_weight: matrix_loader(names.cross_k_weight, d_model, d_model, true)?,
            cross_k_bias: vector_loader(&names.cross_k_bias, d_model)?,
            cross_v_weight: matrix_loader(names.cross_v_weight, d_model, d_model, true)?,
            cross_v_bias: vector_loader(&names.cross_v_bias, d_model)?,
            cross_o_weight: matrix_loader(names.cross_o_weight, d_model, d_model, true)?,
            cross_o_bias: vector_loader(&names.cross_o_bias, d_model)?,
            ffn_ln_weight: vector_loader(&names.ffn_ln_weight, d_model)?,
            ffn_ln_bias: vector_loader(&names.ffn_ln_bias, d_model)?,
            ffn_up_weight: matrix_loader(names.ffn_up_weight, ffn_dim, d_model, true)?,
            ffn_up_bias: vector_loader(&names.ffn_up_bias, ffn_dim)?,
            ffn_down_weight: matrix_loader(names.ffn_down_weight, d_model, ffn_dim, true)?,
            ffn_down_bias: vector_loader(&names.ffn_down_bias, d_model)?,
        });
    }

    Ok(CohereTranscribeDecoderWeights {
        token_embedding,
        positional_embedding,
        emb_ln_weight,
        emb_ln_bias,
        out_ln_weight,
        out_ln_bias,
        output_head_weight,
        output_head_bias,
        layers,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::cohere::tensor_names::{DEC_HEAD_WEIGHT, decoder_layer_tensor_names};
    use crate::models::cohere::weights::CohereMatrixLayout;
    use crate::models::runtime_preflight::build_runtime_tensor_reader_from_preflight;
    use crate::testing::{TinyGgufFixtureSpec, write_tiny_gguf_runtime_source};
    use crate::validate_ggml_runtime_source_path;
    use crate::{
        GgmlAsrRuntimeSourcePreflight, read_gguf_metadata_from_runtime_source,
        read_gguf_tensor_index_from_runtime_source,
    };
    use std::sync::Arc;
    use tempfile::{NamedTempFile, TempPath};

    fn write_runtime_ready_preflight() -> (TempPath, GgmlAsrRuntimeSourcePreflight) {
        let file = NamedTempFile::new().expect("temp file");
        let persisted = file.into_temp_path();
        let spec = TinyGgufFixtureSpec::cohere_oasr_v1_runtime_ready("cohere-runtime-fixture");
        write_tiny_gguf_runtime_source(&persisted, &spec).expect("write fixture");

        let runtime_source =
            validate_ggml_runtime_source_path(&persisted).expect("valid runtime source path");
        let metadata =
            read_gguf_metadata_from_runtime_source(&runtime_source).expect("read gguf metadata");
        let tensor_index = read_gguf_tensor_index_from_runtime_source(&runtime_source)
            .expect("read gguf tensor index");
        (
            persisted,
            GgmlAsrRuntimeSourcePreflight {
                runtime_source,
                metadata: Arc::new(metadata),
                tensor_index: Arc::new(tensor_index),
            },
        )
    }

    #[test]
    fn loads_decoder_weights_from_runtime_ready_fixture() {
        let (_runtime_path, preflight) = write_runtime_ready_preflight();
        let reader = build_runtime_tensor_reader_from_preflight(&preflight).expect("tensor reader");
        let metadata = super::super::runtime_contract::parse_cohere_transcribe_execution_metadata(
            &preflight.metadata,
        )
        .expect("parse metadata");

        let weights =
            load_cohere_transcribe_decoder_weights_from_reader(&reader, metadata).expect("weights");
        assert_eq!(weights.layers.len(), 2);
        assert_eq!(weights.token_embedding.rows, 32);
        assert_eq!(weights.token_embedding.cols, 16);
        assert_eq!(weights.output_head_bias.values.len(), 32);
        assert_eq!(weights.layers[0].ffn_up_bias.values.len(), 32);
        assert_eq!(
            weights.layers[1].cross_o_weight.layout,
            CohereMatrixLayout::ColumnsByRows
        );
        assert!(weights.output_head_weight.raw_ggml.is_some());
    }

    #[test]
    fn accepts_transposed_matrix_layout_for_decoder_head() {
        let file = NamedTempFile::new().expect("temp file");
        let persisted = file.into_temp_path();
        let spec = TinyGgufFixtureSpec::cohere_oasr_v1_runtime_ready("cohere-runtime-fixture")
            .with_tensor_shape(DEC_HEAD_WEIGHT, [16_u64, 32_u64]);
        write_tiny_gguf_runtime_source(&persisted, &spec).expect("write fixture");

        let runtime_source =
            validate_ggml_runtime_source_path(&persisted).expect("valid runtime source path");
        let metadata =
            read_gguf_metadata_from_runtime_source(&runtime_source).expect("read gguf metadata");
        let tensor_index = read_gguf_tensor_index_from_runtime_source(&runtime_source)
            .expect("read gguf tensor index");
        let preflight = GgmlAsrRuntimeSourcePreflight {
            runtime_source,
            metadata: Arc::new(metadata),
            tensor_index: Arc::new(tensor_index),
        };
        let reader = build_runtime_tensor_reader_from_preflight(&preflight).expect("tensor reader");
        let metadata = super::super::runtime_contract::parse_cohere_transcribe_execution_metadata(
            &preflight.metadata,
        )
        .expect("parse metadata");

        let weights =
            load_cohere_transcribe_decoder_weights_from_reader(&reader, metadata).expect("weights");
        assert_eq!(
            weights.output_head_weight.layout,
            CohereMatrixLayout::ColumnsByRows
        );
        assert!(weights.output_head_weight.raw_ggml.is_some());
    }

    #[test]
    fn rejects_missing_decoder_tensor() {
        let file = NamedTempFile::new().expect("temp file");
        let persisted = file.into_temp_path();
        let layer0 = decoder_layer_tensor_names(0);
        let spec = TinyGgufFixtureSpec::cohere_oasr_v1_runtime_ready("cohere-runtime-fixture")
            .without_tensor(&layer0.attn_q_weight);
        write_tiny_gguf_runtime_source(&persisted, &spec).expect("write fixture");

        let runtime_source =
            validate_ggml_runtime_source_path(&persisted).expect("valid runtime source path");
        let metadata =
            read_gguf_metadata_from_runtime_source(&runtime_source).expect("read gguf metadata");
        let tensor_index = read_gguf_tensor_index_from_runtime_source(&runtime_source)
            .expect("read gguf tensor index");
        let preflight = GgmlAsrRuntimeSourcePreflight {
            runtime_source,
            metadata: Arc::new(metadata),
            tensor_index: Arc::new(tensor_index),
        };
        let reader = build_runtime_tensor_reader_from_preflight(&preflight).expect("tensor reader");
        let metadata = super::super::runtime_contract::parse_cohere_transcribe_execution_metadata(
            &preflight.metadata,
        )
        .expect("parse metadata");

        let error = load_cohere_transcribe_decoder_weights_from_reader(&reader, metadata)
            .expect_err("missing tensor must fail");
        assert!(matches!(
            error,
            CohereDecoderWeightsError::InvalidTensorShape { ref tensor_name, .. }
                if tensor_name == "dec.blk.0.attn_q.weight"
        ));
    }
}
