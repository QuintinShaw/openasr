use crate::GgufTensorDataReader;
use crate::models::mapped_token_embedding::{
    MappedTokenEmbeddingError, MappedTokenEmbeddingTable,
    load_mapped_token_embedding_table_from_reader,
};

use super::runtime_contract::Qwen3AsrExecutionMetadata;
use super::tensor_names::TOKEN_EMBD_WEIGHT;

pub(crate) fn load_qwen3_token_embedding_table_from_reader(
    reader: &GgufTensorDataReader,
    metadata: Qwen3AsrExecutionMetadata,
) -> Result<MappedTokenEmbeddingTable, MappedTokenEmbeddingError> {
    load_mapped_token_embedding_table_from_reader(
        reader,
        TOKEN_EMBD_WEIGHT,
        metadata.llm_d_model,
        metadata.vocab_size,
    )
}
