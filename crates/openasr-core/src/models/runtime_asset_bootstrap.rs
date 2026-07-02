use thiserror::Error;

use crate::GgufTensorDataReader;

use super::ggml_asr_executor::GgmlAsrRuntimeSourcePreflight;
use super::runtime_preflight::{
    RuntimeSourceTensorReaderError, build_runtime_tensor_reader_from_preflight,
};
use super::runtime_tensor_contract_registry::{
    RuntimeTensorContractMetadata, RuntimeTensorContractRegistryError,
    validate_builtin_runtime_tensor_contract_preflight,
};

#[derive(Debug)]
pub(crate) struct BuiltinRuntimeAssetBootstrap {
    pub metadata: RuntimeTensorContractMetadata,
    pub tensor_reader: GgufTensorDataReader,
}

#[derive(Debug, Error)]
pub(crate) enum BuiltinRuntimeAssetBootstrapError {
    #[error("runtime tensor contract preflight failed: {source}")]
    RuntimeContractPreflight {
        #[source]
        source: RuntimeTensorContractRegistryError,
    },
    #[error("runtime tensor reader build failed: {source}")]
    TensorReaderBuild {
        #[source]
        source: RuntimeSourceTensorReaderError,
    },
}

pub(crate) fn build_builtin_runtime_asset_bootstrap(
    model_architecture: &str,
    preflight: &GgmlAsrRuntimeSourcePreflight,
) -> Result<BuiltinRuntimeAssetBootstrap, BuiltinRuntimeAssetBootstrapError> {
    let metadata =
        validate_builtin_runtime_tensor_contract_preflight(model_architecture, preflight).map_err(
            |source| BuiltinRuntimeAssetBootstrapError::RuntimeContractPreflight { source },
        )?;
    let tensor_reader = build_runtime_tensor_reader_from_preflight(preflight)
        .map_err(|source| BuiltinRuntimeAssetBootstrapError::TensorReaderBuild { source })?;
    Ok(BuiltinRuntimeAssetBootstrap {
        metadata,
        tensor_reader,
    })
}
