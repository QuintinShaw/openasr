mod arena_weight_pipeline;
mod backend;
mod cpu_graph;
mod env_flags;
mod ffi;
mod gguf_c_parser_sandbox;
mod gguf_metadata;
mod gguf_tensor_data;
mod gguf_tensor_index;
mod gguf_write;
mod package_probe;
mod runtime_source;

pub(crate) use arena_weight_pipeline::{
    ArenaAllocError, WeightSlot, alloc_static_f16, alloc_static_f32, bind_loaded,
    upload_static_f16, upload_static_f32,
};
pub(crate) use backend::ensure_backends_loaded;
pub use backend::{
    GgmlBackend, GgmlBackendDevice, GgmlBackendKind, GgmlCpuFeatures, GgmlDeviceMemory,
    GgmlRuntimeError, GgmlRuntimeInfo, ggml_available_devices, ggml_hip_tuning_summary,
    ggml_native_build_enabled, ggml_runtime_boot_summary, ggml_runtime_info,
};
pub use cpu_graph::{
    AutoGpuPolicy, GgmlCpuBinaryOp, GgmlCpuGraphBackend, GgmlCpuGraphConfig, GgmlCpuGraphError,
    GgmlCpuGraphRunner, GgmlCpuGraphThreadingWorkload, RequestBackendOverrideGuard,
    RequestBackendPreference, install_request_backend_override, request_backend_override,
};
pub(crate) use cpu_graph::{
    GgmlCpuGraphBuilder, GgmlCpuTensor, GgmlLoadedTensor, GgmlLoadedWeightContext,
    GgmlPersistentGraphSession, GgmlRopeExtParams, GgmlStaticTensor, GgmlStaticTensorArena,
};
pub(crate) use env_flags::{env_toggle_with_raw, env_var_truthy};
pub(crate) use ffi::{GGML_TYPE_F16, GGML_TYPE_F32};
pub(crate) use gguf_c_parser_sandbox::load_gguf_metadata_and_tensor_index_with_c_parser_sandbox;
pub use gguf_c_parser_sandbox::{
    GGUF_C_PARSER_SANDBOX_HELPER_ARG, GgufCParserSandboxError,
    render_gguf_c_parser_sandbox_child_output,
};
pub use gguf_metadata::{
    GgufMetadata, GgufMetadataReadError, GgufMetadataValue, read_gguf_metadata,
    read_gguf_metadata_from_runtime_source,
};
pub use gguf_tensor_data::{
    GgufHostTensorPayload, GgufOwnedWeightTensorPayload, GgufTensorDataReadError,
    GgufTensorDataReader, GgufWeightTensorElementType, GgufWeightTensorPayload,
};
pub use gguf_tensor_index::{
    GgufTensorIndex, GgufTensorIndexReadError, GgufTensorMetadata, read_gguf_tensor_index,
    read_gguf_tensor_index_from_runtime_source,
};
pub(crate) use gguf_write::{
    GgufWriteTensor, GgufWriteTensorType, GgufWriteValue, quantize_f32_to_ggml_tensor_data,
    write_gguf_file_v0,
};
pub use package_probe::{
    GgmlPackageExtensionHint, GgmlPackageFormat, GgmlPackageModelIdentityProbe, GgmlPackageProbe,
    GgmlPackageProbeError, OPENASR_RUNTIME_PACK_EXTENSION, has_openasr_runtime_pack_extension,
    probe_ggml_package_model_identity, probe_ggml_package_path,
};
pub use runtime_source::{
    GgmlRuntimeSource, GgmlRuntimeSourcePathError, validate_ggml_runtime_source_path,
};
