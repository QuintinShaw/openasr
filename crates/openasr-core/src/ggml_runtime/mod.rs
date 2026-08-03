mod arena_weight_pipeline;
mod backend;
pub(crate) mod backend_memory;
pub(crate) mod backend_memory_admission;
mod cpu_graph;
mod env_flags;
mod ffi;
mod gguf_c_parser_sandbox;
pub mod gguf_header;
mod gguf_metadata;
mod gguf_tensor_data;
mod gguf_tensor_index;
mod gguf_write;
mod job_cancel;
mod kv_element;
mod package_probe;
mod runtime_preflight;
mod runtime_source;

/// Engine-wide GGUF header safety envelope. These are format/resource limits,
/// not model-context limits; tensor payload bytes remain governed by the
/// model pack and execution-memory planner.
pub(crate) const MAX_RUNTIME_GGUF_TENSORS: u64 = 1_000_000;
pub(crate) const MAX_RUNTIME_GGUF_METADATA_ENTRIES: u64 = 100_000;
pub(crate) const MAX_RUNTIME_GGUF_STRING_BYTES: u64 = 16 * 1024 * 1024;
pub(crate) const MAX_RUNTIME_GGUF_ARRAY_ELEMENTS: u64 = 16 * 1024 * 1024;
pub(crate) const MAX_RUNTIME_GGUF_HEADER_BYTES: u64 = 256 * 1024 * 1024;

pub(crate) const fn runtime_gguf_parse_limits() -> ffi::GgufParseLimits {
    ffi::GgufParseLimits {
        max_tensors: MAX_RUNTIME_GGUF_TENSORS,
        max_kv: MAX_RUNTIME_GGUF_METADATA_ENTRIES,
        max_string_bytes: MAX_RUNTIME_GGUF_STRING_BYTES,
        max_array_elements: MAX_RUNTIME_GGUF_ARRAY_ELEMENTS,
        max_header_bytes: MAX_RUNTIME_GGUF_HEADER_BYTES,
    }
}

pub(crate) use arena_weight_pipeline::{
    ArenaAllocError, WeightSlot, alloc_static_f16, alloc_static_f32, bind_loaded,
    upload_static_f16, upload_static_f32,
};
pub use backend::{
    GgmlBackend, GgmlBackendDevice, GgmlBackendKind, GgmlCpuFeatures, GgmlDeviceMemory,
    GgmlRuntimeError, GgmlRuntimeInfo, ggml_available_devices, ggml_hip_tuning_summary,
    ggml_native_build_enabled, ggml_runtime_boot_summary, ggml_runtime_info,
};
pub(crate) use backend::{
    accelerated_device_rank, ensure_backends_loaded, preferred_accelerated_device,
};
pub use cpu_graph::{
    AutoGpuPolicy, GgmlCpuBinaryOp, GgmlCpuGraphBackend, GgmlCpuGraphConfig, GgmlCpuGraphError,
    GgmlCpuGraphRunner, GgmlCpuGraphThreadingWorkload, RequestBackendOverrideGuard,
    RequestBackendPreference, ResolvedFamilyRuntimeInput, install_request_backend_override,
    request_backend_override, resolve_request_execution_route,
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
pub(crate) use gguf_metadata::{
    bounded_gguf_parser_payload_wire_multiplier, bounded_gguf_parser_structural_bytes,
    read_gguf_metadata_from_runtime_source_with_limits,
};
pub use gguf_tensor_data::{
    GgufHostTensorPayload, GgufOwnedWeightTensorPayload, GgufTensorDataReadError,
    GgufTensorDataReader, GgufWeightTensorElementType, GgufWeightTensorPayload,
};
pub(crate) use gguf_tensor_data::{dequantize_ggml_row_to_f32, ggml_row_size_bytes};
pub(crate) use gguf_tensor_index::read_gguf_tensor_index_from_runtime_source_with_limits;
pub use gguf_tensor_index::{
    GgufTensorIndex, GgufTensorIndexReadError, GgufTensorMetadata, read_gguf_tensor_index,
    read_gguf_tensor_index_from_runtime_source,
};
pub use gguf_write::{BUILD_COMMIT_ENV, OASR_METADATA_KEY_BUILD_COMMIT};
pub(crate) use gguf_write::{
    GgufWriteTensor, GgufWriteTensorType, GgufWriteValue, quantize_f32_to_ggml_tensor_data,
    write_gguf_file_v0,
};
pub(crate) use job_cancel::{
    InheritedJobCancelGuard, arm_thread_job_cancel_flag, cancel_flag_requested_from_data,
    disarm_thread_job_cancel_flag_if_current, thread_job_cancel_flag,
};
#[cfg(test)]
pub(crate) use job_cancel::{thread_job_cancel_flag_data, thread_job_cancel_requested};
pub(crate) use kv_element::GgmlKvElementType;
#[cfg(test)]
pub(crate) use kv_element::dequantize_q8_0_rows;
pub(crate) use package_probe::probe_ggml_package_file;
pub use package_probe::{
    GgmlPackageExtensionHint, GgmlPackageFormat, GgmlPackageModelIdentityProbe, GgmlPackageProbe,
    GgmlPackageProbeError, OPENASR_RUNTIME_PACK_EXTENSION, has_openasr_runtime_pack_extension,
    probe_ggml_package_model_identity, probe_ggml_package_path,
};
pub use runtime_preflight::GgufRuntimeSourcePreflight;
pub(crate) use runtime_preflight::{
    RuntimeSourceMetadataAndTensorIndexPreflightError, RuntimeSourceTensorReaderError,
    build_runtime_tensor_reader_from_preflight, load_runtime_source_metadata_and_tensor_index,
    load_runtime_source_metadata_and_tensor_index_from_source,
};
pub use runtime_source::{
    GgmlRuntimeSource, GgmlRuntimeSourcePathError, validate_ggml_runtime_source_path,
};
pub(crate) use runtime_source::{StrongFileIdentity, resolve_content_id, unreadable_content_id};
