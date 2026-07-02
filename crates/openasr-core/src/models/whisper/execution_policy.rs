use crate::ggml_runtime::{GgmlCpuGraphBackend, GgmlCpuGraphConfig, env_toggle_with_raw};

const OPENASR_WHISPER_GGML_DISABLE_ENCODER_FLASH_ATTN: &str =
    "OPENASR_WHISPER_GGML_DISABLE_ENCODER_FLASH_ATTN";
const OPENASR_WHISPER_GGML_DISABLE_DECODER_CROSS_FLASH_ATTN: &str =
    "OPENASR_WHISPER_GGML_DISABLE_DECODER_CROSS_FLASH_ATTN";
const OPENASR_WHISPER_GGML_DISABLE_DECODER_SELF_FLASH_ATTN: &str =
    "OPENASR_WHISPER_GGML_DISABLE_DECODER_SELF_FLASH_ATTN";
const OPENASR_WHISPER_GGML_DISABLE_PARALLEL_ENCODER_AND_DECODER_STATIC: &str =
    "OPENASR_WHISPER_GGML_DISABLE_PARALLEL_ENCODER_AND_DECODER_STATIC";
const OPENASR_WHISPER_GGML_ENABLE_PARALLEL_ENCODER_AND_DECODER_STATIC: &str =
    "OPENASR_WHISPER_GGML_ENABLE_PARALLEL_ENCODER_AND_DECODER_STATIC";
const OPENASR_WHISPER_GGML_DISABLE_PERSISTENT_CROSS_CACHE_F16_UPLOAD: &str =
    "OPENASR_WHISPER_GGML_DISABLE_PERSISTENT_CROSS_CACHE_F16_UPLOAD";
const OPENASR_WHISPER_GGML_ENABLE_PERSISTENT_CROSS_CACHE_F16_UPLOAD: &str =
    "OPENASR_WHISPER_GGML_ENABLE_PERSISTENT_CROSS_CACHE_F16_UPLOAD";
pub(crate) fn whisper_encoder_flash_attention_enabled() -> bool {
    std::env::var_os(OPENASR_WHISPER_GGML_DISABLE_ENCODER_FLASH_ATTN).is_none()
}

pub(crate) fn whisper_decoder_cross_flash_attention_enabled() -> bool {
    std::env::var_os(OPENASR_WHISPER_GGML_DISABLE_DECODER_CROSS_FLASH_ATTN).is_none()
}

pub(crate) fn whisper_decoder_self_flash_attention_enabled() -> bool {
    std::env::var_os(OPENASR_WHISPER_GGML_DISABLE_DECODER_SELF_FLASH_ATTN).is_none()
}

pub(crate) fn whisper_parallel_encoder_and_decoder_static_enabled(
    backend: GgmlCpuGraphBackend,
    allow_persistent_session_reuse: bool,
) -> bool {
    let default_enabled = !(allow_persistent_session_reuse && backend.is_gpu_class());
    env_toggle_with_raw(
        std::env::var(OPENASR_WHISPER_GGML_DISABLE_PARALLEL_ENCODER_AND_DECODER_STATIC)
            .ok()
            .as_deref(),
        std::env::var(OPENASR_WHISPER_GGML_ENABLE_PARALLEL_ENCODER_AND_DECODER_STATIC)
            .ok()
            .as_deref(),
        default_enabled,
    )
}

pub(crate) fn whisper_decoder_persistent_cross_cache_f16_upload_enabled(
    backend: GgmlCpuGraphBackend,
    requires_f32_rhs: bool,
) -> bool {
    let default_enabled = matches!(backend, GgmlCpuGraphBackend::Cpu)
        || (backend.is_gpu_class() && !requires_f32_rhs);
    whisper_decoder_persistent_cross_cache_f16_upload_enabled_with_env(
        std::env::var(OPENASR_WHISPER_GGML_DISABLE_PERSISTENT_CROSS_CACHE_F16_UPLOAD)
            .ok()
            .as_deref(),
        std::env::var(OPENASR_WHISPER_GGML_ENABLE_PERSISTENT_CROSS_CACHE_F16_UPLOAD)
            .ok()
            .as_deref(),
        default_enabled,
    )
}

pub(crate) fn whisper_decoder_persistent_cross_cache_f16_upload_enabled_with_env(
    disable_raw: Option<&str>,
    enable_raw: Option<&str>,
    default_enabled: bool,
) -> bool {
    env_toggle_with_raw(disable_raw, enable_raw, default_enabled)
}

pub(crate) fn whisper_decoder_persistent_cross_cache_default_f32_rhs_on_cpu_enabled() -> bool {
    whisper_decoder_persistent_cross_cache_default_f32_rhs_on_cpu_enabled_with_env(
        std::env::var(GgmlCpuGraphConfig::CPU_ACCELERATOR_ENV)
            .ok()
            .as_deref(),
    )
}

pub(crate) fn whisper_decoder_persistent_cross_cache_default_f32_rhs_on_cpu_enabled_with_env(
    raw: Option<&str>,
) -> bool {
    GgmlCpuGraphConfig::cpu_accelerator_enabled_with_env(raw, GgmlCpuGraphBackend::Cpu)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn longform_metal_disables_parallel_encoder_and_decoder_static_by_default() {
        assert!(!whisper_parallel_encoder_and_decoder_static_enabled(
            GgmlCpuGraphBackend::Metal,
            true,
        ));
    }

    #[test]
    fn short_metal_keeps_parallel_encoder_and_decoder_static_enabled() {
        assert!(whisper_parallel_encoder_and_decoder_static_enabled(
            GgmlCpuGraphBackend::Metal,
            false,
        ));
    }

    #[test]
    fn longform_cpu_keeps_parallel_encoder_and_decoder_static_enabled() {
        assert!(whisper_parallel_encoder_and_decoder_static_enabled(
            GgmlCpuGraphBackend::Cpu,
            true,
        ));
    }

    #[test]
    fn longform_gpu_disables_parallel_encoder_and_decoder_static_by_default() {
        assert!(!whisper_parallel_encoder_and_decoder_static_enabled(
            GgmlCpuGraphBackend::Gpu,
            true,
        ));
    }

    #[test]
    fn gpu_persistent_cross_cache_f16_upload_matches_metal_policy() {
        assert!(whisper_decoder_persistent_cross_cache_f16_upload_enabled(
            GgmlCpuGraphBackend::Gpu,
            false,
        ));
        assert!(!whisper_decoder_persistent_cross_cache_f16_upload_enabled(
            GgmlCpuGraphBackend::Gpu,
            true,
        ));
    }
}
