use crate::ggml_runtime::GgmlCpuGraphConfig;

pub(crate) fn configure_model_graph_config(
    mut config: GgmlCpuGraphConfig,
    has_explicit_scheduler_override: bool,
) -> GgmlCpuGraphConfig {
    // GPU-class backends (Metal + the discrete-GPU lane) run the model graph on a
    // single GPU backend by default: no CPU-fallback scheduler, which is both
    // faster (no host round-trips) and the precondition for decode-graph reuse
    // (the multi-backend scheduler's `sched_alloc_graph` drops per-token inputs
    // of a reused graph). Callers can still force the scheduler via the env knob.
    if !has_explicit_scheduler_override && config.backend.is_gpu_class() {
        config.use_scheduler = false;
    }
    config
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ggml_runtime::GgmlCpuGraphBackend;

    #[test]
    fn keeps_explicit_scheduler_override_on_metal() {
        let config = configure_model_graph_config(
            GgmlCpuGraphConfig {
                backend: GgmlCpuGraphBackend::Metal,
                use_scheduler: true,
                ..GgmlCpuGraphConfig::conservative_default()
            },
            true,
        );

        assert!(config.use_scheduler);
    }

    #[test]
    fn defaults_metal_graphs_to_non_scheduler() {
        let config = configure_model_graph_config(
            GgmlCpuGraphConfig {
                backend: GgmlCpuGraphBackend::Metal,
                use_scheduler: true,
                ..GgmlCpuGraphConfig::conservative_default()
            },
            false,
        );

        assert!(!config.use_scheduler);
    }

    #[test]
    fn keeps_cpu_scheduler_setting_when_not_overridden() {
        let config = configure_model_graph_config(
            GgmlCpuGraphConfig {
                backend: GgmlCpuGraphBackend::Cpu,
                use_scheduler: true,
                ..GgmlCpuGraphConfig::conservative_default()
            },
            false,
        );

        assert!(config.use_scheduler);
    }

    #[test]
    fn defaults_gpu_graphs_to_non_scheduler() {
        // The discrete-GPU lane (HIP/CUDA/Vulkan) runs single-backend like Metal:
        // no CPU-fallback scheduler by default, which is the precondition for
        // correct decode-graph reuse (the multi-backend scheduler drops per-token
        // inputs of a reused graph, producing degenerate repeated tokens).
        let config = configure_model_graph_config(
            GgmlCpuGraphConfig {
                backend: GgmlCpuGraphBackend::Gpu,
                use_scheduler: true,
                ..GgmlCpuGraphConfig::conservative_default()
            },
            false,
        );

        assert!(!config.use_scheduler);
    }

    #[test]
    fn keeps_explicit_scheduler_override_on_gpu() {
        let config = configure_model_graph_config(
            GgmlCpuGraphConfig {
                backend: GgmlCpuGraphBackend::Gpu,
                use_scheduler: true,
                ..GgmlCpuGraphConfig::conservative_default()
            },
            true,
        );

        assert!(config.use_scheduler);
    }
}
