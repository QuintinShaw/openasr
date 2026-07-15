use crate::ggml_runtime::{GgmlCpuGraphBackend, GgmlCpuGraphConfig, GgmlCpuGraphThreadingWorkload};
use crate::models::graph_runtime_config::{
    ModelMetalRuntimeOverrides, configure_model_runtime_graph_config_from_env,
    has_explicit_thread_override,
};

/// Right-sized from the measured full-encoder forward graph (~11.1k nodes /
/// ~12.4k tensors for the streaming chunk window). The graph topology is
/// architecture-bound (layers x ops-per-layer), not sequence-length-bound —
/// longer audio grows tensor dimensions, not the op count — so the node count
/// stays ~constant across inputs. 65,536 keeps >5x headroom on both node and
/// tensor counts. The previous 2,000,000 over-reserved the cgraph object alone
/// by ~79 MB and, paired with a hand-tuned 2 GiB context (see
/// [`GgmlCpuGraphConfig::metadata_context_bytes`]), OOM'd CPU transcription.
const FULL_ENCODER_GRAPH_SIZE: usize = 65_536;

/// The encoder runs on the GPU only when the REQUEST explicitly asks for
/// accelerated execution. Auto stays CPU: on the measured M1 host every Metal
/// configuration loses to CPU for this model's chunked workload (the per-chunk
/// graph is too small to amortize GPU dispatch). Users on stronger GPUs can
/// force it with execution_target=accelerated, and what runs is what was
/// requested — no silent downgrade.
///
/// Delegates to the shared `resolve_family_runtime_backend` gate (declared via
/// this architecture's `auto_gpu_enabled = false`) rather than hand-rolling
/// the override check, so any provenance label resolving through the same
/// gate can never drift from what this function actually decided.
fn encoder_gpu_enabled() -> bool {
    GgmlCpuGraphConfig::resolve_family_runtime_backend(false).is_gpu_class()
}

pub(crate) fn xasr_zipformer_encoder_graph_config() -> GgmlCpuGraphConfig {
    xasr_zipformer_encoder_graph_config_with_overrides(
        configure_model_runtime_graph_config_from_env(
            GgmlCpuGraphConfig::default(),
            ModelMetalRuntimeOverrides {
                default_use_scheduler_when_unset: None,
                default_n_threads_when_unset: None,
            },
        ),
        has_explicit_thread_override(),
        encoder_gpu_enabled(),
    )
}

/// Pure encoder-graph policy: env-derived inputs are dependency-injected so this
/// can be unit-tested without mutating process-global env (which races across
/// the parallel test runner). Mirrors the cohere `*_with_overrides` idiom.
fn xasr_zipformer_encoder_graph_config_with_overrides(
    mut config: GgmlCpuGraphConfig,
    has_explicit_thread_override: bool,
    encoder_gpu_enabled: bool,
) -> GgmlCpuGraphConfig {
    config.graph_size = config.graph_size.max(FULL_ENCODER_GRAPH_SIZE);
    config.context_bytes = config
        .context_bytes
        .max(GgmlCpuGraphConfig::metadata_context_bytes(
            config.graph_size,
        ));
    // X-ASR uses depthwise conv (CONV_2D_DW) in the encoder-embed and conv_module
    // paths. The Metal backend has no fused CONV_2D_DW kernel, and a scheduler
    // CPU-fallback can't move the op because the prepared graph's tensors are
    // pre-allocated to the GPU buffer. Instead the graph builder emits the
    // im2col-based depthwise conv (Metal-native) on GPU-class backends, so the
    // whole encoder runs on a single GPU backend with no scheduler. Default
    // fail-closed to CPU; only an explicit accelerated request keeps the GPU
    // backend (single-backend).
    if config.backend.is_gpu_class() {
        if encoder_gpu_enabled {
            config.use_scheduler = false;
        } else {
            config.backend = GgmlCpuGraphBackend::Cpu;
            config.use_scheduler = false;
        }
    }
    // The streaming encoder runs a small (29-frame) chunk graph per hop, so it is
    // latency-bound and oversubscription-sensitive like an autoregressive
    // decoder, not a wide batched encoder. A single-host thread sweep on this
    // 8-core machine put the `Decoder` profile (4 threads) well ahead of the
    // `EncoderPrelude` profile (7 threads) — do not widen without a fresh sweep.
    if !has_explicit_thread_override && config.backend == GgmlCpuGraphBackend::Cpu {
        config.n_threads = GgmlCpuGraphConfig::resolve_runtime_thread_count_for(
            GgmlCpuGraphBackend::Cpu,
            GgmlCpuGraphThreadingWorkload::Decoder,
        );
    }
    config
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_with(backend: GgmlCpuGraphBackend, n_threads: Option<usize>) -> GgmlCpuGraphConfig {
        GgmlCpuGraphConfig {
            backend,
            n_threads,
            use_scheduler: backend.is_gpu_class(),
            ..GgmlCpuGraphConfig::conservative_default()
        }
    }

    #[test]
    fn config_reserves_full_encoder_graph_capacity() {
        let config = xasr_zipformer_encoder_graph_config_with_overrides(
            base_with(GgmlCpuGraphBackend::Cpu, None),
            false,
            false,
        );
        assert!(config.graph_size >= FULL_ENCODER_GRAPH_SIZE);
        assert!(
            config.context_bytes
                >= GgmlCpuGraphConfig::metadata_context_bytes(FULL_ENCODER_GRAPH_SIZE)
        );
    }

    #[test]
    fn full_encoder_contexts_stay_within_cpu_commit_budget() {
        // Regression guard for the CPU-transcription OOM: the embed runner, the
        // full-encoder runner, and the persistent graph session each allocate one
        // no_alloc metadata context at the same time. `ggml_init` always mallocs
        // the full `mem_size` even with `no_alloc=true`, so the pre-fix 2 GiB x3 =
        // 6 GiB tripped `_aligned_malloc` -> NULL -> GGML_ASSERT. Sizing each
        // context from `graph_size` keeps all three comfortably resident.
        let config = xasr_zipformer_encoder_graph_config_with_overrides(
            base_with(GgmlCpuGraphBackend::Cpu, None),
            false,
            false,
        );
        // Three coexisting contexts must stay well under a CPU commit budget...
        assert!(config.context_bytes * 3 < 256 * 1024 * 1024);
        // ...while still exceeding the ~7 MB the measured 11.1k-node graph uses.
        assert!(config.context_bytes > 7 * 1024 * 1024);
    }

    #[test]
    fn gpu_encoder_falls_back_to_cpu_when_gate_disabled() {
        let config = xasr_zipformer_encoder_graph_config_with_overrides(
            base_with(GgmlCpuGraphBackend::Metal, None),
            false,
            false,
        );

        assert_eq!(config.backend, GgmlCpuGraphBackend::Cpu);
        assert!(!config.use_scheduler);
    }

    #[test]
    fn gpu_encoder_keeps_single_gpu_backend_when_gate_enabled() {
        let config = xasr_zipformer_encoder_graph_config_with_overrides(
            base_with(GgmlCpuGraphBackend::Metal, None),
            false,
            true,
        );

        // GPU runs single-backend (im2col depthwise conv is Metal-native), so no
        // multi-backend scheduler / CPU fallback.
        assert_eq!(config.backend, GgmlCpuGraphBackend::Metal);
        assert!(!config.use_scheduler);
    }

    #[test]
    fn config_uses_chunk_friendly_cpu_threads_when_unset() {
        let config = xasr_zipformer_encoder_graph_config_with_overrides(
            base_with(GgmlCpuGraphBackend::Cpu, None),
            false,
            false,
        );

        assert_eq!(
            config.n_threads,
            GgmlCpuGraphConfig::resolve_runtime_thread_count_for(
                GgmlCpuGraphBackend::Cpu,
                GgmlCpuGraphThreadingWorkload::Decoder,
            )
        );
    }

    #[test]
    fn config_keeps_explicit_cpu_threads() {
        let config = xasr_zipformer_encoder_graph_config_with_overrides(
            base_with(GgmlCpuGraphBackend::Cpu, Some(2)),
            true,
            false,
        );

        assert_eq!(config.n_threads, Some(2));
    }
}
