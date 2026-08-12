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
pub(super) const FULL_ENCODER_GRAPH_SIZE: usize = 65_536;

/// The stateless predictor and joiner use three tiny persistent graphs. Keep
/// their runner independent from the 65k-node streaming encoder so the head
/// does not inherit another full-encoder metadata context.
pub(super) const DEVICE_HEAD_GRAPH_SIZE: usize = 64;

/// Auto prefers the accelerator on the generic GPU lane (HIP/CUDA/Vulkan),
/// and only falls back to CPU when no accelerator is present or the request
/// targets Apple Silicon Metal specifically (see below). An explicit
/// `execution_target=cpu` or `=accelerated` always wins (the gate only ever
/// pins Auto, never overrides an explicit preference).
///
/// The earlier CPU-pinned Auto default predates the encoder-weight-placement
/// fix (#139): the streaming encoder's weights were pinned off the GPU
/// buffer, so a Metal request never actually offloaded the encoder and paid
/// GPU dispatch overhead on a per-chunk graph too small to amortize it -- a
/// net loss measured on the M1 host. With weights correctly placed so the
/// encoder truly resides on the GPU buffer, a first re-measurement put Metal
/// at parity-to-faster than CPU, but a later, cleaner platform audit found
/// Metal itself still 1.97x *slower* than CPU end-to-end (dispatch-bound: a
/// 29-frame chunk graph rebuilt/dispatched every hop is too small to amortize
/// Metal's per-dispatch overhead) while the generic GPU lane was never
/// re-measured to regress. `auto_gpu_policy = ExceptMetal` reflects that:
/// Auto now falls back to CPU on Apple Silicon Metal specifically while
/// leaving CUDA/HIP/Vulkan untouched (this remains the *final* form for the
/// streaming path -- unlike moonshine's decode-graph fix, there's no known
/// architectural fix for a chunk graph this small being dispatch-bound on
/// Metal). An explicit `--backend metal` request still gets Metal. Backend
/// choice only ever changes which backend Auto picks, never correctness:
/// output stays byte-identical between CPU and Metal.
///
pub(crate) fn xasr_zipformer_encoder_graph_config(
    backend: GgmlCpuGraphBackend,
) -> GgmlCpuGraphConfig {
    // `backend` is resolved by whoever built this request (this
    // architecture's `auto_gpu_policy = ExceptMetal`), so the base config
    // below already reflects the gate -- no separate re-check needed.
    xasr_zipformer_encoder_graph_config_with_overrides(
        configure_model_runtime_graph_config_from_env(
            GgmlCpuGraphConfig::runtime_default_for_resolved_backend(backend),
            ModelMetalRuntimeOverrides {
                default_use_scheduler_when_unset: None,
                default_n_threads_when_unset: None,
            },
        ),
        has_explicit_thread_override(),
    )
}

pub(crate) fn xasr_zipformer_device_head_graph_config(
    backend: GgmlCpuGraphBackend,
) -> GgmlCpuGraphConfig {
    let mut config = configure_model_runtime_graph_config_from_env(
        GgmlCpuGraphConfig::runtime_default_for_resolved_backend(backend),
        ModelMetalRuntimeOverrides {
            default_use_scheduler_when_unset: Some(false),
            default_n_threads_when_unset: Some(1),
        },
    );
    config.set_graph_node_capacity(DEVICE_HEAD_GRAPH_SIZE);
    if config.backend.is_gpu_class() {
        // X-ASR advertises FullDevice, and every predictor/joiner op used by
        // this graph is supported by the direct Metal/GPU backend.
        config.use_scheduler = false;
    }
    config
}

/// Pure encoder-graph policy: env-derived inputs are dependency-injected so this
/// can be unit-tested without mutating process-global env (which races across
/// the parallel test runner). Mirrors the cohere `*_with_overrides` idiom.
fn xasr_zipformer_encoder_graph_config_with_overrides(
    mut config: GgmlCpuGraphConfig,
    has_explicit_thread_override: bool,
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
    // whole encoder runs on the resolved single GPU backend with no scheduler.
    // Auto-mode backend policy is resolved before this function; a family graph
    // config must never reinterpret that resolved request contract.
    if config.backend.is_gpu_class() {
        config.use_scheduler = false;
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
        );
        // Three coexisting contexts must stay well under a CPU commit budget...
        assert!(config.context_bytes * 3 < 256 * 1024 * 1024);
        // ...while still exceeding the ~7 MB the measured 11.1k-node graph uses.
        assert!(config.context_bytes > 7 * 1024 * 1024);
    }

    #[test]
    fn gpu_encoder_keeps_the_resolved_single_gpu_backend() {
        let config = xasr_zipformer_encoder_graph_config_with_overrides(
            base_with(GgmlCpuGraphBackend::Metal, None),
            false,
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
        );

        assert_eq!(config.n_threads, Some(2));
    }
}
