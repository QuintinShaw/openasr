//! firered-aed ggml graph backend/threading policy.
//!
//! Stage 2/3 landed CPU-only by design (correctness-first, GPU staged as an
//! explicit follow-up once decoder/executor parity was established -- see the
//! prior module docs on [`super::encoder_graph`] and [`super::decoder_graph`]
//! for the CPU-only-era history). That parity is now verified end to end
//! (CPU vs Metal transcripts match byte-for-byte on real packs), so this
//! mirrors the cohere/moonshine template -- dynamic backend resolution via
//! [`configure_model_runtime_graph_config_from_env`] (Metal auto-selected
//! through the generic runtime-default resolver), with an explicit per-stage
//! opt-out that falls back to CPU.
//!
//! Note this is narrower than it may read: firered-aed's own executor never
//! batches *multiple* longform slices into one graph call the way cohere's
//! `batched_decode` can (each call here still encodes/decodes exactly one
//! window -- see `executor.rs` module docs), so there is no
//! `prefer_cpu_backend` request-level override to thread through here. That
//! is NOT the same claim as "firered-aed has no longform support" (issue
//! #158's actual bug, and easy to misread this comment as): the *outer*
//! per-file longform slicer in `native_transcribe` is architecture-agnostic
//! and already calls this executor once per slice for every builtin family,
//! firered-aed included, with its window length capped to this
//! architecture's declared `GlobalQuadratic` safety ceiling (issue #68's
//! `encoder_attention_span`) and, defensively, to the encoder's baked
//! rel-pos-table capacity (`FireRedAedExecutionMetadata::encoder_max_frames`,
//! enforced in `executor.rs`).

use crate::ggml_runtime::{GgmlCpuGraphBackend, GgmlCpuGraphConfig, GgmlCpuGraphThreadingWorkload};
#[cfg(test)]
use crate::models::graph_runtime_config::configure_model_runtime_graph_config;
use crate::models::graph_runtime_config::{
    ModelMetalRuntimeOverrides, configure_model_runtime_graph_config_from_env,
    gpu_stage_enabled_for_backend, has_explicit_thread_override,
};

const FIRERED_ENCODER_GRAPH_SIZE: usize = 32_768;
const FIRERED_DECODER_GRAPH_SIZE: usize = 8192;

const OPENASR_FIRERED_ENABLE_ENCODER_METAL: &str = "OPENASR_FIRERED_ENABLE_ENCODER_METAL";
const OPENASR_FIRERED_ENABLE_DECODER_METAL: &str = "OPENASR_FIRERED_ENABLE_DECODER_METAL";
const OPENASR_FIRERED_ENABLE_ENCODER_GPU: &str = "OPENASR_FIRERED_ENABLE_ENCODER_GPU";
const OPENASR_FIRERED_ENABLE_DECODER_GPU: &str = "OPENASR_FIRERED_ENABLE_DECODER_GPU";

/// Shared base for both stages: everything except the Metal scheduler
/// default, which the encoder and decoder set independently (see
/// [`firered_encoder_graph_config`] / [`firered_decoder_graph_config`]) --
/// the same encoder/decoder split moonshine's `graph_config` uses for the
/// same reason (decode-graph reuse).
fn firered_runtime_graph_config_with_scheduler_default(
    backend: GgmlCpuGraphBackend,
    default_use_scheduler_when_unset: Option<bool>,
) -> GgmlCpuGraphConfig {
    configure_model_runtime_graph_config_from_env(
        GgmlCpuGraphConfig::runtime_default_for_resolved_backend(backend),
        ModelMetalRuntimeOverrides {
            default_use_scheduler_when_unset,
            default_n_threads_when_unset: Some(1),
        },
    )
}

pub(crate) fn firered_encoder_graph_config(backend: GgmlCpuGraphBackend) -> GgmlCpuGraphConfig {
    // `no_alloc` metadata context sized from the actual node count (see
    // `GgmlCpuGraphConfig::metadata_context_bytes`); previously a flat
    // hardcoded 512 MiB per cached encoder runtime (see the thread-local
    // cache in `executor.rs`).
    //
    // The encoder keeps the scheduler on for Metal: the conformer forward
    // graph was built and parity-verified under multi-backend scheduling and
    // has not been re-verified with the scheduler off. Only the decoder's
    // `use_scheduler` default changed (see `firered_decoder_graph_config`).
    let mut config = firered_runtime_graph_config_with_scheduler_default(backend, Some(true));
    config.graph_size = config.graph_size.max(FIRERED_ENCODER_GRAPH_SIZE);
    config.context_bytes = config
        .context_bytes
        .max(GgmlCpuGraphConfig::metadata_context_bytes(
            config.graph_size,
        ));
    if config.backend.is_gpu_class() && !firered_encoder_gpu_enabled(config.backend) {
        config.backend = GgmlCpuGraphBackend::Cpu;
        config.use_scheduler = false;
    }
    if !has_explicit_thread_override() {
        config.n_threads = GgmlCpuGraphConfig::resolve_runtime_thread_count_for(
            config.backend,
            GgmlCpuGraphThreadingWorkload::EncoderPrelude,
        );
    }
    config
}

/// Decode-graph reuse (`nn::decoder::reusable_decode_graph_supported`) only
/// activates when the backend is GPU-class *and* the scheduler is off (a
/// multi-backend scheduler's `sched_alloc_graph` drops the per-token inputs
/// a reused, in-place-KV graph depends on). The decoder previously inherited
/// the encoder's `default_use_scheduler_when_unset: Some(true)`, which would
/// keep the reusable incremental-step graph in `decoder_graph` permanently
/// disabled on Metal and force a full graph rebuild every decode token
/// (measured at ~21% of the per-token decode step on l-v2 q4_k). Leaving
/// this `None` keeps the base default (scheduler-off on GPU-class backends,
/// see `configure_model_graph_config`), exactly mirroring moonshine's
/// decoder-tier fix. This is a pure backend/scheduling choice: output must
/// stay byte-identical (pinned by the reused-vs-fresh logits test in
/// `decoder_graph` and the firered golden tests), since it does not change
/// which arithmetic runs, only whether the graph is rebuilt per token.
/// `OPENASR_GGML_USE_SCHEDULER=1` remains the explicit escape hatch (it also
/// disables reuse, restoring the rebuild-per-token path).
pub(crate) fn firered_decoder_graph_config(backend: GgmlCpuGraphBackend) -> GgmlCpuGraphConfig {
    // See the matching comment in `firered_encoder_graph_config`: this is a
    // `no_alloc` metadata pool sized from the actual node count, not the real
    // tensor bytes (those live in the arena's own backend buffer).
    let mut config = firered_runtime_graph_config_with_scheduler_default(backend, None);
    config.graph_size = config.graph_size.max(FIRERED_DECODER_GRAPH_SIZE);
    config.context_bytes = config
        .context_bytes
        .max(GgmlCpuGraphConfig::metadata_context_bytes(
            config.graph_size,
        ));
    if config.backend.is_gpu_class() && !firered_decoder_gpu_enabled(config.backend) {
        config.backend = GgmlCpuGraphBackend::Cpu;
        config.use_scheduler = false;
    }
    if !has_explicit_thread_override() {
        config.n_threads = GgmlCpuGraphConfig::resolve_runtime_thread_count_for(
            config.backend,
            GgmlCpuGraphThreadingWorkload::Decoder,
        );
    }
    config
}

fn firered_encoder_gpu_enabled(backend: GgmlCpuGraphBackend) -> bool {
    gpu_stage_enabled_for_backend(
        backend,
        OPENASR_FIRERED_ENABLE_ENCODER_GPU,
        true,
        Some(OPENASR_FIRERED_ENABLE_ENCODER_METAL),
        true,
    )
}

fn firered_decoder_gpu_enabled(backend: GgmlCpuGraphBackend) -> bool {
    gpu_stage_enabled_for_backend(
        backend,
        OPENASR_FIRERED_ENABLE_DECODER_GPU,
        true,
        Some(OPENASR_FIRERED_ENABLE_DECODER_METAL),
        true,
    )
}

/// Test-only mirror of [`firered_runtime_graph_config_with_scheduler_default`]
/// with the env/TLS reads replaced by explicit flags, so the scheduler-default
/// pins below stay deterministic regardless of the test environment (same
/// pattern as `whisper::graph_config` / `qwen::graph_config`).
#[cfg(test)]
fn firered_runtime_graph_config_with_explicit_overrides(
    base: GgmlCpuGraphConfig,
    has_explicit_scheduler_override: bool,
    has_explicit_thread_override: bool,
    default_use_scheduler_when_unset: Option<bool>,
) -> GgmlCpuGraphConfig {
    configure_model_runtime_graph_config(
        base,
        has_explicit_scheduler_override,
        has_explicit_thread_override,
        ModelMetalRuntimeOverrides {
            default_use_scheduler_when_unset,
            default_n_threads_when_unset: Some(1),
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metal_base(use_scheduler: bool) -> GgmlCpuGraphConfig {
        GgmlCpuGraphConfig {
            backend: GgmlCpuGraphBackend::Metal,
            use_scheduler,
            ..GgmlCpuGraphConfig::conservative_default()
        }
    }

    /// Pin the decoder tier's Metal scheduler default to OFF: decode-graph
    /// reuse (`nn::decoder::reusable_decode_graph_supported`) requires
    /// `gpu_class && !scheduler`, so a scheduler-on default here would turn
    /// the reusable incremental decode graph back into dead code (the exact
    /// regression moonshine had before commit 879677ac).
    #[test]
    fn decoder_metal_scheduler_default_stays_off_for_decode_graph_reuse() {
        let config = firered_runtime_graph_config_with_explicit_overrides(
            metal_base(false),
            false,
            false,
            None,
        );
        assert!(
            !config.use_scheduler,
            "firered decoder tier must default the Metal scheduler off so the reusable \
             incremental decode graph stays reachable"
        );
    }

    /// The encoder tier keeps the scheduler-on Metal default it was
    /// parity-verified under.
    #[test]
    fn encoder_metal_scheduler_default_stays_on() {
        let config = firered_runtime_graph_config_with_explicit_overrides(
            metal_base(false),
            false,
            false,
            Some(true),
        );
        assert!(config.use_scheduler);
    }

    /// An explicit `OPENASR_GGML_USE_SCHEDULER` override must keep winning on
    /// the decoder tier (it is the escape hatch that restores the
    /// rebuild-per-token decode path).
    #[test]
    fn decoder_metal_explicit_scheduler_override_still_wins() {
        let config = firered_runtime_graph_config_with_explicit_overrides(
            metal_base(true),
            true,
            false,
            None,
        );
        assert!(config.use_scheduler);
    }

    #[test]
    fn encoder_gpu_defaults_to_unified_gpu_lane() {
        assert!(firered_encoder_gpu_enabled(GgmlCpuGraphBackend::Gpu));
    }

    #[test]
    fn decoder_gpu_defaults_to_unified_gpu_lane() {
        assert!(firered_decoder_gpu_enabled(GgmlCpuGraphBackend::Gpu));
    }

    #[test]
    fn encoder_and_decoder_gpu_keep_cpu_and_metal_defaults() {
        assert!(firered_encoder_gpu_enabled(GgmlCpuGraphBackend::Cpu));
        assert!(firered_encoder_gpu_enabled(GgmlCpuGraphBackend::Metal));
        assert!(firered_decoder_gpu_enabled(GgmlCpuGraphBackend::Cpu));
        assert!(firered_decoder_gpu_enabled(GgmlCpuGraphBackend::Metal));
    }

    #[test]
    fn encoder_graph_size_floor_is_preserved() {
        assert!(
            firered_encoder_graph_config(GgmlCpuGraphBackend::Cpu).graph_size
                >= FIRERED_ENCODER_GRAPH_SIZE
        );
    }

    #[test]
    fn decoder_graph_size_floor_is_preserved() {
        assert!(
            firered_decoder_graph_config(GgmlCpuGraphBackend::Cpu).graph_size
                >= FIRERED_DECODER_GRAPH_SIZE
        );
    }
}
