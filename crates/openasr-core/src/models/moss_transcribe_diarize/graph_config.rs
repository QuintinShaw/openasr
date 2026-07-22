//! moss-transcribe-diarize ggml graph backend/threading policy. Correctness-
//! first: no per-stage GPU opt-out gating yet (perf tuning is out of scope
//! this stage, see `mod.rs`'s stage-status note) -- just the shared
//! env-driven backend resolution every other family's runtime graph config
//! uses (`AutoGpuPolicy::AllBackends`, mirrors this family's arch-descriptor
//! entry).

use crate::ggml_runtime::GgmlCpuGraphConfig;
use crate::models::graph_runtime_config::{
    ModelMetalRuntimeOverrides, configure_model_runtime_graph_config_from_env,
};

const MOSS_TD_ENCODER_GRAPH_SIZE: usize = 16_384;

pub(crate) fn moss_td_runtime_graph_config() -> GgmlCpuGraphConfig {
    configure_model_runtime_graph_config_from_env(
        GgmlCpuGraphConfig::default(),
        ModelMetalRuntimeOverrides {
            default_use_scheduler_when_unset: Some(true),
            default_n_threads_when_unset: Some(1),
        },
    )
}

pub(crate) fn moss_td_encoder_graph_config() -> GgmlCpuGraphConfig {
    let mut config = moss_td_runtime_graph_config();
    config.graph_size = config.graph_size.max(MOSS_TD_ENCODER_GRAPH_SIZE);
    config.context_bytes = config
        .context_bytes
        .max(GgmlCpuGraphConfig::metadata_context_bytes(
            config.graph_size,
        ));
    config
}
