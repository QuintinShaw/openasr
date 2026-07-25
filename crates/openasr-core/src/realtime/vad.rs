mod streaming;
pub use streaming::{
    StreamingVadEngine, StreamingVadEngineError, default_streaming_vad_config,
    resolve_streaming_vad_mode,
};

include!("vad/core.rs");
include!("vad/tests.rs");
