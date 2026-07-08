pub(crate) mod config;
pub mod package_import;
pub mod prompt;
pub mod runtime;
pub(crate) mod tensor_names;
pub mod tokenizer;

pub use config::{Hymt2ConfigError, Hymt2ExecutionMetadata};
pub use package_import::{
    HYMT2_PINNED_SOURCE_GGUF_SHA256, Hymt2ImportError, Hymt2ImportRequest, Hymt2ImportResult,
    import_hymt2_gguf_to_runtime_pack,
};
pub use runtime::{
    Hymt2DecodeResult, Hymt2DecodeTimings, Hymt2PrefixCacheConfig, Hymt2PrefixReuseReport,
    Hymt2Runtime, Hymt2RuntimeError, Hymt2TranslationSessionCache,
};
pub use tokenizer::Hymt2Tokenizer;

// Pull-time contract validation for translation runtime packs (Hy-MT2,
// `general.architecture = "hunyuan-dense"`) is dispatched through
// `crate::models::aux_pack_registry`, alongside the other auxiliary (non-ASR)
// families (diarization, punctuation) -- one table instead of a per-family
// function called from an ad hoc chain in `api::backend::native`. The contract
// itself is still the cheap [`runtime::Hymt2Runtime`] probe (metadata +
// tensor-index validation, no weight materialization).
