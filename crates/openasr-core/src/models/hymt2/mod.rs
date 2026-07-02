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

/// Pull-time contract for translation runtime packs.
///
/// Returns `Some` only when the pack declares the Hy-MT2 GGUF architecture
/// (`general.architecture = "hunyuan-dense"`); ASR packs fall through to
/// family-adapter selection. The contract is the cheap [`runtime::Hymt2Runtime`]
/// probe (metadata + tensor-index validation, no weight materialization), so
/// `openasr pull` stays fail-closed for translation packs without paying a
/// full model load.
pub(crate) fn validate_translation_runtime_pack_contract(
    path: &std::path::Path,
    metadata: &crate::GgufMetadata,
) -> Option<Result<(), String>> {
    let architecture = metadata.get_string(crate::arch::GENERAL_ARCHITECTURE_KEY)?;
    if architecture.trim() != config::HUNYUAN_DENSE_ARCHITECTURE_VALUE {
        return None;
    }
    Some(
        runtime::Hymt2Runtime::probe_path(path)
            .map(|_| ())
            .map_err(|error| error.to_string()),
    )
}
