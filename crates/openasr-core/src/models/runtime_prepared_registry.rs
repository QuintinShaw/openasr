use std::sync::Arc;

use thiserror::Error;

use super::cohere::{
    CoherePreparedRuntime, CoherePreparedRuntimeError, build_cohere_prepared_runtime,
};
use super::ggml_asr_executor::GgmlAsrRuntimeSourcePreflight;
use super::prepared_runtime_cache::PreparedRuntimeCache;
use super::qwen::{
    Qwen3AsrPreparedRuntime, Qwen3AsrPreparedRuntimeError, build_qwen_prepared_runtime,
};

// The per-family prepared runtimes differ in size (qwen carries the LLM decode
// state), but this enum is always held behind an `Arc` in the runtime cache, so
// the variant-size delta never lands on the stack — boxing would only add an
// indirection on every weight access for no real benefit.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone)]
pub(crate) enum BuiltinPreparedRuntime {
    CohereTranscribe(CoherePreparedRuntime),
    Qwen3Asr(Qwen3AsrPreparedRuntime),
}

impl BuiltinPreparedRuntime {
    pub(crate) fn as_cohere_transcribe(&self) -> Option<&CoherePreparedRuntime> {
        match self {
            Self::CohereTranscribe(runtime) => Some(runtime),
            _ => None,
        }
    }

    pub(crate) fn as_qwen3_asr(&self) -> Option<&Qwen3AsrPreparedRuntime> {
        match self {
            Self::Qwen3Asr(runtime) => Some(runtime),
            _ => None,
        }
    }

    #[cfg(test)]
    pub(crate) fn into_qwen3_asr(self) -> Option<Qwen3AsrPreparedRuntime> {
        match self {
            Self::Qwen3Asr(runtime) => Some(runtime),
            _ => None,
        }
    }
}

#[derive(Debug, Error)]
pub(crate) enum BuiltinPreparedRuntimeRegistryError {
    #[error("unknown builtin prepared runtime architecture '{model_architecture}'")]
    UnknownArchitecture { model_architecture: String },
    #[error("builtin cohere prepared runtime build failed: {source}")]
    CohereTranscribeBuild {
        #[source]
        source: CoherePreparedRuntimeError,
    },
    #[error("builtin qwen prepared runtime build failed: {source}")]
    Qwen3AsrBuild {
        #[source]
        source: Qwen3AsrPreparedRuntimeError,
    },
}

#[derive(Debug, Default, Clone)]
pub(crate) struct BuiltinPreparedRuntimeCache {
    runtimes_by_path: PreparedRuntimeCache<BuiltinPreparedRuntime>,
}

impl BuiltinPreparedRuntimeCache {
    pub(crate) fn prepared_runtime_for_preflight<E, B, P>(
        &self,
        model_architecture: &str,
        preflight: &GgmlAsrRuntimeSourcePreflight,
        map_build_error: B,
        map_poisoned_lock: P,
    ) -> Result<Arc<BuiltinPreparedRuntime>, E>
    where
        B: Fn(BuiltinPreparedRuntimeRegistryError) -> E,
        P: Fn() -> E,
    {
        self.runtimes_by_path.get_or_try_insert_with(
            preflight.runtime_source.path(),
            || {
                build_builtin_prepared_runtime(model_architecture, preflight)
                    .map_err(map_build_error)
            },
            map_poisoned_lock,
        )
    }

    fn with_typed_runtime_for_preflight<T, E, B, P, M, U, R>(
        &self,
        model_architecture: &str,
        preflight: &GgmlAsrRuntimeSourcePreflight,
        map_build_error: B,
        map_poisoned_lock: P,
        project: fn(&BuiltinPreparedRuntime) -> Option<&T>,
        map_wrong_variant: M,
        use_runtime: U,
    ) -> Result<R, E>
    where
        B: Fn(BuiltinPreparedRuntimeRegistryError) -> E,
        P: Fn() -> E,
        M: FnOnce() -> E,
        U: FnOnce(&T) -> Result<R, E>,
    {
        let prepared_runtime = self.prepared_runtime_for_preflight(
            model_architecture,
            preflight,
            map_build_error,
            map_poisoned_lock,
        )?;
        let prepared_runtime = project(prepared_runtime.as_ref()).ok_or_else(map_wrong_variant)?;
        use_runtime(prepared_runtime)
    }

    pub(crate) fn with_cohere_transcribe_runtime_for_preflight<E, B, P, M, U, R>(
        &self,
        model_architecture: &str,
        preflight: &GgmlAsrRuntimeSourcePreflight,
        map_build_error: B,
        map_poisoned_lock: P,
        map_wrong_variant: M,
        use_runtime: U,
    ) -> Result<R, E>
    where
        B: Fn(BuiltinPreparedRuntimeRegistryError) -> E,
        P: Fn() -> E,
        M: FnOnce() -> E,
        U: FnOnce(&CoherePreparedRuntime) -> Result<R, E>,
    {
        self.with_typed_runtime_for_preflight(
            model_architecture,
            preflight,
            map_build_error,
            map_poisoned_lock,
            BuiltinPreparedRuntime::as_cohere_transcribe,
            map_wrong_variant,
            use_runtime,
        )
    }

    pub(crate) fn with_qwen3_asr_runtime_for_preflight<E, B, P, M, U, R>(
        &self,
        model_architecture: &str,
        preflight: &GgmlAsrRuntimeSourcePreflight,
        map_build_error: B,
        map_poisoned_lock: P,
        map_wrong_variant: M,
        use_runtime: U,
    ) -> Result<R, E>
    where
        B: Fn(BuiltinPreparedRuntimeRegistryError) -> E,
        P: Fn() -> E,
        M: FnOnce() -> E,
        U: FnOnce(&Qwen3AsrPreparedRuntime) -> Result<R, E>,
    {
        self.with_typed_runtime_for_preflight(
            model_architecture,
            preflight,
            map_build_error,
            map_poisoned_lock,
            BuiltinPreparedRuntime::as_qwen3_asr,
            map_wrong_variant,
            use_runtime,
        )
    }

    /// Evicts every cached prepared runtime (idle_unload); see
    /// `PreparedRuntimeCache::clear`.
    pub(crate) fn clear(&self) {
        self.runtimes_by_path.clear();
    }
}

pub(crate) fn build_builtin_prepared_runtime(
    model_architecture: &str,
    preflight: &GgmlAsrRuntimeSourcePreflight,
) -> Result<BuiltinPreparedRuntime, BuiltinPreparedRuntimeRegistryError> {
    match model_architecture {
        crate::COHERE_TRANSCRIBE_GGML_ARCHITECTURE_ID => build_cohere_prepared_runtime(preflight)
            .map(BuiltinPreparedRuntime::CohereTranscribe)
            .map_err(
                |source| BuiltinPreparedRuntimeRegistryError::CohereTranscribeBuild { source },
            ),
        crate::QWEN3_ASR_GGML_ARCHITECTURE_ID => build_qwen_prepared_runtime(preflight)
            .map(BuiltinPreparedRuntime::Qwen3Asr)
            .map_err(|source| BuiltinPreparedRuntimeRegistryError::Qwen3AsrBuild { source }),
        _ => Err(BuiltinPreparedRuntimeRegistryError::UnknownArchitecture {
            model_architecture: model_architecture.to_string(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use tempfile::{NamedTempFile, TempPath};

    use super::*;
    use crate::models::ggml_asr_executor::GgmlAsrRuntimeSourcePreflight;
    use crate::testing::{TinyGgufFixtureSpec, write_tiny_gguf_runtime_source};

    fn write_cohere_preflight() -> (TempPath, GgmlAsrRuntimeSourcePreflight) {
        let file = NamedTempFile::new().expect("temp file");
        let persisted = file.into_temp_path();
        let spec = TinyGgufFixtureSpec::cohere_oasr_v1_runtime_ready("cohere-runtime-fixture");
        write_tiny_gguf_runtime_source(&persisted, &spec).expect("write fixture");

        let runtime_source =
            crate::validate_ggml_runtime_source_path(&persisted).expect("runtime source path");
        let metadata =
            crate::read_gguf_metadata_from_runtime_source(&runtime_source).expect("metadata");
        let tensor_index = crate::read_gguf_tensor_index_from_runtime_source(&runtime_source)
            .expect("tensor index");
        (
            persisted,
            GgmlAsrRuntimeSourcePreflight {
                runtime_source,
                metadata: Arc::new(metadata),
                tensor_index: Arc::new(tensor_index),
            },
        )
    }

    #[test]
    fn fails_closed_on_unknown_architecture() {
        let (_runtime_path, preflight) = write_cohere_preflight();

        let error = build_builtin_prepared_runtime("unknown-arch", &preflight)
            .expect_err("unknown builtin arch must fail closed");
        assert!(matches!(
            error,
            BuiltinPreparedRuntimeRegistryError::UnknownArchitecture { model_architecture }
            if model_architecture == "unknown-arch"
        ));
    }

    #[test]
    fn builtin_prepared_runtime_cache_reuses_runtime_for_same_path() {
        let (_runtime_path, preflight) = write_cohere_preflight();
        let cache = BuiltinPreparedRuntimeCache::default();

        let runtime_a = cache
            .prepared_runtime_for_preflight(
                crate::COHERE_TRANSCRIBE_GGML_ARCHITECTURE_ID,
                &preflight,
                |error| error,
                || BuiltinPreparedRuntimeRegistryError::UnknownArchitecture {
                    model_architecture: "poisoned".to_string(),
                },
            )
            .expect("runtime a");
        let runtime_b = cache
            .prepared_runtime_for_preflight(
                crate::COHERE_TRANSCRIBE_GGML_ARCHITECTURE_ID,
                &preflight,
                |error| error,
                || BuiltinPreparedRuntimeRegistryError::UnknownArchitecture {
                    model_architecture: "poisoned".to_string(),
                },
            )
            .expect("runtime b");

        assert!(Arc::ptr_eq(&runtime_a, &runtime_b));
        assert!(runtime_a.as_ref().as_cohere_transcribe().is_some());
    }

    #[test]
    fn clear_evicts_the_prepared_runtime_so_the_next_call_rebuilds_it() {
        // idle_unload's actual production path: `clear()` is what
        // `Qwen3AsrGgmlExecutor::unload_idle_state` /
        // `CohereTranscribeGgmlExecutor::unload_idle_state` call. Proves the
        // real (not stub) prepared-runtime build is evicted and a later
        // request just rebuilds it -- functions normally, pays the cold cost
        // again -- exactly the documented idle_unload contract.
        let (_runtime_path, preflight) = write_cohere_preflight();
        let cache = BuiltinPreparedRuntimeCache::default();
        let build = |cache: &BuiltinPreparedRuntimeCache| {
            cache
                .prepared_runtime_for_preflight(
                    crate::COHERE_TRANSCRIBE_GGML_ARCHITECTURE_ID,
                    &preflight,
                    |error| error,
                    || BuiltinPreparedRuntimeRegistryError::UnknownArchitecture {
                        model_architecture: "poisoned".to_string(),
                    },
                )
                .expect("prepared runtime")
        };

        let runtime_a = build(&cache);
        cache.clear();
        let runtime_b = build(&cache);

        assert!(
            !Arc::ptr_eq(&runtime_a, &runtime_b),
            "clear() must evict the cached runtime so the next call rebuilds it"
        );
        assert!(runtime_b.as_ref().as_cohere_transcribe().is_some());

        // After the rebuild, the cache is warm again: a third call reuses it.
        let runtime_c = build(&cache);
        assert!(Arc::ptr_eq(&runtime_b, &runtime_c));
    }

    #[test]
    fn typed_runtime_helper_fails_closed_on_variant_mismatch() {
        let (_runtime_path, preflight) = write_cohere_preflight();
        let cache = BuiltinPreparedRuntimeCache::default();

        let error = cache
            .with_qwen3_asr_runtime_for_preflight(
                crate::COHERE_TRANSCRIBE_GGML_ARCHITECTURE_ID,
                &preflight,
                |error| error.to_string(),
                || "poisoned".to_string(),
                || "wrong-variant".to_string(),
                |_| Ok::<(), String>(()),
            )
            .expect_err("typed helper must fail closed on variant mismatch");

        assert_eq!(error, "wrong-variant");
    }
}
