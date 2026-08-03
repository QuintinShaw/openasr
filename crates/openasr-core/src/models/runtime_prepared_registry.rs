use thiserror::Error;

use crate::ggml_runtime::GgmlCpuGraphBackend;

use super::cohere::{
    CoherePreparedRuntime, CoherePreparedRuntimeError, build_cohere_prepared_runtime,
};
use super::ggml_asr_executor::GgmlAsrRuntimeSourcePreflight;
use super::prepared_runtime_cache::{
    HostNeutralPreparedRuntime, PreparedRuntimeCache, PreparedRuntimeHandle,
    PreparedRuntimeQuoteContext, SystemMemoryMaterialization,
};
use super::qwen::{
    Qwen3AsrPreparedRuntime, Qwen3AsrPreparedRuntimeError, build_qwen_prepared_runtime,
};
use super::system_memory_owner::{SystemMemoryAllocationQuote, SystemMemoryOwnerError};

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

impl SystemMemoryMaterialization for BuiltinPreparedRuntime {
    fn retained_system_memory_bytes(&self) -> Result<u64, String> {
        match self {
            Self::CohereTranscribe(runtime) => runtime.retained_system_memory_bytes(),
            Self::Qwen3Asr(runtime) => runtime.retained_system_memory_bytes(),
        }
    }
}

impl HostNeutralPreparedRuntime for BuiltinPreparedRuntime {
    fn system_memory_quote(
        context: PreparedRuntimeQuoteContext<'_>,
        pack_content_id: &str,
    ) -> Result<SystemMemoryAllocationQuote, SystemMemoryOwnerError> {
        match context.model_architecture {
            crate::COHERE_TRANSCRIBE_GGML_ARCHITECTURE_ID => {
                CoherePreparedRuntime::system_memory_quote(context, pack_content_id)
            }
            crate::QWEN3_ASR_GGML_ARCHITECTURE_ID => {
                Qwen3AsrPreparedRuntime::system_memory_quote(context, pack_content_id)
            }
            architecture => Err(SystemMemoryOwnerError::capacity_failure(
                "prepared_runtime_quote",
                format!("unknown builtin prepared runtime architecture '{architecture}'"),
            )),
        }
    }
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
    #[error("builtin prepared runtime system-memory admission failed: {source}")]
    SystemMemoryCapacity {
        #[source]
        source: SystemMemoryOwnerError,
    },
}

/// The resolved-input identity a builtin prepared-runtime lookup is keyed
/// and built from: which architecture, which already-preflighted runtime
/// source, and which backend this request resolved to. Grouped into one
/// value (rather than three parallel arguments) because they always travel
/// together from the executor's `request` down through the cache lookup to
/// the actual build call.
#[derive(Clone, Copy)]
pub(crate) struct PreparedRuntimeLookup<'a> {
    pub(crate) model_architecture: &'a str,
    pub(crate) preflight: &'a GgmlAsrRuntimeSourcePreflight,
    pub(crate) backend: GgmlCpuGraphBackend,
}

#[derive(Debug, Default, Clone)]
pub(crate) struct BuiltinPreparedRuntimeCache {
    runtimes_by_path: PreparedRuntimeCache<BuiltinPreparedRuntime>,
}

impl BuiltinPreparedRuntimeCache {
    pub(crate) fn ready_for_preflight(
        &self,
        preflight: &GgmlAsrRuntimeSourcePreflight,
    ) -> Option<PreparedRuntimeHandle<BuiltinPreparedRuntime>> {
        self.runtimes_by_path.ready(&preflight.runtime_source)
    }

    pub(crate) fn prepared_runtime_for_preflight<E, B, P>(
        &self,
        lookup: PreparedRuntimeLookup<'_>,
        map_build_error: B,
        map_poisoned_lock: P,
    ) -> Result<PreparedRuntimeHandle<BuiltinPreparedRuntime>, E>
    where
        B: Fn(BuiltinPreparedRuntimeRegistryError) -> E,
        P: Fn() -> E,
    {
        self.runtimes_by_path.get_or_try_insert_with(
            &lookup.preflight.runtime_source,
            PreparedRuntimeQuoteContext {
                model_architecture: lookup.model_architecture,
                metadata: &lookup.preflight.metadata,
                tensor_index: &lookup.preflight.tensor_index,
                backend: lookup.backend,
            },
            || build_builtin_prepared_runtime(lookup).map_err(&map_build_error),
            map_poisoned_lock,
            |source| {
                map_build_error(BuiltinPreparedRuntimeRegistryError::SystemMemoryCapacity {
                    source,
                })
            },
        )
    }

    #[cfg(test)]
    fn with_typed_runtime_for_preflight<T, E, B, P, M, U, R>(
        &self,
        lookup: PreparedRuntimeLookup<'_>,
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
        let prepared_runtime =
            self.prepared_runtime_for_preflight(lookup, map_build_error, map_poisoned_lock)?;
        let prepared_runtime = project(prepared_runtime.as_ref()).ok_or_else(map_wrong_variant)?;
        use_runtime(prepared_runtime)
    }

    #[cfg(test)]
    pub(crate) fn with_qwen3_asr_runtime_for_preflight<E, B, P, M, U, R>(
        &self,
        lookup: PreparedRuntimeLookup<'_>,
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
            lookup,
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

    /// Evicts exactly the cached prepared runtime for `pack_content_id`; see
    /// `PreparedRuntimeCache::evict_content_id`. Used after a pull
    /// install/replace to release the *old* content id's now-unreachable
    /// resident state without touching any other cached identity.
    pub(crate) fn evict_content_id(&self, pack_content_id: &str) {
        self.runtimes_by_path.evict_content_id(pack_content_id);
    }
}

pub(crate) fn build_builtin_prepared_runtime(
    lookup: PreparedRuntimeLookup<'_>,
) -> Result<BuiltinPreparedRuntime, BuiltinPreparedRuntimeRegistryError> {
    match lookup.model_architecture {
        crate::COHERE_TRANSCRIBE_GGML_ARCHITECTURE_ID => {
            build_cohere_prepared_runtime(lookup.preflight, lookup.backend)
                .map(BuiltinPreparedRuntime::CohereTranscribe)
                .map_err(
                    |source| BuiltinPreparedRuntimeRegistryError::CohereTranscribeBuild { source },
                )
        }
        crate::QWEN3_ASR_GGML_ARCHITECTURE_ID => {
            build_qwen_prepared_runtime(lookup.preflight, lookup.backend)
                .map(BuiltinPreparedRuntime::Qwen3Asr)
                .map_err(|source| BuiltinPreparedRuntimeRegistryError::Qwen3AsrBuild { source })
        }
        _ => Err(BuiltinPreparedRuntimeRegistryError::UnknownArchitecture {
            model_architecture: lookup.model_architecture.to_string(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::Arc;

    use tempfile::{NamedTempFile, TempPath};

    use super::*;
    use crate::models::ggml_asr_executor::GgmlAsrRuntimeSourcePreflight;
    use crate::testing::{TinyGgufFixtureSpec, write_tiny_gguf_runtime_source};

    #[test]
    fn quote_dispatch_uses_integration_identity_not_raw_gguf_alias() {
        for (integration_architecture, gguf_alias) in [
            (
                crate::COHERE_TRANSCRIBE_GGML_ARCHITECTURE_ID,
                crate::arch::hparams::COHERE_TRANSCRIBE_ARCHITECTURE_VALUE,
            ),
            (
                crate::QWEN3_ASR_GGML_ARCHITECTURE_ID,
                crate::arch::hparams::QWEN3_ARCHITECTURE_VALUE,
            ),
        ] {
            assert_ne!(integration_architecture, gguf_alias);
            let mut values = BTreeMap::new();
            values.insert(
                "general.architecture".to_string(),
                crate::ggml_runtime::GgufMetadataValue::String(gguf_alias.to_string()),
            );
            let metadata = crate::GgufMetadata::from_values_for_test(values);
            let tensor_index = crate::GgufTensorIndex::empty_for_test("alias.gguf".into());
            let result = BuiltinPreparedRuntime::system_memory_quote(
                PreparedRuntimeQuoteContext {
                    model_architecture: integration_architecture,
                    metadata: &metadata,
                    tensor_index: &tensor_index,
                    backend: GgmlCpuGraphBackend::Cpu,
                },
                "alias-regression",
            );
            if let Err(error) = result {
                assert!(
                    !error.to_string().contains("unknown builtin"),
                    "integration identity must dispatch before the family validates its raw GGUF alias: {error}"
                );
            }
        }
    }

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

        let error = build_builtin_prepared_runtime(PreparedRuntimeLookup {
            model_architecture: "unknown-arch",
            preflight: &preflight,
            backend: GgmlCpuGraphBackend::Cpu,
        })
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

        let lookup = PreparedRuntimeLookup {
            model_architecture: crate::COHERE_TRANSCRIBE_GGML_ARCHITECTURE_ID,
            preflight: &preflight,
            backend: GgmlCpuGraphBackend::Cpu,
        };
        let runtime_a = cache
            .prepared_runtime_for_preflight(
                lookup,
                |error| error,
                || BuiltinPreparedRuntimeRegistryError::UnknownArchitecture {
                    model_architecture: "poisoned".to_string(),
                },
            )
            .expect("runtime a");
        let runtime_b = cache
            .prepared_runtime_for_preflight(
                lookup,
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
        let lookup = PreparedRuntimeLookup {
            model_architecture: crate::COHERE_TRANSCRIBE_GGML_ARCHITECTURE_ID,
            preflight: &preflight,
            backend: GgmlCpuGraphBackend::Cpu,
        };
        let build = |cache: &BuiltinPreparedRuntimeCache| {
            cache
                .prepared_runtime_for_preflight(
                    lookup,
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
                PreparedRuntimeLookup {
                    model_architecture: crate::COHERE_TRANSCRIBE_GGML_ARCHITECTURE_ID,
                    preflight: &preflight,
                    backend: GgmlCpuGraphBackend::Cpu,
                },
                |error: BuiltinPreparedRuntimeRegistryError| error.to_string(),
                || "poisoned".to_string(),
                || "wrong-variant".to_string(),
                |_| Ok::<(), String>(()),
            )
            .expect_err("typed helper must fail closed on variant mismatch");

        assert_eq!(error, "wrong-variant");
    }
}
