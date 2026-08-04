use std::{collections::BTreeMap, sync::Arc};

use thiserror::Error;

use crate::arch::OpenAsrArchitectureRegistry;

use super::ggml_asr_executor::{GgmlAsrStreamingExecutor, GgmlAsrViewExecutor};

/// One service-owned concrete executor projected into its offline and
/// streaming Interfaces. Both trait objects point at the same allocation.
#[derive(Clone)]
pub(crate) struct BuiltinExecutorHandle {
    view: Arc<dyn GgmlAsrViewExecutor>,
    streaming: Arc<dyn GgmlAsrStreamingExecutor>,
}

impl BuiltinExecutorHandle {
    pub(crate) fn view(&self) -> Arc<dyn GgmlAsrViewExecutor> {
        Arc::clone(&self.view)
    }

    pub(crate) fn streaming(&self) -> Arc<dyn GgmlAsrStreamingExecutor> {
        Arc::clone(&self.streaming)
    }
}

/// Monomorphized only while a service root is prepared. Dynamic dispatch is
/// stored after construction and never added to tensor/op/token hot loops.
pub(crate) fn materialize_builtin_executor<E>() -> BuiltinExecutorHandle
where
    E: Default + GgmlAsrViewExecutor + GgmlAsrStreamingExecutor + 'static,
{
    let executor = Arc::new(E::default());
    BuiltinExecutorHandle {
        view: Arc::clone(&executor) as Arc<dyn GgmlAsrViewExecutor>,
        streaming: executor as Arc<dyn GgmlAsrStreamingExecutor>,
    }
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub(crate) enum BuiltinExecutorComponentRegistryError {
    #[error(
        "unknown builtin executor component '{executor_component_id}' for architecture '{model_architecture}'"
    )]
    UnknownExecutorComponent {
        model_architecture: String,
        executor_component_id: String,
    },
    #[error("duplicate builtin architecture '{model_architecture}' in executor inventory")]
    DuplicateModelArchitecture { model_architecture: String },
    #[error(
        "builtin executor component mismatch for architecture '{model_architecture}': declared='{declared}', actual='{actual}'"
    )]
    ExecutorComponentMismatch {
        model_architecture: String,
        declared: String,
        actual: &'static str,
    },
    #[error(
        "builtin executor decoder-state contract failed for architecture '{model_architecture}': {reason}"
    )]
    DecoderStateContractFailed {
        model_architecture: String,
        reason: String,
    },
    #[error(
        "builtin executor decoder-state topology mismatch for architecture '{model_architecture}': declared={declared:?}, actual={actual:?}"
    )]
    DecoderStateTopologyMismatch {
        model_architecture: String,
        declared: crate::arch::OpenAsrDecoderStateTopology,
        actual: crate::arch::OpenAsrDecoderStateTopology,
    },
}

pub(crate) fn materialize_builtin_executors_by_model_architecture(
    stateful_executors: &BuiltinStatefulExecutorScope,
) -> Result<
    BTreeMap<&'static str, Arc<dyn GgmlAsrViewExecutor>>,
    BuiltinExecutorComponentRegistryError,
> {
    let mut executors_by_model_architecture = BTreeMap::new();

    for descriptor in OpenAsrArchitectureRegistry::with_builtins().descriptors() {
        let executor = stateful_executors
            .view(descriptor.identity.model_architecture)
            .ok_or_else(
                || BuiltinExecutorComponentRegistryError::UnknownExecutorComponent {
                    model_architecture: descriptor.identity.model_architecture.to_string(),
                    executor_component_id: descriptor
                        .execution_contract
                        .executor_component_id
                        .to_string(),
                },
            )?;
        let actual_executor_id = executor.executor_id();
        if actual_executor_id != descriptor.execution_contract.executor_component_id {
            return Err(
                BuiltinExecutorComponentRegistryError::ExecutorComponentMismatch {
                    model_architecture: descriptor.identity.model_architecture.to_string(),
                    declared: descriptor
                        .execution_contract
                        .executor_component_id
                        .to_string(),
                    actual: actual_executor_id,
                },
            );
        }
        let family_descriptor = descriptor.ggml_family_adapter_descriptor();
        let actual_state_topology = executor
            .decoder_state_contract(&family_descriptor)
            .map(decoder_state_topology)
            .map_err(
                |error| BuiltinExecutorComponentRegistryError::DecoderStateContractFailed {
                    model_architecture: descriptor.identity.model_architecture.to_string(),
                    reason: error.to_string(),
                },
            )?;
        if actual_state_topology != descriptor.topology_contract.decoder_state_topology {
            return Err(
                BuiltinExecutorComponentRegistryError::DecoderStateTopologyMismatch {
                    model_architecture: descriptor.identity.model_architecture.to_string(),
                    declared: descriptor.topology_contract.decoder_state_topology,
                    actual: actual_state_topology,
                },
            );
        }
        executors_by_model_architecture.insert(descriptor.identity.model_architecture, executor);
    }

    Ok(executors_by_model_architecture)
}

fn decoder_state_topology(
    contract: super::ggml_asr_executor::GgmlAsrDecoderStateContract,
) -> crate::arch::OpenAsrDecoderStateTopology {
    use crate::arch::OpenAsrDecoderStateTopology;
    use crate::capacity::topology::StateKind;

    match contract {
        super::ggml_asr_executor::GgmlAsrDecoderStateContract::NoPersistentState => {
            OpenAsrDecoderStateTopology::None
        }
        super::ggml_asr_executor::GgmlAsrDecoderStateContract::Planned {
            streams:
                [
                    super::ggml_asr_executor::GgmlAsrDecoderStateStreamContract {
                        kind: StateKind::SelfAttentionKv,
                        ..
                    },
                ],
            ..
        } => OpenAsrDecoderStateTopology::CausalSelfAttentionKv,
        super::ggml_asr_executor::GgmlAsrDecoderStateContract::Planned { streams, .. }
            if streams.len() == 2
                && streams
                    .iter()
                    .any(|stream| stream.kind == StateKind::SelfAttentionKv)
                && streams
                    .iter()
                    .any(|stream| stream.kind == StateKind::CrossAttentionKv) =>
        {
            OpenAsrDecoderStateTopology::EncoderDecoderSelfAndCrossAttentionKv
        }
        super::ggml_asr_executor::GgmlAsrDecoderStateContract::Planned { .. } => {
            OpenAsrDecoderStateTopology::FamilyDefinedTokenScaledPersistent
        }
    }
}

/// Stateful builtin executors owned by one [`NativeExecutionServices`](crate::NativeExecutionServices)
/// root. Offline and streaming dispatches built for that root receive clones
/// of these same allocations, while independently constructed service roots
/// never share cached prepared weights.
pub(crate) struct BuiltinStatefulExecutorScope {
    by_model_architecture: BTreeMap<&'static str, BuiltinExecutorHandle>,
}

impl BuiltinStatefulExecutorScope {
    pub(crate) fn new() -> Result<Self, BuiltinExecutorComponentRegistryError> {
        let mut by_model_architecture = BTreeMap::new();
        for descriptor in OpenAsrArchitectureRegistry::with_builtins().descriptors() {
            let model_architecture = descriptor.identity.model_architecture;
            let handle = (descriptor.execution_contract.runtime_factory)();
            if by_model_architecture
                .insert(model_architecture, handle)
                .is_some()
            {
                return Err(
                    BuiltinExecutorComponentRegistryError::DuplicateModelArchitecture {
                        model_architecture: model_architecture.to_string(),
                    },
                );
            }
        }
        Ok(Self {
            by_model_architecture,
        })
    }

    pub(crate) fn view(&self, model_architecture: &str) -> Option<Arc<dyn GgmlAsrViewExecutor>> {
        self.by_model_architecture
            .get(model_architecture)
            .map(BuiltinExecutorHandle::view)
    }

    pub(crate) fn streaming(
        &self,
        model_architecture: &str,
    ) -> Option<Arc<dyn GgmlAsrStreamingExecutor>> {
        self.by_model_architecture
            .get(model_architecture)
            .map(BuiltinExecutorHandle::streaming)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn executor_data_ptr<T: ?Sized>(executor: &Arc<T>) -> *const () {
        Arc::as_ptr(executor) as *const ()
    }

    #[test]
    fn builtin_inventory_projects_one_allocation_per_scope() {
        let first_scope = BuiltinStatefulExecutorScope::new().expect("first executor scope");
        let second_scope = BuiltinStatefulExecutorScope::new().expect("second executor scope");
        let first_offline = materialize_builtin_executors_by_model_architecture(&first_scope)
            .expect("first offline executor map");
        let second_offline = materialize_builtin_executors_by_model_architecture(&second_scope)
            .expect("second offline executor map");

        for descriptor in OpenAsrArchitectureRegistry::with_builtins().descriptors() {
            let model_architecture = descriptor.identity.model_architecture;
            let first_offline_executor = first_offline
                .get(model_architecture)
                .unwrap_or_else(|| panic!("missing offline executor for {model_architecture}"));
            let second_offline_executor = second_offline
                .get(model_architecture)
                .unwrap_or_else(|| panic!("missing offline executor for {model_architecture}"));
            let first_streaming_executor = first_scope
                .streaming(model_architecture)
                .unwrap_or_else(|| panic!("missing streaming executor for {model_architecture}"));
            let second_streaming_executor = second_scope
                .streaming(model_architecture)
                .unwrap_or_else(|| panic!("missing streaming executor for {model_architecture}"));

            assert_eq!(
                executor_data_ptr(first_offline_executor),
                executor_data_ptr(&first_streaming_executor),
                "{model_architecture} offline and streaming facets must share one allocation"
            );
            assert_ne!(
                executor_data_ptr(first_offline_executor),
                executor_data_ptr(second_offline_executor),
                "{model_architecture} offline facets must be isolated between scopes"
            );
            assert_ne!(
                executor_data_ptr(&first_streaming_executor),
                executor_data_ptr(&second_streaming_executor),
                "{model_architecture} streaming facets must be isolated between scopes"
            );
        }
    }

    #[test]
    fn builtin_inventory_matches_executor_topology_contracts() {
        let scope = BuiltinStatefulExecutorScope::new().expect("executor scope");

        for descriptor in OpenAsrArchitectureRegistry::with_builtins().descriptors() {
            let model_architecture = descriptor.identity.model_architecture;
            let executor = scope
                .view(model_architecture)
                .unwrap_or_else(|| panic!("missing executor for {model_architecture}"));
            assert_eq!(
                executor.executor_id(),
                descriptor.execution_contract.executor_component_id,
                "{model_architecture} executor id must match the inventory"
            );

            let contract = executor
                .decoder_state_contract(&descriptor.ggml_family_adapter_descriptor())
                .unwrap_or_else(|error| {
                    panic!("{model_architecture} decoder-state contract failed: {error}")
                });
            assert_eq!(
                decoder_state_topology(contract),
                descriptor.topology_contract.decoder_state_topology,
                "{model_architecture} executor topology must match the inventory"
            );
        }
    }

    #[test]
    fn builtin_inventory_matches_executor_phrase_bias_contracts() {
        let scope = BuiltinStatefulExecutorScope::new().expect("executor scope");

        for descriptor in OpenAsrArchitectureRegistry::with_builtins().descriptors() {
            let model_architecture = descriptor.identity.model_architecture;
            let executor = scope
                .view(model_architecture)
                .unwrap_or_else(|| panic!("missing executor for {model_architecture}"));
            let expected = descriptor
                .execution_contract
                .phrase_bias
                .is_structurally_supported();
            assert_eq!(
                executor.supports_phrase_bias(),
                expected,
                "{model_architecture} executor phrase-bias support must match the inventory"
            );
        }
    }

    #[test]
    fn builtin_inventory_matches_concrete_executor_adapter_bindings() {
        let scope = BuiltinStatefulExecutorScope::new().expect("executor scope");

        for descriptor in OpenAsrArchitectureRegistry::with_builtins().descriptors() {
            let model_architecture = descriptor.identity.model_architecture;
            let selected_family = descriptor.ggml_family_adapter_descriptor();
            let executor = scope
                .view(model_architecture)
                .unwrap_or_else(|| panic!("missing executor for {model_architecture}"));
            let provided = executor
                .adapter_binding_strategy_for(&selected_family)
                .unwrap_or_else(|error| {
                    panic!("{model_architecture} adapter-binding lookup failed: {error}")
                });

            assert_eq!(
                provided, descriptor.execution_contract.adapter_binding,
                "{model_architecture} concrete executor adapter binding must match the inventory"
            );
        }
    }
}
