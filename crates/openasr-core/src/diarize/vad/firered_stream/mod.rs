//! FireRedVAD **Stream-VAD** (`FireRedTeam/FireRedVAD`, Apache-2.0,
//! `Stream-VAD/model.pth.tar`): a causal (`N2 = 0`, no lookahead) DFSMN
//! voice-activity detector. Vendored as a ~2.3 MB `f32` safetensors blob
//! baked in via `include_bytes!` (no ggml/.oasr/catalog involvement), so it
//! is always available.
//!
//! This is the **sole VAD engine** in OpenASR: because it is strictly
//! causal, the same checkpoint backs both realtime endpointing
//! ([`crate::realtime`]'s `VadMode::ExternalProbability` path, via
//! [`FireRedStreamingVad`]) and long-form speech slicing (the
//! [`crate::longform::LongFormVadProvider`] seam, via
//! [`FireRedStreamVadProvider`]) and diarization's speech-region resolution.
//! There is no other neural engine and no runtime engine-selection
//! mechanism to opt out of it.

mod frontend;
mod ggml_runtime;
mod model;
mod provider;
mod realtime_runtime;
mod streaming;
mod weights;

#[cfg(test)]
mod tests;

use std::sync::{Arc, Mutex};

use crate::models::{
    native_execution_services::{current_execution_cache_attempt_id, current_runtime_receipts},
    runtime_receipts::RuntimeOwnerGuard,
    system_memory_owner::{
        AdmittedHostObject, SystemMemoryAllocationOutcome, SystemMemoryOwner,
        SystemMemoryOwnerError,
    },
};

pub use model::FireRedStreamVadModel;
pub(crate) use provider::PolicyResolvedFireRedStreamVadProvider;
pub use provider::{FireRedStreamVadError, FireRedStreamVadProvider};
pub(crate) use realtime_runtime::{FireRedRealtimeVadRuntime, FireRedRealtimeVadSession};
pub use streaming::FireRedStreamingVad;

pub(crate) fn execution_capabilities() -> crate::device::execution_policy::ExecutionCapabilities {
    use crate::device::{
        execution_policy::{AcceleratedPlacementCapabilities, ExecutionCapabilities},
        execution_route::ExecutionProvider,
    };

    // Feature extraction remains host-side preprocessing on every backend;
    // the complete neural DFSMN graph executes on the selected device.
    ExecutionCapabilities::new(true)
        .with_provider(
            ExecutionProvider::Metal,
            AcceleratedPlacementCapabilities::FULL_DEVICE,
        )
        .with_provider(
            ExecutionProvider::Cuda,
            AcceleratedPlacementCapabilities::FULL_DEVICE,
        )
        .with_provider(
            ExecutionProvider::Vulkan,
            AcceleratedPlacementCapabilities::FULL_DEVICE,
        )
}

/// Realtime sessions keep automatic selection on CPU. Explicit accelerated
/// requests still use the unified stateful runtime and its replay contract.
pub(crate) const AUTO_GPU_POLICY: crate::ggml_runtime::AutoGpuPolicy =
    crate::ggml_runtime::AutoGpuPolicy::Never;

/// Offline slicing uses CUDA/Vulkan automatically while Metal remains an
/// explicit opt-in until its product-level latency evidence is promoted.
pub(crate) const OFFLINE_AUTO_GPU_POLICY: crate::ggml_runtime::AutoGpuPolicy =
    crate::ggml_runtime::AutoGpuPolicy::ExceptMetal;

pub(crate) type AdmittedFireRedStreamVadModel = AdmittedHostObject<FireRedStreamVadModel>;

/// NES-scoped handle to the admitted embedded Stream-VAD weights.
#[derive(Clone)]
pub struct SharedFireRedStreamVadModel {
    owner: AdmittedFireRedStreamVadModel,
}

impl std::ops::Deref for SharedFireRedStreamVadModel {
    type Target = FireRedStreamVadModel;

    fn deref(&self) -> &Self::Target {
        &self.owner
    }
}

impl SharedFireRedStreamVadModel {
    #[cfg(test)]
    pub(crate) fn committed_requested_bytes(&self) -> u64 {
        self.owner.committed_requested_bytes()
    }

    fn from_admitted(owner: AdmittedFireRedStreamVadModel) -> Self {
        Self { owner }
    }
}

pub(super) fn receipt_owner(
    component: &str,
    content: Option<&str>,
    source: Option<&str>,
) -> Option<RuntimeOwnerGuard> {
    let collector = current_runtime_receipts()?;
    if !collector.is_available() {
        return None;
    }
    let descriptor = collector.host_neutral_owner_descriptor(component, content, source)?;
    Some(collector.start_owner(descriptor, current_execution_cache_attempt_id()))
}

fn admit_embedded_model() -> Result<AdmittedFireRedStreamVadModel, SystemMemoryOwnerError> {
    let quote = FireRedStreamVadModel::system_memory_quote()
        .map_err(|reason| SystemMemoryOwnerError::capacity_failure("host_state_quote", reason))?;
    SystemMemoryOwner::try_allocate(quote, || {
        let model = FireRedStreamVadModel::embedded().map_err(|error| error.to_string())?;
        let retained = model.retained_system_memory_bytes()?;
        Ok(SystemMemoryAllocationOutcome::new(
            model, retained, retained,
        ))
    })
    .map(Arc::new)
}

/// The Stream-VAD model owned by the installed [`crate::NativeExecutionServices`]
/// root (~2.3 MB parsed weights). Returns `None` only if the vendored weights
/// blob fails to parse or admission fails. Callers should treat that as an
/// unexpected fail-closed condition, not a routine fallback.
pub fn shared_model() -> Option<SharedFireRedStreamVadModel> {
    if let Some(slot) = crate::models::native_execution_services::current_stream_vad_embedded_slot()
    {
        let mut guard = slot.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(existing) = guard.as_ref() {
            existing.record_receipt_reuse();
            return Some(SharedFireRedStreamVadModel::from_admitted(Arc::clone(
                existing,
            )));
        }
        let admitted = admit_embedded_model().ok()?;
        *guard = Some(Arc::clone(&admitted));
        return Some(SharedFireRedStreamVadModel::from_admitted(admitted));
    }
    admit_embedded_model()
        .ok()
        .map(SharedFireRedStreamVadModel::from_admitted)
}

pub(crate) type StreamVadEmbeddedSlot = Arc<Mutex<Option<AdmittedFireRedStreamVadModel>>>;
