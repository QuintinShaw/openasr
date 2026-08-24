//! Read-only bridge to backend-native captured graph lifecycle evidence.
//!
//! This adapter is deliberately policy-free. It cannot enable capture, create
//! a graph, or select a provider. The optional ABI is resolved from the live
//! backend registry and queried only by an installed lifecycle collector.

use std::{ffi::c_void, mem, ptr::NonNull};

use thiserror::Error;

use super::ffi;

const GRAPH_LIFECYCLE_API_PROC: &[u8] = b"ggml_backend_graph_lifecycle_get_api_v1\0";
const KNOWN_FLAGS: u32 = ffi::GGML_BACKEND_GRAPH_LIFECYCLE_CAPTURE_SUPPORTED_V1
    | ffi::GGML_BACKEND_GRAPH_LIFECYCLE_CAPTURE_ENABLED_V1
    | ffi::GGML_BACKEND_GRAPH_LIFECYCLE_EXECUTABLE_PRESENT_V1
    | ffi::GGML_BACKEND_GRAPH_LIFECYCLE_GRAPH_TRACKED_V1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NativeGraphExecutableChange {
    Instantiated,
    Updated,
    Replaced,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct NativeGraphLifecycleObservation {
    pub capture_supported: bool,
    pub graph_tracked: bool,
    pub capture_enabled: bool,
    pub executable_generation: Option<u64>,
    pub last_executable_change: Option<NativeGraphExecutableChange>,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum BackendGraphLifecycleBinding {
    Unavailable,
    Incompatible,
    Available {
        observe: ffi::GgmlBackendGraphLifecycleObserveV1Fn,
    },
}

impl BackendGraphLifecycleBinding {
    /// Resolve the optional table once for a live backend handle. An absent
    /// proc is a supported state for CPU/Vulkan and builds without this
    /// extension. A present but malformed table remains distinguishable so an
    /// evidence run fails closed when it tries to observe it.
    pub(crate) unsafe fn resolve(backend: NonNull<c_void>) -> Self {
        // SAFETY: caller guarantees the backend remains live.
        let device = unsafe { ffi::ggml_backend_get_device(backend.as_ptr()) };
        if device.is_null() {
            return Self::Unavailable;
        }
        // SAFETY: the device belongs to the live backend.
        let registry = unsafe { ffi::ggml_backend_dev_backend_reg(device) };
        if registry.is_null() {
            return Self::Unavailable;
        }
        // SAFETY: static NUL-terminated name and live registry.
        let proc = unsafe {
            ffi::ggml_backend_reg_get_proc_address(
                registry,
                GRAPH_LIFECYCLE_API_PROC.as_ptr().cast(),
            )
        };
        if proc.is_null() {
            return Self::Unavailable;
        }
        // SAFETY: the versioned proc name fixes the getter signature.
        let get_api: ffi::GgmlBackendGraphLifecycleGetApiV1Fn = unsafe { mem::transmute(proc) };
        // SAFETY: getter was obtained from the live backend registry.
        let Some(api) = (unsafe { get_api().as_ref() }) else {
            return Self::Incompatible;
        };
        if api.struct_size < mem::size_of::<ffi::GgmlBackendGraphLifecycleApiV1>() as u32
            || api.abi_version != ffi::GGML_BACKEND_GRAPH_LIFECYCLE_ABI_V1
            || api.capabilities != 0
        {
            return Self::Incompatible;
        }
        match api.observe {
            Some(observe) => Self::Available { observe },
            None => Self::Incompatible,
        }
    }

    pub(crate) fn observe(
        self,
        backend: NonNull<c_void>,
        graph: NonNull<c_void>,
    ) -> Result<Option<NativeGraphLifecycleObservation>, BackendGraphLifecycleError> {
        let observe = match self {
            Self::Unavailable => return Ok(None),
            Self::Incompatible => return Err(BackendGraphLifecycleError::Incompatible),
            Self::Available { observe } => observe,
        };
        let mut raw = ffi::GgmlBackendGraphLifecycleObservationV1 {
            struct_size: mem::size_of::<ffi::GgmlBackendGraphLifecycleObservationV1>() as u32,
            ..Default::default()
        };
        // SAFETY: backend and graph are live for the enclosing compute, and
        // `raw` is a correctly sized writable v1 observation.
        let status = unsafe { observe(backend.as_ptr(), graph.as_ptr(), &mut raw) };
        if status != ffi::GGML_STATUS_SUCCESS {
            return Err(BackendGraphLifecycleError::Status { status });
        }
        parse_observation(raw).map(Some)
    }
}

fn parse_observation(
    raw: ffi::GgmlBackendGraphLifecycleObservationV1,
) -> Result<NativeGraphLifecycleObservation, BackendGraphLifecycleError> {
    if raw.struct_size < mem::size_of::<ffi::GgmlBackendGraphLifecycleObservationV1>() as u32
        || raw.abi_version != ffi::GGML_BACKEND_GRAPH_LIFECYCLE_ABI_V1
        || raw.flags & !KNOWN_FLAGS != 0
    {
        return Err(BackendGraphLifecycleError::InvalidObservation);
    }
    let capture_supported = raw.flags & ffi::GGML_BACKEND_GRAPH_LIFECYCLE_CAPTURE_SUPPORTED_V1 != 0;
    let graph_tracked = raw.flags & ffi::GGML_BACKEND_GRAPH_LIFECYCLE_GRAPH_TRACKED_V1 != 0;
    let capture_enabled = raw.flags & ffi::GGML_BACKEND_GRAPH_LIFECYCLE_CAPTURE_ENABLED_V1 != 0;
    let executable_present =
        raw.flags & ffi::GGML_BACKEND_GRAPH_LIFECYCLE_EXECUTABLE_PRESENT_V1 != 0;
    let capture_flags_without_support =
        !capture_supported && (graph_tracked || capture_enabled || executable_present);
    let graph_flags_without_tracking = !graph_tracked && (capture_enabled || executable_present);
    let executable_without_enablement = executable_present && !capture_enabled;
    if capture_flags_without_support
        || graph_flags_without_tracking
        || executable_without_enablement
        || executable_present != (raw.executable_generation != 0)
    {
        return Err(BackendGraphLifecycleError::InvalidObservation);
    }
    let last_executable_change = match raw.last_executable_change {
        ffi::GGML_BACKEND_GRAPH_EXECUTABLE_CHANGE_NONE_V1 => None,
        ffi::GGML_BACKEND_GRAPH_EXECUTABLE_CHANGE_INSTANTIATED_V1 => {
            Some(NativeGraphExecutableChange::Instantiated)
        }
        ffi::GGML_BACKEND_GRAPH_EXECUTABLE_CHANGE_UPDATED_V1 => {
            Some(NativeGraphExecutableChange::Updated)
        }
        ffi::GGML_BACKEND_GRAPH_EXECUTABLE_CHANGE_REPLACED_V1 => {
            Some(NativeGraphExecutableChange::Replaced)
        }
        _ => return Err(BackendGraphLifecycleError::InvalidObservation),
    };
    if executable_present != last_executable_change.is_some() {
        return Err(BackendGraphLifecycleError::InvalidObservation);
    }
    Ok(NativeGraphLifecycleObservation {
        capture_supported,
        graph_tracked,
        capture_enabled,
        executable_generation: executable_present.then_some(raw.executable_generation),
        last_executable_change,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub(crate) enum BackendGraphLifecycleError {
    #[error("backend graph lifecycle ABI v1 is incompatible")]
    Incompatible,
    #[error("backend graph lifecycle observation failed with ggml status {status}")]
    Status { status: i32 },
    #[error("backend graph lifecycle ABI returned an invalid observation")]
    InvalidObservation,
    #[error(
        "backend graph lifecycle capture support or enablement changed within one graph generation"
    )]
    CapturePolicyDrift,
    #[error("backend graph lifecycle stopped tracking a live graph generation")]
    GraphTrackingDisappeared,
    #[error("backend graph lifecycle did not track a capture-capable graph after compute")]
    GraphNotTrackedAfterCompute,
    #[error("backend graph lifecycle capture executable disappeared within one graph generation")]
    CaptureExecutableDisappeared,
    #[error(
        "backend graph lifecycle capture generation regressed: previous={previous}, actual={actual}"
    )]
    CaptureGenerationRegressed { previous: u64, actual: u64 },
    #[error("backend graph lifecycle capture generation changed without a native change reason")]
    CaptureGenerationChangedWithoutReason,
    #[error("backend graph lifecycle capture generation changed outside the measured compute")]
    CaptureGenerationChangedOutsideCompute,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn observation(
        flags: u32,
        generation: u64,
        change: u32,
    ) -> ffi::GgmlBackendGraphLifecycleObservationV1 {
        ffi::GgmlBackendGraphLifecycleObservationV1 {
            struct_size: mem::size_of::<ffi::GgmlBackendGraphLifecycleObservationV1>() as u32,
            abi_version: ffi::GGML_BACKEND_GRAPH_LIFECYCLE_ABI_V1,
            flags,
            last_executable_change: change,
            executable_generation: generation,
        }
    }

    #[test]
    fn graph_lifecycle_ffi_layout_matches_v1_header() {
        assert_eq!(
            mem::size_of::<ffi::GgmlBackendGraphLifecycleObservationV1>(),
            24
        );
        assert_eq!(
            mem::offset_of!(
                ffi::GgmlBackendGraphLifecycleObservationV1,
                executable_generation
            ),
            16
        );
    }

    #[test]
    fn native_capture_observation_requires_consistent_flags_generation_and_change() {
        let parsed = parse_observation(observation(
            KNOWN_FLAGS,
            7,
            ffi::GGML_BACKEND_GRAPH_EXECUTABLE_CHANGE_INSTANTIATED_V1,
        ))
        .expect("valid observation");
        assert!(parsed.capture_supported);
        assert!(parsed.graph_tracked);
        assert!(parsed.capture_enabled);
        assert_eq!(parsed.executable_generation, Some(7));
        assert_eq!(
            parsed.last_executable_change,
            Some(NativeGraphExecutableChange::Instantiated)
        );

        for raw in [
            observation(
                ffi::GGML_BACKEND_GRAPH_LIFECYCLE_CAPTURE_ENABLED_V1,
                0,
                ffi::GGML_BACKEND_GRAPH_EXECUTABLE_CHANGE_NONE_V1,
            ),
            observation(
                ffi::GGML_BACKEND_GRAPH_LIFECYCLE_CAPTURE_SUPPORTED_V1,
                9,
                ffi::GGML_BACKEND_GRAPH_EXECUTABLE_CHANGE_UPDATED_V1,
            ),
            observation(
                KNOWN_FLAGS,
                9,
                ffi::GGML_BACKEND_GRAPH_EXECUTABLE_CHANGE_NONE_V1,
            ),
            observation(
                ffi::GGML_BACKEND_GRAPH_LIFECYCLE_CAPTURE_SUPPORTED_V1
                    | ffi::GGML_BACKEND_GRAPH_LIFECYCLE_GRAPH_TRACKED_V1
                    | ffi::GGML_BACKEND_GRAPH_LIFECYCLE_EXECUTABLE_PRESENT_V1,
                9,
                ffi::GGML_BACKEND_GRAPH_EXECUTABLE_CHANGE_UPDATED_V1,
            ),
        ] {
            assert_eq!(
                parse_observation(raw),
                Err(BackendGraphLifecycleError::InvalidObservation)
            );
        }
    }
}
