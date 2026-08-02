use thiserror::Error;

use super::{NativeAsrHardwareTarget, NativeAsrRuntimeReadiness};
use crate::device::execution_route::ExecutionRouteError;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum NativeAsrError {
    #[error("Native ASR model pack is unsupported: {reason}.")]
    UnsupportedModelPack { reason: String },
    #[error("Native ASR model asset is missing locally: {path}.")]
    MissingLocalModelAsset { path: std::path::PathBuf },
    #[error("Native ASR hardware target is unsupported: {target}.")]
    UnsupportedHardwareTarget { target: NativeAsrHardwareTarget },
    #[error("Native ASR provider is unavailable: {provider}.")]
    ProviderUnavailable { provider: String },
    #[error("Backend '{backend}' does not support true streaming ASR.")]
    BackendDoesNotSupportTrueStreaming { backend: String },
    #[error("Voice ID is available only for file transcription, not Native ASR realtime sessions.")]
    VoiceIdUnsupportedForRealtime,
    #[error(
        "Phrase bias / hotword boosting is not supported by the '{model_family}' native model family ({adapter}). The request was rejected instead of silently ignoring phrase_bias."
    )]
    PhraseBiasUnsupportedByModel {
        adapter: String,
        model_family: String,
    },
    #[error("Native ASR execution device was not found: {detail}.")]
    ExecutionDeviceNotFound { detail: String },
    #[error("Native ASR execution device is not exactly addressable: {detail}.")]
    ExecutionDeviceNotAddressable { detail: String },
    #[error("Native ASR execution device failed to initialize: {detail}.")]
    ExecutionDeviceInitFailed { detail: String },
    #[error("Native ASR session is closed.")]
    SessionClosed,
    #[error("Native ASR session failed: {message}.")]
    SessionFailed { message: String },
}

impl NativeAsrError {
    pub(super) fn invalid_streaming_session_config(message: impl Into<String>) -> Self {
        Self::SessionFailed {
            message: format!(
                "invalid Native ASR streaming session config: {}",
                message.into()
            ),
        }
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(super) fn session_backpressure(message: impl Into<String>) -> Self {
        Self::SessionFailed {
            message: format!(
                "Native ASR session backpressure exceeded: {}",
                message.into()
            ),
        }
    }

    /// Production mapping used when a native session/stream hits a typed
    /// execution-route failure (Exact miss / not-addressable / init failed /
    /// accelerated unavailable). Keep in lockstep with
    /// [`crate::BackendError::from_execution_route_error`].
    pub fn from_execution_route_error(error: ExecutionRouteError) -> Self {
        match error {
            ExecutionRouteError::DeviceNotFound { detail } => {
                Self::ExecutionDeviceNotFound { detail }
            }
            ExecutionRouteError::NotAddressable { detail } => {
                Self::ExecutionDeviceNotAddressable { detail }
            }
            ExecutionRouteError::InitFailed { detail } => {
                Self::ExecutionDeviceInitFailed { detail }
            }
            ExecutionRouteError::AcceleratedUnavailable => Self::ProviderUnavailable {
                provider: "accelerated".to_string(),
            },
        }
    }
}

impl TryFrom<NativeAsrRuntimeReadiness> for NativeAsrError {
    type Error = NativeAsrRuntimeReadiness;

    fn try_from(readiness: NativeAsrRuntimeReadiness) -> Result<Self, Self::Error> {
        match readiness {
            NativeAsrRuntimeReadiness::Ready => Err(NativeAsrRuntimeReadiness::Ready),
            other => Ok(other.into_error()),
        }
    }
}

impl NativeAsrRuntimeReadiness {
    fn into_error(self) -> NativeAsrError {
        match self {
            Self::Ready => NativeAsrError::SessionFailed {
                message: "runtime readiness was Ready".to_string(),
            },
            Self::UnsupportedModelPack { reason } => {
                NativeAsrError::UnsupportedModelPack { reason }
            }
            Self::MissingLocalModelAsset { path } => {
                NativeAsrError::MissingLocalModelAsset { path }
            }
            Self::UnsupportedHardwareTarget { target } => {
                NativeAsrError::UnsupportedHardwareTarget { target }
            }
            Self::ProviderUnavailable { provider } => {
                NativeAsrError::ProviderUnavailable { provider }
            }
            Self::BackendDoesNotSupportTrueStreaming { backend } => {
                NativeAsrError::BackendDoesNotSupportTrueStreaming { backend }
            }
        }
    }
}
