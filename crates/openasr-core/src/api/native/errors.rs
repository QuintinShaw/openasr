use thiserror::Error;

use super::{NativeAsrHardwareTarget, NativeAsrRuntimeReadiness};

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
    #[error(
        "Phrase bias / hotword boosting is not supported by the '{model_family}' native model family ({adapter}). The request was rejected instead of silently ignoring phrase_bias."
    )]
    PhraseBiasUnsupportedByModel {
        adapter: String,
        model_family: String,
    },
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
