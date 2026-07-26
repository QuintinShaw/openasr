use crate::ggml_runtime::GgmlCpuGraphError;

/// The allocation failures that must retain their identity while a Qwen graph
/// error crosses direct, streaming, and serve-batch decode boundaries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum Qwen3AsrGraphAllocationFailure {
    Context {
        stage: &'static str,
        requested_bytes: usize,
    },
    Host {
        stage: &'static str,
        requested_bytes: usize,
    },
    BackendBuffer {
        backend: String,
    },
}

impl Qwen3AsrGraphAllocationFailure {
    pub(super) fn from_graph_error(error: &GgmlCpuGraphError) -> Option<Self> {
        match error {
            GgmlCpuGraphError::ContextAllocationFailed {
                stage,
                requested_bytes,
            } => Some(Self::Context {
                stage,
                requested_bytes: *requested_bytes,
            }),
            GgmlCpuGraphError::HostAllocationFailed {
                stage,
                requested_bytes,
            } => Some(Self::Host {
                stage,
                requested_bytes: *requested_bytes,
            }),
            GgmlCpuGraphError::BackendBufferAllocationFailed { backend } => {
                Some(Self::BackendBuffer {
                    backend: backend.clone(),
                })
            }
            _ => None,
        }
    }
}
