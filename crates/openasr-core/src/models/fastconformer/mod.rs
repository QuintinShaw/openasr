//! Shared FastConformer encoder infrastructure: the dw-striding subsampling
//! prelude + `nn::encoder::conformer_block` stack that `parakeet_ctc` and
//! `parakeet_tdt` both build byte-for-byte identically (the TDT checkpoint
//! honors `scale_input`/no-bias metadata the CTC checkpoint does not, but
//! the graph shape and math are the same builder either way).
//!
//! This module owns the weight-loading (BatchNorm fold, zero-bias synthesis
//! for bias-free checkpoints) and graph-building (arena alloc/upload/bind,
//! subsampling conv chain, conformer layer loop) skeleton and nothing else:
//! it never picks a tail (CTC head vs. joint encoder projection) or owns a
//! family error type. Each family keeps its own `encoder_weights.rs` /
//! `encoder_graph.rs` with its own public types (`Parakeet*EncoderGraph`,
//! `Parakeet*MelFeatures`, ...) and error enum; those enums implement
//! [`FastConformerGraphError`] / [`FastConformerWeightsError`] (a handful of
//! one-line constructors) so the shared builders stay generic over `E`,
//! mirroring the `map_err` pattern `nn::encoder::conformer_block` already
//! uses (see AGENTS.md: keep infrastructure model-agnostic).
//!
//! Numeric behavior is carried over byte-for-byte from the pre-refactor
//! per-family copies -- nothing here changes the math, only where it lives.

pub(crate) mod graph;
pub(crate) mod weights;

pub(crate) use graph::{
    FastConformerEncoderCore, FastConformerStackConfig, alloc_static, bind_loaded,
    build_conformer_stack, upload_graph_f32, upload_static,
};
pub(crate) use weights::{
    FastConformerLayerWeights, NamedTensor, load_fastconformer_layer,
    load_fastconformer_subsampling, load_named,
};

use crate::ggml_runtime::{GgmlCpuGraphError, GgufTensorDataReadError};

/// Error hook the shared graph builders need. Each family's own graph-build
/// error enum implements this with two one-line constructors mapping onto
/// its existing variants, so the shared code never owns (or has to spell)
/// a model-specific error type.
pub(crate) trait FastConformerGraphError: Sized {
    fn graph_build_failed(step: &'static str, source: GgmlCpuGraphError) -> Self;
    fn shape(reason: String) -> Self;
}

/// Error hook the shared weight loaders need. `From<GgufTensorDataReadError>`
/// lets `?` keep working at every GGUF read call site exactly as it did in
/// each family's own (now-shared) loader -- both existing error enums
/// already derive this via `#[error(...)] Read(#[from] GgufTensorDataReadError)`.
pub(crate) trait FastConformerWeightsError: Sized + From<GgufTensorDataReadError> {
    fn batchnorm_fold(reason: String) -> Self;
}
