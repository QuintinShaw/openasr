//! Generic ggml static-tensor-arena weight-residency plumbing.
//!
//! `models/{parakeet_ctc,parakeet_tdt,sensevoice,wav2vec2_ctc}/encoder_graph.rs`
//! each independently reimplemented the identical
//! alloc(rank-dispatched)/upload(f32,f16)/bind-zero-copy skeleton for their
//! per-layer weights, differing only in (a) their own local graph-build error
//! enum and (b) whether rank-4 tensors are supported (subsampling convs need
//! it; FSMN/conv kernels in other families do not). This module owns exactly
//! that shared plumbing and nothing else: it never sees a model-specific
//! tensor name, block-wiring decision, or math op, so it stays usable by any
//! future family with the same "zero-copy bind from the mmap'd pack, else
//! f32/f16 arena upload" shape without encoding that family's semantics here
//! (see AGENTS.md: keep infrastructure model-agnostic).
//!
//! Each family keeps a thin same-named wrapper (`alloc_static`,
//! `upload_static`, `bind_loaded`, ...) around these functions that maps
//! `ArenaAllocError` / the `bind_loaded` error string into its own error enum,
//! so every existing call site in those files is unchanged -- only the
//! function *bodies* moved here.
//!
//! `models/parakeet_tdt/encoder_graph.rs` has no legacy no-pack arena
//! fallback, so it never constructs `WeightSlot::Arena` (see that module for
//! why); the type stays shared regardless since the fallback variant existing
//! but unused is exactly the same "kept for parity" convention already used
//! by `models/moonshine`.

use super::{
    GGML_TYPE_F16, GGML_TYPE_F32, GgmlCpuGraphError, GgmlCpuTensor, GgmlLoadedTensor,
    GgmlLoadedWeightContext, GgmlStaticTensor, GgmlStaticTensorArena,
};

/// A 2-D linear weight: either an arena tensor (f32-uploaded -- legacy / no
/// runtime pack) or a zero-copy leaf bound to the mmap'd pack (native
/// q4_K/f16/f32, no host copy + no arena upload).
#[derive(Clone, Copy)]
pub(crate) enum WeightSlot {
    // No current family constructs this variant (every bindable weight's
    // host payload is dropped, or never materialized, at load, so binding
    // failure always fails closed rather than falling back) -- kept for
    // parity with a possible future non-mmap fallback, same rationale
    // `models::moonshine` already documents for its own (now-replaced) copy.
    #[allow(dead_code)]
    Arena(GgmlStaticTensor),
    Loaded(GgmlLoadedTensor),
}

impl WeightSlot {
    pub(crate) fn graph<'a>(self, arena: &GgmlStaticTensorArena) -> GgmlCpuTensor<'a> {
        match self {
            Self::Arena(handle) => arena.graph_tensor(handle),
            Self::Loaded(tensor) => tensor.as_graph_tensor(),
        }
    }
}

/// Bind a 2-D linear zero-copy from the mmap'd pack (`loaded`) by its on-disk
/// name. Returns `Err(reason)` -- a formatted message, not a family error type
/// -- if the loaded context is absent or the tensor is missing; the caller
/// wraps `reason` in its own "Shape"-style error variant and the returned
/// tensor in its own `WeightSlot`-shaped type (this never constructs `Arena`
/// -- some families' `WeightSlot` has no such variant at all -- so callers
/// that want the shared enum wrap it themselves: `.map(WeightSlot::Loaded)`).
/// FAILS CLOSED: the host f32 values for bound weights are dropped (or never
/// materialized) at load in every current caller of this pipeline, so there
/// is no arena fallback here -- uploading an empty buffer would silently
/// corrupt the graph.
pub(crate) fn bind_loaded(
    loaded: Option<&GgmlLoadedWeightContext>,
    name: &str,
) -> Result<GgmlLoadedTensor, String> {
    match loaded.and_then(|ctx| ctx.tensor(name)) {
        Some(tensor) => Ok(tensor),
        None => Err(format!(
            "2-D linear '{name}' could not be bound zero-copy from the runtime pack \
             (loaded weight context missing or tensor absent); host payload was dropped"
        )),
    }
}

/// A rank-dispatch failure: `dims` didn't match any rank this pipeline
/// allocates for the caller's `support_rank4` setting. Carried as owned data
/// (not a family error type) so this module stays error-enum-agnostic; wrap
/// `GgmlCpuGraphError` transparently via `?` and format `UnsupportedRank`
/// yourself (the message text differs slightly per family: "unsupported
/// rank" vs. "f16 depthwise" vs. "f16 fsmn kernel", etc.).
pub(crate) enum ArenaAllocError {
    Graph(GgmlCpuGraphError),
    UnsupportedRank(Vec<usize>),
}

impl From<GgmlCpuGraphError> for ArenaAllocError {
    fn from(source: GgmlCpuGraphError) -> Self {
        Self::Graph(source)
    }
}

/// Allocate a static arena tensor matching a host weight's stored dims,
/// f32-typed. `support_rank4` gates the 4-D case some families' subsampling
/// convs need and others never emit -- the only thing that varies
/// family-to-family in this allocation shape.
pub(crate) fn alloc_static_f32(
    arena: &GgmlStaticTensorArena,
    dims: &[usize],
    values_len: usize,
    step: &'static str,
    support_rank4: bool,
) -> Result<GgmlStaticTensor, ArenaAllocError> {
    match dims {
        [] | [_] => Ok(arena.new_tensor_1d_f32(values_len, step)?),
        [ne0, ne1] => Ok(arena.new_tensor_2d_f32(*ne0, *ne1, step)?),
        [ne0, ne1, ne2] => Ok(arena.new_tensor_3d_f32(*ne0, *ne1, *ne2, step)?),
        [ne0, ne1, ne2, ne3] if support_rank4 => {
            Ok(arena.new_tensor_4d_typed(*ne0, *ne1, *ne2, *ne3, GGML_TYPE_F32, step)?)
        }
        _ => Err(ArenaAllocError::UnsupportedRank(dims.to_vec())),
    }
}

/// Allocate an f16 arena tensor (ggml `conv_2d_dw` requires an f16 kernel;
/// also used for FSMN kernels). `support_rank4` mirrors `alloc_static_f32`.
pub(crate) fn alloc_static_f16(
    arena: &GgmlStaticTensorArena,
    dims: &[usize],
    step: &'static str,
    support_rank4: bool,
) -> Result<GgmlStaticTensor, ArenaAllocError> {
    match dims {
        [ne0, ne1, ne2] => Ok(arena.new_tensor_3d_typed(*ne0, *ne1, *ne2, GGML_TYPE_F16, step)?),
        [ne0, ne1, ne2, ne3] if support_rank4 => {
            Ok(arena.new_tensor_4d_typed(*ne0, *ne1, *ne2, *ne3, GGML_TYPE_F16, step)?)
        }
        _ => Err(ArenaAllocError::UnsupportedRank(dims.to_vec())),
    }
}

/// Upload f32 host values into an already-allocated arena tensor.
pub(crate) fn upload_static_f32(
    arena: &mut GgmlStaticTensorArena,
    tensor: GgmlStaticTensor,
    values: &[f32],
    step: &'static str,
) -> Result<(), GgmlCpuGraphError> {
    arena.set_f32_slice(tensor, values, step)
}

/// Convert f32 host values to f16 bits and upload into an already-allocated
/// f16 arena tensor. `f32_to_f16_bits` is threaded through by the caller
/// (rather than imported here) so this module stays free of any dependency on
/// `models::*` -- every current family already has the conversion in scope.
pub(crate) fn upload_static_f16(
    arena: &mut GgmlStaticTensorArena,
    tensor: GgmlStaticTensor,
    values: &[f32],
    step: &'static str,
    f32_to_f16_bits: impl Fn(f32) -> u16,
) -> Result<(), GgmlCpuGraphError> {
    let bits: Vec<u16> = values.iter().copied().map(f32_to_f16_bits).collect();
    arena.set_f16_bits_slice(tensor, &bits, step)
}
