use std::ffi::{c_char, c_int, c_void};

pub(crate) type GgmlBackendRaw = *mut c_void;
pub(crate) type GgmlBackendDevRaw = *mut c_void;
pub(crate) type GgmlBackendBufferRaw = *mut c_void;
pub(crate) type GgmlBackendBufferTypeRaw = *mut c_void;
pub(crate) type GgmlBackendSchedRaw = *mut c_void;
pub(crate) type GgmlBackendRegRaw = *mut c_void;
pub(crate) type GgmlContextRaw = *mut c_void;
pub(crate) type GgmlTensorRaw = *mut c_void;
pub(crate) type GgmlCgraphRaw = *mut c_void;
pub(crate) type GgufContextRaw = *mut c_void;
pub(crate) const GGML_MAX_DIMS: usize = 4;

pub(crate) type GgmlToFloatFn = unsafe extern "C" fn(x: *const c_void, y: *mut f32, k: i64);

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct GgmlTensorLayoutPrefix {
    pub type_: c_int,
    pub buffer: *mut c_void,
    pub ne: [i64; GGML_MAX_DIMS],
    pub nb: [usize; GGML_MAX_DIMS],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct GgufInitParams {
    pub no_alloc: bool,
    pub ctx: *mut *mut c_void,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct GgmlInitParams {
    pub mem_size: usize,
    pub mem_buffer: *mut c_void,
    pub no_alloc: bool,
}

pub(crate) const GGML_BACKEND_DEVICE_TYPE_CPU: c_int = 0;
pub(crate) const GGML_BACKEND_DEVICE_TYPE_GPU: c_int = 1;
pub(crate) const GGML_BACKEND_DEVICE_TYPE_IGPU: c_int = 2;
pub(crate) const GGML_BACKEND_DEVICE_TYPE_ACCEL: c_int = 3;
pub(crate) const GGML_BACKEND_DEVICE_TYPE_META: c_int = 4;

pub(crate) const GGML_STATUS_SUCCESS: c_int = 0;
pub(crate) const GGML_BACKEND_BUFFER_USAGE_WEIGHTS: c_int = 1;

pub(crate) const GGML_TYPE_F32: c_int = 0;
pub(crate) const GGML_TYPE_F16: c_int = 1;
pub(crate) const GGML_TYPE_Q4_0: c_int = 2;
pub(crate) const GGML_TYPE_Q8_0: c_int = 8;
pub(crate) const GGML_TYPE_Q3_K: c_int = 11;
pub(crate) const GGML_TYPE_Q4_K: c_int = 12;
pub(crate) const GGML_TYPE_Q5_K: c_int = 13;
pub(crate) const GGML_TYPE_Q6_K: c_int = 14;
pub(crate) const GGML_TYPE_I32: c_int = 26;

#[allow(dead_code)]
pub(crate) const GGML_ROPE_TYPE_NEOX: c_int = 2;

/// GPT-J / interleaved RoPE layout (rotates adjacent pairs x[2i], x[2i+1]).
/// Matches HuggingFace `repeat_interleave(2)` rotary embedding (e.g. Moonshine).
#[allow(dead_code)]
pub(crate) const GGML_ROPE_TYPE_NORMAL: c_int = 0;

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct GgmlTypeTraits {
    pub type_name: *const c_char,
    pub blck_size: i64,
    pub blck_size_interleave: i64,
    pub type_size: usize,
    pub is_quantized: bool,
    pub to_float: Option<GgmlToFloatFn>,
    pub from_float_ref: *const c_void,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct GgmlBackendDevCaps {
    pub async_: bool,
    pub host_buffer: bool,
    pub buffer_from_host_ptr: bool,
    pub events: bool,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct GgmlBackendDevProps {
    pub name: *const c_char,
    pub description: *const c_char,
    pub memory_free: usize,
    pub memory_total: usize,
    pub type_: c_int,
    pub device_id: *const c_char,
    pub caps: GgmlBackendDevCaps,
}

pub(crate) const GGUF_TYPE_UINT32: c_int = 4;
pub(crate) const GGUF_TYPE_FLOAT32: c_int = 6;
pub(crate) const GGUF_TYPE_BOOL: c_int = 7;
pub(crate) const GGUF_TYPE_STRING: c_int = 8;
pub(crate) const GGUF_TYPE_ARRAY: c_int = 9;
pub(crate) const GGUF_TYPE_UINT64: c_int = 10;

unsafe extern "C" {
    pub(crate) fn ggml_backend_name(backend: GgmlBackendRaw) -> *const c_char;
    pub(crate) fn ggml_backend_free(backend: GgmlBackendRaw);
    pub(crate) fn ggml_backend_init_best() -> GgmlBackendRaw;
    pub(crate) fn ggml_backend_init_by_type(
        dev_type: c_int,
        params: *const c_char,
    ) -> GgmlBackendRaw;
    // GGML_BACKEND_DL: register backend plugin DLLs (ggml-<name>-*.dll / .so).
    // Must run before any registry query under DL, where the static GGML_USE_*
    // backend registration is compiled out (empty registry, init_best returns
    // null, otherwise). `load_all` scans next to the executable + GGML_BACKEND_PATH
    // (the base installer's CPU variants); `load_all_from_path` scans an explicit
    // directory — used to register the downloaded GPU packs under
    // OPENASR_HOME/backends/<vendor>/<version>/.
    pub(crate) fn ggml_backend_load_all();
    pub(crate) fn ggml_backend_load_all_from_path(dir_path: *const c_char);
    pub(crate) fn ggml_backend_buffer_free(buffer: GgmlBackendBufferRaw);
    pub(crate) fn ggml_backend_buffer_is_host(buffer: GgmlBackendBufferRaw) -> bool;
    pub(crate) fn ggml_backend_buffer_set_usage(buffer: GgmlBackendBufferRaw, usage: c_int);
    pub(crate) fn ggml_backend_graph_compute(
        backend: GgmlBackendRaw,
        cgraph: GgmlCgraphRaw,
    ) -> c_int;
    pub(crate) fn ggml_backend_sched_new(
        backends: *mut GgmlBackendRaw,
        bufts: *mut GgmlBackendBufferTypeRaw,
        n_backends: c_int,
        graph_size: usize,
        parallel: bool,
        op_offload: bool,
    ) -> GgmlBackendSchedRaw;
    pub(crate) fn ggml_backend_sched_free(sched: GgmlBackendSchedRaw);
    pub(crate) fn ggml_backend_sched_reset(sched: GgmlBackendSchedRaw);
    pub(crate) fn ggml_backend_sched_alloc_graph(
        sched: GgmlBackendSchedRaw,
        cgraph: GgmlCgraphRaw,
    ) -> bool;
    pub(crate) fn ggml_backend_sched_graph_compute(
        sched: GgmlBackendSchedRaw,
        cgraph: GgmlCgraphRaw,
    ) -> c_int;
    pub(crate) fn ggml_backend_tensor_set(
        tensor: GgmlTensorRaw,
        data: *const c_void,
        offset: usize,
        size: usize,
    );
    pub(crate) fn ggml_backend_tensor_get(
        tensor: GgmlTensorRaw,
        data: *mut c_void,
        offset: usize,
        size: usize,
    );
    pub(crate) fn ggml_backend_tensor_alloc(
        buffer: GgmlBackendBufferRaw,
        tensor: GgmlTensorRaw,
        addr: *mut c_void,
    ) -> c_int;
    pub(crate) fn ggml_backend_alloc_ctx_tensors(
        ctx: GgmlContextRaw,
        backend: GgmlBackendRaw,
    ) -> GgmlBackendBufferRaw;
    pub(crate) fn ggml_backend_get_device(backend: GgmlBackendRaw) -> GgmlBackendDevRaw;

    pub(crate) fn ggml_backend_dev_count() -> usize;
    pub(crate) fn ggml_backend_dev_get(index: usize) -> GgmlBackendDevRaw;
    pub(crate) fn ggml_backend_dev_name(device: GgmlBackendDevRaw) -> *const c_char;
    pub(crate) fn ggml_backend_dev_description(device: GgmlBackendDevRaw) -> *const c_char;
    pub(crate) fn ggml_backend_dev_type(device: GgmlBackendDevRaw) -> c_int;
    pub(crate) fn ggml_backend_dev_memory(
        device: GgmlBackendDevRaw,
        free: *mut usize,
        total: *mut usize,
    );
    pub(crate) fn ggml_backend_dev_get_props(
        device: GgmlBackendDevRaw,
        props: *mut GgmlBackendDevProps,
    );
    pub(crate) fn ggml_backend_dev_init(
        device: GgmlBackendDevRaw,
        params: *const c_char,
    ) -> GgmlBackendRaw;
    pub(crate) fn ggml_backend_dev_buffer_from_host_ptr(
        device: GgmlBackendDevRaw,
        ptr: *mut c_void,
        size: usize,
        max_tensor_size: usize,
    ) -> GgmlBackendBufferRaw;
    pub(crate) fn ggml_backend_dev_supports_op(
        device: GgmlBackendDevRaw,
        op: GgmlTensorRaw,
    ) -> bool;
    pub(crate) fn ggml_backend_dev_backend_reg(device: GgmlBackendDevRaw) -> GgmlBackendRegRaw;
    pub(crate) fn ggml_backend_reg_get_proc_address(
        reg: GgmlBackendRegRaw,
        name: *const c_char,
    ) -> *mut c_void;

    // The host sets CPU threads through the registry proc-address table
    // (`backend_set_n_threads`), which works under GGML_BACKEND_DL where the
    // ggml-cpu plugin's symbols are not linked into the core. The macOS
    // BLAS-accelerator path calls ggml_backend_blas_set_n_threads directly.
    #[cfg(target_os = "macos")]
    pub(crate) fn ggml_backend_blas_init() -> GgmlBackendRaw;
    #[cfg(target_os = "macos")]
    pub(crate) fn ggml_backend_blas_set_n_threads(backend: GgmlBackendRaw, n_threads: c_int);
    pub(crate) fn ggml_init(params: GgmlInitParams) -> GgmlContextRaw;
    pub(crate) fn ggml_reset(ctx: GgmlContextRaw);
    pub(crate) fn ggml_free(ctx: GgmlContextRaw);
    pub(crate) fn ggml_blck_size(type_: c_int) -> i64;
    pub(crate) fn ggml_type_size(type_: c_int) -> usize;
    pub(crate) fn ggml_row_size(type_: c_int, ne: i64) -> usize;
    #[allow(dead_code)]
    pub(crate) fn ggml_is_quantized(type_: c_int) -> bool;
    pub(crate) fn ggml_get_type_traits(type_: c_int) -> *const GgmlTypeTraits;
    pub(crate) fn ggml_quantize_chunk(
        type_: c_int,
        src: *const f32,
        dst: *mut c_void,
        start: i64,
        nrows: i64,
        n_per_row: i64,
        imatrix: *const f32,
    ) -> usize;
    pub(crate) fn ggml_get_data(tensor: GgmlTensorRaw) -> *mut c_void;
    pub(crate) fn ggml_get_first_tensor(ctx: GgmlContextRaw) -> GgmlTensorRaw;
    pub(crate) fn ggml_get_next_tensor(ctx: GgmlContextRaw, tensor: GgmlTensorRaw)
    -> GgmlTensorRaw;
    pub(crate) fn ggml_get_name(tensor: GgmlTensorRaw) -> *const c_char;
    pub(crate) fn ggml_nbytes(tensor: GgmlTensorRaw) -> usize;
    pub(crate) fn ggml_new_tensor_1d(ctx: GgmlContextRaw, type_: c_int, ne0: i64) -> GgmlTensorRaw;
    pub(crate) fn ggml_new_tensor_2d(
        ctx: GgmlContextRaw,
        type_: c_int,
        ne0: i64,
        ne1: i64,
    ) -> GgmlTensorRaw;
    pub(crate) fn ggml_new_tensor_3d(
        ctx: GgmlContextRaw,
        type_: c_int,
        ne0: i64,
        ne1: i64,
        ne2: i64,
    ) -> GgmlTensorRaw;
    pub(crate) fn ggml_new_tensor_4d(
        ctx: GgmlContextRaw,
        type_: c_int,
        ne0: i64,
        ne1: i64,
        ne2: i64,
        ne3: i64,
    ) -> GgmlTensorRaw;
    pub(crate) fn ggml_set_name(tensor: GgmlTensorRaw, name: *const c_char) -> GgmlTensorRaw;
    pub(crate) fn ggml_add(
        ctx: GgmlContextRaw,
        a: GgmlTensorRaw,
        b: GgmlTensorRaw,
    ) -> GgmlTensorRaw;
    pub(crate) fn ggml_sub(
        ctx: GgmlContextRaw,
        a: GgmlTensorRaw,
        b: GgmlTensorRaw,
    ) -> GgmlTensorRaw;
    pub(crate) fn ggml_mul(
        ctx: GgmlContextRaw,
        a: GgmlTensorRaw,
        b: GgmlTensorRaw,
    ) -> GgmlTensorRaw;
    pub(crate) fn ggml_div(
        ctx: GgmlContextRaw,
        a: GgmlTensorRaw,
        b: GgmlTensorRaw,
    ) -> GgmlTensorRaw;
    pub(crate) fn ggml_sqr(ctx: GgmlContextRaw, a: GgmlTensorRaw) -> GgmlTensorRaw;
    pub(crate) fn ggml_sqrt(ctx: GgmlContextRaw, a: GgmlTensorRaw) -> GgmlTensorRaw;
    pub(crate) fn ggml_log(ctx: GgmlContextRaw, a: GgmlTensorRaw) -> GgmlTensorRaw;
    pub(crate) fn ggml_mul_mat(
        ctx: GgmlContextRaw,
        a: GgmlTensorRaw,
        b: GgmlTensorRaw,
    ) -> GgmlTensorRaw;
    pub(crate) fn ggml_get_rows(
        ctx: GgmlContextRaw,
        a: GgmlTensorRaw,
        b: GgmlTensorRaw,
    ) -> GgmlTensorRaw;
    pub(crate) fn ggml_argmax(ctx: GgmlContextRaw, a: GgmlTensorRaw) -> GgmlTensorRaw;
    #[cfg(test)]
    pub(crate) fn ggml_top_k(ctx: GgmlContextRaw, a: GgmlTensorRaw, k: c_int) -> GgmlTensorRaw;
    pub(crate) fn ggml_scale(ctx: GgmlContextRaw, a: GgmlTensorRaw, s: f32) -> GgmlTensorRaw;
    pub(crate) fn ggml_sum(ctx: GgmlContextRaw, a: GgmlTensorRaw) -> GgmlTensorRaw;
    pub(crate) fn ggml_sum_rows(ctx: GgmlContextRaw, a: GgmlTensorRaw) -> GgmlTensorRaw;
    pub(crate) fn ggml_mean(ctx: GgmlContextRaw, a: GgmlTensorRaw) -> GgmlTensorRaw;
    pub(crate) fn ggml_norm(ctx: GgmlContextRaw, a: GgmlTensorRaw, eps: f32) -> GgmlTensorRaw;
    // group normalize along ne0*ne1*n_groups (ggml.h:1382). wav2vec2 base uses
    // feat_extract_norm=="group" with n_groups == n_channels (per-channel norm).
    pub(crate) fn ggml_group_norm(
        ctx: GgmlContextRaw,
        a: GgmlTensorRaw,
        n_groups: c_int,
        eps: f32,
    ) -> GgmlTensorRaw;
    // concat a and b along `dim` (ggml.h:1084). Used to stitch per-group conv_1d
    // outputs back into one [out_channels, T] tensor for the grouped pos-conv.
    pub(crate) fn ggml_concat(
        ctx: GgmlContextRaw,
        a: GgmlTensorRaw,
        b: GgmlTensorRaw,
        dim: c_int,
    ) -> GgmlTensorRaw;
    #[allow(dead_code)]
    pub(crate) fn ggml_rms_norm(ctx: GgmlContextRaw, a: GgmlTensorRaw, eps: f32) -> GgmlTensorRaw;
    #[allow(dead_code)]
    pub(crate) fn ggml_repeat(
        ctx: GgmlContextRaw,
        a: GgmlTensorRaw,
        b: GgmlTensorRaw,
    ) -> GgmlTensorRaw;
    #[allow(dead_code)]
    pub(crate) fn ggml_repeat_4d(
        ctx: GgmlContextRaw,
        a: GgmlTensorRaw,
        ne0: i64,
        ne1: i64,
        ne2: i64,
        ne3: i64,
    ) -> GgmlTensorRaw;
    pub(crate) fn ggml_soft_max(ctx: GgmlContextRaw, a: GgmlTensorRaw) -> GgmlTensorRaw;
    pub(crate) fn ggml_soft_max_ext(
        ctx: GgmlContextRaw,
        a: GgmlTensorRaw,
        mask: GgmlTensorRaw,
        scale: f32,
        max_bias: f32,
    ) -> GgmlTensorRaw;
    pub(crate) fn ggml_flash_attn_ext(
        ctx: GgmlContextRaw,
        q: GgmlTensorRaw,
        k: GgmlTensorRaw,
        v: GgmlTensorRaw,
        mask: GgmlTensorRaw,
        scale: f32,
        max_bias: f32,
        logit_softcap: f32,
    ) -> GgmlTensorRaw;
    pub(crate) fn ggml_can_repeat(a: GgmlTensorRaw, b: GgmlTensorRaw) -> bool;
    pub(crate) fn ggml_gelu(ctx: GgmlContextRaw, a: GgmlTensorRaw) -> GgmlTensorRaw;
    pub(crate) fn ggml_gelu_erf(ctx: GgmlContextRaw, a: GgmlTensorRaw) -> GgmlTensorRaw;
    pub(crate) fn ggml_tanh(ctx: GgmlContextRaw, a: GgmlTensorRaw) -> GgmlTensorRaw;
    pub(crate) fn ggml_relu(ctx: GgmlContextRaw, a: GgmlTensorRaw) -> GgmlTensorRaw;
    pub(crate) fn ggml_sigmoid(ctx: GgmlContextRaw, a: GgmlTensorRaw) -> GgmlTensorRaw;
    pub(crate) fn ggml_softplus(ctx: GgmlContextRaw, a: GgmlTensorRaw) -> GgmlTensorRaw;
    pub(crate) fn ggml_exp(ctx: GgmlContextRaw, a: GgmlTensorRaw) -> GgmlTensorRaw;
    #[allow(dead_code)]
    pub(crate) fn ggml_silu(ctx: GgmlContextRaw, a: GgmlTensorRaw) -> GgmlTensorRaw;
    #[allow(dead_code)]
    pub(crate) fn ggml_cast(ctx: GgmlContextRaw, a: GgmlTensorRaw, type_: c_int) -> GgmlTensorRaw;
    pub(crate) fn ggml_cont(ctx: GgmlContextRaw, a: GgmlTensorRaw) -> GgmlTensorRaw;
    pub(crate) fn ggml_nelements(tensor: GgmlTensorRaw) -> i64;
    pub(crate) fn ggml_is_transposed(tensor: GgmlTensorRaw) -> bool;
    pub(crate) fn ggml_is_contiguous(tensor: GgmlTensorRaw) -> bool;
    pub(crate) fn ggml_reshape_2d(
        ctx: GgmlContextRaw,
        a: GgmlTensorRaw,
        ne0: i64,
        ne1: i64,
    ) -> GgmlTensorRaw;
    pub(crate) fn ggml_reshape_1d(ctx: GgmlContextRaw, a: GgmlTensorRaw, ne0: i64)
    -> GgmlTensorRaw;
    pub(crate) fn ggml_reshape_3d(
        ctx: GgmlContextRaw,
        a: GgmlTensorRaw,
        ne0: i64,
        ne1: i64,
        ne2: i64,
    ) -> GgmlTensorRaw;
    pub(crate) fn ggml_reshape_4d(
        ctx: GgmlContextRaw,
        a: GgmlTensorRaw,
        ne0: i64,
        ne1: i64,
        ne2: i64,
        ne3: i64,
    ) -> GgmlTensorRaw;
    pub(crate) fn ggml_view_2d(
        ctx: GgmlContextRaw,
        a: GgmlTensorRaw,
        ne0: i64,
        ne1: i64,
        nb1: usize,
        offset: usize,
    ) -> GgmlTensorRaw;
    pub(crate) fn ggml_view_1d(
        ctx: GgmlContextRaw,
        a: GgmlTensorRaw,
        ne0: i64,
        offset: usize,
    ) -> GgmlTensorRaw;
    pub(crate) fn ggml_view_3d(
        ctx: GgmlContextRaw,
        a: GgmlTensorRaw,
        ne0: i64,
        ne1: i64,
        ne2: i64,
        nb1: usize,
        nb2: usize,
        offset: usize,
    ) -> GgmlTensorRaw;
    #[allow(dead_code)]
    pub(crate) fn ggml_view_4d(
        ctx: GgmlContextRaw,
        a: GgmlTensorRaw,
        ne0: i64,
        ne1: i64,
        ne2: i64,
        ne3: i64,
        nb1: usize,
        nb2: usize,
        nb3: usize,
        offset: usize,
    ) -> GgmlTensorRaw;
    pub(crate) fn ggml_cpy(
        ctx: GgmlContextRaw,
        a: GgmlTensorRaw,
        b: GgmlTensorRaw,
    ) -> GgmlTensorRaw;
    #[allow(dead_code)]
    pub(crate) fn ggml_set_rows(
        ctx: GgmlContextRaw,
        a: GgmlTensorRaw,
        b: GgmlTensorRaw,
        c: GgmlTensorRaw,
    ) -> GgmlTensorRaw;
    pub(crate) fn ggml_transpose(ctx: GgmlContextRaw, a: GgmlTensorRaw) -> GgmlTensorRaw;
    pub(crate) fn ggml_permute(
        ctx: GgmlContextRaw,
        a: GgmlTensorRaw,
        axis0: c_int,
        axis1: c_int,
        axis2: c_int,
        axis3: c_int,
    ) -> GgmlTensorRaw;
    pub(crate) fn ggml_conv_1d(
        ctx: GgmlContextRaw,
        a: GgmlTensorRaw,
        b: GgmlTensorRaw,
        s0: c_int,
        p0: c_int,
        d0: c_int,
    ) -> GgmlTensorRaw;
    pub(crate) fn ggml_conv_2d(
        ctx: GgmlContextRaw,
        a: GgmlTensorRaw,
        b: GgmlTensorRaw,
        s0: c_int,
        s1: c_int,
        p0: c_int,
        p1: c_int,
        d0: c_int,
        d1: c_int,
    ) -> GgmlTensorRaw;
    pub(crate) fn ggml_conv_2d_dw(
        ctx: GgmlContextRaw,
        a: GgmlTensorRaw,
        b: GgmlTensorRaw,
        s0: c_int,
        s1: c_int,
        p0: c_int,
        p1: c_int,
        d0: c_int,
        d1: c_int,
    ) -> GgmlTensorRaw;
    pub(crate) fn ggml_conv_2d_dw_direct(
        ctx: GgmlContextRaw,
        a: GgmlTensorRaw,
        b: GgmlTensorRaw,
        stride0: c_int,
        stride1: c_int,
        pad0: c_int,
        pad1: c_int,
        dilation0: c_int,
        dilation1: c_int,
    ) -> GgmlTensorRaw;
    #[allow(dead_code)]
    pub(crate) fn ggml_rope_ext(
        ctx: GgmlContextRaw,
        a: GgmlTensorRaw,
        b: GgmlTensorRaw,
        c: GgmlTensorRaw,
        n_dims: c_int,
        mode: c_int,
        n_ctx_orig: c_int,
        freq_base: f32,
        freq_scale: f32,
        ext_factor: f32,
        attn_factor: f32,
        beta_fast: f32,
        beta_slow: f32,
    ) -> GgmlTensorRaw;
    pub(crate) fn ggml_new_graph_custom(
        ctx: GgmlContextRaw,
        size: usize,
        grads: bool,
    ) -> GgmlCgraphRaw;
    pub(crate) fn ggml_build_forward_expand(cgraph: GgmlCgraphRaw, tensor: GgmlTensorRaw);
    pub(crate) fn ggml_set_input(tensor: GgmlTensorRaw);
    pub(crate) fn ggml_set_output(tensor: GgmlTensorRaw);
    // Sizing helpers for no_alloc metadata contexts. `ggml_tensor_overhead`
    // returns the per-tensor bookkeeping cost (ggml_object + ggml_tensor) and
    // `ggml_graph_overhead_custom` the bytes a cgraph of `size` nodes consumes
    // inside its context — together they give the EXACT capacity a metadata-only
    // context needs, mirroring llama.cpp's compute-meta buffer sizing.
    pub(crate) fn ggml_tensor_overhead() -> usize;
    pub(crate) fn ggml_graph_overhead_custom(size: usize, grads: bool) -> usize;
    // ggml_cpu_has_* / ggml_cpu_get_* are not declared here: under
    // GGML_BACKEND_DL they live in the loaded ggml-cpu plugin, not the linked
    // core. GgmlCpuFeatures::detect() reads CPU features via the Rust stdlib
    // instead (build-mode-agnostic).

    pub(crate) fn gguf_init_from_file(
        fname: *const c_char,
        params: GgufInitParams,
    ) -> GgufContextRaw;
    pub(crate) fn gguf_free(ctx: GgufContextRaw);
    pub(crate) fn gguf_get_n_kv(ctx: *const c_void) -> i64;
    pub(crate) fn gguf_get_key(ctx: *const c_void, key_id: i64) -> *const c_char;
    pub(crate) fn gguf_get_kv_type(ctx: *const c_void, key_id: i64) -> c_int;
    pub(crate) fn gguf_get_arr_type(ctx: *const c_void, key_id: i64) -> c_int;
    pub(crate) fn gguf_get_val_u32(ctx: *const c_void, key_id: i64) -> u32;
    pub(crate) fn gguf_get_val_u64(ctx: *const c_void, key_id: i64) -> u64;
    pub(crate) fn gguf_get_val_f32(ctx: *const c_void, key_id: i64) -> f32;
    pub(crate) fn gguf_get_val_bool(ctx: *const c_void, key_id: i64) -> bool;
    pub(crate) fn gguf_get_val_str(ctx: *const c_void, key_id: i64) -> *const c_char;
    pub(crate) fn gguf_get_arr_n(ctx: *const c_void, key_id: i64) -> usize;
    pub(crate) fn gguf_get_arr_data(ctx: *const c_void, key_id: i64) -> *const c_void;
    pub(crate) fn gguf_get_arr_str(ctx: *const c_void, key_id: i64, i: usize) -> *const c_char;
    pub(crate) fn gguf_get_data_offset(ctx: *const c_void) -> usize;
    pub(crate) fn gguf_get_n_tensors(ctx: *const c_void) -> i64;
    pub(crate) fn gguf_get_tensor_offset(ctx: *const c_void, tensor_id: i64) -> usize;
    pub(crate) fn gguf_get_tensor_name(ctx: *const c_void, tensor_id: i64) -> *const c_char;
    pub(crate) fn gguf_get_tensor_n_dims(ctx: *const c_void, tensor_id: i64) -> u32;
    pub(crate) fn gguf_get_tensor_dim(ctx: *const c_void, tensor_id: i64, dim: c_int) -> i64;
    pub(crate) fn gguf_get_tensor_type(ctx: *const c_void, tensor_id: i64) -> c_int;
    pub(crate) fn gguf_get_tensor_size(ctx: *const c_void, tensor_id: i64) -> usize;
    pub(crate) fn ggml_type_name(type_: c_int) -> *const c_char;

    pub(crate) fn gguf_init_empty() -> GgufContextRaw;
    pub(crate) fn gguf_set_val_u32(ctx: GgufContextRaw, key: *const c_char, val: u32);
    #[cfg(test)]
    pub(crate) fn gguf_set_val_u64(ctx: GgufContextRaw, key: *const c_char, val: u64);
    #[cfg(test)]
    pub(crate) fn gguf_set_val_f32(ctx: GgufContextRaw, key: *const c_char, val: f32);
    #[cfg(test)]
    pub(crate) fn gguf_set_val_bool(ctx: GgufContextRaw, key: *const c_char, val: bool);
    pub(crate) fn gguf_set_val_str(ctx: GgufContextRaw, key: *const c_char, val: *const c_char);
    pub(crate) fn gguf_set_arr_data(
        ctx: GgufContextRaw,
        key: *const c_char,
        type_: c_int,
        data: *const c_void,
        n: usize,
    );
    pub(crate) fn gguf_set_arr_str(
        ctx: GgufContextRaw,
        key: *const c_char,
        data: *const *const c_char,
        n: usize,
    );
    pub(crate) fn gguf_add_tensor(ctx: GgufContextRaw, tensor: GgmlTensorRaw);
    pub(crate) fn gguf_set_tensor_type(ctx: GgufContextRaw, name: *const c_char, type_: c_int);
    pub(crate) fn gguf_set_tensor_data(
        ctx: GgufContextRaw,
        name: *const c_char,
        data: *const c_void,
    );
    pub(crate) fn gguf_write_to_file(
        ctx: *const c_void,
        fname: *const c_char,
        only_meta: bool,
    ) -> bool;

    #[cfg(target_os = "macos")]
    pub(crate) fn ggml_backend_metal_init() -> GgmlBackendRaw;
}
