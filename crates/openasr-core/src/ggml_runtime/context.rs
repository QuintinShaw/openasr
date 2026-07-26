use std::{
    alloc::{Layout, alloc, dealloc},
    ffi::c_void,
    ptr::NonNull,
};

use thiserror::Error;

use super::ffi;

/// The caller-owned buffers supplied to `ggml_try_init` are deliberately
/// over-aligned to ggml's current 64-byte requirement. Keeping that alignment
/// fixed in Rust makes allocation failure return a typed error before C sees a
/// pointer, rather than relying on ggml's legacy allocating initializer.
const CONTEXT_ALIGNMENT: usize = 64;

#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum GgmlContextAllocationError {
    #[error("ggml context allocation failed at {stage} (requested_bytes={requested_bytes})")]
    AllocationFailed {
        stage: &'static str,
        requested_bytes: usize,
    },
    #[error(
        "ggml context allocation layout is invalid at {stage} (requested_bytes={requested_bytes})"
    )]
    InvalidLayout {
        stage: &'static str,
        requested_bytes: usize,
    },
    #[error(
        "ggml context initialization rejected caller-owned storage (requested_bytes={requested_bytes})"
    )]
    InitializationFailed { requested_bytes: usize },
}

#[derive(Debug)]
struct AlignedAllocation {
    ptr: NonNull<u8>,
    layout: Layout,
}

impl AlignedAllocation {
    fn new_with_allocator(
        stage: &'static str,
        requested_bytes: usize,
        allocate: impl FnOnce(Layout) -> *mut u8,
    ) -> Result<Self, GgmlContextAllocationError> {
        let size = requested_bytes.max(CONTEXT_ALIGNMENT);
        let layout = Layout::from_size_align(size, CONTEXT_ALIGNMENT).map_err(|_| {
            GgmlContextAllocationError::InvalidLayout {
                stage,
                requested_bytes,
            }
        })?;
        let ptr =
            NonNull::new(allocate(layout)).ok_or(GgmlContextAllocationError::AllocationFailed {
                stage,
                requested_bytes,
            })?;
        Ok(Self { ptr, layout })
    }

    fn as_mut_ptr(&self) -> *mut c_void {
        self.ptr.as_ptr().cast()
    }
}

impl Drop for AlignedAllocation {
    fn drop(&mut self) {
        unsafe { dealloc(self.ptr.as_ptr(), self.layout) };
    }
}

/// A fallible, caller-owned ggml context. Its declaration order and `Drop`
/// implementation guarantee that `ggml_free` observes both storage buffers
/// alive, then they are released exactly once after ggml has detached from them.
#[derive(Debug)]
pub(crate) struct GgmlCallerOwnedContext {
    raw: NonNull<c_void>,
    // These fields are intentionally unread after initialization: they own the
    // buffers whose addresses ggml retains until `ggml_free` returns.
    _context_storage: AlignedAllocation,
    _pool_storage: AlignedAllocation,
}

impl GgmlCallerOwnedContext {
    pub(crate) fn new(requested_bytes: usize) -> Result<Self, GgmlContextAllocationError> {
        Self::new_with_allocator(requested_bytes, |layout| unsafe { alloc(layout) })
    }

    fn new_with_allocator(
        requested_bytes: usize,
        mut allocate: impl FnMut(Layout) -> *mut u8,
    ) -> Result<Self, GgmlContextAllocationError> {
        let context_bytes = unsafe { ffi::ggml_context_size() };
        let context_storage =
            AlignedAllocation::new_with_allocator("context", context_bytes, |layout| {
                allocate(layout)
            })?;
        let pool_storage =
            AlignedAllocation::new_with_allocator("pool", requested_bytes, |layout| {
                allocate(layout)
            })?;
        let raw = unsafe {
            ffi::ggml_try_init(
                ffi::GgmlInitParams {
                    mem_size: requested_bytes,
                    mem_buffer: pool_storage.as_mut_ptr(),
                    no_alloc: true,
                },
                context_storage.as_mut_ptr(),
                context_bytes,
            )
        };
        let raw = NonNull::new(raw)
            .ok_or(GgmlContextAllocationError::InitializationFailed { requested_bytes })?;
        Ok(Self {
            raw,
            _context_storage: context_storage,
            _pool_storage: pool_storage,
        })
    }

    pub(crate) fn raw(&self) -> NonNull<c_void> {
        self.raw
    }
}

impl Drop for GgmlCallerOwnedContext {
    fn drop(&mut self) {
        // `ggml_free` does not free caller-owned storage, but it can still read
        // the context and pool while releasing ggml bookkeeping. Field drops run
        // only after this method returns, preserving the required lifetime.
        unsafe { ffi::ggml_free(self.raw.as_ptr()) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn caller_owned_context_supports_independent_second_context() {
        let first = GgmlCallerOwnedContext::new(64 * 1024).unwrap();
        let second = GgmlCallerOwnedContext::new(64 * 1024).unwrap();
        assert_ne!(first.raw(), second.raw());
    }

    #[test]
    fn qwen_sized_pool_failure_is_typed_without_attempting_real_oom() {
        let mut calls = 0;
        let error = GgmlCallerOwnedContext::new_with_allocator(768 * 1024 * 1024, |layout| {
            calls += 1;
            if calls == 1 {
                unsafe { alloc(layout) }
            } else {
                std::ptr::null_mut()
            }
        })
        .unwrap_err();
        assert_eq!(
            error,
            GgmlContextAllocationError::AllocationFailed {
                stage: "pool",
                requested_bytes: 768 * 1024 * 1024,
            }
        );
    }

    #[test]
    fn overflowed_layout_is_typed() {
        let error = AlignedAllocation::new_with_allocator("pool", usize::MAX, |_| {
            panic!("invalid layout must not call allocator")
        })
        .unwrap_err();
        assert_eq!(
            error,
            GgmlContextAllocationError::InvalidLayout {
                stage: "pool",
                requested_bytes: usize::MAX,
            }
        );
    }

    #[test]
    fn zero_sized_pool_has_a_live_context() {
        let context = GgmlCallerOwnedContext::new(0).unwrap();
        let _raw = context.raw();
    }
}
