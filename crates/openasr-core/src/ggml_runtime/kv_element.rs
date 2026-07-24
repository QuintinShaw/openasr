//! Shared KV-cache element types for decoder-only LLM paths.
//!
//! Runtime temporary state only: never part of `.oasr` packs, catalog tags, or
//! model signatures. Host and resident storage share this vocabulary so every
//! Qwen-shaped family (qwen3-asr / mimo / firered2-llm / moss) can opt into the
//! same typed path without copying quant logic.

use std::ffi::c_void;

use super::cpu_graph::GgmlCpuGraphBackend;
use super::ffi;

/// Element type used by host-side and/or device-resident LLM KV caches.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum GgmlKvElementType {
    F32,
    F16,
    Q8_0,
}

impl GgmlKvElementType {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::Q8_0 => "q8_0",
        }
    }

    pub(crate) const fn ggml_type(self) -> i32 {
        match self {
            Self::F32 => ffi::GGML_TYPE_F32,
            Self::F16 => ffi::GGML_TYPE_F16,
            Self::Q8_0 => ffi::GGML_TYPE_Q8_0,
        }
    }

    #[allow(dead_code)]
    pub(crate) const fn is_quantized(self) -> bool {
        matches!(self, Self::Q8_0)
    }

    pub(crate) fn block_size(self) -> usize {
        let blck = unsafe { ffi::ggml_blck_size(self.ggml_type()) };
        usize::try_from(blck).unwrap_or(0)
    }

    #[allow(dead_code)]
    pub(crate) fn type_size(self) -> usize {
        unsafe { ffi::ggml_type_size(self.ggml_type()) }
    }

    /// Bytes for one logical row of `head_dim` values (one KV head at one position).
    pub(crate) fn row_nbytes(self, head_dim: usize) -> Result<usize, String> {
        self.validate_head_dim(head_dim)?;
        let ne0 = i64::try_from(head_dim).map_err(|_| {
            format!(
                "kv element type {}: head_dim={head_dim} does not fit i64",
                self.as_str()
            )
        })?;
        let nbytes = unsafe { ffi::ggml_row_size(self.ggml_type(), ne0) };
        if nbytes == 0 {
            return Err(format!(
                "kv element type {}: ggml_row_size(head_dim={head_dim}) returned 0",
                self.as_str()
            ));
        }
        Ok(nbytes)
    }

    pub(crate) fn plane_nbytes(
        self,
        head_dim: usize,
        max_positions: usize,
        kv_heads: usize,
    ) -> Result<usize, String> {
        let row = self.row_nbytes(head_dim)?;
        row.checked_mul(max_positions)
            .and_then(|n| n.checked_mul(kv_heads))
            .ok_or_else(|| {
                format!(
                    "kv element type {}: plane byte count overflow (head_dim={head_dim}, max_positions={max_positions}, kv_heads={kv_heads})",
                    self.as_str()
                )
            })
    }

    pub(crate) fn validate_head_dim(self, head_dim: usize) -> Result<(), String> {
        if head_dim == 0 {
            return Err(format!(
                "kv element type {}: head_dim must be positive",
                self.as_str()
            ));
        }
        let block = self.block_size();
        if block == 0 {
            return Err(format!(
                "kv element type {}: ggml block size is unavailable",
                self.as_str()
            ));
        }
        if !head_dim.is_multiple_of(block) {
            return Err(format!(
                "kv element type {}: head_dim={head_dim} is not divisible by block_size={block}",
                self.as_str()
            ));
        }
        Ok(())
    }

    /// Phase-1 Q8 is CPU/Metal only. F16/F32 keep the existing backend surface.
    pub(crate) fn supports_backend(self, backend: GgmlCpuGraphBackend) -> bool {
        match self {
            Self::F32 | Self::F16 => true,
            Self::Q8_0 => matches!(
                backend,
                GgmlCpuGraphBackend::Cpu | GgmlCpuGraphBackend::Metal
            ),
        }
    }

    /// Quantize `row_count` contiguous rows of `head_dim` f32 values into the
    /// native ggml layout for this type. F32/F16 are not quantized here.
    pub(crate) fn quantize_rows_from_f32(
        self,
        values: &[f32],
        head_dim: usize,
        row_count: usize,
    ) -> Result<Vec<u8>, String> {
        match self {
            Self::F32 | Self::F16 => Err(format!(
                "kv element type {}: quantize_rows_from_f32 is only valid for quantized types",
                self.as_str()
            )),
            Self::Q8_0 => quantize_q8_0_rows(values, head_dim, row_count),
        }
    }
}

fn quantize_q8_0_rows(
    values: &[f32],
    head_dim: usize,
    row_count: usize,
) -> Result<Vec<u8>, String> {
    GgmlKvElementType::Q8_0.validate_head_dim(head_dim)?;
    if row_count == 0 {
        return Ok(Vec::new());
    }
    let expected_elems = head_dim.checked_mul(row_count).ok_or_else(|| {
        format!("q8_0 quantize element count overflow (head_dim={head_dim}, rows={row_count})")
    })?;
    if values.len() != expected_elems {
        return Err(format!(
            "q8_0 quantize input length mismatch: got {} values, expected {expected_elems} (head_dim={head_dim}, rows={row_count})",
            values.len()
        ));
    }
    if values.iter().any(|v| !v.is_finite()) {
        return Err("q8_0 quantize rejected non-finite values".to_string());
    }
    let row_nbytes = GgmlKvElementType::Q8_0.row_nbytes(head_dim)?;
    let expected_bytes = row_nbytes.checked_mul(row_count).ok_or_else(|| {
        format!("q8_0 quantize byte count overflow (row_nbytes={row_nbytes}, rows={row_count})")
    })?;
    let mut bytes = vec![0_u8; expected_bytes];
    let head_dim_i64 = i64::try_from(head_dim)
        .map_err(|_| format!("q8_0 quantize head_dim={head_dim} does not fit i64"))?;
    let row_count_i64 = i64::try_from(row_count)
        .map_err(|_| format!("q8_0 quantize row_count={row_count} does not fit i64"))?;
    let produced = unsafe {
        ffi::ggml_quantize_chunk(
            ffi::GGML_TYPE_Q8_0,
            values.as_ptr(),
            bytes.as_mut_ptr().cast::<c_void>(),
            0,
            row_count_i64,
            head_dim_i64,
            std::ptr::null(),
        )
    };
    if produced != expected_bytes {
        return Err(format!(
            "q8_0 quantize size mismatch: expected {expected_bytes} bytes, got {produced}"
        ));
    }
    Ok(bytes)
}

/// Dequantize packed q8_0 rows back to f32 for tests and diagnostics.
#[cfg(test)]
pub(crate) fn dequantize_q8_0_rows(
    bytes: &[u8],
    head_dim: usize,
    row_count: usize,
) -> Result<Vec<f32>, String> {
    GgmlKvElementType::Q8_0.validate_head_dim(head_dim)?;
    if row_count == 0 {
        return Ok(Vec::new());
    }
    let row_nbytes = GgmlKvElementType::Q8_0.row_nbytes(head_dim)?;
    let expected_bytes = row_nbytes.checked_mul(row_count).ok_or_else(|| {
        format!("q8_0 dequantize byte count overflow (row_nbytes={row_nbytes}, rows={row_count})")
    })?;
    if bytes.len() != expected_bytes {
        return Err(format!(
            "q8_0 dequantize input length mismatch: got {} bytes, expected {expected_bytes}",
            bytes.len()
        ));
    }
    let traits_ptr = unsafe { ffi::ggml_get_type_traits(ffi::GGML_TYPE_Q8_0) };
    if traits_ptr.is_null() {
        return Err("q8_0 type traits unavailable".to_string());
    }
    let to_float = unsafe { (*traits_ptr).to_float }
        .ok_or_else(|| "q8_0 type traits missing to_float".to_string())?;
    let mut out = vec![0.0_f32; head_dim.saturating_mul(row_count)];
    let head_dim_i64 = i64::try_from(head_dim)
        .map_err(|_| format!("q8_0 dequantize head_dim={head_dim} does not fit i64"))?;
    for row in 0..row_count {
        let src_off = row
            .checked_mul(row_nbytes)
            .ok_or_else(|| "q8_0 dequantize source offset overflow".to_string())?;
        let dst_off = row
            .checked_mul(head_dim)
            .ok_or_else(|| "q8_0 dequantize destination offset overflow".to_string())?;
        unsafe {
            to_float(
                bytes.as_ptr().add(src_off).cast::<c_void>(),
                out.as_mut_ptr().add(dst_off),
                head_dim_i64,
            );
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q8_0_row_layout_matches_ggml_block_geometry() {
        let head_dim = 128;
        assert_eq!(GgmlKvElementType::Q8_0.block_size(), 32);
        assert_eq!(GgmlKvElementType::Q8_0.type_size(), 34);
        assert_eq!(
            GgmlKvElementType::Q8_0.row_nbytes(head_dim).expect("row"),
            136
        );
        assert_eq!(
            GgmlKvElementType::Q8_0
                .plane_nbytes(head_dim, 8, 2)
                .expect("plane"),
            136 * 8 * 2
        );
        assert!(GgmlKvElementType::Q8_0.validate_head_dim(40).is_err());
        assert!(GgmlKvElementType::Q8_0.validate_head_dim(0).is_err());
    }

    #[test]
    fn q8_0_quantize_roundtrips_within_block_scale_tolerance() {
        let head_dim = 64;
        let row_count = 3;
        let mut values = Vec::with_capacity(head_dim * row_count);
        for row in 0..row_count {
            for dim in 0..head_dim {
                values.push((row as f32) * 0.25 + (dim as f32) * 0.01 - 0.5);
            }
        }
        let bytes = GgmlKvElementType::Q8_0
            .quantize_rows_from_f32(&values, head_dim, row_count)
            .expect("quantize");
        let restored = dequantize_q8_0_rows(&bytes, head_dim, row_count).expect("dequant");
        assert_eq!(restored.len(), values.len());
        let mut max_abs = 0.0_f32;
        for (a, b) in values.iter().zip(restored.iter()) {
            max_abs = max_abs.max((a - b).abs());
        }
        // q8_0 is 8-bit with per-32-block scale; unit-scale rows stay well under 0.05.
        assert!(
            max_abs < 0.05,
            "q8_0 roundtrip max abs err {max_abs} exceeded tolerance"
        );
    }

    #[test]
    fn q8_0_backend_surface_is_cpu_and_metal_only() {
        assert!(GgmlKvElementType::Q8_0.supports_backend(GgmlCpuGraphBackend::Cpu));
        assert!(GgmlKvElementType::Q8_0.supports_backend(GgmlCpuGraphBackend::Metal));
        assert!(!GgmlKvElementType::Q8_0.supports_backend(GgmlCpuGraphBackend::Gpu));
        assert!(GgmlKvElementType::F16.supports_backend(GgmlCpuGraphBackend::Gpu));
        assert!(GgmlKvElementType::Q8_0.is_quantized());
        assert!(!GgmlKvElementType::F32.is_quantized());
        assert_eq!(GgmlKvElementType::Q8_0.type_size(), 34);
    }
}
