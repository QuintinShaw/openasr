use std::{
    fs::{self, File},
    path::{Path, PathBuf},
    sync::Arc,
};

use memmap2::Mmap;
use thiserror::Error;

use crate::nn::half::f16_bits_slice_to_f32;

use super::{
    GgmlRuntimeSource, GgmlRuntimeSourcePathError, GgufMetadataReadError, GgufTensorIndex,
    GgufTensorIndexReadError, GgufTensorMetadata, ffi, read_gguf_metadata,
    read_gguf_metadata_from_runtime_source, read_gguf_tensor_index_from_runtime_source,
    validate_ggml_runtime_source_path,
};

const GGUF_DEFAULT_ALIGNMENT_BYTES: u64 = 32;
const GGUF_MIN_ALIGNMENT_BYTES: u64 = 8;
const GGUF_MAX_WEIGHT_TENSOR_RANK: usize = 4;
const GGML_TYPE_F32: i32 = 0;
const GGML_TYPE_F16: i32 = 1;
#[derive(Debug)]
pub struct GgufTensorDataReader {
    tensor_index: Arc<GgufTensorIndex>,
    tensor_data_alignment_bytes: u64,
    mmap: Arc<Mmap>,
}

impl GgufTensorDataReader {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, GgufTensorDataReadError> {
        let runtime_source = validate_ggml_runtime_source_path(path)?;
        Self::from_runtime_source(&runtime_source)
    }

    pub fn from_runtime_source(
        runtime_source: &GgmlRuntimeSource,
    ) -> Result<Self, GgufTensorDataReadError> {
        let tensor_index = read_gguf_tensor_index_from_runtime_source(runtime_source)?;
        let metadata = read_gguf_metadata_from_runtime_source(runtime_source)?;
        let tensor_data_alignment_bytes =
            parse_tensor_alignment(runtime_source.path(), metadata.get_u32("general.alignment"))?;
        Self::from_tensor_index_and_alignment(Arc::new(tensor_index), tensor_data_alignment_bytes)
    }

    pub fn from_tensor_index(
        tensor_index: GgufTensorIndex,
    ) -> Result<Self, GgufTensorDataReadError> {
        Self::from_tensor_index_shared(Arc::new(tensor_index))
    }

    pub fn from_tensor_index_shared(
        tensor_index: Arc<GgufTensorIndex>,
    ) -> Result<Self, GgufTensorDataReadError> {
        let metadata = read_gguf_metadata(tensor_index.path())?;
        let tensor_data_alignment_bytes =
            parse_tensor_alignment(tensor_index.path(), metadata.get_u32("general.alignment"))?;
        Self::from_tensor_index_and_alignment(tensor_index, tensor_data_alignment_bytes)
    }

    pub fn tensor_index(&self) -> &GgufTensorIndex {
        self.tensor_index.as_ref()
    }

    pub fn tensor_data_alignment_bytes(&self) -> u64 {
        self.tensor_data_alignment_bytes
    }

    pub(crate) fn backing_mmap(&self) -> Arc<Mmap> {
        Arc::clone(&self.mmap)
    }

    pub fn host_tensor_bytes_by_name(
        &self,
        tensor_name: &str,
    ) -> Result<GgufHostTensorPayload<'_>, GgufTensorDataReadError> {
        let tensor = self.tensor_index.get(tensor_name).ok_or_else(|| {
            GgufTensorDataReadError::TensorNotFound {
                path: self.tensor_index.path().to_path_buf(),
                tensor_name: tensor_name.to_string(),
            }
        })?;
        self.host_tensor_bytes_internal(tensor)
    }

    pub fn host_tensor_bytes_by_id(
        &self,
        tensor_id: usize,
    ) -> Result<GgufHostTensorPayload<'_>, GgufTensorDataReadError> {
        let tensor = self.tensor_index.tensors().get(tensor_id).ok_or_else(|| {
            GgufTensorDataReadError::TensorIndexOutOfBounds {
                path: self.tensor_index.path().to_path_buf(),
                tensor_id,
                tensor_count: self.tensor_index.tensors().len(),
            }
        })?;
        self.host_tensor_bytes_internal(tensor)
    }

    pub fn host_tensor_bytes_copy_by_name(
        &self,
        tensor_name: &str,
    ) -> Result<Vec<u8>, GgufTensorDataReadError> {
        let payload = self.host_tensor_bytes_by_name(tensor_name)?;
        Ok(payload.bytes.to_vec())
    }

    pub fn host_tensor_f32_copy_by_name(
        &self,
        tensor_name: &str,
        expected_shape: &[u64],
    ) -> Result<Vec<f32>, GgufTensorDataReadError> {
        let payload = self.host_tensor_bytes_by_name(tensor_name)?;
        self.host_tensor_f32_copy_from_payload(payload, expected_shape)
    }

    pub fn host_tensor_f32_copy_by_id(
        &self,
        tensor_id: usize,
        expected_shape: &[u64],
    ) -> Result<Vec<f32>, GgufTensorDataReadError> {
        let payload = self.host_tensor_bytes_by_id(tensor_id)?;
        self.host_tensor_f32_copy_from_payload(payload, expected_shape)
    }

    pub fn host_tensor_f32_copy_dequantized_by_name(
        &self,
        tensor_name: &str,
        expected_shape: &[u64],
    ) -> Result<Vec<f32>, GgufTensorDataReadError> {
        let payload = self.host_tensor_bytes_by_name(tensor_name)?;
        self.host_tensor_f32_copy_dequantized_from_payload(payload, expected_shape)
    }

    pub fn host_tensor_f16_bits_copy_by_name(
        &self,
        tensor_name: &str,
        expected_shape: &[u64],
    ) -> Result<Vec<u16>, GgufTensorDataReadError> {
        let payload = self.host_tensor_bytes_by_name(tensor_name)?;
        self.host_tensor_f16_bits_copy_from_payload(payload, expected_shape)
    }

    pub fn host_tensor_f16_bits_copy_by_id(
        &self,
        tensor_id: usize,
        expected_shape: &[u64],
    ) -> Result<Vec<u16>, GgufTensorDataReadError> {
        let payload = self.host_tensor_bytes_by_id(tensor_id)?;
        self.host_tensor_f16_bits_copy_from_payload(payload, expected_shape)
    }

    pub fn weight_tensor_payload_by_name(
        &self,
        tensor_name: &str,
    ) -> Result<GgufWeightTensorPayload<'_>, GgufTensorDataReadError> {
        let payload = self.host_tensor_bytes_by_name(tensor_name)?;
        self.weight_tensor_payload_from_host(payload)
    }

    pub fn owned_weight_tensor_payload_by_name(
        &self,
        tensor_name: &str,
    ) -> Result<GgufOwnedWeightTensorPayload, GgufTensorDataReadError> {
        let payload = self.host_tensor_bytes_by_name(tensor_name)?;
        self.owned_weight_tensor_payload_from_host(payload)
    }

    pub fn weight_tensor_payload_by_id(
        &self,
        tensor_id: usize,
    ) -> Result<GgufWeightTensorPayload<'_>, GgufTensorDataReadError> {
        let payload = self.host_tensor_bytes_by_id(tensor_id)?;
        self.weight_tensor_payload_from_host(payload)
    }

    fn from_tensor_index_and_alignment(
        tensor_index: Arc<GgufTensorIndex>,
        tensor_data_alignment_bytes: u64,
    ) -> Result<Self, GgufTensorDataReadError> {
        if tensor_data_alignment_bytes == 0
            || !tensor_data_alignment_bytes.is_multiple_of(GGUF_MIN_ALIGNMENT_BYTES)
        {
            return Err(GgufTensorDataReadError::InvalidTensorDataAlignment {
                path: tensor_index.path().to_path_buf(),
                alignment: tensor_data_alignment_bytes,
            });
        }

        let file = File::open(tensor_index.path()).map_err(|source| {
            GgufTensorDataReadError::OpenFile {
                path: tensor_index.path().to_path_buf(),
                source,
            }
        })?;
        let mmap =
            unsafe { Mmap::map(&file) }.map_err(|source| GgufTensorDataReadError::MapFile {
                path: tensor_index.path().to_path_buf(),
                source,
            })?;

        let mapped_len = u64::try_from(mmap.len()).map_err(|_| {
            GgufTensorDataReadError::MappedLengthPlatformOverflow {
                path: tensor_index.path().to_path_buf(),
                length: mmap.len(),
            }
        })?;
        let file_size = fs::metadata(tensor_index.path())
            .map_err(|source| GgufTensorDataReadError::SourceMetadata {
                path: tensor_index.path().to_path_buf(),
                source,
            })?
            .len();
        if mapped_len != file_size {
            return Err(GgufTensorDataReadError::MappedLengthMismatch {
                path: tensor_index.path().to_path_buf(),
                mapped_len,
                file_size,
            });
        }

        Ok(Self {
            tensor_index,
            tensor_data_alignment_bytes,
            mmap: Arc::new(mmap),
        })
    }

    fn host_tensor_bytes_internal<'a>(
        &'a self,
        tensor: &'a GgufTensorMetadata,
    ) -> Result<GgufHostTensorPayload<'a>, GgufTensorDataReadError> {
        let data_section_offset = self.tensor_index.data_section_offset_bytes();
        let relative_offset = tensor
            .offset_bytes
            .checked_sub(data_section_offset)
            .ok_or_else(|| GgufTensorDataReadError::TensorOffsetBeforeDataSection {
                path: self.tensor_index.path().to_path_buf(),
                tensor_name: tensor.name.clone(),
                tensor_offset: tensor.offset_bytes,
                data_section_offset,
            })?;

        if relative_offset % self.tensor_data_alignment_bytes != 0 {
            return Err(GgufTensorDataReadError::TensorOffsetAlignmentViolation {
                path: self.tensor_index.path().to_path_buf(),
                tensor_name: tensor.name.clone(),
                tensor_offset: tensor.offset_bytes,
                data_section_offset,
                alignment: self.tensor_data_alignment_bytes,
            });
        }

        let start = usize::try_from(tensor.offset_bytes).map_err(|_| {
            GgufTensorDataReadError::TensorOffsetPlatformOverflow {
                path: self.tensor_index.path().to_path_buf(),
                tensor_name: tensor.name.clone(),
                offset: tensor.offset_bytes,
            }
        })?;
        let size = usize::try_from(tensor.size_bytes).map_err(|_| {
            GgufTensorDataReadError::TensorSizePlatformOverflow {
                path: self.tensor_index.path().to_path_buf(),
                tensor_name: tensor.name.clone(),
                size_bytes: tensor.size_bytes,
            }
        })?;
        let end = start.checked_add(size).ok_or_else(|| {
            GgufTensorDataReadError::TensorRangeOverflow {
                path: self.tensor_index.path().to_path_buf(),
                tensor_name: tensor.name.clone(),
                offset: tensor.offset_bytes,
                size_bytes: tensor.size_bytes,
            }
        })?;
        if end > self.mmap.len() {
            return Err(GgufTensorDataReadError::TensorRangeOutOfBounds {
                path: self.tensor_index.path().to_path_buf(),
                tensor_name: tensor.name.clone(),
                offset: tensor.offset_bytes,
                size_bytes: tensor.size_bytes,
                file_size: u64::try_from(self.mmap.len()).unwrap_or(u64::MAX),
            });
        }

        Ok(GgufHostTensorPayload {
            metadata: tensor,
            start,
            bytes: &self.mmap[start..end],
        })
    }

    fn host_tensor_f32_copy_from_payload(
        &self,
        payload: GgufHostTensorPayload<'_>,
        expected_shape: &[u64],
    ) -> Result<Vec<f32>, GgufTensorDataReadError> {
        validate_expected_shape(payload.metadata, expected_shape, self.tensor_index.path())?;
        validate_tensor_type(payload.metadata, GGML_TYPE_F32, self.tensor_index.path())?;
        validate_typed_tensor_storage(payload.metadata, 4, self.tensor_index.path())?;

        let num_elements = checked_num_elements(payload.metadata, self.tensor_index.path())?;
        let num_elements_usize = usize::try_from(num_elements).map_err(|_| {
            GgufTensorDataReadError::TensorElementCountPlatformOverflow {
                path: self.tensor_index.path().to_path_buf(),
                tensor_name: payload.metadata.name.clone(),
                num_elements,
            }
        })?;

        if cfg!(target_endian = "little") {
            // GGUF tensor data is little-endian; mmap offsets are normally alignment padded.
            let (prefix, aligned, suffix) = unsafe { payload.bytes.align_to::<f32>() };
            if prefix.is_empty() && suffix.is_empty() && aligned.len() == num_elements_usize {
                return Ok(aligned.to_vec());
            }
        }

        Ok(payload
            .bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect())
    }

    fn host_tensor_f16_bits_copy_from_payload(
        &self,
        payload: GgufHostTensorPayload<'_>,
        expected_shape: &[u64],
    ) -> Result<Vec<u16>, GgufTensorDataReadError> {
        validate_expected_shape(payload.metadata, expected_shape, self.tensor_index.path())?;
        validate_tensor_type(payload.metadata, GGML_TYPE_F16, self.tensor_index.path())?;
        validate_typed_tensor_storage(payload.metadata, 2, self.tensor_index.path())?;

        let num_elements = checked_num_elements(payload.metadata, self.tensor_index.path())?;
        let num_elements_usize = usize::try_from(num_elements).map_err(|_| {
            GgufTensorDataReadError::TensorElementCountPlatformOverflow {
                path: self.tensor_index.path().to_path_buf(),
                tensor_name: payload.metadata.name.clone(),
                num_elements,
            }
        })?;

        if cfg!(target_endian = "little") {
            // F16 is stored as raw little-endian bits; keep it lossless.
            let (prefix, aligned, suffix) = unsafe { payload.bytes.align_to::<u16>() };
            if prefix.is_empty() && suffix.is_empty() && aligned.len() == num_elements_usize {
                return Ok(aligned.to_vec());
            }
        }

        Ok(payload
            .bytes
            .chunks_exact(2)
            .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
            .collect())
    }

    fn host_tensor_f32_copy_dequantized_from_payload(
        &self,
        payload: GgufHostTensorPayload<'_>,
        expected_shape: &[u64],
    ) -> Result<Vec<f32>, GgufTensorDataReadError> {
        validate_expected_shape(payload.metadata, expected_shape, self.tensor_index.path())?;
        match payload.metadata.ggml_type {
            GGML_TYPE_F32 => self.host_tensor_f32_copy_from_payload(payload, expected_shape),
            GGML_TYPE_F16 => {
                let values =
                    self.host_tensor_f16_bits_copy_from_payload(payload, expected_shape)?;
                Ok(f16_bits_slice_to_f32(&values))
            }
            _ => self.host_tensor_quantized_dequantize_to_f32_from_payload(payload),
        }
    }

    fn host_tensor_quantized_dequantize_to_f32_from_payload(
        &self,
        payload: GgufHostTensorPayload<'_>,
    ) -> Result<Vec<f32>, GgufTensorDataReadError> {
        let num_elements = checked_num_elements(payload.metadata, self.tensor_index.path())?;
        let num_elements_usize = usize::try_from(num_elements).map_err(|_| {
            GgufTensorDataReadError::TensorElementCountPlatformOverflow {
                path: self.tensor_index.path().to_path_buf(),
                tensor_name: payload.metadata.name.clone(),
                num_elements,
            }
        })?;
        let ne0 = *payload.metadata.dims.first().ok_or_else(|| {
            GgufTensorDataReadError::TensorRankUnsupportedForWeightMaterialization {
                path: self.tensor_index.path().to_path_buf(),
                tensor_name: payload.metadata.name.clone(),
                rank: 0,
                max_supported_rank: GGUF_MAX_WEIGHT_TENSOR_RANK,
            }
        })?;
        let ne0_i64 =
            i64::try_from(ne0).map_err(|_| GgufTensorDataReadError::TensorDimPlatformOverflow {
                path: self.tensor_index.path().to_path_buf(),
                tensor_name: payload.metadata.name.clone(),
                dim_index: 0,
                dim_value: ne0,
            })?;
        let block_size = unsafe { ffi::ggml_blck_size(payload.metadata.ggml_type) };
        if block_size <= 0 {
            return Err(
                GgufTensorDataReadError::TensorTypeUnsupportedForWeightMaterialization {
                    path: self.tensor_index.path().to_path_buf(),
                    tensor_name: payload.metadata.name.clone(),
                    ggml_type: payload.metadata.ggml_type,
                    type_name: payload.metadata.type_name.clone(),
                },
            );
        }
        let block_size_u64 = u64::try_from(block_size).map_err(|_| {
            GgufTensorDataReadError::TensorTypeUnsupportedForWeightMaterialization {
                path: self.tensor_index.path().to_path_buf(),
                tensor_name: payload.metadata.name.clone(),
                ggml_type: payload.metadata.ggml_type,
                type_name: payload.metadata.type_name.clone(),
            }
        })?;
        if ne0 % block_size_u64 != 0 {
            return Err(GgufTensorDataReadError::TensorStorageWidthMismatch {
                path: self.tensor_index.path().to_path_buf(),
                tensor_name: payload.metadata.name.clone(),
                expected_bytes: block_size_u64,
                actual_bytes: ne0,
            });
        }
        let row_size = unsafe { ffi::ggml_row_size(payload.metadata.ggml_type, ne0_i64) };
        let rows = payload
            .metadata
            .dims
            .iter()
            .skip(1)
            .try_fold(1_u64, |acc, dim| acc.checked_mul(*dim))
            .ok_or_else(|| GgufTensorDataReadError::TensorElementCountOverflow {
                path: self.tensor_index.path().to_path_buf(),
                tensor_name: payload.metadata.name.clone(),
                dims: payload.metadata.dims.clone(),
            })?;
        let expected_bytes_u64 = (row_size as u64).checked_mul(rows).ok_or_else(|| {
            GgufTensorDataReadError::TensorStorageWidthOverflow {
                path: self.tensor_index.path().to_path_buf(),
                tensor_name: payload.metadata.name.clone(),
                num_elements,
                element_size_bytes: row_size as u64,
            }
        })?;
        let actual_bytes_u64 = u64::try_from(payload.bytes.len()).map_err(|_| {
            GgufTensorDataReadError::TensorPayloadLengthPlatformOverflow {
                path: self.tensor_index.path().to_path_buf(),
                tensor_name: payload.metadata.name.clone(),
                payload_len: payload.bytes.len(),
            }
        })?;
        if expected_bytes_u64 != actual_bytes_u64 {
            return Err(GgufTensorDataReadError::TensorPayloadLengthMismatch {
                path: self.tensor_index.path().to_path_buf(),
                tensor_name: payload.metadata.name.clone(),
                expected_bytes: expected_bytes_u64,
                actual_bytes: actual_bytes_u64,
            });
        }

        let traits_ptr = unsafe { ffi::ggml_get_type_traits(payload.metadata.ggml_type) };
        if traits_ptr.is_null() {
            return Err(
                GgufTensorDataReadError::TensorTypeUnsupportedForWeightMaterialization {
                    path: self.tensor_index.path().to_path_buf(),
                    tensor_name: payload.metadata.name.clone(),
                    ggml_type: payload.metadata.ggml_type,
                    type_name: payload.metadata.type_name.clone(),
                },
            );
        }
        let to_float = unsafe { (*traits_ptr).to_float }.ok_or_else(|| {
            GgufTensorDataReadError::TensorTypeUnsupportedForWeightMaterialization {
                path: self.tensor_index.path().to_path_buf(),
                tensor_name: payload.metadata.name.clone(),
                ggml_type: payload.metadata.ggml_type,
                type_name: payload.metadata.type_name.clone(),
            }
        })?;

        let rows_usize = usize::try_from(rows).map_err(|_| {
            GgufTensorDataReadError::TensorElementCountPlatformOverflow {
                path: self.tensor_index.path().to_path_buf(),
                tensor_name: payload.metadata.name.clone(),
                num_elements: rows,
            }
        })?;
        let ne0_usize = usize::try_from(ne0).map_err(|_| {
            GgufTensorDataReadError::TensorDimPlatformOverflow {
                path: self.tensor_index.path().to_path_buf(),
                tensor_name: payload.metadata.name.clone(),
                dim_index: 0,
                dim_value: ne0,
            }
        })?;
        let mut values = vec![0.0_f32; num_elements_usize];
        for row_idx in 0..rows_usize {
            let src_offset = row_idx * row_size;
            let src_ptr = payload.bytes[src_offset..]
                .as_ptr()
                .cast::<std::ffi::c_void>();
            let dst_ptr = values[row_idx * ne0_usize..].as_mut_ptr();
            unsafe {
                to_float(src_ptr, dst_ptr, ne0_i64);
            }
        }
        Ok(values)
    }

    fn weight_tensor_payload_from_host<'a>(
        &self,
        payload: GgufHostTensorPayload<'a>,
    ) -> Result<GgufWeightTensorPayload<'a>, GgufTensorDataReadError> {
        let (element_type, element_size_bytes) = match payload.metadata.ggml_type {
            GGML_TYPE_F32 => (GgufWeightTensorElementType::F32, 4_u64),
            GGML_TYPE_F16 => (GgufWeightTensorElementType::F16, 2_u64),
            ggml_type if unsafe { ffi::ggml_is_quantized(ggml_type) } => {
                (GgufWeightTensorElementType::RawGgml { ggml_type }, 0_u64)
            }
            _ => {
                return Err(
                    GgufTensorDataReadError::TensorTypeUnsupportedForWeightMaterialization {
                        path: self.tensor_index.path().to_path_buf(),
                        tensor_name: payload.metadata.name.clone(),
                        ggml_type: payload.metadata.ggml_type,
                        type_name: payload.metadata.type_name.clone(),
                    },
                );
            }
        };

        let rank = payload.metadata.rank();
        if rank == 0 || rank > GGUF_MAX_WEIGHT_TENSOR_RANK {
            return Err(
                GgufTensorDataReadError::TensorRankUnsupportedForWeightMaterialization {
                    path: self.tensor_index.path().to_path_buf(),
                    tensor_name: payload.metadata.name.clone(),
                    rank,
                    max_supported_rank: GGUF_MAX_WEIGHT_TENSOR_RANK,
                },
            );
        }

        let mut dims = Vec::with_capacity(rank);
        for (dim_index, dim_value) in payload.metadata.dims.iter().enumerate() {
            let dim_value_usize = usize::try_from(*dim_value).map_err(|_| {
                GgufTensorDataReadError::TensorDimPlatformOverflow {
                    path: self.tensor_index.path().to_path_buf(),
                    tensor_name: payload.metadata.name.clone(),
                    dim_index,
                    dim_value: *dim_value,
                }
            })?;
            if dim_value_usize == 0 {
                return Err(
                    GgufTensorDataReadError::TensorRankUnsupportedForWeightMaterialization {
                        path: self.tensor_index.path().to_path_buf(),
                        tensor_name: payload.metadata.name.clone(),
                        rank,
                        max_supported_rank: GGUF_MAX_WEIGHT_TENSOR_RANK,
                    },
                );
            }
            dims.push(dim_value_usize);
        }

        let num_elements_u64 = checked_num_elements(payload.metadata, self.tensor_index.path())?;
        let num_elements = usize::try_from(num_elements_u64).map_err(|_| {
            GgufTensorDataReadError::TensorElementCountPlatformOverflow {
                path: self.tensor_index.path().to_path_buf(),
                tensor_name: payload.metadata.name.clone(),
                num_elements: num_elements_u64,
            }
        })?;

        let expected_len_u64 = match element_type {
            GgufWeightTensorElementType::F32 | GgufWeightTensorElementType::F16 => {
                validate_typed_tensor_storage(
                    payload.metadata,
                    element_size_bytes,
                    self.tensor_index.path(),
                )?;
                num_elements_u64
                    .checked_mul(element_size_bytes)
                    .ok_or_else(|| GgufTensorDataReadError::TensorStorageWidthOverflow {
                        path: self.tensor_index.path().to_path_buf(),
                        tensor_name: payload.metadata.name.clone(),
                        num_elements: num_elements_u64,
                        element_size_bytes,
                    })?
            }
            GgufWeightTensorElementType::RawGgml { ggml_type } => {
                checked_row_major_ggml_tensor_bytes(
                    payload.metadata,
                    ggml_type,
                    self.tensor_index.path(),
                )?
            }
        };
        let actual_len_u64 = u64::try_from(payload.bytes.len()).map_err(|_| {
            GgufTensorDataReadError::TensorPayloadLengthPlatformOverflow {
                path: self.tensor_index.path().to_path_buf(),
                tensor_name: payload.metadata.name.clone(),
                payload_len: payload.bytes.len(),
            }
        })?;
        if expected_len_u64 != actual_len_u64 {
            return Err(GgufTensorDataReadError::TensorPayloadLengthMismatch {
                path: self.tensor_index.path().to_path_buf(),
                tensor_name: payload.metadata.name.clone(),
                expected_bytes: expected_len_u64,
                actual_bytes: actual_len_u64,
            });
        }

        Ok(GgufWeightTensorPayload {
            metadata: payload.metadata,
            bytes: payload.bytes,
            dims,
            num_elements,
            element_type,
        })
    }

    fn owned_weight_tensor_payload_from_host(
        &self,
        payload: GgufHostTensorPayload<'_>,
    ) -> Result<GgufOwnedWeightTensorPayload, GgufTensorDataReadError> {
        let borrowed = self.weight_tensor_payload_from_host(payload)?;
        Ok(GgufOwnedWeightTensorPayload {
            metadata: borrowed.metadata.clone(),
            dims: borrowed.dims.clone(),
            num_elements: borrowed.num_elements,
            element_type: borrowed.element_type,
            mmap: Arc::clone(&self.mmap),
            start: payload.start,
            len: borrowed.bytes.len(),
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct GgufHostTensorPayload<'a> {
    pub metadata: &'a GgufTensorMetadata,
    pub start: usize,
    pub bytes: &'a [u8],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufWeightTensorElementType {
    F32,
    F16,
    RawGgml { ggml_type: i32 },
}

impl GgufWeightTensorElementType {
    pub fn ggml_type(self) -> i32 {
        match self {
            Self::F32 => GGML_TYPE_F32,
            Self::F16 => GGML_TYPE_F16,
            Self::RawGgml { ggml_type } => ggml_type,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GgufWeightTensorPayload<'a> {
    pub metadata: &'a GgufTensorMetadata,
    pub bytes: &'a [u8],
    pub dims: Vec<usize>,
    pub num_elements: usize,
    pub element_type: GgufWeightTensorElementType,
}

#[derive(Debug, Clone)]
pub struct GgufOwnedWeightTensorPayload {
    pub metadata: GgufTensorMetadata,
    pub dims: Vec<usize>,
    pub num_elements: usize,
    pub element_type: GgufWeightTensorElementType,
    mmap: Arc<Mmap>,
    start: usize,
    len: usize,
}

impl GgufOwnedWeightTensorPayload {
    pub fn bytes(&self) -> &[u8] {
        &self.mmap[self.start..self.start + self.len]
    }
}

#[derive(Debug, Error)]
pub enum GgufTensorDataReadError {
    #[error(transparent)]
    InvalidRuntimeSource(#[from] GgmlRuntimeSourcePathError),
    #[error(transparent)]
    TensorIndexRead(#[from] GgufTensorIndexReadError),
    #[error(transparent)]
    MetadataRead(#[from] GgufMetadataReadError),
    #[error("could not inspect gguf runtime source metadata for '{path}': {source}")]
    SourceMetadata {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error(
        "could not open gguf runtime source file '{path}' for tensor materialization: {source}"
    )]
    OpenFile {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error(
        "could not memory-map gguf runtime source file '{path}' for tensor materialization: {source}"
    )]
    MapFile {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("mapped file length does not fit in u64 for '{path}': length={length}")]
    MappedLengthPlatformOverflow { path: PathBuf, length: usize },
    #[error(
        "mapped file length mismatch for '{path}': mapped_len={mapped_len}, file_size={file_size}"
    )]
    MappedLengthMismatch {
        path: PathBuf,
        mapped_len: u64,
        file_size: u64,
    },
    #[error("gguf tensor data alignment is invalid for '{path}': alignment={alignment}")]
    InvalidTensorDataAlignment { path: PathBuf, alignment: u64 },
    #[error("gguf tensor '{tensor_name}' not found in '{path}'")]
    TensorNotFound { path: PathBuf, tensor_name: String },
    #[error(
        "gguf tensor index out of bounds in '{path}': tensor_id={tensor_id}, tensor_count={tensor_count}"
    )]
    TensorIndexOutOfBounds {
        path: PathBuf,
        tensor_id: usize,
        tensor_count: usize,
    },
    #[error(
        "gguf tensor '{tensor_name}' in '{path}' has offset before data section: tensor_offset={tensor_offset}, data_section_offset={data_section_offset}"
    )]
    TensorOffsetBeforeDataSection {
        path: PathBuf,
        tensor_name: String,
        tensor_offset: u64,
        data_section_offset: u64,
    },
    #[error(
        "gguf tensor '{tensor_name}' in '{path}' violates tensor-data alignment: tensor_offset={tensor_offset}, data_section_offset={data_section_offset}, alignment={alignment}"
    )]
    TensorOffsetAlignmentViolation {
        path: PathBuf,
        tensor_name: String,
        tensor_offset: u64,
        data_section_offset: u64,
        alignment: u64,
    },
    #[error("gguf tensor '{tensor_name}' offset does not fit usize in '{path}': offset={offset}")]
    TensorOffsetPlatformOverflow {
        path: PathBuf,
        tensor_name: String,
        offset: u64,
    },
    #[error(
        "gguf tensor '{tensor_name}' size does not fit usize in '{path}': size_bytes={size_bytes}"
    )]
    TensorSizePlatformOverflow {
        path: PathBuf,
        tensor_name: String,
        size_bytes: u64,
    },
    #[error(
        "gguf tensor '{tensor_name}' in '{path}' has range overflow: offset={offset}, size_bytes={size_bytes}"
    )]
    TensorRangeOverflow {
        path: PathBuf,
        tensor_name: String,
        offset: u64,
        size_bytes: u64,
    },
    #[error(
        "gguf tensor '{tensor_name}' in '{path}' exceeds mapped file bounds: offset={offset}, size_bytes={size_bytes}, file_size={file_size}"
    )]
    TensorRangeOutOfBounds {
        path: PathBuf,
        tensor_name: String,
        offset: u64,
        size_bytes: u64,
        file_size: u64,
    },
    #[error(
        "gguf tensor '{tensor_name}' in '{path}' has shape mismatch: expected={expected:?}, actual={actual:?}"
    )]
    TensorShapeMismatch {
        path: PathBuf,
        tensor_name: String,
        expected: Vec<u64>,
        actual: Vec<u64>,
    },
    #[error(
        "gguf tensor '{tensor_name}' in '{path}' has type mismatch: expected={expected}, actual={actual} ({type_name})"
    )]
    TensorTypeMismatch {
        path: PathBuf,
        tensor_name: String,
        expected: i32,
        actual: i32,
        type_name: String,
    },
    #[error("gguf tensor '{tensor_name}' in '{path}' has element-count overflow for dims {dims:?}")]
    TensorElementCountOverflow {
        path: PathBuf,
        tensor_name: String,
        dims: Vec<u64>,
    },
    #[error(
        "gguf tensor '{tensor_name}' element count does not fit usize in '{path}': num_elements={num_elements}"
    )]
    TensorElementCountPlatformOverflow {
        path: PathBuf,
        tensor_name: String,
        num_elements: u64,
    },
    #[error(
        "gguf tensor '{tensor_name}' in '{path}' has storage-width mismatch: expected={expected_bytes}, actual={actual_bytes}"
    )]
    TensorStorageWidthMismatch {
        path: PathBuf,
        tensor_name: String,
        expected_bytes: u64,
        actual_bytes: u64,
    },
    #[error(
        "gguf tensor '{tensor_name}' in '{path}' has storage-width overflow: num_elements={num_elements}, element_size_bytes={element_size_bytes}"
    )]
    TensorStorageWidthOverflow {
        path: PathBuf,
        tensor_name: String,
        num_elements: u64,
        element_size_bytes: u64,
    },
    #[error(
        "gguf tensor '{tensor_name}' in '{path}' has offset not aligned to element width {element_size_bytes}: offset={offset}"
    )]
    TensorElementOffsetMisaligned {
        path: PathBuf,
        tensor_name: String,
        element_size_bytes: u64,
        offset: u64,
    },
    #[error(
        "gguf tensor '{tensor_name}' in '{path}' has size not aligned to element width {element_size_bytes}: size_bytes={size_bytes}"
    )]
    TensorElementSizeMisaligned {
        path: PathBuf,
        tensor_name: String,
        element_size_bytes: u64,
        size_bytes: u64,
    },
    #[error(
        "gguf tensor '{tensor_name}' in '{path}' uses unsupported type for weight materialization: ggml_type={ggml_type} ({type_name})"
    )]
    TensorTypeUnsupportedForWeightMaterialization {
        path: PathBuf,
        tensor_name: String,
        ggml_type: i32,
        type_name: String,
    },
    #[error(
        "gguf tensor '{tensor_name}' in '{path}' uses quantized type without row traits for raw weight materialization: ggml_type={ggml_type} ({type_name})"
    )]
    QuantizedTensorMissingRowTraits {
        path: PathBuf,
        tensor_name: String,
        ggml_type: i32,
        type_name: String,
    },
    #[error(
        "gguf tensor '{tensor_name}' in '{path}' has row width not aligned to quant block size: ggml_type={ggml_type} ({type_name}), block_size={block_size}, ne0={ne0}"
    )]
    QuantizedTensorRowWidthNotBlockAligned {
        path: PathBuf,
        tensor_name: String,
        ggml_type: i32,
        type_name: String,
        block_size: u64,
        ne0: u64,
    },
    #[error(
        "gguf tensor '{tensor_name}' in '{path}' has unsupported rank for weight materialization: rank={rank}, max_supported_rank={max_supported_rank}"
    )]
    TensorRankUnsupportedForWeightMaterialization {
        path: PathBuf,
        tensor_name: String,
        rank: usize,
        max_supported_rank: usize,
    },
    #[error(
        "gguf tensor '{tensor_name}' in '{path}' has dim that does not fit usize: dim_index={dim_index}, value={dim_value}"
    )]
    TensorDimPlatformOverflow {
        path: PathBuf,
        tensor_name: String,
        dim_index: usize,
        dim_value: u64,
    },
    #[error(
        "gguf tensor '{tensor_name}' payload length does not fit u64 in '{path}': payload_len={payload_len}"
    )]
    TensorPayloadLengthPlatformOverflow {
        path: PathBuf,
        tensor_name: String,
        payload_len: usize,
    },
    #[error(
        "gguf tensor '{tensor_name}' payload length mismatch in '{path}': expected_bytes={expected_bytes}, actual_bytes={actual_bytes}"
    )]
    TensorPayloadLengthMismatch {
        path: PathBuf,
        tensor_name: String,
        expected_bytes: u64,
        actual_bytes: u64,
    },
}

fn parse_tensor_alignment(
    path: &Path,
    alignment: Option<u32>,
) -> Result<u64, GgufTensorDataReadError> {
    let alignment = alignment
        .map(u64::from)
        .unwrap_or(GGUF_DEFAULT_ALIGNMENT_BYTES);
    if alignment == 0 || !alignment.is_multiple_of(GGUF_MIN_ALIGNMENT_BYTES) {
        return Err(GgufTensorDataReadError::InvalidTensorDataAlignment {
            path: path.to_path_buf(),
            alignment,
        });
    }
    Ok(alignment)
}

fn validate_expected_shape(
    tensor: &GgufTensorMetadata,
    expected_shape: &[u64],
    path: &Path,
) -> Result<(), GgufTensorDataReadError> {
    if tensor.has_shape(expected_shape) {
        return Ok(());
    }
    Err(GgufTensorDataReadError::TensorShapeMismatch {
        path: path.to_path_buf(),
        tensor_name: tensor.name.clone(),
        expected: expected_shape.to_vec(),
        actual: tensor.dims.clone(),
    })
}

fn validate_tensor_type(
    tensor: &GgufTensorMetadata,
    expected_type: i32,
    path: &Path,
) -> Result<(), GgufTensorDataReadError> {
    if tensor.ggml_type == expected_type {
        return Ok(());
    }
    Err(GgufTensorDataReadError::TensorTypeMismatch {
        path: path.to_path_buf(),
        tensor_name: tensor.name.clone(),
        expected: expected_type,
        actual: tensor.ggml_type,
        type_name: tensor.type_name.clone(),
    })
}

fn checked_num_elements(
    tensor: &GgufTensorMetadata,
    path: &Path,
) -> Result<u64, GgufTensorDataReadError> {
    tensor
        .num_elements()
        .ok_or_else(|| GgufTensorDataReadError::TensorElementCountOverflow {
            path: path.to_path_buf(),
            tensor_name: tensor.name.clone(),
            dims: tensor.dims.clone(),
        })
}

fn validate_typed_tensor_storage(
    tensor: &GgufTensorMetadata,
    element_size_bytes: u64,
    path: &Path,
) -> Result<(), GgufTensorDataReadError> {
    if !tensor.offset_bytes.is_multiple_of(element_size_bytes) {
        return Err(GgufTensorDataReadError::TensorElementOffsetMisaligned {
            path: path.to_path_buf(),
            tensor_name: tensor.name.clone(),
            element_size_bytes,
            offset: tensor.offset_bytes,
        });
    }
    if !tensor.size_bytes.is_multiple_of(element_size_bytes) {
        return Err(GgufTensorDataReadError::TensorElementSizeMisaligned {
            path: path.to_path_buf(),
            tensor_name: tensor.name.clone(),
            element_size_bytes,
            size_bytes: tensor.size_bytes,
        });
    }

    let num_elements = checked_num_elements(tensor, path)?;
    let expected_bytes = num_elements
        .checked_mul(element_size_bytes)
        .ok_or_else(|| GgufTensorDataReadError::TensorStorageWidthOverflow {
            path: path.to_path_buf(),
            tensor_name: tensor.name.clone(),
            num_elements,
            element_size_bytes,
        })?;
    if expected_bytes != tensor.size_bytes {
        return Err(GgufTensorDataReadError::TensorStorageWidthMismatch {
            path: path.to_path_buf(),
            tensor_name: tensor.name.clone(),
            expected_bytes,
            actual_bytes: tensor.size_bytes,
        });
    }

    Ok(())
}

fn checked_row_major_ggml_tensor_bytes(
    tensor: &GgufTensorMetadata,
    ggml_type: i32,
    path: &Path,
) -> Result<u64, GgufTensorDataReadError> {
    let ne0 = *tensor.dims.first().ok_or_else(|| {
        GgufTensorDataReadError::TensorRankUnsupportedForWeightMaterialization {
            path: path.to_path_buf(),
            tensor_name: tensor.name.clone(),
            rank: tensor.rank(),
            max_supported_rank: GGUF_MAX_WEIGHT_TENSOR_RANK,
        }
    })?;
    let ne0_i64 =
        i64::try_from(ne0).map_err(|_| GgufTensorDataReadError::TensorElementCountOverflow {
            path: path.to_path_buf(),
            tensor_name: tensor.name.clone(),
            dims: tensor.dims.clone(),
        })?;
    let block_size = unsafe { ffi::ggml_blck_size(ggml_type) };
    if block_size <= 0 {
        return Err(GgufTensorDataReadError::QuantizedTensorMissingRowTraits {
            path: path.to_path_buf(),
            tensor_name: tensor.name.clone(),
            ggml_type,
            type_name: tensor.type_name.clone(),
        });
    }
    let block_size_u64 = u64::try_from(block_size).map_err(|_| {
        GgufTensorDataReadError::TensorElementCountOverflow {
            path: path.to_path_buf(),
            tensor_name: tensor.name.clone(),
            dims: tensor.dims.clone(),
        }
    })?;
    if ne0 % block_size_u64 != 0 {
        return Err(
            GgufTensorDataReadError::QuantizedTensorRowWidthNotBlockAligned {
                path: path.to_path_buf(),
                tensor_name: tensor.name.clone(),
                ggml_type,
                type_name: tensor.type_name.clone(),
                block_size: block_size_u64,
                ne0,
            },
        );
    }

    let row_size = unsafe { ffi::ggml_row_size(ggml_type, ne0_i64) };
    let rows = tensor
        .dims
        .iter()
        .skip(1)
        .try_fold(1_u64, |acc, dim| acc.checked_mul(*dim))
        .ok_or_else(|| GgufTensorDataReadError::TensorStorageWidthOverflow {
            path: path.to_path_buf(),
            tensor_name: tensor.name.clone(),
            num_elements: tensor.num_elements().unwrap_or(u64::MAX),
            element_size_bytes: row_size as u64,
        })?;
    (row_size as u64).checked_mul(rows).ok_or_else(|| {
        GgufTensorDataReadError::TensorStorageWidthOverflow {
            path: path.to_path_buf(),
            tensor_name: tensor.name.clone(),
            num_elements: rows,
            element_size_bytes: row_size as u64,
        }
    })
}

#[cfg(test)]
mod tests {
    use std::{fs, path::Path};

    use tempfile::NamedTempFile;

    use super::{GgufTensorDataReadError, GgufTensorDataReader, GgufWeightTensorElementType};

    const GGUF_VERSION_V3: u32 = 3;
    const GGUF_TYPE_UINT32: u32 = 4;
    const GGML_TYPE_F32: i32 = 0;
    const GGML_TYPE_F16: i32 = 1;
    const GGML_TYPE_Q8_0: i32 = 8;

    struct TensorFixture<'a> {
        name: &'a str,
        dims: Vec<u64>,
        ggml_type: i32,
        payload: Vec<u8>,
        offset_override: Option<u64>,
    }

    fn push_u32(bytes: &mut Vec<u8>, value: u32) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn push_i32(bytes: &mut Vec<u8>, value: i32) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn push_u64(bytes: &mut Vec<u8>, value: u64) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn push_gguf_string(bytes: &mut Vec<u8>, value: &str) {
        push_u64(bytes, value.len() as u64);
        bytes.extend_from_slice(value.as_bytes());
    }

    fn align_up_u64(value: u64, alignment: u64) -> u64 {
        debug_assert!(alignment > 0);
        (value + alignment - 1) & !(alignment - 1)
    }

    fn write_fixture(path: &Path, alignment: u32, tensors: &[TensorFixture<'_>]) {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        push_u32(&mut bytes, GGUF_VERSION_V3);
        push_u64(&mut bytes, tensors.len() as u64);
        push_u64(&mut bytes, 1); // n_kv

        push_gguf_string(&mut bytes, "general.alignment");
        push_u32(&mut bytes, GGUF_TYPE_UINT32);
        push_u32(&mut bytes, alignment);

        let mut running_offset = 0_u64;
        for tensor in tensors {
            let offset = tensor.offset_override.unwrap_or(running_offset);
            push_gguf_string(&mut bytes, tensor.name);
            push_u32(&mut bytes, tensor.dims.len() as u32);
            for dim in &tensor.dims {
                push_u64(&mut bytes, *dim);
            }
            push_i32(&mut bytes, tensor.ggml_type);
            push_u64(&mut bytes, offset);

            if tensor.offset_override.is_none() {
                running_offset =
                    align_up_u64(offset + tensor.payload.len() as u64, alignment as u64);
            }
        }

        let data_section_offset = align_up_u64(bytes.len() as u64, alignment as u64);
        bytes.resize(data_section_offset as usize, 0);

        let mut data_blob_size = 0_u64;
        for tensor in tensors {
            let offset = tensor
                .offset_override
                .unwrap_or_else(|| align_up_u64(data_blob_size, alignment as u64));
            let end = offset + tensor.payload.len() as u64;
            data_blob_size = data_blob_size.max(end);
        }
        let data_blob_size = align_up_u64(data_blob_size, alignment as u64);
        let mut data_blob = vec![0_u8; usize::try_from(data_blob_size).expect("blob size")];
        let mut implicit_cursor = 0_u64;
        for tensor in tensors {
            let offset = tensor.offset_override.unwrap_or_else(|| {
                let aligned = align_up_u64(implicit_cursor, alignment as u64);
                implicit_cursor = aligned + tensor.payload.len() as u64;
                aligned
            });
            let start = usize::try_from(offset).expect("tensor offset");
            let end = start + tensor.payload.len();
            data_blob[start..end].copy_from_slice(&tensor.payload);
        }

        bytes.extend_from_slice(&data_blob);
        fs::write(path, bytes).expect("write gguf fixture");
    }

    #[test]
    fn materializes_f32_and_f16_payloads() {
        let file = NamedTempFile::new().expect("temp file");
        let f32_values = [1.0_f32, -2.5_f32];
        let mut f32_bytes = Vec::new();
        for value in f32_values {
            f32_bytes.extend_from_slice(&value.to_le_bytes());
        }
        let mut f16_bytes = Vec::new();
        f16_bytes.extend_from_slice(&0x3c00_u16.to_le_bytes());
        f16_bytes.extend_from_slice(&0x3800_u16.to_le_bytes());

        write_fixture(
            file.path(),
            32,
            &[
                TensorFixture {
                    name: "encoder.f32",
                    dims: vec![2],
                    ggml_type: GGML_TYPE_F32,
                    payload: f32_bytes,
                    offset_override: None,
                },
                TensorFixture {
                    name: "encoder.f16",
                    dims: vec![2],
                    ggml_type: GGML_TYPE_F16,
                    payload: f16_bytes,
                    offset_override: None,
                },
            ],
        );

        let reader = GgufTensorDataReader::from_path(file.path()).expect("create tensor reader");
        let f32 = reader
            .host_tensor_f32_copy_by_name("encoder.f32", &[2])
            .expect("materialize f32");
        assert_eq!(f32, vec![1.0, -2.5]);

        let f16 = reader
            .host_tensor_f16_bits_copy_by_name("encoder.f16", &[2])
            .expect("materialize f16");
        assert_eq!(f16, vec![0x3c00, 0x3800]);

        let f32_by_id = reader
            .host_tensor_f32_copy_by_id(0, &[2])
            .expect("materialize f32 by id");
        assert_eq!(f32_by_id, vec![1.0, -2.5]);

        let f16_by_id = reader
            .host_tensor_f16_bits_copy_by_id(1, &[2])
            .expect("materialize f16 by id");
        assert_eq!(f16_by_id, vec![0x3c00, 0x3800]);

        let f32_payload = reader
            .weight_tensor_payload_by_name("encoder.f32")
            .expect("materialize f32 weight payload");
        assert_eq!(f32_payload.dims, vec![2]);
        assert_eq!(f32_payload.num_elements, 2);
        assert_eq!(f32_payload.element_type, GgufWeightTensorElementType::F32);

        let f16_payload = reader
            .weight_tensor_payload_by_id(1)
            .expect("materialize f16 weight payload");
        assert_eq!(f16_payload.dims, vec![2]);
        assert_eq!(f16_payload.num_elements, 2);
        assert_eq!(f16_payload.element_type, GgufWeightTensorElementType::F16);

        let bytes = reader
            .host_tensor_bytes_copy_by_name("encoder.f32")
            .expect("materialize bytes");
        assert_eq!(bytes.len(), 8);
    }

    #[test]
    fn materializes_quantized_weight_payload_without_dequantizing() {
        let file = NamedTempFile::new().expect("temp file");
        let q8_row = vec![0_u8; 34];
        write_fixture(
            file.path(),
            32,
            &[TensorFixture {
                name: "llm.q8",
                dims: vec![32, 1],
                ggml_type: GGML_TYPE_Q8_0,
                payload: q8_row.clone(),
                offset_override: None,
            }],
        );

        let reader = GgufTensorDataReader::from_path(file.path()).expect("create tensor reader");
        let payload = reader
            .weight_tensor_payload_by_name("llm.q8")
            .expect("materialize q8 payload");
        assert_eq!(payload.dims, vec![32]);
        assert_eq!(payload.num_elements, 32);
        assert_eq!(
            payload.element_type,
            GgufWeightTensorElementType::RawGgml {
                ggml_type: GGML_TYPE_Q8_0
            }
        );
        assert_eq!(payload.bytes, q8_row.as_slice());
    }

    #[test]
    fn fails_closed_on_shape_mismatch() {
        let file = NamedTempFile::new().expect("temp file");
        let mut f32_bytes = Vec::new();
        f32_bytes.extend_from_slice(&1.0_f32.to_le_bytes());
        f32_bytes.extend_from_slice(&2.0_f32.to_le_bytes());

        write_fixture(
            file.path(),
            32,
            &[TensorFixture {
                name: "encoder.weight",
                dims: vec![2],
                ggml_type: GGML_TYPE_F32,
                payload: f32_bytes,
                offset_override: None,
            }],
        );

        let reader = GgufTensorDataReader::from_path(file.path()).expect("create tensor reader");
        let error = reader
            .host_tensor_f32_copy_by_name("encoder.weight", &[1, 2])
            .expect_err("shape mismatch must fail");
        assert!(matches!(
            error,
            GgufTensorDataReadError::TensorShapeMismatch { .. }
        ));
    }

    #[test]
    fn fails_closed_on_type_mismatch() {
        let file = NamedTempFile::new().expect("temp file");
        let mut f16_bytes = Vec::new();
        f16_bytes.extend_from_slice(&0x3c00_u16.to_le_bytes());
        f16_bytes.extend_from_slice(&0x3800_u16.to_le_bytes());

        write_fixture(
            file.path(),
            32,
            &[TensorFixture {
                name: "encoder.weight",
                dims: vec![2],
                ggml_type: GGML_TYPE_F16,
                payload: f16_bytes,
                offset_override: None,
            }],
        );

        let reader = GgufTensorDataReader::from_path(file.path()).expect("create tensor reader");
        let error = reader
            .host_tensor_f32_copy_by_name("encoder.weight", &[2])
            .expect_err("type mismatch must fail");
        assert!(matches!(
            error,
            GgufTensorDataReadError::TensorTypeMismatch { .. }
        ));
    }

    #[test]
    fn fails_closed_on_alignment_invalid_offset() {
        let file = NamedTempFile::new().expect("temp file");
        let mut f32_bytes = Vec::new();
        f32_bytes.extend_from_slice(&1.0_f32.to_le_bytes());
        f32_bytes.extend_from_slice(&2.0_f32.to_le_bytes());

        write_fixture(
            file.path(),
            32,
            &[TensorFixture {
                name: "encoder.weight",
                dims: vec![2],
                ggml_type: GGML_TYPE_F32,
                payload: f32_bytes,
                offset_override: Some(4),
            }],
        );

        let error = GgufTensorDataReader::from_path(file.path())
            .expect_err("misaligned tensor offset must fail during tensor-index read");
        assert!(matches!(error, GgufTensorDataReadError::TensorIndexRead(_)));
    }
}
