//! Model-semantic persistent-state capacity primitives.
//!
//! Family-specific frontend and decode topology belongs in each family's
//! `capacity` module and is exposed through [`topology::DecoderStateTopology`].
//! This module intentionally contains no host-memory heuristic, frontend
//! registry, execution policy, or device budget: physical commitment is
//! quoted by the selected backend and admitted by `DeviceMemoryBrokerSet`.

use crate::nn::decoder::LlmKvCacheSpec;

pub(crate) mod decode_schedule;
pub(crate) mod topology;

/// Decoder KV-cache geometry read from the loaded model contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct KvGeometry {
    /// Decoder layers, each contributing one K and one V row per position.
    pub n_layers: usize,
    /// KV heads per layer after GQA.
    pub kv_heads: usize,
    /// Values in one KV head row.
    pub head_dim: usize,
}

/// Logical bytes per decoder position for one sequence, split by the storage
/// copies the family runtime actually owns.
///
/// These values describe token-scaled state; they are not a prediction of
/// native physical commitment. Alignment, allocator blocks, imports, backend
/// caches, and transient workspaces are priced by the backend memory ABI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct KvBytesPerPosition {
    pub host: u64,
    pub resident: u64,
}

impl KvBytesPerPosition {
    #[cfg(test)]
    pub(crate) fn total(self) -> u64 {
        self.host.saturating_add(self.resident)
    }
}

/// Exact logical KV payload for one position and one sequence.
pub(crate) fn kv_bytes_per_position(
    geometry: &KvGeometry,
    spec: LlmKvCacheSpec,
) -> Result<KvBytesPerPosition, String> {
    if geometry.n_layers == 0 || geometry.kv_heads == 0 {
        return Err(format!(
            "kv geometry must have positive n_layers and kv_heads (got {geometry:?})"
        ));
    }
    let rows_per_position = geometry
        .n_layers
        .checked_mul(2)
        .and_then(|kv_rows| kv_rows.checked_mul(geometry.kv_heads))
        .ok_or_else(|| format!("kv geometry row count overflowed: {geometry:?}"))?;
    let rows_per_position = u64::try_from(rows_per_position)
        .map_err(|_| format!("kv geometry row count exceeds u64: {geometry:?}"))?;
    let host_row = u64::try_from(spec.host.row_nbytes(geometry.head_dim)?)
        .map_err(|_| "kv host row size exceeds u64".to_owned())?;
    let resident_row = u64::try_from(spec.resident.row_nbytes(geometry.head_dim)?)
        .map_err(|_| "kv resident row size exceeds u64".to_owned())?;
    Ok(KvBytesPerPosition {
        host: host_row
            .checked_mul(rows_per_position)
            .ok_or_else(|| "kv host byte count overflowed".to_owned())?,
        resident: resident_row
            .checked_mul(rows_per_position)
            .ok_or_else(|| "kv resident byte count overflowed".to_owned())?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logical_kv_bytes_follow_loaded_geometry_and_element_types() {
        let geometry = KvGeometry {
            n_layers: 28,
            kv_heads: 8,
            head_dim: 128,
        };
        let bytes = kv_bytes_per_position(&geometry, LlmKvCacheSpec::DEFAULT).unwrap();
        assert_eq!(bytes.host, 224 * 1024);
        assert_eq!(bytes.resident, 112 * 1024);
        assert_eq!(bytes.total(), 336 * 1024);
    }

    #[test]
    fn invalid_or_overflowing_geometry_fails_closed() {
        assert!(
            kv_bytes_per_position(
                &KvGeometry {
                    n_layers: 0,
                    kv_heads: 8,
                    head_dim: 128,
                },
                LlmKvCacheSpec::DEFAULT,
            )
            .is_err()
        );
        assert!(
            kv_bytes_per_position(
                &KvGeometry {
                    n_layers: usize::MAX,
                    kv_heads: usize::MAX,
                    head_dim: 128,
                },
                LlmKvCacheSpec::DEFAULT,
            )
            .is_err()
        );
    }
}
