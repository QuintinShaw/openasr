use std::collections::BTreeMap;

use super::*;

pub const SAFETENSORS_HEADER_LENGTH_PREFIX_BYTES: usize = 8;
pub const SAFETENSORS_HEADER_MAX_BYTES_V0: u64 = 128 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SafetensorsTensorHeaderV0 {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub data_offsets: [u64; 2],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SafetensorsHeaderV0 {
    pub header_length_bytes: u64,
    pub data_length_bytes: u64,
    pub metadata: BTreeMap<String, String>,
    pub tensors: Vec<SafetensorsTensorHeaderV0>,
}

mod dtype;
mod header;
mod header_io;
mod header_parse;

pub use header::load_safetensors_header_v0;
