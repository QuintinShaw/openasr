use super::*;

use super::header_io::read_header_and_data_lengths_from_file;
use super::header_parse::parse_safetensors_header_json_object;

pub fn load_safetensors_header_v0(
    path: impl AsRef<Path>,
) -> Result<SafetensorsHeaderV0, WhisperLocalSourceError> {
    let (header_length_bytes, data_length_bytes, header_bytes) =
        read_header_and_data_lengths_from_file(path.as_ref())?;
    parse_safetensors_header_json_object(header_length_bytes, data_length_bytes, &header_bytes)
}
