use std::{fs, path::Path};

use crate::NativeAsrError;

pub fn load_wav_16khz_mono_f32_v0(
    path: impl AsRef<Path>,
    reader_label: &str,
    input_label: &str,
) -> Result<Vec<f32>, NativeAsrError> {
    let bytes = fs::read(path.as_ref()).map_err(|error| NativeAsrError::SessionFailed {
        message: format!(
            "{reader_label} could not read WAV input '{}': {error}",
            path.as_ref().display()
        ),
    })?;
    parse_wav_16khz_mono_f32(&bytes, input_label)
}

fn parse_wav_16khz_mono_f32(bytes: &[u8], input_label: &str) -> Result<Vec<f32>, NativeAsrError> {
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(wav_error("input is not a RIFF/WAVE file"));
    }
    let mut cursor = 12_usize;
    let mut fmt: Option<WavFormat> = None;
    let mut data: Option<&[u8]> = None;
    while cursor.checked_add(8).is_some_and(|end| end <= bytes.len()) {
        let chunk_id = &bytes[cursor..cursor + 4];
        let chunk_len =
            u32::from_le_bytes(bytes[cursor + 4..cursor + 8].try_into().unwrap()) as usize;
        cursor += 8;
        let chunk_end = cursor
            .checked_add(chunk_len)
            .ok_or_else(|| wav_error("WAV chunk length overflows usize"))?;
        if chunk_end > bytes.len() {
            return Err(wav_error("WAV chunk extends past end of file"));
        }
        match chunk_id {
            b"fmt " => fmt = Some(parse_wav_fmt(&bytes[cursor..chunk_end])?),
            b"data" => data = Some(&bytes[cursor..chunk_end]),
            _ => {}
        }
        cursor = chunk_end + (chunk_len % 2);
    }
    let fmt = fmt.ok_or_else(|| wav_error("missing fmt chunk"))?;
    let data = data.ok_or_else(|| wav_error("missing data chunk"))?;
    if fmt.channels != 1 || fmt.sample_rate_hz != 16_000 || !matches!(fmt.audio_format, 1 | 3) {
        return Err(wav_error(format!(
            "expected 16 kHz mono PCM16 or float32 WAV input for {input_label}"
        )));
    }
    match (fmt.audio_format, fmt.bits_per_sample) {
        (1, 16) => parse_pcm16_samples(data),
        (3, 32) => parse_float32_samples(data),
        _ => Err(wav_error(format!(
            "expected PCM16 or IEEE-float32 sample payload for {input_label}"
        ))),
    }
}

#[derive(Debug, Clone, Copy)]
struct WavFormat {
    audio_format: u16,
    channels: u16,
    sample_rate_hz: u32,
    bits_per_sample: u16,
}

fn parse_wav_fmt(bytes: &[u8]) -> Result<WavFormat, NativeAsrError> {
    if bytes.len() < 16 {
        return Err(wav_error("fmt chunk is shorter than 16 bytes"));
    }
    Ok(WavFormat {
        audio_format: u16::from_le_bytes(bytes[0..2].try_into().unwrap()),
        channels: u16::from_le_bytes(bytes[2..4].try_into().unwrap()),
        sample_rate_hz: u32::from_le_bytes(bytes[4..8].try_into().unwrap()),
        bits_per_sample: u16::from_le_bytes(bytes[14..16].try_into().unwrap()),
    })
}

fn parse_pcm16_samples(data: &[u8]) -> Result<Vec<f32>, NativeAsrError> {
    parse_wav_samples(data, SampleEncoding::Pcm16)
}

fn parse_float32_samples(data: &[u8]) -> Result<Vec<f32>, NativeAsrError> {
    parse_wav_samples(data, SampleEncoding::Float32)
}

fn parse_wav_samples(data: &[u8], encoding: SampleEncoding) -> Result<Vec<f32>, NativeAsrError> {
    let sample_width = encoding.byte_width();
    if !data.len().is_multiple_of(sample_width) {
        return Err(wav_error(encoding.width_error()));
    }
    data.chunks_exact(sample_width)
        .map(|chunk| encoding.parse(chunk))
        .collect()
}

#[derive(Clone, Copy)]
enum SampleEncoding {
    Pcm16,
    Float32,
}

impl SampleEncoding {
    const fn byte_width(self) -> usize {
        match self {
            Self::Pcm16 => 2,
            Self::Float32 => 4,
        }
    }

    const fn width_error(self) -> &'static str {
        match self {
            Self::Pcm16 => "PCM16 data chunk has an odd byte length",
            Self::Float32 => "float32 data chunk length is not divisible by 4",
        }
    }

    fn parse(self, chunk: &[u8]) -> Result<f32, NativeAsrError> {
        match self {
            Self::Pcm16 => Ok(
                i16::from_le_bytes(chunk.try_into().expect("chunk width must match")) as f32
                    / 32768.0,
            ),
            Self::Float32 => {
                let sample = f32::from_le_bytes(chunk.try_into().expect("chunk width must match"));
                if sample.is_finite() {
                    Ok(sample)
                } else {
                    Err(wav_error("float32 WAV data contains non-finite samples"))
                }
            }
        }
    }
}

fn wav_error(message: impl Into<String>) -> NativeAsrError {
    NativeAsrError::SessionFailed {
        message: message.into(),
    }
}
