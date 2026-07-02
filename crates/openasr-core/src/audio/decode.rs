use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::Path,
};

pub(crate) fn probe_wav_duration_inner(path: &Path) -> std::io::Result<Option<f64>> {
    let mut file = File::open(path)?;
    probe_wav_duration_from_file(&mut file)
}

fn probe_wav_duration_from_file(file: &mut File) -> std::io::Result<Option<f64>> {
    let Some((fmt, data_size)) = read_wav_fmt_and_data_size(file)? else {
        return Ok(None);
    };
    if fmt.audio_format != 1 && fmt.audio_format != 3 {
        return Ok(None);
    }
    if fmt.sample_rate == 0 || fmt.channels == 0 || fmt.bits_per_sample == 0 {
        return Ok(None);
    }

    let bytes_per_second =
        fmt.sample_rate as f64 * fmt.channels as f64 * (fmt.bits_per_sample as f64 / 8.0);
    if bytes_per_second <= 0.0 {
        return Ok(None);
    }

    let duration = data_size as f64 / bytes_per_second;
    if duration.is_finite() && duration > 0.0 {
        Ok(Some(duration))
    } else {
        Ok(None)
    }
}

fn read_wav_fmt_and_data_size(file: &mut File) -> std::io::Result<Option<(WavFmt, u32)>> {
    let mut header = [0_u8; 12];
    if file.read_exact(&mut header).is_err() {
        return Ok(None);
    }
    if &header[0..4] != b"RIFF" || &header[8..12] != b"WAVE" {
        return Ok(None);
    }

    let mut fmt: Option<WavFmt> = None;
    let mut data_size: Option<u32> = None;

    loop {
        let mut chunk_header = [0_u8; 8];
        match file.read_exact(&mut chunk_header) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(error) => return Err(error),
        }

        let chunk_id = &chunk_header[0..4];
        let chunk_size = u32::from_le_bytes(chunk_header[4..8].try_into().expect("chunk size"));

        match chunk_id {
            b"fmt " => {
                if chunk_size > 4096 {
                    return Ok(None);
                }
                let mut bytes = vec![0_u8; chunk_size as usize];
                file.read_exact(&mut bytes)?;
                if chunk_size % 2 == 1 {
                    file.seek(SeekFrom::Current(1))?;
                }
                fmt = parse_wav_fmt(&bytes);
            }
            b"data" => {
                data_size = Some(chunk_size);
                file.seek(SeekFrom::Current(padded_chunk_size(chunk_size) as i64))?;
            }
            _ => {
                file.seek(SeekFrom::Current(padded_chunk_size(chunk_size) as i64))?;
            }
        }

        if fmt.is_some() && data_size.is_some() {
            break;
        }
    }

    Ok(match (fmt, data_size) {
        (Some(fmt), Some(data_size)) => Some((fmt, data_size)),
        _ => None,
    })
}

fn parse_wav_fmt(bytes: &[u8]) -> Option<WavFmt> {
    if bytes.len() < 16 {
        return None;
    }

    Some(WavFmt {
        audio_format: u16::from_le_bytes(bytes[0..2].try_into().ok()?),
        channels: u16::from_le_bytes(bytes[2..4].try_into().ok()?),
        sample_rate: u32::from_le_bytes(bytes[4..8].try_into().ok()?),
        bits_per_sample: u16::from_le_bytes(bytes[14..16].try_into().ok()?),
    })
}

fn padded_chunk_size(size: u32) -> u64 {
    u64::from(size) + u64::from(size % 2)
}

#[derive(Debug, Clone, Copy)]
struct WavFmt {
    audio_format: u16,
    channels: u16,
    sample_rate: u32,
    bits_per_sample: u16,
}
