use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

pub const MIN_SMOKE_FRAMES: usize = 10;
pub const NON_SILENT_PEAK_THRESHOLD: i32 = 512;

pub fn frame_peak(samples: &[i16]) -> i32 {
    samples
        .iter()
        .map(|sample| i32::from(*sample).abs())
        .max()
        .unwrap_or(0)
}

pub fn write_smoke_wav(prefix: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "{prefix}-{}.wav",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time")
            .as_nanos()
    ));
    let sample_rate = 48_000_u32;
    let seconds = 4_u32;
    let samples = sample_rate * seconds;
    let data_bytes = samples * 2;
    let mut wav = Vec::with_capacity(44 + data_bytes as usize);
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&(36 + data_bytes).to_le_bytes());
    wav.extend_from_slice(b"WAVEfmt ");
    wav.extend_from_slice(&16_u32.to_le_bytes());
    wav.extend_from_slice(&1_u16.to_le_bytes());
    wav.extend_from_slice(&1_u16.to_le_bytes());
    wav.extend_from_slice(&sample_rate.to_le_bytes());
    wav.extend_from_slice(&(sample_rate * 2).to_le_bytes());
    wav.extend_from_slice(&2_u16.to_le_bytes());
    wav.extend_from_slice(&16_u16.to_le_bytes());
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_bytes.to_le_bytes());

    for index in 0..samples {
        let phase = index as f32 * 440.0 * std::f32::consts::TAU / sample_rate as f32;
        let sample = (phase.sin() * i16::MAX as f32 * 0.35).round() as i16;
        wav.extend_from_slice(&sample.to_le_bytes());
    }

    fs::write(&path, wav).expect("write smoke wav");
    path
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_peak_handles_i16_min_without_overflow() {
        assert_eq!(frame_peak(&[0, i16::MIN, i16::MAX]), 32_768);
    }

    #[test]
    fn writes_riff_pcm16_mono_wav() {
        let path = write_smoke_wav("openasr-system-audio-helper-test");
        let bytes = fs::read(&path).expect("read smoke wav");
        let _ = fs::remove_file(path);

        assert!(bytes.len() > 44);
        assert_eq!(&bytes[0..4], b"RIFF");
        assert_eq!(&bytes[8..12], b"WAVE");
        assert_eq!(&bytes[20..22], &1_u16.to_le_bytes());
        assert_eq!(&bytes[22..24], &1_u16.to_le_bytes());
        assert_eq!(&bytes[34..36], &16_u16.to_le_bytes());
    }
}
