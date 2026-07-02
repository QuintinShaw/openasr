use serde::Serialize;
use thiserror::Error;

pub const DEFAULT_REALTIME_SAMPLE_RATE_HZ: u32 = 16_000;
pub const DEFAULT_REALTIME_CHANNELS: u16 = 1;
const SUPPORTED_FRAME_DURATIONS_MS: &[u32] = &[10, 20, 30];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum RealtimeAudioEncoding {
    #[serde(rename = "pcm_s16le")]
    PcmS16Le,
}

impl RealtimeAudioEncoding {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::PcmS16Le => "pcm_s16le",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct RealtimeAudioFormat {
    pub encoding: RealtimeAudioEncoding,
    pub sample_rate_hz: u32,
    pub channels: u16,
}

impl RealtimeAudioFormat {
    pub fn pcm16_mono_16khz() -> Self {
        Self {
            encoding: RealtimeAudioEncoding::PcmS16Le,
            sample_rate_hz: DEFAULT_REALTIME_SAMPLE_RATE_HZ,
            channels: DEFAULT_REALTIME_CHANNELS,
        }
    }

    pub fn validate_normalized(self) -> Result<(), RealtimeFrameError> {
        if self.sample_rate_hz != DEFAULT_REALTIME_SAMPLE_RATE_HZ {
            return Err(RealtimeFrameError::UnsupportedSampleRate {
                sample_rate_hz: self.sample_rate_hz,
            });
        }
        if self.channels != DEFAULT_REALTIME_CHANNELS {
            return Err(RealtimeFrameError::UnsupportedChannelCount {
                channels: self.channels,
            });
        }
        Ok(())
    }

    pub fn sample_count_for_duration_ms(
        self,
        duration_ms: u32,
    ) -> Result<usize, RealtimeFrameError> {
        self.validate_normalized()?;
        validate_frame_duration_ms(duration_ms)?;
        Ok((self.sample_rate_hz as usize * duration_ms as usize) / 1_000)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RealtimeAudioFrame {
    pub seq: u64,
    pub start_ms: u64,
    pub format: RealtimeAudioFormat,
    samples: Vec<i16>,
}

impl RealtimeAudioFrame {
    pub fn new(
        seq: u64,
        start_ms: u64,
        format: RealtimeAudioFormat,
        samples: Vec<i16>,
    ) -> Result<Self, RealtimeFrameError> {
        format.validate_normalized()?;
        let frame = Self {
            seq,
            start_ms,
            format,
            samples,
        };
        let duration_ms = frame.duration_ms()?;
        validate_frame_duration_ms(duration_ms)?;
        Ok(frame)
    }

    pub fn from_pcm16le_bytes(
        seq: u64,
        start_ms: u64,
        format: RealtimeAudioFormat,
        bytes: &[u8],
    ) -> Result<Self, RealtimeFrameError> {
        let samples = pcm16le_bytes_to_samples(bytes)?;
        Self::new(seq, start_ms, format, samples)
    }

    pub fn samples(&self) -> &[i16] {
        &self.samples
    }

    pub fn into_samples(self) -> Vec<i16> {
        self.samples
    }

    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }

    pub fn duration_ms(&self) -> Result<u32, RealtimeFrameError> {
        let sample_rate = self.format.sample_rate_hz as u64;
        let numerator = self.samples.len() as u64 * 1_000;
        if (numerator / sample_rate) * sample_rate != numerator {
            return Err(RealtimeFrameError::NonIntegralFrameDuration {
                sample_count: self.samples.len(),
                sample_rate_hz: self.format.sample_rate_hz,
            });
        }
        Ok((numerator / sample_rate) as u32)
    }

    pub fn end_ms(&self) -> u64 {
        self.start_ms
            + self
                .duration_ms()
                .expect("RealtimeAudioFrame is validated at construction") as u64
    }

    pub fn byte_len(&self) -> usize {
        self.samples.len() * std::mem::size_of::<i16>()
    }
}

pub fn validate_frame_duration_ms(duration_ms: u32) -> Result<(), RealtimeFrameError> {
    if SUPPORTED_FRAME_DURATIONS_MS.contains(&duration_ms) {
        Ok(())
    } else {
        Err(RealtimeFrameError::UnsupportedFrameDuration {
            duration_ms,
            supported_ms: SUPPORTED_FRAME_DURATIONS_MS,
        })
    }
}

pub fn pcm16le_bytes_to_samples(bytes: &[u8]) -> Result<Vec<i16>, RealtimeFrameError> {
    if bytes.len() & 1 != 0 {
        return Err(RealtimeFrameError::OddPcm16ByteLength {
            byte_len: bytes.len(),
        });
    }

    Ok(bytes
        .chunks_exact(2)
        .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
        .collect())
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum RealtimeFrameError {
    #[error(
        "Unsupported realtime sample rate {sample_rate_hz} Hz. M48A accepts normalized 16 kHz audio only."
    )]
    UnsupportedSampleRate { sample_rate_hz: u32 },
    #[error(
        "Unsupported realtime channel count {channels}. M48A accepts normalized mono audio only."
    )]
    UnsupportedChannelCount { channels: u16 },
    #[error("Unsupported realtime frame duration {duration_ms} ms. Use one of: {supported_ms:?}.")]
    UnsupportedFrameDuration {
        duration_ms: u32,
        supported_ms: &'static [u32],
    },
    #[error(
        "Realtime frame sample count {sample_count} is not an exact millisecond duration at {sample_rate_hz} Hz."
    )]
    NonIntegralFrameDuration {
        sample_count: usize,
        sample_rate_hz: u32,
    },
    #[error("PCM16LE audio requires an even byte length, got {byte_len} bytes.")]
    OddPcm16ByteLength { byte_len: usize },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_default_format_and_frame_duration() {
        let format = RealtimeAudioFormat::pcm16_mono_16khz();
        assert_eq!(format.sample_count_for_duration_ms(20), Ok(320));
    }

    #[test]
    fn rejects_unsupported_sample_rate_and_channels() {
        let sample_rate = RealtimeAudioFormat {
            sample_rate_hz: 48_000,
            ..RealtimeAudioFormat::pcm16_mono_16khz()
        }
        .validate_normalized()
        .unwrap_err();
        assert!(matches!(
            sample_rate,
            RealtimeFrameError::UnsupportedSampleRate {
                sample_rate_hz: 48_000
            }
        ));

        let channels = RealtimeAudioFormat {
            channels: 2,
            ..RealtimeAudioFormat::pcm16_mono_16khz()
        }
        .validate_normalized()
        .unwrap_err();
        assert!(matches!(
            channels,
            RealtimeFrameError::UnsupportedChannelCount { channels: 2 }
        ));
    }

    #[test]
    fn converts_pcm16le_bytes_exactly() {
        let bytes = [0x01, 0x00, 0xff, 0x7f, 0x00, 0x80];
        assert_eq!(pcm16le_bytes_to_samples(&bytes), Ok(vec![1, 32767, -32768]));
    }

    #[test]
    fn rejects_odd_pcm16_byte_length() {
        let error = pcm16le_bytes_to_samples(&[0, 1, 2]).unwrap_err();
        assert_eq!(
            error,
            RealtimeFrameError::OddPcm16ByteLength { byte_len: 3 }
        );
    }

    #[test]
    fn frame_reports_offsets() {
        let frame =
            RealtimeAudioFrame::new(7, 40, RealtimeAudioFormat::pcm16_mono_16khz(), vec![0; 320])
                .unwrap();
        assert_eq!(frame.duration_ms(), Ok(20));
        assert_eq!(frame.end_ms(), 60);
    }

    #[test]
    fn rejects_non_exact_sample_count() {
        for sample_count in [159, 161, 319, 321, 479, 481] {
            let error = RealtimeAudioFrame::new(
                1,
                0,
                RealtimeAudioFormat::pcm16_mono_16khz(),
                vec![0; sample_count],
            )
            .unwrap_err();
            assert_eq!(
                error,
                RealtimeFrameError::NonIntegralFrameDuration {
                    sample_count,
                    sample_rate_hz: 16_000
                }
            );
        }
    }
}
