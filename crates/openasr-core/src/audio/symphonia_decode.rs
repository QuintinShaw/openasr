//! In-process audio decoding via symphonia (pure Rust, no external process).
//!
//! This is the default decode path for `prepare_audio_input`: m4a/AAC-LC
//! (isomp4, including the `.qta` QuickTime container), mp3, flac, ogg/vorbis,
//! and non-conformant wav all decode here without shelling out to ffmpeg or
//! afconvert. Anything this module cannot decode (HE-AAC, Opus, corrupt
//! files, containers/codecs outside the enabled symphonia features) returns
//! `None` so the caller falls back to the existing external converter chain
//! -- this module never produces a hard error, only "handled" or "not
//! handled".

use std::{fs::File, io::ErrorKind};

use rubato::{FftFixedIn, Resampler};
use symphonia::core::{
    audio::{AudioBufferRef, Signal},
    codecs::{CODEC_TYPE_AAC, CODEC_TYPE_NULL, CodecType, DecoderOptions},
    errors::Error as SymphoniaError,
    formats::FormatOptions,
    io::MediaSourceStream,
    meta::MetadataOptions,
    probe::Hint,
};

use std::path::Path;

const TARGET_SAMPLE_RATE: u32 = 16_000;
// FFT resampler chunk size: large enough to amortize FFT overhead, small
// enough to keep peak memory low for long recordings.
const RESAMPLE_CHUNK_FRAMES: usize = 4096;
const RESAMPLE_SUB_CHUNKS: usize = 2;

/// Attempt to decode `path` to a 16 kHz mono PCM16 WAV entirely in-process.
/// Returns `None` (never an error) if the container/codec is not supported
/// by the enabled symphonia features, or if decoding otherwise fails --
/// callers should fall back to an external converter in that case.
pub(crate) fn try_decode_to_pcm16_mono_16k_wav(
    path: &Path,
    extension: Option<&str>,
) -> Option<Vec<u8>> {
    let mono = decode_to_mono_f32(path, extension)?;
    if mono.samples.is_empty() {
        return None;
    }
    let resampled = if mono.sample_rate == TARGET_SAMPLE_RATE {
        mono.samples
    } else {
        resample_mono_to_16k(&mono.samples, mono.sample_rate)?
    };
    Some(encode_pcm16_mono_16k_wav(&resampled))
}

struct DecodedMono {
    samples: Vec<f32>,
    sample_rate: u32,
}

fn decode_to_mono_f32(path: &Path, extension: Option<&str>) -> Option<DecodedMono> {
    let file = File::open(path).ok()?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(extension) = extension {
        hint.with_extension(extension);
    }

    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .ok()?;
    let mut format = probed.format;

    let track = format
        .tracks()
        .iter()
        .find(|track| track.codec_params.codec != CODEC_TYPE_NULL)?;
    let track_id = track.id;

    if let Some(extra_data) = track.codec_params.extra_data.as_deref()
        && is_unsupported_aac_extension(&track.codec_params.codec, extra_data)
    {
        return None;
    }

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .ok()?;

    let mut samples: Vec<f32> = Vec::new();
    let mut sample_rate: Option<u32> = None;

    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(SymphoniaError::IoError(error)) if error.kind() == ErrorKind::UnexpectedEof => {
                break;
            }
            Err(SymphoniaError::ResetRequired) => break,
            Err(_) => return None,
        };

        if packet.track_id() != track_id {
            continue;
        }

        match decoder.decode(&packet) {
            Ok(decoded) => {
                let rate = decoded.spec().rate;
                sample_rate.get_or_insert(rate);
                push_downmixed_samples(&decoded, &mut samples);
            }
            // A single corrupt/undecodable packet does not doom the whole
            // stream; skip it and keep decoding (matches symphonia's own
            // player example, which treats DecodeError as recoverable).
            Err(SymphoniaError::DecodeError(_)) => continue,
            Err(SymphoniaError::IoError(error)) if error.kind() == ErrorKind::UnexpectedEof => {
                break;
            }
            Err(_) => return None,
        }
    }

    let sample_rate = sample_rate?;
    if sample_rate == 0 {
        return None;
    }

    Some(DecodedMono {
        samples,
        sample_rate,
    })
}

/// Detects explicit-signaling HE-AAC (SBR / PS) from the ISO 14496-3
/// `AudioSpecificConfig` so callers can fall back to an external converter
/// instead of silently producing bandwidth-limited audio: the plain AAC-LC
/// decoder these features enable ignores the SBR high-band extension. This
/// only recognizes *explicit* backward-compatible signaling (object type 5 =
/// SBR, 29 = PS), which is how mainstream m4a/mp4 encoders signal HE-AAC;
/// implicit signaling in raw ADTS streams is not detected here.
fn is_unsupported_aac_extension(codec: &CodecType, extra_data: &[u8]) -> bool {
    if *codec != CODEC_TYPE_AAC {
        return false;
    }
    let Some(&first_byte) = extra_data.first() else {
        return false;
    };
    let audio_object_type = first_byte >> 3;
    matches!(audio_object_type, 5 | 29)
}

fn push_downmixed_samples(decoded: &AudioBufferRef<'_>, out: &mut Vec<f32>) {
    let channels = decoded.spec().channels.count().max(1);
    let frames = decoded.frames();
    out.reserve(frames);
    // Every symphonia sample type (including the 24-bit `i24`/`u24` wrappers)
    // has a `FromSample<S> for f32` conversion, so a single generic path
    // covers both the fast mono case and the multi-channel downmix (a plain
    // arithmetic mean across channels).
    match decoded {
        AudioBufferRef::U8(buf) => downmix(buf, channels, out),
        AudioBufferRef::U16(buf) => downmix(buf, channels, out),
        AudioBufferRef::U24(buf) => downmix(buf, channels, out),
        AudioBufferRef::U32(buf) => downmix(buf, channels, out),
        AudioBufferRef::S8(buf) => downmix(buf, channels, out),
        AudioBufferRef::S16(buf) => downmix(buf, channels, out),
        AudioBufferRef::S24(buf) => downmix(buf, channels, out),
        AudioBufferRef::S32(buf) => downmix(buf, channels, out),
        AudioBufferRef::F32(buf) => downmix(buf, channels, out),
        AudioBufferRef::F64(buf) => downmix(buf, channels, out),
    }
}

fn downmix<S>(buf: &symphonia::core::audio::AudioBuffer<S>, channels: usize, out: &mut Vec<f32>)
where
    S: symphonia::core::sample::Sample,
    f32: symphonia::core::conv::FromSample<S>,
{
    let frames = buf.frames();
    if channels == 1 {
        out.extend(
            buf.chan(0)
                .iter()
                .map(|&s| <f32 as symphonia::core::conv::FromSample<S>>::from_sample(s)),
        );
        return;
    }
    for frame in 0..frames {
        let sum: f32 = (0..channels)
            .map(|channel| {
                <f32 as symphonia::core::conv::FromSample<S>>::from_sample(buf.chan(channel)[frame])
            })
            .sum();
        out.push(sum / channels as f32);
    }
}

/// Resamples mono `input` at `input_rate` Hz to 16 kHz using a pure-Rust FFT
/// resampler (rubato), processing fixed-size chunks and flushing the
/// resampler's internal delay at the end so no trailing audio is dropped.
fn resample_mono_to_16k(input: &[f32], input_rate: u32) -> Option<Vec<f32>> {
    let mut resampler = FftFixedIn::<f32>::new(
        input_rate as usize,
        TARGET_SAMPLE_RATE as usize,
        RESAMPLE_CHUNK_FRAMES,
        RESAMPLE_SUB_CHUNKS,
        1,
    )
    .ok()?;

    let mut output: Vec<f32> = Vec::with_capacity(
        input.len() * TARGET_SAMPLE_RATE as usize / input_rate.max(1) as usize
            + RESAMPLE_CHUNK_FRAMES,
    );
    let mut position = 0usize;

    while position + RESAMPLE_CHUNK_FRAMES <= input.len() {
        let chunk = vec![input[position..position + RESAMPLE_CHUNK_FRAMES].to_vec()];
        let processed = resampler.process(&chunk, None).ok()?;
        output.extend_from_slice(&processed[0]);
        position += RESAMPLE_CHUNK_FRAMES;
    }

    if position < input.len() {
        let remainder = vec![input[position..].to_vec()];
        let processed = resampler.process_partial(Some(&remainder), None).ok()?;
        output.extend_from_slice(&processed[0]);
    } else {
        let processed = resampler.process_partial::<Vec<f32>>(None, None).ok()?;
        output.extend_from_slice(&processed[0]);
    }

    Some(output)
}

/// Encodes mono f32 samples (expected in `[-1.0, 1.0]`) as a canonical PCM16
/// mono 16 kHz WAV file, matching the format `api::audio_io` expects.
fn encode_pcm16_mono_16k_wav(samples: &[f32]) -> Vec<u8> {
    let data_size = (samples.len() * 2) as u32;
    let mut bytes = Vec::with_capacity(44 + data_size as usize);
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&(36 + data_size).to_le_bytes());
    bytes.extend_from_slice(b"WAVE");
    bytes.extend_from_slice(b"fmt ");
    bytes.extend_from_slice(&16_u32.to_le_bytes());
    bytes.extend_from_slice(&1_u16.to_le_bytes()); // PCM
    bytes.extend_from_slice(&1_u16.to_le_bytes()); // mono
    bytes.extend_from_slice(&TARGET_SAMPLE_RATE.to_le_bytes());
    let byte_rate = TARGET_SAMPLE_RATE * 2;
    bytes.extend_from_slice(&byte_rate.to_le_bytes());
    bytes.extend_from_slice(&2_u16.to_le_bytes()); // block align
    bytes.extend_from_slice(&16_u16.to_le_bytes()); // bits per sample
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&data_size.to_le_bytes());
    for &sample in samples {
        let clamped = sample.clamp(-1.0, 1.0);
        let quantized = (clamped * i16::MAX as f32).round() as i16;
        bytes.extend_from_slice(&quantized.to_le_bytes());
    }
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wav_encoder_round_trips_through_audio_io_reader() {
        let samples = vec![0.0_f32, 0.5, -0.5, 1.0, -1.0];
        let bytes = encode_pcm16_mono_16k_wav(&samples);

        let parsed =
            crate::api::audio_io::load_wav_16khz_mono_f32_v0(write_temp(&bytes), "test", "test")
                .unwrap();

        assert_eq!(parsed.len(), samples.len());
        for (expected, actual) in samples.iter().zip(parsed.iter()) {
            assert!((expected - actual).abs() < 0.001, "{expected} vs {actual}");
        }
    }

    #[test]
    fn resample_preserves_frame_count_ratio() {
        let input: Vec<f32> = (0..48_000)
            .map(|index| (index as f32 / 48_000.0 * std::f32::consts::TAU * 440.0).sin())
            .collect();

        let output = resample_mono_to_16k(&input, 48_000).unwrap();

        // 48kHz -> 16kHz is a 3:1 ratio; allow slack for resampler group delay.
        let expected = input.len() / 3;
        let tolerance = RESAMPLE_CHUNK_FRAMES;
        assert!(
            output.len().abs_diff(expected) <= tolerance,
            "expected ~{expected} samples, got {}",
            output.len()
        );
    }

    fn write_temp(bytes: &[u8]) -> std::path::PathBuf {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.keep().join("out.wav");
        std::fs::write(&path, bytes).unwrap();
        path
    }
}
