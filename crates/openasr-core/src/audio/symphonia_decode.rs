//! In-process audio decoding via symphonia (pure Rust, no external process).
//!
//! This is the default decode path for `prepare_audio_input`: m4a/AAC-LC/ALAC
//! (isomp4, including the `.qta` QuickTime container), mp3, flac, ogg/vorbis,
//! mkv/webm (vorbis track only -- see below), and non-conformant wav all
//! decode here without shelling out to ffmpeg or afconvert. Anything this
//! module cannot decode (HE-AAC, Opus in any container, corrupt files,
//! containers/codecs outside the enabled symphonia features) reports
//! [`SymphoniaOutcome::Unsupported`] (never a hard error) so the caller falls
//! back to the existing external converter chain.
//!
//! Opus is the big absence: symphonia has never shipped an Opus decoder (still
//! true as of 0.6.0), so `.opus` files, and Opus tracks inside `.ogg`/`.webm`,
//! always fall through regardless of which demuxer/codec features are
//! enabled here. [`SymphoniaOutcome::Unsupported`] carries a `codec_label`
//! when the demuxer could name the codec anyway, so callers can still tell
//! the user *which* codec was the problem instead of a bare failure.
//!
//! # Untrusted input and third-party demuxer bugs
//!
//! `path` is arbitrary user-supplied bytes reaching a third-party demuxer
//! (symphonia's format readers, including `symphonia-format-mkv`), which is
//! outside this workspace's control and not guaranteed panic-free on
//! malformed input -- e.g. a webm/mkv file whose first EBML element-size byte
//! is `0x00` currently triggers a `debug_assert`-style subtract-overflow
//! panic in `symphonia-format-mkv 0.5.5`'s vint reader (`ebml.rs`), since it
//! computes `7 - byte.leading_zeros()` without checking that
//! `leading_zeros() <= 7`. Per `AGENTS.md`'s trust-boundary invariant
//! ("panic-free on untrusted input"), every symphonia entry point below runs
//! inside [`std::panic::catch_unwind`] and turns a caught panic into
//! [`SymphoniaOutcome::ParserPanicked`] / [`ProbeOutcome::ParserPanicked`],
//! which callers report as a typed "internal parser error" rather than
//! letting the process crash or misreporting the file as corrupt.

use std::{fs::File, io::ErrorKind, panic::catch_unwind, path::Path};

use rubato::{FftFixedIn, Resampler};
use symphonia::core::{
    audio::{AudioBufferRef, Signal},
    codecs::{
        CODEC_TYPE_AAC, CODEC_TYPE_ALAC, CODEC_TYPE_FLAC, CODEC_TYPE_MP3, CODEC_TYPE_NULL,
        CODEC_TYPE_OPUS, CODEC_TYPE_VORBIS, CodecType, DecoderOptions,
    },
    errors::Error as SymphoniaError,
    formats::FormatOptions,
    io::MediaSourceStream,
    meta::MetadataOptions,
    probe::Hint,
};

pub(crate) const TARGET_SAMPLE_RATE_HZ: u32 = 16_000;
// FFT resampler chunk size: large enough to amortize FFT overhead, small
// enough to keep peak memory low for long recordings.
const RESAMPLE_CHUNK_FRAMES: usize = 4096;
const RESAMPLE_SUB_CHUNKS: usize = 2;

/// The decoded file's *source* format, before this module's mono-downmix and
/// 16 kHz resample -- e.g. `{ sample_rate_hz: 44100, channels: 2 }` for a
/// typical music-app m4a export. Surfaced so callers (see
/// `prepare::try_symphonia_prepare`) can report the true source format for
/// diagnostics without a second, separate probe of the same file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DecodedAudioSourceFormat {
    pub(crate) sample_rate_hz: u32,
    pub(crate) channels: u16,
}

/// Result of attempting the in-process symphonia decode path.
pub(crate) enum SymphoniaOutcome {
    /// Decoded successfully: ready-to-use 16 kHz mono f32 samples, plus the
    /// file's true source format (sample rate/channels, before this module's
    /// downmix/resample) for diagnostics. Callers hand these samples
    /// straight to the rest of the pipeline in memory -- no WAV encode, disk
    /// write, or re-read/re-parse round trip.
    Decoded(Vec<f32>, DecodedAudioSourceFormat),
    /// Not decodable in-process (unsupported codec, malformed stream, or an
    /// otherwise-empty result) -- fall back to the external converter chain.
    /// `codec_label` is populated whenever the demuxer identified the track's
    /// codec before decoding failed (see [`codec_type_label`]), even though
    /// no decoder for it is linked into this build.
    Unsupported { codec_label: Option<String> },
    /// The symphonia demuxer/decoder itself panicked on this input (a
    /// third-party bug hit via malformed/adversarial bytes -- see the module
    /// docs). Callers must treat this exactly like `Unsupported` for control
    /// flow (fall back to the external converter) but should report it as an
    /// internal parser error rather than an unsupported codec.
    ParserPanicked,
}

/// Attempt to decode `path` to 16 kHz mono f32 samples entirely in-process.
/// Never panics, even on adversarial input (see module docs): a panic inside
/// the underlying symphonia demuxer/decoder is caught and reported as
/// [`SymphoniaOutcome::ParserPanicked`].
pub(crate) fn try_decode_to_pcm16_mono_16k(
    path: &Path,
    extension: Option<&str>,
) -> SymphoniaOutcome {
    // `path: &Path` and `extension: Option<&str>` are plain shared references
    // to data with no interior mutability, so this closure's captured
    // environment is `UnwindSafe` on its own merits -- no `AssertUnwindSafe`
    // needed. `decode_attempt` also allocates and owns all of its mutable
    // state (the symphonia reader, decoder, sample buffer) locally, so
    // nothing mutable crosses the unwind boundary either way.
    let attempt = match catch_unwind(|| decode_attempt(path, extension)) {
        Ok(attempt) => attempt,
        Err(_) => return SymphoniaOutcome::ParserPanicked,
    };

    let Some(mono) = attempt.mono else {
        return SymphoniaOutcome::Unsupported {
            codec_label: attempt.codec_label,
        };
    };
    if mono.samples.is_empty() {
        return SymphoniaOutcome::Unsupported {
            codec_label: attempt.codec_label,
        };
    }
    let source_format = DecodedAudioSourceFormat {
        sample_rate_hz: mono.sample_rate,
        channels: mono.channels,
    };
    let resampled = if mono.sample_rate == TARGET_SAMPLE_RATE_HZ {
        mono.samples
    } else {
        match resample_mono_to_16k(&mono.samples, mono.sample_rate) {
            Some(resampled) => resampled,
            None => {
                return SymphoniaOutcome::Unsupported {
                    codec_label: attempt.codec_label,
                };
            }
        }
    };
    SymphoniaOutcome::Decoded(resampled, source_format)
}

/// Result of [`probe_codec_label`]: names the codec of a file's first real
/// track without requiring a decoder for it, for building diagnostic
/// messages when the in-process decode path was never attempted (the
/// explicit-ffmpeg escape hatch bypasses it entirely -- see `prepare.rs`).
pub(crate) enum ProbeOutcome {
    /// The demuxer identified the track's codec.
    Codec(String),
    /// The container itself could not be probed (unrecognized/corrupt
    /// bytes), which is not this function's concern to diagnose.
    Unknown,
    /// The symphonia probe panicked on this input; see the module docs.
    ParserPanicked,
}

/// Probes `path` far enough to name the audio codec of its first real track,
/// without requiring a decoder for that codec to be compiled in. Symphonia's
/// codec type registry (`CODEC_TYPE_*`) is populated by every demuxer
/// regardless of which decoder features this build enables, so this can
/// identify e.g. Opus in a container symphonia can parse but not decode --
/// letting callers report a precise "this codec is unsupported" error instead
/// of an opaque conversion failure. Never panics (see module docs).
pub(crate) fn probe_codec_label(path: &Path, extension: Option<&str>) -> ProbeOutcome {
    // Same `UnwindSafe` reasoning as `try_decode_to_pcm16_mono_16k` above:
    // both captured arguments are plain shared references with no interior
    // mutability.
    match catch_unwind(|| probe_codec_label_inner(path, extension)) {
        Ok(Some(label)) => ProbeOutcome::Codec(label),
        Ok(None) => ProbeOutcome::Unknown,
        Err(_) => ProbeOutcome::ParserPanicked,
    }
}

fn probe_codec_label_inner(path: &Path, extension: Option<&str>) -> Option<String> {
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
    let track = probed
        .format
        .tracks()
        .iter()
        .find(|track| track.codec_params.codec != CODEC_TYPE_NULL)?;

    Some(codec_type_label(track.codec_params.codec))
}

/// Human-readable name for the codecs symphonia's format registry can
/// identify, whether or not this build links a decoder for them.
fn codec_type_label(codec: CodecType) -> String {
    match codec {
        CODEC_TYPE_OPUS => "Opus".to_string(),
        CODEC_TYPE_VORBIS => "Vorbis".to_string(),
        CODEC_TYPE_AAC => "AAC".to_string(),
        CODEC_TYPE_MP3 => "MP3".to_string(),
        CODEC_TYPE_FLAC => "FLAC".to_string(),
        CODEC_TYPE_ALAC => "ALAC".to_string(),
        other => format!("codec {other}"),
    }
}

struct DecodedMono {
    samples: Vec<f32>,
    sample_rate: u32,
    /// The source track's channel count *before* this function's mono
    /// downmix (which always collapses to 1) -- captured from the first
    /// successfully decoded packet's `AudioBufferRef::spec()`, same as
    /// `sample_rate` above.
    channels: u16,
}

/// Combines demuxing, codec identification, and decoding into a single pass
/// so a caller reporting a failed decode doesn't need to re-open and re-probe
/// the file just to name the codec (see `codec_label` on the returned
/// struct). Never itself panics on symphonia's behalf -- run this through
/// `catch_unwind` (as `try_decode_to_pcm16_mono_16k` does), not directly.
struct DecodeAttempt {
    /// Populated as soon as a real (non-null) track is found, even if
    /// decoding it then fails -- see [`codec_type_label`].
    codec_label: Option<String>,
    mono: Option<DecodedMono>,
}

fn decode_attempt(path: &Path, extension: Option<&str>) -> DecodeAttempt {
    let mut codec_label = None;
    let mono = decode_to_mono_f32(path, extension, &mut codec_label);
    DecodeAttempt { codec_label, mono }
}

fn decode_to_mono_f32(
    path: &Path,
    extension: Option<&str>,
    codec_label: &mut Option<String>,
) -> Option<DecodedMono> {
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
    *codec_label = Some(codec_type_label(track.codec_params.codec));

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
    let mut channels: Option<u16> = None;

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
                let spec = decoded.spec();
                sample_rate.get_or_insert(spec.rate);
                channels.get_or_insert(spec.channels.count() as u16);
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
    // `.max(1)` mirrors `push_downmixed_samples`'s own floor: a spec reporting
    // zero channels is nonsensical, so treat it the same as "unknown" rather
    // than surfacing an impossible `0` in a diagnostics field.
    let channels = channels.unwrap_or(1).max(1);

    Some(DecodedMono {
        samples,
        sample_rate,
        channels,
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
///
/// The main loop uses `process_into_buffer` with a pair of buffers allocated
/// once up front (`Resampler::input_buffer_allocate` /
/// `output_buffer_allocate`, sized to what `FftFixedIn` needs per call) and
/// reused across every chunk, instead of the convenience `process()` +
/// `chunk.to_vec()` pairing that used to allocate a fresh input `Vec` and a
/// fresh output `Vec<Vec<f32>>` for every `RESAMPLE_CHUNK_FRAMES` chunk (a
/// 10-minute 48 kHz input is ~2160 chunks). The numeric path is unchanged --
/// `process_into_buffer` is what `process()` itself calls internally after
/// allocating its buffers (see `Resampler::process` in rubato); only the
/// buffer lifetime moved from per-chunk to per-call.
fn resample_mono_to_16k(input: &[f32], input_rate: u32) -> Option<Vec<f32>> {
    let mut resampler = FftFixedIn::<f32>::new(
        input_rate as usize,
        TARGET_SAMPLE_RATE_HZ as usize,
        RESAMPLE_CHUNK_FRAMES,
        RESAMPLE_SUB_CHUNKS,
        1,
    )
    .ok()?;

    let mut output: Vec<f32> = Vec::with_capacity(
        input.len() * TARGET_SAMPLE_RATE_HZ as usize / input_rate.max(1) as usize
            + RESAMPLE_CHUNK_FRAMES,
    );
    let mut position = 0usize;

    // `FftFixedIn::input_frames_max()` is the fixed `RESAMPLE_CHUNK_FRAMES`
    // chunk size, so this input buffer is reused verbatim (only its contents
    // change) across every full-chunk iteration below.
    let mut input_buffer = resampler.input_buffer_allocate(true);
    let mut output_buffer = resampler.output_buffer_allocate(true);

    while position + RESAMPLE_CHUNK_FRAMES <= input.len() {
        input_buffer[0].copy_from_slice(&input[position..position + RESAMPLE_CHUNK_FRAMES]);
        let (_, out_len) = resampler
            .process_into_buffer(&input_buffer, &mut output_buffer, None)
            .ok()?;
        output.extend_from_slice(&output_buffer[0][..out_len]);
        position += RESAMPLE_CHUNK_FRAMES;
    }

    // Tail handling (a short final chunk, plus the zero-input flush that
    // drains the resampler's internal delay line) runs at most twice per
    // call regardless of input length, so it keeps using the
    // `process_partial_into_buffer` convenience method for its zero-padding;
    // only the *output* buffer is the shared, pre-allocated one.
    if position < input.len() {
        let remainder = [input[position..].to_vec()];
        let (_, out_len) = resampler
            .process_partial_into_buffer(Some(&remainder), &mut output_buffer, None)
            .ok()?;
        output.extend_from_slice(&output_buffer[0][..out_len]);
    } else {
        let (_, out_len) = resampler
            .process_partial_into_buffer(Option::<&[Vec<f32>]>::None, &mut output_buffer, None)
            .ok()?;
        output.extend_from_slice(&output_buffer[0][..out_len]);
    }

    Some(output)
}

#[cfg(test)]
mod tests {
    use super::*;

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

    /// A minimal webm/mkv EBML header whose size vint is the single byte
    /// `0x00`: `symphonia-format-mkv 0.5.5`'s `read_vint` computes
    /// `7 - byte.leading_zeros()` without checking that `leading_zeros() <=
    /// 7`, so `leading_zeros(0x00) == 8` underflows that subtraction and
    /// panics (`attempt to subtract with overflow`) in a debug/overflow-
    /// checked build. The probe needs a 16-byte window to recognize the
    /// container at all (see `symphonia_core::probe::Probe::next`), hence
    /// the trailing padding -- this is the smallest input that reaches the
    /// buggy line.
    fn malformed_webm_vint_zero_bytes() -> Vec<u8> {
        let mut bytes = vec![0x1A, 0x45, 0xDF, 0xA3, 0x00];
        bytes.extend(std::iter::repeat_n(0xAA, 11));
        bytes
    }

    #[test]
    fn malformed_webm_vint_underflow_is_caught_not_panicked() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("malformed.webm");
        std::fs::write(&path, malformed_webm_vint_zero_bytes()).unwrap();

        // Before the `catch_unwind` guard, this call panicked (verified via a
        // standalone repro against symphonia-format-mkv 0.5.5 directly); it
        // must now report `ParserPanicked` and let the caller fall back to
        // the external converter chain instead of crashing the process.
        assert!(matches!(
            try_decode_to_pcm16_mono_16k(&path, Some("webm")),
            SymphoniaOutcome::ParserPanicked
        ));
        assert!(matches!(
            probe_codec_label(&path, Some("webm")),
            ProbeOutcome::ParserPanicked
        ));
    }
}
