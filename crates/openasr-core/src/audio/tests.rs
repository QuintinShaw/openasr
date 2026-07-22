use std::{f32::consts::TAU, fs, path::PathBuf};

use super::*;

#[test]
fn recognized_extensions_are_case_insensitive() {
    let temp = tempfile::tempdir().unwrap();
    let wav = temp.path().join("sample.WAV");
    fs::write(&wav, b"not a wav").unwrap();

    let info = probe_audio_input(&wav).unwrap();

    assert_eq!(info.extension.as_deref(), Some("wav"));
    assert!(info.recognized_extension);
    assert!(info.issues.is_empty());
}

#[test]
fn unknown_extension_is_marked_but_does_not_error() {
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("sample.unknownaudio");
    fs::write(&input, b"audio bytes").unwrap();

    let info = probe_audio_input(&input).unwrap();

    assert_eq!(info.extension.as_deref(), Some("unknownaudio"));
    assert!(!info.recognized_extension);
    assert_eq!(
        info.issues,
        vec![AudioInputIssue::UnknownExtension(
            "unknownaudio".to_string()
        )]
    );
}

#[test]
fn qta_extension_is_recognized() {
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("voice memo.qta");
    fs::write(&input, b"not a real mov").unwrap();

    let info = probe_audio_input(&input).unwrap();

    assert_eq!(info.extension.as_deref(), Some("qta"));
    assert!(info.recognized_extension);
    assert!(info.issues.is_empty());
}

#[test]
fn missing_file_errors() {
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("missing.wav");

    let error = probe_audio_input(&input).unwrap_err().to_string();

    assert!(error.contains("Input file not found:"));
    assert!(error.contains("Please provide a valid audio or video file path."));
}

#[test]
fn directory_input_errors() {
    let temp = tempfile::tempdir().unwrap();

    let error = probe_audio_input(temp.path()).unwrap_err().to_string();

    assert!(error.contains("Input path is a directory:"));
    assert!(error.contains("Please provide a valid audio or video file path."));
}

#[cfg(unix)]
#[test]
fn non_regular_input_errors() {
    let error = probe_audio_input("/dev/null").unwrap_err().to_string();

    assert!(error.contains("Input path is not a regular file:"));
    assert!(error.contains("Please provide a valid audio or video file path."));
}

#[test]
fn wav_duration_is_parsed_for_fixture_if_supported() {
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|path| path.parent())
        .unwrap()
        .join("fixtures/jfk.wav");
    let info = probe_audio_input(fixture).unwrap();

    if let Some(duration) = info.duration_seconds {
        assert!(duration > 0.0);
    }
}

#[test]
fn unsupported_or_malformed_wav_returns_unknown_duration_without_panic() {
    let temp = tempfile::tempdir().unwrap();
    let wav = temp.path().join("bad.wav");
    fs::write(&wav, b"RIFF\x04\x00\x00\x00WAVE").unwrap();

    let info = probe_audio_input(&wav).unwrap();

    assert_eq!(info.duration_seconds, None);
}

#[test]
fn non_wav_duration_is_unknown() {
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("sample.mp3");
    fs::write(&input, b"not parsed").unwrap();

    let info = probe_audio_input(&input).unwrap();

    assert_eq!(info.duration_seconds, None);
}

#[test]
fn no_extension_file_can_be_probed_without_failing() {
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("sample");
    fs::write(&input, b"audio bytes").unwrap();

    let info = probe_audio_input(&input).unwrap();

    assert_eq!(info.extension, None);
    assert!(!info.recognized_extension);
    assert!(info.issues.is_empty());
}

#[test]
fn wav_duration_probe_reads_pcm_duration() {
    let temp = tempfile::tempdir().unwrap();
    let wav = temp.path().join("tone.wav");
    write_test_wav(&wav, 16_000, 1, 16, 16_000);

    let duration = probe_wav_duration(&wav).unwrap();

    assert!((duration - 1.0).abs() < 0.001);
}

#[test]
fn wav_passthrough_does_not_require_ffmpeg_for_external_backend() {
    let temp = tempfile::tempdir().unwrap();
    let wav = temp.path().join("sample.wav");
    fs::write(&wav, b"not a real wav").unwrap();

    let prepared =
        prepare_audio_input(&wav, &AudioPreparationOptions::new(BackendKind::Native)).unwrap();

    assert_eq!(prepared.path(), wav.as_path());
    assert!(!prepared.is_converted());
}

#[test]
fn native_pcm16_mono_16khz_wav_passes_through() {
    let temp = tempfile::tempdir().unwrap();
    let wav = temp.path().join("sample.wav");
    write_test_wav(&wav, 16_000, 1, 16, 16_000);

    let prepared =
        prepare_audio_input(&wav, &AudioPreparationOptions::new(BackendKind::Native)).unwrap();

    assert_eq!(prepared.path(), wav.as_path());
    assert!(!prepared.is_converted());
}

#[test]
fn native_float_wav_passthrough_without_ffmpeg() {
    let temp = tempfile::tempdir().unwrap();
    let wav = temp.path().join("sample.wav");
    write_float_wav(&wav, 16_000, 1, 16_000);

    let prepared =
        prepare_audio_input(&wav, &AudioPreparationOptions::new(BackendKind::Native)).unwrap();

    assert_eq!(prepared.path(), wav.as_path());
    assert!(!prepared.is_converted());
}

#[test]
#[cfg(not(target_os = "macos"))]
fn native_non_wav_conversion_mode_requires_ffmpeg_when_enabled() {
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("sample.mp3");
    fs::write(&input, b"mock bytes").unwrap();

    let error = prepare_audio_input(
        &input,
        &AudioPreparationOptions::new(BackendKind::Native).with_native_non_wav_conversion(true),
    )
    .unwrap_err();

    assert!(matches!(
        error,
        AudioPreparationError::MissingFfmpeg {
            backend: BackendKind::Native,
            ..
        }
    ));
}

#[test]
#[cfg(not(target_os = "macos"))]
fn native_qta_input_is_recognized_and_reaches_ffmpeg_conversion() {
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("sample.qta");
    fs::write(&input, b"mock bytes").unwrap();

    let error = prepare_audio_input(
        &input,
        &AudioPreparationOptions::new(BackendKind::Native).with_native_non_wav_conversion(true),
    )
    .unwrap_err();

    // A `.qta` file must reach the ffmpeg conversion step (and fail only
    // because no ffmpeg binary is configured in this test), not get rejected
    // upfront as an unrecognized extension.
    assert!(matches!(
        error,
        AudioPreparationError::MissingFfmpeg {
            backend: BackendKind::Native,
            ..
        }
    ));
}

// On macOS these same inputs reach the afconvert fallback instead of erroring
// with MissingFfmpeg -- see the macos-only tests below (`native_non_wav_*` /
// `native_qta_*` counterparts) that assert the conversion actually happens.
#[test]
#[cfg(target_os = "macos")]
fn native_non_wav_conversion_falls_back_to_afconvert_when_ffmpeg_absent() {
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("sample.mp3");
    fs::write(&input, b"mock bytes, not a real mp3").unwrap();

    let error = prepare_audio_input(
        &input,
        &AudioPreparationOptions::new(BackendKind::Native).with_native_non_wav_conversion(true),
    )
    .unwrap_err();

    // Reaches afconvert (present at /usr/bin/afconvert on every macOS
    // install) and fails there because the fixture bytes are not a real MP3
    // stream -- proof the fallback is wired up rather than short-circuiting
    // to MissingFfmpeg.
    assert!(matches!(
        error,
        AudioPreparationError::ConversionFailed { tool, .. } if tool == "afconvert"
    ));
}

#[test]
#[cfg(target_os = "macos")]
fn native_qta_input_reaches_afconvert_conversion() {
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("sample.qta");
    fs::write(&input, b"mock bytes, not a real mov").unwrap();

    let error = prepare_audio_input(
        &input,
        &AudioPreparationOptions::new(BackendKind::Native).with_native_non_wav_conversion(true),
    )
    .unwrap_err();

    assert!(matches!(
        error,
        AudioPreparationError::ConversionFailed { tool, .. } if tool == "afconvert"
    ));
}

#[test]
#[cfg(target_os = "macos")]
fn native_m4a_input_converts_via_afconvert_without_ffmpeg() {
    // With symphonia now the default in-process decoder for m4a/AAC-LC, this
    // real m4a normally decodes via `try_symphonia_prepare` rather than
    // reaching afconvert -- but either path must land here on a valid,
    // converted 16 kHz WAV, so this still covers the end-to-end contract.
    let fixture_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|path| path.parent())
        .unwrap()
        .join("fixtures");
    let temp = tempfile::tempdir().unwrap();
    let m4a = temp.path().join("jfk.m4a");
    let status = std::process::Command::new("/usr/bin/afconvert")
        .arg("-f")
        .arg("m4af")
        .arg("-d")
        .arg("aac")
        .arg(fixture_dir.join("jfk.wav"))
        .arg(&m4a)
        .status()
        .expect("afconvert must be available to build the m4a fixture");
    assert!(status.success());

    let prepared = prepare_audio_input(
        &m4a,
        &AudioPreparationOptions::new(BackendKind::Native).with_native_non_wav_conversion(true),
    )
    .expect("decoding a real m4a without ffmpeg configured should succeed");

    assert!(prepared.is_converted());
    let bytes = fs::read(prepared.path()).unwrap();
    assert_eq!(&bytes[0..4], b"RIFF");
    assert_eq!(&bytes[8..12], b"WAVE");
}

fn crate_fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn prepare_native_conversion(path: &Path) -> Result<PreparedAudioInput, AudioPreparationError> {
    prepare_audio_input(
        path,
        &AudioPreparationOptions::new(BackendKind::Native).with_native_non_wav_conversion(true),
    )
}

fn assert_prepared_16k_mono_wav(prepared: &PreparedAudioInput) {
    assert!(prepared.is_converted());
    let bytes = fs::read(prepared.path()).unwrap();
    let samples = crate::api::audio_io::load_wav_16khz_mono_f32_v0(prepared.path(), "test", "test")
        .expect("prepared output must be a valid 16 kHz mono WAV");
    assert_eq!(&bytes[0..4], b"RIFF");
    assert_eq!(&bytes[8..12], b"WAVE");
    assert!(!samples.is_empty());
}

#[test]
fn symphonia_decodes_m4a_aac_lc_in_process() {
    let prepared = prepare_native_conversion(&crate_fixture("tone_mono.m4a"))
        .expect("m4a/AAC-LC should decode via the in-process symphonia path");
    assert_prepared_16k_mono_wav(&prepared);
}

#[test]
fn symphonia_decodes_qta_container_in_process() {
    let prepared = prepare_native_conversion(&crate_fixture("tone_mono.qta"))
        .expect(".qta (mov/m4a container) should decode via the in-process symphonia path");
    assert_prepared_16k_mono_wav(&prepared);
}

#[test]
fn symphonia_decodes_mp3_in_process() {
    let prepared = prepare_native_conversion(&crate_fixture("tone_mono.mp3"))
        .expect("mp3 should decode via the in-process symphonia path");
    assert_prepared_16k_mono_wav(&prepared);
}

#[test]
fn symphonia_decodes_flac_in_process() {
    let prepared = prepare_native_conversion(&crate_fixture("tone_mono.flac"))
        .expect("flac should decode via the in-process symphonia path");
    assert_prepared_16k_mono_wav(&prepared);
}

#[test]
fn symphonia_decodes_ogg_vorbis_in_process() {
    let prepared = prepare_native_conversion(&crate_fixture("tone_stereo.ogg"))
        .expect("ogg/vorbis should decode (and downmix) via the in-process symphonia path");
    assert_prepared_16k_mono_wav(&prepared);
}

#[test]
fn symphonia_decodes_webm_vorbis_in_process() {
    let prepared = prepare_native_conversion(&crate_fixture("tone_stereo.webm"))
        .expect("webm/vorbis should decode via the in-process symphonia mkv path");
    assert_prepared_16k_mono_wav(&prepared);
}

#[test]
fn symphonia_decodes_m4a_alac_in_process() {
    let prepared = prepare_native_conversion(&crate_fixture("tone_mono_alac.m4a"))
        .expect("m4a/ALAC should decode via the in-process symphonia alac path");
    assert_prepared_16k_mono_wav(&prepared);
}

#[test]
#[cfg(target_os = "macos")]
fn webm_opus_names_the_codec_when_afconvert_also_cannot_open_it() {
    // symphonia can demux the webm container (mkv feature) but has no Opus
    // decoder, so this falls to afconvert -- which cannot parse WebM/Matroska
    // at all (unlike a bare Opus-in-Ogg file, see the ogg/opus test below).
    // The resulting error must name the detected codec rather than reading
    // like the file is simply corrupt.
    let error = prepare_native_conversion(&crate_fixture("tone_mono_opus.webm")).unwrap_err();

    let message = error.to_string();
    assert!(
        matches!(error, AudioPreparationError::ConversionFailed { tool, .. } if tool == "afconvert")
    );
    assert!(
        message.contains("Detected audio codec: Opus"),
        "error should name the detected codec: {message}"
    );
}

#[test]
#[cfg(not(target_os = "macos"))]
fn webm_opus_names_the_codec_in_the_missing_ffmpeg_hint() {
    // On a non-macOS host with no ffmpeg configured, there is no afconvert
    // fallback either: the `MissingFfmpeg` hint itself must name the detected
    // codec instead of a generic "this format" placeholder.
    let error = prepare_native_conversion(&crate_fixture("tone_mono_opus.webm")).unwrap_err();

    let message = error.to_string();
    assert!(matches!(error, AudioPreparationError::MissingFfmpeg { .. }));
    assert!(
        message.contains("Opus"),
        "missing-ffmpeg hint should name the detected codec: {message}"
    );
}

#[test]
fn symphonia_decodes_and_resamples_non_conformant_wav() {
    // A real (non-16k-mono) wav is not passed through blindly: it is decoded
    // and resampled via the same symphonia path as the other formats.
    let prepared = prepare_native_conversion(&crate_fixture("tone_stereo_44100.wav"))
        .expect("non-conformant wav should decode via the in-process symphonia path");
    assert_prepared_16k_mono_wav(&prepared);
}

#[test]
#[cfg(target_os = "macos")]
fn he_aac_falls_back_to_afconvert_when_symphonia_cannot_decode_it() {
    // HE-AAC (SBR) is outside what the enabled symphonia `aac` feature can
    // correctly decode (see `is_unsupported_aac_extension`); this must fall
    // back to afconvert, which macOS ships and can decode HE-AAC, rather than
    // silently emitting bandwidth-limited audio.
    let prepared = prepare_native_conversion(&crate_fixture("tone_heaac.m4a"))
        .expect("HE-AAC should fall back to afconvert and still succeed");
    assert_prepared_16k_mono_wav(&prepared);
}

#[test]
#[cfg(not(target_os = "macos"))]
fn opus_in_ogg_is_unsupported_by_symphonia_and_requires_ffmpeg() {
    // The enabled symphonia features intentionally exclude an Opus decoder
    // (see Cargo.toml), so a real Opus-in-Ogg file must fall back to an
    // external converter instead of silently mis-decoding, and this host has
    // neither ffmpeg configured nor (being non-macOS) an afconvert fallback.
    let error = prepare_native_conversion(&crate_fixture("tone_opus.ogg")).unwrap_err();
    assert!(matches!(error, AudioPreparationError::MissingFfmpeg { .. }));
}

#[test]
fn explicit_ffmpeg_bin_skips_symphonia_even_for_a_decodable_format() {
    // A bare (PATH-relative) command name is accepted by `resolve_conversion_tool`
    // without an existence check, so this deterministically proves the
    // in-process decoder was *not* tried: if it had been, this real, valid m4a
    // fixture would have decoded successfully instead of failing to spawn a
    // nonexistent tool.
    let options = AudioPreparationOptions::new(BackendKind::Native)
        .with_native_non_wav_conversion(true)
        .with_ffmpeg_bin(Some(PathBuf::from("openasr-test-nonexistent-ffmpeg")))
        .with_ffmpeg_bin_explicit(true);

    let error = prepare_audio_input(crate_fixture("tone_mono.m4a"), &options).unwrap_err();

    assert!(matches!(
        error,
        AudioPreparationError::ConversionSpawn { tool, .. } if tool == "ffmpeg"
    ));
}

fn write_test_wav(path: &Path, sample_rate: u32, channels: u16, bits_per_sample: u16, frames: u32) {
    write_wav(
        path,
        sample_rate,
        channels,
        bits_per_sample,
        frames,
        |index| {
            let phase = index as f32 / sample_rate as f32 * 440.0 * TAU;
            SampleBytes::I16((phase.sin() * i16::MAX as f32) as i16)
        },
    );
}

fn write_float_wav(path: &Path, sample_rate: u32, channels: u16, frames: u32) {
    write_wav(path, sample_rate, channels, 32, frames, |index| {
        let phase = index as f32 / sample_rate as f32 * 440.0 * TAU;
        SampleBytes::F32(phase.sin())
    });
}

fn write_wav<F>(
    path: &Path,
    sample_rate: u32,
    channels: u16,
    bits_per_sample: u16,
    frames: u32,
    mut sample_at: F,
) where
    F: FnMut(u32) -> SampleBytes,
{
    let audio_format = if bits_per_sample == 32 { 3_u16 } else { 1_u16 };
    let data_size = frames * channels as u32 * (bits_per_sample as u32 / 8);
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&(36 + data_size).to_le_bytes());
    bytes.extend_from_slice(b"WAVE");
    bytes.extend_from_slice(b"fmt ");
    bytes.extend_from_slice(&16_u32.to_le_bytes());
    bytes.extend_from_slice(&audio_format.to_le_bytes());
    bytes.extend_from_slice(&channels.to_le_bytes());
    bytes.extend_from_slice(&sample_rate.to_le_bytes());
    let byte_rate = sample_rate * channels as u32 * (bits_per_sample as u32 / 8);
    bytes.extend_from_slice(&byte_rate.to_le_bytes());
    let block_align = channels * (bits_per_sample / 8);
    bytes.extend_from_slice(&block_align.to_le_bytes());
    bytes.extend_from_slice(&bits_per_sample.to_le_bytes());
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&data_size.to_le_bytes());

    for index in 0..frames {
        match sample_at(index) {
            SampleBytes::I16(sample) => bytes.extend_from_slice(&sample.to_le_bytes()),
            SampleBytes::F32(sample) => bytes.extend_from_slice(&sample.to_le_bytes()),
        }
    }

    fs::write(path, bytes).unwrap();
}

enum SampleBytes {
    I16(i16),
    F32(f32),
}
