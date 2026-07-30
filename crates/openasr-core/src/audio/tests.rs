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

/// Regression guard for the WAVE_FORMAT_EXTENSIBLE admission mismatch: before
/// this fix, `wav_is_already_conformant` parsed the `fmt ` chunk's top-level
/// `audio_format` field directly (never unwrapping 0xFFFE), so *every*
/// extensible wav -- including the canonical, cbSize=22, full-GUID shape
/// macOS `afconvert -f WAVE` always emits -- was misclassified as
/// non-conformant and pushed through symphonia's stricter extensible parser
/// instead of the cheap passthrough this already-conformant file qualifies
/// for. `prepare_native_conversion` (not the bare `BackendKind::Native`
/// options the other passthrough tests above use) is required to actually
/// exercise `wav_is_already_conformant`: without
/// `with_native_non_wav_conversion(true)` the native backend passes every wav
/// through unconditionally, never reaching this check at all.
#[test]
fn wave_format_extensible_conformant_wav_passes_through_without_symphonia() {
    let temp = tempfile::tempdir().unwrap();
    let wav = temp.path().join("extensible.wav");
    fs::write(&wav, extensible_pcm16_mono_16k_wav(40, 22, 8)).unwrap();

    let prepared = prepare_native_conversion(&wav)
        .expect("an already-conformant WAVE_FORMAT_EXTENSIBLE wav must decode");

    assert_eq!(prepared.path(), wav.as_path());
    assert!(
        !prepared.is_converted(),
        "an already-conformant extensible wav must pass through, not go through symphonia"
    );
}

/// Same admission check, but for the shorter/non-canonical extensible shapes
/// real hardware recorders and conferencing systems emit: a `fmt ` chunk
/// shorter than symphonia's own extensible parser requires (which needs the
/// full 40-byte chunk, cbSize exactly 22, and the complete 16-byte SubFormat
/// GUID matched against the standard PCM GUID) but long enough (>= 26 bytes)
/// for `api::audio_io::parse_wav_fmt`'s lenient unwrap -- first two SubFormat
/// GUID bytes only -- to read the real PCM tag. Before this fix these were
/// misclassified as non-conformant, reached symphonia's stricter parser, and
/// (with no `fmt`/GUID match) fell through to the external ffmpeg/afconvert
/// chain -- a silent behavior change from passthrough to hard failure on
/// hosts without ffmpeg configured.
#[test]
fn wave_format_extensible_short_fmt_chunk_passes_through_without_symphonia() {
    let temp = tempfile::tempdir().unwrap();
    let wav = temp.path().join("extensible_short.wav");
    // 26-byte fmt content: the 16 fixed fields, a non-canonical cbSize (8,
    // not symphonia's required 22), and only the first two SubFormat GUID
    // bytes (the PCM tag) -- no channel mask, no full GUID.
    fs::write(&wav, extensible_pcm16_mono_16k_wav(26, 8, 8)).unwrap();

    let prepared = prepare_native_conversion(&wav)
        .expect("a short-fmt-chunk WAVE_FORMAT_EXTENSIBLE wav must decode");

    assert_eq!(prepared.path(), wav.as_path());
    assert!(
        !prepared.is_converted(),
        "a short-fmt-chunk extensible wav must pass through, not go through symphonia"
    );
}

/// Builds a WAVE_FORMAT_EXTENSIBLE 16 kHz mono PCM16 wav with an explicit
/// `fmt` chunk content length and `cbSize` value, to cover both the
/// canonical (40-byte, cbSize=22) shape and shorter, non-canonical shapes.
/// Only the PCM SubFormat GUID's first two bytes are ever written (the part
/// `api::audio_io::parse_wav_fmt` actually reads); any GUID bytes past
/// `fmt_content_len` are simply absent, matching a real short fmt chunk
/// rather than a truncated-but-intended-full one.
fn extensible_pcm16_mono_16k_wav(fmt_content_len: usize, cb_size: u16, frames: u32) -> Vec<u8> {
    assert!(
        fmt_content_len >= 26,
        "fixture must be long enough for the SubFormat GUID's leading PCM tag"
    );
    let data_size = frames * 2;
    let mut fmt_content = Vec::with_capacity(fmt_content_len);
    fmt_content.extend_from_slice(&0xFFFE_u16.to_le_bytes()); // audio_format (extensible)
    fmt_content.extend_from_slice(&1_u16.to_le_bytes()); // channels
    fmt_content.extend_from_slice(&16_000_u32.to_le_bytes()); // sample rate
    fmt_content.extend_from_slice(&32_000_u32.to_le_bytes()); // byte rate
    fmt_content.extend_from_slice(&2_u16.to_le_bytes()); // block align
    fmt_content.extend_from_slice(&16_u16.to_le_bytes()); // bits per sample
    fmt_content.extend_from_slice(&cb_size.to_le_bytes()); // cbSize
    while fmt_content.len() < fmt_content_len {
        // First two SubFormat GUID bytes are the PCM tag (0x0001 LE); the
        // rest of this padding (channel mask + remaining GUID bytes) is
        // irrelevant to `parse_wav_fmt`'s lenient check.
        if fmt_content.len() == 24 {
            fmt_content.extend_from_slice(&1_u16.to_le_bytes());
        } else {
            fmt_content.push(0);
        }
    }
    fmt_content.truncate(fmt_content_len);

    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&(4 + 8 + fmt_content_len as u32 + 8 + data_size).to_le_bytes());
    bytes.extend_from_slice(b"WAVE");
    bytes.extend_from_slice(b"fmt ");
    bytes.extend_from_slice(&(fmt_content_len as u32).to_le_bytes());
    bytes.extend_from_slice(&fmt_content);
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&data_size.to_le_bytes());
    for index in 0..frames {
        bytes.extend_from_slice(&(index as i16).to_le_bytes());
    }
    bytes
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

    assert_prepared_16k_mono_audio(&prepared);
}

/// The in-process symphonia decode path (the one this m4a/AAC-LC fixture
/// takes, see the test above) hands back samples already resident in memory
/// instead of writing a prepared WAV to a temp dir and immediately
/// re-reading it back -- assert both halves of that contract directly.
#[test]
fn symphonia_in_memory_decode_never_writes_a_temp_wav_to_disk() {
    let prepared = prepare_native_conversion(&crate_fixture("tone_mono.m4a"))
        .expect("m4a/AAC-LC should decode via the in-process symphonia path");

    assert!(
        prepared.temp_dir.is_none(),
        "the in-memory decode path must not create a temp dir"
    );
    let samples = prepared
        .samples()
        .expect("in-process symphonia decode should hand back in-memory samples");
    assert!(!samples.is_empty());
    assert!(prepared.is_converted());
}

/// `tests/fixtures/*.{m4a,mp3,flac,ogg,opus,webm,qta,wav}` are all
/// synthesized, not recorded audio, so they can be regenerated if a fixture
/// is lost or a new one is needed. Recipe (from a 0.5s 440 Hz mono/stereo
/// sine `source.wav` generated via `ffmpeg -f lavfi -i
/// "sine=frequency=440:duration=0.5" -ar 16000 -ac <1|2> source.wav`):
///
/// ```text
/// ffmpeg -i source.wav -c:a alac tone_mono_alac.m4a
/// ffmpeg -i source_stereo.wav -c:a vorbis -strict -2 -ac 2 tone_stereo.webm
/// ffmpeg -i source.wav -c:a libopus tone_mono_opus.webm
/// ffmpeg -i source.wav -c:a libopus tone_opus.ogg
/// ffmpeg -i source.wav -c:a libopus tone.opus
/// ffmpeg -i source.wav -c:a adpcm_ms tone_mono_adpcm_ms.wav
/// ```
///
/// `malformed_vint_zero.webm` is not synthesized audio at all -- see
/// `symphonia_decode::tests::malformed_webm_vint_zero_bytes` for its exact
/// (adversarial, hand-crafted) bytes and why they trigger a third-party
/// demuxer bug.
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

/// Asserts `prepared` is a converted, non-empty 16 kHz mono decode,
/// regardless of which conversion path produced it: the in-process symphonia
/// path hands back samples already resident in memory
/// ([`PreparedAudioInput::samples`]), while the external ffmpeg/afconvert
/// conversion path still writes a real prepared WAV to disk.
fn assert_prepared_16k_mono_audio(prepared: &PreparedAudioInput) {
    assert!(prepared.is_converted());
    let samples: Vec<f32> = match prepared.samples() {
        Some(samples) => samples.to_vec(),
        None => {
            let bytes = fs::read(prepared.path()).unwrap();
            assert_eq!(&bytes[0..4], b"RIFF");
            assert_eq!(&bytes[8..12], b"WAVE");
            crate::api::audio_io::load_wav_16khz_mono_f32_v0(prepared.path(), "test", "test")
                .expect("prepared output must be a valid 16 kHz mono WAV")
        }
    };
    assert!(!samples.is_empty());
}

#[test]
fn symphonia_decodes_m4a_aac_lc_in_process() {
    let prepared = prepare_native_conversion(&crate_fixture("tone_mono.m4a"))
        .expect("m4a/AAC-LC should decode via the in-process symphonia path");
    assert_prepared_16k_mono_audio(&prepared);
}

#[test]
fn symphonia_decodes_qta_container_in_process() {
    let prepared = prepare_native_conversion(&crate_fixture("tone_mono.qta"))
        .expect(".qta (mov/m4a container) should decode via the in-process symphonia path");
    assert_prepared_16k_mono_audio(&prepared);
}

#[test]
fn symphonia_decodes_mp3_in_process() {
    let prepared = prepare_native_conversion(&crate_fixture("tone_mono.mp3"))
        .expect("mp3 should decode via the in-process symphonia path");
    assert_prepared_16k_mono_audio(&prepared);
}

#[test]
fn symphonia_decodes_flac_in_process() {
    let prepared = prepare_native_conversion(&crate_fixture("tone_mono.flac"))
        .expect("flac should decode via the in-process symphonia path");
    assert_prepared_16k_mono_audio(&prepared);
}

#[test]
fn symphonia_decodes_ogg_vorbis_in_process() {
    let prepared = prepare_native_conversion(&crate_fixture("tone_stereo.ogg"))
        .expect("ogg/vorbis should decode (and downmix) via the in-process symphonia path");
    assert_prepared_16k_mono_audio(&prepared);
}

#[test]
fn symphonia_decodes_webm_vorbis_in_process() {
    let prepared = prepare_native_conversion(&crate_fixture("tone_stereo.webm"))
        .expect("webm/vorbis should decode via the in-process symphonia mkv path");
    assert_prepared_16k_mono_audio(&prepared);
}

#[test]
fn symphonia_decodes_m4a_alac_in_process() {
    let prepared = prepare_native_conversion(&crate_fixture("tone_mono_alac.m4a"))
        .expect("m4a/ALAC should decode via the in-process symphonia alac path");
    assert_prepared_16k_mono_audio(&prepared);
}

/// Bare ADTS `.aac` (no m4a/mp4 container) -- what WeChat and many other
/// recorders/voice-memo apps actually emit, as opposed to `.m4a`. Symphonia's
/// `AdtsReader` (registered by the already-enabled `aac` feature) demuxes
/// this directly; no new feature or dependency needed.
#[test]
fn symphonia_decodes_bare_adts_aac_in_process() {
    let prepared = prepare_native_conversion(&crate_fixture("tone_mono.aac"))
        .expect("bare ADTS .aac should decode via the in-process symphonia path");
    assert_prepared_16k_mono_audio(&prepared);
}

/// `.m4b` (audiobook) is the same ISO/MP4 container as `.m4a`/`.mp4` under a
/// different extension -- symphonia's `isomp4` reader already lists it
/// (`IsoMp4Reader::query`), so this only needed the extension added to
/// `RECOGNIZED_EXTENSIONS`, no code or dependency change.
#[test]
fn symphonia_decodes_m4b_audiobook_container_in_process() {
    let prepared = prepare_native_conversion(&crate_fixture("tone_mono.m4b"))
        .expect(".m4b (isomp4 container) should decode via the in-process symphonia path");
    assert_prepared_16k_mono_audio(&prepared);
}

/// AIFF (PCM), decoded by symphonia's `aiff` feature -- which reuses the
/// already-vendored `symphonia-format-riff` crate (the same one that reads
/// `.wav`), so enabling it added no new dependency.
#[test]
fn symphonia_decodes_aiff_in_process() {
    let prepared = prepare_native_conversion(&crate_fixture("tone_mono.aiff"))
        .expect("aiff should decode via the in-process symphonia path");
    assert_prepared_16k_mono_audio(&prepared);
}

/// Apple's Core Audio Format, decoded by symphonia's `caf` feature
/// (`symphonia-format-caf`, pulled in new for this).
#[test]
fn symphonia_decodes_caf_in_process() {
    let prepared = prepare_native_conversion(&crate_fixture("tone_mono.caf"))
        .expect("caf should decode via the in-process symphonia path");
    assert_prepared_16k_mono_audio(&prepared);
}

/// Windows Media Audio: symphonia has no ASF/WMA demuxer at all, so a real
/// `.wma` upload must fall through to the external ffmpeg/afconvert
/// conversion chain (same shape as HE-AAC) rather than being rejected
/// upfront as an unrecognized extension.
#[test]
#[cfg(target_os = "macos")]
fn wma_falls_back_to_afconvert_or_succeeds_without_symphonia_support() {
    // `tone_mono.wma` is a real ffmpeg-encoded (wmav2) file; afconvert cannot
    // decode WMA, so without ffmpeg configured this must land on a typed
    // conversion failure -- proof the extension reached the external-converter
    // step instead of a bare "unsupported extension" rejection.
    let error = prepare_native_conversion(&crate_fixture("tone_mono.wma")).unwrap_err();
    assert!(
        matches!(&error, AudioPreparationError::ConversionFailed { tool, .. } if tool == "afconvert"),
        "a real .wma file must reach (and fail inside) the external converter, not be \
         rejected as an unrecognized extension: {error}"
    );
}

#[test]
#[cfg(not(target_os = "macos"))]
fn wma_reaches_missing_ffmpeg_without_symphonia_support() {
    let error = prepare_native_conversion(&crate_fixture("tone_mono.wma")).unwrap_err();
    assert!(
        matches!(error, AudioPreparationError::MissingFfmpeg { .. }),
        "a real .wma file must reach the ffmpeg-required path, not be rejected as an \
         unrecognized extension: {error}"
    );
}

/// Beyond routing, this proves ffmpeg genuinely decodes the format when it is
/// configured: `tone_mono.wma` is a real ffmpeg-encoded (wmav2) file, and this
/// workspace's ffmpeg links a WMA decoder, so the external conversion must
/// fully succeed, not merely be attempted.
#[test]
fn wma_converts_successfully_when_ffmpeg_is_available() {
    let ffmpeg_available = std::process::Command::new("ffmpeg")
        .arg("-version")
        .output()
        .is_ok_and(|output| output.status.success());
    if !ffmpeg_available {
        eprintln!("ffmpeg not on PATH; skipping the .wma external-conversion test");
        return;
    }
    let options = AudioPreparationOptions::new(BackendKind::Native)
        .with_native_non_wav_conversion(true)
        .with_ffmpeg_bin(Some(PathBuf::from("ffmpeg")))
        .with_ffmpeg_bin_explicit(true);

    let prepared = prepare_audio_input(crate_fixture("tone_mono.wma"), &options)
        .expect("a real .wma file must convert via the external ffmpeg fallback");
    assert_prepared_16k_mono_audio(&prepared);
}

/// AMR (mono voice codec, common on older phones and some IM voice
/// messages): like WMA, symphonia has no demuxer for it at all, so this only
/// ever reaches ffmpeg. `tone.amr` is a real, decodable AMR-NB bitstream
/// (silent content -- there is no local AMR encoder to synthesize a tone
/// with -- but real frames a real decoder accepts), and this workspace's
/// ffmpeg build genuinely links an AMR-NB decoder, so this asserts the full
/// external conversion actually succeeds when ffmpeg is configured.
#[test]
fn amr_converts_successfully_when_ffmpeg_is_available() {
    let ffmpeg_available = std::process::Command::new("ffmpeg")
        .arg("-version")
        .output()
        .is_ok_and(|output| output.status.success());
    if !ffmpeg_available {
        eprintln!("ffmpeg not on PATH; skipping the .amr external-conversion test");
        return;
    }
    let options = AudioPreparationOptions::new(BackendKind::Native)
        .with_native_non_wav_conversion(true)
        .with_ffmpeg_bin(Some(PathBuf::from("ffmpeg")))
        .with_ffmpeg_bin_explicit(true);

    let prepared = prepare_audio_input(crate_fixture("tone.amr"), &options)
        .expect("a real, decodable .amr file must convert via the external ffmpeg fallback");
    assert_prepared_16k_mono_audio(&prepared);
}

#[test]
#[cfg(target_os = "macos")]
fn malformed_webm_falls_back_to_typed_error_instead_of_panicking() {
    // `malformed_vint_zero.webm` is a 16-byte fixture (the minimum symphonia's
    // probe needs to recognize the mkv/webm marker at all) whose EBML header
    // size field is the single byte `0x00`. `symphonia-format-mkv 0.5.5`'s
    // vint reader computes `7 - byte.leading_zeros()` without checking that
    // `leading_zeros() <= 7`, so this underflows and panics in a
    // debug/overflow-checked build (verified via a standalone repro against
    // the crate directly). This is exactly the kind of untrusted, arbitrary
    // upload webm is a reachable surface for, so the panic-free trust-
    // boundary invariant in AGENTS.md requires it be caught and turned into a
    // typed error, not crash the process or be misreported as a corrupt file.
    let error = prepare_native_conversion(&crate_fixture("malformed_vint_zero.webm")).unwrap_err();

    let message = error.to_string();
    assert!(
        matches!(error, AudioPreparationError::ConversionFailed { tool, .. } if tool == "afconvert")
    );
    assert!(
        message.contains("internal error while inspecting this file"),
        "error should report a parser-internal-error, not a bare tool failure: {message}"
    );
}

#[test]
#[cfg(not(target_os = "macos"))]
fn malformed_webm_falls_back_to_typed_error_instead_of_panicking() {
    // See the macOS counterpart above for the vint-underflow panic this
    // guards against. Without ffmpeg configured and no afconvert fallback,
    // this must land on `MissingFfmpeg` (not a panic), with the hint naming
    // the parser-internal-error condition.
    let error = prepare_native_conversion(&crate_fixture("malformed_vint_zero.webm")).unwrap_err();

    let message = error.to_string();
    assert!(matches!(error, AudioPreparationError::MissingFfmpeg { .. }));
    assert!(
        message.contains("malformed or corrupted"),
        "missing-ffmpeg hint should report the parser-internal-error condition: {message}"
    );
}

#[test]
fn symphonia_decodes_webm_opus_in_process() {
    // The mkv/webm demuxer surfaces the Opus packets and the bundled libopus
    // decodes them in-process -- no ffmpeg/afconvert fallback involved.
    let prepared = prepare_native_conversion(&crate_fixture("tone_mono_opus.webm"))
        .expect("webm/opus should decode via the in-process symphonia + libopus path");
    assert_decoded_opus_tone(&prepared);
}

#[test]
fn symphonia_decodes_and_resamples_non_conformant_wav() {
    // A real (non-16k-mono) wav is not passed through blindly: it is decoded
    // and resampled via the same symphonia path as the other formats.
    let prepared = prepare_native_conversion(&crate_fixture("tone_stereo_44100.wav"))
        .expect("non-conformant wav should decode via the in-process symphonia path");
    assert_prepared_16k_mono_audio(&prepared);
}

#[test]
fn symphonia_decodes_ms_adpcm_wav_in_process() {
    // MS-ADPCM (WAVE_FORMAT_ADPCM 0x0002) is a common output of dictaphones,
    // older recording software, and conferencing systems (see #159); the
    // `wav` demuxer identifies this format tag regardless of build features,
    // but only the `adpcm` symphonia feature links a decoder for it. With
    // that feature enabled this decodes entirely in-process, the same as any
    // other non-conformant wav, instead of the old behavior of silently
    // passing the undecoded ADPCM bytes straight through.
    let prepared = prepare_native_conversion(&crate_fixture("tone_mono_adpcm_ms.wav"))
        .expect("MS-ADPCM wav should decode via the in-process symphonia adpcm path");
    assert_prepared_16k_mono_audio(&prepared);
}

/// Regression guard for the structural asymmetry this change fixes: a `.wav`
/// symphonia cannot parse at all (corrupt/garbage bytes, not merely an
/// unsupported-but-identifiable codec) used to be handed back byte-for-byte
/// untouched (`prepare_external_input`'s old wav-only leniency), which
/// silently produced "prepared" audio the rest of the pipeline cannot
/// actually use and only failed much later with a generic, misleading
/// "expected 16 kHz mono PCM" error. It must fail closed here instead, going
/// through the exact same external-converter fallback (and failing there)
/// that any other undecodable input already used -- proving passthrough is
/// now reachable only for an already-conformant wav.
#[test]
#[cfg(target_os = "macos")]
fn garbage_bytes_wav_fails_closed_instead_of_passing_through() {
    let temp = tempfile::tempdir().unwrap();
    let wav = temp.path().join("garbage.wav");
    fs::write(&wav, garbage_bytes()).unwrap();

    let error = prepare_native_conversion(&wav).unwrap_err();

    assert!(
        matches!(&error, AudioPreparationError::ConversionFailed { tool, .. } if tool == "afconvert"),
        "garbage-byte wav must fail closed through the external converter, not pass through: {error}"
    );
}

#[test]
#[cfg(not(target_os = "macos"))]
fn garbage_bytes_wav_fails_closed_instead_of_passing_through() {
    let temp = tempfile::tempdir().unwrap();
    let wav = temp.path().join("garbage.wav");
    fs::write(&wav, garbage_bytes()).unwrap();

    let error = prepare_native_conversion(&wav).unwrap_err();

    assert!(
        matches!(&error, AudioPreparationError::MissingFfmpeg { .. }),
        "garbage-byte wav must fail closed with a typed error, not pass through: {error}"
    );
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
    assert_prepared_16k_mono_audio(&prepared);
}

#[test]
fn symphonia_decodes_opus_in_ogg_in_process() {
    // A real Opus-in-Ogg file decodes entirely in-process (symphonia ogg
    // demuxer + bundled libopus), with the RFC 7845 pre-skip removed and the
    // Ogg granule end-trim applied.
    let prepared = prepare_native_conversion(&crate_fixture("tone_opus.ogg"))
        .expect("ogg/opus should decode via the in-process symphonia + libopus path");
    assert_decoded_opus_tone(&prepared);
}

#[test]
fn symphonia_decodes_bare_opus_file_in_process() {
    // `.opus` is the conventional extension for a bare Ogg-Opus file; it must
    // be recognized and take the same in-process path.
    let prepared = prepare_native_conversion(&crate_fixture("tone.opus"))
        .expect(".opus should decode via the in-process symphonia + libopus path");
    assert_decoded_opus_tone(&prepared);
}

#[test]
fn opus_extension_is_recognized() {
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("voice.opus");
    fs::write(&input, b"not a real ogg stream").unwrap();

    let info = probe_audio_input(&input).unwrap();

    assert_eq!(info.extension.as_deref(), Some("opus"));
    assert!(info.recognized_extension);
    assert!(info.issues.is_empty());
}

/// Horizontal coverage for every extension added alongside `.aac`: each must
/// be recognized at the probe stage regardless of whether decoding it is an
/// in-process symphonia path (`aac`/`m4b`/`aiff`/`aif`/`aifc`/`caf`) or an
/// external ffmpeg-only fallback (`wma`/`amr`) -- see `RECOGNIZED_EXTENSIONS`'s
/// doc comment in `audio/types.rs` for which is which.
#[test]
fn newly_supported_extensions_are_all_recognized_at_the_probe_stage() {
    for extension in ["aac", "m4b", "aiff", "aif", "aifc", "caf", "wma", "amr"] {
        let temp = tempfile::tempdir().unwrap();
        let input = temp.path().join(format!("voice.{extension}"));
        fs::write(&input, b"not real audio bytes").unwrap();

        let info = probe_audio_input(&input).unwrap();

        assert_eq!(info.extension.as_deref(), Some(extension));
        assert!(
            info.recognized_extension,
            "{extension} should be recognized"
        );
        assert!(info.issues.is_empty());
    }
}

/// Writes a copy of the `tone.opus` fixture cut off mid-stream (60%) and
/// returns the path plus the guard that keeps the temp dir alive.
fn write_truncated_opus_fixture() -> (tempfile::TempDir, PathBuf) {
    let full = fs::read(crate_fixture("tone.opus")).unwrap();
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("truncated.opus");
    fs::write(&path, &full[..full.len() * 3 / 5]).unwrap();
    (temp, path)
}

/// A file cut off mid-stream must fail closed, never panic or fabricate
/// audio. Symphonia's ogg demuxer needs the complete stream (a truncated
/// file already fails its probe), so the in-process path hands this to the
/// external converter chain. On macOS that means afconvert, which exits 0
/// but writes an *empty* WAV for a truncated Ogg-Opus stream (a CoreAudio
/// leniency quirk -- no samples, no fabricated audio) or fails with a typed
/// conversion error; both outcomes are acceptable here.
#[test]
#[cfg(target_os = "macos")]
fn truncated_opus_is_handled_without_panicking() {
    let (_temp, path) = write_truncated_opus_fixture();

    match prepare_native_conversion(&path) {
        Ok(_) | Err(AudioPreparationError::ConversionFailed { .. }) => {}
        Err(error) => panic!("truncated opus must fail closed with a typed error: {error}"),
    }
}

/// The non-macOS counterpart: with no ffmpeg configured and no afconvert
/// fallback, the typed `MissingFfmpeg` surfaces instead of a panic.
#[test]
#[cfg(not(target_os = "macos"))]
fn truncated_opus_is_handled_without_panicking() {
    let (_temp, path) = write_truncated_opus_fixture();

    let error = prepare_native_conversion(&path).unwrap_err();
    assert!(
        matches!(error, AudioPreparationError::MissingFfmpeg { .. }),
        "truncated opus must fail closed with a typed error: {error}"
    );
}

/// Corrupt Opus bytes must fail closed with a typed error (never a panic,
/// never fabricated audio): on macOS the fallback reaches afconvert, which
/// cannot open the corrupt stream; elsewhere there is no converter at all and
/// the typed `MissingFfmpeg` surfaces. Either way the error is precise.
#[test]
#[cfg(target_os = "macos")]
fn corrupt_opus_fails_closed_with_a_typed_error() {
    assert_corrupt_opus_is_typed_error(corrupt_opus_head_bytes());
    assert_corrupt_opus_is_typed_error(garbage_bytes());
}

#[test]
#[cfg(not(target_os = "macos"))]
fn corrupt_opus_fails_closed_with_a_typed_error() {
    assert_corrupt_opus_is_typed_error(corrupt_opus_head_bytes());
    assert_corrupt_opus_is_typed_error(garbage_bytes());
}

fn assert_corrupt_opus_is_typed_error(bytes: Vec<u8>) {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("corrupt.opus");
    fs::write(&path, &bytes).unwrap();

    let error = prepare_native_conversion(&path).unwrap_err();
    let message = error.to_string();

    #[cfg(target_os = "macos")]
    assert!(
        matches!(&error, AudioPreparationError::ConversionFailed { tool, .. } if tool == "afconvert"),
        "corrupt opus must fail closed through the external converter: {message}"
    );
    #[cfg(not(target_os = "macos"))]
    assert!(
        matches!(&error, AudioPreparationError::MissingFfmpeg { .. }),
        "corrupt opus must fail closed with a typed error: {message}"
    );
}

/// The real `tone.opus` fixture with its `OpusHead` magic scrambled: the Ogg
/// container still parses, but the Opus stream can no longer be identified.
fn corrupt_opus_head_bytes() -> Vec<u8> {
    let mut bytes = fs::read(crate_fixture("tone.opus")).unwrap();
    let magic = bytes
        .windows(8)
        .position(|window| window == b"OpusHead")
        .expect("the fixture must contain an OpusHead packet");
    bytes[magic] = b'X';
    bytes
}

fn garbage_bytes() -> Vec<u8> {
    vec![0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x11, 0x22, 0x33]
}

/// Asserts `prepared` is the in-process decode of one of the 0.5 s 440 Hz
/// mono Opus tone fixtures: in-memory samples (no temp WAV round trip), about
/// 0.5 s at 16 kHz, non-silent, with the true source format (48 kHz -- Opus
/// always decodes at 48 kHz -- mono) reported for diagnostics. Tolerances,
/// not exact sample counts: encoder padding, the pre-skip trim, and the
/// resampler's group delay all move the count by small amounts (#176).
fn assert_decoded_opus_tone(prepared: &PreparedAudioInput) {
    assert!(prepared.is_converted());
    assert!(
        prepared.temp_dir.is_none(),
        "the in-process opus decode path must not create a temp dir"
    );
    let samples = prepared
        .samples()
        .expect("in-process opus decode should hand back in-memory samples");

    // 0.5 s of content: exactly 24,000 samples at the decoder's 48 kHz --
    // pinned pre-resample by `opus_tone_decodes_to_the_exact_rfc7845_sample_
    // count` -- resampled to ~8,000 at 16 kHz. The slack here is only the
    // FFT resampler's group delay (~2%, shared by every format this module
    // resamples, pulls the Ogg/WebM counts short) plus at most one packet
    // (~120 ms, pushes the webm count long: its millisecond timecodes skip
    // the Ogg end-trim); the *decode* length itself is exact and asserted
    // at 48 kHz.
    let expected = 8_000_usize;
    assert!(
        samples.len().abs_diff(expected) <= 1_000,
        "expected ~{expected} samples, got {}",
        samples.len()
    );

    // lavfi's `sine` source defaults to amplitude 1/8, so the tone's RMS is
    // ~0.088 (verified against ffmpeg's own decode of the same fixture); the
    // range only has to prove the audio is real, not silent or clipped.
    let rms =
        (samples.iter().map(|sample| sample * sample).sum::<f32>() / samples.len() as f32).sqrt();
    assert!(
        (0.03..=0.3).contains(&rms),
        "decoded tone should be audible, not silent or clipped: RMS {rms}"
    );

    assert_eq!(prepared.original().sample_rate_hz, Some(48_000));
    assert_eq!(prepared.original().channels, Some(1));
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

/// The explicit-ffmpeg escape hatch skips the in-process decode attempt
/// entirely (see the test above), so if the configured "ffmpeg" then fails to
/// convert a wav codec this build *does* decode in-process (MS-ADPCM here),
/// the resulting error must still name the detected codec instead of a bare,
/// unhelpful tool failure -- this is the whole point of
/// `symphonia_decode::probe_codec_label` / `Diagnostic::Codec`. `/usr/bin/
/// false` stands in for a real-but-broken ffmpeg binary: it exists (so this
/// reaches `ConversionFailed`, not a spawn error) and always exits non-zero
/// regardless of its arguments.
#[test]
#[cfg(unix)]
fn explicit_ffmpeg_bin_failure_reports_the_detected_codec_in_the_error() {
    let options = AudioPreparationOptions::new(BackendKind::Native)
        .with_native_non_wav_conversion(true)
        .with_ffmpeg_bin(Some(PathBuf::from("/usr/bin/false")))
        .with_ffmpeg_bin_explicit(true);

    let error = prepare_audio_input(crate_fixture("tone_mono_adpcm_ms.wav"), &options).unwrap_err();

    let message = error.to_string();
    assert!(matches!(
        error,
        AudioPreparationError::ConversionFailed { .. }
    ));
    assert!(
        message.contains("MS ADPCM"),
        "the error should name the detected codec: {message}"
    );
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
