use std::path::Path;

use super::{AudioInputInfo, AudioInputIssue, RECOGNIZED_EXTENSIONS, decode, probe_wav_duration};

pub(super) fn probe_audio_details(path: &Path) -> AudioInputInfo {
    let extension = normalized_extension(path);
    let recognized_extension = is_recognized_extension(extension.as_deref());
    let duration_seconds = wav_duration_if_supported(path, extension.as_deref());
    // WAV's fmt chunk names the source sample rate/channel count directly, no
    // decode needed. Other recognized formats (m4a/mp3/flac/ogg/...) only
    // reveal their true source format once actually decoded -- see
    // `prepare::try_symphonia_prepare`, which fills these fields in on this
    // same `AudioInputInfo` after a successful in-process decode. Left `None`
    // here for anything this probe cannot cheaply determine, never guessed.
    let (sample_rate_hz, channels) = wav_source_format_if_supported(path, extension.as_deref());

    AudioInputInfo {
        path: path.to_path_buf(),
        extension: extension.clone(),
        recognized_extension,
        duration_seconds,
        sample_rate_hz,
        channels,
        issues: collect_issues(extension, recognized_extension),
    }
}

fn normalized_extension(path: &Path) -> Option<String> {
    path.extension()
        .and_then(|extension| extension.to_str())
        .filter(|extension| !extension.is_empty())
        .map(|extension| extension.to_ascii_lowercase())
}

fn is_recognized_extension(extension: Option<&str>) -> bool {
    extension.is_some_and(|extension| RECOGNIZED_EXTENSIONS.contains(&extension))
}

fn wav_duration_if_supported(path: &Path, extension: Option<&str>) -> Option<f64> {
    if extension == Some("wav") {
        probe_wav_duration(path)
    } else {
        None
    }
}

fn wav_source_format_if_supported(
    path: &Path,
    extension: Option<&str>,
) -> (Option<u32>, Option<u16>) {
    if extension != Some("wav") {
        return (None, None);
    }
    match decode::probe_wav_pcm_shape(path) {
        Ok(Some(fmt)) => (Some(fmt.sample_rate), Some(fmt.channels)),
        _ => (None, None),
    }
}

fn collect_issues(extension: Option<String>, recognized_extension: bool) -> Vec<AudioInputIssue> {
    match (extension, recognized_extension) {
        (Some(extension), false) => vec![AudioInputIssue::UnknownExtension(extension)],
        _ => Vec::new(),
    }
}
