use std::path::Path;

use super::{AudioInputInfo, AudioInputIssue, RECOGNIZED_EXTENSIONS, probe_wav_duration};

pub(super) fn probe_audio_details(path: &Path) -> AudioInputInfo {
    let extension = normalized_extension(path);
    let recognized_extension = is_recognized_extension(extension.as_deref());
    let duration_seconds = wav_duration_if_supported(path, extension.as_deref());

    AudioInputInfo {
        path: path.to_path_buf(),
        extension: extension.clone(),
        recognized_extension,
        duration_seconds,
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

fn collect_issues(extension: Option<String>, recognized_extension: bool) -> Vec<AudioInputIssue> {
    match (extension, recognized_extension) {
        (Some(extension), false) => vec![AudioInputIssue::UnknownExtension(extension)],
        _ => Vec::new(),
    }
}
