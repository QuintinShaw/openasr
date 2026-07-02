use std::path::{Path, PathBuf};

use crate::BackendKind;

pub(crate) const RECOGNIZED_EXTENSIONS: &[&str] =
    &["wav", "mp3", "mp4", "m4a", "webm", "flac", "ogg"];

#[derive(Debug, Clone, PartialEq)]
pub struct AudioInputInfo {
    pub path: PathBuf,
    pub extension: Option<String>,
    pub recognized_extension: bool,
    pub duration_seconds: Option<f64>,
    pub issues: Vec<AudioInputIssue>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AudioInputIssue {
    UnknownExtension(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudioPreparationOptions {
    pub backend: BackendKind,
    pub ffmpeg_bin: Option<PathBuf>,
    pub native_non_wav_requires_conversion: bool,
}

impl AudioPreparationOptions {
    pub fn new(backend: BackendKind) -> Self {
        Self {
            backend,
            ffmpeg_bin: None,
            native_non_wav_requires_conversion: false,
        }
    }

    pub fn with_ffmpeg_bin(mut self, ffmpeg_bin: Option<PathBuf>) -> Self {
        self.ffmpeg_bin = ffmpeg_bin;
        self
    }

    pub fn with_native_non_wav_conversion(mut self, enabled: bool) -> Self {
        self.native_non_wav_requires_conversion = enabled;
        self
    }
}

#[derive(Debug)]
pub struct PreparedAudioInput {
    pub(crate) original: AudioInputInfo,
    pub(crate) prepared_path: PathBuf,
    pub(crate) temp_dir: Option<tempfile::TempDir>,
}

impl PreparedAudioInput {
    pub fn path(&self) -> &Path {
        &self.prepared_path
    }

    pub fn original(&self) -> &AudioInputInfo {
        &self.original
    }

    pub fn is_converted(&self) -> bool {
        self.temp_dir.is_some()
    }
}
