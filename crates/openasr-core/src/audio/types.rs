use std::path::{Path, PathBuf};

use crate::BackendKind;

// `qta` is macOS QuickTime Player's audio recording extension -- a MOV
// container (ffmpeg's `mov,mp4,m4a,3gp,3g2,mj2` demuxer probes and decodes it
// like any other MOV/M4A file, no special handling needed beyond recognizing
// the extension).
pub(crate) const RECOGNIZED_EXTENSIONS: &[&str] =
    &["wav", "mp3", "mp4", "m4a", "webm", "flac", "ogg", "qta"];

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
    /// Whether `ffmpeg_bin` (when set) came from an explicit user choice --
    /// `--ffmpeg-bin`, `OPENASR_FFMPEG_BIN`, or `media.ffmpeg_bin` in config --
    /// as opposed to auto-discovering `ffmpeg` on `PATH`. The in-process
    /// symphonia decode path is the default for recognized non-WAV formats and
    /// is only skipped in favor of external conversion when this is `true`:
    /// a system that merely happens to have ffmpeg on PATH should not disable
    /// it (see `crates/openasr-core/src/audio/prepare.rs`).
    pub ffmpeg_bin_explicit: bool,
    pub native_non_wav_requires_conversion: bool,
}

impl AudioPreparationOptions {
    pub fn new(backend: BackendKind) -> Self {
        Self {
            backend,
            ffmpeg_bin: None,
            ffmpeg_bin_explicit: false,
            native_non_wav_requires_conversion: false,
        }
    }

    pub fn with_ffmpeg_bin(mut self, ffmpeg_bin: Option<PathBuf>) -> Self {
        self.ffmpeg_bin = ffmpeg_bin;
        self
    }

    /// Marks `ffmpeg_bin` as an explicit user choice rather than a PATH
    /// auto-discovery result. No-op if `ffmpeg_bin` is `None`.
    pub fn with_ffmpeg_bin_explicit(mut self, explicit: bool) -> Self {
        self.ffmpeg_bin_explicit = explicit && self.ffmpeg_bin.is_some();
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
