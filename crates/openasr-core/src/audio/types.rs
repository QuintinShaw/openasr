use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

use crate::BackendKind;

// `qta` is macOS QuickTime Player's audio recording extension -- a MOV
// container (ffmpeg's `mov,mp4,m4a,3gp,3g2,mj2` demuxer probes and decodes it
// like any other MOV/M4A file, no special handling needed beyond recognizing
// the extension).
//
// Every extension here is *reachable*, not necessarily decodable in-process:
// `webm` in particular is a container, not a codec, and most real-world
// `.webm`/`.ogg` audio uses Opus, which symphonia has never shipped a decoder
// for (still true as of symphonia 0.6). Those files fall through to the
// external ffmpeg/afconvert conversion chain in `prepare.rs` -- with ffmpeg
// on PATH they still transcribe; without it, the error names the detected
// codec instead of pretending the file is corrupt (see
// `symphonia_decode::probe_codec_label` and `prepare::codec_note`).
pub(crate) const RECOGNIZED_EXTENSIONS: &[&str] =
    &["wav", "mp3", "mp4", "m4a", "webm", "flac", "ogg", "qta"];

#[derive(Debug, Clone, PartialEq)]
pub struct AudioInputInfo {
    pub path: PathBuf,
    pub extension: Option<String>,
    pub recognized_extension: bool,
    pub duration_seconds: Option<f64>,
    /// The *source* file's sample rate in Hz, before any resampling this
    /// crate's normalization pipeline applies -- e.g. `8000` for a phone-call
    /// recording or `44100`/`48000` for a typical music-app export. `None`
    /// when the source rate could not be determined (an unrecognized
    /// extension, or a format this crate does not decode in-process --
    /// callers must not fabricate a value in that case; see
    /// `crate::api::backend::request_context`'s privacy/honesty contract).
    pub sample_rate_hz: Option<u32>,
    /// The source file's channel count, before this pipeline's mono downmix.
    /// Same "probed, never fabricated" contract as `sample_rate_hz`.
    pub channels: Option<u16>,
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

/// The decoded/prepared audio a [`PreparedAudioInput`] hands to downstream
/// consumers. The WAV-passthrough and external ffmpeg/afconvert conversion
/// paths (`audio::prepare`) still route through a real file on disk -- either
/// the untouched original or the external tool's output -- because that is
/// the cheapest or only option there. The in-process symphonia decode path
/// (the default for m4a/mp3/flac/ogg/webm/non-conformant wav) instead hands
/// back the fully-decoded 16 kHz mono f32 samples it already has resident in
/// memory, so callers never have to write them to a temporary WAV and
/// immediately re-read + re-parse it back into the exact same samples.
///
/// Deliberately `Arc<Vec<f32>>`, not `Arc<[f32]>`: wrapping the already-owned
/// `Vec<f32>` the symphonia decode produced is a plain pointer move (`Arc::new`
/// only allocates the small refcount header), whereas `Arc<[f32]>::from(vec)`
/// would copy every sample into a new combined allocation. It also lets the
/// sole consumer (`native_transcribe::resolve_prepared_audio_samples`) reclaim
/// the exact same `Vec<f32>` via `Arc::try_unwrap` with zero copy whenever it
/// is the last handle, instead of always cloning the samples out from behind
/// a shared reference.
#[derive(Debug)]
pub(crate) enum PreparedAudioSamples {
    Path(PathBuf),
    InMemory(Arc<Vec<f32>>),
}

#[derive(Debug)]
pub struct PreparedAudioInput {
    pub(crate) original: AudioInputInfo,
    pub(crate) samples: PreparedAudioSamples,
    pub(crate) temp_dir: Option<tempfile::TempDir>,
}

impl PreparedAudioInput {
    /// A path identifying this prepared input: the real WAV to read bytes
    /// from for the WAV-passthrough and external-conversion paths (see
    /// [`Self::samples`]), or -- for the in-process symphonia decode path,
    /// which writes nothing to disk -- the *original* source file, purely
    /// for display/logging. Callers that need the decoded audio itself
    /// should prefer [`Self::samples`] and only fall back to reading this
    /// path when it returns `None`.
    pub fn path(&self) -> &Path {
        match &self.samples {
            PreparedAudioSamples::Path(path) => path,
            PreparedAudioSamples::InMemory(_) => &self.original.path,
        }
    }

    pub fn original(&self) -> &AudioInputInfo {
        &self.original
    }

    /// Ready-to-decode 16 kHz mono f32 samples already resident in memory,
    /// when the in-process symphonia decode path produced them directly.
    /// `None` for the WAV-passthrough and external ffmpeg/afconvert
    /// conversion paths, which hand back a file via [`Self::path`] instead.
    pub fn samples(&self) -> Option<&[f32]> {
        match &self.samples {
            PreparedAudioSamples::InMemory(samples) => Some(samples.as_slice()),
            PreparedAudioSamples::Path(_) => None,
        }
    }

    /// Cheap `Arc` clone (a refcount bump, not a data copy) of
    /// [`Self::samples`], for attaching to a
    /// [`crate::TranscriptionRequest`]/`NativeAsrOfflineRequest` so the
    /// native backend can decode straight from memory instead of re-reading
    /// [`Self::path`] from disk.
    pub fn shared_samples(&self) -> Option<Arc<Vec<f32>>> {
        match &self.samples {
            PreparedAudioSamples::InMemory(samples) => Some(Arc::clone(samples)),
            PreparedAudioSamples::Path(_) => None,
        }
    }

    pub fn is_converted(&self) -> bool {
        self.temp_dir.is_some() || matches!(self.samples, PreparedAudioSamples::InMemory(_))
    }

    /// Best-effort duration of the prepared audio in seconds. Prefers the
    /// cheap probed source-file duration (wav's fmt/data chunk sizes,
    /// `original().duration_seconds`); falls back to counting the in-memory
    /// samples for the symphonia decode path, or re-probing the prepared WAV
    /// on disk for the external-conversion path. `None` only when nothing
    /// here can determine it (e.g. an unrecognized-extension passthrough).
    pub fn duration_seconds(&self) -> Option<f64> {
        if let Some(duration) = self.original.duration_seconds {
            return Some(duration);
        }
        match &self.samples {
            PreparedAudioSamples::InMemory(samples) => Some(
                samples.len() as f64 / f64::from(super::symphonia_decode::TARGET_SAMPLE_RATE_HZ),
            ),
            PreparedAudioSamples::Path(path) if self.temp_dir.is_some() => {
                super::probe_wav_duration(path)
            }
            PreparedAudioSamples::Path(_) => None,
        }
    }
}
