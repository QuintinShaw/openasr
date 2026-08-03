use std::{
    fmt,
    ops::{Deref, Range},
    path::{Path, PathBuf},
    sync::Arc,
};

use crate::BackendKind;

// `qta` is macOS QuickTime Player's audio recording extension -- a MOV
// container (ffmpeg's `mov,mp4,m4a,3gp,3g2,mj2` demuxer probes and decodes it
// like any other MOV/M4A file, no special handling needed beyond recognizing
// the extension). `mov` itself (QuickTime movies with an audio track, e.g. a
// screen recording) rides the exact same demuxer and needs the same nothing.
// `m4b` (audiobook) is the same ISO/MP4 container under a
// different extension -- symphonia's `isomp4` reader already lists it as a
// recognized extension alongside `mp4`/`m4a` (see
// `symphonia_format_isomp4::demuxer::IsoMp4Reader::query`), so no extra
// feature or code is needed beyond listing it here. `aac` is a bare ADTS
// stream (WeChat and many other recorders/voice-memo apps emit this, not an
// m4a/mp4 container) -- decoded by the `AdtsReader` the already-enabled
// symphonia `aac` feature registers. `aiff`/`aif`/`aifc` and `caf` are Apple's
// two other common PCM/compressed containers, decoded in-process by the
// symphonia `aiff`/`caf` features (both enabled in the workspace `Cargo.toml`
// alongside the format's already-enabled codecs -- pcm/aac/alac/flac/mp3).
//
// Every extension here is *reachable*, not necessarily decodable in-process:
// `webm` in particular is a container, not a codec. Opus -- the most common
// codec inside `.opus`/`.ogg`/`.webm` audio -- decodes in-process through the
// bundled libopus (see `audio/opus_decode`); anything the in-process path
// still cannot handle (HE-AAC, Opus multistream >2ch, `wma`, `amr`, ...)
// falls through to the external ffmpeg/afconvert conversion chain in
// `prepare.rs` -- with ffmpeg on PATH it still transcribes; without it, the
// error names the detected codec instead of pretending the file is corrupt
// (see `symphonia_decode::probe_codec_label` and `prepare::codec_note`).
// `wma`/`amr` are listed here even though symphonia has no ASF/AMR demuxer at
// all (unlike HE-AAC, which symphonia's `isomp4`/`aac` *can* at least name):
// ffmpeg decodes both, so this only changes whether that fallback is ever
// attempted, not whether every upload of one succeeds.
pub(crate) const RECOGNIZED_EXTENSIONS: &[&str] = &[
    "wav", "mp3", "mp4", "m4a", "m4b", "mov", "webm", "flac", "ogg", "opus", "qta", "aac", "aiff",
    "aif", "aifc", "caf", "wma", "amr",
];

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
/// The in-memory variant is an immutable [`PcmBuffer`]. Cloning it only bumps
/// an `Arc` refcount; downstream consumers derive [`PcmSlice`] views into the
/// same allocation instead of cloning a whole recording or each long-form
/// chunk.
#[derive(Debug)]
pub(crate) enum PreparedAudioSamples {
    Path(PathBuf),
    InMemory(PcmBuffer),
}

/// Immutable, shared normalized PCM backing.
///
/// `Arc<Vec<f32>>` is intentional rather than `Arc<[f32]>`: decoded audio is
/// already a `Vec`, so this wraps it without moving every sample into a second
/// allocation. The vector is never exposed mutably. Long-form decode,
/// recording-level Voice ID, and forced alignment share it through
/// [`PcmSlice`] ranges.
#[derive(Clone)]
pub(crate) struct PcmBuffer {
    backing: Arc<Vec<f32>>,
}

impl PcmBuffer {
    pub(crate) fn from_vec(samples: Vec<f32>) -> Self {
        Self {
            backing: Arc::new(samples),
        }
    }

    pub(crate) fn from_shared(samples: Arc<Vec<f32>>) -> Self {
        Self { backing: samples }
    }

    pub(crate) fn shared_backing(&self) -> Arc<Vec<f32>> {
        Arc::clone(&self.backing)
    }

    pub(crate) fn as_slice(&self) -> &[f32] {
        self.backing.as_slice()
    }

    pub(crate) fn len(&self) -> usize {
        self.backing.len()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.backing.is_empty()
    }

    /// Bytes reserved by the normalized PCM sample allocation.
    pub(crate) fn resident_bytes(&self) -> u64 {
        u64::try_from(self.backing.capacity())
            .unwrap_or(u64::MAX)
            .saturating_mul(std::mem::size_of::<f32>() as u64)
    }

    pub(crate) fn full_slice(&self) -> PcmSlice {
        PcmSlice {
            backing: Arc::clone(&self.backing),
            range: 0..self.backing.len(),
        }
    }

    /// Creates a range relative to this full backing. This follows ordinary
    /// slice indexing semantics and panics for an invalid range; native
    /// long-form ranges have already been validated by the planner.
    pub(crate) fn slice(&self, range: Range<usize>) -> PcmSlice {
        assert!(range.start <= range.end, "PCM range start exceeds end");
        assert!(range.end <= self.backing.len(), "PCM range exceeds backing");
        PcmSlice {
            backing: Arc::clone(&self.backing),
            range,
        }
    }

    /// Stable test identity for one allocation. It deliberately identifies
    /// the backing object rather than a slice's first sample (empty and offset
    /// slices still share an identity).
    #[cfg(test)]
    pub(crate) fn backing_identity(&self) -> usize {
        Arc::as_ptr(&self.backing) as usize
    }
}

impl From<Vec<f32>> for PcmBuffer {
    fn from(samples: Vec<f32>) -> Self {
        Self::from_vec(samples)
    }
}

impl From<Arc<Vec<f32>>> for PcmBuffer {
    fn from(samples: Arc<Vec<f32>>) -> Self {
        Self::from_shared(samples)
    }
}

impl Deref for PcmBuffer {
    type Target = [f32];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl AsRef<[f32]> for PcmBuffer {
    fn as_ref(&self) -> &[f32] {
        self.as_slice()
    }
}

impl fmt::Debug for PcmBuffer {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PcmBuffer")
            .field("samples", &self.len())
            .field("resident_bytes", &self.resident_bytes())
            .finish_non_exhaustive()
    }
}

impl PartialEq for PcmBuffer {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

/// Immutable owned view into a [`PcmBuffer`].
///
/// A view is `Arc + Range`: cloning or sending it to a long-form worker never
/// copies samples, and its public surface only exposes `&[f32]`.
#[derive(Clone)]
pub(crate) struct PcmSlice {
    backing: Arc<Vec<f32>>,
    range: Range<usize>,
}

impl PcmSlice {
    pub(crate) fn as_slice(&self) -> &[f32] {
        &self.backing[self.range.clone()]
    }

    #[cfg(test)]
    pub(crate) fn range(&self) -> Range<usize> {
        self.range.clone()
    }

    #[cfg(test)]
    pub(crate) fn backing_identity(&self) -> usize {
        Arc::as_ptr(&self.backing) as usize
    }

    /// Creates a sub-view with a range relative to this view.
    pub(crate) fn slice(&self, range: Range<usize>) -> Self {
        assert!(range.start <= range.end, "PCM sub-range start exceeds end");
        assert!(range.end <= self.range.len(), "PCM sub-range exceeds view");
        let start = self.range.start + range.start;
        let end = self.range.start + range.end;
        Self {
            backing: Arc::clone(&self.backing),
            range: start..end,
        }
    }
}

impl From<Vec<f32>> for PcmSlice {
    fn from(samples: Vec<f32>) -> Self {
        PcmBuffer::from_vec(samples).full_slice()
    }
}

impl From<PcmBuffer> for PcmSlice {
    fn from(buffer: PcmBuffer) -> Self {
        buffer.full_slice()
    }
}

impl Deref for PcmSlice {
    type Target = [f32];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl AsRef<[f32]> for PcmSlice {
    fn as_ref(&self) -> &[f32] {
        self.as_slice()
    }
}

impl fmt::Debug for PcmSlice {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PcmSlice")
            .field("range", &self.range)
            .field("backing_samples", &self.backing.len())
            .finish_non_exhaustive()
    }
}

impl PartialEq for PcmSlice {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
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
            PreparedAudioSamples::InMemory(samples) => Some(samples.shared_backing()),
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

#[cfg(test)]
mod pcm_tests {
    use super::*;

    #[test]
    fn pcm_views_keep_one_backing_and_only_shift_the_sample_pointer() {
        let buffer = PcmBuffer::from_vec((0..32).map(|sample| sample as f32).collect());
        let first = buffer.slice(4..20);
        let nested = first.slice(3..9);

        assert_eq!(buffer.backing_identity(), first.backing_identity());
        assert_eq!(first.backing_identity(), nested.backing_identity());
        assert_eq!(first.range(), 4..20);
        assert_eq!(nested.range(), 7..13);
        assert_eq!(nested.as_slice(), &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        assert_eq!(first.as_ptr(), buffer.as_ptr().wrapping_add(4));
        assert_eq!(nested.as_ptr(), buffer.as_ptr().wrapping_add(7));
    }

    #[test]
    fn pcm_clones_do_not_duplicate_resident_sample_bytes() {
        let mut samples = Vec::with_capacity(64);
        samples.extend([0.0, 1.0, 2.0]);
        let buffer = PcmBuffer::from_vec(samples);
        let clone = buffer.clone();
        let view = buffer.full_slice();

        assert_eq!(buffer.backing_identity(), clone.backing_identity());
        assert_eq!(buffer.backing_identity(), view.backing_identity());
        assert_eq!(buffer.resident_bytes(), 64 * 4);
        assert_eq!(clone.resident_bytes(), buffer.resident_bytes());
    }
}
