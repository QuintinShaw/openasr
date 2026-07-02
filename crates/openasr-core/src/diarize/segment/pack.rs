//! Runtime resolution of the pulled pyannote segmentation pack.
//!
//! Resolved by the shared [`crate::diarize::pack`] resolver from
//! `OPENASR_PYANNOTE_PACK` or `openasr_home()/models/pyannote*/`, loaded once into
//! a process-wide segmenter. Absence is graceful (callers fall back to
//! VAD-segment diarization). The pack payload is a GGUF `.oasr` pack (the
//! catalog/pull format); a raw `.safetensors` is still accepted as the dev fast
//! path. The loader sniffs the file magic to pick the path.

use std::path::Path;
use std::sync::OnceLock;

use super::PyannoteSegmenter;

static SHARED: OnceLock<PyannoteSegmenter> = OnceLock::new();

const PACK_ENV: &str = "OPENASR_PYANNOTE_PACK";

/// The process-wide pyannote segmenter, or `None` if no pack is installed.
///
/// Only a successful load is cached: a probe while the pack is absent must not
/// poison the cache for the rest of the daemon's life (the pack can be pulled
/// mid-daemon and has to be picked up on the next request).
pub fn shared_segmenter() -> Option<&'static PyannoteSegmenter> {
    if let Some(segmenter) = SHARED.get() {
        return Some(segmenter);
    }
    let path = crate::diarize::pack::resolve_pack(PACK_ENV, "pyannote")?;
    let segmenter = load_segmenter(&path)?;
    // A concurrent loader may have won the race; either value came from the
    // same pack, so keep whichever landed first.
    let _ = SHARED.set(segmenter);
    SHARED.get()
}

/// Load the segmenter from a resolved pack path, choosing the GGUF `.oasr` loader
/// or the raw safetensors fast path by sniffing the file magic.
fn load_segmenter(path: &Path) -> Option<PyannoteSegmenter> {
    if crate::diarize::pack::is_gguf(path) {
        PyannoteSegmenter::from_oasr(path).ok()
    } else {
        let bytes = std::fs::read(path).ok()?;
        PyannoteSegmenter::from_safetensors(&bytes).ok()
    }
}
