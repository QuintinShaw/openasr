//! Runtime resolution of the installed FireRedPunc capability-pack file (the
//! `punctuation` feature's post-processing model). Resolved from
//! `OPENASR_FIRERED_PUNC_PACK` or the standard
//! `openasr_home()/models/*firered-punc*/` location, mirroring
//! `models::qwen::forced_aligner_pack`'s resolution built on the shared
//! `crate::capability_pack` resolver (this pack is neither diarization nor
//! forced alignment, but the same "installed support model" shape).

use std::path::PathBuf;

const FIRERED_PUNC_PACK_ENV: &str = "OPENASR_FIRERED_PUNC_PACK";
const FIRERED_PUNC_INSTALLED_DIR_HINT: &str = "firered-punc";

/// The resolved path to the installed FireRedPunc pack, or `None` if no pack
/// is installed. Callers must treat `None` as "punctuation stays off" -- this
/// capability pack is never auto-downloaded by the transcription path.
pub(crate) fn resolve_firered_punc_pack_path() -> Option<PathBuf> {
    crate::capability_pack::resolve_installed_capability_pack(
        FIRERED_PUNC_PACK_ENV,
        FIRERED_PUNC_INSTALLED_DIR_HINT,
    )
}
