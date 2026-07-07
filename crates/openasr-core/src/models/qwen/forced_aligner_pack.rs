//! Runtime resolution of the Qwen3-ForcedAligner-0.6B capability-pack file
//! (the `word-timestamps` feature's `ForcedAligner` role, `catalog::
//! word_timestamps_forced_aligner_pack`). Resolved from
//! `OPENASR_FORCED_ALIGNER_PACK` or the standard
//! `openasr_home()/models/*forced-aligner*/` location, mirroring
//! `diarize::embed::pack`'s WeSpeaker resolution but built on the shared
//! `crate::capability_pack` resolver directly (this pack is not diarization).

use std::path::PathBuf;

const FORCED_ALIGNER_PACK_ENV: &str = "OPENASR_FORCED_ALIGNER_PACK";
const FORCED_ALIGNER_INSTALLED_DIR_HINT: &str = "forced-aligner";

/// The resolved path to the installed Qwen3-ForcedAligner pack, or `None` if
/// no pack is installed.
pub(crate) fn resolve_forced_aligner_pack_path() -> Option<PathBuf> {
    crate::capability_pack::resolve_installed_capability_pack(
        FORCED_ALIGNER_PACK_ENV,
        FORCED_ALIGNER_INSTALLED_DIR_HINT,
    )
}
