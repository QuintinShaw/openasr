//! Runtime resolution of the Qwen3-ForcedAligner-0.6B capability-pack file
//! (the `word-timestamps` feature's `ForcedAligner` role, `catalog::
//! word_timestamps_forced_aligner_pack`). Resolved from
//! `OPENASR_FORCED_ALIGNER_PACK` or the installed pack whose model id contains
//! `forced-aligner` (`qwen3-forced-aligner-0.6b`), mirroring
//! `diarize::embed::pack`'s ReDimNet2-B6 resolution but built on the shared
//! `crate::capability_pack` resolver directly (this pack is not diarization).

use std::path::PathBuf;

const FORCED_ALIGNER_PACK_ENV: &str = "OPENASR_FORCED_ALIGNER_PACK";
const FORCED_ALIGNER_INSTALLED_MODEL_ID_HINT: &str = "forced-aligner";
const FORCED_ALIGNER_MODEL_ID: &str = "qwen3-forced-aligner-0.6b";
const FORCED_ALIGNER_PREFERRED_QUANT: &str = "q8_0";
pub(crate) const FORCED_ALIGNER_PACK_PREFERENCE: crate::capability_pack::CapabilityPackPreference =
    crate::capability_pack::CapabilityPackPreference::new(
        FORCED_ALIGNER_MODEL_ID,
        FORCED_ALIGNER_INSTALLED_MODEL_ID_HINT,
        FORCED_ALIGNER_PREFERRED_QUANT,
    );

/// The resolved path to the installed Qwen3-ForcedAligner pack, or `None` if
/// no pack is installed.
pub(crate) fn resolve_forced_aligner_pack_path() -> Option<PathBuf> {
    crate::capability_pack::resolve_installed_capability_pack(
        FORCED_ALIGNER_PACK_ENV,
        FORCED_ALIGNER_PACK_PREFERENCE,
    )
}
