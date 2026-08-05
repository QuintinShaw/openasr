//! Path resolution for the optional DiariZen pack.
//!
//! Persistent runtime ownership intentionally lives in the injected execution
//! service root; this module contains no process-global runtime cache.

use std::path::PathBuf;

const PACK_ENV: &str = "OPENASR_DIARIZEN_PACK";
const INSTALLED_MODEL_ID_HINT: &str = super::DIARIZEN_MODEL_ID;
const PREFERRED_QUANT: &str = "fp16";
pub(crate) const DIARIZEN_PACK_PREFERENCE: crate::capability_pack::CapabilityPackPreference =
    crate::capability_pack::CapabilityPackPreference::new(
        super::DIARIZEN_MODEL_ID,
        INSTALLED_MODEL_ID_HINT,
        PREFERRED_QUANT,
    );

pub(crate) fn diarizen_pack_path() -> Option<PathBuf> {
    crate::diarize::pack::resolve_pack(PACK_ENV, DIARIZEN_PACK_PREFERENCE)
}

pub fn diarizen_pack_installed() -> bool {
    diarizen_pack_path().is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn installed_pack_identity_is_exact_and_stable() {
        assert_eq!(PACK_ENV, "OPENASR_DIARIZEN_PACK");
        assert_eq!(INSTALLED_MODEL_ID_HINT, "diarizen-large-s80-v2");
    }
}
