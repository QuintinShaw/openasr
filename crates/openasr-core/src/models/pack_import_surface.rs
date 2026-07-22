//! Force-linked pack-import surfaces for native families.
//!
//! Architecture integration descriptors name a convert symbol or external
//! tooling path. This module is the compiled registration table that proves
//! each `CoreConvert` symbol still exists and is linked into the crate:
//! naming a deleted or private convert entry fails to compile here. File-on-disk
//! checks alone are intentionally insufficient.

use std::collections::BTreeMap;

#[cfg(test)]
use crate::arch::{OpenAsrArchitectureRegistry, OpenAsrPackImportSurface};

fn link_symbol<F>(symbol: &'static str, function: F) -> (&'static str, usize) {
    // Keep the convert entry reachable so deleting or privatizing it fails this
    // module's compile, not only a stringly path check.
    (symbol, std::ptr::from_ref(&function) as usize)
}

/// Returns the set of core convert symbols that are force-linked into this
/// crate, keyed by symbol name. Presence in this map is what makes a
/// `OpenAsrPackImportSurface::CoreConvert` declaration real.
pub(crate) fn linked_core_pack_import_symbols() -> BTreeMap<&'static str, usize> {
    [
        link_symbol(
            "convert_local_cohere_source_to_runtime_pack",
            crate::models::cohere::convert_local_cohere_source_to_runtime_pack,
        ),
        link_symbol(
            "convert_local_whisper_hf_source_to_runtime_pack",
            crate::models::whisper::convert_local_whisper_hf_source_to_runtime_pack,
        ),
        link_symbol(
            "convert_local_qwen_source_to_runtime_pack",
            crate::models::qwen::convert_local_qwen_source_to_runtime_pack,
        ),
        link_symbol(
            "convert_local_parakeet_ctc_source_to_runtime_pack",
            crate::models::parakeet_ctc::convert_local_parakeet_ctc_source_to_runtime_pack,
        ),
        link_symbol(
            "convert_local_parakeet_tdt_source_to_runtime_pack",
            crate::models::parakeet_tdt::convert_local_parakeet_tdt_source_to_runtime_pack,
        ),
        link_symbol(
            "convert_local_wav2vec2_ctc_source_to_runtime_pack",
            crate::models::wav2vec2_ctc::convert_local_wav2vec2_ctc_source_to_runtime_pack,
        ),
        link_symbol(
            "convert_local_xasr_zipformer_source_to_runtime_pack",
            crate::models::xasr_zipformer::convert_local_xasr_zipformer_source_to_runtime_pack,
        ),
        link_symbol(
            "convert_local_moonshine_source_to_runtime_pack",
            crate::models::moonshine::convert_local_moonshine_source_to_runtime_pack,
        ),
        link_symbol(
            "convert_local_dolphin_wenet_source_to_runtime_pack",
            crate::models::dolphin::convert_local_dolphin_wenet_source_to_runtime_pack,
        ),
        link_symbol(
            "convert_local_sensevoice_source_to_runtime_pack",
            crate::models::sensevoice::convert_local_sensevoice_source_to_runtime_pack,
        ),
        link_symbol(
            "convert_local_firered_aed_source_to_runtime_pack",
            crate::models::firered_aed::convert_local_firered_aed_source_to_runtime_pack,
        ),
        link_symbol(
            "convert_local_firered_llm_source_to_runtime_pack",
            crate::models::firered_llm::convert_local_firered_llm_source_to_runtime_pack,
        ),
        link_symbol(
            "convert_local_moss_transcribe_diarize_source_to_runtime_pack",
            crate::models::moss_transcribe_diarize::convert_local_moss_transcribe_diarize_source_to_runtime_pack,
        ),
        link_symbol(
            "convert_local_granite_speech_source_to_runtime_pack",
            crate::models::granite_speech::convert_local_granite_speech_source_to_runtime_pack,
        ),
    ]
    .into_iter()
    .collect()
}

/// Ensures every architecture-declared core convert surface is present in the
/// force-linked table above.
#[cfg(test)]
pub(crate) fn assert_architecture_pack_imports_are_linked() {
    let linked = linked_core_pack_import_symbols();
    for descriptor in OpenAsrArchitectureRegistry::with_builtins().descriptors() {
        match descriptor.integration.pack_import {
            OpenAsrPackImportSurface::CoreConvert { symbol } => {
                assert!(
                    linked.contains_key(symbol),
                    "native family '{}' declares core pack-import symbol '{}' but it is not force-linked in pack_import_surface",
                    descriptor.model_family,
                    symbol
                );
            }
            OpenAsrPackImportSurface::ExternalTooling { .. } => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_architecture_core_pack_import_is_force_linked() {
        assert_architecture_pack_imports_are_linked();
    }

    #[test]
    fn half_wired_core_pack_import_symbol_is_rejected() {
        let linked = linked_core_pack_import_symbols();
        assert!(
            !linked.contains_key("convert_local_does_not_exist"),
            "unknown symbols must not appear in the linked table"
        );
    }
}
