//! Shared resolution of pulled diarization model-pack files. Support packs load
//! tens-of-MB weights from files (not vendored), resolved from an env override
//! or from the installed model store.
//! Thin diarization-flavored wrappers over the model-agnostic resolver in
//! `crate::capability_pack` (also used by the Qwen3-ForcedAligner
//! word-timestamps capability pack).

#[cfg(test)]
use std::path::Path;
use std::path::PathBuf;

/// Resolve a pack path: the `env_var` override (if it points at a file), else the
/// installed pack whose model id contains `model_id_hint`.
pub(super) fn resolve_pack(env_var: &str, model_id_hint: &str) -> Option<PathBuf> {
    crate::capability_pack::resolve_installed_capability_pack(env_var, model_id_hint)
}

/// Test-only format discriminator for raw-source versus converted-pack parity.
#[cfg(test)]
pub(super) fn is_gguf(path: &Path) -> bool {
    crate::capability_pack::is_gguf_capability_pack(path)
}
