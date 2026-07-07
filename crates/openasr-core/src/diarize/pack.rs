//! Shared resolution of pulled diarization model-pack files. Support packs load
//! tens-of-MB weights from files (not vendored), resolved from an env override
//! or the standard installed location under `openasr_home()/models/<model>/`.
//! Thin diarization-flavored wrappers over the model-agnostic resolver in
//! `crate::capability_pack` (also used by the Qwen3-ForcedAligner
//! word-timestamps capability pack).

use std::path::{Path, PathBuf};

/// Resolve a pack path: the `env_var` override (if it points at a file), else the
/// first `.safetensors`/`.oasr` under a `models/*` directory whose name contains
/// `dir_substr`.
pub(super) fn resolve_pack(env_var: &str, dir_substr: &str) -> Option<PathBuf> {
    crate::capability_pack::resolve_installed_capability_pack(env_var, dir_substr)
}

/// Whether `path` is a GGUF (`.oasr`) pack, by sniffing the 4-byte magic rather
/// than trusting the extension. A diarization pack may be delivered as either a
/// pulled GGUF `.oasr` or a raw `.safetensors` (the dev fast path), so the
/// loaders branch on this.
pub(super) fn is_gguf(path: &Path) -> bool {
    crate::capability_pack::is_gguf_capability_pack(path)
}
