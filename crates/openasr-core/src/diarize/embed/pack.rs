//! Runtime resolution and selection of the speaker-embedder weight pack.
//!
//! Two embedders can be installed during the ReDimNet2-B6 migration:
//! ReDimNet2-B6 (`OPENASR_REDIMNET_PACK` / installed-dir hint `redimnet`,
//! 192-d, ggml graph) and the legacy WeSpeaker ResNet34
//! (`OPENASR_WESPEAKER_PACK` / installed-dir hint `wespeaker`, 256-d,
//! pure-Rust). [`choose_embedder_pack`] is the one selection rule: ReDimNet2
//! wins whenever its pack is present, WeSpeaker is used only as a fallback,
//! and both absent resolves to `None` -- callers fall back to "diarization
//! unavailable" rather than panicking. Removing WeSpeaker entirely is a later,
//! separately approved step (see `redimnet::mod` docs / `HANDOFF.md`).

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use super::RedimNet2Embedder;
use super::SpeakerEmbedder;
use super::WeSpeakerEmbedder;

static SHARED_EMBEDDER: OnceLock<SharedEmbedderState> = OnceLock::new();

const WESPEAKER_PACK_ENV: &str = "OPENASR_WESPEAKER_PACK";
const WESPEAKER_INSTALLED_DIR_HINT: &str = "wespeaker";
const REDIMNET_PACK_ENV: &str = "OPENASR_REDIMNET_PACK";
const REDIMNET_INSTALLED_DIR_HINT: &str = "redimnet";

/// Human-readable label for ReDimNet2-B6's embedding space (documentation /
/// audit metadata only). The actual runtime compatibility gate is the pack's
/// content fingerprint (`SpeakerEmbedderIdentity::pack_fingerprint`, a sha256
/// of the `.oasr` file) plus `embedding_dim`, not this string -- a re-export
/// or repack of the same checkpoint keeps the same fingerprint and stays
/// compatible even if this label changes.
pub(crate) const REDIMNET_EMBEDDING_SPACE_VERSION: &str = "redimnet2-b6-cn-v1";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpeakerEmbedderIdentity {
    pub embedding_dim: usize,
    pub pack_fingerprint: String,
}

struct SharedEmbedderState {
    embedder: Box<dyn SpeakerEmbedder>,
    identity: SpeakerEmbedderIdentity,
}

/// Which embedder [`choose_embedder_pack`] picked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EmbedderKind {
    RedimNet2,
    WeSpeaker,
}

/// The one selection rule between the two coexisting embedders: ReDimNet2
/// wins whenever its pack is present (regardless of whether WeSpeaker is also
/// installed); WeSpeaker is used only when ReDimNet2 is absent; neither
/// present resolves to `None`. Pure and side-effect-free so it is directly
/// unit-testable against every presence/absence combination without touching
/// the filesystem or the process-wide cache below.
fn choose_embedder_pack(
    redimnet: Option<PathBuf>,
    wespeaker: Option<PathBuf>,
) -> Option<(EmbedderKind, PathBuf)> {
    if let Some(path) = redimnet {
        return Some((EmbedderKind::RedimNet2, path));
    }
    wespeaker.map(|path| (EmbedderKind::WeSpeaker, path))
}

fn redimnet_pack_path() -> Option<PathBuf> {
    crate::diarize::pack::resolve_pack(REDIMNET_PACK_ENV, REDIMNET_INSTALLED_DIR_HINT)
}

fn wespeaker_pack_path() -> Option<PathBuf> {
    crate::diarize::pack::resolve_pack(WESPEAKER_PACK_ENV, WESPEAKER_INSTALLED_DIR_HINT)
}

/// Whether an active embedder pack is resolvable right now (either ReDimNet2
/// or WeSpeaker, env override or installed location), without loading the
/// weights. Capability reporting uses this presence probe; the actual load
/// and final fail-closed gate stay in [`shared_embedder`].
pub fn embedder_pack_installed() -> bool {
    choose_embedder_pack(redimnet_pack_path(), wespeaker_pack_path()).is_some()
}

/// The process-wide active embedder (ReDimNet2 if installed, else WeSpeaker),
/// or `None` if neither pack is installed.
///
/// Only a successful load is cached. A failed resolve/load must NOT poison the
/// cache: capability reporting re-probes the filesystem on every ask, so a
/// daemon that saw a diarize request before the pack was installed has to pick
/// the pack up on the next request, not after a restart.
pub fn shared_embedder() -> Option<&'static dyn SpeakerEmbedder> {
    shared_embedder_state().map(|state| state.embedder.as_ref())
}

/// Metadata for the process-wide active embedder, including the content
/// fingerprint stored next to enrolled voice-match embeddings.
pub fn shared_embedder_identity() -> Option<&'static SpeakerEmbedderIdentity> {
    shared_embedder_state().map(|state| &state.identity)
}

fn shared_embedder_state() -> Option<&'static SharedEmbedderState> {
    if let Some(state) = SHARED_EMBEDDER.get() {
        return Some(state);
    }
    let (kind, path) = choose_embedder_pack(redimnet_pack_path(), wespeaker_pack_path())?;
    let state = load_embedder_state(kind, &path)?;
    let _ = SHARED_EMBEDDER.set(state);
    SHARED_EMBEDDER.get()
}

fn load_embedder_state(kind: EmbedderKind, path: &Path) -> Option<SharedEmbedderState> {
    let embedder: Box<dyn SpeakerEmbedder> = match kind {
        // ReDimNet2 is GGUF-only (design decision #1: a ggml-native artifact,
        // no pure-Rust safetensors fast path like WeSpeaker's dev shortcut).
        EmbedderKind::RedimNet2 => Box::new(RedimNet2Embedder::from_oasr(path).ok()?),
        EmbedderKind::WeSpeaker => Box::new(load_wespeaker_embedder(path)?),
    };
    let identity = SpeakerEmbedderIdentity {
        embedding_dim: embedder.embedding_dim(),
        pack_fingerprint: pack_fingerprint(path)?,
    };
    Some(SharedEmbedderState { embedder, identity })
}

/// Load the WeSpeaker embedder from a resolved pack path, choosing the GGUF
/// `.oasr` loader or the raw safetensors fast path by sniffing the file magic.
fn load_wespeaker_embedder(path: &Path) -> Option<WeSpeakerEmbedder> {
    if crate::diarize::pack::is_gguf(path) {
        WeSpeakerEmbedder::from_oasr(path).ok()
    } else {
        let bytes = std::fs::read(path).ok()?;
        WeSpeakerEmbedder::from_safetensors(&bytes).ok()
    }
}

fn pack_fingerprint(path: &Path) -> Option<String> {
    use sha2::{Digest, Sha256};
    use std::io::Read;

    let mut file = std::fs::File::open(path).ok()?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer).ok()?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Some(format!("sha256:{:x}", hasher.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(name: &str) -> PathBuf {
        PathBuf::from(name)
    }

    #[test]
    fn choose_embedder_pack_prefers_redimnet_when_both_present() {
        let picked = choose_embedder_pack(Some(p("redimnet.oasr")), Some(p("wespeaker.oasr")));
        assert_eq!(picked, Some((EmbedderKind::RedimNet2, p("redimnet.oasr"))));
    }

    #[test]
    fn choose_embedder_pack_uses_redimnet_alone() {
        let picked = choose_embedder_pack(Some(p("redimnet.oasr")), None);
        assert_eq!(picked, Some((EmbedderKind::RedimNet2, p("redimnet.oasr"))));
    }

    #[test]
    fn choose_embedder_pack_falls_back_to_wespeaker_alone() {
        let picked = choose_embedder_pack(None, Some(p("wespeaker.oasr")));
        assert_eq!(picked, Some((EmbedderKind::WeSpeaker, p("wespeaker.oasr"))));
    }

    #[test]
    fn choose_embedder_pack_is_none_when_neither_present() {
        assert_eq!(choose_embedder_pack(None, None), None);
    }

    #[test]
    fn redimnet_embedding_space_version_is_pinned() {
        assert_eq!(REDIMNET_EMBEDDING_SPACE_VERSION, "redimnet2-b6-cn-v1");
    }
}
