//! Runtime resolution of the WeSpeaker speaker-embedder weight pack.
//!
//! WeSpeaker ResNet34 is the only speaker embedder. It is resolved from
//! `OPENASR_WESPEAKER_PACK` or the standard `openasr_home()/models/wespeaker*/`
//! location. Absence is graceful: callers fall back to "diarization
//! unavailable". Pack payloads are GGUF `.oasr` files; raw `.safetensors` are
//! still accepted as dev fast paths.

use std::path::Path;
use std::sync::OnceLock;

use super::SpeakerEmbedder;
use super::WeSpeakerEmbedder;

static WESPEAKER_SHARED: OnceLock<SharedWeSpeakerEmbedder> = OnceLock::new();

const WESPEAKER_PACK_ENV: &str = "OPENASR_WESPEAKER_PACK";
const WESPEAKER_INSTALLED_DIR_HINT: &str = "wespeaker";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpeakerEmbedderIdentity {
    pub embedding_dim: usize,
    pub pack_fingerprint: String,
}

struct SharedWeSpeakerEmbedder {
    embedder: WeSpeakerEmbedder,
    identity: SpeakerEmbedderIdentity,
}

/// Whether the WeSpeaker embedder pack is resolvable right now (env override or
/// installed location), without loading the weights. Capability reporting uses
/// this presence probe; the actual load and final fail-closed gate stay in
/// [`shared_embedder`].
pub fn embedder_pack_installed() -> bool {
    crate::diarize::pack::resolve_pack(WESPEAKER_PACK_ENV, WESPEAKER_INSTALLED_DIR_HINT).is_some()
}

/// The process-wide WeSpeaker embedder, or `None` if no pack is installed.
///
/// Only a successful load is cached. A failed resolve/load must NOT poison the
/// cache: capability reporting re-probes the filesystem on every ask, so a
/// daemon that saw a diarize request before the pack was installed has to pick
/// the pack up on the next request, not after a restart.
pub fn shared_embedder() -> Option<&'static dyn SpeakerEmbedder> {
    shared_wespeaker_state().map(|state| &state.embedder as &dyn SpeakerEmbedder)
}

/// Metadata for the process-wide WeSpeaker embedder, including the content
/// fingerprint stored next to enrolled voice-match embeddings.
pub fn shared_embedder_identity() -> Option<&'static SpeakerEmbedderIdentity> {
    shared_wespeaker_state().map(|state| &state.identity)
}

fn shared_wespeaker_state() -> Option<&'static SharedWeSpeakerEmbedder> {
    if let Some(state) = WESPEAKER_SHARED.get() {
        return Some(state);
    }
    let path =
        crate::diarize::pack::resolve_pack(WESPEAKER_PACK_ENV, WESPEAKER_INSTALLED_DIR_HINT)?;
    let embedder = load_wespeaker_embedder(&path)?;
    let identity = SpeakerEmbedderIdentity {
        embedding_dim: embedder.embedding_dim(),
        pack_fingerprint: pack_fingerprint(&path)?,
    };
    let _ = WESPEAKER_SHARED.set(SharedWeSpeakerEmbedder { embedder, identity });
    WESPEAKER_SHARED.get()
}

/// Load the embedder from a resolved pack path, choosing the GGUF `.oasr` loader
/// or the raw safetensors fast path by sniffing the file magic.
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
