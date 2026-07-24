//! Runtime resolution of the speaker-embedder weight pack.
//!
//! Only ReDimNet2-B6 is supported (`OPENASR_REDIMNET_PACK` / installed-dir hint
//! `redimnet`, 192-d, ggml graph). When the pack is absent, resolution returns
//! `None` and callers fail closed with a clear "install redimnet2-b6-cn" error
//! rather than falling back to any other embedder.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use super::RedimNet2Embedder;
use super::SpeakerEmbedder;

static SHARED_EMBEDDER: OnceLock<SharedEmbedderState> = OnceLock::new();

const REDIMNET_PACK_ENV: &str = "OPENASR_REDIMNET_PACK";
const REDIMNET_INSTALLED_DIR_HINT: &str = "redimnet";

/// Catalog / pull id of the only supported speaker-embedder pack.
pub const SPEAKER_EMBEDDER_PACK_ID: &str = "redimnet2-b6-cn";

/// User-facing label for the only supported speaker-embedder pack.
pub const SPEAKER_EMBEDDER_PACK_LABEL: &str =
    "ReDimNet2-B6 speaker-embedder pack (redimnet2-b6-cn)";

/// Fail-closed reason when Voice ID enrollment cannot resolve the pack.
pub const VOICE_ID_EMBEDDER_PACK_MISSING_REASON: &str = "creating a voice id requires the ReDimNet2-B6 speaker-embedder pack (redimnet2-b6-cn); install the pack first";

/// Fail-closed reason when legacy voice-match enrollment cannot resolve the pack.
pub const VOICE_MATCH_EMBEDDER_PACK_MISSING_REASON: &str = "creating a voice match profile requires the ReDimNet2-B6 speaker-embedder pack (redimnet2-b6-cn); install the pack first";

/// Fail-closed reason when diarize was accepted by capability probe but the pack
/// then failed to load (path present, weights unusable).
pub const DIARIZATION_EMBEDDER_LOAD_FAILED_REASON: &str = "Diarization was requested but the ReDimNet2-B6 speaker-embedder pack (redimnet2-b6-cn) could not be loaded.";

/// Fail-closed reason when realtime diarize is requested without the pack.
pub const REALTIME_DIARIZATION_EMBEDDER_MISSING_REASON: &str = "Realtime diarization needs the ReDimNet2-B6 speaker-embedder pack (redimnet2-b6-cn); install it or omit diarize=true.";

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

fn redimnet_pack_path() -> Option<PathBuf> {
    crate::diarize::pack::resolve_pack(REDIMNET_PACK_ENV, REDIMNET_INSTALLED_DIR_HINT)
}

/// Whether the ReDimNet2-B6 embedder pack is resolvable right now (env override
/// or installed location), without loading the weights. Capability reporting
/// uses this presence probe; the actual load and final fail-closed gate stay
/// in [`shared_embedder`].
pub fn embedder_pack_installed() -> bool {
    redimnet_pack_path().is_some()
}

/// The process-wide active ReDimNet2-B6 embedder, or `None` if the pack is not
/// installed.
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
    let path = redimnet_pack_path()?;
    let state = load_embedder_state(&path)?;
    let _ = SHARED_EMBEDDER.set(state);
    SHARED_EMBEDDER.get()
}

fn load_embedder_state(path: &Path) -> Option<SharedEmbedderState> {
    // ReDimNet2 is GGUF-only (a ggml-native artifact; no safetensors fast path).
    let embedder: Box<dyn SpeakerEmbedder> = Box::new(RedimNet2Embedder::from_oasr(path).ok()?);
    let identity = SpeakerEmbedderIdentity {
        embedding_dim: embedder.embedding_dim(),
        pack_fingerprint: pack_fingerprint(path)?,
    };
    Some(SharedEmbedderState { embedder, identity })
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

    #[test]
    fn redimnet_embedding_space_version_is_pinned() {
        assert_eq!(REDIMNET_EMBEDDING_SPACE_VERSION, "redimnet2-b6-cn-v1");
    }

    #[test]
    fn redimnet_pack_env_name_is_stable() {
        assert_eq!(REDIMNET_PACK_ENV, "OPENASR_REDIMNET_PACK");
        assert_eq!(REDIMNET_INSTALLED_DIR_HINT, "redimnet");
    }

    #[test]
    fn missing_embedder_reasons_name_redimnet2_b6_cn() {
        assert_eq!(SPEAKER_EMBEDDER_PACK_ID, "redimnet2-b6-cn");
        for reason in [
            VOICE_ID_EMBEDDER_PACK_MISSING_REASON,
            VOICE_MATCH_EMBEDDER_PACK_MISSING_REASON,
            DIARIZATION_EMBEDDER_LOAD_FAILED_REASON,
            REALTIME_DIARIZATION_EMBEDDER_MISSING_REASON,
            SPEAKER_EMBEDDER_PACK_LABEL,
        ] {
            assert!(
                reason.contains(SPEAKER_EMBEDDER_PACK_ID),
                "reason must name the install id: {reason}"
            );
            assert!(
                reason.contains("ReDimNet2-B6"),
                "reason must name the pack family: {reason}"
            );
            assert!(
                !reason.to_ascii_lowercase().contains("wespeaker"),
                "reason must not mention WeSpeaker: {reason}"
            );
            assert!(
                !reason.contains("active speaker-embedder"),
                "reason must not use the retired dual-path wording: {reason}"
            );
        }
    }
}
