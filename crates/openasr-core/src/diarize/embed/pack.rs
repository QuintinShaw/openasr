//! Runtime resolution of the speaker-embedder weight pack.
//!
//! Only ReDimNet2-B6 is supported (`OPENASR_REDIMNET_PACK` / installed model-id
//! hint `redimnet`, 192-d, ggml graph). When the pack is absent, resolution returns
//! `None` and callers fail closed with a clear "install redimnet2-b6-cn" error
//! rather than falling back to any other embedder.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use super::RedimNet2Embedder;
use super::SpeakerEmbedder;

static SHARED_EMBEDDER: OnceLock<SharedEmbedderState> = OnceLock::new();

const REDIMNET_PACK_ENV: &str = "OPENASR_REDIMNET_PACK";
const REDIMNET_INSTALLED_MODEL_ID_HINT: &str = "redimnet";

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

/// Fail-closed reason when the source-independent identity stage
/// (`diarize::voice_id::name_speakers_across_scopes`) cannot relate speaker
/// labels to known people because the embedder is unavailable, and skipping
/// silently would hide a real degrade: an enrolled person going unmatched, or
/// two in-decoder scopes staying artificially separate. See that function's
/// doc comment for exactly when this fires versus when the same absence is a
/// legitimate no-op.
pub const VOICE_ID_NAMING_EMBEDDER_MISSING_REASON: &str = "Voice ID needs the ReDimNet2-B6 speaker-embedder pack (redimnet2-b6-cn) to identify speakers, but it is missing or could not be loaded. Reinstall the pack, or turn off Voice ID.";

/// Human-readable label for ReDimNet2-B6's embedding space (documentation /
/// audit metadata only). The actual runtime compatibility gate is the pack's
/// content fingerprint (`SpeakerEmbedderIdentity::pack_fingerprint`, the
/// sha256 of the `.oasr` file -- for an installed pack, its content-addressed
/// object digest, which is the same value) plus `embedding_dim`, not this
/// string -- a re-export or repack of the same checkpoint keeps the same
/// fingerprint and stays compatible even if this label changes.
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
    crate::diarize::pack::resolve_pack(REDIMNET_PACK_ENV, REDIMNET_INSTALLED_MODEL_ID_HINT)
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

/// Content fingerprint of the embedder pack: `sha256:<hex>`.
///
/// An installed pack is a sealed content-addressed object *under this
/// process's own model store root*, so its fingerprint is read straight from
/// the object path without re-reading the weights -- the same trust the model
/// load path takes (see `content_store`'s integrity chain: hashed once at
/// admission, sealed read-only since, `model-pack verify` re-proves on
/// demand, and `content_store::trusted_object_digest`'s `models_root` anchor,
/// which is what tells a real installed object apart from a same-shaped path
/// elsewhere on disk). The value is identical to what hashing the bytes
/// returns, so enrollments fingerprinted either way interoperate. Anything
/// the gate declines -- an env-override pack, an unsealed object, or a path
/// outside the resolved model store -- is hashed the slow way: those are
/// arbitrary paths with no digest to trust.
fn pack_fingerprint(path: &Path) -> Option<String> {
    use sha2::{Digest, Sha256};
    use std::io::Read;

    let mut file = std::fs::File::open(path).ok()?;
    let sealed = file.metadata().ok()?.permissions().readonly();
    if let Some(models_root) = crate::content_store::default_models_root()
        && let Some(digest) =
            crate::content_store::trusted_object_digest(path, sealed, &models_root)
    {
        return Some(format!("sha256:{digest}"));
    }
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
        assert_eq!(REDIMNET_INSTALLED_MODEL_ID_HINT, "redimnet");
    }

    fn sha256_hex(bytes: &[u8]) -> String {
        use sha2::{Digest, Sha256};
        format!("{:x}", Sha256::digest(bytes))
    }

    fn write_object_at_layout(root: &Path, digest: &str, bytes: &[u8], read_only: bool) -> PathBuf {
        let object = root
            .join("models")
            .join("objects")
            .join("sha256")
            .join(digest)
            .join("content");
        std::fs::create_dir_all(object.parent().expect("object path has parent"))
            .expect("create digest dir");
        std::fs::write(&object, bytes).expect("write fixture");
        let mut permissions = std::fs::metadata(&object)
            .expect("stat fixture")
            .permissions();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            permissions.set_mode(if read_only { 0o444 } else { 0o644 });
        }
        #[cfg(not(unix))]
        permissions.set_readonly(read_only);
        std::fs::set_permissions(&object, permissions).expect("set fixture mode");
        object
    }

    /// The trusted half, pinned by construction: bytes that do not hash to
    /// the digest their path names can only fingerprint to that path digest
    /// if it was read, not recomputed. `pack_fingerprint` anchors trust to
    /// `default_models_root()`, so this test points `OPENASR_HOME` at the
    /// fixture's own tempdir -- nextest's per-test process isolation makes
    /// this safe (see AGENTS.md's note on why nextest, not `cargo test`, is
    /// required for this workspace).
    #[test]
    fn pack_fingerprint_trusts_a_sealed_object_without_hashing() {
        let dir = tempfile::tempdir().expect("tempdir");
        unsafe { std::env::set_var("OPENASR_HOME", dir.path()) };
        let named_digest = "ab".repeat(32);
        let bytes = b"embedder-fingerprint-trust-fixture";
        assert_ne!(
            sha256_hex(bytes),
            named_digest,
            "the fixture must not accidentally hash to the named digest"
        );
        let object = write_object_at_layout(dir.path(), &named_digest, bytes, true);

        assert_eq!(
            pack_fingerprint(&object),
            Some(format!("sha256:{named_digest}"))
        );
    }

    /// The fail-closed half as its own pin: an unsealed object's fingerprint
    /// is the hash of its bytes and never the digest its path claims.
    #[test]
    fn pack_fingerprint_unsealed_object_falls_back_to_hashing() {
        let dir = tempfile::tempdir().expect("tempdir");
        unsafe { std::env::set_var("OPENASR_HOME", dir.path()) };
        let named_digest = "ef".repeat(32);
        let bytes = b"embedder-fingerprint-fallback-fixture";
        let object = write_object_at_layout(dir.path(), &named_digest, bytes, false);

        let fingerprint = pack_fingerprint(&object).expect("fingerprint");
        assert_eq!(fingerprint, format!("sha256:{}", sha256_hex(bytes)));
        assert_ne!(fingerprint, format!("sha256:{named_digest}"));
    }

    /// The same adversarial shape pinned in `content_store`'s own tests: a
    /// same-shaped sealed path that is not under the resolved model store
    /// root must never be trusted, even though `OPENASR_HOME` is set and
    /// resolvable.
    #[test]
    fn pack_fingerprint_rejects_a_same_shaped_path_outside_the_models_root() {
        let dir = tempfile::tempdir().expect("tempdir");
        unsafe { std::env::set_var("OPENASR_HOME", dir.path()) };
        let attacker_digest = "99".repeat(32);
        let bytes = b"attacker-controlled-bytes";
        let object = dir
            .path()
            .join("totally-unrelated")
            .join("objects")
            .join("sha256")
            .join(&attacker_digest)
            .join("content");
        std::fs::create_dir_all(object.parent().expect("object path has parent"))
            .expect("create digest dir");
        std::fs::write(&object, bytes).expect("write fixture");
        let mut permissions = std::fs::metadata(&object)
            .expect("stat fixture")
            .permissions();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            permissions.set_mode(0o444);
        }
        #[cfg(not(unix))]
        permissions.set_readonly(true);
        std::fs::set_permissions(&object, permissions).expect("set fixture mode");

        let fingerprint = pack_fingerprint(&object).expect("fingerprint");
        assert_eq!(fingerprint, format!("sha256:{}", sha256_hex(bytes)));
        assert_ne!(fingerprint, format!("sha256:{attacker_digest}"));
    }

    #[test]
    fn missing_embedder_reasons_name_redimnet2_b6_cn() {
        assert_eq!(SPEAKER_EMBEDDER_PACK_ID, "redimnet2-b6-cn");
        for reason in [
            VOICE_ID_EMBEDDER_PACK_MISSING_REASON,
            VOICE_MATCH_EMBEDDER_PACK_MISSING_REASON,
            DIARIZATION_EMBEDDER_LOAD_FAILED_REASON,
            REALTIME_DIARIZATION_EMBEDDER_MISSING_REASON,
            VOICE_ID_NAMING_EMBEDDER_MISSING_REASON,
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
