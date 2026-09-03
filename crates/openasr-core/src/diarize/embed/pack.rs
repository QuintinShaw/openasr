//! Runtime resolution of the speaker-embedder weight pack.
//!
//! Default capability and catalog required pack remain ReDimNet2-B6
//! (`OPENASR_REDIMNET_PACK` / installed model-id hint `redimnet2-b6-cn`).
//! WeSpeaker ResNet loads only on an explicit preference or
//! `OPENASR_WESPEAKER_PACK`. A broken override fails closed and never falls
//! back.

#[cfg(test)]
use std::path::Path;
use std::path::PathBuf;

use crate::arch::GENERAL_ARCHITECTURE_KEY;
use crate::config::VoiceIdEmbedderPreference;
use crate::models::{
    aux_pack_registry::{AuxPackKind, REDIMNET2_GGML_ARCHITECTURE_ID},
    pack_verifier::{PackCandidate, PackRoute, PackVerifier},
};

use super::EmbedError;

const REDIMNET_PACK_ENV: &str = "OPENASR_REDIMNET_PACK";
const REDIMNET_INSTALLED_MODEL_ID_HINT: &str = "redimnet2-b6-cn";
const REDIMNET_PREFERRED_QUANT: &str = "fp16";
const WESPEAKER_PACK_ENV: &str = "OPENASR_WESPEAKER_PACK";
const WESPEAKER_INSTALLED_MODEL_ID_HINT: &str = "wespeaker";
const WESPEAKER_PREFERRED_QUANT: &str = "fp16";
pub(crate) const REDIMNET_PACK_PREFERENCE: crate::capability_pack::CapabilityPackPreference =
    crate::capability_pack::CapabilityPackPreference::new(
        SPEAKER_EMBEDDER_PACK_ID,
        REDIMNET_INSTALLED_MODEL_ID_HINT,
        REDIMNET_PREFERRED_QUANT,
    );
pub(crate) const WESPEAKER_PACK_PREFERENCE: crate::capability_pack::CapabilityPackPreference =
    crate::capability_pack::CapabilityPackPreference::new(
        WESPEAKER_EMBEDDER_PACK_ID,
        WESPEAKER_INSTALLED_MODEL_ID_HINT,
        WESPEAKER_PREFERRED_QUANT,
    );

/// Catalog / pull id of the default speaker-embedder pack.
pub const SPEAKER_EMBEDDER_PACK_ID: &str = "redimnet2-b6-cn";
/// Catalog / pull id of the optional WeSpeaker ResNet34 pack.
pub const WESPEAKER_EMBEDDER_PACK_ID: &str = "wespeaker-voxceleb-resnet34-lm";

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
/// Frontend identity label for ReDimNet2-B6. Stable contract, not a human label.
pub const REDIMNET_FRONTEND_VERSION: &str = "redimnet-tfmel-v1";
pub(crate) const WESPEAKER_EMBEDDING_SPACE_VERSION: &str = "wespeaker-resnet-v1";
pub(crate) const WESPEAKER_FRONTEND_VERSION: &str = "wespeaker-kaldi-hamming-v1";
const MODEL_ID_METADATA_KEY: &str = "openasr.model.id";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpeakerEmbedderFamily {
    ReDimNet2,
    WeSpeakerResNet,
}

impl SpeakerEmbedderFamily {
    pub fn architecture_id(self) -> &'static str {
        match self {
            Self::ReDimNet2 => REDIMNET2_GGML_ARCHITECTURE_ID,
            Self::WeSpeakerResNet => {
                crate::models::aux_pack_registry::WESPEAKER_RESNET_ARCHITECTURE_ID
            }
        }
    }

    /// Default catalog pull id when this family is selected but no size has
    /// been chosen. WeSpeaker sizes share the architecture; the installed
    /// pack's `openasr.model.id` is the source of truth on identity.
    pub fn default_catalog_model_id(self) -> &'static str {
        match self {
            Self::ReDimNet2 => SPEAKER_EMBEDDER_PACK_ID,
            Self::WeSpeakerResNet => WESPEAKER_EMBEDDER_PACK_ID,
        }
    }

    pub fn display_name(self) -> &'static str {
        match self {
            Self::ReDimNet2 => "ReDimNet2-B6",
            Self::WeSpeakerResNet => "WeSpeaker ResNet",
        }
    }

    pub fn pack_env(self) -> &'static str {
        match self {
            Self::ReDimNet2 => REDIMNET_PACK_ENV,
            Self::WeSpeakerResNet => WESPEAKER_PACK_ENV,
        }
    }

    pub fn missing_install_reason(self) -> String {
        format!(
            "{} speaker-embedder pack ({}) is not installed; install it or unset {}",
            self.display_name(),
            self.default_catalog_model_id(),
            self.pack_env(),
        )
    }
}

/// Content identity of one embedding space. Space labels live on this value
/// so Voice ID does not guess family from dimension or calibration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpeakerEmbedderIdentity {
    pub family: SpeakerEmbedderFamily,
    pub embedding_dim: usize,
    pub pack_fingerprint: String,
    pub catalog_model_id: String,
    pub space_family: &'static str,
    pub space_model_id: &'static str,
    pub model_version: &'static str,
    pub frontend_version: &'static str,
    pub calibration_version: &'static str,
}

impl SpeakerEmbedderIdentity {
    pub fn redimnet2(
        pack_fingerprint: impl Into<String>,
        catalog_model_id: impl Into<String>,
    ) -> Self {
        Self {
            family: SpeakerEmbedderFamily::ReDimNet2,
            embedding_dim: 192,
            pack_fingerprint: pack_fingerprint.into(),
            catalog_model_id: catalog_model_id.into(),
            space_family: "redimnet",
            space_model_id: "redimnet2-b6",
            model_version: REDIMNET_EMBEDDING_SPACE_VERSION,
            frontend_version: REDIMNET_FRONTEND_VERSION,
            calibration_version: crate::diarize::calibration::REDIMNET_CALIBRATION_VERSION,
        }
    }

    pub fn wespeaker_resnet(
        pack_fingerprint: impl Into<String>,
        catalog_model_id: impl Into<String>,
    ) -> Self {
        Self {
            family: SpeakerEmbedderFamily::WeSpeakerResNet,
            embedding_dim: 256,
            pack_fingerprint: pack_fingerprint.into(),
            catalog_model_id: catalog_model_id.into(),
            space_family: "wespeaker",
            space_model_id: "wespeaker-resnet",
            model_version: WESPEAKER_EMBEDDING_SPACE_VERSION,
            frontend_version: WESPEAKER_FRONTEND_VERSION,
            calibration_version: crate::diarize::calibration::WESPEAKER_CALIBRATION_VERSION,
        }
    }

    /// Fixture identity that must never collide with a production space.
    pub fn unlabeled_fixture(
        family: SpeakerEmbedderFamily,
        embedding_dim: usize,
        pack_fingerprint: impl Into<String>,
    ) -> Self {
        Self {
            family,
            embedding_dim,
            pack_fingerprint: pack_fingerprint.into(),
            catalog_model_id: "unknown".to_string(),
            space_family: "unknown",
            space_model_id: "unknown",
            model_version: "unknown",
            frontend_version: "unknown",
            calibration_version: "unknown",
        }
    }
}

pub(crate) struct PreparedSelectedEmbedder {
    pub(crate) family: SpeakerEmbedderFamily,
    pub(crate) catalog_model_id: String,
    pub(crate) source: PreparedEmbedderSource,
}

pub(crate) struct PreparedEmbedderSource {
    verified_pack: crate::models::pack_verifier::VerifiedPack,
    content_id: String,
}

impl PreparedEmbedderSource {
    pub(crate) fn into_parts(
        self,
    ) -> (
        crate::models::pack_verifier::VerifiedPack,
        crate::ggml_runtime::GgufRuntimeSourcePreflight,
        String,
    ) {
        let preflight = self.verified_pack.preflight().clone();
        (self.verified_pack, preflight, self.content_id)
    }
}

pub(crate) fn redimnet_pack_path() -> Option<PathBuf> {
    crate::diarize::pack::resolve_pack(REDIMNET_PACK_ENV, REDIMNET_PACK_PREFERENCE)
}

pub(crate) fn wespeaker_pack_path() -> Option<PathBuf> {
    crate::diarize::pack::resolve_pack(WESPEAKER_PACK_ENV, WESPEAKER_PACK_PREFERENCE)
}

/// Default capability probe: whether the ReDimNet2-B6 pack is resolvable.
/// WeSpeaker presence does not flip diarization capability on.
pub fn embedder_pack_installed() -> bool {
    redimnet_pack_path().is_some()
}

fn wespeaker_env_override_set() -> bool {
    std::env::var_os(WESPEAKER_PACK_ENV).is_some_and(|value| !value.is_empty())
}

/// Select the embedder family and pin its pack identity without constructing a
/// runtime. `OPENASR_WESPEAKER_PACK` forces WeSpeaker. Explicit WeSpeaker
/// preference never falls back to ReDimNet. A present-but-invalid pack fails
/// closed.
pub(crate) fn prepare_embedder(
    preference: VoiceIdEmbedderPreference,
) -> Result<Option<PreparedSelectedEmbedder>, EmbedError> {
    let (family, path) =
        if wespeaker_env_override_set() || preference == VoiceIdEmbedderPreference::WeSpeaker {
            (
                SpeakerEmbedderFamily::WeSpeakerResNet,
                wespeaker_pack_path(),
            )
        } else {
            (SpeakerEmbedderFamily::ReDimNet2, redimnet_pack_path())
        };
    let Some(path) = path else {
        if family == SpeakerEmbedderFamily::WeSpeakerResNet {
            return Err(EmbedError::Unavailable(family.missing_install_reason()));
        }
        return Ok(None);
    };
    let verified_pack = PackVerifier
        .verify_candidate(PackCandidate::new(&path))
        .map_err(|error| EmbedError::Unavailable(format!("{}: {error}", path.display())))?;
    if !matches!(
        verified_pack.route(),
        PackRoute::Aux {
            kind: AuxPackKind::Diarization,
            ..
        }
    ) {
        return Err(EmbedError::Unavailable(format!(
            "{}: pack route is not auxiliary diarization: {:?}",
            path.display(),
            verified_pack.route()
        )));
    }
    let architecture = verified_pack
        .preflight()
        .metadata()
        .get_string(GENERAL_ARCHITECTURE_KEY)
        .map(str::trim)
        .unwrap_or("");
    if architecture != family.architecture_id() {
        return Err(EmbedError::Unavailable(format!(
            "{}: general.architecture is '{architecture}', expected '{}' for {}",
            path.display(),
            family.architecture_id(),
            family.default_catalog_model_id()
        )));
    }
    let catalog_model_id = verified_pack
        .preflight()
        .metadata()
        .get_string(MODEL_ID_METADATA_KEY)
        .map(str::trim)
        .filter(|id| !id.is_empty())
        .unwrap_or(family.default_catalog_model_id())
        .to_string();
    let content_id = verified_pack.content_id().to_string();
    Ok(Some(PreparedSelectedEmbedder {
        family,
        catalog_model_id,
        source: PreparedEmbedderSource {
            verified_pack,
            content_id,
        },
    }))
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
#[cfg(test)]
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
        assert_eq!(REDIMNET_INSTALLED_MODEL_ID_HINT, "redimnet2-b6-cn");
        assert_eq!(WESPEAKER_PACK_ENV, "OPENASR_WESPEAKER_PACK");
        assert_eq!(WESPEAKER_EMBEDDER_PACK_ID, "wespeaker-voxceleb-resnet34-lm");
    }

    #[test]
    fn pack_fingerprint_tracks_same_path_replacement_and_deletion() {
        let dir = tempfile::tempdir().expect("tempdir");
        let pack = dir.path().join("redimnet.oasr");
        std::fs::write(&pack, b"GGUFredimnet-content-a").expect("write a");
        let fingerprint_a = pack_fingerprint(&pack).expect("fingerprint a");

        std::fs::write(&pack, b"GGUFredimnet-content-b").expect("replace b");
        let fingerprint_b = pack_fingerprint(&pack).expect("fingerprint b");
        assert_ne!(
            fingerprint_a, fingerprint_b,
            "same-path replacement must produce a new content identity"
        );

        std::fs::remove_file(&pack).expect("delete pack");
        assert!(
            pack_fingerprint(&pack).is_none(),
            "deleted pack must not retain a content identity"
        );
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
        let named_digest = "ab".repeat(32);
        let bytes = b"embedder-fingerprint-trust-fixture";
        assert_ne!(
            sha256_hex(bytes),
            named_digest,
            "the fixture must not accidentally hash to the named digest"
        );
        let object = write_object_at_layout(dir.path(), &named_digest, bytes, true);

        crate::test_process_env::with_test_process_env(
            [("OPENASR_HOME", Some(dir.path().as_os_str().to_os_string()))],
            || {
                assert_eq!(
                    pack_fingerprint(&object),
                    Some(format!("sha256:{named_digest}"))
                );
            },
        );
    }

    /// The fail-closed half as its own pin: an unsealed object's fingerprint
    /// is the hash of its bytes and never the digest its path claims.
    #[test]
    fn pack_fingerprint_unsealed_object_falls_back_to_hashing() {
        let dir = tempfile::tempdir().expect("tempdir");
        let named_digest = "ef".repeat(32);
        let bytes = b"embedder-fingerprint-fallback-fixture";
        let object = write_object_at_layout(dir.path(), &named_digest, bytes, false);

        crate::test_process_env::with_test_process_env(
            [("OPENASR_HOME", Some(dir.path().as_os_str().to_os_string()))],
            || {
                let fingerprint = pack_fingerprint(&object).expect("fingerprint");
                assert_eq!(fingerprint, format!("sha256:{}", sha256_hex(bytes)));
                assert_ne!(fingerprint, format!("sha256:{named_digest}"));
            },
        );
    }

    /// The same adversarial shape pinned in `content_store`'s own tests: a
    /// same-shaped sealed path that is not under the resolved model store
    /// root must never be trusted, even though `OPENASR_HOME` is set and
    /// resolvable.
    #[test]
    fn pack_fingerprint_rejects_a_same_shaped_path_outside_the_models_root() {
        let dir = tempfile::tempdir().expect("tempdir");
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

        crate::test_process_env::with_test_process_env(
            [("OPENASR_HOME", Some(dir.path().as_os_str().to_os_string()))],
            || {
                let fingerprint = pack_fingerprint(&object).expect("fingerprint");
                assert_eq!(fingerprint, format!("sha256:{}", sha256_hex(bytes)));
                assert_ne!(fingerprint, format!("sha256:{attacker_digest}"));
            },
        );
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

    #[test]
    fn selected_wespeaker_reason_names_its_pack_id() {
        let reason = SpeakerEmbedderFamily::WeSpeakerResNet.missing_install_reason();
        assert!(reason.contains(WESPEAKER_EMBEDDER_PACK_ID));
        assert!(reason.contains("WeSpeaker ResNet"));
        assert!(reason.contains(WESPEAKER_PACK_ENV));
        assert_eq!(
            SpeakerEmbedderFamily::ReDimNet2.default_catalog_model_id(),
            SPEAKER_EMBEDDER_PACK_ID
        );
    }

    fn isolated_empty_embedder_home() {
        let dir = tempfile::tempdir().expect("tempdir");
        crate::test_process_env::with_test_process_env(
            [
                ("OPENASR_HOME", Some(dir.path().as_os_str().to_os_string())),
                (WESPEAKER_PACK_ENV, None),
                (REDIMNET_PACK_ENV, None),
                ("OPENASR_MODELS_DIR", None),
            ],
            || {
                let error = match prepare_embedder(VoiceIdEmbedderPreference::WeSpeaker) {
                    Ok(_) => panic!("WeSpeaker preference with no pack must fail closed"),
                    Err(error) => error,
                };
                match error {
                    EmbedError::Unavailable(reason) => {
                        assert!(
                            reason.contains(WESPEAKER_EMBEDDER_PACK_ID),
                            "missing WeSpeaker pack must name its catalog id, got {reason}"
                        );
                        assert!(
                            reason.contains("WeSpeaker ResNet"),
                            "missing WeSpeaker pack must name the family, got {reason}"
                        );
                    }
                    other => panic!("expected Unavailable, got {other}"),
                }
                assert!(
                    prepare_embedder(VoiceIdEmbedderPreference::ReDimNet2)
                        .expect("ReDimNet absence is not an error at prepare")
                        .is_none(),
                    "default ReDimNet preference with no pack must yield None, not a silent WeSpeaker skip"
                );
            },
        );
    }

    #[test]
    fn prepare_embedder_wespeaker_preference_fails_closed_when_pack_missing() {
        isolated_empty_embedder_home();
    }

    #[test]
    fn identity_constructors_carry_space_labels_without_guessing() {
        let redimnet = SpeakerEmbedderIdentity::redimnet2("sha256:rd", "redimnet2-b6-cn");
        assert_eq!(redimnet.space_family, "redimnet");
        assert_eq!(redimnet.space_model_id, "redimnet2-b6");
        assert_eq!(redimnet.frontend_version, REDIMNET_FRONTEND_VERSION);
        assert_eq!(redimnet.embedding_dim, 192);

        let wespeaker = SpeakerEmbedderIdentity::wespeaker_resnet(
            "sha256:ws",
            "wespeaker-voxceleb-resnet152-lm",
        );
        assert_eq!(wespeaker.space_family, "wespeaker");
        assert_eq!(
            wespeaker.catalog_model_id,
            "wespeaker-voxceleb-resnet152-lm"
        );
        assert_eq!(wespeaker.frontend_version, WESPEAKER_FRONTEND_VERSION);
        assert_eq!(wespeaker.embedding_dim, 256);

        let fixture = SpeakerEmbedderIdentity::unlabeled_fixture(
            SpeakerEmbedderFamily::ReDimNet2,
            2,
            "voice-id-identity-tests-v1",
        );
        assert_eq!(fixture.space_family, "unknown");
        assert_eq!(fixture.embedding_dim, 2);
        assert_ne!(fixture.space_family, redimnet.space_family);
    }
}
