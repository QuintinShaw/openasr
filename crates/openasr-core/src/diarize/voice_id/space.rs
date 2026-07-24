//! Complete embedding-space identity for Voice ID.
//!
//! Two embeddings are comparable only when their `space_id` values are equal.
//! Matching dimension alone is never enough: pack fingerprint, model, frontend,
//! calibration, and matcher policy all participate in the identity.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::diarize::calibration::{
    REDIMNET_CALIBRATION, REDIMNET_CALIBRATION_VERSION, SpeakerCalibrationProfile,
    WESPEAKER_CALIBRATION, WESPEAKER_CALIBRATION_VERSION,
};
use crate::diarize::embed::SpeakerEmbedderIdentity;

const REDIMNET_MODEL_VERSION: &str = "redimnet2-b6-cn-v1";

/// Matcher policy version for quality-aware medoid prototypes + person-level
/// margin. Bump when scoring, clustering distance, prototype cap, or support
/// bonus rules change in a way that invalidates stored prototypes or thresholds.
pub const MATCHER_POLICY_VERSION: &str = "person-medoid-v1";

/// Frontend identity labels. These are stable contracts, not human labels.
pub const WESPEAKER_FRONTEND_VERSION: &str = "wespeaker-fbank-v1";
pub const REDIMNET_FRONTEND_VERSION: &str = "redimnet-tfmel-v1";

/// Marker used when a v1 profile is imported without full model/calibration
/// provenance. Such spaces are never eligible for matching.
pub const LEGACY_UNVERIFIABLE_V1_MARKER: &str = "legacy-unverifiable-v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbeddingSpace {
    pub space_id: String,
    pub dimension: usize,
    pub pack_fingerprint: String,
    pub embedder_family: String,
    pub embedder_model_id: String,
    pub embedder_model_version: String,
    pub frontend_version: String,
    pub calibration_version: String,
    pub matcher_policy_version: String,
    /// True only for migrated v1 profiles whose model/calibration provenance
    /// cannot be reconstructed. Matchers must refuse these spaces.
    #[serde(default)]
    pub legacy_unverifiable: bool,
}

impl EmbeddingSpace {
    pub fn from_parts(
        dimension: usize,
        pack_fingerprint: impl Into<String>,
        embedder_family: impl Into<String>,
        embedder_model_id: impl Into<String>,
        embedder_model_version: impl Into<String>,
        frontend_version: impl Into<String>,
        calibration_version: impl Into<String>,
        matcher_policy_version: impl Into<String>,
    ) -> Self {
        let mut space = Self {
            space_id: String::new(),
            dimension,
            pack_fingerprint: pack_fingerprint.into(),
            embedder_family: embedder_family.into(),
            embedder_model_id: embedder_model_id.into(),
            embedder_model_version: embedder_model_version.into(),
            frontend_version: frontend_version.into(),
            calibration_version: calibration_version.into(),
            matcher_policy_version: matcher_policy_version.into(),
            legacy_unverifiable: false,
        };
        space.space_id = space.compute_space_id();
        space
    }

    /// Build a matchable space for the currently active embedder identity plus
    /// the embedder's calibration profile.
    pub fn for_active_embedder(
        identity: &SpeakerEmbedderIdentity,
        calibration: SpeakerCalibrationProfile,
    ) -> Self {
        let (family, model_id, model_version, frontend, cal_version) =
            describe_embedder(identity.embedding_dim, calibration);
        Self::from_parts(
            identity.embedding_dim,
            identity.pack_fingerprint.clone(),
            family,
            model_id,
            model_version,
            frontend,
            cal_version,
            MATCHER_POLICY_VERSION,
        )
    }

    /// Reconstruct a non-matchable legacy space from a v1 profile's dim + pack
    /// fingerprint. The space is retained for export/delete/display only.
    pub fn legacy_unverifiable_v1(dimension: usize, pack_fingerprint: impl Into<String>) -> Self {
        let mut space = Self {
            space_id: String::new(),
            dimension,
            pack_fingerprint: pack_fingerprint.into(),
            embedder_family: LEGACY_UNVERIFIABLE_V1_MARKER.to_string(),
            embedder_model_id: LEGACY_UNVERIFIABLE_V1_MARKER.to_string(),
            embedder_model_version: LEGACY_UNVERIFIABLE_V1_MARKER.to_string(),
            frontend_version: LEGACY_UNVERIFIABLE_V1_MARKER.to_string(),
            calibration_version: LEGACY_UNVERIFIABLE_V1_MARKER.to_string(),
            matcher_policy_version: LEGACY_UNVERIFIABLE_V1_MARKER.to_string(),
            legacy_unverifiable: true,
        };
        space.space_id = space.compute_space_id();
        space
    }

    pub fn is_matchable(&self) -> bool {
        !self.legacy_unverifiable
            && self.embedder_family != LEGACY_UNVERIFIABLE_V1_MARKER
            && self.calibration_version != LEGACY_UNVERIFIABLE_V1_MARKER
            && self.matcher_policy_version == MATCHER_POLICY_VERSION
    }

    pub fn is_comparable_to(&self, other: &EmbeddingSpace) -> bool {
        self.is_matchable() && other.is_matchable() && self.space_id == other.space_id
    }

    fn compute_space_id(&self) -> String {
        // Canonical serialization: fixed field order, no whitespace, so the
        // space_id is stable across processes and language boundaries.
        let canonical = format!(
            concat!(
                "{{\"calibration_version\":{},",
                "\"dimension\":{},",
                "\"embedder_family\":{},",
                "\"embedder_model_id\":{},",
                "\"embedder_model_version\":{},",
                "\"frontend_version\":{},",
                "\"legacy_unverifiable\":{},",
                "\"matcher_policy_version\":{},",
                "\"pack_fingerprint\":{}}}"
            ),
            json_string(&self.calibration_version),
            self.dimension,
            json_string(&self.embedder_family),
            json_string(&self.embedder_model_id),
            json_string(&self.embedder_model_version),
            json_string(&self.frontend_version),
            if self.legacy_unverifiable {
                "true"
            } else {
                "false"
            },
            json_string(&self.matcher_policy_version),
            json_string(&self.pack_fingerprint),
        );
        let digest = Sha256::digest(canonical.as_bytes());
        format!("space_sha256:{digest:x}")
    }
}

fn describe_embedder(
    embedding_dim: usize,
    calibration: SpeakerCalibrationProfile,
) -> (
    &'static str,
    &'static str,
    &'static str,
    &'static str,
    &'static str,
) {
    // Calibration identity is structural: compare by the enrollment threshold
    // pair that uniquely distinguishes the two shipped profiles today.
    let is_redimnet = calibration.enrollment_default_match_similarity
        == REDIMNET_CALIBRATION.enrollment_default_match_similarity
        && calibration.enrollment_match_margin == REDIMNET_CALIBRATION.enrollment_match_margin
        && embedding_dim == 192;
    if is_redimnet {
        (
            "redimnet",
            "redimnet2-b6",
            REDIMNET_MODEL_VERSION,
            REDIMNET_FRONTEND_VERSION,
            REDIMNET_CALIBRATION_VERSION,
        )
    } else if embedding_dim == 256
        && calibration.enrollment_default_match_similarity
            == WESPEAKER_CALIBRATION.enrollment_default_match_similarity
    {
        (
            "wespeaker",
            "wespeaker-voxceleb-resnet34-lm",
            "wespeaker-v1",
            WESPEAKER_FRONTEND_VERSION,
            WESPEAKER_CALIBRATION_VERSION,
        )
    } else {
        // Unknown embedder: still form a space so fail-closed comparison works,
        // but use explicit unknown markers rather than guessing a family.
        ("unknown", "unknown", "unknown", "unknown", "unknown")
    }
}

fn json_string(value: &str) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| format!("\"{value}\""))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn space_id_is_stable_and_sensitive_to_calibration() {
        let a = EmbeddingSpace::from_parts(
            192,
            "sha256:abc",
            "redimnet",
            "redimnet2-b6",
            "redimnet2-b6-cn-v1",
            REDIMNET_FRONTEND_VERSION,
            REDIMNET_CALIBRATION_VERSION,
            MATCHER_POLICY_VERSION,
        );
        let b = EmbeddingSpace::from_parts(
            192,
            "sha256:abc",
            "redimnet",
            "redimnet2-b6",
            "redimnet2-b6-cn-v1",
            REDIMNET_FRONTEND_VERSION,
            REDIMNET_CALIBRATION_VERSION,
            MATCHER_POLICY_VERSION,
        );
        let c = EmbeddingSpace::from_parts(
            192,
            "sha256:abc",
            "redimnet",
            "redimnet2-b6",
            "redimnet2-b6-cn-v1",
            REDIMNET_FRONTEND_VERSION,
            "redimnet2-b6-cal-v2",
            MATCHER_POLICY_VERSION,
        );
        assert_eq!(a.space_id, b.space_id);
        assert_ne!(a.space_id, c.space_id);
        assert!(a.is_comparable_to(&b));
        assert!(!a.is_comparable_to(&c));
    }

    #[test]
    fn legacy_space_is_not_matchable() {
        let legacy = EmbeddingSpace::legacy_unverifiable_v1(256, "sha256:old");
        assert!(legacy.legacy_unverifiable);
        assert!(!legacy.is_matchable());
        let modern = EmbeddingSpace::from_parts(
            256,
            "sha256:old",
            "wespeaker",
            "wespeaker-voxceleb-resnet34-lm",
            "wespeaker-v1",
            WESPEAKER_FRONTEND_VERSION,
            WESPEAKER_CALIBRATION_VERSION,
            MATCHER_POLICY_VERSION,
        );
        assert!(!legacy.is_comparable_to(&modern));
    }
}
