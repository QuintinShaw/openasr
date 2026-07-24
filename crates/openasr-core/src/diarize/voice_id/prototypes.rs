//! Quality-aware prototype construction.
//!
//! Sample embeddings are the source of truth. Prototypes are a rebuildable index:
//! quality-weighted clustering inside one EmbeddingSpace, medoid (real sample)
//! selection, and a per-person active cap.

use super::domain::{PersonPrototype, PrototypeMember, SampleEmbedding, SampleQuality};
use super::ids::{PersonId, PrototypeId, SampleId};
use super::space::{EmbeddingSpace, MATCHER_POLICY_VERSION};
use crate::diarize::contract::SpeakerEmbedding;

/// Maximum active prototypes exposed per person in one embedding space.
pub const MAX_PROTOTYPES_PER_PERSON: usize = 3;
/// Default cosine-distance merge threshold used when calibration does not
/// supply a tighter one. Cosine distance = 1 - cosine similarity.
pub const DEFAULT_CLUSTER_COSINE_DISTANCE: f32 = 0.35;

#[derive(Debug, Clone)]
pub struct PrototypeSample {
    pub sample_id: SampleId,
    pub embedding: SpeakerEmbedding,
    pub quality: SampleQuality,
}

/// Build up to [`MAX_PROTOTYPES_PER_PERSON`] quality-aware medoid prototypes.
///
/// Clustering is single-linkage on cosine distance with a calibrated threshold.
/// The medoid of each cluster is the real member that minimizes the
/// quality-weighted average distance to other members (not a mean vector).
pub fn build_person_prototypes(
    person_id: &PersonId,
    space: &EmbeddingSpace,
    samples: &[PrototypeSample],
    cluster_cosine_distance: f32,
) -> Vec<PersonPrototype> {
    if samples.is_empty() || !space.is_matchable() {
        return Vec::new();
    }

    let dim = space.dimension;
    let usable: Vec<&PrototypeSample> = samples
        .iter()
        .filter(|s| s.embedding.dim() == dim)
        .collect();
    if usable.is_empty() {
        return Vec::new();
    }

    let clusters = cluster_samples(&usable, cluster_cosine_distance.clamp(0.05, 0.8));
    let mut prototypes: Vec<(f32, PersonPrototype)> = clusters
        .into_iter()
        .filter_map(|cluster| {
            let (medoid_idx, medoid_quality) = select_medoid(&cluster)?;
            let medoid = cluster[medoid_idx];
            let members = cluster
                .iter()
                .map(|sample| PrototypeMember {
                    sample_id: sample.sample_id.clone(),
                    quality_weight: sample.quality.weight(),
                })
                .collect::<Vec<_>>();
            // Rank key: higher total quality and larger clusters first so the
            // cap keeps the most informative prototypes.
            let rank = members.iter().map(|m| m.quality_weight).sum::<f32>()
                + medoid_quality
                + (members.len() as f32) * 0.05;
            Some((
                rank,
                PersonPrototype {
                    prototype_id: PrototypeId::generate(),
                    person_id: person_id.clone(),
                    space: space.clone(),
                    medoid_sample_id: medoid.sample_id.clone(),
                    medoid_embedding: medoid.embedding.clone(),
                    policy_version: MATCHER_POLICY_VERSION.to_string(),
                    members,
                },
            ))
        })
        .collect();

    prototypes.sort_by(|a, b| b.0.total_cmp(&a.0));
    prototypes.truncate(MAX_PROTOTYPES_PER_PERSON);
    prototypes.into_iter().map(|(_, p)| p).collect()
}

/// Build [`PrototypeSample`] values from stored embeddings + quality rows.
#[allow(dead_code)] // used by future service rebuild helpers and external callers
pub fn prototype_samples_from_embeddings(
    embeddings: &[SampleEmbedding],
    qualities: &[(SampleId, SampleQuality)],
) -> Vec<PrototypeSample> {
    embeddings
        .iter()
        .filter_map(|embedding| {
            let quality = qualities
                .iter()
                .find(|(id, _)| id == &embedding.sample_id)
                .map(|(_, q)| q.clone())?;
            Some(PrototypeSample {
                sample_id: embedding.sample_id.clone(),
                embedding: embedding.embedding.clone(),
                quality,
            })
        })
        .collect()
}

fn cluster_samples<'a>(
    samples: &[&'a PrototypeSample],
    max_distance: f32,
) -> Vec<Vec<&'a PrototypeSample>> {
    let n = samples.len();
    let mut parent: Vec<usize> = (0..n).collect();

    fn find(parent: &mut [usize], mut i: usize) -> usize {
        while parent[i] != i {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        i
    }

    for i in 0..n {
        for j in (i + 1)..n {
            let distance = 1.0 - samples[i].embedding.cosine(&samples[j].embedding);
            if distance <= max_distance {
                let pi = find(&mut parent, i);
                let pj = find(&mut parent, j);
                if pi != pj {
                    // Union by attaching the lower-quality root under the higher
                    // one so medoid selection starts from a better representative.
                    if samples[pi].quality.weight() >= samples[pj].quality.weight() {
                        parent[pj] = pi;
                    } else {
                        parent[pi] = pj;
                    }
                }
            }
        }
    }

    let mut groups: Vec<Vec<&PrototypeSample>> = Vec::new();
    let mut root_index: Vec<Option<usize>> = vec![None; n];
    for (i, sample) in samples.iter().enumerate() {
        let root = find(&mut parent, i);
        if let Some(group_idx) = root_index[root] {
            groups[group_idx].push(*sample);
        } else {
            root_index[root] = Some(groups.len());
            groups.push(vec![*sample]);
        }
    }
    groups
}

fn select_medoid(cluster: &[&PrototypeSample]) -> Option<(usize, f32)> {
    if cluster.is_empty() {
        return None;
    }
    if cluster.len() == 1 {
        return Some((0, cluster[0].quality.weight()));
    }

    let mut best_idx = 0usize;
    let mut best_score = f32::INFINITY;
    for (i, candidate) in cluster.iter().enumerate() {
        let mut weighted_distance = 0.0f32;
        let mut weight_sum = 0.0f32;
        for (j, other) in cluster.iter().enumerate() {
            if i == j {
                continue;
            }
            let w = other.quality.weight();
            weighted_distance += (1.0 - candidate.embedding.cosine(&other.embedding)) * w;
            weight_sum += w;
        }
        // Prefer higher-quality candidates when distances tie: subtract a small
        // quality term so low-quality outliers are not chosen as medoids.
        let score = if weight_sum > 0.0 {
            weighted_distance / weight_sum - candidate.quality.weight() * 0.01
        } else {
            0.0 - candidate.quality.weight() * 0.01
        };
        if score < best_score {
            best_score = score;
            best_idx = i;
        }
    }
    Some((best_idx, cluster[best_idx].quality.weight()))
}

/// Person-level score for one prototype against a query embedding.
///
/// `medoid_similarity + bounded support bonus from the top-2 quality-weighted
/// member similarities`. The bonus is capped so extra samples cannot dominate.
pub fn score_prototype(
    prototype: &PersonPrototype,
    query: &SpeakerEmbedding,
    member_embeddings: &[(SampleId, SpeakerEmbedding, f32)],
) -> f32 {
    if prototype.medoid_embedding.dim() != query.dim() {
        return f32::NEG_INFINITY;
    }
    let medoid_sim = prototype.medoid_embedding.cosine(query);
    let mut member_scores: Vec<f32> = member_embeddings
        .iter()
        .filter(|(id, emb, _)| {
            emb.dim() == query.dim()
                && prototype
                    .members
                    .iter()
                    .any(|member| &member.sample_id == id)
        })
        .map(|(_, emb, weight)| emb.cosine(query) * weight.clamp(0.05, 1.0))
        .collect();
    member_scores.sort_by(|a, b| b.total_cmp(a));
    let support = member_scores.iter().take(2).sum::<f32>();
    // Bound the support term so a person with many near-duplicate samples cannot
    // outscore a well-matched single-sample person without real acoustic support.
    let support_bonus = (support * 0.08).clamp(0.0, 0.06);
    medoid_sim + support_bonus
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diarize::calibration::REDIMNET_CALIBRATION_VERSION;
    use crate::diarize::voice_id::space::{MATCHER_POLICY_VERSION, REDIMNET_FRONTEND_VERSION};

    fn space() -> EmbeddingSpace {
        EmbeddingSpace::from_parts(
            2,
            "sha256:test",
            "test",
            "test-model",
            "v1",
            REDIMNET_FRONTEND_VERSION,
            REDIMNET_CALIBRATION_VERSION,
            MATCHER_POLICY_VERSION,
        )
    }

    fn q(speech: f32, snr: f32) -> SampleQuality {
        SampleQuality {
            speech_seconds: speech,
            snr_estimate: snr,
            clipping_ratio: 0.0,
            vad_coverage: 0.8,
            accepted_reason: "test".into(),
        }
    }

    #[test]
    fn low_quality_outlier_is_not_medoid_when_better_cluster_exists() {
        // Generate real ids then override via constructing samples with generate.
        let person = PersonId::generate();
        let good = PrototypeSample {
            sample_id: SampleId::generate(),
            embedding: SpeakerEmbedding::l2_normalized(vec![1.0, 0.0]),
            quality: q(12.0, 25.0),
        };
        let also_good = PrototypeSample {
            sample_id: SampleId::generate(),
            embedding: SpeakerEmbedding::l2_normalized(vec![0.98, 0.1]),
            quality: q(11.0, 22.0),
        };
        let bad = PrototypeSample {
            sample_id: SampleId::generate(),
            embedding: SpeakerEmbedding::l2_normalized(vec![0.0, 1.0]),
            quality: q(5.5, 6.0),
        };
        let prototypes = build_person_prototypes(
            &person,
            &space(),
            &[good.clone(), also_good.clone(), bad.clone()],
            0.3,
        );
        assert!(!prototypes.is_empty());
        // The first prototype (highest rank) should pick a high-quality sample
        // from the tight cluster, never the orthogonal low-quality outlier when
        // a better cluster exists.
        let top = &prototypes[0];
        assert_ne!(top.medoid_sample_id, bad.sample_id);
        assert!(
            top.medoid_sample_id == good.sample_id || top.medoid_sample_id == also_good.sample_id
        );
    }

    #[test]
    fn prototype_cap_is_respected() {
        let person = PersonId::generate();
        let mut samples = Vec::new();
        // Five mutually distant samples -> five clusters, capped to 3.
        let dirs = [
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![-1.0, 0.0],
            vec![0.0, -1.0],
            vec![0.7, 0.7],
        ];
        for dir in dirs {
            samples.push(PrototypeSample {
                sample_id: SampleId::generate(),
                embedding: SpeakerEmbedding::l2_normalized(dir),
                quality: q(10.0, 20.0),
            });
        }
        let prototypes = build_person_prototypes(&person, &space(), &samples, 0.1);
        assert!(prototypes.len() <= MAX_PROTOTYPES_PER_PERSON);
    }

    #[test]
    fn support_bonus_is_bounded() {
        let person = PersonId::generate();
        let medoid_id = SampleId::generate();
        let member_id = SampleId::generate();
        let prototype = PersonPrototype {
            prototype_id: PrototypeId::generate(),
            person_id: person,
            space: space(),
            medoid_sample_id: medoid_id.clone(),
            medoid_embedding: SpeakerEmbedding::l2_normalized(vec![1.0, 0.0]),
            policy_version: MATCHER_POLICY_VERSION.into(),
            members: vec![
                PrototypeMember {
                    sample_id: medoid_id.clone(),
                    quality_weight: 1.0,
                },
                PrototypeMember {
                    sample_id: member_id.clone(),
                    quality_weight: 1.0,
                },
            ],
        };
        let query = SpeakerEmbedding::l2_normalized(vec![1.0, 0.0]);
        let members = vec![
            (
                medoid_id,
                SpeakerEmbedding::l2_normalized(vec![1.0, 0.0]),
                1.0,
            ),
            (
                member_id,
                SpeakerEmbedding::l2_normalized(vec![1.0, 0.0]),
                1.0,
            ),
        ];
        let score = score_prototype(&prototype, &query, &members);
        assert!(score <= 1.0 + 0.06 + 1e-5);
        assert!(score >= 1.0);
    }
}
