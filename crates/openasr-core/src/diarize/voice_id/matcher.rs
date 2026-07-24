//! Person-level Voice ID matcher.
//!
//! Ranking is always by person_id. Multiple samples/prototypes of the same
//! person reinforce that person and never become their own runner-up.

use super::domain::{CandidateScope, PersonMatch, PersonPrototype, PersonStatus};
use super::ids::PersonId;
use super::prototypes::score_prototype;
use super::space::EmbeddingSpace;
use crate::diarize::contract::SpeakerEmbedding;

#[derive(Debug, Clone)]
pub struct MatcherPerson {
    pub person_id: PersonId,
    pub display_name: String,
    pub status: PersonStatus,
    pub prototypes: Vec<PersonPrototype>,
    /// sample_id -> (embedding, quality_weight) for support bonus.
    pub member_embeddings: Vec<(super::ids::SampleId, SpeakerEmbedding, f32)>,
}

#[derive(Debug, Clone)]
pub struct PersonMatcher {
    space: EmbeddingSpace,
    persons: Vec<MatcherPerson>,
    accept_threshold: f32,
    margin: f32,
}

impl PersonMatcher {
    pub fn new(
        space: EmbeddingSpace,
        persons: Vec<MatcherPerson>,
        accept_threshold: f32,
        margin: f32,
    ) -> Self {
        let persons = persons
            .into_iter()
            .filter(|person| {
                person.status.allows_matching()
                    && person
                        .prototypes
                        .iter()
                        .all(|proto| proto.space.is_comparable_to(&space))
            })
            .collect();
        Self {
            space,
            persons,
            accept_threshold: accept_threshold.clamp(0.0, 1.0),
            margin: margin.max(0.0),
        }
    }

    pub fn space(&self) -> &EmbeddingSpace {
        &self.space
    }

    pub fn is_empty(&self) -> bool {
        self.persons.is_empty()
    }

    pub fn with_scope(mut self, scope: &CandidateScope) -> Self {
        match scope {
            CandidateScope::AllCompatible => self,
            CandidateScope::Explicit(ids) => {
                if ids.is_empty() {
                    self.persons.clear();
                } else {
                    self.persons
                        .retain(|person| ids.iter().any(|id| id == &person.person_id));
                }
                self
            }
        }
    }

    /// Best person match under the matcher's default accept threshold + margin.
    ///
    /// Returns `None` (Unknown) when:
    /// - no candidates
    /// - best score < accept threshold
    /// - a runner-up person exists and margin is not met
    pub fn best_match(&self, query: &SpeakerEmbedding) -> Option<PersonMatch> {
        self.best_match_with_gates(query, self.accept_threshold, self.margin, 0.0)
    }

    /// Best person match with caller-supplied gates (batch vs streaming anchors).
    ///
    /// `threshold_tolerance` lowers the accept floor (used by native streaming
    /// once a person already owns a session anchor). Ranking stays per-person:
    /// a person's own prototypes never become their runner-up.
    pub fn best_match_with_gates(
        &self,
        query: &SpeakerEmbedding,
        accept_threshold: f32,
        margin: f32,
        threshold_tolerance: f32,
    ) -> Option<PersonMatch> {
        if !self.space.is_matchable() || query.dim() != self.space.dimension {
            return None;
        }

        let accept_threshold = accept_threshold.clamp(0.0, 1.0);
        let margin = margin.max(0.0);
        let threshold_tolerance = threshold_tolerance.max(0.0);
        let effective_threshold = (accept_threshold - threshold_tolerance).clamp(0.0, 1.0);

        let mut best: Option<PersonMatch> = None;
        let mut runner_up: Option<f32> = None;

        for person in &self.persons {
            let Some(person_match) = self.score_person(person, query, accept_threshold) else {
                continue;
            };
            match &best {
                None => best = Some(person_match),
                Some(current) if person_match.score <= current.score => {
                    runner_up = Some(
                        runner_up
                            .map(|v| v.max(person_match.score))
                            .unwrap_or(person_match.score),
                    );
                }
                Some(current) => {
                    runner_up = Some(
                        runner_up
                            .map(|v| v.max(current.score))
                            .unwrap_or(current.score),
                    );
                    best = Some(person_match);
                }
            }
        }

        let mut best = best?;
        if best.score < effective_threshold {
            return None;
        }
        if let Some(second) = runner_up {
            best.runner_up_score = Some(second);
            if best.score - second < margin {
                return None;
            }
        }
        Some(best)
    }

    /// Highest person score and the default accept threshold (debug / logging).
    pub fn best_score_and_threshold(&self, query: &SpeakerEmbedding) -> Option<(f32, f32)> {
        if !self.space.is_matchable() || query.dim() != self.space.dimension {
            return None;
        }
        let mut best: Option<f32> = None;
        for person in &self.persons {
            let Some(person_match) = self.score_person(person, query, self.accept_threshold) else {
                continue;
            };
            best = Some(best.map_or(person_match.score, |current| {
                current.max(person_match.score)
            }));
        }
        best.map(|score| (score, self.accept_threshold))
    }

    fn score_person(
        &self,
        person: &MatcherPerson,
        query: &SpeakerEmbedding,
        accept_threshold: f32,
    ) -> Option<PersonMatch> {
        let mut best: Option<(f32, &PersonPrototype)> = None;
        for prototype in &person.prototypes {
            if !prototype.space.is_comparable_to(&self.space) {
                continue;
            }
            let score = score_prototype(prototype, query, &person.member_embeddings);
            if !score.is_finite() {
                continue;
            }
            match best {
                Some((current, _)) if score <= current => {}
                _ => best = Some((score, prototype)),
            }
        }
        let (score, prototype) = best?;
        Some(PersonMatch {
            person_id: person.person_id.clone(),
            display_name: person.display_name.clone(),
            score,
            threshold: accept_threshold,
            runner_up_score: None,
            matched_prototype_id: prototype.prototype_id.clone(),
            matched_sample_id: prototype.medoid_sample_id.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diarize::calibration::REDIMNET_CALIBRATION_VERSION;
    use crate::diarize::contract::SpeakerEmbedding;
    use crate::diarize::voice_id::domain::PrototypeMember;
    use crate::diarize::voice_id::ids::{PrototypeId, SampleId};
    use crate::diarize::voice_id::space::{MATCHER_POLICY_VERSION, REDIMNET_FRONTEND_VERSION};

    fn space() -> EmbeddingSpace {
        EmbeddingSpace::from_parts(
            2,
            "sha256:test",
            "test",
            "test",
            "v1",
            REDIMNET_FRONTEND_VERSION,
            REDIMNET_CALIBRATION_VERSION,
            MATCHER_POLICY_VERSION,
        )
    }

    fn person(name: &str, embedding: Vec<f32>, extra_embeddings: Vec<Vec<f32>>) -> MatcherPerson {
        let person_id = PersonId::generate();
        let medoid_id = SampleId::generate();
        let space = space();
        let mut members = vec![PrototypeMember {
            sample_id: medoid_id.clone(),
            quality_weight: 1.0,
        }];
        let mut member_embeddings = vec![(
            medoid_id.clone(),
            SpeakerEmbedding::l2_normalized(embedding.clone()),
            1.0,
        )];
        for extra in extra_embeddings {
            let id = SampleId::generate();
            members.push(PrototypeMember {
                sample_id: id.clone(),
                quality_weight: 1.0,
            });
            member_embeddings.push((id, SpeakerEmbedding::l2_normalized(extra), 1.0));
        }
        MatcherPerson {
            person_id: person_id.clone(),
            display_name: name.to_string(),
            status: PersonStatus::Active,
            prototypes: vec![PersonPrototype {
                prototype_id: PrototypeId::generate(),
                person_id,
                space,
                medoid_sample_id: medoid_id,
                medoid_embedding: SpeakerEmbedding::l2_normalized(embedding),
                policy_version: MATCHER_POLICY_VERSION.into(),
                members,
            }],
            member_embeddings,
        }
    }

    #[test]
    fn same_display_name_persons_compete_in_margin() {
        let alice_a = person("Alice", vec![1.0, 0.0], vec![]);
        let alice_b = person("Alice", vec![0.95, 0.05], vec![]);
        let matcher = PersonMatcher::new(space(), vec![alice_a, alice_b], 0.5, 0.15);
        // Query almost equidistant / small margin -> Unknown.
        let query = SpeakerEmbedding::l2_normalized(vec![0.97, 0.03]);
        assert!(matcher.best_match(&query).is_none());
    }

    #[test]
    fn same_person_samples_do_not_self_compete() {
        let only = person(
            "Bob",
            vec![1.0, 0.0],
            vec![vec![0.99, 0.01], vec![0.98, 0.02]],
        );
        let matcher = PersonMatcher::new(space(), vec![only], 0.5, 0.15);
        let query = SpeakerEmbedding::l2_normalized(vec![1.0, 0.0]);
        let result = matcher
            .best_match(&query)
            .expect("single person should match");
        assert!(result.runner_up_score.is_none());
        assert!(result.score >= 0.5);
    }

    #[test]
    fn empty_explicit_scope_disables_matching() {
        let only = person("Bob", vec![1.0, 0.0], vec![]);
        let matcher = PersonMatcher::new(space(), vec![only], 0.5, 0.0)
            .with_scope(&CandidateScope::Explicit(vec![]));
        assert!(matcher.is_empty());
        assert!(
            matcher
                .best_match(&SpeakerEmbedding::l2_normalized(vec![1.0, 0.0]))
                .is_none()
        );
    }

    #[test]
    fn consent_revoked_person_is_excluded() {
        let mut revoked = person("Carol", vec![1.0, 0.0], vec![]);
        revoked.status = PersonStatus::ConsentRevoked;
        let matcher = PersonMatcher::new(space(), vec![revoked], 0.5, 0.0);
        assert!(matcher.is_empty());
    }

    #[test]
    fn space_mismatch_is_fail_closed() {
        let p = person("D", vec![1.0, 0.0], vec![]);
        let other_space = EmbeddingSpace::from_parts(
            2,
            "sha256:other",
            "test",
            "test",
            "v1",
            REDIMNET_FRONTEND_VERSION,
            REDIMNET_CALIBRATION_VERSION,
            MATCHER_POLICY_VERSION,
        );
        let matcher = PersonMatcher::new(other_space, vec![p], 0.5, 0.0);
        // Prototypes filtered out due to space mismatch.
        assert!(matcher.is_empty());
    }

    #[test]
    fn rename_does_not_affect_identity_key() {
        let mut p = person("Old", vec![1.0, 0.0], vec![]);
        let id = p.person_id.clone();
        p.display_name = "New".into();
        let matcher = PersonMatcher::new(space(), vec![p], 0.5, 0.0);
        let m = matcher
            .best_match(&SpeakerEmbedding::l2_normalized(vec![1.0, 0.0]))
            .unwrap();
        assert_eq!(m.person_id, id);
        assert_eq!(m.display_name, "New");
    }
}
