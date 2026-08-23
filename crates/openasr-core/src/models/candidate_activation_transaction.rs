//! Output-plan-bound candidate activation.
//!
//! Runtime ownership and durable model activation share this one entry. A
//! candidate cannot be published until staged-owner attestation has exercised
//! the selected [`GgmlDecodeOutputPlan`]. Failed attestation or a failed
//! commit leaves the previous durable selection and active runtime untouched.

use crate::ggml_runtime::GgmlDecodeOutputPlan;

/// Snapshot taken before an activation attempt mutates durable selection or
/// the in-process active runtime.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreviousActivationState<Selection, Runtime> {
    pub durable_selection: Selection,
    pub active_runtime: Runtime,
}

/// Candidate facts resolved once for the activation attempt.
///
/// The selected output plan is part of the immutable identity that staged
/// owner attestation must reproduce. A generic GPU-class lane is not a plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActivationFacts {
    pub selected_output_plan: GgmlDecodeOutputPlan,
}

/// Owner constructed for this attempt, not yet visible as the active runtime.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StagedActivationOwner {
    output_plan: GgmlDecodeOutputPlan,
}

impl StagedActivationOwner {
    pub const fn new(output_plan: GgmlDecodeOutputPlan) -> Self {
        Self { output_plan }
    }

    pub const fn output_plan(&self) -> GgmlDecodeOutputPlan {
        self.output_plan
    }
}

/// Evidence produced only by the shipped attestation contract.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutputPlanAttestationEvidence {
    pub output_plan: GgmlDecodeOutputPlan,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ActivationError {
    OutputPlanMismatch {
        selected: GgmlDecodeOutputPlan,
        staged: GgmlDecodeOutputPlan,
    },
    Commit(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FailedActivation<Selection, Runtime> {
    pub error: ActivationError,
    pub previous: PreviousActivationState<Selection, Runtime>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommittedActivation<Selection, Runtime> {
    pub evidence: OutputPlanAttestationEvidence,
    pub durable_selection: Selection,
    pub active_runtime: Runtime,
}

/// The one activation entry used by production default-model rebind.
///
/// Attestation always runs before persist or in-memory publication. Callers
/// must not mutate durable selection or the active runtime before this
/// function returns `Ok`. Output-plan mismatch returns `previous` without
/// invoking persist or publish. A persist failure likewise leaves the active
/// runtime unpublished. A publish failure restores the previous durable
/// selection.
pub fn activate_runtime_with_output_plan<Selection, Runtime, PersistErr, PublishErr, RestoreErr>(
    previous: PreviousActivationState<Selection, Runtime>,
    facts: ActivationFacts,
    staged_owner: StagedActivationOwner,
    next_durable_selection: Selection,
    next_active_runtime: Runtime,
    persist_durable: impl FnOnce(&Selection) -> Result<(), PersistErr>,
    publish_runtime: impl FnOnce(&Runtime) -> Result<(), PublishErr>,
    restore_durable: impl FnOnce(&Selection) -> Result<(), RestoreErr>,
) -> Result<CommittedActivation<Selection, Runtime>, FailedActivation<Selection, Runtime>>
where
    PersistErr: std::fmt::Display,
    PublishErr: std::fmt::Display,
    RestoreErr: std::fmt::Display,
{
    match attest_staged_output_plan(&facts, &staged_owner) {
        Ok(evidence) => commit_attested_activation(
            evidence,
            &previous.durable_selection,
            next_durable_selection,
            next_active_runtime,
            persist_durable,
            publish_runtime,
            restore_durable,
        )
        .map_err(|error| FailedActivation { error, previous }),
        Err(error) => Err(FailedActivation { error, previous }),
    }
}

fn attest_staged_output_plan(
    facts: &ActivationFacts,
    staged_owner: &StagedActivationOwner,
) -> Result<OutputPlanAttestationEvidence, ActivationError> {
    let selected = facts.selected_output_plan;
    let staged = staged_owner.output_plan();
    if selected != staged {
        return Err(ActivationError::OutputPlanMismatch { selected, staged });
    }
    Ok(OutputPlanAttestationEvidence {
        output_plan: selected,
    })
}

fn commit_attested_activation<Selection, Runtime, PersistErr, PublishErr, RestoreErr>(
    evidence: OutputPlanAttestationEvidence,
    previous_durable_selection: &Selection,
    next_durable_selection: Selection,
    next_active_runtime: Runtime,
    persist_durable: impl FnOnce(&Selection) -> Result<(), PersistErr>,
    publish_runtime: impl FnOnce(&Runtime) -> Result<(), PublishErr>,
    restore_durable: impl FnOnce(&Selection) -> Result<(), RestoreErr>,
) -> Result<CommittedActivation<Selection, Runtime>, ActivationError>
where
    PersistErr: std::fmt::Display,
    PublishErr: std::fmt::Display,
    RestoreErr: std::fmt::Display,
{
    persist_durable(&next_durable_selection)
        .map_err(|error| ActivationError::Commit(error.to_string()))?;
    if let Err(error) = publish_runtime(&next_active_runtime) {
        let restore_error = restore_durable(previous_durable_selection)
            .err()
            .map(|restore| format!("; restore previous durable failed: {restore}"))
            .unwrap_or_default();
        return Err(ActivationError::Commit(format!("{error}{restore_error}")));
    }
    Ok(CommittedActivation {
        evidence,
        durable_selection: next_durable_selection,
        active_runtime: next_active_runtime,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::default_selection::{
        DefaultModelResolution, persist as persist_default_selection, resolve,
    };
    use crate::testing::{TinyGgufFixtureSpec, write_tiny_gguf_runtime_source};
    use crate::{InstalledPack, QuantPreference};
    use sha2::Digest;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    fn write_installed_pack(
        home: &Path,
        model_id: &str,
        quant: &str,
        suffix: &str,
    ) -> InstalledPack {
        let filename = format!("{model_id}-{quant}.oasr");
        let models = home.join("models");
        let staged = models.join("fixture-source").join(&filename);
        std::fs::create_dir_all(staged.parent().expect("staged parent"))
            .expect("create fixture dir");
        let spec = TinyGgufFixtureSpec::whisper_oasr_v1_encoder_graph_one_layer(model_id);
        write_tiny_gguf_runtime_source(&staged, &spec).expect("write tiny gguf runtime source");
        let bytes = std::fs::read(&staged).expect("read fixture pack");
        std::fs::remove_dir_all(models.join("fixture-source")).expect("drop fixture staging dir");

        let sha256 = format!("{:x}", sha2::Sha256::digest(&bytes));
        let path = models.join("objects/sha256").join(&sha256).join("content");
        std::fs::create_dir_all(path.parent().expect("object parent")).expect("create object dir");
        std::fs::write(&path, &bytes).expect("write object");

        let pack = InstalledPack {
            model_id: model_id.to_string(),
            display_name: model_id.to_string(),
            quant: quant.to_string(),
            suffix: suffix.to_string(),
            pull: format!("{model_id}:{suffix}"),
            filename,
            path,
            url: format!("https://example.test/{model_id}-{quant}.oasr"),
            hf_revision: "0123456789abcdef0123456789abcdef01234567".to_string(),
            sha256,
            size_bytes: bytes.len() as u64,
            installed_at_unix_seconds: 1,
            source: None,
        };
        let ref_path = models
            .join("refs")
            .join(model_id)
            .join(format!("{quant}.json"));
        std::fs::create_dir_all(ref_path.parent().expect("ref parent")).expect("create ref dir");
        std::fs::write(
            &ref_path,
            serde_json::to_string_pretty(&pack).expect("serialize installed pack"),
        )
        .expect("write model ref");
        pack
    }

    #[test]
    fn shipped_activation_rejects_mismatched_output_plan_and_keeps_previous_runtime() {
        let temp = tempfile::tempdir().unwrap();
        let home = temp.path();
        let previous_pack = write_installed_pack(home, "whisper-tiny", "q4_0", "q4");
        let next_pack = write_installed_pack(home, "whisper-base", "q4_0", "q4");
        persist_default_selection(home, &previous_pack, QuantPreference::Auto)
            .expect("seed previous durable selection");

        let active_runtime = Arc::new(Mutex::new(previous_pack.path.clone()));
        let persist_calls = Arc::new(AtomicUsize::new(0));
        let publish_calls = Arc::new(AtomicUsize::new(0));
        let previous = PreviousActivationState {
            durable_selection: previous_pack.clone(),
            active_runtime: previous_pack.path.clone(),
        };

        let failed = activate_runtime_with_output_plan(
            previous.clone(),
            ActivationFacts {
                selected_output_plan: GgmlDecodeOutputPlan::FullLogits,
            },
            StagedActivationOwner::new(GgmlDecodeOutputPlan::NativeFirstMaxToken),
            next_pack.clone(),
            next_pack.path.clone(),
            {
                let persist_calls = Arc::clone(&persist_calls);
                let home = home.to_path_buf();
                move |pack: &InstalledPack| {
                    persist_calls.fetch_add(1, Ordering::SeqCst);
                    persist_default_selection(&home, pack, QuantPreference::Auto)
                }
            },
            {
                let publish_calls = Arc::clone(&publish_calls);
                let active_runtime = Arc::clone(&active_runtime);
                move |path: &PathBuf| {
                    publish_calls.fetch_add(1, Ordering::SeqCst);
                    *active_runtime.lock().expect("runtime slot") = path.clone();
                    Ok::<(), String>(())
                }
            },
            |_previous: &InstalledPack| Ok::<(), String>(()),
        )
        .expect_err("mismatched output-plan attestation must fail closed");

        assert_eq!(
            failed.error,
            ActivationError::OutputPlanMismatch {
                selected: GgmlDecodeOutputPlan::FullLogits,
                staged: GgmlDecodeOutputPlan::NativeFirstMaxToken,
            }
        );
        assert_eq!(failed.previous, previous);
        assert_eq!(persist_calls.load(Ordering::SeqCst), 0);
        assert_eq!(publish_calls.load(Ordering::SeqCst), 0);
        assert_eq!(
            resolve(home, None).expect("resolve after failed activation"),
            DefaultModelResolution::Installed(previous_pack.clone())
        );
        assert_eq!(
            active_runtime.lock().expect("runtime slot").as_path(),
            previous_pack.path.as_path()
        );
    }

    #[test]
    fn shipped_activation_commits_only_after_matching_output_plan_attestation() {
        let temp = tempfile::tempdir().unwrap();
        let home = temp.path();
        let previous_pack = write_installed_pack(home, "whisper-tiny", "q4_0", "q4");
        let next_pack = write_installed_pack(home, "whisper-base", "q4_0", "q4");
        persist_default_selection(home, &previous_pack, QuantPreference::Auto)
            .expect("seed previous durable selection");

        let active_runtime = Arc::new(Mutex::new(previous_pack.path.clone()));
        let committed = activate_runtime_with_output_plan(
            PreviousActivationState {
                durable_selection: previous_pack.clone(),
                active_runtime: previous_pack.path.clone(),
            },
            ActivationFacts {
                selected_output_plan: GgmlDecodeOutputPlan::FullLogits,
            },
            StagedActivationOwner::new(GgmlDecodeOutputPlan::FullLogits),
            next_pack.clone(),
            next_pack.path.clone(),
            {
                let home = home.to_path_buf();
                move |pack: &InstalledPack| {
                    persist_default_selection(&home, pack, QuantPreference::Auto)
                }
            },
            {
                let active_runtime = Arc::clone(&active_runtime);
                move |path: &PathBuf| {
                    *active_runtime.lock().expect("runtime slot") = path.clone();
                    Ok::<(), String>(())
                }
            },
            |_previous: &InstalledPack| Ok::<(), String>(()),
        )
        .expect("matching output-plan attestation must commit");

        assert_eq!(
            committed.evidence.output_plan,
            GgmlDecodeOutputPlan::FullLogits
        );
        assert_eq!(
            resolve(home, None).expect("resolve after committed activation"),
            DefaultModelResolution::Installed(next_pack.clone())
        );
        assert_eq!(
            active_runtime.lock().expect("runtime slot").as_path(),
            next_pack.path.as_path()
        );
    }

    #[test]
    fn failed_commit_after_attestation_does_not_publish_runtime() {
        let temp = tempfile::tempdir().unwrap();
        let home = temp.path();
        let previous_pack = write_installed_pack(home, "whisper-tiny", "q4_0", "q4");
        let next_pack = write_installed_pack(home, "whisper-base", "q4_0", "q4");
        persist_default_selection(home, &previous_pack, QuantPreference::Auto)
            .expect("seed previous durable selection");
        let active_runtime = Arc::new(Mutex::new(previous_pack.path.clone()));

        let failed = activate_runtime_with_output_plan(
            PreviousActivationState {
                durable_selection: previous_pack.clone(),
                active_runtime: previous_pack.path.clone(),
            },
            ActivationFacts {
                selected_output_plan: GgmlDecodeOutputPlan::CompleteScores,
            },
            StagedActivationOwner::new(GgmlDecodeOutputPlan::CompleteScores),
            next_pack,
            PathBuf::from("/tmp/next-must-not-publish.oasr"),
            |_pack: &InstalledPack| Err("persist rejected"),
            {
                let active_runtime = Arc::clone(&active_runtime);
                move |path: &PathBuf| {
                    *active_runtime.lock().expect("runtime slot") = path.clone();
                    Ok::<(), String>(())
                }
            },
            |_previous: &InstalledPack| Ok::<(), String>(()),
        )
        .expect_err("persist failure must not publish");

        assert!(matches!(failed.error, ActivationError::Commit(_)));
        assert_eq!(
            resolve(home, None).expect("resolve after persist failure"),
            DefaultModelResolution::Installed(previous_pack.clone())
        );
        assert_eq!(
            active_runtime.lock().expect("runtime slot").as_path(),
            previous_pack.path.as_path()
        );
    }
}
