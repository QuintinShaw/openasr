//! Model-management endpoints (local list, default get/set, delete, import)
//! and installed-pack/default-pack resolution. Pure code-motion from `lib.rs`.

use crate::*;

pub(crate) async fn local_models(
    Extension(distribution): Extension<DistributionContext>,
) -> Result<Json<LocalModelsResponse>, ApiError> {
    let home = distribution.openasr_home()?;
    let packs = list_installed_packs(&home).map_err(ApiError::Pull)?;
    let default_pull =
        resolve_default_pack(&home, distribution.catalog_source())?.map(|pack| pack.pull);
    Ok(Json(LocalModelsResponse {
        object: "list",
        data: packs
            .into_iter()
            .map(|pack| {
                let is_default = default_pull.as_deref() == Some(pack.pull.as_str());
                LocalModelResponse { pack, is_default }
            })
            .collect(),
    }))
}

pub(crate) async fn default_model(
    State(runtime): State<ServerRuntime>,
    Extension(distribution): Extension<DistributionContext>,
) -> Result<Json<DefaultModelResponse>, ApiError> {
    let home = distribution.openasr_home()?;
    Ok(Json(default_model_response(
        &home,
        distribution.catalog_source(),
        runtime.model_pack_path.current().as_deref(),
    )?))
}

pub(crate) async fn set_default_model(
    State(runtime): State<ServerRuntime>,
    Extension(distribution): Extension<DistributionContext>,
    Json(request): Json<SetDefaultRequest>,
) -> Result<Json<DefaultModelResponse>, ApiError> {
    let home = distribution.openasr_home()?;
    let pack = resolve_installed_pack_for_default(&home, distribution.catalog_source(), &request)?;
    let preference = request.quant_preference_for_pack(&pack);
    let intent = crate::realtime::realtime_execution_target_preference(&home)
        .map(openasr_core::device::execution_policy::ExecutionIntent::from)
        .unwrap_or(openasr_core::device::execution_policy::ExecutionIntent::Auto);
    activate_default_model_blocking(
        &runtime,
        &home,
        &pack,
        preference,
        intent,
        DefaultModelActivationMode::PersistSelection,
    )?;

    Ok(Json(default_model_response(
        &home,
        distribution.catalog_source(),
        runtime.model_pack_path.current().as_deref(),
    )?))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DefaultModelActivationMode {
    PersistSelection,
    ReactivateDurableSelection,
}

pub(crate) fn activate_default_model_blocking(
    runtime: &ServerRuntime,
    home: &Path,
    pack: &InstalledPack,
    preference: QuantPreference,
    intent: openasr_core::device::execution_policy::ExecutionIntent,
    mode: DefaultModelActivationMode,
) -> Result<openasr_core::DefaultModelActivationIdentity, ApiError> {
    let _activation_barrier = runtime.begin_native_activation()?;
    if runtime.backend == BackendKind::Native && runtime.native_rebind_blocked() {
        return Err(ApiError::Conflict(
            "Cannot change the default model while a native transcription or realtime session is running."
                .to_string(),
        ));
    }
    // Bridge the entire candidate transaction across verification, warmup,
    // durable commit, and live publication. The idle reaper serializes its
    // exclusive teardown claim against this guard, so it cannot evict the
    // candidate after attestation but before publication.
    let _activation_activity =
        (runtime.backend == BackendKind::Native).then(NativeActivityGuard::enter);

    runtime
        .model_pack_path
        .inject_activation_failure(ModelActivationFailpoint::PackVerification)?;
    let verified_pack = openasr_core::PackVerifier
        .verify_candidate(openasr_core::PackCandidate::new(pack.path.clone()))
        .map_err(|error| {
            ApiError::BadRequest(format!(
                "default model pack is not a verified activation pack: {error}"
            ))
        })?;
    runtime
        .model_pack_path
        .inject_activation_failure(ModelActivationFailpoint::CandidateResolution)?;
    let services = runtime.native_execution.execution_services();
    let resolved_activation = openasr_core::resolve_default_model_activation(
        services.as_ref(),
        &verified_pack,
        intent,
        pack.pull.clone(),
        pack.path.clone(),
    )
    .map_err(|reason| {
        ApiError::BadRequest(format!("default model activation plan failed: {reason}"))
    })?;
    let facts = resolved_activation.facts().clone();
    let identity = facts.identity().clone();
    let journal = match mode {
        DefaultModelActivationMode::PersistSelection => {
            openasr_core::DefaultModelActivationJournalFactory::persist_selection(
                home.to_path_buf(),
                pack.clone(),
                preference,
            )
        }
        DefaultModelActivationMode::ReactivateDurableSelection => {
            openasr_core::DefaultModelActivationJournalFactory::reactivate_durable_selection(
                home.to_path_buf(),
                pack.clone(),
                preference,
            )
        }
    };
    let journal = match runtime.model_pack_path.selection_write_fault() {
        Some(fault) if mode == DefaultModelActivationMode::PersistSelection => {
            journal.with_selection_write_fault_for_test(fault)
        }
        _ => journal,
    };
    let prepared = journal.prepare(
        openasr_core::DefaultModelActivationCandidate {
            pull: pack.pull.clone(),
            path: pack.path.clone(),
            pack_content_id: verified_pack.content_id().to_string(),
        },
        facts,
    );
    debug_assert_eq!(prepared.stage(), openasr_core::ActivationStage::Prepared);
    runtime
        .model_pack_path
        .inject_activation_failure(ModelActivationFailpoint::QuoteObservation)?;
    let quote = resolved_activation.quote().map_err(|reason| {
        ApiError::BadRequest(format!("default model activation quote failed: {reason}"))
    })?;
    runtime
        .model_pack_path
        .inject_activation_failure(ModelActivationFailpoint::BrokerReservation)?;
    let reservation = quote.reserve(services.as_ref()).map_err(|reason| {
        ApiError::BadRequest(format!("default model activation reserve failed: {reason}"))
    })?;
    let reservation_context = reservation.context();
    let reserved = prepared.reserve(reservation);
    debug_assert_eq!(reserved.stage(), openasr_core::ActivationStage::Reserved);

    if let Err(error) = runtime
        .model_pack_path
        .inject_activation_failure(ModelActivationFailpoint::NativeMaterialization)
    {
        let _ = reserved.rollback_activation();
        return Err(error);
    }

    let staged = NativeActivationStagedOwner::stage(
        runtime,
        identity.path().to_path_buf(),
        identity.pack_content_id().to_string(),
    );
    let materialized = reserved.materialize(std::iter::once(staged));
    debug_assert_eq!(
        materialized.stage(),
        openasr_core::ActivationStage::Materialized
    );

    let pending = materialized.begin_attestation(NativeActivationAttestation {
        identity: identity.clone(),
        runtime: runtime.clone(),
        home: home.to_path_buf(),
        reservation_context,
    });
    debug_assert_eq!(
        pending.stage(),
        openasr_core::ActivationStage::AttestationPending
    );

    let attested = match pending.attest() {
        openasr_core::AttestationOutcome::Attested(attested) => attested,
        openasr_core::AttestationOutcome::Rejected {
            transaction,
            source,
        } => {
            let _ = transaction.rollback_activation();
            return Err(map_attestation_error(source));
        }
        openasr_core::AttestationOutcome::MustQuarantine {
            transaction,
            source,
        } => {
            let _ = transaction.quarantine_activation();
            return Err(map_attestation_error(source));
        }
    };
    debug_assert_eq!(attested.stage(), openasr_core::ActivationStage::Attested);

    attested.commit_activation().map_err(|source| {
        ApiError::BadRequest(format!("default model was not committed: {source}"))
    })?;

    runtime
        .model_pack_path
        .inject_activation_failure(ModelActivationFailpoint::DurableCommitBeforeLivePublish)?;

    // Durable V2 is the commit frontier. Live publication is the non-fallible
    // follow-up: it must not run before persist, and a failed persist must
    // leave the previous live path in place.
    let published_identity = identity.clone();
    runtime.publish_attested_native_model(identity);
    Ok(published_identity)
}

struct NativeActivationStagedOwner {
    runtime: ServerRuntime,
    candidate_path: PathBuf,
    candidate_content_id: String,
    previous_path: Option<PathBuf>,
}

impl NativeActivationStagedOwner {
    fn stage(
        runtime: &ServerRuntime,
        candidate_path: PathBuf,
        candidate_content_id: String,
    ) -> Self {
        Self {
            runtime: runtime.clone(),
            candidate_path,
            candidate_content_id,
            previous_path: runtime.model_pack_path.current(),
        }
    }

    fn discard_candidate(&mut self) -> Result<(), String> {
        if self.runtime.backend != BackendKind::Native {
            return Ok(());
        }
        crate::idle_activity::forget_native_model_residency(
            &crate::idle_activity::NativeRuntimeResidencyKey::attested(
                self.candidate_content_id.clone(),
            ),
        );
        if self.previous_path.as_deref() == Some(self.candidate_path.as_path()) {
            return Ok(());
        }
        self.runtime
            .native_execution
            .execution_services()
            .evict_prepared_runtime_content_id(&self.candidate_content_id);
        Ok(())
    }
}

impl openasr_core::StagedOwner for NativeActivationStagedOwner {
    type Error = String;

    fn teardown(&mut self) -> Result<(), Self::Error> {
        self.discard_candidate()
    }

    fn quarantine(&mut self) -> Result<(), Self::Error> {
        self.discard_candidate()
    }
}

struct NativeActivationAttestation {
    identity: openasr_core::DefaultModelActivationIdentity,
    runtime: ServerRuntime,
    home: PathBuf,
    reservation_context: openasr_core::ActivationReservationContext,
}

fn validate_native_activation_probe(
    snapshot: openasr_core::NativeExecutionReceiptSnapshot,
    plan: &openasr_core::DefaultModelActivationPlan,
    lane: &openasr_core::DefaultModelActivationLane,
) -> Result<(), String> {
    let candidate = lane.candidate();
    snapshot
        .attest_activation(
            plan.pack_content_id(),
            plan.resolved_runtime(),
            candidate.device.route.provider,
            &candidate.device.route.stable_id,
            candidate.placement,
        )
        .map_err(|error| error.to_string())
}

impl
    openasr_core::TypedAttestation<
        openasr_core::DefaultModelActivationPlan,
        openasr_core::DefaultModelActivationLane,
    > for NativeActivationAttestation
{
    type Identity = openasr_core::DefaultModelActivationIdentity;
    type Evidence = openasr_core::DefaultModelActivationEvidence;
    type Error = String;

    fn identity(&self) -> &Self::Identity {
        &self.identity
    }

    fn attest(
        &self,
        facts: &openasr_core::ResolvedExecutionFacts<
            openasr_core::DefaultModelActivationPlan,
            openasr_core::DefaultModelActivationLane,
            Self::Identity,
        >,
    ) -> Result<Self::Evidence, openasr_core::AttestationFailure<Self::Error>> {
        let plan = facts.plan();
        let lane = facts.exact_lane();
        if !plan.matches_identity(&self.identity) || lane.candidate() != self.identity.candidate() {
            return Err(openasr_core::AttestationFailure::Rejected(
                "activation facts drifted before native attestation".to_string(),
            ));
        }
        if self.runtime.backend != BackendKind::Native {
            return Ok(openasr_core::DefaultModelActivationEvidence::new(
                self.identity.clone(),
            ));
        }
        self.runtime
            .model_pack_path
            .inject_activation_failure(ModelActivationFailpoint::FirstComputeAttestation)
            .map_err(|error| openasr_core::AttestationFailure::Rejected(error.to_string()))?;
        // The candidate warmup must run *in* the replacement generation.
        // Advancing after warmup would instantly stale both its worker TLS and
        // its exact resident marker, causing a duplicate first-request warmup
        // and a false `/health.model_resident=false` after a valid commit.
        crate::idle_activity::advance_native_runtime_generation();
        let probe = crate::realtime::probe_native_activation_blocking(
            self.runtime.clone(),
            Some(crate::realtime::NativeWarmupTarget::attested(
                self.identity.path().to_path_buf(),
                self.identity.pack_content_id(),
            )),
            Some(self.home.clone()),
            Some(self.reservation_context),
        )
        .map_err(|error| openasr_core::AttestationFailure::Rejected(error.to_string()))?;
        if let Some(snapshot) = probe {
            validate_native_activation_probe(snapshot, plan, lane)
                .map_err(openasr_core::AttestationFailure::Rejected)?;
        }
        self.runtime
            .model_pack_path
            .inject_activation_failure(ModelActivationFailpoint::Reconciliation)
            .map_err(|error| openasr_core::AttestationFailure::Rejected(error.to_string()))?;
        let services = self.runtime.native_execution.execution_services();
        let reconciliation = services
            .runtime_receipts()
            .reconcile_live_leases_quiescent(services.memory_broker());
        if !matches!(
            reconciliation,
            openasr_core::runtime_receipts::LeaseReceiptShadow::Matched
        ) {
            return Err(openasr_core::AttestationFailure::Rejected(format!(
                "default model activation owner reconciliation failed: {reconciliation:?}"
            )));
        }
        Ok(openasr_core::DefaultModelActivationEvidence::new(
            self.identity.clone(),
        ))
    }
}

fn map_attestation_error(source: openasr_core::AttestationError<String>) -> ApiError {
    match source {
        openasr_core::AttestationError::Contract(
            openasr_core::AttestationFailure::Rejected(reason)
            | openasr_core::AttestationFailure::MustQuarantine(reason),
        ) => ApiError::BadRequest(reason),
        openasr_core::AttestationError::ContractIdentityMismatch => {
            ApiError::BadRequest("activation attestation contract identity mismatch".to_string())
        }
        openasr_core::AttestationError::EvidenceIdentityMismatch => {
            ApiError::BadRequest("activation attestation evidence identity mismatch".to_string())
        }
    }
}

pub(crate) async fn delete_model(
    State(runtime): State<ServerRuntime>,
    AxumPath(id): AxumPath<String>,
    Extension(distribution): Extension<DistributionContext>,
) -> Result<Json<DeleteModelResponse>, ApiError> {
    let home = distribution.openasr_home()?;
    let default_pull =
        resolve_default_pack(&home, distribution.catalog_source())?.map(|pack| pack.pull);
    let _activation_barrier = runtime.begin_native_activation()?;
    if runtime.backend == BackendKind::Native && runtime.native_rebind_blocked() {
        return Err(ApiError::Conflict(
            "Cannot delete a model while a native transcription or realtime session is running."
                .to_string(),
        ));
    }
    let removed = remove_model_pack_with_execution_services(
        &home,
        &id,
        Some(distribution.native_execution_services.as_ref()),
    )
    .map_err(ApiError::Pull)?;
    if removed
        .as_ref()
        .is_some_and(|pack| default_pull.as_deref() == Some(pack.pull.as_str()))
    {
        let clear_outcome = clear_default_model_selection(&home)?;
        match &clear_outcome {
            openasr_core::default_selection::DefaultSelectionCommitOutcome::NotCommitted {
                reason,
            } => {
                return Err(ApiError::BadRequest(format!(
                    "default clear was not committed: {reason}"
                )));
            }
            openasr_core::default_selection::DefaultSelectionCommitOutcome::V2Committed => {}
            openasr_core::default_selection::DefaultSelectionCommitOutcome::V2CommittedProjectionFailed {
                reason,
            } => eprintln!(
                "openasr-server: default V2 clear committed; legacy projection repair is pending: {reason}"
            ),
        }
        runtime.clear_active_native_model();
    }
    Ok(Json(DeleteModelResponse {
        deleted: removed.is_some(),
        pack: removed,
    }))
}

pub(crate) async fn import_local_model(
    Extension(distribution): Extension<DistributionContext>,
    Json(request): Json<ImportLocalModelRequest>,
) -> Result<Json<ImportLocalModelResponse>, ApiError> {
    let home = distribution.openasr_home()?;
    let path = resolve_local_pull_source_path(request.path)?;
    let catalog = load_catalog_for_optional_source(distribution.catalog_source(), &home)
        .map_err(ApiError::Catalog)?;
    let resolved = resolve_catalog_model_pack_from_path(&catalog, &path).map_err(ApiError::Pull)?;
    ensure_explicit_model_license_acceptance(&resolved, request.accept_license == Some(true))?;
    let mut progress = |_| {};
    let installed = install_catalog_model_pack_from_path_with_execution_services(
        &catalog,
        path,
        &home,
        Some(distribution.native_execution_services.as_ref()),
        &mut progress,
    )
    .map_err(ApiError::Pull)?;
    Ok(Json(ImportLocalModelResponse {
        object: "model.local_import",
        installed,
    }))
}

pub(crate) fn matching_installed_pack(
    home: &Path,
    resolved: &ResolvedCatalogPull,
) -> Result<Option<InstalledPack>, PullError> {
    Ok(list_installed_packs(home)?.into_iter().find(|pack| {
        pack.pull == resolved.pull
            && pack.sha256 == resolved.sha256
            && pack.size_bytes == resolved.size_bytes
            && pack.hf_revision == resolved.hf_revision
    }))
}

// Default-model resolution/persistence is NOT reimplemented here: it is owned
// by `openasr_core::default_selection` (fail-closed: config.default_model wins,
// the default.json pointer is a fallback only, and a configured-but-uninstalled
// default never silently substitutes a different installed pack). This module
// stays a thin delegate so the server, the CLI, and (eventually) the desktop
// shell all read the same resolver -- see docs/default-model-resolution.md.

pub(crate) fn resolve_default_pack(
    home: &Path,
    catalog_source: Option<CatalogSource<'_>>,
) -> Result<Option<InstalledPack>, ApiError> {
    let catalog = catalog_source
        .map(|source| load_catalog_for_source(source, home))
        .transpose()
        .map_err(ApiError::Catalog)?;
    Ok(
        openasr_core::default_selection::resolve_with_catalog(home, catalog.as_ref())?
            .into_installed_pack(),
    )
}

pub(crate) fn default_model_response(
    home: &Path,
    catalog_source: Option<CatalogSource<'_>>,
    active_pack_path: Option<&Path>,
) -> Result<DefaultModelResponse, ApiError> {
    let catalog = catalog_source
        .map(|source| load_catalog_for_source(source, home))
        .transpose()
        .map_err(ApiError::Catalog)?;
    let resolution = openasr_core::default_selection::resolve_with_catalog(home, catalog.as_ref())?;
    let status = match &resolution {
        openasr_core::default_selection::DefaultModelResolution::Installed(_) => "installed",
        openasr_core::default_selection::DefaultModelResolution::NotInstalled(_) => "not_installed",
        openasr_core::default_selection::DefaultModelResolution::Unset => "unset",
    };
    // The `default_model` field reports the bare model identity; the quant lives in
    // `default_pull`/`pack.pull`. Appending the quant here would duplicate it (with a
    // different spelling) and diverge from the persisted bare `config.default_model`.
    let default_model = match &resolution {
        openasr_core::default_selection::DefaultModelResolution::Installed(pack) => {
            Some(pack.model_id.clone())
        }
        openasr_core::default_selection::DefaultModelResolution::NotInstalled(reference) => {
            Some(reference.clone())
        }
        openasr_core::default_selection::DefaultModelResolution::Unset => None,
    };
    let pack = resolution.into_installed_pack();
    let activation = if pack
        .as_ref()
        .is_some_and(|pack| active_pack_path == Some(pack.path.as_path()))
    {
        DefaultModelActivationState::Committed
    } else {
        DefaultModelActivationState::Unavailable
    };

    Ok(DefaultModelResponse {
        object: "model.default",
        default_model,
        default_model_status: status,
        default_pull: pack.as_ref().map(|pack| pack.pull.clone()),
        pack,
        activation,
    })
}

pub(crate) fn select_launch_pack_from_list(
    packs: &[InstalledPack],
    reference: &str,
    preference: &QuantPreference,
    catalog: Option<&openasr_core::ModelCatalog>,
) -> Option<InstalledPack> {
    let request = LaunchPackRequest {
        model_ref: reference,
        preference,
        catalog,
        host_profile: host_quant_recommendation_profile(),
    };
    resolve_launch_pack(packs, &request)
        .ok()
        .map(|selection| selection.pack)
}

pub(crate) fn resolve_installed_pack_for_default(
    home: &Path,
    catalog_source: Option<CatalogSource<'_>>,
    request: &SetDefaultRequest,
) -> Result<InstalledPack, ApiError> {
    let reference = request.reference()?;
    if request.is_auto_request() {
        let packs = list_installed_packs(home).map_err(ApiError::Pull)?;
        let catalog = catalog_source
            .map(|source| load_catalog_for_source(source, home))
            .transpose()
            .map_err(ApiError::Catalog)?;
        if let Some(pack) = select_launch_pack_from_list(
            &packs,
            &reference,
            &QuantPreference::Auto,
            catalog.as_ref(),
        ) {
            return Ok(pack);
        }
    }
    find_installed_pack_reference(home, catalog_source, &reference)?
        .ok_or_else(|| ApiError::BadRequest(format!("Installed model pack not found: {reference}")))
}

pub(crate) fn find_installed_pack_reference(
    home: &Path,
    catalog_source: Option<CatalogSource<'_>>,
    reference: &str,
) -> Result<Option<InstalledPack>, ApiError> {
    let packs = list_installed_packs(home).map_err(ApiError::Pull)?;
    if let Some(pack) =
        resolve_installed_pack_reference(&packs, reference).map_err(ApiError::Pull)?
    {
        return Ok(Some(pack));
    }
    let Some(catalog_source) = catalog_source else {
        return Ok(None);
    };
    let catalog = load_catalog_for_source(catalog_source, home).map_err(ApiError::Catalog)?;
    resolve_installed_pack_reference_with_catalog(&packs, &catalog, reference)
        .map_err(ApiError::Pull)
}

#[cfg(test)]
pub(crate) fn persist_default_pack(
    home: &Path,
    pack: &InstalledPack,
    quant_preference: QuantPreference,
) -> Result<openasr_core::default_selection::DefaultSelectionCommitOutcome, ApiError> {
    Ok(openasr_core::default_selection::persist_detailed(
        home,
        pack,
        quant_preference,
    )?)
}

pub(crate) fn clear_default_model_selection(
    home: &Path,
) -> Result<openasr_core::default_selection::DefaultSelectionCommitOutcome, ApiError> {
    Ok(openasr_core::default_selection::clear_detailed(home)?)
}
