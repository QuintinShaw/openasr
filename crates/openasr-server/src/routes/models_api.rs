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
    Extension(distribution): Extension<DistributionContext>,
) -> Result<Json<DefaultModelResponse>, ApiError> {
    let home = distribution.openasr_home()?;
    Ok(Json(default_model_response(
        &home,
        distribution.catalog_source(),
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
    if runtime.backend == BackendKind::Native && runtime.native_rebind_blocked() {
        return Err(ApiError::Conflict(
            "Cannot change the default model while a native transcription or realtime session is running."
                .to_string(),
        ));
    }
    let previous_runtime = runtime.model_pack_path.current();
    let previous_durable = resolve_default_pack(&home, distribution.catalog_source())?;
    let previous_preference = openasr_core::load_config_document(&home)
        .map(|document| document.preferences.quant_preference)
        .unwrap_or(QuantPreference::Auto);
    let selected_output_plan = openasr_core::ggml_runtime::ResolvedFamilyRuntimeInput::resolve(
        None,
        openasr_core::ggml_runtime::AutoGpuPolicy::AllBackends,
    )
    .output_plan();
    // Production owners are materialized for this same resolved plan. The
    // shipped activation entry attests that identity before durable commit.
    let staged_owner = openasr_core::StagedActivationOwner::new(selected_output_plan);
    openasr_core::activate_runtime_with_output_plan(
        openasr_core::PreviousActivationState {
            durable_selection: previous_durable.clone(),
            active_runtime: previous_runtime.clone(),
        },
        openasr_core::ActivationFacts {
            selected_output_plan,
        },
        staged_owner,
        Some(pack.clone()),
        Some(pack.path.clone()),
        |next| {
            let pack = next.as_ref().ok_or_else(|| {
                ApiError::BadRequest("activation candidate is missing a pack".to_string())
            })?;
            persist_default_pack(&home, pack, preference)
        },
        |next_path| {
            if runtime.backend != BackendKind::Native {
                return Ok(());
            }
            runtime.rebind_native_model_pack(next_path.clone())
        },
        |previous| match previous {
            Some(previous_pack) => {
                persist_default_pack(&home, previous_pack, previous_preference.clone())
            }
            None => clear_default_model_selection(&home),
        },
    )
    .map_err(map_activation_error)?;
    if runtime.backend == BackendKind::Native {
        crate::realtime::spawn_boot_native_warmup(runtime.clone());
    }
    Ok(Json(default_model_response(
        &home,
        distribution.catalog_source(),
    )?))
}

fn map_activation_error(
    failed: openasr_core::FailedActivation<
        Option<openasr_core::InstalledPack>,
        Option<std::path::PathBuf>,
    >,
) -> ApiError {
    match failed.error {
        openasr_core::ActivationError::OutputPlanMismatch { selected, staged } => {
            ApiError::BadRequest(format!(
                "default model activation failed output-plan attestation: selected={selected:?} staged={staged:?}"
            ))
        }
        openasr_core::ActivationError::Commit(reason) => ApiError::BadRequest(reason),
    }
}

pub(crate) async fn delete_model(
    AxumPath(id): AxumPath<String>,
    Extension(distribution): Extension<DistributionContext>,
) -> Result<Json<DeleteModelResponse>, ApiError> {
    let home = distribution.openasr_home()?;
    let default_pull =
        resolve_default_pack(&home, distribution.catalog_source())?.map(|pack| pack.pull);
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
        clear_default_model_selection(&home)?;
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

    Ok(DefaultModelResponse {
        object: "model.default",
        default_model,
        default_model_status: status,
        default_pull: pack.as_ref().map(|pack| pack.pull.clone()),
        pack,
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

pub(crate) fn persist_default_pack(
    home: &Path,
    pack: &InstalledPack,
    quant_preference: QuantPreference,
) -> Result<(), ApiError> {
    Ok(openasr_core::default_selection::persist(
        home,
        pack,
        quant_preference,
    )?)
}

pub(crate) fn clear_default_model_selection(home: &Path) -> Result<(), ApiError> {
    Ok(openasr_core::default_selection::clear(home)?)
}
