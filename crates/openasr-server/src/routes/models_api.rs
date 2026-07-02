//! Model-management endpoints (local list, default get/set, delete, import)
//! and installed-pack/default-pack resolution. Pure code-motion from `lib.rs`.

use crate::*;

pub(crate) async fn local_models(
    Extension(distribution): Extension<DistributionContext>,
) -> Result<Json<LocalModelsResponse>, ApiError> {
    distribution.ensure_restart_resumes_started();
    let home = distribution.openasr_home()?;
    let packs = list_installed_packs(&home).map_err(ApiError::Pull)?;
    let default_pull =
        resolve_default_pack(&home, distribution.catalog_url())?.map(|pack| pack.pull);
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
        distribution.catalog_url(),
    )?))
}

pub(crate) async fn set_default_model(
    Extension(distribution): Extension<DistributionContext>,
    Json(request): Json<SetDefaultRequest>,
) -> Result<Json<DefaultModelResponse>, ApiError> {
    let home = distribution.openasr_home()?;
    let pack = resolve_installed_pack_for_default(&home, distribution.catalog_url(), &request)?;
    let preference = request.quant_preference_for_pack(&pack);
    persist_default_pack(&home, &pack, preference)?;
    Ok(Json(default_model_response(
        &home,
        distribution.catalog_url(),
    )?))
}

pub(crate) async fn delete_model(
    AxumPath(id): AxumPath<String>,
    Extension(distribution): Extension<DistributionContext>,
) -> Result<Json<DeleteModelResponse>, ApiError> {
    let home = distribution.openasr_home()?;
    let default_pull =
        resolve_default_pack(&home, distribution.catalog_url())?.map(|pack| pack.pull);
    let removed = remove_model_pack(&home, &id).map_err(ApiError::Pull)?;
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
    let catalog =
        load_model_catalog(distribution.catalog_url(), &home).map_err(ApiError::Catalog)?;
    let mut progress = |_| {};
    let installed = install_catalog_model_pack_from_path(&catalog, path, &home, &mut progress)
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

pub(crate) fn resolve_default_pack(
    home: &Path,
    catalog_url: Option<&str>,
) -> Result<Option<InstalledPack>, ApiError> {
    let packs = list_installed_packs(home).map_err(ApiError::Pull)?;
    let catalog = catalog_url
        .map(|catalog_url| load_model_catalog(Some(catalog_url), home))
        .transpose()
        .map_err(ApiError::Catalog)?;
    let document = load_config_document(home).map_err(ApiError::Config)?;
    let pointer = read_default_pack_pointer(home).map_err(ApiError::Pull)?;
    if matches!(
        document.preferences.quant_preference,
        QuantPreference::Pinned { .. }
    ) && let Some(pointer) = pointer.as_ref()
    {
        let pointer_preference = QuantPreference::pinned(&pointer.quant);
        // Use the BARE model id (preferring config.default_model when set) so
        // installed_packs_for_model enumerates ALL quants: this lets the Pinned-hit /
        // Pinned-missing-fallback ladder work, and honors `config set default_model`.
        // Passing the quant-tagged pointer.pull would pre-filter to one quant and make
        // the fallback unreachable (default model would break if that quant is removed).
        let reference = document
            .config
            .default_model
            .as_deref()
            .unwrap_or(pointer.model_id.as_str());
        return Ok(select_launch_pack_from_list(
            &packs,
            reference,
            &pointer_preference,
            catalog.as_ref(),
        ));
    }

    let Some(default_model) = document
        .config
        .default_model
        .as_deref()
        .or_else(|| pointer.as_ref().map(|pointer| pointer.model_id.as_str()))
    else {
        return Ok(None);
    };
    Ok(select_launch_pack_from_list(
        &packs,
        default_model,
        &document.preferences.quant_preference,
        catalog.as_ref(),
    ))
}

pub(crate) fn default_model_response(
    home: &Path,
    catalog_url: Option<&str>,
) -> Result<DefaultModelResponse, ApiError> {
    let pack = resolve_default_pack(home, catalog_url)?;
    // The `default_model` field reports the bare model identity; the quant lives in
    // `default_pull`/`pack.pull`. Appending the quant here would duplicate it (with a
    // different spelling) and diverge from the persisted bare `config.default_model`.
    let default_model = pack.as_ref().map(|pack| pack.model_id.clone()).or_else(|| {
        load_config(home)
            .ok()
            .and_then(|config| config.default_model)
    });

    Ok(DefaultModelResponse {
        object: "model.default",
        default_model,
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
    catalog_url: Option<&str>,
    request: &SetDefaultRequest,
) -> Result<InstalledPack, ApiError> {
    let reference = request.reference()?;
    if request.is_auto_request() {
        let packs = list_installed_packs(home).map_err(ApiError::Pull)?;
        let catalog = catalog_url
            .map(|catalog_url| load_model_catalog(Some(catalog_url), home))
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
    find_installed_pack_reference(home, catalog_url, &reference)?
        .ok_or_else(|| ApiError::BadRequest(format!("Installed model pack not found: {reference}")))
}

pub(crate) fn find_installed_pack_reference(
    home: &Path,
    catalog_url: Option<&str>,
    reference: &str,
) -> Result<Option<InstalledPack>, ApiError> {
    let packs = list_installed_packs(home).map_err(ApiError::Pull)?;
    if let Some(pack) =
        resolve_installed_pack_reference(&packs, reference).map_err(ApiError::Pull)?
    {
        return Ok(Some(pack));
    }
    let Some(catalog_url) = catalog_url else {
        return Ok(None);
    };
    let catalog = load_model_catalog(Some(catalog_url), home).map_err(ApiError::Catalog)?;
    resolve_installed_pack_reference_with_catalog(&packs, &catalog, reference)
        .map_err(ApiError::Pull)
}

pub(crate) fn persist_default_pack(
    home: &Path,
    pack: &InstalledPack,
    quant_preference: QuantPreference,
) -> Result<(), ApiError> {
    save_default_model_selection(home, pack.model_id.clone(), quant_preference)
        .map_err(ApiError::Config)?;
    persist_default_pack_pointer(home, pack).map_err(ApiError::Pull)
}

pub(crate) fn clear_default_model_selection(home: &Path) -> Result<(), ApiError> {
    let mut document = load_config_document(home).map_err(ApiError::Config)?;
    document.config.default_model = None;
    document.preferences.quant_preference = QuantPreference::Auto;
    save_config_document(home, &document).map_err(ApiError::Config)?;

    let path = default_pack_pointer_path(home);
    match fs::remove_file(&path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(source) => Err(ApiError::Pull(PullError::Io { path, source })),
    }
}
