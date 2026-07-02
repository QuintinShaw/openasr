//! `/v1/history` list/get/delete handlers plus the retention-pruning helpers.
//! Pure code-motion from `lib.rs`; shared crate-root items come via
//! `use crate::*`, history-store + retention types from `openasr_core`.

use axum::{
    Extension, Json,
    extract::Path as AxumPath,
    response::{IntoResponse, Response},
};
use openasr_core::config::{HistoryRetentionPolicy, load_config_document};
use openasr_core::realtime::history::DaemonHistoryStore;

use crate::*;

pub(crate) async fn history_list(
    Extension(distribution): Extension<DistributionContext>,
) -> Result<Json<HistoryListResponse>, ApiError> {
    let home = distribution.openasr_home()?;
    let store = DaemonHistoryStore::open(&home);
    prune_history_for_preferences(&distribution, &store)?;
    Ok(Json(HistoryListResponse {
        object: "list",
        data: store.list().map_err(ApiError::History)?,
    }))
}

pub(crate) async fn history_get(
    AxumPath(id): AxumPath<String>,
    Extension(distribution): Extension<DistributionContext>,
) -> Result<Response, ApiError> {
    let home = distribution.openasr_home()?;
    let store = DaemonHistoryStore::open(&home);
    prune_history_for_preferences(&distribution, &store)?;
    let detail = store
        .get(&id)
        .map_err(ApiError::History)?
        .ok_or_else(|| ApiError::NotFound(format!("History entry not found: {id}")))?;
    Ok(Json(detail).into_response())
}

pub(crate) async fn history_delete(
    AxumPath(id): AxumPath<String>,
    Extension(distribution): Extension<DistributionContext>,
) -> Result<Json<DeleteHistoryResponse>, ApiError> {
    let home = distribution.openasr_home()?;
    let store = DaemonHistoryStore::open(&home);
    if !store.delete(&id).map_err(ApiError::History)? {
        return Err(ApiError::NotFound(format!("History entry not found: {id}")));
    }
    Ok(Json(DeleteHistoryResponse { deleted: true, id }))
}

pub(crate) fn prune_history_for_preferences(
    distribution: &DistributionContext,
    store: &DaemonHistoryStore,
) -> Result<usize, ApiError> {
    let home = distribution.openasr_home()?;
    let preferences = load_config_document(&home)
        .map_err(ApiError::Config)?
        .preferences;
    prune_history_store(store, preferences.history_retention)
}

pub(crate) fn prune_history_store(
    store: &DaemonHistoryStore,
    retention: HistoryRetentionPolicy,
) -> Result<usize, ApiError> {
    if let Some(max_entries) = retention.max_entries() {
        return store
            .retain_most_recent(max_entries)
            .map_err(ApiError::History);
    }

    if let Some(max_age_seconds) = retention.max_age_seconds() {
        let cutoff = unix_seconds_now().saturating_sub(max_age_seconds);
        return store.delete_older_than(cutoff).map_err(ApiError::History);
    }

    Ok(0)
}
