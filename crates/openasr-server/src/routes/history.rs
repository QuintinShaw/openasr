//! `/v1/history` list/get/delete handlers plus the retention-pruning helpers.
//! Pure code-motion from `lib.rs`; shared crate-root items come via
//! `use crate::*`, history-store + retention types from `openasr_core`.

use axum::{
    Extension, Json,
    extract::{Path as AxumPath, Query},
    response::{IntoResponse, Response},
};
use openasr_core::config::{HistoryRetentionPolicy, load_config_document};
use openasr_core::realtime::history::{DaemonHistoryKind, DaemonHistoryQuery, DaemonHistoryStore};
use serde::Deserialize;

use crate::*;

/// Default and max page size for `GET /v1/history`. Keeps a runaway
/// `limit` query param from forcing a full-table scan/response on a large
/// history.
const DEFAULT_HISTORY_LIMIT: usize = 50;
const MAX_HISTORY_LIMIT: usize = 500;

#[derive(Debug, Default, Deserialize)]
pub(crate) struct HistoryListQuery {
    #[serde(default)]
    pub(crate) search: Option<String>,
    #[serde(default)]
    pub(crate) kind: Option<String>,
    #[serde(default)]
    pub(crate) limit: Option<usize>,
    #[serde(default)]
    pub(crate) offset: Option<usize>,
}

impl HistoryListQuery {
    fn into_store_query(self) -> Result<(DaemonHistoryQuery, usize, usize), ApiError> {
        let kind = match self.kind.as_deref() {
            None => None,
            Some(raw) => Some(parse_history_kind(raw)?),
        };
        let limit = self.limit.unwrap_or(DEFAULT_HISTORY_LIMIT).min(MAX_HISTORY_LIMIT);
        let offset = self.offset.unwrap_or(0);
        let search = self
            .search
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty());
        Ok((
            DaemonHistoryQuery {
                search,
                kind,
                limit: Some(limit),
                offset,
            },
            limit,
            offset,
        ))
    }
}

fn parse_history_kind(raw: &str) -> Result<DaemonHistoryKind, ApiError> {
    match raw {
        "file" => Ok(DaemonHistoryKind::File),
        "live" => Ok(DaemonHistoryKind::Live),
        other => Err(ApiError::BadRequest(format!(
            "Unsupported history kind '{other}'. Use one of: file, live."
        ))),
    }
}

pub(crate) async fn history_list(
    Extension(distribution): Extension<DistributionContext>,
    Query(query): Query<HistoryListQuery>,
) -> Result<Json<HistoryListResponse>, ApiError> {
    let home = distribution.openasr_home()?;
    let store = DaemonHistoryStore::open(&home);
    prune_history_for_preferences(&distribution, &store)?;
    let (store_query, limit, offset) = query.into_store_query()?;
    let page = store.query(&store_query).map_err(ApiError::History)?;
    Ok(Json(HistoryListResponse {
        object: "list",
        data: page.entries,
        total: page.total,
        limit,
        offset,
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
