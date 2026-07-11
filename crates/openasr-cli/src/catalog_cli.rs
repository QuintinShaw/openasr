use std::{
    env,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};
use openasr_core::{
    ModelCatalog, default_catalog_url, load_local_catalog_file_with_identity, load_model_catalog,
};

const OPENASR_CATALOG_URL: &str = "OPENASR_CATALOG_URL";

pub(super) fn load_cli_model_catalog(openasr_home: &Path) -> Result<Option<ModelCatalog>> {
    if let Some(catalog_url) = env::var(OPENASR_CATALOG_URL)
        .ok()
        .filter(|value| !value.trim().is_empty())
    {
        return load_catalog(Some(catalog_url.as_str()), openasr_home)
            .map(Some)
            .with_context(|| format!("Could not load model catalog from {OPENASR_CATALOG_URL}"));
    }

    for path in local_catalog_candidates()? {
        if path.is_file() {
            // `model-registry/catalog.json` discovered relative to the repo
            // checkout is the pre-deployment source of truth for the
            // canonical catalog identity (mirroring the binary's embedded
            // snapshot), so it is verified against that identity -- not the
            // incidental local path -- accepting either the production
            // signature (the committed `catalog.signature.json` as-is) or a
            // locally (uncommitted) dev-signed one for previewing staged
            // edits. See `load_local_catalog_file_with_identity`.
            return load_local_catalog_file_with_identity(
                &path,
                default_catalog_url(),
                openasr_home,
            )
            .map(Some)
            .with_context(|| format!("Could not load local model catalog '{}'", path.display()));
        }
    }

    Ok(None)
}

fn load_catalog(catalog_url: Option<&str>, openasr_home: &Path) -> Result<ModelCatalog> {
    Ok(load_model_catalog(catalog_url, openasr_home)?)
}

fn local_catalog_candidates() -> Result<Vec<PathBuf>> {
    let mut candidates = Vec::new();
    candidates.push(
        env::current_dir()
            .context("Could not resolve current directory for local catalog discovery")?
            .join("model-registry/catalog.json"),
    );
    candidates
        .push(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../model-registry/catalog.json"));
    candidates.sort();
    candidates.dedup();
    Ok(candidates)
}
