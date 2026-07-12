use std::{
    env,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};
use openasr_core::{
    ModelCatalog, OPENASR_CATALOG_FILE_ENV_VAR, default_catalog_url,
    load_local_catalog_file_with_identity, load_model_catalog, resolve_local_catalog_env_override,
};

const OPENASR_CATALOG_URL: &str = "OPENASR_CATALOG_URL";

pub(super) fn load_cli_model_catalog(openasr_home: &Path) -> Result<Option<ModelCatalog>> {
    // `OPENASR_CATALOG_FILE`/`OPENASR_CATALOG_IDENTITY` (bytes from a local
    // file, verification identity declared explicitly) take precedence over
    // `OPENASR_CATALOG_URL`: this is what lets `openasr serve` load a
    // desktop-bundled, production-signed catalog file under its real
    // `https://` identity instead of the incidental `file://<install path>` a
    // bare `OPENASR_CATALOG_URL` override would assert (and fail signature
    // verification against). See `resolve_local_catalog_env_override`'s doc
    // comment. A half-configured pair (one var set, not the other) warns to
    // stderr and falls through to `OPENASR_CATALOG_URL` handling below, same
    // as if neither were set.
    let (local_override, warning) = resolve_local_catalog_env_override();
    if let Some(warning) = warning {
        eprintln!(
            "warning: {warning} Falling back to {OPENASR_CATALOG_URL} / the default catalog."
        );
    }
    if let Some(local_override) = local_override {
        return load_local_catalog_file_with_identity(
            &local_override.path,
            &local_override.identity,
            openasr_home,
        )
        .map(Some)
        .with_context(|| {
            format!("Could not load model catalog from {OPENASR_CATALOG_FILE_ENV_VAR}")
        });
    }

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
            // incidental local path. Because that identity is the production
            // `https://` `DEFAULT_CATALOG_URL`, ONLY the production signature
            // verifies here (the committed `catalog.signature.json` as-is);
            // the public local-dev key is not accepted for this call, so a
            // malicious CWD cannot substitute a dev-signed catalog for the
            // canonical one just by being `cd`-ed into. To preview staged,
            // uncommitted catalog edits with the dev key instead, use an
            // explicit `OPENASR_CATALOG_URL=file://<path>` override (which
            // goes through `load_model_catalog`, verified against that
            // literal `file://` identity, not this one). See
            // `load_local_catalog_file_with_identity`.
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

#[cfg(test)]
mod tests {
    use super::*;
    use openasr_core::OPENASR_CATALOG_IDENTITY_ENV_VAR;

    /// The env vars this module reads are process-global, and `cargo test`
    /// runs unit tests within one binary on multiple threads by default, so
    /// tests mutating them must serialize against each other (see the
    /// matching lock in `openasr-server`'s `tests.rs`). `cargo nextest`, this
    /// repo's canonical runner, isolates each test in its own process and
    /// would not need this, but a plain `cargo test` run could still race.
    fn catalog_env_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
        LOCK.get_or_init(|| std::sync::Mutex::new(()))
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    struct CatalogEnvGuard {
        url: Option<String>,
        file: Option<String>,
        identity: Option<String>,
    }

    impl CatalogEnvGuard {
        fn capture() -> Self {
            Self {
                url: env::var(OPENASR_CATALOG_URL).ok(),
                file: env::var(openasr_core::OPENASR_CATALOG_FILE_ENV_VAR).ok(),
                identity: env::var(OPENASR_CATALOG_IDENTITY_ENV_VAR).ok(),
            }
        }

        /// The OLD desktop wiring this PR replaces: a bare
        /// `OPENASR_CATALOG_URL=file://<path>` override.
        fn set_url_override(url: &str) -> Self {
            let guard = Self::capture();
            unsafe {
                env::set_var(OPENASR_CATALOG_URL, url);
                env::remove_var(openasr_core::OPENASR_CATALOG_FILE_ENV_VAR);
                env::remove_var(OPENASR_CATALOG_IDENTITY_ENV_VAR);
            }
            guard
        }

        /// The NEW mechanism: bytes from `path`, verified against the
        /// separately-declared `identity`.
        fn set_local_file_override(path: &Path, identity: &str) -> Self {
            let guard = Self::capture();
            unsafe {
                env::set_var(openasr_core::OPENASR_CATALOG_FILE_ENV_VAR, path);
                env::set_var(OPENASR_CATALOG_IDENTITY_ENV_VAR, identity);
                env::remove_var(OPENASR_CATALOG_URL);
            }
            guard
        }
    }

    impl Drop for CatalogEnvGuard {
        fn drop(&mut self) {
            unsafe {
                match self.url.take() {
                    Some(value) => env::set_var(OPENASR_CATALOG_URL, value),
                    None => env::remove_var(OPENASR_CATALOG_URL),
                }
                match self.file.take() {
                    Some(value) => env::set_var(openasr_core::OPENASR_CATALOG_FILE_ENV_VAR, value),
                    None => env::remove_var(openasr_core::OPENASR_CATALOG_FILE_ENV_VAR),
                }
                match self.identity.take() {
                    Some(value) => env::set_var(OPENASR_CATALOG_IDENTITY_ENV_VAR, value),
                    None => env::remove_var(OPENASR_CATALOG_IDENTITY_ENV_VAR),
                }
            }
        }
    }

    /// Copies the real, committed, PRODUCTION-signed
    /// `model-registry/catalog.json` and its `catalog.signature.json` pair --
    /// byte-for-byte what desktop bundles into `Contents/Resources` -- into
    /// `dir`, returning the copied catalog path.
    fn copy_bundled_production_catalog_to(dir: &Path) -> PathBuf {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../model-registry");
        let catalog_path = dir.join("catalog.json");
        std::fs::copy(root.join("catalog.json"), &catalog_path).expect("copy bundled catalog.json");
        std::fs::copy(
            root.join(openasr_core::CATALOG_SIGNATURE_FILE_NAME),
            catalog_path.with_file_name(openasr_core::CATALOG_SIGNATURE_FILE_NAME),
        )
        .expect("copy bundled catalog.signature.json");
        catalog_path
    }

    #[test]
    fn serve_startup_rejects_bundled_catalog_via_bare_file_url_override() {
        let _lock = catalog_env_lock();
        // Reproduces the 0.1.13 desktop packaging regression at the ACTUAL
        // path `openasr serve` hits at startup (`resolve_serve_model_source`
        // -> `resolve_model_source_for_backend` -> this function): the old
        // `sidecar.rs::resolve_catalog_url` set
        // `OPENASR_CATALOG_URL=file:///Applications/.../catalog.json`, using
        // the install path as both fetch source and verification identity
        // for the exact production-signed catalog desktop bundles. This must
        // fail -- manually reproduced end-to-end via a real `openasr serve`
        // process crashing with "Could not load model catalog from
        // OPENASR_CATALOG_URL" before this PR's fix.
        let temp = tempfile::tempdir().unwrap();
        let home = temp.path().join("home");
        let catalog_path = copy_bundled_production_catalog_to(temp.path());
        let file_url = format!("file://{}", catalog_path.display());

        let _guard = CatalogEnvGuard::set_url_override(&file_url);
        let error = load_cli_model_catalog(&home).unwrap_err().to_string();
        assert!(
            error.contains(&format!(
                "Could not load model catalog from {OPENASR_CATALOG_URL}"
            )),
            "{error}"
        );
    }

    #[test]
    fn serve_startup_accepts_bundled_catalog_via_file_and_identity_override() {
        let _lock = catalog_env_lock();
        // The fix: the SAME bundled bytes, but `OPENASR_CATALOG_FILE` (bytes)
        // + `OPENASR_CATALOG_IDENTITY` (the real production identity the
        // signature is bound to) instead of folding both into a single
        // `file://` URL. `load_cli_model_catalog` must prefer this over
        // `OPENASR_CATALOG_URL`, and the load must succeed -- this is what
        // `openasr serve` (and `search`/`show`) will see once the desktop
        // sidecar is updated to set these two vars for the bundled catalog.
        let temp = tempfile::tempdir().unwrap();
        let home = temp.path().join("home");
        let catalog_path = copy_bundled_production_catalog_to(temp.path());
        let identity = default_catalog_url();

        let _guard = CatalogEnvGuard::set_local_file_override(&catalog_path, identity);
        let catalog = load_cli_model_catalog(&home)
            .expect("bundled catalog + declared identity must verify")
            .expect("a matching override must produce Some(catalog)");
        assert!(!catalog.models.is_empty());
    }

    #[test]
    fn catalog_local_override_takes_precedence_over_catalog_url() {
        let _lock = catalog_env_lock();
        // If both `OPENASR_CATALOG_URL` and `OPENASR_CATALOG_FILE`/`_IDENTITY`
        // are set, the explicit local-file override wins -- silently
        // preferring the legacy `catalog_url` here would resurrect the exact
        // regression this PR fixes for any caller that (redundantly) sets
        // both.
        let temp = tempfile::tempdir().unwrap();
        let home = temp.path().join("home");
        let catalog_path = copy_bundled_production_catalog_to(temp.path());
        let identity = default_catalog_url();
        let bogus_file_url = format!("file://{}", catalog_path.display());

        let _url_guard = CatalogEnvGuard::set_url_override(&bogus_file_url);
        unsafe {
            env::set_var(openasr_core::OPENASR_CATALOG_FILE_ENV_VAR, &catalog_path);
            env::set_var(OPENASR_CATALOG_IDENTITY_ENV_VAR, identity);
        }
        let catalog = load_cli_model_catalog(&home)
            .expect("local override must win and verify")
            .expect("a matching override must produce Some(catalog)");
        assert!(!catalog.models.is_empty());
    }
}
