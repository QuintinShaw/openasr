//! Identity vs transport.
//!
//! Signed catalog bytes, pack URLs, and `files[].url` stay on their canonical
//! hosts. This module is the only place that may swap the *fetch* host for
//! availability (China catalog replica, ModelScope, `dl.bug.im`). Callers
//! must not construct ModelScope or CNB URLs themselves.

use std::env;
#[cfg(unix)]
use std::fs;

pub const CANONICAL_CATALOG_ENDPOINT: &str = "https://catalog.openasr.org";
pub const CHINA_CATALOG_ENDPOINT: &str = "https://catalog.bug.im";
pub const CANONICAL_DL_ENDPOINT: &str = "https://dl.openasr.org";
pub const CHINA_DL_ENDPOINT: &str = "https://dl.bug.im";
pub const GITHUB_RELEASE_DOWNLOAD: &str =
    "https://github.com/QuintinShaw/openasr/releases/download";

pub const CATALOG_ENDPOINT_ENV: &str = "OPENASR_CATALOG_ENDPOINT";
pub const DL_ENDPOINT_ENV: &str = "OPENASR_DL_ENDPOINT";

/// ModelScope org that hosts OpenASR packs. Hugging Face uses `OpenASR`;
/// ModelScope's path is lowercase `openasr` (live 2026-08-26:
/// `https://www.modelscope.cn/models/openasr/qwen3-asr-0.6b`).
pub const MODELSCOPE_OWNER: &str = "openasr";
pub const MODELSCOPE_ORIGIN: &str = "https://www.modelscope.cn";
/// ModelScope Hub only accepts commits on the default branch. Hugging Face
/// git SHAs are not valid ModelScope refs (`commit rejected by repository
/// policy`). Integrity stays the signed catalog sha256, not this branch name.
pub const MODELSCOPE_DEFAULT_REVISION: &str = "master";

const ALLOWED_CATALOG_ENDPOINTS: &[&str] = &[CANONICAL_CATALOG_ENDPOINT, CHINA_CATALOG_ENDPOINT];
const ALLOWED_DL_ENDPOINTS: &[&str] = &[CANONICAL_DL_ENDPOINT, CHINA_DL_ENDPOINT];

/// Rewrite a signed catalog *identity* URL onto the configured transport
/// endpoint. Verification still uses `url` unchanged. Only the live
/// `catalog.openasr.org` prefix and the retired Hugging Face catalog object
/// paths are rewritten — never an arbitrary `huggingface.co` pack URL.
pub(crate) fn apply_catalog_endpoint(url: &str) -> String {
    rewrite_catalog_identity(url, &resolved_catalog_endpoint())
}

/// Transport URLs for one catalog identity. An explicit allowlisted
/// `OPENASR_CATALOG_ENDPOINT` pins a single host. Otherwise Cloudflare and
/// Aliyun ESA are both listed (preferred first) so the caller can race them.
pub(crate) fn catalog_transport_urls(identity_url: &str) -> Vec<String> {
    let canonical = rewrite_catalog_identity(identity_url, CANONICAL_CATALOG_ENDPOINT);
    let china = rewrite_catalog_identity(identity_url, CHINA_CATALOG_ENDPOINT);
    if canonical == china {
        return vec![identity_url.to_string()];
    }
    if let Some(pinned) = endpoint_from_env(CATALOG_ENDPOINT_ENV) {
        if is_allowed_https_endpoint(&pinned, ALLOWED_CATALOG_ENDPOINTS) {
            return vec![rewrite_catalog_identity(identity_url, &pinned)];
        }
        return vec![canonical];
    }
    if prefer_china_transport_without_catalog_env() {
        vec![china, canonical]
    } else {
        vec![canonical, china]
    }
}

fn rewrite_catalog_identity(url: &str, endpoint: &str) -> String {
    if let Some(rest) = url.strip_prefix(CANONICAL_CATALOG_ENDPOINT) {
        return format!("{endpoint}{rest}");
    }
    const LEGACY_HF_CATALOG: &str = "https://huggingface.co/OpenASR/catalog/resolve/main/";
    if let Some(name) = url.strip_prefix(LEGACY_HF_CATALOG)
        && (name == "catalog.json" || name == "catalog.signature.json")
    {
        return format!("{endpoint}/v1/{name}");
    }
    url.to_string()
}

/// Swap `https://dl.openasr.org/...` onto the China (or override) download
/// origin. Non-dl URLs pass through. Used for kernel/vendor bytes whose
/// signed identity stays on `dl.openasr.org`.
pub(crate) fn apply_dl_endpoint(url: &str) -> String {
    rewrite_origin_prefix(url, CANONICAL_DL_ENDPOINT, &resolved_dl_endpoint(), "")
}

/// Ordered fetch URLs for a signed release artifact. Identity stays the
/// catalog/GitHub URL; this list is transportation only.
///
/// China: `dl.bug.im` → official → GitHub.
/// Overseas: GitHub → official. An explicit `OPENASR_DL_ENDPOINT` pins one host.
pub(crate) fn artifact_fetch_urls(url: &str) -> Vec<String> {
    if url.starts_with("file://") {
        return vec![url.to_string()];
    }
    if let Some(pinned) = endpoint_from_env(DL_ENDPOINT_ENV) {
        if is_allowed_https_endpoint(&pinned, ALLOWED_DL_ENDPOINTS) {
            return vec![rewrite_origin_prefix(
                &official_artifact_url(url).unwrap_or_else(|| url.to_string()),
                CANONICAL_DL_ENDPOINT,
                &pinned,
                "",
            )];
        }
        return vec![apply_dl_endpoint(url)];
    }
    let Some(artifact) = parse_release_artifact(url) else {
        return vec![apply_dl_endpoint(url)];
    };
    let github = format!(
        "{GITHUB_RELEASE_DOWNLOAD}/{}/{}",
        artifact.tag, artifact.filename
    );
    let official = artifact.official;
    let china = rewrite_origin_prefix(&official, CANONICAL_DL_ENDPOINT, CHINA_DL_ENDPOINT, "");
    let mut urls = if prefer_china_transport_without_catalog_env() {
        vec![china, official, github]
    } else {
        vec![github, official]
    };
    urls.dedup();
    urls
}

fn official_artifact_url(url: &str) -> Option<String> {
    parse_release_artifact(url).map(|artifact| artifact.official)
}

struct ReleaseArtifact {
    tag: String,
    filename: String,
    official: String,
}

fn parse_release_artifact(url: &str) -> Option<ReleaseArtifact> {
    if let Some(rest) = url.strip_prefix(GITHUB_RELEASE_DOWNLOAD) {
        let rest = rest.strip_prefix('/')?;
        let (tag, filename) = split_tag_filename(rest)?;
        let official = official_url_for_github_tag(tag, filename)?;
        return Some(ReleaseArtifact {
            tag: tag.to_string(),
            filename: filename.to_string(),
            official,
        });
    }
    let path = url
        .strip_prefix(CANONICAL_DL_ENDPOINT)
        .or_else(|| url.strip_prefix(CHINA_DL_ENDPOINT))?;
    if let Some(rest) = path.strip_prefix("/core/") {
        let (tag, filename) = split_tag_filename(rest)?;
        return Some(ReleaseArtifact {
            tag: tag.to_string(),
            filename: filename.to_string(),
            official: format!("{CANONICAL_DL_ENDPOINT}/core/{tag}/{filename}"),
        });
    }
    if let Some(rest) = path.strip_prefix("/cli/") {
        let (tag, filename) = split_tag_filename(rest)?;
        return Some(ReleaseArtifact {
            tag: tag.to_string(),
            filename: filename.to_string(),
            official: format!("{CANONICAL_DL_ENDPOINT}/cli/{tag}/{filename}"),
        });
    }
    if let Some(rest) = path.strip_prefix("/desktop/releases/") {
        let (version, filename) = split_tag_filename(rest)?;
        let version = version.strip_prefix('v')?;
        if version.is_empty() {
            return None;
        }
        return Some(ReleaseArtifact {
            tag: format!("desktop-v{version}"),
            filename: filename.to_string(),
            official: format!("{CANONICAL_DL_ENDPOINT}/desktop/releases/v{version}/{filename}"),
        });
    }
    None
}

fn official_url_for_github_tag(tag: &str, filename: &str) -> Option<String> {
    if let Some(version) = tag.strip_prefix("desktop-v") {
        if version.is_empty() {
            return None;
        }
        return Some(format!(
            "{CANONICAL_DL_ENDPOINT}/desktop/releases/v{version}/{filename}"
        ));
    }
    if !tag.starts_with('v') {
        return None;
    }
    Some(format!("{CANONICAL_DL_ENDPOINT}/core/{tag}/{filename}"))
}

fn split_tag_filename(rest: &str) -> Option<(&str, &str)> {
    let (tag, filename) = rest.split_once('/')?;
    if tag.is_empty()
        || filename.is_empty()
        || filename.contains('/')
        || filename.contains("..")
        || filename.contains('\\')
    {
        return None;
    }
    Some((tag, filename))
}

/// Map a signed Hugging Face pack URL onto ModelScope's resolve path.
///
/// `https://huggingface.co/OpenASR/<repo>/resolve/<rev>/<file>`
/// → `https://www.modelscope.cn/models/openasr/<repo>/resolve/master/<file>`
///
/// The Hugging Face revision is parsed so we only map real `/resolve/` pack
/// URLs; the ModelScope path always uses [`MODELSCOPE_DEFAULT_REVISION`].
/// Owner is always [`MODELSCOPE_OWNER`] (live ModelScope org), not the HF
/// casing. Returns `None` when `url` is not an HF `/resolve/` pack URL.
pub(crate) fn apply_modelscope_endpoint(url: &str) -> Option<String> {
    modelscope_resolve_url(url)
}

pub(crate) fn hf_endpoint_env() -> Option<String> {
    endpoint_from_env("HF_ENDPOINT")
}

pub(crate) fn hf_endpoint_is_set() -> bool {
    hf_endpoint_env().is_some()
}

/// Locale / TZ / `OPENASR_DOWNLOAD_SOURCE=china|global` / explicit catalog
/// endpoint. Desktop should set the env knobs from its one China toggle so
/// this process and the sidecar daemon agree.
pub fn prefer_china_transport() -> bool {
    if let Some(endpoint) = endpoint_from_env(CATALOG_ENDPOINT_ENV)
        && is_allowed_https_endpoint(&endpoint, ALLOWED_CATALOG_ENDPOINTS)
    {
        return endpoint == CHINA_CATALOG_ENDPOINT;
    }
    match env::var("OPENASR_DOWNLOAD_SOURCE")
        .ok()
        .map(|value| value.trim().to_ascii_lowercase())
        .as_deref()
    {
        Some("china") => return true,
        Some("global") => return false,
        _ => {}
    }
    locale_prefers_china_sources()
}

pub(crate) fn locale_prefers_china_sources() -> bool {
    ["LC_ALL", "LC_MESSAGES", "LANG"]
        .iter()
        .filter_map(|key| env::var(key).ok())
        .any(|value| locale_value_prefers_china_sources(&value))
        || env::var("TZ")
            .ok()
            .is_some_and(|value| timezone_value_prefers_china_sources(&value))
        || system_timezone_prefers_china_sources()
}

pub(crate) fn locale_value_prefers_china_sources(value: &str) -> bool {
    // Match desktop `isChinaUser()`: Simplified Chinese language, not every
    // `zh*` locale. Traditional (TW/HK/Hant) is China-transport only when the
    // timezone already matches (Hong Kong / Macau are in the TZ list).
    let lower = value.to_ascii_lowercase();
    let primary = lower
        .split(['.', '@'])
        .next()
        .unwrap_or("")
        .replace('-', "_");
    let mut parts = primary.split('_');
    let lang = parts.next().unwrap_or("");
    let region = parts.next().unwrap_or("");
    lang == "zh" && (region.is_empty() || region == "cn" || region.starts_with("hans"))
}

pub(crate) fn timezone_value_prefers_china_sources(value: &str) -> bool {
    let normalized = value.to_ascii_lowercase().replace('\\', "/");
    [
        "asia/shanghai",
        "asia/chongqing",
        "asia/harbin",
        "asia/urumqi",
        "asia/hong_kong",
        "asia/macau",
        "prc",
    ]
    .iter()
    .any(|needle| normalized.ends_with(needle) || normalized.contains(&format!("/{needle}")))
}

#[cfg(unix)]
fn system_timezone_prefers_china_sources() -> bool {
    fs::read_link("/etc/localtime")
        .ok()
        .and_then(|path| path.to_str().map(str::to_owned))
        .is_some_and(|value| timezone_value_prefers_china_sources(&value))
}

#[cfg(not(unix))]
fn system_timezone_prefers_china_sources() -> bool {
    false
}

fn resolved_catalog_endpoint() -> String {
    resolve_endpoint(
        CATALOG_ENDPOINT_ENV,
        ALLOWED_CATALOG_ENDPOINTS,
        CANONICAL_CATALOG_ENDPOINT,
        CHINA_CATALOG_ENDPOINT,
    )
}

fn resolved_dl_endpoint() -> String {
    if let Some(value) = endpoint_from_env(DL_ENDPOINT_ENV) {
        if is_allowed_https_endpoint(&value, ALLOWED_DL_ENDPOINTS) {
            return value;
        }
        return CANONICAL_DL_ENDPOINT.to_string();
    }
    if prefer_china_transport() {
        CHINA_DL_ENDPOINT.to_string()
    } else {
        CANONICAL_DL_ENDPOINT.to_string()
    }
}

fn resolve_endpoint(env_var: &str, allowlist: &[&str], canonical: &str, china: &str) -> String {
    if let Some(value) = endpoint_from_env(env_var) {
        if is_allowed_https_endpoint(&value, allowlist) {
            return value;
        }
        // Non-allowlist / non-https: fail closed — do not fetch from an
        // attacker-controlled host. Stay on the canonical origin.
        return canonical.to_string();
    }
    if prefer_china_transport_without_catalog_env() {
        china.to_string()
    } else {
        canonical.to_string()
    }
}

/// Like [`prefer_china_transport`], but ignores `OPENASR_CATALOG_ENDPOINT` so
/// catalog resolution can consult download-source / locale without recursion.
fn prefer_china_transport_without_catalog_env() -> bool {
    match env::var("OPENASR_DOWNLOAD_SOURCE")
        .ok()
        .map(|value| value.trim().to_ascii_lowercase())
        .as_deref()
    {
        Some("china") => return true,
        Some("global") => return false,
        _ => {}
    }
    locale_prefers_china_sources()
}

fn endpoint_from_env(var: &str) -> Option<String> {
    env::var(var)
        .ok()
        .map(|value| value.trim().trim_end_matches('/').to_string())
        .filter(|value| !value.is_empty())
}

fn is_allowed_https_endpoint(endpoint: &str, allowlist: &[&str]) -> bool {
    allowlist.contains(&endpoint)
        && endpoint.starts_with("https://")
        && !endpoint.contains(['?', '#', '\\'])
}

/// Rewrite `canonical_origin` (and optionally a legacy origin) onto `endpoint`.
/// `legacy_origin` may be empty.
fn rewrite_origin_prefix(
    url: &str,
    canonical_origin: &str,
    endpoint: &str,
    legacy_origin: &str,
) -> String {
    if let Some(rest) = url.strip_prefix(canonical_origin) {
        return format!("{endpoint}{rest}");
    }
    if !legacy_origin.is_empty()
        && let Some(rest) = url.strip_prefix(legacy_origin)
    {
        return format!("{endpoint}{rest}");
    }
    url.to_string()
}

fn modelscope_resolve_url(hf_url: &str) -> Option<String> {
    let rest = hf_url.strip_prefix("https://huggingface.co/")?;
    let (owner, after_owner) = rest.split_once('/')?;
    let (repo, after_repo) = after_owner.split_once('/')?;
    let after_resolve = after_repo.strip_prefix("resolve/")?;
    let (rev, filename) = after_resolve.split_once('/')?;
    if owner.is_empty()
        || repo.is_empty()
        || rev.is_empty()
        || filename.is_empty()
        || filename.contains("..")
        || filename.contains('\\')
    {
        return None;
    }
    let _ = owner; // HF casing is ignored; ModelScope org is MODELSCOPE_OWNER.
    Some(format!(
        "{MODELSCOPE_ORIGIN}/models/{MODELSCOPE_OWNER}/{repo}/resolve/{MODELSCOPE_DEFAULT_REVISION}/{filename}"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    const IDENTITY: &str = "https://catalog.openasr.org/v1/catalog.json";
    const SIGNATURE: &str = "https://catalog.openasr.org/v1/catalog.signature.json";
    const HUGGING_FACE_ORIGIN: &str = "https://huggingface.co";
    const HF_PACK: &str = "https://huggingface.co/OpenASR/moonshine-tiny/resolve/0123456789abcdef0123456789abcdef01234567/moonshine-tiny-q8_0.oasr";

    #[test]
    fn catalog_rewrite_swaps_only_the_canonical_origin() {
        assert_eq!(
            rewrite_origin_prefix(
                IDENTITY,
                CANONICAL_CATALOG_ENDPOINT,
                CHINA_CATALOG_ENDPOINT,
                HUGGING_FACE_ORIGIN
            ),
            "https://catalog.bug.im/v1/catalog.json"
        );
        assert_eq!(
            rewrite_origin_prefix(
                SIGNATURE,
                CANONICAL_CATALOG_ENDPOINT,
                CHINA_CATALOG_ENDPOINT,
                HUGGING_FACE_ORIGIN
            ),
            "https://catalog.bug.im/v1/catalog.signature.json"
        );
        assert_eq!(
            rewrite_origin_prefix(
                IDENTITY,
                CANONICAL_CATALOG_ENDPOINT,
                CANONICAL_CATALOG_ENDPOINT,
                HUGGING_FACE_ORIGIN
            ),
            IDENTITY
        );
        assert_eq!(
            rewrite_origin_prefix(
                "file:///tmp/catalog.json",
                CANONICAL_CATALOG_ENDPOINT,
                CHINA_CATALOG_ENDPOINT,
                HUGGING_FACE_ORIGIN
            ),
            "file:///tmp/catalog.json"
        );
        assert_eq!(
            rewrite_origin_prefix(
                "https://example.com/v1/catalog.json",
                CANONICAL_CATALOG_ENDPOINT,
                CHINA_CATALOG_ENDPOINT,
                HUGGING_FACE_ORIGIN
            ),
            "https://example.com/v1/catalog.json"
        );
    }

    #[test]
    fn catalog_endpoint_rewrites_live_and_legacy_identities_only() {
        use std::ffi::OsString;
        crate::test_process_env::with_test_process_env(
            [
                (
                    CATALOG_ENDPOINT_ENV,
                    Some(OsString::from(CHINA_CATALOG_ENDPOINT)),
                ),
                ("OPENASR_DOWNLOAD_SOURCE", Some(OsString::from("global"))),
            ],
            || {
                assert_eq!(
                    apply_catalog_endpoint(IDENTITY),
                    "https://catalog.bug.im/v1/catalog.json"
                );
                assert_eq!(
                    apply_catalog_endpoint(SIGNATURE),
                    "https://catalog.bug.im/v1/catalog.signature.json"
                );
                assert_eq!(apply_catalog_endpoint(HF_PACK), HF_PACK);
            },
        );
        crate::test_process_env::with_test_process_env(
            [
                (
                    CATALOG_ENDPOINT_ENV,
                    Some(OsString::from("https://evil.example")),
                ),
                ("OPENASR_DOWNLOAD_SOURCE", Some(OsString::from("global"))),
            ],
            || {
                assert_eq!(apply_catalog_endpoint(IDENTITY), IDENTITY);
            },
        );
        crate::test_process_env::with_test_process_env(
            [
                (CATALOG_ENDPOINT_ENV, None),
                ("OPENASR_DOWNLOAD_SOURCE", Some(OsString::from("global"))),
            ],
            || {
                let legacy = "https://huggingface.co/OpenASR/catalog/resolve/main/catalog.json";
                assert_eq!(
                    apply_catalog_endpoint(legacy),
                    "https://catalog.openasr.org/v1/catalog.json"
                );
            },
        );
    }

    #[test]
    fn non_allowlist_catalog_endpoint_is_rejected() {
        assert!(!is_allowed_https_endpoint(
            "https://evil.example",
            ALLOWED_CATALOG_ENDPOINTS
        ));
        assert!(!is_allowed_https_endpoint(
            "http://catalog.bug.im",
            ALLOWED_CATALOG_ENDPOINTS
        ));
        assert!(!is_allowed_https_endpoint(
            "https://catalog.bug.im/extra",
            ALLOWED_CATALOG_ENDPOINTS
        ));
        assert!(is_allowed_https_endpoint(
            CHINA_CATALOG_ENDPOINT,
            ALLOWED_CATALOG_ENDPOINTS
        ));
        assert!(is_allowed_https_endpoint(
            CANONICAL_CATALOG_ENDPOINT,
            ALLOWED_CATALOG_ENDPOINTS
        ));
    }

    #[test]
    fn modelscope_rewrite_uses_lowercase_owner_and_master_revision() {
        assert_eq!(
            apply_modelscope_endpoint(HF_PACK).as_deref(),
            Some(
                "https://www.modelscope.cn/models/openasr/moonshine-tiny/resolve/master/moonshine-tiny-q8_0.oasr"
            )
        );
        assert_eq!(apply_modelscope_endpoint(IDENTITY), None);
        assert_eq!(
            apply_modelscope_endpoint("https://huggingface.co/OpenASR/moonshine-tiny"),
            None
        );
        assert_eq!(
            apply_modelscope_endpoint("https://huggingface.co/OpenASR/evil/resolve/abc/../secrets"),
            None
        );
    }

    #[test]
    fn dl_rewrite_swaps_only_dl_openasr_origin() {
        let core = "https://dl.openasr.org/core/v0.1.36/plugin.dll";
        assert_eq!(
            rewrite_origin_prefix(core, CANONICAL_DL_ENDPOINT, CHINA_DL_ENDPOINT, ""),
            "https://dl.bug.im/core/v0.1.36/plugin.dll"
        );
        assert_eq!(
            rewrite_origin_prefix(HF_PACK, CANONICAL_DL_ENDPOINT, CHINA_DL_ENDPOINT, ""),
            HF_PACK
        );
    }

    #[test]
    fn chinese_locale_and_timezone_prefer_china_sources() {
        assert!(locale_value_prefers_china_sources("zh-Hans_US.UTF-8"));
        assert!(locale_value_prefers_china_sources("zh_CN.UTF-8"));
        assert!(locale_value_prefers_china_sources("zh.UTF-8"));
        assert!(!locale_value_prefers_china_sources("zh_TW.UTF-8"));
        assert!(!locale_value_prefers_china_sources("zh-Hant-TW"));
        assert!(!locale_value_prefers_china_sources("C.UTF-8"));
        assert!(timezone_value_prefers_china_sources(
            "/var/db/timezone/zoneinfo/Asia/Shanghai"
        ));
        assert!(!timezone_value_prefers_china_sources("America/Los_Angeles"));
    }

    #[test]
    fn catalog_transport_races_both_hosts_when_unpinned() {
        use std::ffi::OsString;
        crate::test_process_env::with_test_process_env(
            [
                (CATALOG_ENDPOINT_ENV, None),
                ("OPENASR_DOWNLOAD_SOURCE", Some(OsString::from("global"))),
            ],
            || {
                assert_eq!(
                    catalog_transport_urls(IDENTITY),
                    vec![
                        IDENTITY.to_string(),
                        "https://catalog.bug.im/v1/catalog.json".to_string(),
                    ]
                );
            },
        );
        crate::test_process_env::with_test_process_env(
            [
                (CATALOG_ENDPOINT_ENV, None),
                ("OPENASR_DOWNLOAD_SOURCE", Some(OsString::from("china"))),
            ],
            || {
                assert_eq!(
                    catalog_transport_urls(IDENTITY),
                    vec![
                        "https://catalog.bug.im/v1/catalog.json".to_string(),
                        IDENTITY.to_string(),
                    ]
                );
            },
        );
        crate::test_process_env::with_test_process_env(
            [
                (
                    CATALOG_ENDPOINT_ENV,
                    Some(OsString::from(CHINA_CATALOG_ENDPOINT)),
                ),
                ("OPENASR_DOWNLOAD_SOURCE", Some(OsString::from("global"))),
            ],
            || {
                assert_eq!(
                    catalog_transport_urls(IDENTITY),
                    vec!["https://catalog.bug.im/v1/catalog.json".to_string()]
                );
            },
        );
        assert_eq!(catalog_transport_urls(HF_PACK), vec![HF_PACK.to_string()]);
    }

    #[test]
    fn artifact_fetch_urls_order_by_region() {
        use std::ffi::OsString;
        let core = "https://dl.openasr.org/core/v0.1.36/plugin.dll";
        let github = format!("{GITHUB_RELEASE_DOWNLOAD}/v0.1.36/plugin.dll");
        let china = "https://dl.bug.im/core/v0.1.36/plugin.dll";
        crate::test_process_env::with_test_process_env(
            [
                (DL_ENDPOINT_ENV, None),
                ("OPENASR_DOWNLOAD_SOURCE", Some(OsString::from("global"))),
            ],
            || {
                assert_eq!(
                    artifact_fetch_urls(core),
                    vec![github.clone(), core.to_string()]
                );
            },
        );
        crate::test_process_env::with_test_process_env(
            [
                (DL_ENDPOINT_ENV, None),
                ("OPENASR_DOWNLOAD_SOURCE", Some(OsString::from("china"))),
            ],
            || {
                assert_eq!(
                    artifact_fetch_urls(core),
                    vec![china.to_string(), core.to_string(), github]
                );
            },
        );
        let desktop = "https://dl.openasr.org/desktop/releases/v0.1.22/OpenASR-Desktop-0.1.22-aarch64.app.tar.gz";
        crate::test_process_env::with_test_process_env(
            [
                (DL_ENDPOINT_ENV, None),
                ("OPENASR_DOWNLOAD_SOURCE", Some(OsString::from("global"))),
            ],
            || {
                assert_eq!(
                    artifact_fetch_urls(desktop)[0],
                    format!(
                        "{GITHUB_RELEASE_DOWNLOAD}/desktop-v0.1.22/OpenASR-Desktop-0.1.22-aarch64.app.tar.gz"
                    )
                );
            },
        );
    }
}
