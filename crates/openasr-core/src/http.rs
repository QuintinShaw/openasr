use std::time::Duration;

pub(crate) fn blocking_client(
    connect_timeout: Duration,
    timeout: Duration,
) -> Result<reqwest::blocking::Client, reqwest::Error> {
    reqwest::blocking::Client::builder()
        .connect_timeout(connect_timeout)
        .timeout(timeout)
        .redirect(reqwest::redirect::Policy::limited(10))
        .build()
}

/// A client that does NOT follow redirects, so the caller can inspect and
/// rewrite each `Location` hop (used by the model downloader to route the
/// Hugging Face CDN redirect through the mirror when `HF_ENDPOINT` is set).
pub(crate) fn blocking_client_no_redirect(
    connect_timeout: Duration,
    timeout: Duration,
) -> Result<reqwest::blocking::Client, reqwest::Error> {
    reqwest::blocking::Client::builder()
        .connect_timeout(connect_timeout)
        .timeout(timeout)
        .redirect(reqwest::redirect::Policy::none())
        .build()
}

pub(crate) fn error_message(error: &reqwest::Error) -> String {
    let mut message = error.to_string();
    let mut source = std::error::Error::source(error);
    while let Some(error) = source {
        message.push_str(": ");
        message.push_str(&error.to_string());
        source = error.source();
    }
    message
}

pub(crate) const HUGGING_FACE_HOST: &str = "https://huggingface.co/";
pub(crate) const HF_MIRROR_ENDPOINT: &str = "https://hf-mirror.com";

/// First-party Cloudflare Worker that proxies `huggingface.co`'s `OpenASR/*`
/// `/resolve/...` endpoint and transparently passes through the 302 into the Xet
/// CDN. Used by [`DownloadSource::Weights`](crate::DownloadSource::Weights) as the
/// primary China source (self-controlled, no dependency on third-party mirrors)
/// and as an overseas fallback.
pub(crate) const WEIGHTS_ENDPOINT: &str = "https://weights.openasr.org";

/// Optional transport host for legacy HF-shaped catalog URLs. The default catalog
/// URL is already `https://catalog.openasr.org/v1/catalog.json`; this endpoint is
/// retained for self-host tests and older pinned catalog identities.
pub(crate) const DEFAULT_CATALOG_ENDPOINT: &str = "https://catalog.openasr.org";

pub(crate) fn apply_default_hf_mirror(url: &str) -> String {
    let endpoint = hf_endpoint().unwrap_or_else(|| HF_MIRROR_ENDPOINT.to_string());
    rewrite_to_mirror(url, Some(&endpoint))
}

/// Swap only the `https://huggingface.co/` host of a resolve URL onto the
/// first-party [`WEIGHTS_ENDPOINT`], preserving the signed `/resolve/<rev>/<file>`
/// path verbatim (so the sha256 gate and the downstream Xet redirect are
/// unaffected). Non-huggingface URLs pass through untouched.
pub(crate) fn apply_weights_endpoint(url: &str) -> String {
    rewrite_to_mirror(url, Some(WEIGHTS_ENDPOINT))
}

/// Rewrite only legacy `https://huggingface.co/...` catalog identities onto the
/// configured catalog endpoint. The current default `/v1/catalog.json` source is
/// already on `catalog.openasr.org`, so it passes through unchanged.
pub(crate) fn apply_catalog_endpoint(url: &str) -> String {
    rewrite_to_mirror(url, Some(catalog_endpoint().as_str()))
}

/// Read a transport-endpoint override from `var`, normalised (trimmed, no trailing
/// slash); `None` when unset or empty. Shared by [`hf_endpoint`] and
/// [`catalog_endpoint`] so both knobs parse identically.
fn endpoint_from_env(var: &str) -> Option<String> {
    std::env::var(var)
        .ok()
        .map(|value| value.trim().trim_end_matches('/').to_string())
        .filter(|value| !value.is_empty())
}

fn hf_endpoint() -> Option<String> {
    endpoint_from_env("HF_ENDPOINT")
}

/// The catalog transport host: `OPENASR_CATALOG_ENDPOINT` if set, else the
/// built-in [`DEFAULT_CATALOG_ENDPOINT`]. Unlike [`hf_endpoint`] this never
/// resolves to `None` — the catalog is always fetched from Cloudflare, never HF.
fn catalog_endpoint() -> String {
    endpoint_from_env("OPENASR_CATALOG_ENDPOINT")
        .unwrap_or_else(|| DEFAULT_CATALOG_ENDPOINT.to_string())
}

fn rewrite_to_mirror(url: &str, endpoint: Option<&str>) -> String {
    match (endpoint, url.strip_prefix(HUGGING_FACE_HOST)) {
        (Some(endpoint), Some(rest)) => format!("{endpoint}/{rest}"),
        _ => url.to_string(),
    }
}

/// Hugging Face stores model files behind a separate CDN (the Xet CAS bridge and
/// the legacy LFS hosts). A `/resolve/...` request 302-redirects to one of those
/// hosts, which — like huggingface.co itself — is unreachable from networks that
pub(crate) fn apply_hf_mirror_redirect_with_endpoint(
    location: &str,
    endpoint: Option<&str>,
) -> String {
    rewrite_cdn_redirect_to_mirror(location, endpoint)
}

pub(crate) fn hf_endpoint_is_set() -> bool {
    hf_endpoint().is_some()
}

pub(crate) fn is_allowed_mirror_host(url: &str) -> bool {
    let Ok(parsed) = reqwest::Url::parse(url) else {
        return false;
    };
    if parsed.scheme() != "https" {
        return false;
    }
    let Some(host) = parsed.host_str().map(str::to_ascii_lowercase) else {
        return false;
    };
    host == "huggingface.co"
        || host.ends_with(".huggingface.co")
        || host == "hf-mirror.com"
        || host == "modelscope.cn"
        || host == "www.modelscope.cn"
}

/// The legacy Hugging Face LFS CDN (`cdn-lfs.huggingface.co`, `cdn-lfs-us-1...`,
/// etc.) lives under the `.huggingface.co` domain and IS proxied by hf-mirror.com,
/// so a `/resolve/...` 302 into it can be host-swapped onto the configured mirror
/// endpoint (keeping the signed query string intact).
const HF_LEGACY_CDN_HOST_SUFFIX: &str = ".huggingface.co";

/// Hugging Face's Xet content-addressed storage is served from the short `.hf.co`
/// domain: the CAS bridge (`cas-bridge.xethub.hf.co`) and the AWS CDN in front of
/// it (`us.aws.cdn.hf.co`), among others. Neither hf-mirror.com nor the
/// weights.openasr.org worker re-serve these Xet CAS paths, so a `/resolve/...`
/// 302 into Xet must be followed VERBATIM -- host-swapping it onto an endpoint
/// yields a 404. (The official mirror tools `hf`/`hfd` reach Xet via its chunk
/// protocol, which this plain-HTTP downloader does not speak.) Any host under
/// `.hf.co` is treated as a Xet frontend and excluded from the rewrite; the legacy
/// LFS CDN stays under `.huggingface.co`, which does not match this suffix.
const HF_XET_CAS_HOST_SUFFIX: &str = ".hf.co";

fn rewrite_cdn_redirect_to_mirror(location: &str, endpoint: Option<&str>) -> String {
    let Some(endpoint) = endpoint else {
        return location.to_string();
    };
    let Some((scheme_host, rest)) = split_scheme_host(location) else {
        return location.to_string();
    };
    let host = scheme_host
        .strip_prefix("https://")
        .or_else(|| scheme_host.strip_prefix("http://"))
        .unwrap_or(scheme_host);
    // Only the legacy LFS CDN (`.huggingface.co`) is proxied by the mirror; every
    // Xet frontend (`.hf.co`, e.g. `us.aws.cdn.hf.co`, `cas-bridge.xethub.hf.co`)
    // is followed verbatim so the transparent Xet redirect is never broken.
    let is_legacy_lfs_cdn =
        host.ends_with(HF_LEGACY_CDN_HOST_SUFFIX) && !host.ends_with(HF_XET_CAS_HOST_SUFFIX);
    if is_legacy_lfs_cdn {
        format!("{endpoint}{rest}")
    } else {
        location.to_string()
    }
}

/// Split an absolute URL into its `scheme://host` prefix and the `/path?query`
/// remainder. Returns `None` for relative or malformed URLs.
fn split_scheme_host(url: &str) -> Option<(&str, &str)> {
    let after_scheme = url.find("://")? + 3;
    let rest_start = url[after_scheme..]
        .find('/')
        .map(|index| after_scheme + index)?;
    Some((&url[..rest_start], &url[rest_start..]))
}

#[cfg(test)]
mod tests {
    use super::{
        DEFAULT_CATALOG_ENDPOINT, rewrite_cdn_redirect_to_mirror, rewrite_to_mirror,
        split_scheme_host,
    };

    #[test]
    fn rewrites_only_huggingface_host_and_only_when_endpoint_set() {
        let pinned = "https://huggingface.co/OpenASR/whisper/resolve/main/model.gguf";
        let mirrored = "https://hf-mirror.com/OpenASR/whisper/resolve/main/model.gguf";
        assert_eq!(
            rewrite_to_mirror(pinned, Some("https://hf-mirror.com")),
            mirrored
        );
        // Unset endpoint or a non-huggingface URL is left untouched.
        assert_eq!(rewrite_to_mirror(pinned, None), pinned);
        assert_eq!(
            rewrite_to_mirror("https://example.com/x", Some("https://hf-mirror.com")),
            "https://example.com/x"
        );
    }

    #[test]
    fn catalog_endpoint_rewrites_only_the_pinned_catalog_host() {
        // The catalog is served from Cloudflare: the pinned huggingface.co host is
        // swapped onto the catalog endpoint while the path (the signed identity)
        // is preserved verbatim.
        let pinned = "https://huggingface.co/OpenASR/catalog/resolve/main/catalog.json";
        assert_eq!(
            rewrite_to_mirror(pinned, Some(DEFAULT_CATALOG_ENDPOINT)),
            "https://catalog.openasr.org/OpenASR/catalog/resolve/main/catalog.json"
        );
        // The sibling signature object rides the same host swap.
        let signature =
            "https://huggingface.co/OpenASR/catalog/resolve/main/catalog.signature.json";
        assert_eq!(
            rewrite_to_mirror(signature, Some(DEFAULT_CATALOG_ENDPOINT)),
            "https://catalog.openasr.org/OpenASR/catalog/resolve/main/catalog.signature.json"
        );
        // Custom / local catalog sources are never rewritten onto the endpoint.
        assert_eq!(
            rewrite_to_mirror("file:///tmp/catalog.json", Some(DEFAULT_CATALOG_ENDPOINT)),
            "file:///tmp/catalog.json"
        );
        assert_eq!(
            rewrite_to_mirror(
                "https://example.com/catalog.json",
                Some(DEFAULT_CATALOG_ENDPOINT)
            ),
            "https://example.com/catalog.json"
        );
    }

    #[test]
    fn splits_scheme_host_from_path() {
        assert_eq!(
            split_scheme_host("https://cas-bridge.xethub.hf.co/xet/abc?sig=1"),
            Some(("https://cas-bridge.xethub.hf.co", "/xet/abc?sig=1"))
        );
        // A URL with no path component or a relative URL cannot be split.
        assert_eq!(split_scheme_host("https://host.example.com"), None);
        assert_eq!(split_scheme_host("/relative/path"), None);
    }

    #[test]
    fn rewrites_hf_cdn_redirect_host_only_when_endpoint_set() {
        let endpoint = Some("https://hf-mirror.com");
        // The Xet CAS bridge is followed VERBATIM, never host-swapped onto the
        // mirror: hf-mirror.com does not proxy the Xet CAS path (rewriting it 404s),
        // and the bridge host is itself reachable. Regression guard — a previous
        // version rewrote this and broke every Xet-backed download.
        assert_eq!(
            rewrite_cdn_redirect_to_mirror(
                "https://cas-bridge.xethub.hf.co/xet-bridge-us/abc?X-Amz-Signature=deadbeef",
                endpoint
            ),
            "https://cas-bridge.xethub.hf.co/xet-bridge-us/abc?X-Amz-Signature=deadbeef"
        );
        // The AWS CDN in front of Xet (`us.aws.cdn.hf.co`) is also a `.hf.co` Xet
        // frontend and must be followed verbatim. It ends with `.hf.co` but NOT
        // `.huggingface.co`; a previous version keyed the exclusion on the literal
        // `xethub.hf.co` suffix only and wrongly host-swapped this active host onto
        // the endpoint (a guaranteed 404). Regression guard.
        assert_eq!(
            rewrite_cdn_redirect_to_mirror(
                "https://us.aws.cdn.hf.co/repos/xx/yy/blob?X-Amz-Signature=deadbeef",
                endpoint
            ),
            "https://us.aws.cdn.hf.co/repos/xx/yy/blob?X-Amz-Signature=deadbeef"
        );
        // The legacy LFS CDN, which hf-mirror DOES proxy, is still host-swapped,
        // keeping the signed query string intact.
        assert_eq!(
            rewrite_cdn_redirect_to_mirror(
                "https://cdn-lfs.huggingface.co/repo/file?x=1",
                endpoint
            ),
            "https://hf-mirror.com/repo/file?x=1"
        );
        // huggingface.co itself is rewritten before the request, not here, and
        // unrelated CDNs / an unset endpoint pass through untouched.
        assert_eq!(
            rewrite_cdn_redirect_to_mirror("https://huggingface.co/a/b", endpoint),
            "https://huggingface.co/a/b"
        );
        assert_eq!(
            rewrite_cdn_redirect_to_mirror("https://cdn.example.com/a/b", endpoint),
            "https://cdn.example.com/a/b"
        );
        assert_eq!(
            rewrite_cdn_redirect_to_mirror("https://cas-bridge.xethub.hf.co/x?s=1", None),
            "https://cas-bridge.xethub.hf.co/x?s=1"
        );
    }
}
