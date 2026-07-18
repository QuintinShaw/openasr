use std::env;
#[cfg(unix)]
use std::fs;

use serde::{Deserialize, Serialize};

use crate::{ResolvedCatalogPull, http};

const DOWNLOAD_SOURCE_ENV: &str = "OPENASR_DOWNLOAD_SOURCE";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DownloadSource {
    #[serde(rename = "hf")]
    Hf,
    #[serde(rename = "hf-mirror")]
    HfMirror,
    /// The first-party `weights.openasr.org` Cloudflare Worker, which proxies the
    /// `huggingface.co` `OpenASR/*` `/resolve/...` endpoint and transparently
    /// passes through the 302 into Hugging Face's Xet CDN (`us.aws.cdn.hf.co`).
    /// The start URL host is swapped to the worker; the Xet redirect is then
    /// followed VERBATIM, exactly like the direct `Hf` source (see
    /// [`super::pull::mirror_endpoint_for_current_url`]). Anonymous only -- the HF
    /// bearer token is never sent to the worker.
    #[serde(rename = "weights")]
    Weights,
}

impl DownloadSource {
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "hf" | "huggingface" | "hugging-face" => Some(Self::Hf),
            "hf-mirror" | "hf_mirror" | "mirror" => Some(Self::HfMirror),
            "weights" | "openasr" | "openasr-weights" => Some(Self::Weights),
            _ => None,
        }
    }

    pub fn as_env_value(self) -> &'static str {
        match self {
            Self::Hf => "hf",
            Self::HfMirror => "hf-mirror",
            Self::Weights => "weights",
        }
    }

    pub(crate) fn url_for(self, pull: &ResolvedCatalogPull) -> Option<String> {
        match self {
            Self::Hf => Some(pull.url.clone()),
            Self::HfMirror => Some(http::apply_default_hf_mirror(&pull.url)),
            Self::Weights => Some(http::apply_weights_endpoint(&pull.url)),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum DownloadSourcePref {
    /// Region-aware chain, region judged locally by [`locale_prefers_china_sources`]
    /// (CLI/server without a desktop front end -- there is no other region signal).
    #[default]
    Auto,
    Pinned {
        source: DownloadSource,
    },
    /// Region-aware chain with the region explicitly supplied by the caller
    /// instead of judged from locale/timezone. Exists for the desktop app,
    /// which performs its own single-point China detection (language OR
    /// timezone) in the frontend and hands the boolean result down here --
    /// core never re-derives the region for this variant, it just orders the
    /// same three-source chain [`auto_source_chain`] would for that verdict.
    AutoRegion {
        prefer_china: bool,
    },
}

impl DownloadSourcePref {
    pub fn pinned(source: DownloadSource) -> Self {
        Self::Pinned { source }
    }

    pub fn auto_region(prefer_china: bool) -> Self {
        Self::AutoRegion { prefer_china }
    }

    pub fn parse_env_value(value: &str) -> Option<Self> {
        let value = value.trim();
        if value.eq_ignore_ascii_case("auto") {
            return Some(Self::Auto);
        }
        if value.eq_ignore_ascii_case("china") {
            return Some(Self::AutoRegion { prefer_china: true });
        }
        if value.eq_ignore_ascii_case("global") {
            return Some(Self::AutoRegion {
                prefer_china: false,
            });
        }
        DownloadSource::parse(value).map(Self::pinned)
    }
}

pub fn resolve_chain(pref: &DownloadSourcePref) -> Vec<DownloadSource> {
    match pref {
        DownloadSourcePref::Pinned { source } => vec![*source],
        DownloadSourcePref::Auto => auto_source_chain(locale_prefers_china_sources()),
        DownloadSourcePref::AutoRegion { prefer_china } => auto_source_chain(*prefer_china),
    }
}

pub(crate) fn source_chain_from_env() -> Vec<DownloadSource> {
    if let Some(pref) = env::var(DOWNLOAD_SOURCE_ENV)
        .ok()
        .and_then(|value| DownloadSourcePref::parse_env_value(&value))
    {
        return resolve_chain(&pref);
    }
    default_source_chain(http::hf_endpoint_is_set(), locale_prefers_china_sources())
}

fn default_source_chain(hf_endpoint_set: bool, prefer_china: bool) -> Vec<DownloadSource> {
    if hf_endpoint_set {
        // An explicit `HF_ENDPOINT` override points the mirror rewrite at a custom
        // endpoint, so honor that mirror first; the first-party worker and the
        // direct source stay as deeper fallbacks.
        return vec![
            DownloadSource::HfMirror,
            DownloadSource::Weights,
            DownloadSource::Hf,
        ];
    }
    auto_source_chain(prefer_china)
}

fn auto_source_chain(prefer_china: bool) -> Vec<DownloadSource> {
    if prefer_china {
        // huggingface.co is walled from China networks (resolve times out), so lead
        // with the first-party worker we control, keep hf-mirror.com as a deeper
        // fallback (no longer a single point of failure), and try the direct source
        // last.
        vec![
            DownloadSource::Weights,
            DownloadSource::HfMirror,
            DownloadSource::Hf,
        ]
    } else {
        // Overseas the direct source is reachable and carries zero worker load, so
        // it leads; the worker and mirror are fallbacks.
        vec![
            DownloadSource::Hf,
            DownloadSource::Weights,
            DownloadSource::HfMirror,
        ]
    }
}

fn locale_prefers_china_sources() -> bool {
    ["LC_ALL", "LC_MESSAGES", "LANG"]
        .iter()
        .filter_map(|key| env::var(key).ok())
        .any(|value| locale_value_prefers_china_sources(&value))
        || env::var("TZ")
            .ok()
            .is_some_and(|value| timezone_value_prefers_china_sources(&value))
        || system_timezone_prefers_china_sources()
}

fn locale_value_prefers_china_sources(value: &str) -> bool {
    value.to_ascii_lowercase().starts_with("zh")
}

fn timezone_value_prefers_china_sources(value: &str) -> bool {
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

#[cfg(test)]
mod tests {
    use crate::LicenseClass;

    use super::*;

    fn resolved_pull() -> ResolvedCatalogPull {
        ResolvedCatalogPull {
            requested: "moonshine-tiny:q8".to_string(),
            model_id: "moonshine-tiny".to_string(),
            display_name: "Moonshine Tiny".to_string(),
            quant: "q8_0".to_string(),
            suffix: "q8".to_string(),
            pull: "moonshine-tiny:q8".to_string(),
            filename: "moonshine-tiny-q8_0.oasr".to_string(),
            url: "https://huggingface.co/OpenASR/moonshine-tiny/resolve/0123456789abcdef0123456789abcdef01234567/moonshine-tiny-q8_0.oasr".to_string(),
            mirrors: Vec::new(),
            hf_revision: "0123456789abcdef0123456789abcdef01234567".to_string(),
            sha256: "a".repeat(64),
            size_bytes: 1024,
            license: "MIT".to_string(),
            license_url: "https://example.invalid/license".to_string(),
            license_class: LicenseClass::Permissive,
        }
    }

    #[test]
    fn pinned_enabled_source_is_strict_single_source() {
        assert_eq!(
            resolve_chain(&DownloadSourcePref::pinned(DownloadSource::Hf)),
            vec![DownloadSource::Hf]
        );
    }

    #[test]
    fn china_context_leads_with_weights_then_mirror_then_direct() {
        // China: first-party worker primary, hf-mirror.com a deeper fallback (no
        // longer the single point of failure), huggingface.co last.
        assert_eq!(
            default_source_chain(false, true),
            vec![
                DownloadSource::Weights,
                DownloadSource::HfMirror,
                DownloadSource::Hf,
            ]
        );
    }

    #[test]
    fn overseas_context_leads_with_direct_then_weights_then_mirror() {
        // Overseas: direct huggingface.co primary (zero worker load), worker and
        // mirror as fallbacks.
        assert_eq!(
            default_source_chain(false, false),
            vec![
                DownloadSource::Hf,
                DownloadSource::Weights,
                DownloadSource::HfMirror,
            ]
        );
    }

    #[test]
    fn default_source_keeps_explicit_hf_endpoint_first() {
        // A pinned HF_ENDPOINT keeps the mirror first regardless of locale; the
        // worker and direct source stay as deeper fallbacks.
        assert_eq!(
            default_source_chain(true, false),
            vec![
                DownloadSource::HfMirror,
                DownloadSource::Weights,
                DownloadSource::Hf,
            ]
        );
        assert_eq!(
            default_source_chain(true, true),
            default_source_chain(true, false)
        );
    }

    #[test]
    fn auto_chain_is_region_aware_and_token_independent() {
        // The Auto preference orders purely by region: China leads with the worker,
        // overseas leads with the direct source. A present HF token never reorders
        // the chain (it only scopes which requests carry Authorization, in pull.rs)
        // -- overseas already leads with the direct source, and China stays on the
        // anonymous worker because huggingface.co is walled regardless of token.
        assert_eq!(
            resolve_chain(&DownloadSourcePref::Auto),
            auto_source_chain(locale_prefers_china_sources())
        );
        assert_eq!(auto_source_chain(false).first(), Some(&DownloadSource::Hf));
        assert_eq!(
            auto_source_chain(true).first(),
            Some(&DownloadSource::Weights)
        );
    }

    #[test]
    fn auto_region_pref_orders_by_the_supplied_flag_not_locale() {
        // AutoRegion is the desktop-driven variant: the region verdict is
        // supplied explicitly, so it must match `auto_source_chain` for that
        // verdict regardless of what the process locale/timezone would say.
        assert_eq!(
            resolve_chain(&DownloadSourcePref::auto_region(true)),
            auto_source_chain(true)
        );
        assert_eq!(
            resolve_chain(&DownloadSourcePref::auto_region(false)),
            auto_source_chain(false)
        );
    }

    #[test]
    fn parses_china_and_global_env_values_as_auto_region() {
        assert_eq!(
            DownloadSourcePref::parse_env_value("china"),
            Some(DownloadSourcePref::auto_region(true))
        );
        assert_eq!(
            DownloadSourcePref::parse_env_value("CHINA"),
            Some(DownloadSourcePref::auto_region(true))
        );
        assert_eq!(
            DownloadSourcePref::parse_env_value("global"),
            Some(DownloadSourcePref::auto_region(false))
        );
        assert_eq!(
            DownloadSourcePref::parse_env_value(" global "),
            Some(DownloadSourcePref::auto_region(false))
        );
    }

    #[test]
    fn auto_region_pref_serde_round_trips_and_uses_tagged_shape() {
        let china = DownloadSourcePref::auto_region(true);
        let json = serde_json::to_value(&china).unwrap();
        assert_eq!(
            json,
            serde_json::json!({"mode": "auto_region", "prefer_china": true})
        );
        assert_eq!(
            serde_json::from_value::<DownloadSourcePref>(json).unwrap(),
            china
        );

        let global = DownloadSourcePref::auto_region(false);
        let json = serde_json::to_value(&global).unwrap();
        assert_eq!(
            json,
            serde_json::json!({"mode": "auto_region", "prefer_china": false})
        );
        assert_eq!(
            serde_json::from_value::<DownloadSourcePref>(json).unwrap(),
            global
        );
    }

    #[test]
    fn legacy_auto_and_pinned_json_still_deserialize_after_adding_auto_region() {
        // Forward-compat precedent in this codebase (see `HistoryRetentionPolicy`
        // in config.rs) is: old values keep parsing, unrecognized new values are
        // an explicit hard parse error rather than a silent fallback. Adding the
        // `auto_region` variant must not disturb the two pre-existing wire shapes.
        assert_eq!(
            serde_json::from_value::<DownloadSourcePref>(serde_json::json!({"mode": "auto"}))
                .unwrap(),
            DownloadSourcePref::Auto
        );
        assert_eq!(
            serde_json::from_value::<DownloadSourcePref>(
                serde_json::json!({"mode": "pinned", "source": "hf-mirror"})
            )
            .unwrap(),
            DownloadSourcePref::pinned(DownloadSource::HfMirror)
        );
    }

    #[test]
    fn chinese_locale_and_timezone_prefer_china_sources() {
        assert!(locale_value_prefers_china_sources("zh-Hans_US.UTF-8"));
        assert!(timezone_value_prefers_china_sources(
            "/var/db/timezone/zoneinfo/Asia/Shanghai"
        ));
        assert!(!locale_value_prefers_china_sources("C.UTF-8"));
        assert!(!timezone_value_prefers_china_sources("America/Los_Angeles"));
    }

    #[test]
    fn parses_cli_source_values() {
        assert_eq!(DownloadSource::parse("hf"), Some(DownloadSource::Hf));
        assert_eq!(
            DownloadSource::parse("hf-mirror"),
            Some(DownloadSource::HfMirror)
        );
        assert_eq!(
            DownloadSource::parse("weights"),
            Some(DownloadSource::Weights)
        );
        assert_eq!(
            DownloadSource::parse("openasr-weights"),
            Some(DownloadSource::Weights)
        );
        assert_eq!(DownloadSource::Weights.as_env_value(), "weights");
        assert_eq!(DownloadSource::parse("modelscope"), None);
        assert_eq!(DownloadSourcePref::parse_env_value("ms"), None);
        assert_eq!(
            DownloadSourcePref::parse_env_value("auto"),
            Some(DownloadSourcePref::Auto)
        );
    }

    #[test]
    fn source_urls_resolve_canonical_and_hf_mirror() {
        let pull = resolved_pull();

        assert_eq!(DownloadSource::Hf.url_for(&pull), Some(pull.url.clone()));
        assert_eq!(
            DownloadSource::HfMirror.url_for(&pull),
            Some("https://hf-mirror.com/OpenASR/moonshine-tiny/resolve/0123456789abcdef0123456789abcdef01234567/moonshine-tiny-q8_0.oasr".to_string())
        );
        // The worker source swaps only the huggingface.co host onto
        // weights.openasr.org; the signed /resolve/<rev>/<file> path is preserved
        // verbatim so the Xet redirect (and sha256) stay intact.
        assert_eq!(
            DownloadSource::Weights.url_for(&pull),
            Some("https://weights.openasr.org/OpenASR/moonshine-tiny/resolve/0123456789abcdef0123456789abcdef01234567/moonshine-tiny-q8_0.oasr".to_string())
        );
    }
}
