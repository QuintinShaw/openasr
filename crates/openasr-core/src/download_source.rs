use std::{env, fs};

use serde::{Deserialize, Serialize};

use crate::{ResolvedCatalogPull, http};

const DOWNLOAD_SOURCE_ENV: &str = "OPENASR_DOWNLOAD_SOURCE";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DownloadSource {
    #[serde(rename = "hf")]
    Hf,
    #[serde(rename = "hf-mirror")]
    HfMirror,
}

impl DownloadSource {
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "hf" | "huggingface" | "hugging-face" => Some(Self::Hf),
            "hf-mirror" | "hf_mirror" | "mirror" => Some(Self::HfMirror),
            _ => None,
        }
    }

    pub fn as_env_value(self) -> &'static str {
        match self {
            Self::Hf => "hf",
            Self::HfMirror => "hf-mirror",
        }
    }

    pub(crate) fn url_for(self, pull: &ResolvedCatalogPull) -> Option<String> {
        match self {
            Self::Hf => Some(pull.url.clone()),
            Self::HfMirror => Some(http::apply_default_hf_mirror(&pull.url)),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum DownloadSourcePref {
    #[default]
    Auto,
    Pinned {
        source: DownloadSource,
    },
}

impl DownloadSourcePref {
    pub fn pinned(source: DownloadSource) -> Self {
        Self::Pinned { source }
    }

    pub fn parse_env_value(value: &str) -> Option<Self> {
        let value = value.trim();
        if value.eq_ignore_ascii_case("auto") {
            return Some(Self::Auto);
        }
        DownloadSource::parse(value).map(Self::pinned)
    }
}

pub fn resolve_chain(pref: &DownloadSourcePref) -> Vec<DownloadSource> {
    match pref {
        DownloadSourcePref::Pinned { source } => vec![*source],
        DownloadSourcePref::Auto => auto_source_chain(locale_prefers_china_sources()),
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
    if hf_endpoint_set || prefer_china {
        vec![DownloadSource::HfMirror, DownloadSource::Hf]
    } else {
        auto_source_chain(prefer_china)
    }
}

fn auto_source_chain(prefer_china: bool) -> Vec<DownloadSource> {
    if prefer_china {
        vec![DownloadSource::HfMirror, DownloadSource::Hf]
    } else {
        vec![DownloadSource::Hf, DownloadSource::HfMirror]
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
    fn default_auto_source_uses_hf_mirror_for_china_context() {
        assert_eq!(
            default_source_chain(false, true),
            vec![DownloadSource::HfMirror, DownloadSource::Hf]
        );
    }

    #[test]
    fn default_source_keeps_explicit_hf_endpoint_first_outside_china_context() {
        assert_eq!(
            default_source_chain(true, false),
            vec![DownloadSource::HfMirror, DownloadSource::Hf]
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
    }
}
