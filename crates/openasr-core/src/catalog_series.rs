use std::{collections::BTreeMap, sync::LazyLock};

use serde::Deserialize;

const CATALOG_SERIES_TOML: &str = include_str!("../catalog-series.toml");

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CatalogSeriesSpec {
    family: String,
    aliases: Vec<String>,
    member_sizes: Vec<String>,
    default_size: String,
}

impl CatalogSeriesSpec {
    #[cfg(test)]
    pub(crate) fn family(&self) -> &str {
        &self.family
    }

    pub(crate) fn default_size(&self) -> &str {
        &self.default_size
    }

    pub(crate) fn matches_alias(&self, value: &str) -> bool {
        self.aliases.iter().any(|alias| alias == value)
    }

    pub(crate) fn contains_family_size(&self, family: &str, size: &str) -> bool {
        self.family == family && self.member_sizes.iter().any(|member| member == size)
    }
}

#[derive(Debug, Deserialize)]
struct CatalogSeriesSpecToml {
    aliases: Vec<String>,
    member_sizes: Vec<String>,
    default_size: String,
}

static CATALOG_SERIES: LazyLock<Vec<CatalogSeriesSpec>> = LazyLock::new(|| {
    let parsed: BTreeMap<String, CatalogSeriesSpecToml> =
        toml::from_str(CATALOG_SERIES_TOML).expect("catalog series taxonomy must parse");
    parsed
        .into_iter()
        .map(|(family, spec)| {
            assert!(
                !family.trim().is_empty(),
                "catalog series family must not be empty"
            );
            assert!(
                spec.aliases.iter().all(|alias| !alias.trim().is_empty()),
                "catalog series '{family}' aliases must not contain empty values"
            );
            assert!(
                spec.member_sizes.iter().all(|size| !size.trim().is_empty()),
                "catalog series '{family}' member_sizes must not contain empty values"
            );
            assert!(
                spec.member_sizes
                    .iter()
                    .any(|size| size == &spec.default_size),
                "catalog series '{family}' default_size must be a member size"
            );
            CatalogSeriesSpec {
                family,
                aliases: spec.aliases,
                member_sizes: spec.member_sizes,
                default_size: spec.default_size,
            }
        })
        .collect()
});

pub(crate) fn catalog_series_spec(model_ref: &str) -> Option<&'static CatalogSeriesSpec> {
    let normalized = model_ref.trim();
    CATALOG_SERIES
        .iter()
        .find(|series| series.matches_alias(normalized))
}

pub(crate) fn family_aliases_match(a: &str, b: &str) -> bool {
    if a.eq_ignore_ascii_case(b) {
        return true;
    }

    let Some(left) = catalog_series_spec(a) else {
        return false;
    };
    let Some(right) = catalog_series_spec(b) else {
        return false;
    };
    std::ptr::eq(left, right)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen_aliases_share_one_series() {
        let aliases = ["qwen", "qwen-asr", "qwen3", "qwen3-asr"];
        let first = catalog_series_spec(aliases[0]).expect("qwen series");
        assert_eq!(first.family(), "qwen");
        assert_eq!(first.default_size(), "0.6b");

        for alias in aliases {
            let spec = catalog_series_spec(alias).expect("alias should resolve");
            assert!(std::ptr::eq(first, spec));
            assert!(family_aliases_match(alias, "qwen3-asr"));
            assert!(family_aliases_match("qwen3-asr", alias));
        }
    }

    #[test]
    fn family_aliases_match_keeps_unknowns_exact_only() {
        assert!(family_aliases_match("whisper", "WHISPER"));
        assert!(!family_aliases_match("qwen", "whisper"));
        assert!(!family_aliases_match("qwen3-asr-0.6b", "qwen"));
    }

    #[test]
    fn xasr_aliases_share_one_series() {
        let aliases = ["xasr", "x-asr", "xasr-zipformer"];
        let first = catalog_series_spec(aliases[0]).expect("xasr series");
        assert_eq!(first.family(), "xasr-zipformer");
        assert_eq!(first.default_size(), "0.16b");

        for alias in aliases {
            let spec = catalog_series_spec(alias).expect("alias should resolve");
            assert!(std::ptr::eq(first, spec));
            assert!(family_aliases_match(alias, "xasr-zipformer"));
            assert!(family_aliases_match("xasr-zipformer", alias));
        }
        assert!(!family_aliases_match("xasr", "qwen"));
    }

    #[test]
    fn cohere_and_moonshine_series_resolve() {
        let cohere = catalog_series_spec("cohere").expect("cohere series");
        assert_eq!(cohere.family(), "cohere");
        assert_eq!(cohere.default_size(), "2b");
        assert!(family_aliases_match("cohere", "cohere-transcribe"));
        assert!(family_aliases_match("cohere-transcribe", "cohere"));

        let moonshine = catalog_series_spec("moonshine").expect("moonshine series");
        assert_eq!(moonshine.family(), "moonshine");
        assert_eq!(moonshine.default_size(), "tiny");
        assert!(!family_aliases_match("cohere", "moonshine"));
    }
}
