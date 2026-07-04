use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    CatalogPullRequest, CatalogQuantRecommendationProfile, ModelCatalog, canonical_quant_tag,
    parse_model_ref, registry::quant_quality_rank, resolve_catalog_pull_with_profile,
};
use crate::{InstalledPack, catalog_series::family_aliases_match};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum QuantPreference {
    #[default]
    Auto,
    Pinned {
        quant: String,
    },
}

impl QuantPreference {
    pub fn pinned(quant: impl AsRef<str>) -> Self {
        Self::Pinned {
            quant: canonical_quant_tag(quant.as_ref()).to_string(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct LaunchPackRequest<'a> {
    pub model_ref: &'a str,
    pub preference: &'a QuantPreference,
    pub catalog: Option<&'a ModelCatalog>,
    pub host_profile: CatalogQuantRecommendationProfile,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LaunchPackSelection {
    pub pack: InstalledPack,
    pub effective_quant: String,
    pub runtime_model_id: String,
    pub reason: LaunchSelectionReason,
    pub notice: Option<LaunchPackNotice>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LaunchSelectionReason {
    Pinned,
    AutoRecommended,
    AutoBestInstalled,
    AutoSingle,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LaunchPackNotice {
    RecommendedNotInstalled { recommended: String, chosen: String },
    AmbiguousAutoFallback { chosen: String },
    PinnedQuantMissingFellBack { requested: String, chosen: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum LaunchPackError {
    #[error("No installed model pack matches '{model_ref}'. Install one first: {pull_hint}")]
    NothingInstalled {
        model_ref: String,
        recommended: Option<String>,
        pull_hint: String,
    },
}

pub fn resolve_launch_pack(
    packs: &[InstalledPack],
    request: &LaunchPackRequest<'_>,
) -> Result<LaunchPackSelection, LaunchPackError> {
    let candidates = installed_packs_for_model(packs, request.model_ref, request.catalog);
    if candidates.is_empty() {
        return Err(nothing_installed_error(request));
    }

    if let QuantPreference::Pinned { quant } = request.preference {
        let requested = canonical_quant_tag(quant);
        if let Some(pack) = find_quant(&candidates, requested) {
            return Ok(selection(pack, LaunchSelectionReason::Pinned, None));
        }
        let fallback = best_installed_pack(&candidates);
        let chosen = canonical_quant_tag(&fallback.quant).to_string();
        return Ok(selection(
            fallback,
            LaunchSelectionReason::AutoBestInstalled,
            Some(LaunchPackNotice::PinnedQuantMissingFellBack {
                requested: requested.to_string(),
                chosen,
            }),
        ));
    }

    if candidates.len() == 1 {
        return Ok(selection(
            candidates[0].clone(),
            LaunchSelectionReason::AutoSingle,
            None,
        ));
    }

    if let Some(recommended) = recommended_quant(request) {
        if let Some(pack) = find_quant(&candidates, &recommended) {
            return Ok(selection(
                pack,
                LaunchSelectionReason::AutoRecommended,
                None,
            ));
        }
        let fallback = best_installed_pack(&candidates);
        let chosen = canonical_quant_tag(&fallback.quant).to_string();
        return Ok(selection(
            fallback,
            LaunchSelectionReason::AutoBestInstalled,
            Some(LaunchPackNotice::RecommendedNotInstalled {
                recommended,
                chosen,
            }),
        ));
    }

    let fallback = best_installed_pack(&candidates);
    let chosen = canonical_quant_tag(&fallback.quant).to_string();
    Ok(selection(
        fallback,
        LaunchSelectionReason::AutoBestInstalled,
        Some(LaunchPackNotice::AmbiguousAutoFallback { chosen }),
    ))
}

pub fn installed_packs_for_model(
    packs: &[InstalledPack],
    model_ref: &str,
    catalog: Option<&ModelCatalog>,
) -> Vec<InstalledPack> {
    let model_ref = model_ref.trim();
    if model_ref.is_empty() {
        return Vec::new();
    }
    let Ok(parsed) = parse_model_ref(model_ref) else {
        return Vec::new();
    };
    let explicit_quant = parsed.tag.as_deref().map(canonical_quant_tag);
    let catalog_model_id = catalog
        .and_then(|catalog| {
            resolve_catalog_pull_with_profile(
                catalog,
                &CatalogPullRequest {
                    reference: parsed.family.clone(),
                    quant: None,
                    size: None,
                },
                None,
            )
            .ok()
        })
        .map(|resolved| resolved.model_id);

    packs
        .iter()
        .filter(|pack| {
            pack.pull == model_ref
                || catalog_model_id.as_deref() == Some(pack.model_id.as_str())
                || pack.model_id == parsed.family
                || family_aliases_match(&pack.model_id, &parsed.family)
        })
        .filter(|pack| {
            explicit_quant.is_none_or(|quant| {
                canonical_quant_tag(&pack.quant) == quant
                    || canonical_quant_tag(&pack.suffix) == quant
            })
        })
        .cloned()
        .collect()
}

fn recommended_quant(request: &LaunchPackRequest<'_>) -> Option<String> {
    let catalog = request.catalog?;
    resolve_catalog_pull_with_profile(
        catalog,
        &CatalogPullRequest {
            reference: request.model_ref.to_string(),
            quant: None,
            size: None,
        },
        Some(request.host_profile),
    )
    .ok()
    .map(|resolved| canonical_quant_tag(&resolved.quant).to_string())
}

fn nothing_installed_error(request: &LaunchPackRequest<'_>) -> LaunchPackError {
    let recommended = recommended_quant(request);
    let pull_hint = recommended
        .as_ref()
        .map(|quant| format!("openasr pull {}:{quant}", request.model_ref.trim()))
        .unwrap_or_else(|| format!("openasr pull {}", request.model_ref.trim()));
    LaunchPackError::NothingInstalled {
        model_ref: request.model_ref.trim().to_string(),
        recommended,
        pull_hint,
    }
}

fn find_quant(packs: &[InstalledPack], quant: &str) -> Option<InstalledPack> {
    packs
        .iter()
        .find(|pack| {
            canonical_quant_tag(&pack.quant) == quant || canonical_quant_tag(&pack.suffix) == quant
        })
        .cloned()
}

fn best_installed_pack(packs: &[InstalledPack]) -> InstalledPack {
    packs
        .iter()
        .max_by(|left, right| {
            quant_quality_rank(&left.quant)
                .cmp(&quant_quality_rank(&right.quant))
                .then_with(|| right.size_bytes.cmp(&left.size_bytes))
        })
        .expect("best_installed_pack requires non-empty candidates")
        .clone()
}

fn selection(
    pack: InstalledPack,
    reason: LaunchSelectionReason,
    notice: Option<LaunchPackNotice>,
) -> LaunchPackSelection {
    let effective_quant = canonical_quant_tag(&pack.quant).to_string();
    let runtime_model_id = format!("{}:{effective_quant}", pack.model_id);
    LaunchPackSelection {
        pack,
        effective_quant,
        runtime_model_id,
        reason,
        notice,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CatalogModel, CatalogModelKind, CatalogQuant, CatalogQuantPerf, LicenseClass};
    use std::path::PathBuf;

    fn pack(model_id: &str, quant: &str, suffix: &str) -> InstalledPack {
        InstalledPack {
            model_id: model_id.to_string(),
            display_name: model_id.to_string(),
            quant: quant.to_string(),
            suffix: suffix.to_string(),
            pull: format!("{model_id}:{suffix}"),
            filename: format!("{model_id}-{quant}.oasr"),
            path: PathBuf::from(format!("/tmp/{model_id}-{quant}.oasr")),
            url: format!("https://example.test/{model_id}-{quant}.oasr"),
            hf_revision: "rev".to_string(),
            sha256: "0".repeat(64),
            size_bytes: match canonical_quant_tag(quant) {
                "fp16" => 4_000,
                "q8_0" => 2_000,
                "q4_k" => 1_000,
                "q3_k" => 750,
                _ => 1,
            },
            installed_at_unix_seconds: 1,
            source: None,
        }
    }

    fn catalog() -> ModelCatalog {
        ModelCatalog {
            schema_version: 1,
            generated_at: "2026-01-01T00:00:00Z".to_string(),
            catalog_url: "https://example.test/catalog.json".to_string(),
            backends: Vec::new(),
            models: vec![CatalogModel {
                id: "qwen3-asr-0.6b".to_string(),
                kind: CatalogModelKind::AsrModel,
                capability: None,
                experimental: false,
                display_name: "Qwen3-ASR 0.6B".to_string(),
                family: "qwen".to_string(),
                aliases: vec!["qwen3".to_string(), "qwen3-asr".to_string()],
                pull_alias: Some("qwen3".to_string()),
                size: "0.6b".to_string(),
                languages: vec!["en".to_string(), "zh".to_string()],
                source_langs: Vec::new(),
                target_langs: Vec::new(),
                vendor: Some("Qwen".to_string()),
                license: "Apache-2.0".to_string(),
                license_url: "https://example.test/license".to_string(),
                license_class: LicenseClass::Permissive,
                hf_repo: "OpenASR/qwen3-asr-0.6b".to_string(),
                hf_revision: "rev".to_string(),
                public: true,
                min_cli_version: "0.0.0".to_string(),
                recommended_quant: "q8_0".to_string(),
                pull_recommended: "qwen3-asr-0.6b:q8".to_string(),
                sort_weight: 0,
                recommended: false,
                prose: None,
                prose_locales: None,
                quants: vec![
                    quant("fp16", "fp16", 4_000),
                    quant("q8_0", "q8", 2_000),
                    quant("q4_k", "q4", 1_000),
                    quant("q3_k", "q3", 750),
                ],
            }],
        }
    }

    fn quant(quant: &str, suffix: &str, peak_rss_bytes: u64) -> CatalogQuant {
        CatalogQuant {
            quant: quant.to_string(),
            suffix: suffix.to_string(),
            pull: format!("qwen3-asr-0.6b:{suffix}"),
            filename: format!("qwen3-asr-0.6b-{quant}.oasr"),
            url: format!("https://example.test/qwen3-asr-0.6b-{quant}.oasr"),
            mirrors: Vec::new(),
            sha256: "0".repeat(64),
            size_bytes: peak_rss_bytes / 2,
            recommended: quant == "q8_0",
            perf: Some(CatalogQuantPerf {
                rtf_cpu: None,
                rtf_metal: None,
                peak_rss_bytes: Some(peak_rss_bytes),
                jfk_wer_vs_fp16: None,
            }),
        }
    }

    fn request<'a>(
        model_ref: &'a str,
        preference: &'a QuantPreference,
        catalog: &'a ModelCatalog,
        memory_budget_bytes: Option<u64>,
    ) -> LaunchPackRequest<'a> {
        LaunchPackRequest {
            model_ref,
            preference,
            catalog: Some(catalog),
            host_profile: CatalogQuantRecommendationProfile {
                memory_budget_bytes,
            },
        }
    }

    #[test]
    fn pinned_quant_selects_matching_installed_pack() {
        let catalog = catalog();
        let preference = QuantPreference::pinned("q4");
        let selection = resolve_launch_pack(
            &[
                pack("qwen3-asr-0.6b", "q8_0", "q8"),
                pack("qwen3-asr-0.6b", "q4_k", "q4"),
            ],
            &request("qwen", &preference, &catalog, Some(2_000)),
        )
        .unwrap();
        assert_eq!(selection.effective_quant, "q4_k");
        assert_eq!(selection.reason, LaunchSelectionReason::Pinned);
        assert_eq!(selection.notice, None);
    }

    #[test]
    fn auto_selects_device_recommended_when_installed() {
        let catalog = catalog();
        let preference = QuantPreference::Auto;
        let selection = resolve_launch_pack(
            &[
                pack("qwen3-asr-0.6b", "q8_0", "q8"),
                pack("qwen3-asr-0.6b", "q4_k", "q4"),
            ],
            &request("qwen3", &preference, &catalog, Some(2_000)),
        )
        .unwrap();
        assert_eq!(selection.effective_quant, "q8_0");
        assert_eq!(selection.reason, LaunchSelectionReason::AutoRecommended);
    }

    #[test]
    fn auto_falls_back_to_best_installed_when_recommended_missing() {
        let catalog = catalog();
        let preference = QuantPreference::Auto;
        let selection = resolve_launch_pack(
            &[
                pack("qwen3-asr-0.6b", "q4_k", "q4"),
                pack("qwen3-asr-0.6b", "q3_k", "q3"),
            ],
            &request("qwen-asr", &preference, &catalog, Some(2_000)),
        )
        .unwrap();
        assert_eq!(selection.effective_quant, "q4_k");
        assert_eq!(selection.reason, LaunchSelectionReason::AutoBestInstalled);
        assert_eq!(
            selection.notice,
            Some(LaunchPackNotice::RecommendedNotInstalled {
                recommended: "q8_0".to_string(),
                chosen: "q4_k".to_string(),
            })
        );
    }

    #[test]
    fn auto_without_catalog_picks_best_installed_with_notice() {
        let preference = QuantPreference::Auto;
        let request = LaunchPackRequest {
            model_ref: "qwen3-asr-0.6b",
            preference: &preference,
            catalog: None,
            host_profile: CatalogQuantRecommendationProfile {
                memory_budget_bytes: None,
            },
        };
        let selection = resolve_launch_pack(
            &[
                pack("qwen3-asr-0.6b", "q3_k", "q3"),
                pack("qwen3-asr-0.6b", "q8_0", "q8"),
            ],
            &request,
        )
        .unwrap();
        assert_eq!(selection.effective_quant, "q8_0");
        assert_eq!(
            selection.notice,
            Some(LaunchPackNotice::AmbiguousAutoFallback {
                chosen: "q8_0".to_string(),
            })
        );
    }

    #[test]
    fn nothing_installed_returns_actionable_pull_hint() {
        let catalog = catalog();
        let preference = QuantPreference::Auto;
        let error = resolve_launch_pack(&[], &request("qwen", &preference, &catalog, Some(1_000)))
            .unwrap_err();
        assert_eq!(
            error,
            LaunchPackError::NothingInstalled {
                model_ref: "qwen".to_string(),
                recommended: Some("q4_k".to_string()),
                pull_hint: "openasr pull qwen:q4_k".to_string(),
            }
        );
    }
}
