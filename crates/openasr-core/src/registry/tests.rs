use std::fs;
use std::path::Path;

use super::validation::{validate_card, validate_unique_ids, validate_variant_index};
use super::*;
use crate::host::{host_quant_recommendation_profile, host_total_memory_bytes};

fn valid_card_toml(id: &str) -> String {
    format!(
        r#"
id = "{id}"
display_name = "Whisper Planning Card"
backend = "native"
task = "transcription"
languages = ["en", "zh"]
size = "tiny"
recommended_hardware = "CPU"
license = "MIT"
features = ["transcription"]
quality_profile = "planning-only"
source = "Native ASR Core planning metadata"
"#
    )
}

fn variant_card(id: &str, family: &str, tag: &str, default_variant: Option<&str>) -> ModelCard {
    let mut card = test_model_card(id);
    card.family = Some(family.to_string());
    card.default_variant = default_variant.map(ToOwned::to_owned);
    card.variant = Some(ModelVariantMetadata {
        tag: tag.to_string(),
        format: "oasr".to_string(),
        quantization: Some(tag.to_string()),
        role: default_variant
            .filter(|default| *default == tag)
            .map(|_| "default".to_string()),
    });
    card
}

fn assert_card_validation_error(card: ModelCard, expected: &str) {
    let error = validate_card(Path::new("whisper-tiny.toml"), &card)
        .unwrap_err()
        .to_string();

    assert!(error.contains(expected), "{error}");
}

// On every platform that ships a RAM probe (macOS sysctl, Linux /proc/meminfo,
// Windows GlobalMemoryStatusEx) the host total must come back positive, so the
// quant recommender budgets against real memory instead of falling back to the
// catalog default. The Windows arm specifically guards against the prior
// `None`-returning gap.
#[cfg(any(target_os = "macos", target_os = "linux", windows))]
#[test]
fn host_total_memory_is_probed_on_supported_platforms() {
    let total = host_total_memory_bytes();
    assert!(
        matches!(total, Some(bytes) if bytes > 0),
        "expected Some(>0) total RAM on a probed platform, got {total:?}"
    );

    let profile = host_quant_recommendation_profile();
    let expected_budget = total.map(|bytes| bytes / 4 * 3);
    assert_eq!(profile.memory_budget_bytes, expected_budget);
}

mod catalog;
mod diagnostics;
mod runtime_registry;
mod schema_validation;
mod source_parsing;
mod variant_selection;
