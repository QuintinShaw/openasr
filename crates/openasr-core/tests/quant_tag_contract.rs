//! Cross-language quant-tag contract.
//!
//! The desktop TypeScript `canonicalQuantTag` must canonicalize exactly like
//! `openasr_core::canonical_quant_tag`. Both sides consume the same JSON
//! fixture; if the mapping ever changes, change the fixture first and both
//! test suites will hold the implementations in lockstep.

use openasr_core::{ModelCard, ModelVariantMetadata, resolve_registry_model_ref};
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    input: String,
    canonical: String,
}

fn variant_card(id: &str, tag: &str, quant: &str) -> ModelCard {
    ModelCard {
        id: id.to_string(),
        family: Some(id.to_string()),
        default_variant: Some(tag.to_string()),
        variant: Some(ModelVariantMetadata {
            tag: tag.to_string(),
            format: "oasr".to_string(),
            quantization: Some(quant.to_string()),
            role: Some("default".to_string()),
        }),
        display_name: id.to_string(),
        backend: "native".to_string(),
        task: "transcription".to_string(),
        languages: vec!["en".to_string()],
        size: "tiny".to_string(),
        recommended_hardware: "CPU".to_string(),
        license: "MIT".to_string(),
        features: vec!["transcription".to_string()],
        quality_profile: "planning-only".to_string(),
        source: "quant-tag-contract".to_string(),
    }
}

#[test]
fn canonical_quant_tag_matches_shared_fixture() {
    let raw = include_str!("fixtures/quant_tag_cases.json");
    let fixture: Fixture = serde_json::from_str(raw).expect("quant tag fixture parses");
    assert!(
        fixture.cases.len() >= 10,
        "fixture must keep meaningful coverage"
    );
    for case in &fixture.cases {
        assert_eq!(
            openasr_core::canonical_quant_tag(&case.input),
            case.canonical,
            "canonical_quant_tag({:?}) drifted from the shared contract",
            case.input
        );
    }
}

#[test]
fn tagged_ref_resolution_uses_canonical_quant_aliases() {
    let raw = include_str!("fixtures/quant_tag_cases.json");
    let fixture: Fixture = serde_json::from_str(raw).expect("quant tag fixture parses");
    for case in &fixture.cases {
        let requested = case.input.trim();
        if requested.is_empty() {
            continue;
        }
        let cards = vec![variant_card(
            "alias-model",
            &case.canonical,
            &case.canonical,
        )];
        let resolved = resolve_registry_model_ref(&cards, &format!("alias-model:{requested}"))
            .unwrap_or_else(|error| {
                panic!(
                    "alias-model:{requested} should match canonical tag {}: {error}",
                    case.canonical
                )
            });
        assert_eq!(resolved.card.id, "alias-model");
    }
}

#[test]
fn published_registry_tag_does_not_match_quant_alias() {
    let cards = vec![variant_card("firered-aed-l-v2", "published", "q4_k")];
    let resolved = resolve_registry_model_ref(&cards, "firered-aed-l-v2:q4")
        .expect("q4 must match the published card's pack quant");
    assert_eq!(resolved.card.variant_tag(), Some("published"));

    let published = resolve_registry_model_ref(&cards, "firered-aed-l-v2:published")
        .expect("published must still match the registry tag");
    assert_eq!(published.card.variant_tag(), Some("published"));

    let error = resolve_registry_model_ref(&cards, "firered-aed-l-v2:q8")
        .unwrap_err()
        .to_string();
    assert!(error.contains("does not have variant tag 'q8'"), "{error}");

    let q4_only = vec![variant_card("firered-aed-l-v2", "q4_k", "q4_k")];
    let error = resolve_registry_model_ref(&q4_only, "firered-aed-l-v2:published")
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("does not have variant tag 'published'"),
        "{error}"
    );
}
