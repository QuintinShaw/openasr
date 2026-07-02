use super::*;

#[test]
fn parse_model_ref_accepts_model_and_tagged_model() {
    assert_eq!(
        parse_model_ref("whisper-tiny").unwrap(),
        ModelRef {
            family: "whisper-tiny".to_string(),
            tag: None
        }
    );
    assert_eq!(
        parse_model_ref("whisper-tiny:q4_0").unwrap(),
        ModelRef {
            family: "whisper-tiny".to_string(),
            tag: Some("q4_0".to_string())
        }
    );
}

#[test]
fn parse_model_ref_rejects_invalid_refs() {
    assert!(matches!(
        parse_model_ref("whisper-tiny:"),
        Err(ModelResolutionError::InvalidRef(_))
    ));
    assert!(matches!(
        parse_model_ref("whisper-tiny:q4_0:extra"),
        Err(ModelResolutionError::InvalidRef(_))
    ));
}

#[test]
fn resolves_no_tag_to_default_variant() {
    let cards = vec![
        variant_card("whisper-tiny-q8", "whisper-tiny", "q8_0", Some("q4_0")),
        variant_card("whisper-tiny", "whisper-tiny", "q4_0", Some("q4_0")),
    ];

    let resolved = resolve_registry_model_ref(&cards, "whisper-tiny").unwrap();

    assert_eq!(resolved.card.id, "whisper-tiny");
    assert_eq!(resolved.family, "whisper-tiny");
    assert_eq!(resolved.tag.as_deref(), Some("q4_0"));
    assert!(resolved.is_default_variant);
}

#[test]
fn resolves_explicit_tag_to_matching_variant() {
    let cards = vec![
        variant_card("whisper-tiny", "whisper-tiny", "q4_0", Some("q4_0")),
        variant_card("whisper-tiny-q8", "whisper-tiny", "q8_0", Some("q4_0")),
    ];

    let resolved = resolve_registry_model_ref(&cards, "whisper-tiny:q8_0").unwrap();

    assert_eq!(resolved.card.id, "whisper-tiny-q8");
    assert_eq!(resolved.tag.as_deref(), Some("q8_0"));
    assert!(!resolved.is_default_variant);
}

#[test]
fn ambiguous_family_without_default_fails_friendly() {
    let cards = vec![
        variant_card("tiny-q4", "whisper-tiny", "q4_0", None),
        variant_card("tiny-q8", "whisper-tiny", "q8_0", None),
    ];

    let error = resolve_registry_model_ref(&cards, "whisper-tiny")
        .unwrap_err()
        .to_string();

    assert!(error.contains("is ambiguous"));
    assert!(error.contains("whisper-tiny:q4_0"));
    assert!(error.contains("whisper-tiny:q8_0"));
}

#[test]
fn existing_direct_ids_still_resolve() {
    let cards = vec![test_model_card("legacy-model")];
    let resolved = resolve_registry_model_ref(&cards, "legacy-model").unwrap();

    assert_eq!(resolved.card.id, "legacy-model");
    assert_eq!(resolved.tag, None);
}

#[test]
fn default_variant_must_exist_when_configured() {
    let cards = vec![variant_card(
        "whisper-tiny",
        "whisper-tiny",
        "q8_0",
        Some("q4_0"),
    )];

    let error = validate_variant_index(&cards).unwrap_err().to_string();
    assert!(error.contains("default_variant 'q4_0' does not match any variant tag"));
}

#[test]
fn duplicate_family_tag_fails_validation() {
    let cards = vec![
        variant_card("tiny-a", "whisper-tiny", "q4_0", Some("q4_0")),
        variant_card("tiny-b", "whisper-tiny", "q4_0", Some("q4_0")),
    ];

    let error = validate_variant_index(&cards).unwrap_err().to_string();
    assert!(error.contains("duplicate variant 'whisper-tiny:q4_0'"));
}

#[test]
fn variant_metadata_requires_non_empty_fields() {
    let mut card = variant_card("whisper-tiny", "whisper-tiny", "q4_0", Some("q4_0"));
    card.variant.as_mut().unwrap().format.clear();
    assert_card_validation_error(card, "variant.format is required");

    let mut card = variant_card("whisper-tiny", "whisper-tiny", "q4_0", Some("q4_0"));
    card.variant.as_mut().unwrap().tag.clear();
    assert_card_validation_error(card, "variant.tag is required");

    let mut card = variant_card("whisper-tiny", "whisper-tiny", "q4_0", Some("q4_0"));
    card.variant.as_mut().unwrap().quantization = Some(String::new());
    assert_card_validation_error(card, "variant.quantization is required");
}
