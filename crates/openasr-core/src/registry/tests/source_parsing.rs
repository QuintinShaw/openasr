use super::*;

#[test]
fn whisper_model_card_resolves_by_family_and_tag() {
    let cards = load_registry(test_model_registry_dir()).unwrap();

    let resolved = resolve_registry_model_ref(&cards, "whisper-small:published").unwrap();

    assert_eq!(resolved.card.id, "whisper-small");
    assert_eq!(resolved.family, "whisper-small");
    assert_eq!(resolved.tag.as_deref(), Some("published"));
    assert_eq!(resolved.card.variant_format(), Some("oasr"));
    assert_eq!(resolved.card.variant_quantization(), Some("q8_0"));
    assert!(resolved.is_default_variant);
}

#[test]
fn qwen_model_card_resolves_by_family_and_tag() {
    let cards = load_registry(test_model_registry_dir()).unwrap();

    let resolved = resolve_registry_model_ref(&cards, "qwen3-asr-0.6b:published").unwrap();

    assert_eq!(resolved.card.id, "qwen3-asr-0.6b");
    assert_eq!(resolved.family, "qwen3-asr-0.6b");
    assert_eq!(resolved.tag.as_deref(), Some("published"));
    assert_eq!(resolved.card.variant_format(), Some("oasr"));
    assert!(resolved.is_default_variant);
}

#[test]
fn cohere_model_card_resolves_by_family_and_tag() {
    let cards = load_registry(test_model_registry_dir()).unwrap();

    let resolved =
        resolve_registry_model_ref(&cards, "cohere-transcribe-03-2026:published").unwrap();

    assert_eq!(resolved.card.id, "cohere-transcribe-03-2026");
    assert_eq!(resolved.family, "cohere-transcribe-03-2026");
    assert_eq!(resolved.tag.as_deref(), Some("published"));
    assert_eq!(resolved.card.variant_format(), Some("oasr"));
    assert_eq!(resolved.card.variant_quantization(), Some("q8_0"));
    assert!(resolved.is_default_variant);
}

#[test]
fn whisper_unknown_tag_lists_available_tags() {
    let cards = load_registry(test_model_registry_dir()).unwrap();

    let error = resolve_registry_model_ref(&cards, "whisper-small:q8_0")
        .unwrap_err()
        .to_string();

    assert!(error.contains("Model family 'whisper-small' does not have variant tag 'q8_0'"));
    assert!(error.contains("Available tags: published"));
}

#[test]
fn model_reference_matching_requires_explicit_quant_when_source_has_quant() {
    assert!(!model_reference_matches_resolved_source(
        "qwen3-asr-0.6b",
        "qwen3-asr-0.6b:q8_0"
    ));
    assert!(model_reference_matches_resolved_source(
        "qwen3-asr-0.6b:q8",
        "qwen3-asr-0.6b:q8_0"
    ));
    assert!(model_reference_matches_resolved_source(
        "qwen3-asr-0.6b:q4_k_m",
        "qwen3-asr-0.6b:q4_k"
    ));
    assert!(!model_reference_matches_resolved_source(
        "qwen3-asr-1.7b:q8",
        "qwen3-asr-0.6b:q8_0"
    ));
    assert!(!model_reference_matches_resolved_source(
        "qwen3-asr-0.6b:q8",
        "qwen3-asr-0.6b"
    ));
}

#[test]
fn bundled_whisper_cards_are_published_packs() {
    let cards = load_registry(test_model_registry_dir()).unwrap();

    for id in ["whisper-small", "whisper-large-v3-turbo"] {
        let card = cards
            .iter()
            .find(|card| card.id == id)
            .unwrap_or_else(|| panic!("missing bundled Whisper published card: {id}"));

        assert_eq!(card.backend, "native");
        assert_eq!(card.variant_format(), Some("oasr"));
        assert_eq!(card.default_variant.as_deref(), Some("published"));
        assert!(card.quality_profile.contains("published"));
    }
}

#[test]
fn bundled_model_cards_have_default_variants() {
    let cards = load_registry(test_model_registry_dir()).unwrap();

    for card in cards {
        let variant = card.variant.as_ref().unwrap();
        assert_eq!(card.default_variant.as_deref(), Some(variant.tag.as_str()));
        assert_eq!(variant.format.as_str(), "oasr");
        assert!(
            variant
                .quantization
                .as_deref()
                .is_none_or(|q| !q.trim().is_empty())
        );
        assert!(card.is_default_variant());
    }
}

#[test]
fn minimal_card_without_download_fields_parses_and_validates() {
    let temp = tempfile::tempdir().unwrap();
    fs::write(
        temp.path().join("whisper-tiny.toml"),
        valid_card_toml("whisper-tiny"),
    )
    .unwrap();

    let cards = load_registry(temp.path()).unwrap();
    validate_card(Path::new("whisper-tiny.toml"), &cards[0]).unwrap();
}
