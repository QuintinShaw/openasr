use super::*;
use std::fs;

fn catalog_json() -> String {
    r#"{
  "schema_version": 1,
  "generated_at": "2026-05-31T00:00:00Z",
  "catalog_url": "https://catalog.openasr.org/v1/catalog.json",
  "models": [
    {
      "id": "moonshine-tiny",
      "kind": "asr-model",
      "display_name": "Moonshine Tiny",
      "family": "moonshine",
      "aliases": ["moonshine", "ambiguous-family"],
      "pull_alias": "moonshine",
      "size": "tiny",
      "languages": ["en"],
      "vendor": "Useful Sensors",
      "license": "MIT",
      "license_url": "https://huggingface.co/UsefulSensors/moonshine-tiny",
      "license_class": "permissive",
      "hf_repo": "OpenASR/moonshine-tiny",
      "hf_revision": "0123456789abcdef0123456789abcdef01234567",
      "public": true,
      "min_cli_version": "0.1.0",
      "recommended_quant": "q8_0",
      "pull_recommended": "moonshine-tiny:q8",
      "prose": {
        "tagline": "Small English ASR",
        "overview": ["Tiny model"],
        "highlights": ["fast"]
      },
      "quants": [
        {
          "quant": "fp16",
          "suffix": "fp16",
          "pull": "moonshine-tiny:fp16",
          "filename": "moonshine-tiny-fp16.oasr",
          "url": "https://huggingface.co/OpenASR/moonshine-tiny/resolve/0123456789abcdef0123456789abcdef01234567/moonshine-tiny-fp16.oasr",
          "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
          "size_bytes": 20,
          "recommended": false,
          "perf": { "rtf_cpu": 0.2, "rtf_metal": 0.1, "peak_rss_bytes": 1000, "jfk_wer_vs_fp16": 0.0 }
        },
        {
          "quant": "q8_0",
          "suffix": "q8",
          "pull": "moonshine-tiny:q8",
          "filename": "moonshine-tiny-q8_0.oasr",
          "url": "https://huggingface.co/OpenASR/moonshine-tiny/resolve/0123456789abcdef0123456789abcdef01234567/moonshine-tiny-q8_0.oasr",
          "sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
          "size_bytes": 10,
          "recommended": true,
          "perf": { "rtf_cpu": 0.1, "rtf_metal": 0.05, "peak_rss_bytes": 800, "jfk_wer_vs_fp16": 0.01 }
        }
      ]
    },
    {
      "id": "moonshine-base",
      "kind": "asr-model",
      "display_name": "Moonshine Base",
      "family": "moonshine",
      "aliases": ["moonshine", "ambiguous-family"],
      "pull_alias": "moonshine",
      "size": "base",
      "languages": ["en"],
      "vendor": "Useful Sensors",
      "license": "MIT",
      "license_url": "https://huggingface.co/UsefulSensors/moonshine-base",
      "license_class": "permissive",
      "hf_repo": "OpenASR/moonshine-base",
      "hf_revision": "0123456789abcdef0123456789abcdef01234567",
      "public": true,
      "min_cli_version": "0.1.0",
      "recommended_quant": "q8_0",
      "pull_recommended": "moonshine-base:q8",
      "quants": [
        {
          "quant": "q8_0",
          "suffix": "q8",
          "pull": "moonshine-base:q8",
          "filename": "moonshine-base-q8_0.oasr",
          "url": "https://huggingface.co/OpenASR/moonshine-base/resolve/0123456789abcdef0123456789abcdef01234567/moonshine-base-q8_0.oasr",
          "sha256": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
          "size_bytes": 30,
          "recommended": true
        }
      ]
    }
  ]
}"#
    .to_string()
}

fn catalog_json_with_first_fp16_mirror(source: &str, url: &str) -> String {
    catalog_json().replace(
        r#""url": "https://huggingface.co/OpenASR/moonshine-tiny/resolve/0123456789abcdef0123456789abcdef01234567/moonshine-tiny-fp16.oasr",
          "sha256":"#,
        &format!(
            r#""url": "https://huggingface.co/OpenASR/moonshine-tiny/resolve/0123456789abcdef0123456789abcdef01234567/moonshine-tiny-fp16.oasr",
          "mirrors": [{{"source": "{source}", "url": "{url}"}}],
          "sha256":"#
        ),
    )
}

fn alias_contract_catalog() -> ModelCatalog {
    ModelCatalog {
        schema_version: 1,
        generated_at: "2026-06-04T00:00:00Z".to_string(),
        catalog_url: "fixture".to_string(),
        backends: Vec::new(),
        models: vec![
            alias_contract_model(
                "qwen3-asr-0.6b",
                "Qwen3-ASR 0.6B",
                "qwen",
                &["qwen3", "qwen3-asr"],
                Some("qwen3"),
                "0.6b",
                true,
            ),
            alias_contract_model(
                "qwen3-asr-1.7b",
                "Qwen3-ASR 1.7B",
                "qwen",
                &["qwen3", "qwen3-asr"],
                Some("qwen3"),
                "1.7b",
                true,
            ),
            alias_contract_model(
                "whisper-small",
                "Whisper Small",
                "whisper",
                &[],
                Some("whisper-small"),
                "small",
                true,
            ),
        ],
    }
}

fn alias_contract_model(
    id: &str,
    display_name: &str,
    family: &str,
    aliases: &[&str],
    pull_alias: Option<&str>,
    size: &str,
    public: bool,
) -> CatalogModel {
    let revision = "0123456789abcdef0123456789abcdef01234567";
    CatalogModel {
        id: id.to_string(),
        kind: CatalogModelKind::AsrModel,
        capability: None,
        experimental: false,
        display_name: display_name.to_string(),
        family: family.to_string(),
        aliases: aliases.iter().map(|alias| (*alias).to_string()).collect(),
        pull_alias: pull_alias.map(ToOwned::to_owned),
        size: size.to_string(),
        languages: vec!["en".to_string(), "zh".to_string()],
        language_mode: None,
        language_default: None,
        source_langs: Vec::new(),
        target_langs: Vec::new(),
        vendor: None,
        license: "Apache-2.0".to_string(),
        license_url: "https://example.invalid/license".to_string(),
        license_class: LicenseClass::Permissive,
        hf_repo: format!("OpenASR/{id}"),
        hf_revision: revision.to_string(),
        public,
        min_cli_version: "0.1.0".to_string(),
        recommended_quant: "q8_0".to_string(),
        pull_recommended: format!("{id}:q8"),
        sort_weight: 0,
        recommended: false,
        prose: None,
        prose_locales: None,
        quants: vec![
            alias_contract_quant(id, "fp16", "fp16", revision),
            alias_contract_quant(id, "q8_0", "q8", revision),
            alias_contract_quant(id, "q4_k", "q4", revision),
        ],
    }
}

fn alias_contract_quant(id: &str, quant: &str, suffix: &str, revision: &str) -> CatalogQuant {
    let peak_rss_bytes = match canonical_quant_tag(quant) {
        "fp16" => 16_u64 * 1024 * 1024 * 1024,
        "q8_0" => 8_u64 * 1024 * 1024 * 1024,
        "q4_k" => 4_u64 * 1024 * 1024 * 1024,
        _ => 0,
    };
    CatalogQuant {
        quant: quant.to_string(),
        suffix: suffix.to_string(),
        pull: format!("{id}:{suffix}"),
        filename: format!("{id}-{quant}.oasr"),
        url: format!("https://huggingface.co/OpenASR/{id}/resolve/{revision}/{id}-{quant}.oasr"),
        mirrors: Vec::new(),
        sha256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_string(),
        size_bytes: 1,
        recommended: quant == "q8_0",
        perf: Some(CatalogQuantPerf {
            rtf_cpu: None,
            rtf_metal: None,
            peak_rss_bytes: Some(peak_rss_bytes),
            jfk_wer_vs_fp16: None,
        }),
    }
}

fn resolve_contract_pull(catalog: &ModelCatalog, reference: &str) -> ResolvedCatalogPull {
    resolve_catalog_pull(
        catalog,
        &CatalogPullRequest {
            reference: reference.to_string(),
            quant: None,
            size: None,
        },
    )
    .unwrap()
}

fn without_qwen_per_model_aliases(mut catalog: ModelCatalog) -> ModelCatalog {
    for model in &mut catalog.models {
        if model.family == "qwen" {
            model.aliases.clear();
            model.pull_alias = None;
        }
    }
    catalog
}

fn runtime_variant_card(id: &str, quantization: &str) -> ModelCard {
    let mut card = test_model_card(id);
    card.family = Some(id.to_string());
    card.default_variant = Some("published".to_string());
    card.variant = Some(ModelVariantMetadata {
        tag: "published".to_string(),
        format: "oasr".to_string(),
        quantization: Some(quantization.to_string()),
        role: Some("default".to_string()),
    });
    card
}

fn capability_pack_model(id: &str, role: CatalogCapabilityRole) -> CatalogModel {
    let revision = "0123456789abcdef0123456789abcdef01234567";
    let mut model = alias_contract_model(id, id, id, &[], None, "embedder", true);
    model.kind = CatalogModelKind::CapabilityPack;
    model.capability = Some(CatalogCapability {
        feature: CATALOG_FEATURE_SPEAKER_DIARIZATION.to_string(),
        role,
    });
    model.recommended_quant = "f32".to_string();
    model.pull_recommended = format!("{id}:f32");
    model.quants = vec![alias_contract_quant(id, "f32", "f32", revision)];
    model
}

fn translation_model(id: &str, public: bool) -> CatalogModel {
    let revision = "0123456789abcdef0123456789abcdef01234567";
    let mut model = alias_contract_model(id, "Hy-MT2 1.8B", "hymt2", &[], None, "1.8b", public);
    model.kind = CatalogModelKind::TranslationModel;
    model.experimental = true;
    model.languages = vec!["en".to_string(), "zh".to_string()];
    model.source_langs = vec!["zh".to_string()];
    model.target_langs = vec!["en".to_string()];
    model.recommended_quant = "q4_k_m".to_string();
    model.pull_recommended = format!("{id}:q4km");
    model.quants = vec![alias_contract_quant(id, "q4_k_m", "q4km", revision)];
    model
}

#[test]
fn catalog_parser_resolves_id_quant_suffix() {
    let catalog = parse_model_catalog(&catalog_json(), "fixture").unwrap();

    let resolved = resolve_catalog_pull(
        &catalog,
        &CatalogPullRequest {
            reference: "moonshine-tiny:q8".to_string(),
            quant: None,
            size: None,
        },
    )
    .unwrap();

    assert_eq!(resolved.model_id, "moonshine-tiny");
    assert_eq!(resolved.quant, "q8_0");
    assert_eq!(resolved.suffix, "q8");
    assert_eq!(resolved.pull, "moonshine-tiny:q8");
    assert_eq!(resolved.license_class, LicenseClass::Permissive);
}

#[test]
fn catalog_parser_defaults_missing_kind_to_asr_model() {
    let contents = catalog_json().replace("      \"kind\": \"asr-model\",\n", "");

    let catalog = parse_model_catalog(&contents, "fixture").unwrap();

    assert!(
        catalog
            .models
            .iter()
            .all(|model| model.kind == CatalogModelKind::AsrModel)
    );
    assert!(catalog.models.iter().all(CatalogModel::is_market_listed));
}

#[test]
fn catalog_capability_packs_are_not_market_listed_but_are_feature_queryable() {
    let mut catalog = alias_contract_catalog();
    catalog.models.push(capability_pack_model(
        "wespeaker-voxceleb-resnet34-lm",
        CatalogCapabilityRole::SpeakerEmbedder,
    ));

    super::validate_model_catalog(&catalog).unwrap();

    let asr_model = catalog
        .models
        .iter()
        .find(|model| model.id == "qwen3-asr-0.6b")
        .unwrap();
    let capability_pack = catalog
        .models
        .iter()
        .find(|model| model.id == "wespeaker-voxceleb-resnet34-lm")
        .unwrap();
    assert!(asr_model.is_market_listed());
    assert!(!capability_pack.is_market_listed());

    let packs = catalog.capability_packs_for_feature(CATALOG_FEATURE_SPEAKER_DIARIZATION);
    assert_eq!(packs.len(), 1);
    assert_eq!(packs[0].id, "wespeaker-voxceleb-resnet34-lm");
}

#[test]
fn catalog_kind_matrix_controls_market_listing() {
    let mut catalog = alias_contract_catalog();
    catalog.models.push(capability_pack_model(
        "wespeaker-voxceleb-resnet34-lm",
        CatalogCapabilityRole::SpeakerEmbedder,
    ));
    catalog.models.push(translation_model("hymt2-1.8b", true));
    catalog
        .models
        .push(translation_model("private-translator", false));

    super::validate_model_catalog(&catalog).unwrap();

    let mut market_ids: Vec<_> = catalog
        .models
        .iter()
        .filter(|model| model.is_market_listed())
        .map(|model| model.id.as_str())
        .collect();
    market_ids.sort_unstable();

    assert_eq!(
        market_ids,
        vec![
            "hymt2-1.8b",
            "qwen3-asr-0.6b",
            "qwen3-asr-1.7b",
            "whisper-small",
        ]
    );
}

#[test]
fn catalog_translation_model_requires_translation_metadata() {
    let mut catalog = alias_contract_catalog();
    let mut model = translation_model("hymt2-1.8b", true);
    model.source_langs.clear();
    catalog.models.push(model);

    let error = super::validate_model_catalog(&catalog)
        .unwrap_err()
        .to_string();

    assert!(error.contains("source_langs"));
    assert!(error.contains("must not be empty"));
}

#[test]
fn catalog_translation_model_rejects_one_letter_language_code() {
    let mut catalog = alias_contract_catalog();
    let mut model = translation_model("hymt2-1.8b", true);
    model.source_langs = vec!["z".to_string()];
    catalog.models.push(model);

    let error = super::validate_model_catalog(&catalog)
        .unwrap_err()
        .to_string();

    assert!(error.contains("source_langs"));
    assert!(error.contains("invalid language code 'z'"));
}

#[test]
fn catalog_non_translation_model_rejects_translation_metadata() {
    let mut catalog = alias_contract_catalog();
    catalog.models[0].source_langs = vec!["zh".to_string()];
    catalog.models[0].target_langs = vec!["en".to_string()];

    let error = super::validate_model_catalog(&catalog)
        .unwrap_err()
        .to_string();

    assert!(error.contains("translation metadata"));
    assert!(error.contains("not translation-model"));
}

#[test]
fn speaker_diarization_required_pack_selects_wespeaker_embedder() {
    let mut catalog = alias_contract_catalog();
    catalog.models.push(capability_pack_model(
        "wespeaker-voxceleb-resnet34-lm",
        CatalogCapabilityRole::SpeakerEmbedder,
    ));
    catalog.models.push(capability_pack_model(
        "pyannote-segmentation-3.0",
        CatalogCapabilityRole::SpeakerSegmenter,
    ));

    let default_required = catalog
        .speaker_diarization_required_embedder_pack()
        .expect("WeSpeaker required pack");
    assert_eq!(default_required.id, "wespeaker-voxceleb-resnet34-lm");
}

#[test]
fn catalog_capability_pack_requires_capability_metadata() {
    let mut catalog = alias_contract_catalog();
    let mut pack = capability_pack_model(
        "wespeaker-voxceleb-resnet34-lm",
        CatalogCapabilityRole::SpeakerEmbedder,
    );
    pack.capability = None;
    catalog.models.push(pack);

    let error = super::validate_model_catalog(&catalog)
        .unwrap_err()
        .to_string();

    assert!(error.contains("kind capability-pack"));
    assert!(error.contains("no capability metadata"));
}

#[test]
fn catalog_asr_model_rejects_capability_metadata() {
    let mut catalog = alias_contract_catalog();
    catalog.models[0].capability = Some(CatalogCapability {
        feature: CATALOG_FEATURE_SPEAKER_DIARIZATION.to_string(),
        role: CatalogCapabilityRole::SpeakerEmbedder,
    });

    let error = super::validate_model_catalog(&catalog)
        .unwrap_err()
        .to_string();

    assert!(error.contains("capability metadata"));
    assert!(error.contains("asr-model"));
}

#[test]
fn canonical_quant_tag_maps_release_aliases_to_disk_names() {
    assert_eq!(canonical_quant_tag("q8"), "q8_0");
    assert_eq!(canonical_quant_tag("q8_0"), "q8_0");
    assert_eq!(canonical_quant_tag("q4"), "q4_k");
    assert_eq!(canonical_quant_tag("q4_k"), "q4_k");
    assert_eq!(canonical_quant_tag("q4_k_m"), "q4_k");
    assert_eq!(canonical_quant_tag("q3"), "q3_k");
    assert_eq!(canonical_quant_tag("q3_k"), "q3_k");
    assert_eq!(canonical_quant_tag("fp16"), "fp16");
}

#[test]
fn catalog_pull_resolves_series_aliases_and_default_sizes() {
    let catalog = alias_contract_catalog();
    let cases = [
        ("qwen", "qwen3-asr-0.6b", "q8_0", "qwen3-asr-0.6b:q8"),
        ("qwen-asr", "qwen3-asr-0.6b", "q8_0", "qwen3-asr-0.6b:q8"),
        ("qwen3", "qwen3-asr-0.6b", "q8_0", "qwen3-asr-0.6b:q8"),
        ("qwen3-asr", "qwen3-asr-0.6b", "q8_0", "qwen3-asr-0.6b:q8"),
        ("qwen:q8", "qwen3-asr-0.6b", "q8_0", "qwen3-asr-0.6b:q8"),
        (
            "qwen3-asr:q4_k_m",
            "qwen3-asr-0.6b",
            "q4_k",
            "qwen3-asr-0.6b:q4",
        ),
        ("whisper", "whisper-small", "q8_0", "whisper-small:q8"),
        ("whisper-small", "whisper-small", "q8_0", "whisper-small:q8"),
        ("whisper:q8", "whisper-small", "q8_0", "whisper-small:q8"),
        (
            "whisper-small:q8_0",
            "whisper-small",
            "q8_0",
            "whisper-small:q8",
        ),
    ];

    for (reference, model_id, quant, pull) in cases {
        let resolved = resolve_contract_pull(&catalog, reference);
        assert_eq!(resolved.model_id, model_id, "{reference}");
        assert_eq!(resolved.quant, quant, "{reference}");
        assert_eq!(resolved.pull, pull, "{reference}");
    }
}

#[test]
fn catalog_series_taxonomy_resolves_without_per_model_aliases() {
    let catalog = without_qwen_per_model_aliases(alias_contract_catalog());
    for reference in ["qwen", "qwen-asr", "qwen3", "qwen3-asr"] {
        let resolved = resolve_contract_pull(&catalog, reference);
        assert_eq!(resolved.model_id, "qwen3-asr-0.6b", "{reference}");
        assert_eq!(resolved.quant, "q8_0", "{reference}");
        assert_eq!(resolved.pull, "qwen3-asr-0.6b:q8", "{reference}");
    }

    let resolved = resolve_catalog_pull(
        &catalog,
        &CatalogPullRequest {
            reference: "qwen3-asr".to_string(),
            quant: Some("q4_k_m".to_string()),
            size: Some("1.7b".to_string()),
        },
    )
    .unwrap();

    assert_eq!(resolved.model_id, "qwen3-asr-1.7b");
    assert_eq!(resolved.quant, "q4_k");
    assert_eq!(resolved.pull, "qwen3-asr-1.7b:q4");
}

#[test]
fn runtime_model_ref_uses_catalog_series_and_quant_aliases() {
    let catalog = alias_contract_catalog();
    let cards = vec![
        runtime_variant_card("qwen3-asr-0.6b", "q8_0"),
        runtime_variant_card("qwen3-asr-1.7b", "q8_0"),
        runtime_variant_card("whisper-small", "q8_0"),
    ];
    let cases = [
        ("qwen", "qwen3-asr-0.6b", "q8_0", "qwen3-asr-0.6b:q8"),
        ("qwen:q8", "qwen3-asr-0.6b", "q8_0", "qwen3-asr-0.6b:q8"),
        (
            "qwen-asr:q8_0",
            "qwen3-asr-0.6b",
            "q8_0",
            "qwen3-asr-0.6b:q8",
        ),
        ("qwen3-asr", "qwen3-asr-0.6b", "q8_0", "qwen3-asr-0.6b:q8"),
        ("whisper", "whisper-small", "q8_0", "whisper-small:q8"),
        ("whisper-small", "whisper-small", "q8_0", "whisper-small:q8"),
        ("whisper:q8", "whisper-small", "q8_0", "whisper-small:q8"),
        (
            "whisper-small:q8_0",
            "whisper-small",
            "q8_0",
            "whisper-small:q8",
        ),
        (
            "qwen3-asr:q4_k_m",
            "qwen3-asr-0.6b",
            "q4_k",
            "qwen3-asr-0.6b:q4",
        ),
    ];

    for (reference, model_id, quant, pull) in cases {
        let resolved = resolve_runtime_model_ref(&cards, Some(&catalog), reference).unwrap();
        assert_eq!(
            resolved.source,
            RuntimeModelRefSource::Catalog,
            "{reference}"
        );
        assert_eq!(resolved.model_id, model_id, "{reference}");
        assert_eq!(resolved.quant.as_deref(), Some(quant), "{reference}");
        assert_eq!(
            resolved.runtime_model_id,
            format!("{model_id}:{quant}"),
            "{reference}"
        );
        assert_eq!(resolved.pull.as_deref(), Some(pull), "{reference}");
        assert_eq!(resolved.card.unwrap().id, model_id, "{reference}");
    }
}

#[test]
fn runtime_model_ref_falls_back_to_registry_variant_refs() {
    let catalog = alias_contract_catalog();
    let cards = vec![runtime_variant_card("qwen3-asr-0.6b", "q8_0")];

    let resolved =
        resolve_runtime_model_ref(&cards, Some(&catalog), "qwen3-asr-0.6b:published").unwrap();

    assert_eq!(resolved.source, RuntimeModelRefSource::Registry);
    assert_eq!(resolved.model_id, "qwen3-asr-0.6b");
    assert_eq!(resolved.quant.as_deref(), Some("q8_0"));
    assert_eq!(resolved.runtime_model_id, "qwen3-asr-0.6b:q8_0");
    assert_eq!(resolved.pull, None);
}

#[test]
fn catalog_pull_size_option_overrides_series_default_size() {
    let catalog = alias_contract_catalog();

    let resolved = resolve_catalog_pull(
        &catalog,
        &CatalogPullRequest {
            reference: "qwen".to_string(),
            quant: None,
            size: Some("1.7b".to_string()),
        },
    )
    .unwrap();

    assert_eq!(resolved.model_id, "qwen3-asr-1.7b");
    assert_eq!(resolved.pull, "qwen3-asr-1.7b:q8");
}

#[test]
fn catalog_pull_treats_reference_and_option_quant_aliases_as_equivalent() {
    let catalog = alias_contract_catalog();

    let resolved = resolve_catalog_pull(
        &catalog,
        &CatalogPullRequest {
            reference: "qwen:q4_k_m".to_string(),
            quant: Some("q4".to_string()),
            size: None,
        },
    )
    .unwrap();

    assert_eq!(resolved.model_id, "qwen3-asr-0.6b");
    assert_eq!(resolved.quant, "q4_k");
    assert_eq!(resolved.pull, "qwen3-asr-0.6b:q4");
}

#[test]
fn catalog_pull_reports_quant_conflicts_after_alias_normalization() {
    let catalog = alias_contract_catalog();

    let error = resolve_catalog_pull(
        &catalog,
        &CatalogPullRequest {
            reference: "qwen:q8".to_string(),
            quant: Some("q4_k_m".to_string()),
            size: None,
        },
    )
    .unwrap_err()
    .to_string();

    assert!(error.contains("Conflicting quant selection"));
    assert!(error.contains("q8"));
    assert!(error.contains("q4_k_m"));
}

#[test]
fn catalog_quant_recommendation_keeps_catalog_default_when_it_fits() {
    let catalog = alias_contract_catalog();
    let model = catalog
        .models
        .iter()
        .find(|model| model.id == "qwen3-asr-0.6b")
        .unwrap();

    let quant = recommend_catalog_quant(
        model,
        CatalogQuantRecommendationProfile {
            memory_budget_bytes: Some(12 * 1024 * 1024 * 1024),
        },
    )
    .unwrap();

    assert_eq!(quant.quant, "q8_0");
}

#[test]
fn catalog_quant_recommendation_downgrades_when_default_exceeds_budget() {
    let catalog = alias_contract_catalog();
    let model = catalog
        .models
        .iter()
        .find(|model| model.id == "qwen3-asr-0.6b")
        .unwrap();

    let quant = recommend_catalog_quant(
        model,
        CatalogQuantRecommendationProfile {
            memory_budget_bytes: Some(6 * 1024 * 1024 * 1024),
        },
    )
    .unwrap();

    assert_eq!(quant.quant, "q4_k");
}

#[test]
fn catalog_quant_recommendation_falls_back_to_default_without_viable_perf_data() {
    let catalog = alias_contract_catalog();
    let model = catalog
        .models
        .iter()
        .find(|model| model.id == "qwen3-asr-0.6b")
        .unwrap();

    let quant = recommend_catalog_quant(
        model,
        CatalogQuantRecommendationProfile {
            memory_budget_bytes: Some(1024),
        },
    )
    .unwrap();

    assert_eq!(quant.quant, "q8_0");
}

#[test]
fn catalog_pull_with_profile_uses_device_recommended_quant_for_bare_reference() {
    let catalog = alias_contract_catalog();
    let bare = CatalogPullRequest {
        reference: "qwen3-asr-0.6b".to_string(),
        quant: None,
        size: None,
    };

    // Roomy budget keeps the catalog default (q8_0).
    let roomy = resolve_catalog_pull_with_profile(
        &catalog,
        &bare,
        Some(CatalogQuantRecommendationProfile {
            memory_budget_bytes: Some(12 * 1024 * 1024 * 1024),
        }),
    )
    .unwrap();
    assert_eq!(roomy.quant, "q8_0");

    // Tight budget downgrades the default to q4_k.
    let tight = resolve_catalog_pull_with_profile(
        &catalog,
        &bare,
        Some(CatalogQuantRecommendationProfile {
            memory_budget_bytes: Some(6 * 1024 * 1024 * 1024),
        }),
    )
    .unwrap();
    assert_eq!(tight.quant, "q4_k");

    // An explicit quant always wins over the device profile.
    let explicit = CatalogPullRequest {
        reference: "qwen3-asr-0.6b:q4_k".to_string(),
        quant: None,
        size: None,
    };
    let pinned = resolve_catalog_pull_with_profile(
        &catalog,
        &explicit,
        Some(CatalogQuantRecommendationProfile {
            memory_budget_bytes: Some(12 * 1024 * 1024 * 1024),
        }),
    )
    .unwrap();
    assert_eq!(pinned.quant, "q4_k");

    // The plain wrapper (no profile) keeps the static catalog default.
    assert_eq!(resolve_catalog_pull(&catalog, &bare).unwrap().quant, "q8_0");
}

#[test]
fn catalog_parser_resolves_bare_id_to_recommended_quant() {
    let catalog = parse_model_catalog(&catalog_json(), "fixture").unwrap();

    let resolved = resolve_catalog_pull(
        &catalog,
        &CatalogPullRequest {
            reference: "moonshine-tiny".to_string(),
            quant: None,
            size: None,
        },
    )
    .unwrap();

    assert_eq!(resolved.pull, "moonshine-tiny:q8");
}

#[test]
fn catalog_parser_resolves_alias_with_size_disambiguation() {
    let catalog = parse_model_catalog(&catalog_json(), "fixture").unwrap();

    let resolved = resolve_catalog_pull(
        &catalog,
        &CatalogPullRequest {
            reference: "ambiguous-family".to_string(),
            quant: None,
            size: Some("base".to_string()),
        },
    )
    .unwrap();

    assert_eq!(resolved.pull, "moonshine-base:q8");
}

#[test]
fn catalog_parser_reports_ambiguous_aliases() {
    let catalog = parse_model_catalog(&catalog_json(), "fixture").unwrap();

    let error = resolve_catalog_pull(
        &catalog,
        &CatalogPullRequest {
            reference: "ambiguous-family".to_string(),
            quant: None,
            size: None,
        },
    )
    .unwrap_err()
    .to_string();

    assert!(error.contains("ambiguous"));
    assert!(error.contains("moonshine-tiny:q8"));
    assert!(error.contains("moonshine-base:q8"));
}

#[test]
fn catalog_loader_caches_file_source_and_falls_back_to_cache() {
    let temp = tempfile::tempdir().unwrap();
    let source_path = temp.path().join("source-catalog.json");
    let home = temp.path().join("home");
    fs::write(&source_path, catalog_json()).unwrap();

    let source = format!("file://{}", source_path.display());
    let catalog = load_model_catalog(Some(&source), &home).unwrap();
    assert_eq!(catalog.models.len(), 2);
    assert!(default_catalog_cache_path(&home).exists());

    fs::remove_file(&source_path).unwrap();
    let cached = load_model_catalog(Some(&source), &home).unwrap();
    assert_eq!(cached.models[0].id, "moonshine-tiny");
}

#[test]
fn catalog_loader_falls_back_to_cache_on_network_failure() {
    let temp = tempfile::tempdir().unwrap();
    let source_path = temp.path().join("source-catalog.json");
    let home = temp.path().join("home");
    fs::write(&source_path, catalog_json()).unwrap();

    let seeded_source = format!("file://{}", source_path.display());
    load_model_catalog(Some(&seeded_source), &home).unwrap();

    let error = load_model_catalog(Some("https://127.0.0.1:1/catalog.json"), &home)
        .unwrap_err()
        .to_string();
    assert!(error.contains("no usable signed cache"), "{error}");
}

#[test]
fn embedded_catalog_snapshot_verifies_and_parses_offline() {
    // The catalog snapshot compiled into the binary is the last-resort OFFLINE
    // fallback (after the network source and the on-disk cache): it must verify
    // against its embedded signature and parse with no network and a fresh home,
    // so a device that has never been online still shows the model list.
    let home = tempfile::tempdir().unwrap();
    let catalog = super::load_embedded_signed_catalog(home.path())
        .expect("embedded catalog should verify and parse offline");
    assert!(!catalog.models.is_empty());
    assert_eq!(catalog.catalog_url, super::DEFAULT_CATALOG_URL);
}

#[test]
fn embedded_catalog_language_mode_matches_core_language_mode_per_family() {
    // The desktop/web "recognition language" selector reads `language_mode`
    // (+ `language_default`) straight off the catalog rather than reimplementing
    // core's per-family LanguageMode resolution client-side. Pin the published
    // catalog's values for one representative model per family so a future
    // catalog regenerate (tooling/publish-model/scripts/_catalog.py's
    // `language_mode_for_model`) that silently drifts from
    // crate::models::language::LanguageMode / ggml_family_adapter's
    // LanguageFamilyHint is caught here, not just in the Python drift check.
    let home = tempfile::tempdir().unwrap();
    let catalog = super::load_embedded_signed_catalog(home.path())
        .expect("embedded catalog should verify and parse offline");
    let find = |id: &str| {
        catalog
            .models
            .iter()
            .find(|model| model.id == id)
            .unwrap_or_else(|| panic!("catalog model '{id}' missing"))
    };

    // Qwen3-ASR: DetectImplicit -- self-detects, no explicit selection.
    let qwen = find("qwen3-asr-1.7b");
    assert_eq!(qwen.language_mode, Some(CatalogLanguageMode::DetectImplicit));
    assert_eq!(qwen.language_default, None);

    // X-ASR zh-en: FixedMultilingual -- built-in bilingual set, no selection.
    let xasr = find("xasr-zh-en");
    assert_eq!(
        xasr.language_mode,
        Some(CatalogLanguageMode::FixedMultilingual)
    );
    assert_eq!(xasr.language_default, None);

    // Cohere transcribe: SpecifyOnly -- always conditioned, "en" default.
    let cohere = find("cohere-transcribe-03-2026");
    assert_eq!(cohere.language_mode, Some(CatalogLanguageMode::SpecifyOnly));
    assert_eq!(cohere.language_default.as_deref(), Some("en"));

    // Moonshine: FixedMonolingual -- intrinsically English.
    let moonshine = find("moonshine-tiny");
    assert_eq!(
        moonshine.language_mode,
        Some(CatalogLanguageMode::FixedMonolingual)
    );
    assert_eq!(moonshine.language_default.as_deref(), Some("en"));

    // Multilingual Whisper: DetectAndSpecify (WhisperVocabGated resolved
    // multilingual from the pack's vocab / the catalog's multi-language list).
    let whisper = find("whisper-base");
    assert_eq!(
        whisper.language_mode,
        Some(CatalogLanguageMode::DetectAndSpecify)
    );
    assert_eq!(whisper.language_default, None);

    // Whisper `*.en`: WhisperVocabGated resolved English-only -> FixedMonolingual.
    let whisper_en = find("whisper-base.en");
    assert_eq!(
        whisper_en.language_mode,
        Some(CatalogLanguageMode::FixedMonolingual)
    );
    assert_eq!(whisper_en.language_default.as_deref(), Some("en"));

    // hymt2 (translation-model) and the diarization capability packs are not
    // GgmlFamilyAdapterDescriptor ASR families -- no source-language axis, so
    // the field is omitted rather than guessed.
    for id in [
        "hymt2-1.8b",
        "pyannote-segmentation-3.0",
        "wespeaker-voxceleb-resnet34-lm",
    ] {
        let model = find(id);
        assert_eq!(model.language_mode, None, "{id} should omit language_mode");
        assert_eq!(model.language_default, None, "{id} should omit language_default");
    }
}

#[test]
fn signed_cache_miss_falls_back_to_embedded_for_default_source() {
    // Wiring: network failed (`error`) and there is no on-disk signed cache, so for
    // the canonical default catalog the loader drops to the embedded snapshot.
    let home = tempfile::tempdir().unwrap();
    let missing_cache = home.path().join("absent-catalog.json");
    let network_error = CatalogError::ReadCatalog {
        catalog_source: DEFAULT_CATALOG_URL.to_string(),
        message: "network unreachable".to_string(),
    };
    let catalog = super::load_cached_signed_catalog(
        DEFAULT_CATALOG_URL,
        home.path(),
        &missing_cache,
        network_error,
    )
    .expect("default-source fallback should reach the embedded catalog");
    assert!(!catalog.models.is_empty());
}

#[test]
fn signed_cache_miss_does_not_substitute_embedded_for_custom_source() {
    // Scoping: an explicit OPENASR_CATALOG_URL override (source != default) must NOT
    // be silently replaced with the bundled official catalog — the original error
    // surfaces instead.
    let home = tempfile::tempdir().unwrap();
    let missing_cache = home.path().join("absent-catalog.json");
    let custom = "https://example.com/my-catalog.json";
    let network_error = CatalogError::ReadCatalog {
        catalog_source: custom.to_string(),
        message: "network unreachable".to_string(),
    };
    let error =
        super::load_cached_signed_catalog(custom, home.path(), &missing_cache, network_error)
            .unwrap_err()
            .to_string();
    assert!(error.contains("no usable signed cache"), "{error}");
}

#[test]
fn embedded_catalog_rejected_on_epoch_rollback() {
    // Anti-rollback: a recorded epoch newer than the embedded snapshot's makes the
    // embedded fallback fail closed rather than downgrade an already-seen catalog.
    let home = tempfile::tempdir().unwrap();
    let verified = crate::catalog_security::verify_catalog_signature_manifest(
        super::EMBEDDED_CATALOG_JSON,
        super::EMBEDDED_CATALOG_SIGNATURE_JSON,
        DEFAULT_CATALOG_URL,
    )
    .expect("embedded manifest verifies");
    crate::catalog_security::record_catalog_epoch(home.path(), verified.catalog_epoch + 1).unwrap();
    let error = super::load_embedded_signed_catalog(home.path())
        .unwrap_err()
        .to_string();
    assert!(error.contains("embedded catalog rejected"), "{error}");
}

#[test]
fn catalog_model_available_for_current_build() {
    // The fixture's min_cli_version (0.1.0) is satisfied by the running build, so it
    // is Available — the complement of the future-min_cli_version RequiresUpdate case.
    let catalog = parse_model_catalog(&catalog_json(), "fixture").unwrap();
    assert!(matches!(
        catalog.models[0].availability(),
        ModelAvailability::Available
    ));
}

#[test]
fn catalog_loader_does_not_cache_invalid_source() {
    let temp = tempfile::tempdir().unwrap();
    let source_path = temp.path().join("source-catalog.json");
    let home = temp.path().join("home");
    fs::write(&source_path, catalog_json()).unwrap();

    let source = format!("file://{}", source_path.display());
    load_model_catalog(Some(&source), &home).unwrap();
    let cache_path = default_catalog_cache_path(&home);
    let cached_before = fs::read_to_string(&cache_path).unwrap();

    fs::write(
        &source_path,
        catalog_json().replace("\"schema_version\": 1", "\"schema_version\": 99"),
    )
    .unwrap();
    let error = load_model_catalog(Some(&source), &home)
        .unwrap_err()
        .to_string();

    assert!(error.contains("Unsupported model catalog schema_version 99"));
    assert_eq!(fs::read_to_string(&cache_path).unwrap(), cached_before);
}

#[test]
fn catalog_parser_rejects_unknown_schema_version() {
    let contents = catalog_json().replace("\"schema_version\": 1", "\"schema_version\": 99");

    let error = parse_model_catalog(&contents, "fixture")
        .unwrap_err()
        .to_string();

    assert!(error.contains("Unsupported model catalog schema_version 99"));
}

#[test]
fn catalog_parser_rejects_string_schema_version() {
    let contents = catalog_json().replace("\"schema_version\": 1", "\"schema_version\": \"1\"");

    let error = parse_model_catalog(&contents, "fixture")
        .unwrap_err()
        .to_string();

    assert!(error.contains("invalid type: string"));
    assert!(error.contains("expected u32"));
}

#[test]
fn catalog_parser_rejects_negative_schema_version() {
    let contents = catalog_json().replace("\"schema_version\": 1", "\"schema_version\": -1");

    let error = parse_model_catalog(&contents, "fixture")
        .unwrap_err()
        .to_string();

    assert!(error.contains("invalid value: integer `-1`"));
    assert!(error.contains("expected u32"));
}

#[test]
fn catalog_parser_rejects_missing_schema_version() {
    let contents = catalog_json().replace("  \"schema_version\": 1,\n", "");

    let error = parse_model_catalog(&contents, "fixture")
        .unwrap_err()
        .to_string();

    assert!(error.contains("missing field `schema_version`"));
}

#[test]
fn catalog_parser_rejects_branch_revision_urls() {
    let contents = catalog_json()
        .replace(
            "\"hf_revision\": \"0123456789abcdef0123456789abcdef01234567\"",
            "\"hf_revision\": \"main\"",
        )
        .replace(
            "/resolve/0123456789abcdef0123456789abcdef01234567/",
            "/resolve/main/",
        );

    let error = parse_model_catalog(&contents, "fixture")
        .unwrap_err()
        .to_string();

    assert!(error.contains("hf_revision must be a 40 hex character commit sha"));
}

#[test]
fn catalog_parser_rejects_untrusted_download_host() {
    let contents = catalog_json().replace(
        "https://huggingface.co/OpenASR/moonshine-tiny/resolve/0123456789abcdef0123456789abcdef01234567/moonshine-tiny-fp16.oasr",
        "https://evil.example/OpenASR/moonshine-tiny/resolve/0123456789abcdef0123456789abcdef01234567/moonshine-tiny-fp16.oasr",
    );

    let error = parse_model_catalog(&contents, "fixture")
        .unwrap_err()
        .to_string();

    assert!(error.contains("URL must be pinned to hf_repo, hf_revision, and filename"));
}

#[test]
fn catalog_parser_rejects_disabled_modelscope_mirror() {
    let mirror_url = "https://modelscope.cn/models/openasr/moonshine-tiny/resolve/bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb/moonshine-tiny-fp16.oasr";
    let contents = catalog_json_with_first_fp16_mirror("modelscope", mirror_url);

    let error = parse_model_catalog(&contents, "fixture")
        .unwrap_err()
        .to_string();

    assert!(error.contains("ModelScope mirrors are disabled"));
}

#[test]
fn catalog_parser_rejects_untrusted_mirror_host() {
    let contents = catalog_json_with_first_fp16_mirror(
        "modelscope",
        "https://evil.example/models/openasr/moonshine-tiny/resolve/0123456789abcdef0123456789abcdef01234567/moonshine-tiny-fp16.oasr",
    );

    let error = parse_model_catalog(&contents, "fixture")
        .unwrap_err()
        .to_string();

    assert!(error.contains("mirror URL host is not allowed"));
}

#[test]
fn catalog_parser_rejects_derived_modelscope_mirror_path() {
    let contents = catalog_json_with_first_fp16_mirror(
        "modelscope",
        "https://modelscope.cn/OpenASR/moonshine-tiny/resolve/0123456789abcdef0123456789abcdef01234567/moonshine-tiny-fp16.oasr",
    );

    let error = parse_model_catalog(&contents, "fixture")
        .unwrap_err()
        .to_string();

    assert!(error.contains("ModelScope mirrors are disabled"));
}

#[test]
fn catalog_parser_rejects_uppercase_modelscope_owner() {
    let contents = catalog_json_with_first_fp16_mirror(
        "modelscope",
        "https://modelscope.cn/models/OpenASR/moonshine-tiny/resolve/0123456789abcdef0123456789abcdef01234567/moonshine-tiny-fp16.oasr",
    );

    let error = parse_model_catalog(&contents, "fixture")
        .unwrap_err()
        .to_string();

    assert!(error.contains("ModelScope mirrors are disabled"));
}

#[test]
fn catalog_parser_rejects_windows_separator_filenames() {
    let contents = catalog_json().replace(
        r#""filename": "moonshine-tiny-q8_0.oasr""#,
        r#""filename": "nested\\moonshine-tiny-q8_0.oasr""#,
    );

    let error = parse_model_catalog(&contents, "fixture")
        .unwrap_err()
        .to_string();

    assert!(error.contains("filename must be a local .oasr basename"));
}

#[test]
fn catalog_with_future_min_cli_version_loads_but_gates_pull() {
    let contents = catalog_json().replace(
        r#""min_cli_version": "0.1.0""#,
        r#""min_cli_version": "999.0.0""#,
    );

    // An older build must still SEE newer models: the catalog parses rather than
    // failing to load wholesale.
    let catalog = parse_model_catalog(&contents, "fixture").expect("catalog should still parse");
    let model = catalog
        .models
        .iter()
        .find(|model| model.min_cli_version == "999.0.0")
        .expect("model with future min_cli_version present");

    // It is surfaced as "requires update" (not hidden, not a load failure).
    assert!(matches!(
        model.availability(),
        ModelAvailability::RequiresUpdate { .. }
    ));

    // ...but actually pulling it is refused with a clear "update OpenASR" error.
    let request = CatalogPullRequest {
        reference: model.id.clone(),
        quant: None,
        size: None,
    };
    let error = resolve_catalog_pull(&catalog, &request)
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("requires OpenASR >="),
        "expected requires-update gate, got: {error}"
    );
}

#[test]
fn catalog_parser_rejects_drifted_pull_strings() {
    let contents = catalog_json().replace(
        "\"pull\": \"moonshine-tiny:q8\"",
        "\"pull\": \"moonshine:q8\"",
    );

    let error = parse_model_catalog(&contents, "fixture")
        .unwrap_err()
        .to_string();

    assert!(error.contains("pull must be '<id>:<suffix>'"));
}

// ---- backends[] : downloadable GPU plugin packs (Phase 2 catalog surface) ----

const BACKEND_SHA_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const BACKEND_SHA_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

fn catalog_json_with_backends(backends_json: &str) -> String {
    catalog_json().replace(
        "  \"models\": [",
        &format!("  \"backends\": [\n{backends_json}\n  ],\n  \"models\": ["),
    )
}

fn valid_hip_backend_json() -> String {
    format!(
        r#"    {{
      "id": "hip-radeon",
      "vendor": "hip",
      "version": "0.13.1+643b5659",
      "display_name": "AMD ROCm (HIP)",
      "targets": ["gfx1100", "gfx1200"],
      "min_cli_version": "0.1.0",
      "files": [
        {{"filename": "ggml-hip.dll", "role": "plugin", "url": "https://example.test/ggml-hip.dll", "sha256": "{BACKEND_SHA_A}", "size_bytes": 1048576}},
        {{"filename": "rocblas-library.zip", "role": "archive", "extract_subdir": "rocblas/library", "url": "https://example.test/rocblas-library.zip", "sha256": "{BACKEND_SHA_B}", "size_bytes": 157286400}}
      ]
    }}"#
    )
}

#[test]
fn catalog_parser_accepts_backend_entries() {
    let catalog = parse_model_catalog(
        &catalog_json_with_backends(&valid_hip_backend_json()),
        "fixture",
    )
    .unwrap();
    assert_eq!(catalog.backends.len(), 1);
    let backend = &catalog.backends[0];
    assert_eq!(backend.id, "hip-radeon");
    assert_eq!(backend.vendor, CatalogBackendVendor::Hip);
    assert_eq!(
        backend.targets,
        vec!["gfx1100".to_string(), "gfx1200".to_string()]
    );
    let plugin = backend
        .files
        .iter()
        .find(|file| file.role == CatalogBackendFileRole::Plugin)
        .expect("plugin file");
    assert_eq!(plugin.filename, "ggml-hip.dll");
    assert!(plugin.extract_subdir.is_none());
    let archive = backend
        .files
        .iter()
        .find(|file| file.role == CatalogBackendFileRole::Archive)
        .expect("archive file");
    assert_eq!(archive.extract_subdir.as_deref(), Some("rocblas/library"));
}

#[test]
fn catalog_without_backends_defaults_to_empty() {
    let catalog = parse_model_catalog(&catalog_json(), "fixture").unwrap();
    assert!(catalog.backends.is_empty());
}

#[test]
fn catalog_parser_rejects_backend_without_plugin() {
    let no_plugin =
        valid_hip_backend_json().replace("\"role\": \"plugin\"", "\"role\": \"runtime\"");
    let error = parse_model_catalog(&catalog_json_with_backends(&no_plugin), "fixture")
        .unwrap_err()
        .to_string();
    assert!(error.contains("exactly one plugin file"));
}

#[test]
fn catalog_parser_rejects_backend_with_bad_sha256() {
    let bad = valid_hip_backend_json().replace(BACKEND_SHA_A, "tooshort");
    let error = parse_model_catalog(&catalog_json_with_backends(&bad), "fixture")
        .unwrap_err()
        .to_string();
    assert!(error.contains("sha256 must be 64 hex characters"));
}

#[test]
fn catalog_parser_rejects_archive_extract_subdir_traversal() {
    let evil = valid_hip_backend_json().replace("rocblas/library", "../../etc");
    let error = parse_model_catalog(&catalog_json_with_backends(&evil), "fixture")
        .unwrap_err()
        .to_string();
    assert!(error.contains("safe relative path"));
}

#[test]
fn catalog_parser_rejects_archive_without_extract_subdir() {
    let no_subdir =
        valid_hip_backend_json().replace(", \"extract_subdir\": \"rocblas/library\"", "");
    let error = parse_model_catalog(&catalog_json_with_backends(&no_subdir), "fixture")
        .unwrap_err()
        .to_string();
    assert!(error.contains("must declare extract_subdir"));
}

#[test]
fn catalog_parser_rejects_extract_subdir_on_non_archive() {
    let bad = valid_hip_backend_json().replace(
        "\"filename\": \"ggml-hip.dll\", \"role\": \"plugin\"",
        "\"filename\": \"ggml-hip.dll\", \"extract_subdir\": \"x\", \"role\": \"plugin\"",
    );
    let error = parse_model_catalog(&catalog_json_with_backends(&bad), "fixture")
        .unwrap_err()
        .to_string();
    assert!(error.contains("not an archive"));
}

#[test]
fn empty_backends_omitted_from_serialized_catalog() {
    let catalog = parse_model_catalog(&catalog_json(), "fixture").unwrap();
    let json = serde_json::to_string(&catalog).unwrap();
    assert!(!json.contains("backends"));
}

#[test]
fn present_backends_round_trip_through_serialization() {
    let catalog = parse_model_catalog(
        &catalog_json_with_backends(&valid_hip_backend_json()),
        "fixture",
    )
    .unwrap();
    let json = serde_json::to_string(&catalog).unwrap();
    let reparsed = parse_model_catalog(&json, "fixture").unwrap();
    assert_eq!(reparsed.backends, catalog.backends);
}

#[test]
fn resolve_catalog_backend_pull_returns_the_matching_pack() {
    let catalog = parse_model_catalog(
        &catalog_json_with_backends(&valid_hip_backend_json()),
        "fixture",
    )
    .unwrap();
    let resolved = resolve_catalog_backend_pull(&catalog, "hip-radeon").unwrap();
    assert_eq!(resolved.backend_id, "hip-radeon");
    assert_eq!(resolved.vendor, CatalogBackendVendor::Hip);
    assert_eq!(resolved.version, "0.13.1+643b5659");
    assert_eq!(resolved.files.len(), 2);
    assert!(
        resolved
            .files
            .iter()
            .any(|file| file.role == CatalogBackendFileRole::Plugin)
    );
}

#[test]
fn resolve_catalog_backend_pull_reports_available_on_unknown_id() {
    let catalog = parse_model_catalog(
        &catalog_json_with_backends(&valid_hip_backend_json()),
        "fixture",
    )
    .unwrap();
    let error = resolve_catalog_backend_pull(&catalog, "cuda").unwrap_err();
    match error {
        BackendResolutionError::UnknownBackend {
            reference,
            available,
        } => {
            assert_eq!(reference, "cuda");
            assert!(available.contains("hip-radeon"));
        }
        other => panic!("expected UnknownBackend, got {other:?}"),
    }
}

#[test]
fn resolve_catalog_backend_pull_errors_when_no_backends() {
    let catalog = parse_model_catalog(&catalog_json(), "fixture").unwrap();
    assert_eq!(
        resolve_catalog_backend_pull(&catalog, "hip-radeon").unwrap_err(),
        BackendResolutionError::NoBackends
    );
}
