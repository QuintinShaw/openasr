use super::*;

fn bundled_catalog() -> ModelCatalog {
    let catalog_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../model-registry/catalog.json");
    let contents = fs::read_to_string(&catalog_path).unwrap();
    parse_model_catalog(&contents, &catalog_path.display().to_string()).unwrap()
}

fn publish_models_core() -> toml::Table {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tooling/publish-model/models-core.toml");
    toml::from_str(&fs::read_to_string(&path).unwrap()).unwrap()
}

#[test]
fn bundled_catalog_signature_verifies_committed_catalog_and_epoch() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../model-registry");
    let catalog_path = root.join("catalog.json");
    let manifest_path = root.join("catalog.signature.json");
    let epoch_path = root.join("catalog.epoch");
    let catalog_contents = fs::read_to_string(catalog_path).unwrap();
    let manifest_contents = fs::read_to_string(manifest_path).unwrap();
    let expected_epoch = fs::read_to_string(epoch_path)
        .unwrap()
        .trim()
        .parse::<u64>()
        .unwrap();

    let verified = crate::verify_catalog_signature_manifest(
        &catalog_contents,
        &manifest_contents,
        default_catalog_url(),
    )
    .unwrap();

    assert_eq!(verified.catalog_epoch, expected_epoch);
    assert_eq!(verified.key_id, crate::CATALOG_SIGNATURE_KEY_ID);
}

#[test]
fn public_catalog_projection_signature_verifies_and_excludes_private_entries() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../model-registry");
    let public_contents = fs::read_to_string(root.join("catalog.public.json")).unwrap();
    let manifest_contents = fs::read_to_string(root.join("catalog.public.signature.json")).unwrap();
    let expected_epoch = fs::read_to_string(root.join("catalog.epoch"))
        .unwrap()
        .trim()
        .parse::<u64>()
        .unwrap();

    // The public projection is signed against the same HF-canonical catalog_url; it
    // is the artifact catalog.openasr.org serves and the binary embeds.
    let verified = crate::verify_catalog_signature_manifest(
        &public_contents,
        &manifest_contents,
        default_catalog_url(),
    )
    .unwrap();
    assert_eq!(verified.catalog_epoch, expected_epoch);
    assert_eq!(verified.key_id, crate::CATALOG_SIGNATURE_KEY_ID);

    let public = parse_model_catalog(&public_contents, "public-projection").unwrap();
    // No staged/private entries may ship in the public artifact.
    assert!(
        public.models.iter().all(|model| model.public),
        "public catalog projection must contain only public:true models"
    );
    // The projection's model set must equal the full catalog's public:true set, so
    // the embedded/served public catalog cannot silently drift from the source.
    let mut projected: Vec<_> = public
        .models
        .iter()
        .map(|model| model.id.as_str())
        .collect();
    projected.sort_unstable();
    let full = bundled_catalog();
    let mut expected: Vec<_> = full
        .models
        .iter()
        .filter(|model| model.public)
        .map(|model| model.id.as_str())
        .collect();
    expected.sort_unstable();
    assert_eq!(
        projected, expected,
        "public projection drifted from the full catalog's public:true set; re-run the catalog publish"
    );
}

#[test]
fn embedded_catalog_fingerprint_matches_committed_public_catalog_and_signature() {
    use sha2::{Digest, Sha256};

    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../model-registry");
    let public_contents = fs::read_to_string(root.join("catalog.public.json")).unwrap();
    let manifest_contents = fs::read_to_string(root.join("catalog.public.signature.json")).unwrap();
    let manifest: crate::CatalogSignatureManifest =
        serde_json::from_str(&manifest_contents).unwrap();

    let expected_sha256 = {
        let mut hasher = Sha256::new();
        hasher.update(public_contents.as_bytes());
        hasher
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
    };

    let (catalog_sha256, catalog_epoch) = crate::embedded_catalog_fingerprint().unwrap();

    // Locks the CLI `catalog-fingerprint` contract: the fingerprint is byte-identical
    // to sha256(model-registry/catalog.public.json) -- the exact bytes `include_str!`
    // embeds -- and the epoch matches the committed signature manifest's epoch, so a
    // packaging gate comparing this against a copied catalog resource is meaningful.
    assert_eq!(catalog_sha256, expected_sha256);
    assert_eq!(catalog_sha256, manifest.catalog_sha256);
    assert_eq!(catalog_epoch, manifest.catalog_epoch);
}

/// Python<->Rust drift contract for the signed catalog's `language_labels` map
/// (design (c), analogous to the canonical quant-tag contract). The Python
/// emitter in tooling/publish-model/scripts/_catalog.py writes the map into the
/// committed catalog; this pins every emitted entry back to
/// `language_display_label`, and requires exactly the curated code set (the
/// Sinitic base codes plus every registered dialect code) so neither side can
/// add, drop, or misspell a label without the other.
#[test]
fn bundled_catalog_language_labels_match_rust_display_table() {
    use crate::models::language::{REGISTERED_DIALECT_CODES, language_display_label};

    let catalog = bundled_catalog();
    let labels = &catalog.language_labels;

    // Exactly the curated set: Sinitic base codes + every registered dialect.
    let mut expected_codes: Vec<String> = vec!["zh".into(), "yue".into(), "wuu".into()];
    expected_codes.extend(REGISTERED_DIALECT_CODES.iter().map(|code| code.to_string()));
    expected_codes.sort_unstable();
    expected_codes.dedup();
    let mut got_codes: Vec<String> = labels.keys().cloned().collect();
    got_codes.sort_unstable();
    assert_eq!(
        got_codes, expected_codes,
        "catalog language_labels code set drifted from the curated display table"
    );

    // Every emitted label must equal the Rust source of truth verbatim.
    for code in &expected_codes {
        let source = language_display_label(code)
            .unwrap_or_else(|| panic!("curated code '{code}' has no language_display_label"));
        let label = &labels[code];
        assert_eq!(
            label.en, source.en,
            "catalog language_labels['{code}'].en drifted from language_display_label"
        );
        assert_eq!(
            label.zh_cn, source.zh_cn,
            "catalog language_labels['{code}'].zh_cn drifted from language_display_label"
        );
    }

    // The public projection the binary embeds must carry the identical map.
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../model-registry");
    let public_contents = fs::read_to_string(root.join("catalog.public.json")).unwrap();
    let public = parse_model_catalog(&public_contents, "public-projection").unwrap();
    assert_eq!(
        public.language_labels, catalog.language_labels,
        "public projection language_labels drifted from the full catalog; re-run the catalog publish"
    );
}

#[test]
fn loads_toml_model_cards_sorted_by_id() {
    let temp = tempfile::tempdir().unwrap();
    fs::write(
        temp.path().join("whisper-tiny.toml"),
        valid_card_toml("whisper-tiny"),
    )
    .unwrap();
    fs::write(
        temp.path().join("whisper-base.toml"),
        valid_card_toml("whisper-base"),
    )
    .unwrap();
    fs::write(temp.path().join("README.md"), "ignored").unwrap();

    let cards = load_registry(temp.path()).unwrap();

    assert_eq!(cards.len(), 2);
    assert_eq!(cards[0].id, "whisper-base");
    assert_eq!(cards[1].id, "whisper-tiny");
}

#[test]
fn bundled_model_cards_parse_successfully() {
    let cards = load_registry(default_registry_dir()).unwrap();

    assert!(cards.len() >= 5);
    assert!(
        cards
            .iter()
            .any(|card| card.id == "cohere-transcribe-03-2026")
    );
    assert!(cards.iter().any(|card| card.id == "qwen3-asr-0.6b"));
    assert!(cards.iter().any(|card| card.id == "qwen3-asr-1.7b"));
    assert!(cards.iter().any(|card| card.id == "whisper-small"));
    assert!(cards.iter().any(|card| card.id == "whisper-large-v3-turbo"));
}

#[test]
fn bundled_registry_ordering_is_deterministic() {
    let cards = load_registry(default_registry_dir()).unwrap();
    let ids: Vec<_> = cards.iter().map(|card| card.id.as_str()).collect();

    assert_eq!(
        ids,
        vec![
            "qwen3-asr-0.6b",
            "cohere-transcribe-03-2026",
            "dolphin-base",
            "dolphin-cn-dialect-base",
            "dolphin-cn-dialect-small",
            "dolphin-small",
            "firered-aed-l-v2",
            "hymt2-1.8b",
            "moonshine-tiny",
            "parakeet-tdt-0.6b-v3",
            "pyannote-segmentation-3.0",
            "qwen3-asr-1.7b",
            "qwen3-forced-aligner-0.6b",
            "sensevoice-small",
            "wespeaker-voxceleb-resnet34-lm",
            "whisper-base",
            "whisper-base.en",
            "whisper-large-v3",
            "whisper-large-v3-turbo",
            "whisper-medium",
            "whisper-medium.en",
            "whisper-small",
            "whisper-small.en",
            "whisper-tiny",
            "whisper-tiny.en",
            "xasr-zh-en",
        ]
    );
}

#[test]
fn bundled_catalog_json_parses_and_matches_registry_cards() {
    let catalog = bundled_catalog();
    let cards = load_registry(default_registry_dir()).unwrap();
    let card_ids: Vec<_> = cards.iter().map(|card| card.id.as_str()).collect();

    for model in &catalog.models {
        assert!(
            card_ids.contains(&model.id.as_str()),
            "catalog model '{}' must have a registry card",
            model.id
        );
        assert!(
            model
                .quants
                .iter()
                .any(|quant| quant.quant == model.recommended_quant && quant.recommended),
            "catalog model '{}' must mark the recommended quant",
            model.id
        );
        for quant in &model.quants {
            assert_eq!(quant.pull, format!("{}:{}", model.id, quant.suffix));
            assert!(
                quant
                    .url
                    .contains(&format!("/resolve/{}/", model.hf_revision)),
                "catalog URL must be pinned to the immutable HF revision"
            );
        }
    }
}

#[test]
fn bundled_catalog_has_at_least_one_public_model() {
    let catalog = bundled_catalog();

    assert!(
        catalog.models.iter().any(|model| model.public),
        "committed catalog must include at least one public model"
    );
}

#[test]
fn bundled_catalog_public_ids_match_current_signed_release_projection() {
    let catalog = bundled_catalog();
    let mut public_ids: Vec<_> = catalog
        .models
        .iter()
        .filter(|model| model.public)
        .map(|model| model.id.as_str())
        .collect();
    public_ids.sort_unstable();

    assert_eq!(
        public_ids,
        vec![
            "cohere-transcribe-03-2026",
            "dolphin-cn-dialect-base",
            "dolphin-cn-dialect-small",
            "hymt2-1.8b",
            "moonshine-tiny",
            "pyannote-segmentation-3.0",
            "qwen3-asr-0.6b",
            "qwen3-asr-1.7b",
            "sensevoice-small",
            "wespeaker-voxceleb-resnet34-lm",
            "whisper-base",
            "whisper-base.en",
            "whisper-large-v3",
            "whisper-large-v3-turbo",
            "whisper-medium",
            "whisper-medium.en",
            "whisper-small",
            "whisper-small.en",
            "whisper-tiny",
            "whisper-tiny.en",
            "xasr-zh-en",
        ],
        "committed signed catalog public set changed; update catalog.json, catalog.signature.json, catalog.epoch, docs, and this release projection gate together"
    );
}

#[test]
fn bundled_catalog_market_listed_ids_exclude_capability_packs() {
    let catalog = bundled_catalog();
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
            "cohere-transcribe-03-2026",
            "dolphin-cn-dialect-base",
            "dolphin-cn-dialect-small",
            "hymt2-1.8b",
            "moonshine-tiny",
            "qwen3-asr-0.6b",
            "qwen3-asr-1.7b",
            "sensevoice-small",
            "whisper-base",
            "whisper-base.en",
            "whisper-large-v3",
            "whisper-large-v3-turbo",
            "whisper-medium",
            "whisper-medium.en",
            "whisper-small",
            "whisper-small.en",
            "whisper-tiny",
            "whisper-tiny.en",
            "xasr-zh-en"
        ],
        "market-listed set is public && kind in (asr-model, translation-model)"
    );
}

#[test]
fn bundled_catalog_declares_speaker_diarization_capability_packs() {
    let catalog = bundled_catalog();
    let packs = catalog.capability_packs_for_feature(CATALOG_FEATURE_SPEAKER_DIARIZATION);
    let mut pack_roles: Vec<_> = packs
        .iter()
        .map(|model| {
            let capability = model.capability.as_ref().unwrap();
            (model.id.as_str(), capability.role)
        })
        .collect();
    pack_roles.sort_unstable_by_key(|(id, _)| *id);

    assert_eq!(
        pack_roles,
        vec![
            (
                "pyannote-segmentation-3.0",
                CatalogCapabilityRole::SpeakerSegmenter
            ),
            (
                "wespeaker-voxceleb-resnet34-lm",
                CatalogCapabilityRole::SpeakerEmbedder
            ),
        ]
    );
    assert!(packs.iter().all(|model| !model.is_market_listed()));
}

#[test]
fn bundled_catalog_public_projection_is_non_empty_and_public_only() {
    let catalog = bundled_catalog();
    let public_models: Vec<_> = catalog
        .models
        .iter()
        .filter(|model| model.public)
        .cloned()
        .collect();

    assert!(
        !public_models.is_empty(),
        "public catalog projection must not be empty"
    );
    assert!(
        public_models.iter().all(|model| model.public),
        "public catalog projection must contain only public models"
    );
}

#[test]
fn bundled_catalog_public_models_are_release_public_in_source_metadata() {
    let catalog = bundled_catalog();
    let source = publish_models_core();

    for model in catalog.models.iter().filter(|model| model.public) {
        assert_eq!(
            source
                .get(&model.id)
                .and_then(|entry| entry.get("release_public"))
                .and_then(|value| value.as_bool()),
            Some(true),
            "public catalog model '{}' must be marked release_public=true in models-core.toml",
            model.id
        );
    }
}

#[test]
fn bundled_catalog_public_models_resolve_recommended_default_quant() {
    let catalog = bundled_catalog();
    let public_models: Vec<_> = catalog.models.iter().filter(|model| model.public).collect();

    assert!(
        !public_models.is_empty(),
        "committed catalog must include public models to validate"
    );
    for model in public_models {
        let recommended = model
            .quants
            .iter()
            .find(|quant| quant.quant == model.recommended_quant && quant.recommended)
            .unwrap_or_else(|| {
                panic!(
                    "public model '{}' must mark recommended_quant '{}' as recommended",
                    model.id, model.recommended_quant
                )
            });

        assert_eq!(
            model.pull_recommended, recommended.pull,
            "public model '{}' pull_recommended must point at the recommended quant",
            model.id
        );

        let default_pull = resolve_catalog_pull(
            &catalog,
            &CatalogPullRequest {
                reference: model.id.clone(),
                quant: None,
                size: None,
            },
        )
        .unwrap();
        assert_eq!(default_pull.model_id, model.id);
        assert_eq!(default_pull.quant, recommended.quant);
        assert_eq!(default_pull.pull, model.pull_recommended);

        let explicit_recommended_pull = resolve_catalog_pull(
            &catalog,
            &CatalogPullRequest {
                reference: model.pull_recommended.clone(),
                quant: None,
                size: None,
            },
        )
        .unwrap();
        assert_eq!(explicit_recommended_pull.model_id, model.id);
        assert_eq!(explicit_recommended_pull.quant, recommended.quant);
        assert_eq!(explicit_recommended_pull.pull, model.pull_recommended);

        let explicit_quant_option_pull = resolve_catalog_pull(
            &catalog,
            &CatalogPullRequest {
                reference: model.id.clone(),
                quant: Some(model.recommended_quant.clone()),
                size: None,
            },
        )
        .unwrap();
        assert_eq!(explicit_quant_option_pull.model_id, model.id);
        assert_eq!(explicit_quant_option_pull.quant, recommended.quant);
        assert_eq!(explicit_quant_option_pull.pull, model.pull_recommended);

        let explicit_suffix_option_pull = resolve_catalog_pull(
            &catalog,
            &CatalogPullRequest {
                reference: model.id.clone(),
                quant: Some(recommended.suffix.clone()),
                size: None,
            },
        )
        .unwrap();
        assert_eq!(explicit_suffix_option_pull.model_id, model.id);
        assert_eq!(explicit_suffix_option_pull.quant, recommended.quant);
        assert_eq!(explicit_suffix_option_pull.pull, model.pull_recommended);
    }
}

#[test]
fn bundled_catalog_public_models_have_legal_license_fields() {
    let catalog = bundled_catalog();
    let public_models: Vec<_> = catalog.models.iter().filter(|model| model.public).collect();

    assert!(
        !public_models.is_empty(),
        "committed catalog must include public models to validate"
    );
    for model in public_models {
        let normalized_license = model.license.to_ascii_lowercase();
        assert!(
            !model.license.trim().is_empty(),
            "public model '{}' license must not be empty",
            model.id
        );
        assert!(
            !normalized_license.contains("pending")
                && !normalized_license.contains("planning")
                && !normalized_license.contains("todo"),
            "public model '{}' license must be release-ready, got '{}'",
            model.id,
            model.license
        );
        assert!(
            model.license_url.starts_with("https://") && model.license_url.len() > "https://".len(),
            "public model '{}' license_url must be an https URL",
            model.id
        );
        assert!(
            matches!(
                &model.license_class,
                LicenseClass::Permissive | LicenseClass::Noncommercial | LicenseClass::Gated
            ),
            "public model '{}' license_class must be a supported catalog class",
            model.id
        );
    }
}

#[test]
fn bundled_model_cards_have_required_metadata() {
    let cards = load_registry(default_registry_dir()).unwrap();

    for card in cards {
        assert!(!card.display_name.is_empty());
        assert!(!card.backend.is_empty());
        assert!(!card.task.is_empty());
        assert!(!card.languages.is_empty());
        assert!(!card.size.is_empty());
        assert!(!card.recommended_hardware.is_empty());
        assert!(!card.license.is_empty());
        assert!(!card.features.is_empty());
        assert!(!card.quality_profile.is_empty());
        assert!(!card.source.is_empty());
    }
}

#[test]
fn bundled_model_ids_are_unique() {
    let cards = load_registry(default_registry_dir()).unwrap();
    validate_unique_ids(&cards).unwrap();
}

#[test]
fn duplicate_model_id_fails_validation() {
    let cards = vec![
        test_model_card("whisper-tiny"),
        test_model_card("whisper-tiny"),
    ];

    let error = validate_unique_ids(&cards).unwrap_err().to_string();
    assert!(error.contains("duplicate model id 'whisper-tiny'"));
}

#[test]
fn missing_required_field_fails_validation() {
    let temp = tempfile::tempdir().unwrap();
    let toml =
        valid_card_toml("whisper-tiny").replace("display_name = \"Whisper Planning Card\"\n", "");
    fs::write(temp.path().join("whisper-tiny.toml"), toml).unwrap();

    let error = load_registry(temp.path()).unwrap_err().to_string();
    assert!(error.contains("Could not parse model card"));
    assert!(error.contains("display_name"));
}

#[test]
fn invalid_backend_fails_validation() {
    let temp = tempfile::tempdir().unwrap();
    let toml = valid_card_toml("whisper-tiny")
        .replace("backend = \"native\"", "backend = \"faster-whisper\"");
    fs::write(temp.path().join("whisper-tiny.toml"), toml).unwrap();

    let error = load_registry(temp.path()).unwrap_err().to_string();
    assert!(error.contains("backend 'faster-whisper' is not supported"));
}

#[test]
fn invalid_task_fails_validation() {
    let temp = tempfile::tempdir().unwrap();
    let toml = valid_card_toml("whisper-tiny")
        .replace("task = \"transcription\"", "task = \"translation\"");
    fs::write(temp.path().join("whisper-tiny.toml"), toml).unwrap();

    let error = load_registry(temp.path()).unwrap_err().to_string();
    assert!(error.contains("task 'translation' is not supported"));
}

#[test]
fn empty_features_fail_validation() {
    let temp = tempfile::tempdir().unwrap();
    let toml =
        valid_card_toml("whisper-tiny").replace("features = [\"transcription\"]", "features = []");
    fs::write(temp.path().join("whisper-tiny.toml"), toml).unwrap();

    let error = load_registry(temp.path()).unwrap_err().to_string();
    assert!(error.contains("features must not be empty"));
}
