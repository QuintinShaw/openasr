use super::*;

fn registry() -> Vec<ModelCard> {
    vec![
        crate::registry::test_model_card("qwen3-asr-0.6b"),
        crate::registry::test_model_card("whisper-large-v3-turbo"),
        crate::registry::test_model_card("whisper-small"),
    ]
}

fn variant_registry() -> Vec<ModelCard> {
    let mut card = crate::registry::test_model_card("whisper-large-v3-turbo");
    card.family = Some("whisper".to_string());
    card.default_variant = Some("candidate".to_string());
    card.variant = Some(crate::ModelVariantMetadata {
        tag: "candidate".to_string(),
        format: "oasr".to_string(),
        quantization: None,
        role: Some("default".to_string()),
    });
    vec![card]
}

fn catalog_model(id: &str, family: &str, aliases: &[&str], size: &str) -> ModelCatalog {
    let revision = "0123456789abcdef0123456789abcdef01234567";
    ModelCatalog {
        schema_version: 1,
        generated_at: "2026-06-04T00:00:00Z".to_string(),
        catalog_url: "fixture".to_string(),
        backends: Vec::new(),
        models: vec![crate::CatalogModel {
            id: id.to_string(),
            kind: crate::CatalogModelKind::AsrModel,
            capability: None,
            experimental: false,
            display_name: id.to_string(),
            family: family.to_string(),
            aliases: aliases.iter().map(|alias| (*alias).to_string()).collect(),
            pull_alias: aliases.first().map(|alias| (*alias).to_string()),
            size: size.to_string(),
            languages: vec!["en".to_string(), "zh".to_string()],
            language_mode: None,
            language_default: None,
            source_langs: Vec::new(),
            target_langs: Vec::new(),
            vendor: None,
            license: "Apache-2.0".to_string(),
            license_url: "https://example.invalid/license".to_string(),
            license_class: crate::LicenseClass::Permissive,
            hf_repo: format!("OpenASR/{id}"),
            hf_revision: revision.to_string(),
            public: true,
            min_cli_version: "0.1.0".to_string(),
            recommended_quant: "q8_0".to_string(),
            pull_recommended: format!("{id}:q8"),
            sort_weight: 0,
            recommended: false,
            prose: None,
            prose_locales: None,
            quants: vec![crate::CatalogQuant {
                quant: "q8_0".to_string(),
                suffix: "q8".to_string(),
                pull: format!("{id}:q8"),
                filename: format!("{id}-q8_0.oasr"),
                url: format!(
                    "https://huggingface.co/OpenASR/{id}/resolve/{revision}/{id}-q8_0.oasr"
                ),
                mirrors: Vec::new(),
                sha256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                    .to_string(),
                size_bytes: 1,
                recommended: true,
                perf: None,
            }],
        }],
    }
}

#[test]
fn missing_config_file_returns_default_config() {
    let temp = tempfile::tempdir().unwrap();
    let config = load_config(temp.path()).unwrap();

    assert_eq!(config.default_model.as_deref(), Some("qwen3-asr-0.6b"));
    assert_eq!(config.default_backend.as_deref(), Some("native"));
    assert_eq!(config.media.ffmpeg_bin, None);
}

#[test]
fn missing_config_file_returns_default_config_document_preferences() {
    let temp = tempfile::tempdir().unwrap();
    let document = load_config_document(temp.path()).unwrap();

    assert_eq!(document.config, OpenAsrConfig::default());
    assert_eq!(document.preferences.version, PREFERENCES_SCHEMA_VERSION);
    assert_eq!(document.preferences.language, None);
    assert!(!document.preferences.diarize);
    assert!(!document.preferences.word_timestamps);
    assert!(!document.preferences.auto_save);
    assert_eq!(document.preferences.hotwords, Vec::<String>::new());
    assert_eq!(document.preferences.theme, AppearanceTheme::System);
    assert_eq!(document.preferences.density, AppearanceDensity::Comfortable);
    assert_eq!(
        document.preferences.dictation_shortcut.as_deref(),
        Some("CommandOrControl+Shift+Space")
    );
    assert_eq!(document.preferences.inference_threads, None);
}

#[test]
fn save_and_load_config_roundtrip() {
    let temp = tempfile::tempdir().unwrap();
    let config = OpenAsrConfig {
        default_model: Some("whisper-small".to_string()),
        default_backend: Some("mock".to_string()),
        media: MediaConfig {
            ffmpeg_bin: Some("/tmp/ffmpeg".to_string()),
        },
        download_source: DownloadSourcePref::Auto,
    };

    save_config(temp.path(), &config).unwrap();
    let loaded = load_config(temp.path()).unwrap();

    assert_eq!(loaded, config);
}

#[test]
fn save_and_load_config_document_roundtrip_preserves_preferences() {
    let temp = tempfile::tempdir().unwrap();
    let document = OpenAsrConfigDocument {
        config: OpenAsrConfig {
            default_model: Some("whisper-small".to_string()),
            default_backend: Some("mock".to_string()),
            media: MediaConfig {
                ffmpeg_bin: Some("/tmp/ffmpeg".to_string()),
            },
            download_source: DownloadSourcePref::Auto,
        },
        preferences: Preferences {
            language: Some("en".to_string()),
            word_timestamps: true,
            auto_save: true,
            launch_at_login: true,
            tray_icon: false,
            output_dir: Some(temp.path().join("transcripts")),
            hotwords: vec!["OpenASR".to_string()],
            hotword_boost: Some(3.5),
            theme: AppearanceTheme::Dark,
            accent_color: Some("#0f766e".to_string()),
            density: AppearanceDensity::Compact,
            push_to_talk: true,
            onboarded: true,
            inference_threads: Some(4),
            execution_target: ExecutionTarget::Cpu,
            history_retention: HistoryRetentionPolicy::Month,
            idle_unload: IdleUnloadPolicy::After10m,
            ..Preferences::default()
        },
    };

    save_config_document(temp.path(), &document).unwrap();
    let loaded = load_config_document(temp.path()).unwrap();

    assert_eq!(loaded, document);
}

#[test]
fn legacy_config_file_defaults_config_document_preferences() {
    let temp = tempfile::tempdir().unwrap();
    std::fs::write(
        config_path(temp.path()),
        r#"{
  "default_model": "whisper-small",
  "default_backend": "mock",
  "media": { "ffmpeg_bin": "/tmp/ffmpeg" }
}
"#,
    )
    .unwrap();

    let loaded = load_config_document(temp.path()).unwrap();

    assert_eq!(
        loaded.config.default_model.as_deref(),
        Some("whisper-small")
    );
    assert_eq!(loaded.preferences, Preferences::default());
}

#[test]
fn save_config_preserves_existing_config_document_preferences() {
    let temp = tempfile::tempdir().unwrap();
    let original = OpenAsrConfigDocument {
        preferences: Preferences {
            language: Some("en".to_string()),
            hotwords: vec!["OpenASR".to_string()],
            inference_threads: Some(2),
            ..Preferences::default()
        },
        ..OpenAsrConfigDocument::default()
    };
    save_config_document(temp.path(), &original).unwrap();

    let updated_config = OpenAsrConfig {
        default_model: Some("whisper-small".to_string()),
        default_backend: Some("mock".to_string()),
        media: MediaConfig::default(),
        download_source: DownloadSourcePref::Auto,
    };
    save_config(temp.path(), &updated_config).unwrap();
    let loaded = load_config_document(temp.path()).unwrap();

    assert_eq!(loaded.config, updated_config);
    assert_eq!(loaded.preferences, original.preferences);
}

#[test]
fn config_document_validation_rejects_bad_preferences() {
    let document = OpenAsrConfigDocument {
        preferences: Preferences {
            hotwords: vec!["OpenASR".to_string(), "openasr".to_string()],
            ..Preferences::default()
        },
        ..OpenAsrConfigDocument::default()
    };

    let error = document.validate(&registry()).unwrap_err().to_string();

    assert!(error.contains("Invalid preference 'hotwords'"));
    assert!(error.contains("duplicate normalized phrases"));
}

#[test]
fn config_document_validation_rejects_unsupported_preferences_version() {
    let document = OpenAsrConfigDocument {
        preferences: Preferences {
            version: PREFERENCES_SCHEMA_VERSION + 1,
            ..Preferences::default()
        },
        ..OpenAsrConfigDocument::default()
    };

    let error = document.validate(&registry()).unwrap_err().to_string();

    assert!(error.contains("Unsupported preferences schema version"));
}

#[test]
fn set_get_unset_supported_keys() {
    let mut config = OpenAsrConfig::default();
    let registry = registry();

    config
        .set(ConfigKey::DefaultModel, "whisper-small", &registry)
        .unwrap();
    config
        .set(ConfigKey::DefaultBackend, "mock", &registry)
        .unwrap();
    config
        .set(ConfigKey::MediaFfmpegBin, "/tmp/ffmpeg", &registry)
        .unwrap();

    assert_eq!(
        config.get(ConfigKey::DefaultModel).as_deref(),
        Some("whisper-small")
    );
    assert_eq!(
        config.get(ConfigKey::DefaultBackend).as_deref(),
        Some("mock")
    );
    assert_eq!(
        config.get(ConfigKey::MediaFfmpegBin).as_deref(),
        Some("/tmp/ffmpeg")
    );

    config.unset(ConfigKey::MediaFfmpegBin);
    assert_eq!(config.get(ConfigKey::MediaFfmpegBin), None);
}

#[test]
fn unknown_key_returns_friendly_error() {
    let error = "missing.key".parse::<ConfigKey>().unwrap_err().to_string();
    assert!(error.contains("Unknown config key 'missing.key'"));
    assert!(error.contains("default_model, default_backend, media.ffmpeg_bin"));
}

#[test]
fn default_backend_rejects_unknown_backend() {
    let mut config = OpenAsrConfig::default();
    let error = config
        .set(ConfigKey::DefaultBackend, "bad-backend", &registry())
        .unwrap_err()
        .to_string();

    assert_eq!(
        error,
        "Unsupported backend 'bad-backend'. Use one of: mock, native."
    );
}

#[test]
fn default_backend_accepts_native() {
    // native is the default backend now and a valid persisted default: it
    // resolves an installed pack by model id (the CLI consent-pulls a missing
    // one), so it no longer has to be passed explicitly.
    let mut config = OpenAsrConfig::default();

    config
        .set(ConfigKey::DefaultBackend, "native", &registry())
        .expect("native is a valid persisted default backend");
    assert_eq!(config.default_backend.as_deref(), Some("native"));
}

#[test]
fn default_model_rejects_unknown_registry_model() {
    let mut config = OpenAsrConfig::default();
    let error = config
        .set(ConfigKey::DefaultModel, "missing-model", &registry())
        .unwrap_err()
        .to_string();

    assert!(error.contains("Unknown model: missing-model"));
    assert!(error.contains("Run `openasr list` to see available models."));
}

#[test]
fn default_model_accepts_variant_tag() {
    let mut config = OpenAsrConfig::default();

    config
        .set(
            ConfigKey::DefaultModel,
            "whisper:candidate",
            &variant_registry(),
        )
        .unwrap();

    assert_eq!(
        config.get(ConfigKey::DefaultModel).as_deref(),
        Some("whisper:candidate")
    );
}

#[test]
fn default_model_with_catalog_preserves_user_reference() {
    let mut config = OpenAsrConfig::default();
    let registry = registry();
    let catalog = catalog_model("qwen3-asr-0.6b", "qwen", &["qwen3", "qwen3-asr"], "0.6b");

    config
        .set_with_catalog(
            ConfigKey::DefaultModel,
            "qwen:q8",
            &registry,
            Some(&catalog),
        )
        .unwrap();

    assert_eq!(
        config.get(ConfigKey::DefaultModel).as_deref(),
        Some("qwen:q8")
    );
    config
        .validate_with_catalog(&registry, Some(&catalog))
        .expect("catalog-validated default must validate with the same catalog");
}

#[test]
fn default_model_with_catalog_preserves_registry_variant_refs() {
    let mut config = OpenAsrConfig::default();
    let registry = variant_registry();
    let catalog = catalog_model("qwen3-asr-0.6b", "qwen", &["qwen3"], "0.6b");

    config
        .set_with_catalog(
            ConfigKey::DefaultModel,
            "whisper:candidate",
            &registry,
            Some(&catalog),
        )
        .unwrap();

    assert_eq!(
        config.get(ConfigKey::DefaultModel).as_deref(),
        Some("whisper:candidate")
    );
}
