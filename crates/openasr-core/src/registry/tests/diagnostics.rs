use super::*;

#[test]
fn missing_tag_fails_with_available_tags() {
    let cards = vec![variant_card(
        "whisper-tiny",
        "whisper-tiny",
        "q4_0",
        Some("q4_0"),
    )];

    let error = resolve_registry_model_ref(&cards, "whisper-tiny:q8_0")
        .unwrap_err()
        .to_string();

    assert!(error.contains("does not have variant tag 'q8_0'"));
    assert!(error.contains("Available tags: q4_0"));
}

#[test]
fn legacy_variant_runtime_metadata_is_rejected() {
    let temp = tempfile::tempdir().unwrap();
    let toml = r#"
id = "legacy-runtime-card"
family = "whisper-tiny"
default_variant = "q4_0"
display_name = "Legacy Runtime Card"
backend = "native"
task = "transcription"
languages = ["en", "zh"]
size = "tiny"
recommended_hardware = "CPU"
license = "MIT"
features = ["transcription"]
quality_profile = "planning-only"
source = "Native ASR Core planning metadata"

[variant]
tag = "q4_0"
format = "oasr"
quantization = "q4_0"
role = "default"

[variant.runtime]
id = "legacy-external-runtime"
kind = "external-process"

[download]
type = "none"
enabled = false
files = []
"#;
    fs::write(temp.path().join("legacy-runtime-card.toml"), toml).unwrap();

    let error = load_registry(temp.path()).unwrap_err().to_string();

    assert!(error.contains("Could not parse model card"), "{error}");
    assert!(error.contains("runtime"), "{error}");
}
