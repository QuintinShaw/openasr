//! Narrow source-shape CI gates for model-family trust boundaries.

use std::path::{Path, PathBuf};

fn models_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("src/models")
}

/// Return the source that is compiled for production, excluding the private
/// `mod tests` block at the end of modules that have one.  Several production
/// modules use `#[cfg(test)]` on individual imports/helpers above that block;
/// splitting at the first attribute would therefore hide real production
/// code from this gate.  The exact module marker keeps those imports in the
/// audited prefix while excluding test-only fixtures and architecture IDs.
fn production_source(path: &Path) -> String {
    let source = std::fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("read production source {}: {error}", path.display()));
    source
        .split_once("\n#[cfg(test)]\nmod tests")
        .map(|(production, _)| production)
        .unwrap_or(source.as_str())
        .to_string()
}

fn assert_production_does_not_reference(path: &Path, symbol: &str) {
    let production = production_source(path);
    assert!(
        !production.contains(symbol),
        "production source {} must derive family behavior from the architecture inventory, not reference {symbol}",
        path.display()
    );
}

fn rust_files_below(root: &Path, output: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(root).expect("read model source directory") {
        let path = entry.expect("read model source entry").path();
        if path.is_dir() {
            rust_files_below(&path, output);
        } else if path.extension().and_then(|value| value.to_str()) == Some("rs") {
            output.push(path);
        }
    }
}

#[test]
fn production_model_importers_cannot_call_the_raw_gguf_writer() {
    let root = models_root();
    let mut files = Vec::new();
    rust_files_below(&root, &mut files);
    let mut violations = Vec::new();
    for path in files {
        let relative = path.strip_prefix(&root).unwrap_or(&path);
        if matches!(
            relative.to_str(),
            Some("oasr_metadata.rs" | "pack_verifier.rs" | "family_source_gates.rs")
        ) {
            continue;
        }
        let source = std::fs::read_to_string(&path).expect("read importer source");
        if source.contains("write_gguf_file_v0(") {
            violations.push(relative.display().to_string());
        }
    }
    assert!(
        violations.is_empty(),
        "production model importers must use OasrPackWriter; raw GGUF calls found in: {}",
        violations.join(", ")
    );
}

#[test]
fn byte_preserving_hymt2_repack_stays_inside_the_oasr_transaction() {
    let path = models_root().join("hymt2/package_import.rs");
    let source = std::fs::read_to_string(&path).expect("read Hy-MT2 importer");
    for required in [
        "OasrPackWriter::begin_repack(",
        "transaction.staging_path()",
        ".commit()",
    ] {
        assert!(
            source.contains(required),
            "Hy-MT2's byte-preserving writer must retain transaction step {required}"
        );
    }
    assert!(
        !source.contains("File::create(&request.output_pack"),
        "custom repackers must never write the final output path directly"
    );
}

#[test]
fn removed_family_architecture_apis_cannot_return() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    rust_files_below(&root, &mut files);
    let forbidden = [
        "OpenAsrFamilyIntegrationDescriptor",
        "GgmlAsrRuntimeSourcePreflight",
        "FamilyDefinitionRegistry",
        "ggml_family_registry",
        "_runtime_descriptor_v1",
        "materialize_builtin_executor_component",
        "shared_decode_driver",
        "block_stack: None",
    ];
    let mut violations = Vec::new();
    for path in files {
        if path.ends_with("models/family_source_gates.rs") {
            continue;
        }
        let source = std::fs::read_to_string(&path).expect("read Rust source");
        for symbol in forbidden {
            if source.contains(symbol) {
                violations.push(format!(
                    "{} contains {symbol}",
                    path.strip_prefix(&root).unwrap_or(&path).display()
                ));
            }
        }
    }
    assert!(
        violations.is_empty(),
        "obsolete model-family APIs are forbidden:\n{}",
        violations.join("\n")
    );
}

#[test]
fn retired_family_apis_cannot_return_to_agent_guidance() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("openasr-core lives under <repo>/crates");
    let guidance = [
        repo_root.join("AGENTS.md"),
        repo_root.join("docs/MODEL_ONBOARDING.md"),
        repo_root.join("docs/design/model-onboarding-contract.md"),
        repo_root.join("docs/design/model-family-lifecycle.md"),
    ];
    let forbidden = [
        "OpenAsrFamilyIntegrationDescriptor",
        "GgmlAsrRuntimeSourcePreflight",
        "FamilyDefinitionRegistry",
        "ggml_family_registry",
        "_runtime_descriptor_v1",
        "materialize_builtin_executor_component",
        "shared_decode_driver",
        "block_stack: None",
    ];
    let mut violations = Vec::new();
    for path in guidance {
        let source = std::fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
        for symbol in forbidden {
            if source.contains(symbol) {
                violations.push(format!("{} contains {symbol}", path.display()));
            }
        }
    }
    assert!(
        violations.is_empty(),
        "agent guidance must not teach retired model-family APIs:\n{}",
        violations.join("\n")
    );
}

#[test]
fn shared_runtime_registries_do_not_reintroduce_family_architecture_matches() {
    let root = models_root();
    for relative in [
        "runtime_prepared_registry.rs",
        "runtime_weight_component_registry.rs",
    ] {
        assert_production_does_not_reference(&root.join(relative), "_GGML_ARCHITECTURE_ID");
    }
    assert_production_does_not_reference(
        &root.join("runtime_weight_component_registry.rs"),
        "OpenAsrArchitectureRegistry",
    );
}

#[test]
fn native_backend_production_does_not_match_dolphin_architecture_directly() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/api/backend/native.rs");
    assert_production_does_not_reference(&path, "DOLPHIN_GGML_ARCHITECTURE_ID");
}

#[test]
fn native_transcribe_production_does_not_match_whisper_architecture_directly() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/api/backend/native_transcribe.rs");
    assert_production_does_not_reference(&path, "WHISPER_GGML_ARCHITECTURE_ID");
}
