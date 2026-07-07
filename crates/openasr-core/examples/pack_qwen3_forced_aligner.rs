//! Publish-only tooling: convert a local Qwen3-ForcedAligner-0.6B safetensors
//! source directory into an `.oasr` runtime pack at a chosen quant.
//!
//! `convert_local_qwen_forced_aligner_source_to_runtime_pack` is deliberately
//! not wired into the `openasr model-pack import` CLI yet (see
//! `models/qwen/forced_aligner_import.rs`): the forced-aligner's NAR decode
//! policy is not dispatchable through the qwen3-asr runtime, so family/CLI
//! wiring is out of scope for this stage. This example exists only to drive
//! the existing converter for the model-pack publishing pipeline
//! (`tooling/publish-model/`), mirroring the `openasr model-pack import
//! <family>` CLI convention for other families.
//!
//! Usage:
//! ```text
//! cargo run --release -p openasr-core --example pack_qwen3_forced_aligner -- \
//!     <source_root> <output_path.oasr> <package_id> <fp16|q8_0|q4_k> \
//!     <source_name> <source_revision> <license_name> <license_source>
//! ```

use std::path::PathBuf;

use openasr_core::models::qwen::{
    Qwen3AsrRuntimeQuantizationMode, Qwen3ForcedAlignerLocalSourceImportRequest,
    convert_local_qwen_forced_aligner_source_to_runtime_pack,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 9 {
        eprintln!(
            "usage: {} <source_root> <output_path.oasr> <package_id> <fp16|q8_0|q4_k> \
             <source_name> <source_revision> <license_name> <license_source>",
            args[0]
        );
        std::process::exit(2);
    }
    let source_root = PathBuf::from(&args[1]);
    let output_root = PathBuf::from(&args[2]);
    let package_id = args[3].clone();
    let quantization = match args[4].as_str() {
        "fp16" => Qwen3AsrRuntimeQuantizationMode::Fp16,
        "q8_0" => Qwen3AsrRuntimeQuantizationMode::Q8_0,
        "q4_k" => Qwen3AsrRuntimeQuantizationMode::Q4_K,
        other => {
            eprintln!("unsupported quantization: {other} (expected fp16, q8_0, or q4_k)");
            std::process::exit(2);
        }
    };
    let source_name = args[5].clone();
    let source_revision = args[6].clone();
    let license_name = args[7].clone();
    let license_source = args[8].clone();

    let request = Qwen3ForcedAlignerLocalSourceImportRequest {
        source_root,
        output_root,
        package_id,
        package_variant: None,
        source_name,
        source_revision,
        license_name,
        license_source,
        quantization,
    };

    match convert_local_qwen_forced_aligner_source_to_runtime_pack(&request) {
        Ok(result) => {
            println!(
                "Imported Qwen3-ForcedAligner local source into runtime pack:\n- output: {}\n- model_id: {}\n- tensor_count: {}",
                result.output_path.display(),
                result.model_id,
                result.tensor_count
            );
        }
        Err(error) => {
            eprintln!("conversion failed: {error}");
            std::process::exit(1);
        }
    }
}
