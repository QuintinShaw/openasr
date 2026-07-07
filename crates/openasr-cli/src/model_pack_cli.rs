use super::*;
use std::collections::BTreeMap;

pub(super) fn model_pack_command(command: ModelPackCommand) -> Result<()> {
    match command {
        ModelPackCommand::Import { command } => import_command(command),
    }
}

fn import_command(command: ImportCommand) -> Result<()> {
    match command {
        ImportCommand::Whisper {
            source_root,
            output_root,
            package_id,
            package_variant,
            model_language,
            source_name,
            source_revision,
            license_name,
            license_source,
            quantization,
        } => import_whisper_local_command(
            &source_root,
            &output_root,
            &package_id,
            package_variant.as_deref(),
            &model_language,
            &source_name,
            &source_revision,
            &license_name,
            &license_source,
            quantization,
        ),
        ImportCommand::Qwen {
            source_root,
            output_root,
            package_id,
            package_variant,
            source_name,
            source_revision,
            license_name,
            license_source,
            quantization,
        } => import_qwen_local_command(
            &source_root,
            &output_root,
            &package_id,
            package_variant.as_deref(),
            &source_name,
            &source_revision,
            &license_name,
            &license_source,
            quantization,
        ),
        ImportCommand::Cohere {
            source_root,
            output_root,
            package_id,
            package_variant,
            source_name,
            source_revision,
            license_name,
            license_source,
            quantization,
        } => import_cohere_local_command(
            &source_root,
            &output_root,
            &package_id,
            package_variant.as_deref(),
            &source_name,
            &source_revision,
            &license_name,
            &license_source,
            quantization,
        ),
        ImportCommand::ParakeetCtc {
            source_root,
            output_root,
            package_id,
            quantization,
        } => {
            import_parakeet_ctc_local_command(&source_root, &output_root, &package_id, quantization)
        }
        ImportCommand::ParakeetTdt {
            source_root,
            output_root,
            package_id,
            quantization,
        } => {
            import_parakeet_tdt_local_command(&source_root, &output_root, &package_id, quantization)
        }
        ImportCommand::Dolphin {
            source_root,
            output_root,
            package_id,
            quantization,
            language_scheme,
        } => import_dolphin_local_command(
            &source_root,
            &output_root,
            &package_id,
            quantization,
            language_scheme,
        ),
        ImportCommand::Sensevoice {
            source_root,
            output_root,
            package_id,
            quantization,
        } => import_sensevoice_local_command(&source_root, &output_root, &package_id, quantization),
        ImportCommand::XasrZipformer {
            source_root,
            output_root,
            package_id,
            quantization,
        } => import_xasr_zipformer_local_command(
            &source_root,
            &output_root,
            &package_id,
            quantization,
        ),
        ImportCommand::Hymt2Gguf {
            source_gguf,
            output_pack,
            package_id,
            license_file,
            notice_file,
        } => import_hymt2_gguf_local_command(
            &source_gguf,
            &output_pack,
            &package_id,
            &license_file,
            &notice_file,
        ),
        ImportCommand::Wav2Vec2Ctc {
            source_root,
            output_root,
            package_id,
            quantization,
        } => {
            import_wav2vec2_ctc_local_command(&source_root, &output_root, &package_id, quantization)
        }
        ImportCommand::Moonshine {
            source_root,
            output_root,
            package_id,
            package_variant,
            source_name,
            source_revision,
            license_name,
            license_source,
            quantization,
        } => import_moonshine_local_command(
            &source_root,
            &output_root,
            &package_id,
            package_variant.as_deref(),
            &source_name,
            &source_revision,
            &license_name,
            &license_source,
            quantization,
        ),
        ImportCommand::Wespeaker {
            source_safetensors,
            output_root,
            package_id,
            source_name,
            source_revision,
            license_name,
            license_source,
            quantization,
        } => import_wespeaker_local_command(
            &source_safetensors,
            &output_root,
            &package_id,
            &source_name,
            &source_revision,
            &license_name,
            &license_source,
            quantization,
        ),
        ImportCommand::Pyannote {
            source_safetensors,
            output_root,
            package_id,
        } => import_pyannote_local_command(&source_safetensors, &output_root, &package_id),
    }
}

#[allow(clippy::too_many_arguments)]
fn import_wespeaker_local_command(
    source_safetensors: &Path,
    output_root: &Path,
    package_id: &str,
    source_name: &str,
    source_revision: &str,
    license_name: &str,
    license_source: &str,
    quantization: ImportWeSpeakerQuantization,
) -> Result<()> {
    let request = openasr_core::WeSpeakerImportRequest {
        source_safetensors: source_safetensors.to_path_buf(),
        output_root: output_root.to_path_buf(),
        model_id: package_id.to_string(),
        source_name: source_name.to_string(),
        source_revision: source_revision.to_string(),
        license_name: license_name.to_string(),
        license_source: license_source.to_string(),
        quantization: match quantization {
            ImportWeSpeakerQuantization::F32 => openasr_core::WeSpeakerRuntimeQuantizationMode::F32,
        },
    };

    ensure_ggml_package_output_suffix(output_root)?;
    let result = openasr_core::convert_local_wespeaker_source_to_runtime_pack(&request)
        .map_err(anyhow::Error::new)?;
    println!(
        "Imported WeSpeaker ResNet34 local source into diarization runtime pack:\n- source: {}\n- output: {}\n- tensor_count: {}\n- quantization: {:?}\n- license: {}",
        source_safetensors.display(),
        result.output_path.display(),
        result.tensor_count,
        quantization,
        license_name
    );
    Ok(())
}

fn import_pyannote_local_command(
    source_safetensors: &Path,
    output_root: &Path,
    package_id: &str,
) -> Result<()> {
    let request = openasr_core::PyannoteImportRequest {
        source_safetensors: source_safetensors.to_path_buf(),
        output_root: output_root.to_path_buf(),
        model_id: package_id.to_string(),
    };

    ensure_ggml_package_output_suffix(output_root)?;
    let result = openasr_core::convert_local_pyannote_source_to_runtime_pack(&request)
        .map_err(anyhow::Error::new)?;
    println!(
        "Imported pyannote-seg local source into diarization runtime pack:\n- source: {}\n- output: {}\n- tensor_count: {}",
        source_safetensors.display(),
        result.output_path.display(),
        result.tensor_count
    );
    Ok(())
}

fn ensure_ggml_package_output_suffix(output_root: &Path) -> Result<()> {
    if path_has_ggml_package_suffix(output_root) {
        return Ok(());
    }
    bail!("output path must end with .oasr (OpenASR native runtime pack).");
}

fn import_sensevoice_local_command(
    source_root: &Path,
    output_root: &Path,
    package_id: &str,
    quantization: ImportSensevoiceQuantization,
) -> Result<()> {
    let request = openasr_core::SenseVoiceImportRequest {
        source_root: source_root.to_path_buf(),
        output_root: output_root.to_path_buf(),
        model_id: package_id.to_string(),
        quantization: match quantization {
            ImportSensevoiceQuantization::Fp16 => openasr_core::SenseVoiceQuantizationMode::Fp16,
            ImportSensevoiceQuantization::Q8_0 => openasr_core::SenseVoiceQuantizationMode::Q8_0,
            ImportSensevoiceQuantization::Q4_K => openasr_core::SenseVoiceQuantizationMode::Q4_K,
        },
    };

    ensure_ggml_package_output_suffix(output_root)?;
    let result = openasr_core::convert_local_sensevoice_source_to_runtime_pack(&request)
        .map_err(anyhow::Error::new)?;
    println!(
        "Imported SenseVoice local source into runtime pack:\n- source: {}\n- output: {}\n- tensor_count: {}\n- vocab_size: {}",
        source_root.display(),
        result.output_path.display(),
        result.tensor_count,
        result.vocab_size
    );
    Ok(())
}

fn import_dolphin_local_command(
    source_root: &Path,
    output_root: &Path,
    package_id: &str,
    quantization: ImportDolphinQuantization,
    language_scheme: ImportDolphinLanguageScheme,
) -> Result<()> {
    let request = openasr_core::DolphinImportRequest {
        safetensors_path: source_root.join("full.safetensors"),
        units_path: source_root.join("units.txt"),
        output_path: output_root.to_path_buf(),
        model_id: package_id.to_string(),
        quantization: match quantization {
            ImportDolphinQuantization::Fp16 => openasr_core::DolphinQuantizationMode::Fp16,
            ImportDolphinQuantization::Q8_0 => openasr_core::DolphinQuantizationMode::Q8_0,
            ImportDolphinQuantization::Q4_K => openasr_core::DolphinQuantizationMode::Q4_K,
        },
        language_scheme: match language_scheme {
            ImportDolphinLanguageScheme::CnDialect => {
                openasr_core::DolphinLanguageScheme::CnDialect
            }
            ImportDolphinLanguageScheme::Multilingual => {
                openasr_core::DolphinLanguageScheme::Multilingual
            }
        },
    };

    ensure_ggml_package_output_suffix(output_root)?;
    let result = openasr_core::convert_local_dolphin_wenet_source_to_runtime_pack(&request)
        .map_err(anyhow::Error::new)?;
    println!(
        "Imported Dolphin local source into runtime pack:\n- source: {}\n- output: {}\n- tensor_count: {}\n- vocab_size: {}",
        source_root.display(),
        result.output_path.display(),
        result.tensor_count,
        result.vocab_size
    );
    Ok(())
}

fn import_parakeet_tdt_local_command(
    source_root: &Path,
    output_root: &Path,
    package_id: &str,
    quantization: ImportParakeetQuantization,
) -> Result<()> {
    let request = openasr_core::ParakeetTdtImportRequest {
        source_root: source_root.to_path_buf(),
        output_root: output_root.to_path_buf(),
        model_id: package_id.to_string(),
        quantization: match quantization {
            ImportParakeetQuantization::Fp16 => openasr_core::ParakeetTdtQuantizationMode::Fp16,
            ImportParakeetQuantization::Q8_0 => openasr_core::ParakeetTdtQuantizationMode::Q8_0,
            ImportParakeetQuantization::Q4_K => openasr_core::ParakeetTdtQuantizationMode::Q4_K,
        },
    };

    ensure_ggml_package_output_suffix(output_root)?;
    let result = openasr_core::convert_local_parakeet_tdt_source_to_runtime_pack(&request)
        .map_err(anyhow::Error::new)?;
    println!(
        "Imported Parakeet-TDT local source into runtime pack:\n- source: {}\n- output: {}\n- tensor_count: {}\n- blank_token_id: {}",
        source_root.display(),
        result.output_path.display(),
        result.tensor_count,
        result.blank_token_id
    );
    Ok(())
}

fn import_parakeet_ctc_local_command(
    source_root: &Path,
    output_root: &Path,
    package_id: &str,
    quantization: ImportParakeetQuantization,
) -> Result<()> {
    let request = openasr_core::ParakeetCtcImportRequest {
        source_root: source_root.to_path_buf(),
        output_root: output_root.to_path_buf(),
        model_id: package_id.to_string(),
        quantization: match quantization {
            ImportParakeetQuantization::Fp16 => openasr_core::ParakeetCtcQuantizationMode::Fp16,
            ImportParakeetQuantization::Q8_0 => openasr_core::ParakeetCtcQuantizationMode::Q8_0,
            ImportParakeetQuantization::Q4_K => openasr_core::ParakeetCtcQuantizationMode::Q4_K,
        },
    };

    ensure_ggml_package_output_suffix(output_root)?;
    let result = openasr_core::convert_local_parakeet_ctc_source_to_runtime_pack(&request)
        .map_err(anyhow::Error::new)?;
    println!(
        "Imported Parakeet-CTC local source into runtime pack:\n- source: {}\n- output: {}\n- tensor_count: {}\n- blank_token_id: {}",
        source_root.display(),
        result.output_path.display(),
        result.tensor_count,
        result.blank_token_id
    );
    Ok(())
}

fn import_xasr_zipformer_local_command(
    source_root: &Path,
    output_root: &Path,
    package_id: &str,
    quantization: ImportXasrZipformerQuantization,
) -> Result<()> {
    let request = openasr_core::XasrZipformerImportRequest {
        source_root: source_root.to_path_buf(),
        output_root: output_root.to_path_buf(),
        model_id: package_id.to_string(),
        quantization: match quantization {
            ImportXasrZipformerQuantization::Fp16 => {
                openasr_core::XasrZipformerQuantizationMode::Fp16
            }
            ImportXasrZipformerQuantization::Q8_0 => {
                openasr_core::XasrZipformerQuantizationMode::Q8_0
            }
            ImportXasrZipformerQuantization::Q4_K => {
                openasr_core::XasrZipformerQuantizationMode::Q4_K
            }
        },
    };

    ensure_ggml_package_output_suffix(output_root)?;
    let result = openasr_core::convert_local_xasr_zipformer_source_to_runtime_pack(&request)
        .map_err(anyhow::Error::new)?;
    println!(
        "Imported X-ASR Zipformer local source into runtime pack:\n- source: {}\n- output: {}\n- tensor_count: {}\n- blank_token_id: {}",
        source_root.display(),
        result.output_path.display(),
        result.tensor_count,
        result.blank_token_id
    );
    Ok(())
}

fn import_hymt2_gguf_local_command(
    source_gguf: &Path,
    output_pack: &Path,
    package_id: &str,
    license_file: &Path,
    notice_file: &Path,
) -> Result<()> {
    ensure_ggml_package_output_suffix(output_pack)?;
    let license_text = std::fs::read_to_string(license_file)
        .with_context(|| format!("read license file {}", license_file.display()))?;
    let notice_text = std::fs::read_to_string(notice_file)
        .with_context(|| format!("read notice file {}", notice_file.display()))?;
    let request = openasr_core::Hymt2ImportRequest {
        source_gguf: source_gguf.to_path_buf(),
        output_pack: output_pack.to_path_buf(),
        model_id: package_id.to_string(),
        quantization: "q4_k_m".to_string(),
        license_text,
        notice_text,
        expected_source_sha256: openasr_core::HYMT2_PINNED_SOURCE_GGUF_SHA256.to_string(),
    };

    let result =
        openasr_core::import_hymt2_gguf_to_runtime_pack(&request).map_err(anyhow::Error::new)?;
    // Fail-closed: the written pack must load through the Hy-MT2 runtime probe.
    let metadata =
        openasr_core::Hymt2Runtime::probe_path(&result.output_path).map_err(anyhow::Error::new)?;
    println!(
        "Imported pinned Hy-MT2 GGUF into translation runtime pack:\n- source: {}\n- source_sha256: {}\n- output: {}\n- pack_sha256: {}\n- size_bytes: {}\n- tensor_count: {}\n- spliced_metadata_entries: {}\n- layers: {}\n- vocab_size: {}",
        source_gguf.display(),
        result.source_sha256,
        result.output_path.display(),
        result.pack_sha256,
        result.pack_size_bytes,
        result.tensor_count,
        result.appended_metadata_entries,
        metadata.layers,
        metadata.vocab_size,
    );
    Ok(())
}

fn import_wav2vec2_ctc_local_command(
    source_root: &Path,
    output_root: &Path,
    package_id: &str,
    quantization: ImportWav2Vec2Quantization,
) -> Result<()> {
    let request = openasr_core::Wav2Vec2CtcImportRequest {
        source_root: source_root.to_path_buf(),
        output_root: output_root.to_path_buf(),
        model_id: package_id.to_string(),
        quantization: match quantization {
            ImportWav2Vec2Quantization::Fp16 => openasr_core::Wav2Vec2CtcQuantizationMode::Fp16,
            ImportWav2Vec2Quantization::Q8_0 => openasr_core::Wav2Vec2CtcQuantizationMode::Q8_0,
            ImportWav2Vec2Quantization::Q4_K => openasr_core::Wav2Vec2CtcQuantizationMode::Q4_K,
        },
    };

    ensure_ggml_package_output_suffix(output_root)?;
    let result = openasr_core::convert_local_wav2vec2_ctc_source_to_runtime_pack(&request)
        .map_err(anyhow::Error::new)?;
    println!(
        "Imported wav2vec2-CTC local source into runtime pack:\n- source: {}\n- output: {}\n- tensor_count: {}\n- blank_token_id: {}",
        source_root.display(),
        result.output_path.display(),
        result.tensor_count,
        result.blank_token_id
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn import_moonshine_local_command(
    source_root: &Path,
    output_root: &Path,
    package_id: &str,
    package_variant: Option<&str>,
    source_name: &str,
    source_revision: &str,
    license_name: &str,
    license_source: &str,
    quantization: ImportMoonshineQuantization,
) -> Result<()> {
    let request = openasr_core::MoonshineLocalSourceImportRequest {
        source_root: source_root.to_path_buf(),
        output_root: output_root.to_path_buf(),
        package_id: package_id.to_string(),
        package_variant: package_variant.map(ToOwned::to_owned),
        source_name: source_name.to_string(),
        source_revision: source_revision.to_string(),
        license_name: license_name.to_string(),
        license_source: license_source.to_string(),
        quantization: match quantization {
            ImportMoonshineQuantization::Fp16 => {
                openasr_core::MoonshineRuntimeQuantizationMode::Fp16
            }
            ImportMoonshineQuantization::Q8_0 => {
                openasr_core::MoonshineRuntimeQuantizationMode::Q8_0
            }
            ImportMoonshineQuantization::Q4_K => {
                openasr_core::MoonshineRuntimeQuantizationMode::Q4_K
            }
        },
    };

    ensure_ggml_package_output_suffix(output_root)?;
    let result = openasr_core::convert_local_moonshine_source_to_runtime_pack(&request)
        .map_err(anyhow::Error::new)?;
    println!(
        "Imported Moonshine local source into runtime pack:\n- source: {}\n- output: {}\n- model.id: {}\n- tensor_count: {}",
        source_root.display(),
        result.output_path.display(),
        result.model_id,
        result.tensor_count
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn import_whisper_local_command(
    source_root: &Path,
    output_root: &Path,
    package_id: &str,
    package_variant: Option<&str>,
    model_language: &str,
    source_name: &str,
    source_revision: &str,
    license_name: &str,
    license_source: &str,
    quantization: ImportWhisperQuantization,
) -> Result<()> {
    let request = WhisperLocalSourceImportRequest {
        source_root: source_root.to_path_buf(),
        output_root: output_root.to_path_buf(),
        package_id: package_id.to_string(),
        package_variant: package_variant.map(ToOwned::to_owned),
        model_language: model_language.to_string(),
        source_name: source_name.to_string(),
        source_revision: source_revision.to_string(),
        license_name: license_name.to_string(),
        license_source: license_source.to_string(),
        quantization: match quantization {
            ImportWhisperQuantization::Fp16 => openasr_core::WhisperRuntimeQuantizationMode::Fp16,
            ImportWhisperQuantization::Q8_0 => openasr_core::WhisperRuntimeQuantizationMode::Q8_0,
            ImportWhisperQuantization::Q4_K => openasr_core::WhisperRuntimeQuantizationMode::Q4_K,
        },
    };

    ensure_ggml_package_output_suffix(output_root)?;
    let result =
        convert_local_whisper_hf_source_to_runtime_pack(&request).map_err(anyhow::Error::new)?;
    println!(
        "Imported Whisper local source into runtime pack:\n- source: {}\n- output: {}\n- model.id: {}\n- tensor_count: {}",
        source_root.display(),
        result.output_path.display(),
        result.model_id,
        result.tensor_count
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn import_qwen_local_command(
    source_root: &Path,
    output_root: &Path,
    package_id: &str,
    package_variant: Option<&str>,
    source_name: &str,
    source_revision: &str,
    license_name: &str,
    license_source: &str,
    quantization: ImportQwen3AsrQuantization,
) -> Result<()> {
    let request = Qwen3AsrLocalSourceImportRequest {
        source_root: source_root.to_path_buf(),
        output_root: output_root.to_path_buf(),
        package_id: package_id.to_string(),
        package_variant: package_variant.map(ToOwned::to_owned),
        source_name: source_name.to_string(),
        source_revision: source_revision.to_string(),
        license_name: license_name.to_string(),
        license_source: license_source.to_string(),
        quantization: match quantization {
            ImportQwen3AsrQuantization::Fp16 => openasr_core::Qwen3AsrRuntimeQuantizationMode::Fp16,
            ImportQwen3AsrQuantization::Q8_0 => openasr_core::Qwen3AsrRuntimeQuantizationMode::Q8_0,
            ImportQwen3AsrQuantization::Q3_K => openasr_core::Qwen3AsrRuntimeQuantizationMode::Q3_K,
            ImportQwen3AsrQuantization::Q4_K => openasr_core::Qwen3AsrRuntimeQuantizationMode::Q4_K,
        },
    };

    ensure_ggml_package_output_suffix(output_root)?;
    let result = convert_local_qwen_source_to_runtime_pack(&request).map_err(anyhow::Error::new)?;
    println!(
        "Imported Qwen local source into runtime pack:\n- source: {}\n- output: {}\n- model.id: {}\n- tensor_count: {}",
        source_root.display(),
        result.output_path.display(),
        result.model_id,
        result.tensor_count
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn import_cohere_local_command(
    source_root: &Path,
    output_root: &Path,
    package_id: &str,
    package_variant: Option<&str>,
    source_name: &str,
    source_revision: &str,
    license_name: &str,
    license_source: &str,
    quantization: ImportCohereQuantization,
) -> Result<()> {
    let request = CohereLocalSourceImportRequest {
        source_root: source_root.to_path_buf(),
        output_root: output_root.to_path_buf(),
        package_id: package_id.to_string(),
        package_variant: package_variant.map(ToOwned::to_owned),
        source_name: source_name.to_string(),
        source_revision: source_revision.to_string(),
        license_name: license_name.to_string(),
        license_source: license_source.to_string(),
        quantization: match quantization {
            ImportCohereQuantization::Fp16 => openasr_core::CohereRuntimeQuantizationMode::Fp16,
            ImportCohereQuantization::Q8_0 => openasr_core::CohereRuntimeQuantizationMode::Q8_0,
            ImportCohereQuantization::Q4_K => openasr_core::CohereRuntimeQuantizationMode::Q4_K,
        },
    };

    ensure_ggml_package_output_suffix(output_root)?;
    let result =
        convert_local_cohere_source_to_runtime_pack(&request).map_err(anyhow::Error::new)?;
    println!(
        "Imported Cohere local source into runtime pack:\n- source: {}\n- output: {}\n- model.id: {}\n- tensor_count: {}",
        source_root.display(),
        result.output_path.display(),
        result.model_id,
        result.tensor_count
    );
    Ok(())
}

pub(super) fn validate_model_pack_path_command(path: &Path) -> Result<()> {
    let package_path = validate_local_ggml_package_cli_path(path)?;
    let probe = openasr_core::probe_ggml_package_path(&package_path).map_err(anyhow::Error::new)?;
    if probe.format != openasr_core::GgmlPackageFormat::GgufCompatible {
        bail!(
            "Model package '{}' uses a reserved non-GGUF container magic and is not accepted by ggml runtime.",
            package_path.display()
        );
    }
    println!(
        "Validated local ggml model package {}.",
        package_path.display()
    );
    println!("No downloads or inference were performed.");
    Ok(())
}

pub(super) fn inspect_model_pack_path_command(path: &Path) -> Result<()> {
    let package_path = validate_local_ggml_package_cli_path(path)?;
    let probe = openasr_core::probe_ggml_package_path(&package_path).map_err(anyhow::Error::new)?;
    let mut rendered = render_ggml_package_inspection(&package_path, &probe);
    if probe.format == openasr_core::GgmlPackageFormat::GgufCompatible {
        rendered.push_str(&render_openasr_runtime_metadata_summary(&package_path));
        rendered.push_str(&render_native_transcription_capability_summary(
            &package_path,
        ));
        rendered.push_str(&render_native_runtime_capability_summary(&package_path));
        rendered.push_str(&render_gguf_tensor_index_summary(&package_path));
    }
    print!("{rendered}");
    Ok(())
}

fn render_ggml_package_inspection(path: &Path, probe: &openasr_core::GgmlPackageProbe) -> String {
    let format = match probe.format {
        openasr_core::GgmlPackageFormat::GgufCompatible => ".oasr (OpenASR native pack)",
        openasr_core::GgmlPackageFormat::UnsupportedOpenAsrContainerReserved => {
            "reserved-openasr-container"
        }
    };
    let extension = match probe.extension_hint {
        openasr_core::GgmlPackageExtensionHint::Oasr => ".oasr",
        openasr_core::GgmlPackageExtensionHint::Gguf => ".gguf",
        openasr_core::GgmlPackageExtensionHint::OtherOrMissing => "<other-or-missing>",
    };

    let mut output = String::new();
    output.push_str(&format!("Path: {}\n", path.display()));
    output.push_str(&format!("Format: {format}\n"));
    output.push_str(&format!("Extension hint: {extension}\n"));
    let identity = openasr_core::probe_ggml_package_model_identity(path);
    if let Some(model_id) = identity.model_id {
        let source_key = identity
            .source_key
            .unwrap_or_else(|| "unknown-metadata-key".to_string());
        output.push_str(&format!("Model identity: {model_id} ({source_key})\n"));
    } else {
        output.push_str("Model identity: <missing>\n");
        if let Some(error) = identity.metadata_read_error {
            output.push_str(&format!("Model identity warning: {error}\n"));
        }
    }
    output.push_str("Runtime contract: ggml package probe only (no manifest decode)\n");
    output.push_str("Warnings:");
    if probe.format == openasr_core::GgmlPackageFormat::GgufCompatible {
        output.push_str(" none\n");
    } else {
        output.push_str(
            "\n- reserved OASR container magic is not accepted on the ggml runtime path\n",
        );
    }
    output
}

fn render_openasr_runtime_metadata_summary(path: &Path) -> String {
    let mut output = String::new();
    output.push_str("OpenASR runtime metadata:\n");
    match openasr_core::read_gguf_metadata(path) {
        Ok(metadata) => output.push_str(&render_openasr_runtime_metadata_values(metadata.values())),
        Err(error) => output.push_str(&format!("- unavailable: {error}\n")),
    }
    output
}

fn render_openasr_runtime_metadata_values(
    values: &BTreeMap<String, openasr_core::GgufMetadataValue>,
) -> String {
    let keys = [
        openasr_core::models::oasr_metadata::OASR_METADATA_KEY_PACKAGE_VERSION,
        openasr_core::models::oasr_metadata::OASR_METADATA_KEY_MODEL_FAMILY,
        openasr_core::models::oasr_metadata::OASR_METADATA_KEY_MODEL_ARCHITECTURE,
        openasr_core::models::oasr_metadata::OASR_METADATA_KEY_AUDIO_FRONTEND,
        openasr_core::models::oasr_metadata::OASR_METADATA_KEY_DECODE_POLICY,
        openasr_core::models::oasr_metadata::OASR_METADATA_KEY_FEATURE_DIARIZATION,
    ];
    keys.iter()
        .map(|key| {
            let value = values
                .get(*key)
                .map(render_gguf_metadata_value)
                .unwrap_or_else(|| "<missing>".to_string());
            format!("- {key}: {value}\n")
        })
        .collect()
}

fn render_gguf_metadata_value(value: &openasr_core::GgufMetadataValue) -> String {
    match value {
        openasr_core::GgufMetadataValue::String(value) => value.clone(),
        openasr_core::GgufMetadataValue::U32(value) => value.to_string(),
        openasr_core::GgufMetadataValue::U64(value) => value.to_string(),
        openasr_core::GgufMetadataValue::Bool(value) => value.to_string(),
        openasr_core::GgufMetadataValue::F32(value) => value.to_string(),
        openasr_core::GgufMetadataValue::StringArray(values) => {
            format!("[{}]", values.join(", "))
        }
        openasr_core::GgufMetadataValue::U32Array(values) => {
            let joined = values
                .iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join(", ");
            format!("[{joined}]")
        }
    }
}

fn render_native_runtime_capability_summary(path: &Path) -> String {
    render_realtime_capabilities(openasr_core::native_runtime_realtime_capabilities_for_path(
        path,
    ))
}

fn render_native_transcription_capability_summary(path: &Path) -> String {
    render_transcription_capabilities(
        openasr_core::native_runtime_transcription_capabilities_for_path(path),
    )
}

fn render_transcription_capabilities(
    capabilities: openasr_core::api::backend::TranscriptionBackendCapabilities,
) -> String {
    let mut output = String::new();
    output.push_str("Native transcription capability:\n");
    output.push_str(&format!("- backend: {}\n", capabilities.backend));
    output.push_str(&render_feature_capability(
        "segment_timestamps",
        capabilities.segment_timestamps,
    ));
    output.push_str(&render_feature_capability(
        "word_timestamps",
        capabilities.word_timestamps,
    ));
    output.push_str(&render_feature_capability(
        "diarization",
        capabilities.diarization,
    ));
    output.push_str(&render_feature_capability(
        "phrase_bias",
        capabilities.phrase_bias,
    ));
    output.push_str(&render_feature_capability(
        "inference_threads",
        capabilities.inference_threads,
    ));
    output.push_str(&render_language_capability(capabilities.language));
    output
}

fn render_language_capability(
    capability: openasr_core::api::backend::LanguageCapability,
) -> String {
    let mut line = format!(
        "- language: mode={} auto={} specify={}",
        capability.mode, capability.auto_supported, capability.specify_supported
    );
    if let Some(default) = capability.default_language {
        line.push_str(&format!(" default={default}"));
    }
    if !capability.fixed_languages.is_empty() {
        line.push_str(&format!(
            " languages={}",
            capability.fixed_languages.join(",")
        ));
    }
    if let Some(reason) = capability.reason {
        line.push_str(&format!(" ({reason})"));
    }
    line.push('\n');
    line
}

fn render_feature_capability(
    name: &str,
    capability: openasr_core::api::backend::BackendFeatureCapability,
) -> String {
    let behavior = match capability.behavior {
        openasr_core::api::backend::BackendCapabilityBehavior::Supported => "supported",
        openasr_core::api::backend::BackendCapabilityBehavior::RejectRequest => "reject_request",
        openasr_core::api::backend::BackendCapabilityBehavior::MetadataOnly => "metadata_only",
    };
    let mut output = format!(
        "- {name}: supported={}, behavior={behavior}",
        capability.supported
    );
    if let Some(reason) = capability.reason {
        output.push_str(&format!(", reason={reason}"));
    }
    output.push('\n');
    output
}

fn render_realtime_capabilities(capabilities: openasr_core::RealtimeBackendCapabilities) -> String {
    let mode = match capabilities.mode {
        openasr_core::RealtimeBackendMode::Unsupported => "unsupported",
        openasr_core::RealtimeBackendMode::FilePerUtteranceFallback => {
            "file_per_utterance_fallback"
        }
        openasr_core::RealtimeBackendMode::TrueStreaming => "true_streaming",
    };
    let mut output = String::new();
    output.push_str("Native realtime capability:\n");
    output.push_str(&format!("- mode: {mode}\n"));
    output.push_str(&format!(
        "- supports_realtime_sessions: {}\n",
        capabilities.supports_realtime_sessions
    ));
    output.push_str(&format!(
        "- supports_partial_results: {}\n",
        capabilities.supports_partial_results
    ));
    output.push_str(&format!(
        "- is_true_streaming: {}\n",
        capabilities.is_true_streaming
    ));
    output.push_str(&format!(
        "- requires_vad_utterance_boundaries: {}\n",
        capabilities.requires_vad_utterance_boundaries
    ));
    output
}

fn render_gguf_tensor_index_summary(path: &Path) -> String {
    let mut output = String::new();
    output.push_str("Tensor index:\n");
    let index = match openasr_core::read_gguf_tensor_index(path) {
        Ok(index) => index,
        Err(error) => {
            output.push_str(&format!("- unavailable: {error}\n"));
            return output;
        }
    };

    let mut per_type: BTreeMap<String, (usize, u128)> = BTreeMap::new();
    let mut total_bytes: u128 = 0;
    for tensor in index.tensors() {
        total_bytes = total_bytes.saturating_add(u128::from(tensor.size_bytes));
        let entry = per_type
            .entry(tensor.type_name.clone())
            .or_insert((0_usize, 0_u128));
        entry.0 = entry.0.saturating_add(1);
        entry.1 = entry.1.saturating_add(u128::from(tensor.size_bytes));
    }

    output.push_str(&format!("- tensors: {}\n", index.tensors().len()));
    output.push_str(&format!("- payload_bytes_total: {total_bytes}\n"));
    output.push_str("- tensor_types:\n");
    for (tensor_type, (count, bytes)) in per_type {
        output.push_str(&format!(
            "  - {tensor_type}: count={count}, bytes={bytes}\n"
        ));
    }
    output
}

pub(super) fn validate_local_ggml_package_cli_path(path: &Path) -> Result<PathBuf> {
    let rendered = path.as_os_str().to_string_lossy();
    if !path.exists() {
        if looks_like_remote_model_pack_path(&rendered) {
            bail!(
                "Model package path must be a local filesystem path; remote URLs are not supported: {rendered}"
            );
        }
        bail!("Model package path '{rendered}' does not exist.");
    }

    let metadata = fs::metadata(path)
        .with_context(|| format!("Could not inspect model package path '{rendered}'"))?;
    if !metadata.is_file() {
        bail!("Model package path '{rendered}' must be a local .oasr file.");
    }

    let canonical = fs::canonicalize(path).with_context(|| {
        format!(
            "Could not resolve local model package path '{}' for suffix validation",
            path.display()
        )
    })?;
    if !path_has_ggml_package_suffix(&canonical) {
        bail!("Model package path '{rendered}' must end with .oasr.");
    }

    Ok(path.to_path_buf())
}

pub(super) fn looks_like_remote_model_pack_path(value: &str) -> bool {
    let Some((scheme, _)) = value.split_once("://") else {
        return false;
    };
    !scheme.is_empty()
        && scheme.chars().all(|character| {
            character.is_ascii_alphanumeric() || matches!(character, '+' | '-' | '.')
        })
}

pub(super) fn path_has_ggml_package_suffix(path: &Path) -> bool {
    // .oasr is OpenASR's sole native runtime-pack format. The legacy .gguf
    // extension is no longer accepted (the on-disk container remains gguf-
    // structured internally, but it is presented and supported only as .oasr).
    // Delegates to core so the CLI and the library converters share one
    // definition of the user-facing pack-extension contract.
    openasr_core::has_openasr_runtime_pack_extension(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::{OsStr, OsString};

    struct EnvVarRestore {
        name: &'static str,
        previous: Option<OsString>,
    }

    impl EnvVarRestore {
        fn set(name: &'static str, value: &OsStr) -> Self {
            let previous = std::env::var_os(name);
            unsafe { std::env::set_var(name, value) };
            Self { name, previous }
        }

        fn remove(name: &'static str) -> Self {
            let previous = std::env::var_os(name);
            unsafe { std::env::remove_var(name) };
            Self { name, previous }
        }
    }

    impl Drop for EnvVarRestore {
        fn drop(&mut self) {
            match self.previous.take() {
                Some(value) => unsafe { std::env::set_var(self.name, value) },
                None => unsafe { std::env::remove_var(self.name) },
            }
        }
    }

    #[test]
    fn runtime_metadata_summary_renders_streaming_feature_gate() {
        let mut values = BTreeMap::new();
        values.insert(
            openasr_core::models::oasr_metadata::OASR_METADATA_KEY_PACKAGE_VERSION.to_string(),
            openasr_core::GgufMetadataValue::String(
                openasr_core::models::oasr_metadata::OASR_PACKAGE_VERSION_V1.to_string(),
            ),
        );
        values.insert(
            openasr_core::models::oasr_metadata::OASR_METADATA_KEY_MODEL_FAMILY.to_string(),
            openasr_core::GgufMetadataValue::String(
                openasr_core::QWEN3_ASR_MODEL_FAMILY.to_string(),
            ),
        );
        let rendered = render_openasr_runtime_metadata_values(&values);

        assert!(rendered.contains("- openasr.package.version: 1"));
        assert!(rendered.contains("- openasr.model.family: qwen3-asr"));
        assert!(rendered.contains("- openasr.features.diarization: <missing>"));
    }

    #[test]
    fn realtime_capability_summary_renders_true_streaming_gate() {
        let rendered = render_realtime_capabilities(
            openasr_core::RealtimeBackendCapabilities::true_streaming_local(),
        );

        assert!(rendered.contains("- mode: true_streaming"));
        assert!(rendered.contains("- supports_realtime_sessions: true"));
        assert!(rendered.contains("- supports_partial_results: true"));
        assert!(rendered.contains("- is_true_streaming: true"));
        assert!(rendered.contains("- requires_vad_utterance_boundaries: false"));
    }

    #[test]
    fn transcription_capability_summary_renders_pack_feature_gates() {
        let mut capabilities =
            openasr_core::api::backend::TranscriptionBackendCapabilities::for_backend_kind(
                openasr_core::BackendKind::Native,
            );
        capabilities.diarization =
            openasr_core::api::backend::BackendFeatureCapability::supported();

        let rendered = render_transcription_capabilities(capabilities);

        assert!(rendered.contains("Native transcription capability:"));
        assert!(rendered.contains("- backend: native"));
        assert!(rendered.contains("- segment_timestamps: supported=true, behavior=supported"));
        assert!(rendered.contains("- word_timestamps: supported=true, behavior=supported"));
        assert!(rendered.contains("- diarization: supported=true, behavior=supported"));
        assert!(rendered.contains("- phrase_bias: supported=true, behavior=supported"));
        assert!(rendered.contains("- inference_threads: supported=true, behavior=supported"));
    }

    #[test]
    fn transcription_capability_summary_derives_declared_cohere_diarization_pack() {
        let temp = tempfile::tempdir().unwrap();
        let runtime_path = temp.path().join("cohere-diarize-runtime.oasr");
        let spec = openasr_core::testing::TinyGgufFixtureSpec::cohere_oasr_v1_runtime_ready(
            "cohere-diarize-runtime-fixture",
        )
        .with_metadata(
            openasr_core::models::oasr_metadata::OASR_METADATA_KEY_FEATURE_DIARIZATION,
            openasr_core::models::oasr_metadata::OASR_FEATURE_DIARIZATION_COHERE_TOKEN_STREAM_V1,
        )
        .with_string_array_metadata("tokenizer.ggml.tokens", cohere_diarization_tokens());
        openasr_core::testing::write_tiny_gguf_runtime_source(&runtime_path, &spec).unwrap();

        let rendered = render_native_transcription_capability_summary(&runtime_path);

        assert!(rendered.contains("- diarization: supported=true, behavior=supported"));
    }

    #[test]
    fn transcription_capability_summary_keeps_base_cohere_diarization_unsupported() {
        let temp = tempfile::tempdir().unwrap();
        let _wespeaker_pack = EnvVarRestore::remove("OPENASR_WESPEAKER_PACK");
        let _home = EnvVarRestore::set("OPENASR_HOME", temp.path().as_os_str());
        let runtime_path = temp.path().join("cohere-runtime.oasr");
        let spec = openasr_core::testing::TinyGgufFixtureSpec::cohere_oasr_v1_runtime_ready(
            "cohere-runtime-fixture",
        );
        openasr_core::testing::write_tiny_gguf_runtime_source(&runtime_path, &spec).unwrap();

        let rendered = render_native_transcription_capability_summary(&runtime_path);

        assert!(rendered.contains("- diarization: supported=false, behavior=reject_request"));
    }

    fn cohere_diarization_tokens() -> [&'static str; 31] {
        [
            "<|startofcontext|>",
            "<|startoftranscript|>",
            "<|emo:undefined|>",
            "<|en|>",
            "<|pnc|>",
            "<|noitn|>",
            "<|notimestamp|>",
            "<|timestamp|>",
            "<|nodiarize|>",
            "<|diarize|>",
            "<|endoftext|>",
            "<|spltoken0|>",
            "▁fixture11",
            "▁fixture12",
            "▁fixture13",
            "▁fixture14",
            "▁fixture15",
            "▁fixture16",
            "▁fixture17",
            "▁fixture18",
            "▁fixture19",
            "▁fixture20",
            "▁fixture21",
            "▁fixture22",
            "▁fixture23",
            "▁fixture24",
            "▁fixture25",
            "▁fixture26",
            "▁fixture27",
            "▁fixture28",
            "▁fixture29",
        ]
    }
}
