//! Bounded-memory `.oasr` requantization through ggml's one quantization
//! implementation.
//!
//! External converters may emit fp16/q8_0 when their host language has no
//! K-quant implementation. This Module verifies the source pack, obtains its
//! semantic tensor policy from the architecture inventory, dequantizes and
//! requantizes one row at a time through ggml, writes through the sealed OASR
//! transaction, and returns the resulting verification proof. It is not a raw
//! GGUF writer and it never keeps the transformed multi-gigabyte pack in RAM.

use std::{collections::BTreeMap, io::Write, path::Path};

use thiserror::Error;

use crate::ggml_runtime::gguf_header::parse_gguf_header;
use crate::{
    VerifiedPack,
    ggml_runtime::{
        GgufMetadataValue, GgufStreamTensorSpec, GgufWriteError, GgufWriteTensorType,
        GgufWriteValue, build_runtime_tensor_reader_from_preflight, dequantize_ggml_row_to_f32,
        ggml_row_size_bytes, quantize_f32_to_ggml_tensor_data_into, write_gguf_file_streaming_v0,
    },
};

use super::{
    ggml_family_adapter::GGML_TOKENIZER_ID_KEY,
    oasr_metadata::{
        OASR_METADATA_KEY_AUDIO_FRONTEND, OASR_METADATA_KEY_BUILD_COMMIT,
        OASR_METADATA_KEY_DECODE_POLICY, OASR_METADATA_KEY_MODEL_ARCHITECTURE,
        OASR_METADATA_KEY_MODEL_FAMILY, OASR_METADATA_KEY_PACKAGE_VERSION, OasrPackWriter,
        PackEnvelope,
    },
    pack_quant::PackQuant,
    pack_quant::TensorQuantizationContract,
    pack_quant_audit::{is_block_quant_type, meets_q8_floor},
    pack_verifier::{PackCandidate, PackRoute, PackVerifier},
};

const PACK_QUANT_KEY: &str = "openasr.pack.quant";
const MODEL_ID_KEY: &str = "openasr.model.id";

#[derive(Debug)]
pub struct RequantizedPack {
    pub verified_pack: VerifiedPack,
    pub converted_tensor_count: usize,
    pub copied_tensor_count: usize,
}

#[derive(Debug, Error)]
pub enum PackRequantError {
    #[error("source pack verification failed: {reason}")]
    SourceVerification { reason: String },
    #[error("only ASR packs can be requantized; source route was {route}")]
    UnsupportedRoute { route: String },
    #[error("target quantization {target} is not supported by the requant seam")]
    UnsupportedTarget { target: &'static str },
    #[error("source pack is missing required string metadata '{key}'")]
    MissingMetadata { key: &'static str },
    #[error("source pack metadata '{key}' has unsupported value type")]
    UnsupportedMetadata { key: String },
    #[error(
        "source pack declares {declared} metadata entries but the lossless rewrite surface parsed {parsed}"
    )]
    IncompleteMetadataSurface { declared: u64, parsed: u64 },
    #[error("source model id '{model_id}' cannot be rebound to target quantization")]
    InvalidModelId { model_id: String },
    #[error("architecture '{architecture}' is absent from the canonical inventory")]
    UnknownArchitecture { architecture: String },
    #[error("no tensor in '{architecture}' is eligible for target {target}")]
    NoEligibleTensor {
        architecture: String,
        target: &'static str,
    },
    #[error(
        "source tensor '{tensor}' uses ggml type {ggml_type}, below its required Q8_0 precision floor"
    )]
    SourceBelowQ8Floor { tensor: String, ggml_type: i32 },
    #[error("could not build requant source reader: {reason}")]
    SourceReader { reason: String },
    #[error("could not start the sealed output transaction: {reason}")]
    OutputTransaction { reason: String },
    #[error("requant write failed: {reason}")]
    Write { reason: String },
    #[error("requant output failed production verification: {reason}")]
    OutputVerification { reason: String },
}

/// Requantize a verified ASR pack to a lower K-quant tier.
///
/// v1 deliberately exposes only Q4_K: it closes the real MiMo external-tool
/// gap without inventing a general arbitrary-precision conversion surface.
/// The inventory projection is exact: decoder matrices eligible for Q4_K are
/// written as Q4_K. Acoustic matrices are copied at their existing F16/F32 or
/// Q8-class precision: Q8_0 is a lower safety bound, not an instruction to
/// discard a source pack's higher acoustic precision. A below-floor acoustic
/// source fails closed because requantization cannot reconstruct lost signal.
pub fn requantize_oasr_pack(
    source: impl AsRef<Path>,
    output: impl AsRef<Path>,
    target: PackQuant,
) -> Result<RequantizedPack, PackRequantError> {
    if target != PackQuant::Q4_K {
        return Err(PackRequantError::UnsupportedTarget {
            target: target.label(),
        });
    }
    let source = PackVerifier
        .verify_candidate(PackCandidate::new(source.as_ref()))
        .map_err(|error| PackRequantError::SourceVerification {
            reason: error.to_string(),
        })?;
    let (model_architecture, envelope_architecture) = match source.route() {
        PackRoute::Asr {
            model_architecture, ..
        } => ((*model_architecture).to_string(), *model_architecture),
        route => {
            return Err(PackRequantError::UnsupportedRoute {
                route: format!("{route:?}"),
            });
        }
    };
    let descriptor = crate::arch::OpenAsrArchitectureRegistry::with_builtins()
        .find_by_model_architecture(&model_architecture)
        .ok_or_else(|| PackRequantError::UnknownArchitecture {
            architecture: model_architecture.clone(),
        })?;
    let preflight = source.preflight();
    let header =
        parse_gguf_header(preflight.runtime_source().backing_bytes()).map_err(|error| {
            PackRequantError::SourceReader {
                reason: format!("could not audit source metadata surface: {error}"),
            }
        })?;
    let parsed_metadata_count =
        u64::try_from(preflight.metadata().values().len()).map_err(|_| {
            PackRequantError::SourceReader {
                reason: "parsed metadata entry count does not fit u64".to_string(),
            }
        })?;
    if header.metadata_count != parsed_metadata_count {
        return Err(PackRequantError::IncompleteMetadataSurface {
            declared: header.metadata_count,
            parsed: parsed_metadata_count,
        });
    }
    let source_quant = preflight.metadata().get_string(PACK_QUANT_KEY);
    if source_quant == Some(target.label()) {
        return Err(PackRequantError::NoEligibleTensor {
            architecture: model_architecture,
            target: target.label(),
        });
    }
    let model_id = preflight
        .metadata()
        .get_string(MODEL_ID_KEY)
        .ok_or(PackRequantError::MissingMetadata { key: MODEL_ID_KEY })?;
    let output_model_id = requantized_model_id(model_id, target.label())?;
    let family_metadata =
        requantized_metadata(preflight.metadata().values(), &output_model_id, target)?;
    let transaction = OasrPackWriter::begin(
        output.as_ref(),
        PackEnvelope::asr(envelope_architecture),
        family_metadata,
    )
    .map_err(|error| PackRequantError::OutputTransaction {
        reason: error.to_string(),
    })?;

    let mut converted_tensor_count = 0_usize;
    let specs = preflight
        .tensor_index()
        .tensors()
        .iter()
        .map(|tensor| {
            let quantization_contract = descriptor.quantization_contract.tensor_classification;
            let preserve_source = preserve_q8_floor_source_tensor(
                quantization_contract,
                &tensor.name,
                tensor.ggml_type,
            )?;
            let target_type = (!preserve_source)
                .then(|| {
                    quantization_contract.target_write_type(&tensor.name, &tensor.dims, target)
                })
                .flatten();
            let ggml_type = if let Some(target_type) = target_type
                && tensor.ggml_type != target_type.ggml_type()
            {
                converted_tensor_count += 1;
                target_type.ggml_type()
            } else {
                tensor.ggml_type
            };
            Ok(GgufStreamTensorSpec {
                name: tensor.name.clone(),
                dims: tensor.dims.clone(),
                ggml_type,
            })
        })
        .collect::<Result<Vec<_>, PackRequantError>>()?;
    if converted_tensor_count == 0 {
        return Err(PackRequantError::NoEligibleTensor {
            architecture: model_architecture,
            target: target.label(),
        });
    }
    let reader = build_runtime_tensor_reader_from_preflight(preflight).map_err(|error| {
        PackRequantError::SourceReader {
            reason: error.to_string(),
        }
    })?;
    write_gguf_file_streaming_v0(
        transaction.staging_path(),
        transaction.sealed_metadata(),
        &specs,
        |index, target_spec, sink| {
            let source_payload = reader.host_tensor_bytes_by_id(index).map_err(|error| {
                GgufWriteError::TensorStreamingProducer {
                    name: target_spec.name.clone(),
                    reason: error.to_string(),
                }
            })?;
            if target_spec.ggml_type == source_payload.metadata.ggml_type {
                sink.write_all(source_payload.bytes).map_err(|error| {
                    GgufWriteError::TensorStreamingProducer {
                        name: target_spec.name.clone(),
                        reason: error.to_string(),
                    }
                })?;
                return Ok(());
            }
            requantize_tensor_rows(
                target_spec,
                source_payload.metadata.ggml_type,
                source_payload.bytes,
                sink,
            )
        },
    )
    .map_err(|error| PackRequantError::Write {
        reason: error.to_string(),
    })?;
    let verified_pack =
        transaction
            .commit()
            .map_err(|error| PackRequantError::OutputVerification {
                reason: error.to_string(),
            })?;
    Ok(RequantizedPack {
        verified_pack,
        converted_tensor_count,
        copied_tensor_count: specs.len() - converted_tensor_count,
    })
}

/// Return whether a tensor must be copied at source precision.
///
/// The semantic Q8 policy is a floor: F16/F32 and Q8-class source storage
/// already satisfies it. Re-encoding those weights to Q8 merely adds
/// quantization error and can cross a behavioral cliff. Conversely,
/// dequantizing a sub-Q8 source and writing Q8 would only relabel already-lost
/// information, so that input is rejected.
fn preserve_q8_floor_source_tensor(
    contract: TensorQuantizationContract,
    tensor: &str,
    source_ggml_type: i32,
) -> Result<bool, PackRequantError> {
    if !contract
        .tensor_role(tensor)
        .is_some_and(crate::models::pack_quant::TensorRole::requires_q8_floor)
    {
        return Ok(false);
    }
    let wire_type = u32::try_from(source_ggml_type).unwrap_or(u32::MAX);
    if is_block_quant_type(wire_type) && !meets_q8_floor(wire_type) {
        return Err(PackRequantError::SourceBelowQ8Floor {
            tensor: tensor.to_string(),
            ggml_type: source_ggml_type,
        });
    }
    Ok(true)
}

fn requantize_tensor_rows(
    target: &GgufStreamTensorSpec,
    source_ggml_type: i32,
    source_bytes: &[u8],
    sink: &mut dyn Write,
) -> Result<(), GgufWriteError> {
    let ne0 =
        usize::try_from(target.dims[0]).map_err(|_| GgufWriteError::TensorStreamingProducer {
            name: target.name.clone(),
            reason: "ne0 does not fit usize".to_string(),
        })?;
    let source_row_bytes = ggml_row_size_bytes(source_ggml_type, ne0).ok_or_else(|| {
        GgufWriteError::TensorStreamingProducer {
            name: target.name.clone(),
            reason: format!("source ggml type {source_ggml_type} has no valid row size"),
        }
    })?;
    let row_count = target
        .dims
        .iter()
        .skip(1)
        .try_fold(1_usize, |acc, dim| {
            usize::try_from(*dim)
                .ok()
                .and_then(|dim| acc.checked_mul(dim))
        })
        .ok_or_else(|| GgufWriteError::TensorStreamingProducer {
            name: target.name.clone(),
            reason: "row count does not fit usize".to_string(),
        })?;
    let expected_source_bytes = source_row_bytes.checked_mul(row_count).ok_or_else(|| {
        GgufWriteError::TensorStreamingProducer {
            name: target.name.clone(),
            reason: "source tensor byte count overflowed".to_string(),
        }
    })?;
    if source_bytes.len() != expected_source_bytes {
        return Err(GgufWriteError::TensorStreamingProducer {
            name: target.name.clone(),
            reason: format!(
                "source payload has {} bytes, expected {expected_source_bytes}",
                source_bytes.len()
            ),
        });
    }
    let target_type = match target.ggml_type {
        ggml_type if ggml_type == GgufWriteTensorType::Q4_K.ggml_type() => {
            GgufWriteTensorType::Q4_K
        }
        ggml_type if ggml_type == GgufWriteTensorType::Q8_0.ggml_type() => {
            GgufWriteTensorType::Q8_0
        }
        ggml_type => {
            return Err(GgufWriteError::TensorStreamingProducer {
                name: target.name.clone(),
                reason: format!("unsupported requant target ggml type {ggml_type}"),
            });
        }
    };
    let mut row = Vec::with_capacity(ne0);
    let mut quantized = Vec::new();
    for source_row in source_bytes.chunks_exact(source_row_bytes) {
        row.clear();
        if source_ggml_type == GgufWriteTensorType::F32.ggml_type() {
            row.extend(
                source_row
                    .chunks_exact(4)
                    .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])),
            );
        } else if source_ggml_type == GgufWriteTensorType::F16.ggml_type() {
            row.extend(source_row.chunks_exact(2).map(|bytes| {
                crate::nn::half::f16_bits_to_f32(u16::from_le_bytes([bytes[0], bytes[1]]))
            }));
        } else {
            dequantize_ggml_row_to_f32(source_ggml_type, source_row, ne0, &mut row).map_err(
                |error| GgufWriteError::TensorStreamingProducer {
                    name: target.name.clone(),
                    reason: error.to_string(),
                },
            )?;
        }
        quantize_f32_to_ggml_tensor_data_into(
            target_type,
            &[target.dims[0]],
            &row,
            &mut quantized,
        )?;
        sink.write_all(&quantized)
            .map_err(|error| GgufWriteError::TensorStreamingProducer {
                name: target.name.clone(),
                reason: error.to_string(),
            })?;
    }
    Ok(())
}

fn requantized_model_id(
    source_model_id: &str,
    target_quant: &str,
) -> Result<String, PackRequantError> {
    let base = match source_model_id.rsplit_once(':') {
        Some((base, suffix)) if ["fp16", "q8_0", "q3_k", "q4_k"].contains(&suffix) => base,
        Some(_) => {
            return Err(PackRequantError::InvalidModelId {
                model_id: source_model_id.to_string(),
            });
        }
        None => source_model_id,
    };
    if base.trim().is_empty() {
        return Err(PackRequantError::InvalidModelId {
            model_id: source_model_id.to_string(),
        });
    }
    Ok(format!("{base}:{target_quant}"))
}

fn requantized_metadata(
    source: &BTreeMap<String, GgufMetadataValue>,
    output_model_id: &str,
    target: PackQuant,
) -> Result<BTreeMap<String, GgufWriteValue>, PackRequantError> {
    let protected = [
        crate::arch::GENERAL_ARCHITECTURE_KEY,
        OASR_METADATA_KEY_PACKAGE_VERSION,
        OASR_METADATA_KEY_MODEL_FAMILY,
        OASR_METADATA_KEY_MODEL_ARCHITECTURE,
        OASR_METADATA_KEY_AUDIO_FRONTEND,
        OASR_METADATA_KEY_DECODE_POLICY,
        GGML_TOKENIZER_ID_KEY,
        OASR_METADATA_KEY_BUILD_COMMIT,
    ];
    let mut output = BTreeMap::new();
    for (key, value) in source {
        if protected.contains(&key.as_str()) || key == MODEL_ID_KEY || key == PACK_QUANT_KEY {
            continue;
        }
        let value = match value {
            GgufMetadataValue::String(value) => GgufWriteValue::String(value.clone()),
            GgufMetadataValue::U32(value) => GgufWriteValue::U32(*value),
            GgufMetadataValue::U64(value) => GgufWriteValue::U64(*value),
            GgufMetadataValue::Bool(value) => GgufWriteValue::Bool(*value),
            GgufMetadataValue::F32(value) if value.is_finite() => GgufWriteValue::F32(*value),
            GgufMetadataValue::F32(_) => {
                return Err(PackRequantError::UnsupportedMetadata { key: key.clone() });
            }
            GgufMetadataValue::StringArray(value) => GgufWriteValue::StringArray(value.clone()),
            GgufMetadataValue::U32Array(value) => GgufWriteValue::U32Array(value.clone()),
        };
        output.insert(key.clone(), value);
    }
    output.insert(
        MODEL_ID_KEY.to_string(),
        GgufWriteValue::String(output_model_id.to_string()),
    );
    output.insert(
        PACK_QUANT_KEY.to_string(),
        GgufWriteValue::String(target.label().to_string()),
    );
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn qwen_requant_fixture() -> crate::testing::TinyGgufFixtureSpec {
        use crate::models::tensor_binding::TensorBindingDescriptorRequirement;

        let mut spec = crate::testing::TinyGgufFixtureSpec::qwen3_asr_oasr_v1_metadata_ready_for_runtime_fail_closed(
            "qwen3-asr-fixture:fp16",
        )
        .with_metadata(PACK_QUANT_KEY, "fp16")
        .with_metadata("qwen3-asr.llm.d_model", "256")
        .with_metadata("qwen3-asr.llm.n_heads", "8")
        .with_metadata("qwen3-asr.llm.n_kv_heads", "8")
        .with_metadata("qwen3-asr.llm.head_dim", "32")
        .with_metadata("qwen3-asr.llm.vocab_size", "256")
        .without_tensor("fixture.tensor");
        for (name, dims) in [
            ("audio.mel_filters", vec![8, 201]),
            ("audio.mel_window", vec![400]),
            ("audio.conv.1.weight", vec![3, 3, 1, 2]),
            ("audio.conv.1.bias", vec![2]),
            ("audio.conv.2.weight", vec![3, 3, 2, 2]),
            ("audio.conv.2.bias", vec![2]),
            ("audio.conv.3.weight", vec![3, 3, 2, 2]),
            ("audio.conv.3.bias", vec![2]),
            ("audio.conv_out.weight", vec![2, 16]),
            ("audio.ln_post.weight", vec![16]),
            ("audio.ln_post.bias", vec![16]),
            ("audio.proj1.weight", vec![16, 16]),
            ("audio.proj1.bias", vec![16]),
            ("audio.proj2.weight", vec![16, 256]),
            ("audio.proj2.bias", vec![256]),
        ] {
            let rank = dims.len();
            spec = spec.with_tensor_shape(name, dims);
            if rank >= 2 {
                spec = spec.with_tensor_f16(name);
            }
        }
        let metadata =
            crate::models::qwen::runtime_contract::parse_qwen3_execution_metadata(&spec.metadata)
                .expect("qwen requant fixture metadata");
        let decoder_contract = crate::models::qwen::QwenDecoderContract::bind(
            crate::models::qwen::QwenDecoderContractGeometry {
                n_layers: metadata.llm_layers,
                d_model: metadata.llm_d_model,
                n_heads: metadata.llm_heads,
                n_kv_heads: metadata.llm_kv_heads,
                head_dim: metadata.llm_head_dim,
                ffn_dim: metadata.llm_d_model,
                vocab_size: metadata.vocab_size,
            },
            crate::models::qwen::runtime_contract::qwen3_asr_decoder_profile(),
        )
        .expect("qwen requant fixture decoder contract");
        for descriptor in crate::models::qwen::runtime_contract::qwen3_runtime_tensor_descriptors(
            metadata,
            &decoder_contract,
        )
        .expect("qwen requant fixture descriptors")
        {
            let dims = match descriptor.requirement {
                TensorBindingDescriptorRequirement::ExactDims(dims) => dims,
                TensorBindingDescriptorRequirement::VectorLen(len) => vec![len],
                TensorBindingDescriptorRequirement::NonEmptyVector => vec![1],
                TensorBindingDescriptorRequirement::Rank2WithDim(dim) => vec![dim, dim],
                TensorBindingDescriptorRequirement::Rank2EitherDims(lhs, rhs)
                | TensorBindingDescriptorRequirement::Rank2OrRank3WithDims(lhs, rhs) => {
                    vec![lhs, rhs]
                }
                TensorBindingDescriptorRequirement::RankAtLeastWithDimAt {
                    min_rank,
                    axis,
                    dim,
                } => {
                    let mut dims = vec![1; min_rank.max(axis.saturating_add(1))];
                    dims[axis] = dim;
                    dims
                }
            };
            let rank = dims.len();
            let tensor_name = descriptor.tensor_name;
            spec =
                spec.with_tensor_shape(tensor_name.clone(), dims.into_iter().map(|dim| dim as u64));
            if rank >= 2 {
                spec = spec.with_tensor_f16(tensor_name);
            }
        }
        spec
    }

    #[test]
    fn model_id_rebind_is_exact_and_does_not_guess_another_suffix() {
        assert_eq!(
            requantized_model_id("mimo-v2.5-asr:q8_0", "q4_k").unwrap(),
            "mimo-v2.5-asr:q4_k"
        );
        assert_eq!(
            requantized_model_id("moss-transcribe-diarize", "q4_k").unwrap(),
            "moss-transcribe-diarize:q4_k"
        );
        assert!(requantized_model_id("mimo-v2.5-asr:custom", "q4_k").is_err());
    }

    #[test]
    fn acoustic_floor_preserves_high_precision_and_rejects_irrecoverable_sources() {
        let contract = TensorQuantizationContract::EntireAcousticPack {
            model_architecture: "fixture-acoustic",
        };
        assert!(
            preserve_q8_floor_source_tensor(
                contract,
                "encoder.weight",
                GgufWriteTensorType::F16.ggml_type()
            )
            .unwrap()
        );
        assert!(
            preserve_q8_floor_source_tensor(
                contract,
                "encoder.weight",
                GgufWriteTensorType::Q8_0.ggml_type()
            )
            .unwrap()
        );
        assert!(matches!(
            preserve_q8_floor_source_tensor(
                contract,
                "encoder.weight",
                GgufWriteTensorType::Q4_K.ggml_type()
            ),
            Err(PackRequantError::SourceBelowQ8Floor { .. })
        ));
    }

    #[test]
    fn semantic_q8_floor_preserves_forced_aligner_boundaries_only() {
        let contract = TensorQuantizationContract::SemanticRolesV1 {
            model_architecture: crate::models::qwen::QWEN3_FORCED_ALIGNER_GGML_ARCHITECTURE_ID,
            classify: crate::models::qwen::forced_aligner_tensor_role,
            quantized_axis: crate::models::pack_quant::QuantizedAxis::First,
        };
        for tensor in ["output.weight", "token_embd.weight"] {
            assert!(
                preserve_q8_floor_source_tensor(
                    contract,
                    tensor,
                    GgufWriteTensorType::Q8_0.ggml_type(),
                )
                .expect("Q8 boundary source satisfies the floor")
            );
            assert!(matches!(
                preserve_q8_floor_source_tensor(
                    contract,
                    tensor,
                    GgufWriteTensorType::Q4_K.ggml_type(),
                ),
                Err(PackRequantError::SourceBelowQ8Floor { .. })
            ));
        }
        assert!(
            !preserve_q8_floor_source_tensor(
                contract,
                "blk.0.ffn_gate.weight",
                GgufWriteTensorType::Q8_0.ggml_type(),
            )
            .expect("decoder matrix has no Q8 floor")
        );
        assert_eq!(
            contract.target_write_type("blk.0.ffn_gate.weight", &[256, 256], PackQuant::Q4_K,),
            Some(GgufWriteTensorType::Q4_K)
        );
    }

    #[test]
    fn existing_output_is_never_overwritten() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("source.oasr");
        let output = dir.path().join("output.oasr");
        crate::testing::write_tiny_gguf_runtime_source(&source, &qwen_requant_fixture())
            .expect("write source fixture");
        std::fs::write(&output, b"caller-owned").unwrap();

        assert!(matches!(
            requantize_oasr_pack(&source, &output, PackQuant::Q4_K),
            Err(PackRequantError::OutputTransaction { .. })
        ));
        assert_eq!(std::fs::read(&output).unwrap(), b"caller-owned");
    }

    #[test]
    fn requant_output_is_verified_and_contains_q4_k_decoder_tensors() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("source.oasr");
        let output = dir.path().join("output.oasr");
        crate::testing::write_tiny_gguf_runtime_source(&source, &qwen_requant_fixture())
            .expect("write source fixture");

        let result = requantize_oasr_pack(&source, &output, PackQuant::Q4_K)
            .expect("requant verified source");
        assert_eq!(result.verified_pack.path(), output);
        assert!(result.converted_tensor_count > 0);
        assert!(result.copied_tensor_count > 0);
        assert_eq!(
            result
                .verified_pack
                .preflight()
                .metadata()
                .get_string(PACK_QUANT_KEY),
            Some("q4_k")
        );
        assert_eq!(
            result
                .verified_pack
                .preflight()
                .metadata()
                .get_string(MODEL_ID_KEY),
            Some("qwen3-asr-fixture:q4_k")
        );
        assert!(
            result
                .verified_pack
                .preflight()
                .tensor_index()
                .tensors()
                .iter()
                .any(|tensor| tensor.ggml_type == GgufWriteTensorType::Q4_K.ggml_type())
        );
        let acoustic = result
            .verified_pack
            .preflight()
            .tensor_index()
            .tensors()
            .iter()
            .find(|tensor| tensor.name == "audio.proj2.weight")
            .expect("acoustic projection");
        assert_eq!(acoustic.ggml_type, GgufWriteTensorType::F16.ggml_type());
        let norm = result
            .verified_pack
            .preflight()
            .tensor_index()
            .tensors()
            .iter()
            .find(|tensor| tensor.name == "blk.0.attn_norm.weight")
            .expect("rank-one decoder norm");
        assert_eq!(norm.dims, vec![256]);
        assert_eq!(norm.ggml_type, GgufWriteTensorType::F32.ggml_type());
    }
}
