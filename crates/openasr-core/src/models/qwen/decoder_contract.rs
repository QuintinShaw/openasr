//! Shared Qwen-shaped LLM decoder pack contract.
//!
//! FunASR-Nano, MOSS-Transcribe-Diarize, MiMo-ASR (backbone), and FireRedASR2-LLM
//! all materialize the same decoder math through [`QwenWholeDecoderPlan`]. Their
//! admission descriptors, fixtures, and access-trace expectations must expand
//! from one semantic description so the 11-tensor layer pattern cannot drift
//! per family.
//!
//! Family adapters supply only:
//! - [`QwenDecoderContractGeometry`] from pack metadata;
//! - [`QwenDecoderContractOptions`] (Qwen3 qk-norm vs Qwen2 qkv-bias);
//! - per-layer [`super::QwenFamilyLlmLayerTensorNames`] (prefix / field spelling);
//! - [`QwenDecoderTailTensorNames`] for norm / logits / embedding constants.
//!
//! Projection weights use ordered ggml `[in, out]` [`ExactDims`] so a transposed
//! pack fails closed at admission (same rule as FastConformer after pack-contract
//! hardening). Tied-embedding decoders omit a separate logits weight via
//! [`QwenDecoderTailTensorNames::output_weight`] = `None`.

use crate::models::tensor_binding::{TensorBindingDescriptor, TensorBindingDescriptorRequirement};

use super::QwenFamilyLlmLayerTensorNames;

/// Pack-metadata geometry for one Qwen-shaped decoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct QwenDecoderContractGeometry {
    pub n_layers: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub ffn_dim: usize,
    pub vocab_size: usize,
}

impl QwenDecoderContractGeometry {
    pub(crate) fn q_dim(self) -> Option<usize> {
        self.n_heads.checked_mul(self.head_dim)
    }

    pub(crate) fn kv_dim(self) -> Option<usize> {
        self.n_kv_heads.checked_mul(self.head_dim)
    }

    /// Structural pins shared by every Qwen-shaped decoder family.
    pub(crate) fn validate_basic(self) -> Result<(), String> {
        if self.n_layers == 0 {
            return Err("qwen decoder n_layers must be positive".to_string());
        }
        for (label, value) in [
            ("d_model", self.d_model),
            ("n_heads", self.n_heads),
            ("n_kv_heads", self.n_kv_heads),
            ("head_dim", self.head_dim),
            ("ffn_dim", self.ffn_dim),
            ("vocab_size", self.vocab_size),
        ] {
            if value == 0 {
                return Err(format!("qwen decoder {label} must be positive"));
            }
        }
        if !self.n_heads.is_multiple_of(self.n_kv_heads) {
            return Err(format!(
                "qwen decoder n_heads ({}) must be a multiple of n_kv_heads ({})",
                self.n_heads, self.n_kv_heads
            ));
        }
        let q_dim = self.q_dim().ok_or_else(|| {
            format!(
                "qwen decoder q_dim overflows: n_heads={} head_dim={}",
                self.n_heads, self.head_dim
            )
        })?;
        let kv_dim = self.kv_dim().ok_or_else(|| {
            format!(
                "qwen decoder kv_dim overflows: n_kv_heads={} head_dim={}",
                self.n_kv_heads, self.head_dim
            )
        })?;
        if q_dim == 0 || kv_dim == 0 {
            return Err("qwen decoder q_dim/kv_dim must be positive".to_string());
        }
        Ok(())
    }
}

/// Typed variation between Qwen2- and Qwen3-class checkpoints.
///
/// This is not a second family registry: each adapter picks the option pair that
/// matches its loader / [`QwenWholeDecoderPlan`] wiring.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct QwenDecoderContractOptions {
    /// Per-head Q/K RMSNorm tensors (`attn_q_norm` / `attn_k_norm`) are present.
    pub qk_norm: bool,
    /// Q/K/V projection bias tensors are present.
    pub qkv_bias: bool,
}

impl QwenDecoderContractOptions {
    pub(crate) const QWEN3: Self = Self {
        qk_norm: true,
        qkv_bias: false,
    };

    /// Qwen2-class option pair (MiMo backbone / FireRedASR2-LLM). Not yet
    /// referenced from production adapters on this commit; kept as the typed
    /// variation constant tests and upcoming migrations share.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) const QWEN2: Self = Self {
        qk_norm: false,
        qkv_bias: true,
    };
}

/// Tail tensor names the family loader already uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct QwenDecoderTailTensorNames<'a> {
    pub output_norm: &'a str,
    /// `None` when embeddings are tied and no separate logits weight is packed
    /// (e.g. MOSS-Transcribe-Diarize).
    pub output_weight: Option<&'a str>,
    pub token_embd: &'a str,
}

fn descriptor(
    tensor_name: impl Into<String>,
    requirement: TensorBindingDescriptorRequirement,
    reason: impl Into<String>,
) -> TensorBindingDescriptor {
    TensorBindingDescriptor {
        tensor_name: tensor_name.into(),
        requirement,
        reason: reason.into(),
    }
}

fn exact_matrix(
    name: String,
    rows_in: usize,
    cols_out: usize,
    reason: &str,
) -> TensorBindingDescriptor {
    descriptor(
        name,
        TensorBindingDescriptorRequirement::ExactDims(vec![rows_in, cols_out]),
        reason,
    )
}

/// One decoder layer's admission descriptors.
///
/// `names` must already reflect `options`: qk-norm / qkv-bias optional fields
/// are read only when the corresponding option is enabled. Extra name strings
/// present while the option is off are ignored (never admitted).
pub(crate) fn qwen_decoder_layer_tensor_descriptors(
    geometry: &QwenDecoderContractGeometry,
    options: QwenDecoderContractOptions,
    names: &QwenFamilyLlmLayerTensorNames,
) -> Result<Vec<TensorBindingDescriptor>, String> {
    geometry.validate_basic()?;
    let d_model = geometry.d_model;
    let q_dim = geometry
        .q_dim()
        .ok_or_else(|| "qwen decoder q_dim overflow".to_string())?;
    let kv_dim = geometry
        .kv_dim()
        .ok_or_else(|| "qwen decoder kv_dim overflow".to_string())?;
    let ffn = geometry.ffn_dim;

    let mut out = Vec::with_capacity(14);
    out.push(descriptor(
        names.attn_norm_name.clone(),
        TensorBindingDescriptorRequirement::VectorLen(d_model),
        "attention RMSNorm must span d_model",
    ));
    out.push(exact_matrix(
        names.attn_q_name.clone(),
        d_model,
        q_dim,
        "query projection must be ggml [d_model, n_heads*head_dim]",
    ));
    out.push(exact_matrix(
        names.attn_k_name.clone(),
        d_model,
        kv_dim,
        "key projection must be ggml [d_model, n_kv_heads*head_dim]",
    ));
    out.push(exact_matrix(
        names.attn_v_name.clone(),
        d_model,
        kv_dim,
        "value projection must be ggml [d_model, n_kv_heads*head_dim]",
    ));
    out.push(exact_matrix(
        names.attn_output_name.clone(),
        q_dim,
        d_model,
        "attention output projection must be ggml [n_heads*head_dim, d_model]",
    ));

    if options.qk_norm {
        let q_norm = names.q_norm_name.as_ref().ok_or_else(|| {
            "qwen decoder options.qk_norm requires q_norm_name on the layer name set".to_string()
        })?;
        let k_norm = names.k_norm_name.as_ref().ok_or_else(|| {
            "qwen decoder options.qk_norm requires k_norm_name on the layer name set".to_string()
        })?;
        out.push(descriptor(
            q_norm.clone(),
            TensorBindingDescriptorRequirement::VectorLen(geometry.head_dim),
            "QK-norm query RMSNorm must span head_dim",
        ));
        out.push(descriptor(
            k_norm.clone(),
            TensorBindingDescriptorRequirement::VectorLen(geometry.head_dim),
            "QK-norm key RMSNorm must span head_dim",
        ));
    }

    if options.qkv_bias {
        let q_bias = names.q_bias_name.as_ref().ok_or_else(|| {
            "qwen decoder options.qkv_bias requires q_bias_name on the layer name set".to_string()
        })?;
        let k_bias = names.k_bias_name.as_ref().ok_or_else(|| {
            "qwen decoder options.qkv_bias requires k_bias_name on the layer name set".to_string()
        })?;
        let v_bias = names.v_bias_name.as_ref().ok_or_else(|| {
            "qwen decoder options.qkv_bias requires v_bias_name on the layer name set".to_string()
        })?;
        out.push(descriptor(
            q_bias.clone(),
            TensorBindingDescriptorRequirement::VectorLen(q_dim),
            "query bias must span n_heads*head_dim",
        ));
        out.push(descriptor(
            k_bias.clone(),
            TensorBindingDescriptorRequirement::VectorLen(kv_dim),
            "key bias must span n_kv_heads*head_dim",
        ));
        out.push(descriptor(
            v_bias.clone(),
            TensorBindingDescriptorRequirement::VectorLen(kv_dim),
            "value bias must span n_kv_heads*head_dim",
        ));
    }

    out.push(descriptor(
        names.ffn_norm_name.clone(),
        TensorBindingDescriptorRequirement::VectorLen(d_model),
        "FFN RMSNorm must span d_model",
    ));
    out.push(exact_matrix(
        names.ffn_gate_name.clone(),
        d_model,
        ffn,
        "FFN gate projection must be ggml [d_model, ffn_dim]",
    ));
    out.push(exact_matrix(
        names.ffn_up_name.clone(),
        d_model,
        ffn,
        "FFN up projection must be ggml [d_model, ffn_dim]",
    ));
    out.push(exact_matrix(
        names.ffn_down_name.clone(),
        ffn,
        d_model,
        "FFN down projection must be ggml [ffn_dim, d_model]",
    ));
    Ok(out)
}

/// Final norm, optional logits head, and token embedding.
pub(crate) fn qwen_decoder_tail_tensor_descriptors(
    geometry: &QwenDecoderContractGeometry,
    tail: QwenDecoderTailTensorNames<'_>,
) -> Result<Vec<TensorBindingDescriptor>, String> {
    geometry.validate_basic()?;
    let d_model = geometry.d_model;
    let vocab = geometry.vocab_size;
    let mut out = vec![
        descriptor(
            tail.output_norm.to_string(),
            TensorBindingDescriptorRequirement::VectorLen(d_model),
            "final RMSNorm before the logits head must span d_model",
        ),
        exact_matrix(
            tail.token_embd.to_string(),
            d_model,
            vocab,
            "token embedding table must be ggml [d_model, vocab]",
        ),
    ];
    if let Some(output_weight) = tail.output_weight {
        out.push(exact_matrix(
            output_weight.to_string(),
            d_model,
            vocab,
            "logits head must be ggml [d_model, vocab]",
        ));
    }
    Ok(out)
}

/// Full decoder-half contract: every layer plus the tail.
pub(crate) fn qwen_decoder_runtime_tensor_descriptors(
    geometry: &QwenDecoderContractGeometry,
    options: QwenDecoderContractOptions,
    mut names_for_layer: impl FnMut(usize) -> QwenFamilyLlmLayerTensorNames,
    tail: QwenDecoderTailTensorNames<'_>,
) -> Result<Vec<TensorBindingDescriptor>, String> {
    geometry.validate_basic()?;
    let mut descriptors = Vec::new();
    for layer in 0..geometry.n_layers {
        descriptors.extend(qwen_decoder_layer_tensor_descriptors(
            geometry,
            options,
            &names_for_layer(layer),
        )?);
    }
    descriptors.extend(qwen_decoder_tail_tensor_descriptors(geometry, tail)?);
    Ok(descriptors)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::tensor_binding::{
        project_fixture_tensors, validate_tensor_binding_descriptors,
    };
    use crate::{GgufTensorIndex, GgufTensorMetadata, ggml_runtime::GgufTensorIndexSnapshot};

    fn qwen3_geometry() -> QwenDecoderContractGeometry {
        QwenDecoderContractGeometry {
            n_layers: 2,
            d_model: 16,
            n_heads: 4,
            n_kv_heads: 2,
            head_dim: 4,
            ffn_dim: 32,
            vocab_size: 64,
        }
    }

    fn qwen2_geometry() -> QwenDecoderContractGeometry {
        QwenDecoderContractGeometry {
            n_layers: 2,
            d_model: 16,
            n_heads: 4,
            n_kv_heads: 4,
            head_dim: 4,
            ffn_dim: 32,
            vocab_size: 64,
        }
    }

    fn qwen3_layer_names(layer: usize) -> QwenFamilyLlmLayerTensorNames {
        let p = format!("blk.{layer}");
        QwenFamilyLlmLayerTensorNames {
            attn_norm_name: format!("{p}.attn_norm.weight"),
            attn_q_name: format!("{p}.attn_q.weight"),
            attn_k_name: format!("{p}.attn_k.weight"),
            attn_v_name: format!("{p}.attn_v.weight"),
            attn_output_name: format!("{p}.attn_output.weight"),
            q_norm_name: Some(format!("{p}.attn_q_norm.weight")),
            k_norm_name: Some(format!("{p}.attn_k_norm.weight")),
            q_bias_name: None,
            k_bias_name: None,
            v_bias_name: None,
            ffn_norm_name: format!("{p}.ffn_norm.weight"),
            ffn_gate_name: format!("{p}.ffn_gate.weight"),
            ffn_up_name: format!("{p}.ffn_up.weight"),
            ffn_down_name: format!("{p}.ffn_down.weight"),
        }
    }

    fn qwen2_layer_names(layer: usize) -> QwenFamilyLlmLayerTensorNames {
        let p = format!("llm.blk.{layer}");
        QwenFamilyLlmLayerTensorNames {
            attn_norm_name: format!("{p}.attn_norm.weight"),
            attn_q_name: format!("{p}.attn_q.weight"),
            attn_k_name: format!("{p}.attn_k.weight"),
            attn_v_name: format!("{p}.attn_v.weight"),
            attn_output_name: format!("{p}.attn_out.weight"),
            q_norm_name: None,
            k_norm_name: None,
            q_bias_name: Some(format!("{p}.attn_q.bias")),
            k_bias_name: Some(format!("{p}.attn_k.bias")),
            v_bias_name: Some(format!("{p}.attn_v.bias")),
            ffn_norm_name: format!("{p}.ffn_norm.weight"),
            ffn_gate_name: format!("{p}.ffn_gate.weight"),
            ffn_up_name: format!("{p}.ffn_up.weight"),
            ffn_down_name: format!("{p}.ffn_down.weight"),
        }
    }

    fn index_from_descriptors(descriptors: &[TensorBindingDescriptor]) -> GgufTensorIndex {
        let tensors = project_fixture_tensors(descriptors)
            .into_iter()
            .enumerate()
            .map(|(index, (name, dims))| GgufTensorMetadata {
                name,
                dims,
                ggml_type: 0,
                type_name: "f32".to_string(),
                size_bytes: 0,
                offset_bytes: index as u64,
            })
            .collect();
        GgufTensorIndex::from_snapshot(GgufTensorIndexSnapshot {
            path: std::path::PathBuf::from("qwen-decoder-contract-test.oasr"),
            data_section_offset_bytes: 0,
            tensors,
        })
        .expect("unique names")
    }

    #[test]
    fn qwen3_layer_count_includes_qk_norm_and_excludes_bias() {
        let g = qwen3_geometry();
        let names = qwen3_layer_names(0);
        let layer =
            qwen_decoder_layer_tensor_descriptors(&g, QwenDecoderContractOptions::QWEN3, &names)
                .expect("qwen3 layer");
        // 5 attn weights + 2 qk-norm + 1 ffn_norm + 3 ffn = 11
        assert_eq!(layer.len(), 11);
        let names_set: std::collections::BTreeSet<_> =
            layer.iter().map(|d| d.tensor_name.as_str()).collect();
        assert!(names_set.contains("blk.0.attn_q_norm.weight"));
        assert!(!names_set.iter().any(|n| n.contains("bias")));
        assert!(matches!(
            layer
                .iter()
                .find(|d| d.tensor_name == "blk.0.attn_q.weight")
                .unwrap()
                .requirement,
            TensorBindingDescriptorRequirement::ExactDims(ref dims) if dims == &[16, 16]
        ));
        // q_dim = 4*4 = 16, kv_dim = 2*4 = 8
        assert!(matches!(
            layer
                .iter()
                .find(|d| d.tensor_name == "blk.0.attn_k.weight")
                .unwrap()
                .requirement,
            TensorBindingDescriptorRequirement::ExactDims(ref dims) if dims == &[16, 8]
        ));
        assert!(matches!(
            layer
                .iter()
                .find(|d| d.tensor_name == "blk.0.attn_output.weight")
                .unwrap()
                .requirement,
            TensorBindingDescriptorRequirement::ExactDims(ref dims) if dims == &[16, 16]
        ));
    }

    #[test]
    fn qwen2_layer_count_includes_bias_and_excludes_qk_norm() {
        let g = qwen2_geometry();
        let names = qwen2_layer_names(0);
        let layer =
            qwen_decoder_layer_tensor_descriptors(&g, QwenDecoderContractOptions::QWEN2, &names)
                .expect("qwen2 layer");
        // 5 attn weights + 3 bias + 1 ffn_norm + 3 ffn = 12
        assert_eq!(layer.len(), 12);
        let names_set: std::collections::BTreeSet<_> =
            layer.iter().map(|d| d.tensor_name.as_str()).collect();
        assert!(names_set.contains("llm.blk.0.attn_q.bias"));
        assert!(names_set.contains("llm.blk.0.attn_out.weight"));
        assert!(
            !names_set
                .iter()
                .any(|n| n.contains("q_norm") || n.contains("k_norm"))
        );
    }

    #[test]
    fn full_decoder_with_tied_embeddings_omits_logits_weight() {
        let g = qwen3_geometry();
        let descriptors = qwen_decoder_runtime_tensor_descriptors(
            &g,
            QwenDecoderContractOptions::QWEN3,
            qwen3_layer_names,
            QwenDecoderTailTensorNames {
                output_norm: "output_norm.weight",
                output_weight: None,
                token_embd: "token_embd.weight",
            },
        )
        .expect("tied decoder");
        // 2 layers * 11 + norm + embd = 24
        assert_eq!(descriptors.len(), 24);
        assert!(descriptors.iter().all(|d| d.tensor_name != "output.weight"));
    }

    #[test]
    fn full_decoder_with_separate_logits_includes_output_weight() {
        let g = qwen2_geometry();
        let descriptors = qwen_decoder_runtime_tensor_descriptors(
            &g,
            QwenDecoderContractOptions::QWEN2,
            qwen2_layer_names,
            QwenDecoderTailTensorNames {
                output_norm: "llm.out_norm.weight",
                output_weight: Some("llm.lm_head.weight"),
                token_embd: "llm.tok_emb.weight",
            },
        )
        .expect("untied decoder");
        // 2 * 12 + norm + embd + logits = 27
        assert_eq!(descriptors.len(), 27);
        assert!(
            descriptors
                .iter()
                .any(|d| d.tensor_name == "llm.lm_head.weight")
        );
    }

    #[test]
    fn fixture_projection_satisfies_the_contract() {
        let g = qwen3_geometry();
        let descriptors = qwen_decoder_runtime_tensor_descriptors(
            &g,
            QwenDecoderContractOptions::QWEN3,
            qwen3_layer_names,
            QwenDecoderTailTensorNames {
                output_norm: "output_norm.weight",
                output_weight: Some("output.weight"),
                token_embd: "token_embd.weight",
            },
        )
        .expect("descriptors");
        let index = index_from_descriptors(&descriptors);
        validate_tensor_binding_descriptors(
            &index,
            &descriptors,
            |name| format!("missing:{name}"),
            |name, dims, reason| format!("{name}:{dims:?}:{reason}"),
        )
        .expect("exact fixture projection must validate");
    }

    #[test]
    fn rejects_transposed_query_projection() {
        let g = qwen3_geometry();
        let names = qwen3_layer_names(0);
        let mut descriptors =
            qwen_decoder_layer_tensor_descriptors(&g, QwenDecoderContractOptions::QWEN3, &names)
                .expect("layer");
        let q = descriptors
            .iter_mut()
            .find(|d| d.tensor_name == "blk.0.attn_q.weight")
            .expect("q");
        // Correct ggml [d_model, q_dim] = [16, 16]; force a deliberate wrong pair
        // that EitherDims would have accepted for a non-square case by using kv.
        let g_rect = QwenDecoderContractGeometry {
            n_kv_heads: 2,
            ..qwen3_geometry()
        };
        let names = qwen3_layer_names(0);
        let mut rect = qwen_decoder_layer_tensor_descriptors(
            &g_rect,
            QwenDecoderContractOptions::QWEN3,
            &names,
        )
        .expect("rect layer");
        let k = rect
            .iter_mut()
            .find(|d| d.tensor_name == "blk.0.attn_k.weight")
            .expect("k");
        // Swap to [kv_dim, d_model] = [8, 16] instead of [16, 8].
        k.requirement = TensorBindingDescriptorRequirement::ExactDims(vec![8, 16]);
        // Build an index that has the transposed k only for that tensor; others correct.
        let mut shapes = project_fixture_tensors(&rect);
        for (name, dims) in &mut shapes {
            if name == "blk.0.attn_k.weight" {
                *dims = vec![8, 16];
            }
        }
        let tensors = shapes
            .into_iter()
            .enumerate()
            .map(|(index, (name, dims))| GgufTensorMetadata {
                name,
                dims,
                ggml_type: 0,
                type_name: "f32".to_string(),
                size_bytes: 0,
                offset_bytes: index as u64,
            })
            .collect();
        let index = GgufTensorIndex::from_snapshot(GgufTensorIndexSnapshot {
            path: std::path::PathBuf::from("qwen-decoder-transpose.oasr"),
            data_section_offset_bytes: 0,
            tensors,
        })
        .expect("unique");
        // Restore correct requirements for validation against the bad index.
        let good = qwen_decoder_layer_tensor_descriptors(
            &g_rect,
            QwenDecoderContractOptions::QWEN3,
            &names,
        )
        .expect("good");
        let err = validate_tensor_binding_descriptors(
            &index,
            &good,
            |name| format!("missing:{name}"),
            |name, dims, reason| format!("{name}:{dims:?}:{reason}"),
        )
        .expect_err("transposed k must fail ExactDims");
        assert!(
            err.contains("blk.0.attn_k.weight"),
            "unexpected error: {err}"
        );
        let _ = descriptors;
        let _ = q;
    }

    #[test]
    fn qk_norm_option_requires_name_slots() {
        let g = qwen3_geometry();
        let mut names = qwen3_layer_names(0);
        names.q_norm_name = None;
        let err =
            qwen_decoder_layer_tensor_descriptors(&g, QwenDecoderContractOptions::QWEN3, &names)
                .expect_err("missing q_norm name must fail");
        assert!(err.contains("q_norm_name"), "{err}");
    }

    #[test]
    fn rejects_n_heads_not_multiple_of_kv_heads() {
        let g = QwenDecoderContractGeometry {
            n_heads: 5,
            n_kv_heads: 2,
            ..qwen3_geometry()
        };
        assert!(g.validate_basic().is_err());
    }
}
