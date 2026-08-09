//! Shared Qwen-shaped LLM decoder pack contract.
//!
//! FunASR-Nano, MOSS-Transcribe-Diarize, MiMo-ASR (backbone), and FireRedASR2-LLM
//! all materialize the same decoder math through [`QwenWholeDecoderPlan`]. Their
//! admission descriptors, fixtures, and access-trace expectations must expand
//! from one semantic description so the per-layer tensor set cannot drift
//! per family. Each layer always emits the base 9 tensors (attn norm, q/k/v/out,
//! ffn norm, gate/up/down); Qwen3 adds 2 qk-norm tensors (11 total) and Qwen2
//! adds 3 qkv-bias tensors (12 total).
//!
//! Family adapters bind exactly once from:
//! - [`QwenDecoderContractGeometry`] from pack metadata;
//! - a [`QwenFamilyDecoderProfile`] that binds the closed Qwen2/Qwen3 variation
//!   to the family's layer-name provider so
//!   admission and whole-decoder planning cannot pick different option pairs;
//! - [`QwenDecoderTailTensorNames`] for norm / logits / embedding constants.
//!
//! Only the resulting [`QwenDecoderContract`] crosses admission, planning,
//! tail loading, host quoting, or backend compilation seams; the three raw
//! inputs are adapter-local construction details.
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

/// Architecture ceilings for untrusted pack geometry. Generous headroom over
/// published Qwen2/Qwen3 ASR checkpoints; parse paths should mirror these so
/// descriptor construction cannot allocate without bound.
pub(crate) const QWEN_DECODER_MAX_LAYERS: usize = 512;
pub(crate) const QWEN_DECODER_MAX_D_MODEL: usize = 65_536;
pub(crate) const QWEN_DECODER_MAX_N_HEADS: usize = 1_024;
pub(crate) const QWEN_DECODER_MAX_HEAD_DIM: usize = 1_024;
pub(crate) const QWEN_DECODER_MAX_FFN_DIM: usize = 262_144;
pub(crate) const QWEN_DECODER_MAX_VOCAB_SIZE: usize = 1_000_000;
/// Cap on total tensor obligations one decoder half may construct
/// (layers * per-layer tensors + tail). Keeps malicious metadata fail-closed.
pub(crate) const QWEN_DECODER_MAX_TENSOR_OBLIGATIONS: usize = 1_000_000;

impl QwenDecoderContractGeometry {
    pub(crate) fn q_dim(self) -> Option<usize> {
        self.n_heads.checked_mul(self.head_dim)
    }

    pub(crate) fn kv_dim(self) -> Option<usize> {
        self.n_kv_heads.checked_mul(self.head_dim)
    }

    /// Exact descriptor count emitted by one layer of `variant`.
    const fn layer_tensor_count(variant: QwenDecoderVariant) -> usize {
        // attn_norm + q/k/v/out + ffn_norm + gate/up/down = 9
        // + optional 2 qk-norm or 3 qkv-bias
        match variant {
            QwenDecoderVariant::Qwen3 => 11,
            QwenDecoderVariant::Qwen2 => 12,
        }
    }

    /// Structural pins shared by every Qwen-shaped decoder family, including
    /// architecture ceilings so untrusted metadata cannot build unbounded
    /// descriptor sets.
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
        for (label, value, max) in [
            ("n_layers", self.n_layers, QWEN_DECODER_MAX_LAYERS),
            ("d_model", self.d_model, QWEN_DECODER_MAX_D_MODEL),
            ("n_heads", self.n_heads, QWEN_DECODER_MAX_N_HEADS),
            ("n_kv_heads", self.n_kv_heads, QWEN_DECODER_MAX_N_HEADS),
            ("head_dim", self.head_dim, QWEN_DECODER_MAX_HEAD_DIM),
            ("ffn_dim", self.ffn_dim, QWEN_DECODER_MAX_FFN_DIM),
            ("vocab_size", self.vocab_size, QWEN_DECODER_MAX_VOCAB_SIZE),
        ] {
            if value > max {
                return Err(format!(
                    "qwen decoder {label} ({value}) exceeds architecture ceiling {max}"
                ));
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

    /// Fail closed before allocating the full descriptor set when a geometry
    /// would construct more obligations than the global ceiling.
    fn tensor_obligation_count(
        self,
        variant: QwenDecoderVariant,
        tail_tensor_count: usize,
    ) -> Result<usize, String> {
        self.validate_basic()?;
        let per_layer = Self::layer_tensor_count(variant);
        let layer_total = self
            .n_layers
            .checked_mul(per_layer)
            .ok_or_else(|| "qwen decoder layer obligation count overflows".to_string())?;
        let total = layer_total
            .checked_add(tail_tensor_count)
            .ok_or_else(|| "qwen decoder total obligation count overflows".to_string())?;
        if total > QWEN_DECODER_MAX_TENSOR_OBLIGATIONS {
            return Err(format!(
                "qwen decoder tensor obligations ({total}) exceed ceiling {QWEN_DECODER_MAX_TENSOR_OBLIGATIONS}"
            ));
        }
        Ok(total)
    }
}

/// Closed set of Qwen-shaped decoder variants.
///
/// Prefer this over independent bool flags: only these two combinations are
/// valid. Invalid pairs such as qk_norm+qkv_bias together cannot be expressed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum QwenDecoderVariant {
    /// Qwen3-class: per-head Q/K RMSNorm, no QKV bias.
    Qwen3,
    /// Qwen2-class: QKV projection bias, no QK-norm (MiMo backbone / FireRed2-LLM).
    Qwen2,
}

impl QwenDecoderVariant {
    const fn qk_norm(self) -> bool {
        matches!(self, Self::Qwen3)
    }

    const fn qkv_bias(self) -> bool {
        matches!(self, Self::Qwen2)
    }

    const fn options(self) -> QwenDecoderContractOptions {
        QwenDecoderContractOptions {
            qk_norm: self.qk_norm(),
            qkv_bias: self.qkv_bias(),
        }
    }
}

/// Private implementation projection of [`QwenDecoderVariant`].
///
/// Fields are private so callers cannot assemble illegal qk_norm/qkv_bias
/// pairs. Construct only via [`QwenDecoderVariant::options`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct QwenDecoderContractOptions {
    qk_norm: bool,
    qkv_bias: bool,
}

impl QwenDecoderContractOptions {
    const fn qk_norm(self) -> bool {
        self.qk_norm
    }

    const fn qkv_bias(self) -> bool {
        self.qkv_bias
    }
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

/// Owned decoder profile for one Qwen-shaped family adapter.
///
/// Binds variant, layer-name provider, and tail names in one value. Not a
/// family registry. Production code should bind this with geometry into
/// [`QwenDecoderContract`] once and pass that value downstream.
#[derive(Debug, Clone, Copy)]
pub(crate) struct QwenFamilyDecoderProfile {
    variant: QwenDecoderVariant,
    names_for_layer: fn(usize) -> QwenFamilyLlmLayerTensorNames,
    tail: QwenDecoderTailTensorNames<'static>,
}

impl QwenFamilyDecoderProfile {
    pub(crate) const fn new(
        variant: QwenDecoderVariant,
        names_for_layer: fn(usize) -> QwenFamilyLlmLayerTensorNames,
        tail: QwenDecoderTailTensorNames<'static>,
    ) -> Self {
        Self {
            variant,
            names_for_layer,
            tail,
        }
    }

    const fn variant(self) -> QwenDecoderVariant {
        self.variant
    }

    const fn options(self) -> QwenDecoderContractOptions {
        self.variant.options()
    }

    const fn names_for_layer(self) -> fn(usize) -> QwenFamilyLlmLayerTensorNames {
        self.names_for_layer
    }

    const fn tail(self) -> QwenDecoderTailTensorNames<'static> {
        self.tail
    }

    /// Descriptor count for the tail half (norm + embd [+ optional logits]).
    const fn tail_tensor_count(self) -> usize {
        if self.tail.output_weight.is_some() {
            3
        } else {
            2
        }
    }
}

/// Geometry + profile bound into one contract value.
///
/// Fields are private: construct only via [`Self::bind`]. Production planner,
/// tail loader, admission descriptors, and host quotes must take this value
/// (or accessors on it) — not separately-threaded geometry/options/names/tail.
#[derive(Debug, Clone, Copy)]
pub(crate) struct QwenDecoderContract {
    geometry: QwenDecoderContractGeometry,
    profile: QwenFamilyDecoderProfile,
}

impl QwenDecoderContract {
    pub(crate) fn bind(
        geometry: QwenDecoderContractGeometry,
        profile: QwenFamilyDecoderProfile,
    ) -> Result<Self, String> {
        geometry.validate_basic()?;
        geometry.tensor_obligation_count(profile.variant(), profile.tail_tensor_count())?;
        validate_tail_names(profile.tail())?;
        for layer_index in 0..geometry.n_layers {
            validate_layer_names(profile.variant(), &(profile.names_for_layer())(layer_index))?;
        }
        Ok(Self { geometry, profile })
    }

    pub(super) const fn geometry(&self) -> QwenDecoderContractGeometry {
        self.geometry
    }

    const fn variant(&self) -> QwenDecoderVariant {
        self.profile.variant()
    }

    const fn options(&self) -> QwenDecoderContractOptions {
        self.profile.options()
    }

    const fn names_for_layer(&self) -> fn(usize) -> QwenFamilyLlmLayerTensorNames {
        self.profile.names_for_layer()
    }

    pub(super) const fn tail(&self) -> QwenDecoderTailTensorNames<'static> {
        self.profile.tail()
    }

    /// Exact number of decoder tensor obligations represented by this proof.
    pub(crate) fn tensor_obligation_count(&self) -> Result<usize, String> {
        self.geometry
            .tensor_obligation_count(self.variant(), self.profile.tail_tensor_count())
    }

    pub(crate) fn runtime_tensor_descriptors(
        &self,
    ) -> Result<Vec<TensorBindingDescriptor>, String> {
        runtime_tensor_descriptors(
            &self.geometry,
            self.options(),
            self.names_for_layer(),
            self.tail(),
            self.profile.tail_tensor_count(),
        )
    }

    pub(super) fn layer_projection(
        &self,
        layer_index: usize,
    ) -> Result<(QwenFamilyLlmLayerTensorNames, Vec<TensorBindingDescriptor>), String> {
        if layer_index >= self.geometry.n_layers {
            return Err(format!(
                "qwen decoder layer index {layer_index} is outside n_layers={}",
                self.geometry.n_layers
            ));
        }
        let names = (self.names_for_layer())(layer_index);
        let descriptors = layer_tensor_descriptors(&self.geometry, self.options(), &names)?;
        Ok((names, descriptors))
    }

    pub(super) fn tail_projection(
        &self,
    ) -> Result<
        (
            QwenDecoderTailTensorNames<'static>,
            Vec<TensorBindingDescriptor>,
        ),
        String,
    > {
        let tail = self.tail();
        let descriptors = tail_tensor_descriptors(&self.geometry, tail)?;
        Ok((tail, descriptors))
    }
}

fn validate_tail_names(tail: QwenDecoderTailTensorNames<'_>) -> Result<(), String> {
    for (label, name) in [
        ("output_norm", tail.output_norm),
        ("token_embd", tail.token_embd),
    ] {
        if name.is_empty() {
            return Err(format!("qwen decoder tail {label} name must not be empty"));
        }
    }
    if let Some(output_weight) = tail.output_weight {
        if output_weight.is_empty() {
            return Err("qwen decoder tail output_weight name must not be empty".to_string());
        }
        if output_weight == tail.output_norm || output_weight == tail.token_embd {
            return Err(
                "qwen decoder untied output_weight must name a distinct tensor".to_string(),
            );
        }
    }
    if tail.output_norm == tail.token_embd {
        return Err("qwen decoder output_norm and token_embd names must differ".to_string());
    }
    Ok(())
}

fn validate_layer_names(
    variant: QwenDecoderVariant,
    names: &QwenFamilyLlmLayerTensorNames,
) -> Result<(), String> {
    let required = [
        names.attn_norm_name.as_str(),
        names.attn_q_name.as_str(),
        names.attn_k_name.as_str(),
        names.attn_v_name.as_str(),
        names.attn_output_name.as_str(),
        names.ffn_norm_name.as_str(),
        names.ffn_gate_name.as_str(),
        names.ffn_up_name.as_str(),
        names.ffn_down_name.as_str(),
    ];
    if required.iter().any(|name| name.is_empty()) {
        return Err("qwen decoder layer tensor names must not be empty".to_string());
    }

    let optional = [
        names.q_norm_name.as_deref(),
        names.k_norm_name.as_deref(),
        names.q_bias_name.as_deref(),
        names.k_bias_name.as_deref(),
        names.v_bias_name.as_deref(),
    ];
    let expected = match variant {
        QwenDecoderVariant::Qwen3 => [true, true, false, false, false],
        QwenDecoderVariant::Qwen2 => [false, false, true, true, true],
    };
    for ((name, expected), label) in optional
        .into_iter()
        .zip(expected)
        .zip(["q_norm", "k_norm", "q_bias", "k_bias", "v_bias"])
    {
        if name.is_some() != expected {
            return Err(format!(
                "qwen decoder {:?} requires {label} name presence={expected}",
                variant
            ));
        }
        if name.is_some_and(str::is_empty) {
            return Err(format!("qwen decoder {label} name must not be empty"));
        }
    }

    let active: Vec<&str> = required
        .into_iter()
        .chain(optional.into_iter().flatten())
        .collect();
    for (index, name) in active.iter().enumerate() {
        if active[index + 1..].contains(name) {
            return Err(format!(
                "qwen decoder layer tensor name '{name}' is duplicated"
            ));
        }
    }
    Ok(())
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
fn layer_tensor_descriptors(
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

    if options.qk_norm() {
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

    if options.qkv_bias() {
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
fn tail_tensor_descriptors(
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
fn runtime_tensor_descriptors(
    geometry: &QwenDecoderContractGeometry,
    options: QwenDecoderContractOptions,
    mut names_for_layer: impl FnMut(usize) -> QwenFamilyLlmLayerTensorNames,
    tail: QwenDecoderTailTensorNames<'_>,
    tail_tensor_count: usize,
) -> Result<Vec<TensorBindingDescriptor>, String> {
    geometry.tensor_obligation_count(
        if options.qk_norm() {
            QwenDecoderVariant::Qwen3
        } else {
            QwenDecoderVariant::Qwen2
        },
        tail_tensor_count,
    )?;
    let mut descriptors = Vec::new();
    for layer in 0..geometry.n_layers {
        descriptors.extend(layer_tensor_descriptors(
            geometry,
            options,
            &names_for_layer(layer),
        )?);
    }
    descriptors.extend(tail_tensor_descriptors(geometry, tail)?);
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

    fn bind_qwen3(tail: QwenDecoderTailTensorNames<'static>) -> QwenDecoderContract {
        QwenDecoderContract::bind(
            qwen3_geometry(),
            QwenFamilyDecoderProfile::new(QwenDecoderVariant::Qwen3, qwen3_layer_names, tail),
        )
        .expect("bind qwen3 contract")
    }

    fn bind_qwen2(tail: QwenDecoderTailTensorNames<'static>) -> QwenDecoderContract {
        QwenDecoderContract::bind(
            qwen2_geometry(),
            QwenFamilyDecoderProfile::new(QwenDecoderVariant::Qwen2, qwen2_layer_names, tail),
        )
        .expect("bind qwen2 contract")
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
        let contract = bind_qwen3(QwenDecoderTailTensorNames {
            output_norm: "output_norm.weight",
            output_weight: Some("output.weight"),
            token_embd: "token_embd.weight",
        });
        let (_, layer) = contract.layer_projection(0).expect("qwen3 layer");
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
        let contract = bind_qwen2(QwenDecoderTailTensorNames {
            output_norm: "llm.out_norm.weight",
            output_weight: Some("llm.lm_head.weight"),
            token_embd: "llm.tok_emb.weight",
        });
        let (_, layer) = contract.layer_projection(0).expect("qwen2 layer");
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
        let contract = bind_qwen3(QwenDecoderTailTensorNames {
            output_norm: "output_norm.weight",
            output_weight: None,
            token_embd: "token_embd.weight",
        });
        let descriptors = contract.runtime_tensor_descriptors().expect("tied decoder");
        // 2 layers * 11 + norm + embd = 24
        assert_eq!(descriptors.len(), 24);
        assert!(descriptors.iter().all(|d| d.tensor_name != "output.weight"));
    }

    #[test]
    fn full_decoder_with_separate_logits_includes_output_weight() {
        let contract = bind_qwen2(QwenDecoderTailTensorNames {
            output_norm: "llm.out_norm.weight",
            output_weight: Some("llm.lm_head.weight"),
            token_embd: "llm.tok_emb.weight",
        });
        let descriptors = contract
            .runtime_tensor_descriptors()
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
        let contract = bind_qwen3(QwenDecoderTailTensorNames {
            output_norm: "output_norm.weight",
            output_weight: Some("output.weight"),
            token_embd: "token_embd.weight",
        });
        let descriptors = contract.runtime_tensor_descriptors().expect("descriptors");
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
        // Correct ggml [d_model, q_dim] = [16, 16]; force a deliberate wrong pair
        // that EitherDims would have accepted for a non-square case by using kv.
        let g_rect = QwenDecoderContractGeometry {
            n_kv_heads: 2,
            ..qwen3_geometry()
        };
        let contract = QwenDecoderContract::bind(
            g_rect,
            QwenFamilyDecoderProfile::new(
                QwenDecoderVariant::Qwen3,
                qwen3_layer_names,
                QwenDecoderTailTensorNames {
                    output_norm: "output_norm.weight",
                    output_weight: Some("output.weight"),
                    token_embd: "token_embd.weight",
                },
            ),
        )
        .expect("rect contract");
        let (_, mut rect) = contract.layer_projection(0).expect("rect layer");
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
        let (_, good) = contract.layer_projection(0).expect("good");
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
    }

    #[test]
    fn qwen3_contract_rejects_missing_qk_norm_name() {
        let g = qwen3_geometry();
        fn missing_q_norm(layer: usize) -> QwenFamilyLlmLayerTensorNames {
            let mut names = qwen3_layer_names(layer);
            names.q_norm_name = None;
            names
        }
        let err = QwenDecoderContract::bind(
            g,
            QwenFamilyDecoderProfile::new(
                QwenDecoderVariant::Qwen3,
                missing_q_norm,
                QwenDecoderTailTensorNames {
                    output_norm: "output_norm.weight",
                    output_weight: Some("output.weight"),
                    token_embd: "token_embd.weight",
                },
            ),
        )
        .expect_err("missing q_norm name must fail at bind");
        assert!(err.contains("q_norm"), "{err}");
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

    #[test]
    fn rejects_geometry_above_architecture_ceilings() {
        let over_layers = QwenDecoderContractGeometry {
            n_layers: QWEN_DECODER_MAX_LAYERS + 1,
            ..qwen3_geometry()
        };
        let err = over_layers.validate_basic().expect_err("layers ceiling");
        assert!(err.contains("n_layers") && err.contains("ceiling"), "{err}");

        let over_vocab = QwenDecoderContractGeometry {
            vocab_size: QWEN_DECODER_MAX_VOCAB_SIZE + 1,
            ..qwen3_geometry()
        };
        assert!(over_vocab.validate_basic().is_err());
    }

    #[test]
    fn rejects_unbounded_obligation_count_before_allocating() {
        // Even within per-field ceilings, refuse a combination that would
        // construct more descriptors than the global obligation budget.
        let g = QwenDecoderContractGeometry {
            n_layers: QWEN_DECODER_MAX_LAYERS,
            d_model: 16,
            n_heads: 4,
            n_kv_heads: 2,
            head_dim: 4,
            ffn_dim: 32,
            vocab_size: 64,
        };
        // Force overflow by asking the internal proof calculator for an
        // enormous tail count. No descriptor allocation may occur first.
        let err = g
            .tensor_obligation_count(QwenDecoderVariant::Qwen3, usize::MAX / 2)
            .expect_err("obligation budget must fail closed");
        assert!(
            err.contains("overflow") || err.contains("ceiling") || err.contains("obligations"),
            "{err}"
        );
    }

    /// Prove variant + names + tail live on one profile: admission via
    /// [`QwenDecoderContract::runtime_tensor_descriptors`] and the layer contract
    /// used by whole-decoder planning both read the same bound value. Flipping
    /// only the variant flips success/failure at both sites together.
    #[test]
    fn contract_bind_rejects_profile_variant_name_mismatch() {
        let g = qwen3_geometry();
        let tail = QwenDecoderTailTensorNames {
            output_norm: "output_norm.weight",
            output_weight: Some("output.weight"),
            token_embd: "token_embd.weight",
        };
        let matched =
            QwenFamilyDecoderProfile::new(QwenDecoderVariant::Qwen3, qwen3_layer_names, tail);
        let mismatched = QwenFamilyDecoderProfile::new(
            // Qwen2 variant demands bias name slots the Qwen3 name provider omits.
            QwenDecoderVariant::Qwen2,
            qwen3_layer_names,
            tail,
        );

        let admission_ok = QwenDecoderContract::bind(g, matched)
            .and_then(|contract| contract.runtime_tensor_descriptors());
        assert!(admission_ok.is_ok(), "{admission_ok:?}");

        let admission_err = QwenDecoderContract::bind(g, mismatched)
            .expect_err("mismatched profile variant must fail at bind");
        assert!(
            admission_err.contains("q_norm") || admission_err.contains("bias"),
            "{admission_err}"
        );
    }

    /// Production planner single-source gate: `QwenWholeDecoderPlan::for_qwen_family`
    /// consumes only a bound [`QwenDecoderContract`]. A matched profile plans; a
    /// mismatched variant fails at bind (before any pack I/O). This is the
    /// production call shape FunASR/MOSS/MiMo/FireRed/Qwen3-ASR use — not a
    /// parallel split-args path.
    #[test]
    fn production_planner_consumes_only_bound_contract() {
        use crate::models::qwen::QwenWholeDecoderPlan;
        use crate::testing::{TinyGgufFixtureSpec, write_tiny_gguf_runtime_source};
        use std::collections::BTreeMap;

        let g = qwen3_geometry();
        let tail = QwenDecoderTailTensorNames {
            output_norm: "output_norm.weight",
            output_weight: Some("output.weight"),
            token_embd: "token_embd.weight",
        };
        let matched =
            QwenFamilyDecoderProfile::new(QwenDecoderVariant::Qwen3, qwen3_layer_names, tail);
        let mismatched =
            QwenFamilyDecoderProfile::new(QwenDecoderVariant::Qwen2, qwen3_layer_names, tail);

        // Matched profile binds and expands; mismatched Qwen2 variant against
        // Qwen3 names fails while constructing the proof, before pack I/O.
        let contract = QwenDecoderContract::bind(g, matched).expect("matched bind");
        assert_eq!(contract.variant(), QwenDecoderVariant::Qwen3);
        let descriptors = contract
            .runtime_tensor_descriptors()
            .expect("matched descriptors");
        let mismatch_err = QwenDecoderContract::bind(g, mismatched)
            .expect_err("mismatched variant must fail contract bind");
        assert!(
            mismatch_err.contains("q_norm") || mismatch_err.contains("bias"),
            "unexpected mismatch error: {mismatch_err}"
        );

        // Minimal fixture covering the full decoder descriptor set so the
        // production planner can expand every layer without missing-tensor noise.
        let temp = tempfile::tempdir().expect("temp");
        let path = temp.path().join("planner-contract.oasr");
        let shapes: BTreeMap<String, Vec<u64>> =
            crate::models::tensor_binding::project_fixture_tensors(&descriptors)
                .into_iter()
                .collect();
        // project_fixture_tensors already encodes ExactDims/VectorLen; just write.
        let mut spec = TinyGgufFixtureSpec::new(BTreeMap::new());
        for (name, dims) in shapes {
            spec = spec.with_tensor_shape(name, dims);
        }
        write_tiny_gguf_runtime_source(&path, &spec).expect("write fixture");
        let reader = crate::ggml_runtime::GgufTensorDataReader::from_path(&path).expect("reader");

        let plan = QwenWholeDecoderPlan::for_qwen_family(&reader, &contract)
            .expect("production planner must accept bound matched contract");
        assert_eq!(plan.layer_count(), g.n_layers);
    }
}
