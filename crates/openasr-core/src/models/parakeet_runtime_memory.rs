//! Count-only and post-build SystemMemory topology shared by Parakeet CTC/TDT.
//!
//! FastConformer loading deliberately materializes only one layer's bound
//! projections at a time, drops those f32 payloads, and retains the small
//! arena-upload tensors until graph construction. The peak oracle below
//! mirrors that order; summing every dequantized tensor in the pack would be a
//! multi-gigabyte false upper bound and would defeat admission on small hosts.

use crate::GgufTensorIndex;
use crate::ggml_runtime::GgufMetadata;
use crate::models::fastconformer::{FastConformerLayerWeights, NamedTensor};
use crate::models::system_memory_owner::{SystemMemoryCapacity, SystemMemoryOwnerError};

const LAYER_SUFFIXES: &[&str] = &[
    "ff1.norm.weight",
    "ff1.norm.bias",
    "ff1.up.weight",
    "ff1.up.bias",
    "ff1.down.weight",
    "ff1.down.bias",
    "attn.norm.weight",
    "attn.norm.bias",
    "attn.q.weight",
    "attn.q.bias",
    "attn.k.weight",
    "attn.k.bias",
    "attn.v.weight",
    "attn.v.bias",
    "attn.out.weight",
    "attn.out.bias",
    "attn.pos.weight",
    "attn.pos_bias_u",
    "attn.pos_bias_v",
    "conv.norm.weight",
    "conv.norm.bias",
    "conv.pw1.weight",
    "conv.pw1.bias",
    "conv.dw.weight",
    "conv.dw.bias",
    "conv.pw2.weight",
    "conv.pw2.bias",
    "ff2.norm.weight",
    "ff2.norm.bias",
    "ff2.up.weight",
    "ff2.up.bias",
    "ff2.down.weight",
    "ff2.down.bias",
    "out.norm.weight",
    "out.norm.bias",
];

const BOUND_LAYER_SUFFIXES: &[&str] = &[
    "ff1.up.weight",
    "ff1.down.weight",
    "attn.q.weight",
    "attn.k.weight",
    "attn.v.weight",
    "attn.out.weight",
    "attn.pos.weight",
    "conv.pw1.weight",
    "conv.pw2.weight",
    "ff2.up.weight",
    "ff2.down.weight",
];

const SYNTHETIC_BIAS_SUFFIXES: &[&str] = &[
    "ff1.up.bias",
    "ff1.down.bias",
    "attn.q.bias",
    "attn.k.bias",
    "attn.v.bias",
    "attn.out.bias",
    "conv.pw1.bias",
    "conv.dw.bias",
    "conv.pw2.bias",
    "ff2.up.bias",
    "ff2.down.bias",
];

const BATCH_NORM_SUFFIXES: &[&str] = &[
    "conv.bn.weight",
    "conv.bn.bias",
    "conv.bn.mean",
    "conv.bn.var",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FastConformerSystemMemoryPlan {
    /// Heap capacity retained by the temporary encoder-weight bundle after
    /// mmap-bound f32 payloads have been released.
    pub(crate) stable_weight_bytes: u64,
    /// Peak of encoder loading or graph construction, excluding tokenizer and
    /// family-specific host decoder/joint state.
    pub(crate) build_peak_bytes: u64,
    /// Rust descriptor Vec retained by the completed graph. Backend buffers
    /// and ggml contexts are admitted by their own physical-domain owners.
    pub(crate) graph_retained_bytes: u64,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct FastConformerMemoryTopology<'a> {
    pub(crate) n_layers: usize,
    pub(crate) hidden_size: usize,
    pub(crate) ffn_dim: usize,
    pub(crate) checkpoint_has_projection_biases: bool,
    pub(crate) bound_tail_weight: &'a str,
    pub(crate) retained_tail_bias: &'a str,
}

pub(crate) fn plan_fastconformer_system_memory(
    tensor_index: &GgufTensorIndex,
    topology: FastConformerMemoryTopology<'_>,
) -> Result<FastConformerSystemMemoryPlan, SystemMemoryOwnerError> {
    let layer_descriptors = element_bytes::<FastConformerLayerWeights>(
        topology.n_layers,
        "parakeet encoder layer descriptors",
    )?;
    let graph_retained_bytes = element_bytes::<crate::models::fastconformer::graph::LayerArena>(
        topology.n_layers,
        "parakeet graph layer descriptors",
    )?;

    let mut stable = layer_descriptors;
    let mut peak = 0_u64;

    let mut subsampling_count = 0_usize;
    let mut subsampling_stable = 0_u64;
    let mut subsampling_bound_values = 0_u64;
    for layer in [0_usize, 2, 3, 5, 6] {
        for kind in ["weight", "bias"] {
            let name = format!("enc.sub.layers.{layer}.{kind}");
            if tensor_index.get(&name).is_some() {
                subsampling_count += 1;
                subsampling_stable = add(
                    subsampling_stable,
                    named_tensor_quote_bytes(tensor_index, &name, true)?,
                    "parakeet subsampling retained bytes",
                )?;
            }
        }
    }
    for (name, retain_values) in [
        ("enc.sub.linear.weight", false),
        ("enc.sub.linear.bias", true),
    ] {
        subsampling_count += 1;
        subsampling_stable = add(
            subsampling_stable,
            named_tensor_quote_bytes(tensor_index, name, retain_values)?,
            "parakeet subsampling retained bytes",
        )?;
        if !retain_values {
            subsampling_bound_values = tensor_f32_bytes(tensor_index, name)?;
        }
    }
    // The loader starts from Vec::new(); its geometric growth can leave spare
    // descriptor capacity. next_power_of_two is a deterministic upper bound
    // for the current push-only implementation.
    let subsampling_descriptor_capacity = subsampling_count
        .checked_next_power_of_two()
        .ok_or_else(|| capacity_error("parakeet subsampling descriptor capacity overflowed"))?;
    subsampling_stable = add(
        subsampling_stable,
        element_bytes::<NamedTensor>(
            subsampling_descriptor_capacity,
            "parakeet subsampling descriptors",
        )?,
        "parakeet subsampling retained bytes",
    )?;
    stable = add(
        stable,
        subsampling_stable,
        "parakeet encoder retained bytes",
    )?;
    peak = peak.max(add(
        stable,
        subsampling_bound_values,
        "parakeet subsampling construction peak",
    )?);

    for layer in 0..topology.n_layers {
        let mut layer_stable = 0_u64;
        let mut layer_bound_values = 0_u64;
        for suffix in LAYER_SUFFIXES {
            let name = format!("enc.blk.{layer}.{suffix}");
            let bound = BOUND_LAYER_SUFFIXES.contains(suffix);
            if tensor_index.get(&name).is_some() {
                layer_stable = add(
                    layer_stable,
                    named_tensor_quote_bytes(tensor_index, &name, !bound)?,
                    "parakeet layer retained bytes",
                )?;
                if bound {
                    layer_bound_values = add(
                        layer_bound_values,
                        tensor_f32_bytes(tensor_index, &name)?,
                        "parakeet layer bound construction bytes",
                    )?;
                }
            } else if !topology.checkpoint_has_projection_biases
                && SYNTHETIC_BIAS_SUFFIXES.contains(suffix)
            {
                let elements =
                    synthetic_bias_elements(suffix, topology.hidden_size, topology.ffn_dim);
                layer_stable = add(
                    layer_stable,
                    synthetic_named_tensor_bytes(&name, elements)?,
                    "parakeet synthesized layer bias bytes",
                )?;
            } else {
                return Err(capacity_error(format!(
                    "required parakeet tensor '{name}' is missing from GGUF index"
                )));
            }
        }

        let mut fold_temporary =
            element_bytes::<f32>(topology.hidden_size, "parakeet batchnorm scale temporary")?;
        for suffix in BATCH_NORM_SUFFIXES {
            let name = format!("enc.blk.{layer}.{suffix}");
            fold_temporary = add(
                fold_temporary,
                named_tensor_quote_bytes(tensor_index, &name, true)?,
                "parakeet batchnorm fold temporary bytes",
            )?;
        }
        let current_layer_peak = add(
            add(
                add(stable, layer_stable, "parakeet layer construction peak")?,
                layer_bound_values,
                "parakeet layer construction peak",
            )?,
            fold_temporary,
            "parakeet layer construction peak",
        )?;
        peak = peak.max(current_layer_peak);
        stable = add(stable, layer_stable, "parakeet encoder retained bytes")?;
    }

    stable = add(
        stable,
        named_tensor_quote_bytes(tensor_index, topology.bound_tail_weight, false)?,
        "parakeet encoder tail retained bytes",
    )?;
    stable = add(
        stable,
        named_tensor_quote_bytes(tensor_index, topology.retained_tail_bias, true)?,
        "parakeet encoder tail retained bytes",
    )?;
    peak = peak.max(add(
        stable,
        tensor_f32_bytes(tensor_index, topology.bound_tail_weight)?,
        "parakeet encoder tail construction peak",
    )?);
    peak = peak.max(add(
        stable,
        graph_retained_bytes,
        "parakeet graph construction peak",
    )?);

    Ok(FastConformerSystemMemoryPlan {
        stable_weight_bytes: stable,
        build_peak_bytes: peak,
        graph_retained_bytes,
    })
}

pub(crate) fn tokenizer_quote_bytes(
    metadata: &GgufMetadata,
    family: &str,
) -> Result<u64, SystemMemoryOwnerError> {
    crate::models::runtime_memory::tokenizer_btree_quote_bytes(metadata, family)
}

pub(crate) fn named_tensor_quote_bytes(
    tensor_index: &GgufTensorIndex,
    name: &str,
    retain_values: bool,
) -> Result<u64, SystemMemoryOwnerError> {
    crate::models::runtime_memory::named_f32_tensor_quote_bytes(
        tensor_index,
        name,
        retain_values,
        "parakeet",
    )
}

pub(crate) fn tensor_f32_bytes(
    tensor_index: &GgufTensorIndex,
    name: &str,
) -> Result<u64, SystemMemoryOwnerError> {
    crate::models::runtime_memory::tensor_f32_bytes(tensor_index, name, "parakeet")
}

pub(crate) fn element_bytes<T>(count: usize, label: &str) -> Result<u64, SystemMemoryOwnerError> {
    crate::models::runtime_memory::element_bytes::<T>(count, "parakeet", label)
}

pub(crate) fn checked_sum(
    values: impl IntoIterator<Item = u64>,
    label: &str,
) -> Result<u64, SystemMemoryOwnerError> {
    crate::models::runtime_memory::checked_sum(values, "parakeet", label)
}

pub(crate) fn add_named_tensor_capacity(
    tensor: &NamedTensor,
    bytes: &mut SystemMemoryCapacity,
    label: &str,
) -> Result<(), String> {
    bytes.add_string(&tensor.name, &format!("{label} name"))?;
    bytes.add_vec(&tensor.dims, &format!("{label} dims"))?;
    bytes.add_vec(&tensor.values, &format!("{label} values"))
}

pub(crate) fn fastconformer_weights_retained_bytes(
    subsampling: &Vec<NamedTensor>,
    layers: &Vec<FastConformerLayerWeights>,
    tail: &[&NamedTensor],
) -> Result<u64, String> {
    let mut bytes = SystemMemoryCapacity::default();
    bytes.add_vec(subsampling, "parakeet subsampling descriptors")?;
    for tensor in subsampling {
        add_named_tensor_capacity(tensor, &mut bytes, "parakeet subsampling tensor")?;
    }
    bytes.add_vec(layers, "parakeet encoder layer descriptors")?;
    for layer in layers {
        for tensor in fastconformer_layer_tensors(layer) {
            add_named_tensor_capacity(tensor, &mut bytes, "parakeet encoder layer tensor")?;
        }
    }
    for tensor in tail {
        add_named_tensor_capacity(tensor, &mut bytes, "parakeet encoder tail tensor")?;
    }
    Ok(bytes.finish())
}

pub(crate) fn graph_retained_bytes(
    core: &crate::models::fastconformer::FastConformerEncoderCore,
) -> Result<u64, String> {
    let mut bytes = SystemMemoryCapacity::default();
    bytes.add_vec(&core.layers, "parakeet graph layer descriptors")?;
    Ok(bytes.finish())
}

fn synthetic_bias_elements(suffix: &str, hidden_size: usize, ffn_dim: usize) -> usize {
    match suffix {
        "ff1.up.bias" | "ff2.up.bias" => ffn_dim,
        "conv.pw1.bias" => hidden_size.saturating_mul(2),
        _ => hidden_size,
    }
}

fn synthetic_named_tensor_bytes(
    name: &str,
    elements: usize,
) -> Result<u64, SystemMemoryOwnerError> {
    checked_sum(
        [
            u64::try_from(name.len())
                .map_err(|_| capacity_error("synthetic tensor name does not fit u64"))?,
            element_bytes::<usize>(1, "synthetic tensor dims")?,
            element_bytes::<f32>(elements, "synthetic tensor values")?,
        ],
        "synthetic parakeet tensor bytes",
    )
}

fn fastconformer_layer_tensors(layer: &FastConformerLayerWeights) -> [&NamedTensor; 35] {
    [
        &layer.ff1_norm_weight,
        &layer.ff1_norm_bias,
        &layer.ff1_up_weight,
        &layer.ff1_up_bias,
        &layer.ff1_down_weight,
        &layer.ff1_down_bias,
        &layer.attn_norm_weight,
        &layer.attn_norm_bias,
        &layer.attn_q_weight,
        &layer.attn_q_bias,
        &layer.attn_k_weight,
        &layer.attn_k_bias,
        &layer.attn_v_weight,
        &layer.attn_v_bias,
        &layer.attn_out_weight,
        &layer.attn_out_bias,
        &layer.attn_pos_weight,
        &layer.attn_pos_bias_u,
        &layer.attn_pos_bias_v,
        &layer.conv_norm_weight,
        &layer.conv_norm_bias,
        &layer.conv_pw1_weight,
        &layer.conv_pw1_bias,
        &layer.conv_dw_weight,
        &layer.conv_dw_bias,
        &layer.conv_pw2_weight,
        &layer.conv_pw2_bias,
        &layer.ff2_norm_weight,
        &layer.ff2_norm_bias,
        &layer.ff2_up_weight,
        &layer.ff2_up_bias,
        &layer.ff2_down_weight,
        &layer.ff2_down_bias,
        &layer.out_norm_weight,
        &layer.out_norm_bias,
    ]
}

fn add(lhs: u64, rhs: u64, label: &str) -> Result<u64, SystemMemoryOwnerError> {
    lhs.checked_add(rhs)
        .ok_or_else(|| capacity_error(format!("{label} overflowed")))
}

fn capacity_error(reason: impl Into<String>) -> SystemMemoryOwnerError {
    SystemMemoryOwnerError::capacity_failure("parakeet_runtime_memory", reason)
}
