//! Shared FastConformer admission tensor contract: the single enumeration of
//! every encoder tensor the dw-striding subsampling prelude + conformer stack
//! consumes, parameterized by the family geometry and the checkpoint's
//! bias-presence. Both `parakeet_ctc` (biases on disk) and `parakeet_tdt`
//! (bias-free checkpoint, zero-bias synthesis) build their runtime tensor
//! contract on this list plus their own tail, so the encoder half of the
//! admission validator cannot drift from the shared loader/graph either way.
//!
//! Shape checks mirror the loader's strictness: the shared loader reads every
//! tensor generically and the graph reshapes by element layout, so norms and
//! biases pin their exact metadata-derived length, 2-D projections pin their
//! two extents in either stored orientation, and the depthwise kernel pins its
//! exact `[conv_kernel, 1, hidden]` ggml layout (the only tensor with a single
//! valid orientation, matching the sensevoice FSMN precedent).
//!
//! Overflow safety: the geometry arrives already bounded by each family's
//! metadata parser (architecture ceilings checked fail-closed at parse time),
//! so the few derived extents computed here use saturating arithmetic as
//! defense in depth -- a hypothetical saturation produces a requirement no
//! pack tensor can satisfy, which stays fail-closed at validation instead of
//! wrapping into an admitting shape.

use crate::models::tensor_binding::{TensorBindingDescriptor, TensorBindingDescriptorRequirement};

/// The metadata-derived geometry one FastConformer encoder's tensor contract
/// is shaped for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FastConformerContractGeometry {
    pub n_layers: usize,
    pub hidden_size: usize,
    pub ffn_dim: usize,
    pub conv_kernel: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    /// Output channels of every dw-striding subsampling conv stage (the
    /// metadata's `subsampling_channels`).
    pub subsampling_channels: usize,
    /// `true` when the checkpoint ships every attention/conv/FFN bias tensor
    /// (parakeet-ctc); `false` when the loader synthesizes zero biases and the
    /// pack must NOT carry them (parakeet-tdt-0.6b-v3's bias-free conversion).
    pub bias_present: bool,
}

fn descriptor(
    tensor_name: String,
    requirement: TensorBindingDescriptorRequirement,
    reason: &str,
) -> TensorBindingDescriptor {
    TensorBindingDescriptor {
        tensor_name,
        requirement,
        reason: reason.to_string(),
    }
}

/// The dw-striding subsampling prelude tensors (`enc.sub.*`): five strided
/// conv stages (layers 0/2/3/5/6) + the flattening linear. The shared graph
/// consumes every one of them unconditionally, so the contract requires the
/// full set even though the weight loader probes per-tensor presence.
pub(crate) fn fastconformer_subsampling_tensor_descriptors(
    geometry: &FastConformerContractGeometry,
) -> Vec<TensorBindingDescriptor> {
    let hidden = geometry.hidden_size;
    let channels = geometry.subsampling_channels;
    let mut descriptors = Vec::new();
    for sub_layer in [0usize, 2, 3, 5, 6] {
        descriptors.push(descriptor(
            format!("enc.sub.layers.{sub_layer}.weight"),
            TensorBindingDescriptorRequirement::RankAtLeastWithDimAt {
                min_rank: 4,
                axis: 3,
                dim: channels,
            },
            "subsampling conv kernel must be rank 4 with the declared channel extent",
        ));
        descriptors.push(descriptor(
            format!("enc.sub.layers.{sub_layer}.bias"),
            TensorBindingDescriptorRequirement::VectorLen(channels),
            "subsampling conv bias must span the declared channel count",
        ));
    }
    descriptors.push(descriptor(
        "enc.sub.linear.weight".to_string(),
        TensorBindingDescriptorRequirement::Rank2WithDim(hidden),
        "subsampling linear must flatten the conv output into hidden_size",
    ));
    descriptors.push(descriptor(
        "enc.sub.linear.bias".to_string(),
        TensorBindingDescriptorRequirement::VectorLen(hidden),
        "subsampling linear bias must span hidden_size",
    ));
    descriptors
}

/// One conformer layer's runtime tensor bindings, shaped for `geometry`.
/// Norms and biases always load; the projection/conv/FFN biases load only when
/// `geometry.bias_present` (a bias-free checkpoint synthesizes zeros, so its
/// pack must not carry those tensors and the contract does not require them).
fn fastconformer_layer_tensor_descriptors(
    geometry: &FastConformerContractGeometry,
    layer: usize,
) -> Vec<TensorBindingDescriptor> {
    let hidden = geometry.hidden_size;
    let ffn = geometry.ffn_dim;
    let n = |suffix: &str| format!("enc.blk.{layer}.{suffix}");
    let mut descriptors = vec![
        descriptor(
            n("ff1.norm.weight"),
            TensorBindingDescriptorRequirement::VectorLen(hidden),
            "first FFN pre-norm gamma must span hidden_size",
        ),
        descriptor(
            n("ff1.norm.bias"),
            TensorBindingDescriptorRequirement::VectorLen(hidden),
            "first FFN pre-norm beta must span hidden_size",
        ),
        descriptor(
            n("ff1.up.weight"),
            TensorBindingDescriptorRequirement::Rank2EitherDims(hidden, ffn),
            "first FFN up projection must map hidden_size to ffn_dim",
        ),
    ];
    if geometry.bias_present {
        descriptors.push(descriptor(
            n("ff1.up.bias"),
            TensorBindingDescriptorRequirement::VectorLen(ffn),
            "first FFN up bias must span ffn_dim",
        ));
    }
    descriptors.push(descriptor(
        n("ff1.down.weight"),
        TensorBindingDescriptorRequirement::Rank2EitherDims(ffn, hidden),
        "first FFN down projection must map ffn_dim to hidden_size",
    ));
    if geometry.bias_present {
        descriptors.push(descriptor(
            n("ff1.down.bias"),
            TensorBindingDescriptorRequirement::VectorLen(hidden),
            "first FFN down bias must span hidden_size",
        ));
    }
    descriptors.extend([
        descriptor(
            n("attn.norm.weight"),
            TensorBindingDescriptorRequirement::VectorLen(hidden),
            "attention pre-norm gamma must span hidden_size",
        ),
        descriptor(
            n("attn.norm.bias"),
            TensorBindingDescriptorRequirement::VectorLen(hidden),
            "attention pre-norm beta must span hidden_size",
        ),
    ]);
    for projection in ["q", "k", "v"] {
        descriptors.push(descriptor(
            n(&format!("attn.{projection}.weight")),
            TensorBindingDescriptorRequirement::Rank2EitherDims(hidden, hidden),
            "attention projection must be hidden_size x hidden_size",
        ));
        if geometry.bias_present {
            descriptors.push(descriptor(
                n(&format!("attn.{projection}.bias")),
                TensorBindingDescriptorRequirement::VectorLen(hidden),
                "attention projection bias must span hidden_size",
            ));
        }
    }
    descriptors.push(descriptor(
        n("attn.out.weight"),
        TensorBindingDescriptorRequirement::Rank2EitherDims(hidden, hidden),
        "attention output projection must be hidden_size x hidden_size",
    ));
    if geometry.bias_present {
        descriptors.push(descriptor(
            n("attn.out.bias"),
            TensorBindingDescriptorRequirement::VectorLen(hidden),
            "attention output bias must span hidden_size",
        ));
    }
    descriptors.extend([
        descriptor(
            n("attn.pos.weight"),
            TensorBindingDescriptorRequirement::Rank2EitherDims(hidden, hidden),
            "relative position projection must be hidden_size x hidden_size",
        ),
        descriptor(
            n("attn.pos_bias_u"),
            TensorBindingDescriptorRequirement::Rank2EitherDims(
                geometry.head_dim,
                geometry.n_heads,
            ),
            "Transformer-XL pos_bias_u must be head_dim x n_heads",
        ),
        descriptor(
            n("attn.pos_bias_v"),
            TensorBindingDescriptorRequirement::Rank2EitherDims(
                geometry.head_dim,
                geometry.n_heads,
            ),
            "Transformer-XL pos_bias_v must be head_dim x n_heads",
        ),
        descriptor(
            n("conv.norm.weight"),
            TensorBindingDescriptorRequirement::VectorLen(hidden),
            "conv module pre-norm gamma must span hidden_size",
        ),
        descriptor(
            n("conv.norm.bias"),
            TensorBindingDescriptorRequirement::VectorLen(hidden),
            "conv module pre-norm beta must span hidden_size",
        ),
        descriptor(
            n("conv.pw1.weight"),
            TensorBindingDescriptorRequirement::Rank2EitherDims(hidden, hidden.saturating_mul(2)),
            "conv pointwise1 must map hidden_size to 2*hidden_size (GLU)",
        ),
    ]);
    if geometry.bias_present {
        descriptors.push(descriptor(
            n("conv.pw1.bias"),
            TensorBindingDescriptorRequirement::VectorLen(hidden.saturating_mul(2)),
            "conv pointwise1 bias must span 2*hidden_size",
        ));
    }
    descriptors.push(descriptor(
        n("conv.dw.weight"),
        TensorBindingDescriptorRequirement::ExactDims(vec![geometry.conv_kernel, 1, hidden]),
        "depthwise conv kernel must be [conv_kernel, 1, hidden_size] for the graph reshape",
    ));
    if geometry.bias_present {
        descriptors.push(descriptor(
            n("conv.dw.bias"),
            TensorBindingDescriptorRequirement::VectorLen(hidden),
            "depthwise conv bias must span hidden_size",
        ));
    }
    // The BatchNorm fold reads all four running statistics unconditionally and
    // folds them into the depthwise weight/bias at load.
    for (suffix, reason) in [
        (
            "conv.bn.weight",
            "conv BatchNorm gamma must span hidden_size",
        ),
        ("conv.bn.bias", "conv BatchNorm beta must span hidden_size"),
        (
            "conv.bn.mean",
            "conv BatchNorm running mean must span hidden_size",
        ),
        (
            "conv.bn.var",
            "conv BatchNorm running variance must span hidden_size",
        ),
    ] {
        descriptors.push(descriptor(
            n(suffix),
            TensorBindingDescriptorRequirement::VectorLen(hidden),
            reason,
        ));
    }
    descriptors.push(descriptor(
        n("conv.pw2.weight"),
        TensorBindingDescriptorRequirement::Rank2EitherDims(hidden, hidden),
        "conv pointwise2 must map hidden_size to hidden_size",
    ));
    if geometry.bias_present {
        descriptors.push(descriptor(
            n("conv.pw2.bias"),
            TensorBindingDescriptorRequirement::VectorLen(hidden),
            "conv pointwise2 bias must span hidden_size",
        ));
    }
    descriptors.extend([
        descriptor(
            n("ff2.norm.weight"),
            TensorBindingDescriptorRequirement::VectorLen(hidden),
            "second FFN pre-norm gamma must span hidden_size",
        ),
        descriptor(
            n("ff2.norm.bias"),
            TensorBindingDescriptorRequirement::VectorLen(hidden),
            "second FFN pre-norm beta must span hidden_size",
        ),
        descriptor(
            n("ff2.up.weight"),
            TensorBindingDescriptorRequirement::Rank2EitherDims(hidden, ffn),
            "second FFN up projection must map hidden_size to ffn_dim",
        ),
    ]);
    if geometry.bias_present {
        descriptors.push(descriptor(
            n("ff2.up.bias"),
            TensorBindingDescriptorRequirement::VectorLen(ffn),
            "second FFN up bias must span ffn_dim",
        ));
    }
    descriptors.push(descriptor(
        n("ff2.down.weight"),
        TensorBindingDescriptorRequirement::Rank2EitherDims(ffn, hidden),
        "second FFN down projection must map ffn_dim to hidden_size",
    ));
    if geometry.bias_present {
        descriptors.push(descriptor(
            n("ff2.down.bias"),
            TensorBindingDescriptorRequirement::VectorLen(hidden),
            "second FFN down bias must span hidden_size",
        ));
    }
    descriptors.extend([
        descriptor(
            n("out.norm.weight"),
            TensorBindingDescriptorRequirement::VectorLen(hidden),
            "block output post-norm gamma must span hidden_size",
        ),
        descriptor(
            n("out.norm.bias"),
            TensorBindingDescriptorRequirement::VectorLen(hidden),
            "block output post-norm beta must span hidden_size",
        ),
    ]);
    descriptors
}

/// The full shared-encoder tensor contract for a FastConformer pack: the
/// subsampling prelude plus every conformer layer. Family tails (CTC head /
/// joint encoder projection) are appended by each family's own contract.
pub(crate) fn fastconformer_encoder_tensor_descriptors(
    geometry: &FastConformerContractGeometry,
) -> Vec<TensorBindingDescriptor> {
    let mut descriptors = fastconformer_subsampling_tensor_descriptors(geometry);
    for layer in 0..geometry.n_layers {
        descriptors.extend(fastconformer_layer_tensor_descriptors(geometry, layer));
    }
    descriptors
}

/// The number of tensor obligations the shared encoder contributes for one
/// geometry, computed from the builders themselves so a family parser's
/// closed-form total-obligation count can never drift from the enumeration
/// it bounds.
pub(crate) fn fastconformer_encoder_descriptor_count(
    geometry: &FastConformerContractGeometry,
) -> usize {
    let subsampling = fastconformer_subsampling_tensor_descriptors(geometry).len();
    let per_layer = fastconformer_layer_tensor_descriptors(geometry, 0).len();
    subsampling.saturating_add(geometry.n_layers.saturating_mul(per_layer))
}
