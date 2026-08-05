//! Shared FastConformer admission tensor contract: the single enumeration of
//! every encoder tensor the dw-striding subsampling prelude + conformer stack
//! consumes, parameterized by the family geometry and the checkpoint's
//! bias-presence. Both `parakeet_ctc` (biases on disk) and `parakeet_tdt`
//! (bias-free checkpoint, zero-bias synthesis) build their runtime tensor
//! contract on this list plus their own tail, so the encoder half of the
//! admission validator cannot drift from the shared loader/graph either way.
//!
//! Shape checks mirror the loader's strictness and the graph's fixed layout:
//! norms and biases pin their exact metadata-derived length, every 2-D
//! projection that feeds `mul_mat` / a fixed graph layout pins its ordered
//! ggml `[in, out]` extents via `ExactDims` (matching the importer's HF
//! `[out, in]` -> ggml reverse), the conformer depthwise kernel pins its
//! exact `[conv_kernel, 1, hidden]` ggml layout, and the dw-striding
//! subsampling prelude pins every conv kernel to the ExactDims the shared
//! graph consumes (3x3 / depthwise / 1x1) plus the flatten width derived
//! from `n_mels` through the same `conv_out_dim` the graph uses.
//!
//! Overflow safety: the geometry arrives already bounded by each family's
//! metadata parser (architecture ceilings checked fail-closed at parse time),
//! so the few derived extents computed here use checked arithmetic and fall
//! back to a saturated requirement no pack tensor can satisfy -- a
//! hypothetical overflow stays fail-closed at validation instead of wrapping
//! into an admitting shape.

use crate::models::tensor_binding::{TensorBindingDescriptor, TensorBindingDescriptorRequirement};

use super::graph::{SUBSAMPLING_KERNEL, SUBSAMPLING_PADDING, SUBSAMPLING_STRIDE};

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
    /// Mel bins feeding the first subsampling conv. Three stride-2 stages
    /// shrink this to the flatten frequency the linear projects from; must
    /// match the family's metadata `n_mels` so the contract pins the same
    /// flatten width the graph builds.
    pub n_mels: usize,
    /// Output channels of every dw-striding subsampling conv stage (the
    /// metadata's `subsampling_channels`).
    pub subsampling_channels: usize,
    /// `true` when the checkpoint ships every attention/conv/FFN bias tensor
    /// (parakeet-ctc); `false` when the loader synthesizes zero biases and the
    /// pack must NOT carry them (parakeet-tdt-0.6b-v3's bias-free conversion).
    pub bias_present: bool,
}

/// Checked form of the graph's three-stage dw-striding frequency shrink:
/// `(input + 2*pad - kernel) / stride + 1` with pad=1, kernel=3, stride=2.
/// Returns `None` when the intermediate add/sub would wrap.
pub(crate) fn conv_out_dim_checked(input: usize) -> Option<usize> {
    let padded = input.checked_add(2usize.checked_mul(SUBSAMPLING_PADDING)?)?;
    let reduced = padded.checked_sub(SUBSAMPLING_KERNEL)?;
    Some(reduced / SUBSAMPLING_STRIDE + 1)
}

/// Frequency extent after the three fixed stride-2 dw-striding stages the
/// shared graph always runs. Same formula as `graph::build_conformer_stack`
/// (`conv_out_dim` thrice); checked so a pathological `n_mels` cannot wrap.
pub(crate) fn fastconformer_subsampled_freq(n_mels: usize) -> Option<usize> {
    let after0 = conv_out_dim_checked(n_mels)?;
    let after1 = conv_out_dim_checked(after0)?;
    conv_out_dim_checked(after1)
}

/// Flatten width the subsampling linear consumes: `channels * freq'` where
/// `freq'` is [`fastconformer_subsampled_freq`]. Checked multiply so an
/// oversize geometry fails closed at contract build rather than wrapping.
pub(crate) fn fastconformer_subsampling_flatten_width(
    channels: usize,
    n_mels: usize,
) -> Option<usize> {
    let freq = fastconformer_subsampled_freq(n_mels)?;
    channels.checked_mul(freq)
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

/// The dw-striding subsampling prelude tensors (`enc.sub.*`): five conv
/// stages (layers 0/2/3/5/6) + the flattening linear. Shapes are the exact
/// ggml layouts the shared graph consumes after the importer reverses HF
/// `[OC,IC,kh,kw]` → `[kw,kh,IC,OC]`:
/// - layer 0: ordinary 3×3, in=1 (mel), out=channels → `[3,3,1,channels]`
/// - layers 2/5: depthwise 3×3 → `[3,3,1,channels]` (ggml `conv_2d_dw`)
/// - layers 3/6: pointwise 1×1 → `[1,1,channels,channels]`
/// - linear: ordered ggml `[flatten, hidden]` (`mul_mat` weight layout) where
///   `flatten = channels * conv_out_dim³(n_mels)`
///
/// The shared graph consumes every one of them unconditionally, so the
/// contract requires the full set even though the weight loader probes
/// per-tensor presence.
pub(crate) fn fastconformer_subsampling_tensor_descriptors(
    geometry: &FastConformerContractGeometry,
) -> Vec<TensorBindingDescriptor> {
    let hidden = geometry.hidden_size;
    let channels = geometry.subsampling_channels;
    // Saturating defense in depth: metadata parsers already bound n_mels and
    // channels, so overflow is unreachable; a saturated requirement matches no
    // pack tensor and stays fail-closed at validation.
    let flatten =
        fastconformer_subsampling_flatten_width(channels, geometry.n_mels).unwrap_or(usize::MAX);
    let k = SUBSAMPLING_KERNEL;
    let ordinary_or_dw = vec![k, k, 1, channels];
    let pointwise = vec![1, 1, channels, channels];
    let mut descriptors = Vec::new();
    for (sub_layer, weight_dims, weight_reason) in [
        (
            0usize,
            ordinary_or_dw.clone(),
            "subsampling conv0 must be the ordinary 3x3 [kw,kh,IC=1,OC=channels] kernel",
        ),
        (
            2usize,
            ordinary_or_dw.clone(),
            "subsampling conv2 must be the depthwise 3x3 [kw,kh,1,channels] kernel",
        ),
        (
            3usize,
            pointwise.clone(),
            "subsampling conv3 must be the pointwise 1x1 [1,1,channels,channels] kernel",
        ),
        (
            5usize,
            ordinary_or_dw.clone(),
            "subsampling conv5 must be the depthwise 3x3 [kw,kh,1,channels] kernel",
        ),
        (
            6usize,
            pointwise,
            "subsampling conv6 must be the pointwise 1x1 [1,1,channels,channels] kernel",
        ),
    ] {
        descriptors.push(descriptor(
            format!("enc.sub.layers.{sub_layer}.weight"),
            TensorBindingDescriptorRequirement::ExactDims(weight_dims),
            weight_reason,
        ));
        descriptors.push(descriptor(
            format!("enc.sub.layers.{sub_layer}.bias"),
            TensorBindingDescriptorRequirement::VectorLen(channels),
            "subsampling conv bias must span the declared channel count",
        ));
    }
    descriptors.push(descriptor(
        "enc.sub.linear.weight".to_string(),
        TensorBindingDescriptorRequirement::ExactDims(vec![flatten, hidden]),
        "subsampling linear must be ggml [flatten, hidden] for mul_mat (channels * freq' -> hidden_size)",
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
    let pw1_out = hidden.saturating_mul(2);
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
            TensorBindingDescriptorRequirement::ExactDims(vec![hidden, ffn]),
            "first FFN up projection must be ggml [hidden, ffn] for mul_mat",
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
        TensorBindingDescriptorRequirement::ExactDims(vec![ffn, hidden]),
        "first FFN down projection must be ggml [ffn, hidden] for mul_mat",
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
            TensorBindingDescriptorRequirement::ExactDims(vec![hidden, hidden]),
            "attention projection must be ggml [hidden, hidden] for mul_mat",
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
        TensorBindingDescriptorRequirement::ExactDims(vec![hidden, hidden]),
        "attention output projection must be ggml [hidden, hidden] for mul_mat",
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
            TensorBindingDescriptorRequirement::ExactDims(vec![hidden, hidden]),
            "relative position projection must be ggml [hidden, hidden] for mul_mat",
        ),
        descriptor(
            n("attn.pos_bias_u"),
            TensorBindingDescriptorRequirement::ExactDims(vec![
                geometry.head_dim,
                geometry.n_heads,
            ]),
            "Transformer-XL pos_bias_u must be ggml [head_dim, n_heads] after importer reverse",
        ),
        descriptor(
            n("attn.pos_bias_v"),
            TensorBindingDescriptorRequirement::ExactDims(vec![
                geometry.head_dim,
                geometry.n_heads,
            ]),
            "Transformer-XL pos_bias_v must be ggml [head_dim, n_heads] after importer reverse",
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
            TensorBindingDescriptorRequirement::ExactDims(vec![hidden, pw1_out]),
            "conv pointwise1 must be ggml [hidden, 2*hidden] for mul_mat (GLU)",
        ),
    ]);
    if geometry.bias_present {
        descriptors.push(descriptor(
            n("conv.pw1.bias"),
            TensorBindingDescriptorRequirement::VectorLen(pw1_out),
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
        TensorBindingDescriptorRequirement::ExactDims(vec![hidden, hidden]),
        "conv pointwise2 must be ggml [hidden, hidden] for mul_mat",
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
            TensorBindingDescriptorRequirement::ExactDims(vec![hidden, ffn]),
            "second FFN up projection must be ggml [hidden, ffn] for mul_mat",
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
        TensorBindingDescriptorRequirement::ExactDims(vec![ffn, hidden]),
        "second FFN down projection must be ggml [ffn, hidden] for mul_mat",
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::tensor_binding::validate_tensor_binding_descriptors;

    fn geometry() -> FastConformerContractGeometry {
        FastConformerContractGeometry {
            n_layers: 1,
            hidden_size: 16,
            ffn_dim: 32,
            conv_kernel: 3,
            n_heads: 2,
            head_dim: 8,
            n_mels: 80,
            subsampling_channels: 4,
            bias_present: true,
        }
    }

    fn tensor_index_from_shapes(shapes: &[(String, Vec<u64>)]) -> crate::GgufTensorIndex {
        let tensors = shapes
            .iter()
            .enumerate()
            .map(|(index, (name, dims))| crate::GgufTensorMetadata {
                name: name.clone(),
                dims: dims.clone(),
                ggml_type: 0,
                type_name: "f32".to_string(),
                size_bytes: 0,
                offset_bytes: index as u64,
            })
            .collect();
        crate::GgufTensorIndex::from_snapshot(crate::ggml_runtime::GgufTensorIndexSnapshot {
            path: std::path::PathBuf::from("fastconformer-subsampling-contract-test.oasr"),
            data_section_offset_bytes: 0,
            tensors,
        })
        .expect("unique tensor names")
    }

    #[test]
    fn subsampled_freq_and_flatten_match_three_stride2_stages() {
        // n_mels=80 → 40 → 20 → 10; channels=256 → flatten=2560.
        assert_eq!(fastconformer_subsampled_freq(80), Some(10));
        assert_eq!(fastconformer_subsampling_flatten_width(256, 80), Some(2560));
        // n_mels=128 → 64 → 32 → 16; channels=24 → flatten=384.
        assert_eq!(fastconformer_subsampled_freq(128), Some(16));
        assert_eq!(fastconformer_subsampling_flatten_width(24, 128), Some(384));
    }

    #[test]
    fn subsampling_descriptors_pin_exact_graph_kernel_shapes() {
        let g = geometry();
        let descriptors = fastconformer_subsampling_tensor_descriptors(&g);
        let by_name: std::collections::BTreeMap<_, _> = descriptors
            .iter()
            .map(|d| (d.tensor_name.as_str(), &d.requirement))
            .collect();
        let channels = g.subsampling_channels;
        let flatten = fastconformer_subsampling_flatten_width(channels, g.n_mels).unwrap();
        assert_eq!(
            by_name["enc.sub.layers.0.weight"],
            &TensorBindingDescriptorRequirement::ExactDims(vec![3, 3, 1, channels])
        );
        assert_eq!(
            by_name["enc.sub.layers.2.weight"],
            &TensorBindingDescriptorRequirement::ExactDims(vec![3, 3, 1, channels])
        );
        assert_eq!(
            by_name["enc.sub.layers.3.weight"],
            &TensorBindingDescriptorRequirement::ExactDims(vec![1, 1, channels, channels])
        );
        assert_eq!(
            by_name["enc.sub.layers.5.weight"],
            &TensorBindingDescriptorRequirement::ExactDims(vec![3, 3, 1, channels])
        );
        assert_eq!(
            by_name["enc.sub.layers.6.weight"],
            &TensorBindingDescriptorRequirement::ExactDims(vec![1, 1, channels, channels])
        );
        assert_eq!(
            by_name["enc.sub.linear.weight"],
            &TensorBindingDescriptorRequirement::ExactDims(vec![flatten, g.hidden_size])
        );
    }

    #[test]
    fn conv_out_dim_checked_rejects_overflowing_input() {
        assert_eq!(conv_out_dim_checked(0), None);
        assert_eq!(conv_out_dim_checked(usize::MAX), None);
        assert_eq!(fastconformer_subsampled_freq(usize::MAX), None);
    }

    #[test]
    fn layer_descriptors_pin_ordered_ggml_in_out_weight_dims() {
        let g = geometry();
        let descriptors = fastconformer_layer_tensor_descriptors(&g, 0);
        let by_name: std::collections::BTreeMap<_, _> = descriptors
            .iter()
            .map(|d| (d.tensor_name.as_str(), &d.requirement))
            .collect();
        assert_eq!(
            by_name["enc.blk.0.ff1.up.weight"],
            &TensorBindingDescriptorRequirement::ExactDims(vec![g.hidden_size, g.ffn_dim])
        );
        assert_eq!(
            by_name["enc.blk.0.ff1.down.weight"],
            &TensorBindingDescriptorRequirement::ExactDims(vec![g.ffn_dim, g.hidden_size])
        );
        assert_eq!(
            by_name["enc.blk.0.attn.q.weight"],
            &TensorBindingDescriptorRequirement::ExactDims(vec![g.hidden_size, g.hidden_size])
        );
        assert_eq!(
            by_name["enc.blk.0.attn.pos_bias_u"],
            &TensorBindingDescriptorRequirement::ExactDims(vec![g.head_dim, g.n_heads])
        );
        assert_eq!(
            by_name["enc.blk.0.conv.pw1.weight"],
            &TensorBindingDescriptorRequirement::ExactDims(vec![g.hidden_size, g.hidden_size * 2])
        );
        assert_eq!(
            by_name["enc.blk.0.conv.pw2.weight"],
            &TensorBindingDescriptorRequirement::ExactDims(vec![g.hidden_size, g.hidden_size])
        );
    }

    #[test]
    fn rejects_subsampling_kernel_with_wrong_spatial_extent() {
        let g = geometry();
        let mut shapes = crate::models::tensor_binding::project_fixture_tensors(
            &fastconformer_subsampling_tensor_descriptors(&g),
        );
        // Corrupt only axis-3-matching but kh≠3: still channels on axis 3, rank 4,
        // so the old RankAtLeastWithDimAt contract would have admitted it.
        let corrupted = shapes
            .iter_mut()
            .find(|(name, _)| name == "enc.sub.layers.0.weight")
            .expect("conv0 weight");
        corrupted.1 = vec![5, 5, 1, g.subsampling_channels as u64];
        let index = tensor_index_from_shapes(&shapes);
        let err = validate_tensor_binding_descriptors(
            &index,
            &fastconformer_subsampling_tensor_descriptors(&g),
            |name| format!("missing {name}"),
            |name, shape, reason| format!("{name} {:?}: {reason}", shape),
        )
        .expect_err("wrong kh must fail ExactDims");
        assert!(
            err.contains("enc.sub.layers.0.weight"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn admits_exact_graph_aligned_subsampling_shapes() {
        let g = geometry();
        let descriptors = fastconformer_subsampling_tensor_descriptors(&g);
        let shapes = crate::models::tensor_binding::project_fixture_tensors(&descriptors);
        let index = tensor_index_from_shapes(&shapes);
        validate_tensor_binding_descriptors(
            &index,
            &descriptors,
            |name| format!("missing {name}"),
            |name, shape, reason| format!("{name} {:?}: {reason}", shape),
        )
        .expect("ExactDims fixture projection must satisfy the contract");
    }

    #[test]
    fn exact_dims_requirement_itself_rejects_wrong_kh() {
        let descriptor = TensorBindingDescriptor {
            tensor_name: "enc.sub.layers.0.weight".to_string(),
            requirement: TensorBindingDescriptorRequirement::ExactDims(vec![3, 3, 1, 4]),
            reason: "pin".to_string(),
        };
        let bad = validate_tensor_binding_descriptors(
            &tensor_index_from_shapes(&[("enc.sub.layers.0.weight".to_string(), vec![5, 5, 1, 4])]),
            std::slice::from_ref(&descriptor),
            |_| "missing".to_string(),
            |_, _, reason| reason,
        );
        assert!(bad.is_err());
        let good = validate_tensor_binding_descriptors(
            &tensor_index_from_shapes(&[("enc.sub.layers.0.weight".to_string(), vec![3, 3, 1, 4])]),
            std::slice::from_ref(&descriptor),
            |_| "missing".to_string(),
            |_, _, reason| reason,
        );
        assert!(good.is_ok());
    }
}
