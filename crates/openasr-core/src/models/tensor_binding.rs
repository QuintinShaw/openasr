use crate::{GgufTensorIndex, GgufTensorMetadata};

/// The tensor names one family's runtime contract allows its weight loaders
/// to read. Built from the family's binding-descriptor enumeration, it makes
/// that enumeration the loader's authoritative read list: a read of any name
/// the contract does not cover fails closed at load time, so loader/contract
/// name drift cannot survive in either direction.
#[derive(Debug, Clone, Default)]
pub(crate) struct TensorReadGuard {
    names: std::collections::BTreeSet<String>,
}

impl TensorReadGuard {
    pub(crate) fn from_descriptors(descriptors: &[TensorBindingDescriptor]) -> Self {
        Self {
            names: descriptors
                .iter()
                .map(|descriptor| descriptor.tensor_name.clone())
                .collect(),
        }
    }

    pub(crate) fn contains(&self, name: &str) -> bool {
        self.names.contains(name)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TensorBindingRequirement<'a> {
    ExactDims(&'a [usize]),
    VectorLen(usize),
    NonEmptyVector,
    Rank2WithDim(usize),
    Rank2EitherDims(usize, usize),
    Rank2OrRank3WithDims(usize, usize),
    RankAtLeastWithDimAt {
        min_rank: usize,
        axis: usize,
        dim: usize,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TensorBindingSpec<'a> {
    pub tensor_name: &'a str,
    pub requirement: TensorBindingRequirement<'a>,
    pub reason: &'a str,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum TensorBindingDescriptorRequirement {
    ExactDims(Vec<usize>),
    VectorLen(usize),
    NonEmptyVector,
    Rank2WithDim(usize),
    Rank2EitherDims(usize, usize),
    Rank2OrRank3WithDims(usize, usize),
    RankAtLeastWithDimAt {
        min_rank: usize,
        axis: usize,
        dim: usize,
    },
}

impl TensorBindingDescriptorRequirement {
    fn as_requirement(&self) -> TensorBindingRequirement<'_> {
        match self {
            Self::ExactDims(expected) => TensorBindingRequirement::ExactDims(expected),
            Self::VectorLen(expected_len) => TensorBindingRequirement::VectorLen(*expected_len),
            Self::NonEmptyVector => TensorBindingRequirement::NonEmptyVector,
            Self::Rank2WithDim(expected_dim) => {
                TensorBindingRequirement::Rank2WithDim(*expected_dim)
            }
            Self::Rank2EitherDims(lhs, rhs) => {
                TensorBindingRequirement::Rank2EitherDims(*lhs, *rhs)
            }
            Self::Rank2OrRank3WithDims(first, second) => {
                TensorBindingRequirement::Rank2OrRank3WithDims(*first, *second)
            }
            Self::RankAtLeastWithDimAt {
                min_rank,
                axis,
                dim,
            } => TensorBindingRequirement::RankAtLeastWithDimAt {
                min_rank: *min_rank,
                axis: *axis,
                dim: *dim,
            },
        }
    }
}

impl From<TensorBindingRequirement<'_>> for TensorBindingDescriptorRequirement {
    fn from(requirement: TensorBindingRequirement<'_>) -> Self {
        match requirement {
            TensorBindingRequirement::ExactDims(expected) => Self::ExactDims(expected.to_vec()),
            TensorBindingRequirement::VectorLen(expected_len) => Self::VectorLen(expected_len),
            TensorBindingRequirement::NonEmptyVector => Self::NonEmptyVector,
            TensorBindingRequirement::Rank2WithDim(expected_dim) => {
                Self::Rank2WithDim(expected_dim)
            }
            TensorBindingRequirement::Rank2EitherDims(lhs, rhs) => Self::Rank2EitherDims(lhs, rhs),
            TensorBindingRequirement::Rank2OrRank3WithDims(first, second) => {
                Self::Rank2OrRank3WithDims(first, second)
            }
            TensorBindingRequirement::RankAtLeastWithDimAt {
                min_rank,
                axis,
                dim,
            } => Self::RankAtLeastWithDimAt {
                min_rank,
                axis,
                dim,
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TensorBindingDescriptor {
    pub tensor_name: String,
    pub requirement: TensorBindingDescriptorRequirement,
    pub reason: String,
}

impl TensorBindingDescriptor {
    fn as_spec(&self) -> TensorBindingSpec<'_> {
        TensorBindingSpec {
            tensor_name: &self.tensor_name,
            requirement: self.requirement.as_requirement(),
            reason: &self.reason,
        }
    }
}

impl From<TensorBindingSpec<'_>> for TensorBindingDescriptor {
    fn from(spec: TensorBindingSpec<'_>) -> Self {
        Self {
            tensor_name: spec.tensor_name.to_string(),
            requirement: spec.requirement.into(),
            reason: spec.reason.to_string(),
        }
    }
}

pub(crate) fn validate_tensor_binding_descriptors<E>(
    index: &GgufTensorIndex,
    bindings: &[TensorBindingDescriptor],
    missing: impl Fn(&str) -> E + Copy,
    invalid: impl Fn(&str, &[u64], String) -> E + Copy,
) -> Result<(), E> {
    for binding in bindings {
        let tensor = require_tensor(index, &binding.tensor_name, missing)?;
        validate_tensor_binding(&tensor.dims, binding.as_spec(), invalid)?;
    }
    Ok(())
}

pub(crate) fn tensor_binding_descriptors(
    bindings: &[TensorBindingSpec<'_>],
) -> Vec<TensorBindingDescriptor> {
    bindings
        .iter()
        .copied()
        .map(TensorBindingDescriptor::from)
        .collect()
}

pub(crate) fn require_tensor<'a, E>(
    index: &'a GgufTensorIndex,
    tensor_name: &str,
    missing: impl Fn(&str) -> E,
) -> Result<&'a GgufTensorMetadata, E> {
    index.get(tensor_name).ok_or_else(|| missing(tensor_name))
}

pub(crate) fn validate_tensor_binding<E>(
    dims: &[u64],
    spec: TensorBindingSpec<'_>,
    invalid: impl FnOnce(&str, &[u64], String) -> E,
) -> Result<(), E> {
    let valid = match spec.requirement {
        TensorBindingRequirement::ExactDims(expected) => exact_dims_match(dims, expected),
        TensorBindingRequirement::VectorLen(expected_len) => dims == [expected_len as u64],
        TensorBindingRequirement::NonEmptyVector => dims.len() == 1 && dims[0] > 0,
        TensorBindingRequirement::Rank2WithDim(expected_dim) => {
            dims.len() == 2
                && (dims[0] as usize == expected_dim || dims[1] as usize == expected_dim)
        }
        TensorBindingRequirement::Rank2EitherDims(lhs, rhs) => {
            dims.len() == 2
                && ((dims[0] as usize == lhs && dims[1] as usize == rhs)
                    || (dims[0] as usize == rhs && dims[1] as usize == lhs))
        }
        TensorBindingRequirement::Rank2OrRank3WithDims(first, second) => {
            (dims.len() == 2 || dims.len() == 3)
                && dims.iter().any(|value| *value as usize == first)
                && dims.iter().any(|value| *value as usize == second)
        }
        TensorBindingRequirement::RankAtLeastWithDimAt {
            min_rank,
            axis,
            dim,
        } => dims.len() >= min_rank && dims.get(axis).is_some_and(|value| *value as usize == dim),
    };
    if valid {
        return Ok(());
    }
    Err(invalid(spec.tensor_name, dims, spec.reason.to_string()))
}

pub(crate) fn render_shape(shape: &[u64]) -> String {
    let parts = shape
        .iter()
        .map(u64::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    format!("[{parts}]")
}

/// Ordered extent match for graph-consumed matrices.
///
/// Accepts an exact rank/extent match, or the same non-unit subsequence after
/// stripping unit axes. Shipped packs sometimes keep a squeezed kernel/channel
/// axis as a leading or trailing `1` (for example FastConformer
/// `conv.pw1.weight` as `[1, hidden, 2*hidden]` vs contract `[hidden, 2*hidden]`);
/// those layouts are equivalent for `mul_mat` after the loader binds the tensor.
/// Transposed non-unit extents still fail closed.
pub(crate) fn exact_dims_match(dims: &[u64], expected: &[usize]) -> bool {
    let got: Vec<usize> = dims.iter().map(|d| *d as usize).collect();
    if got == expected {
        return true;
    }
    let strip =
        |values: &[usize]| -> Vec<usize> { values.iter().copied().filter(|v| *v != 1).collect() };
    let got_core = strip(&got);
    let exp_core = strip(expected);
    !got_core.is_empty() && got_core == exp_core
}

/// Projects one valid dims choice for a tensor-binding requirement, for
/// runtime-ready test fixtures: every family's fixture tensors stamp through
/// this single map, so a requirement kind can never disagree with the fixture
/// dims it produces. GGUF reads a tensor index back with trailing-1 dims
/// trimmed, so projections keep their last extent > 1 wherever the rank must
/// survive (`VectorLen(1)` is the lone `[1]` that reads back as rank 1).
#[cfg(any(test, feature = "testing"))]
pub(crate) fn project_fixture_dims(requirement: &TensorBindingDescriptorRequirement) -> Vec<u64> {
    let mut dims = match requirement {
        TensorBindingDescriptorRequirement::ExactDims(expected) => {
            expected.iter().map(|dim| *dim as u64).collect()
        }
        TensorBindingDescriptorRequirement::VectorLen(len) => vec![*len as u64],
        TensorBindingDescriptorRequirement::NonEmptyVector => vec![2],
        TensorBindingDescriptorRequirement::Rank2WithDim(dim) => vec![2, *dim as u64],
        TensorBindingDescriptorRequirement::Rank2EitherDims(lhs, rhs) => {
            vec![*lhs as u64, *rhs as u64]
        }
        TensorBindingDescriptorRequirement::Rank2OrRank3WithDims(first, second) => {
            vec![*first as u64, *second as u64]
        }
        TensorBindingDescriptorRequirement::RankAtLeastWithDimAt {
            min_rank,
            axis,
            dim,
        } => {
            let mut dims = vec![2_u64; *min_rank];
            dims[*axis] = *dim as u64;
            dims
        }
    };
    // A trailing 1 would read back as a lower rank. Swap a non-unit extent to
    // the end when one exists (rank-2+ shapes only), preserving both extents;
    // an all-unit shape keeps its rank by bumping the last extent.
    if dims.len() >= 2 && *dims.last().expect("rank >= 2") == 1 {
        let last = dims.len() - 1;
        match dims.iter().position(|value| *value != 1) {
            Some(non_unit) => dims.swap(non_unit, last),
            None => dims[last] = 2,
        }
    }
    dims
}

/// Projects the single tensor-binding enumeration into a runtime-ready fixture
/// tensor set (tensor name + valid dims), so the fixture and the admission
/// validator agree through one enumeration.
#[cfg(any(test, feature = "testing"))]
pub(crate) fn project_fixture_tensors(
    bindings: &[TensorBindingDescriptor],
) -> Vec<(String, Vec<u64>)> {
    bindings
        .iter()
        .map(|binding| {
            (
                binding.tensor_name.clone(),
                project_fixture_dims(&binding.requirement),
            )
        })
        .collect()
}

/// Compare a traced full weight load against the binding-descriptor
/// enumeration: exact name-set equality both directions, then each traced
/// read's dims must satisfy its descriptor's requirement at the precision it
/// declares. Shared by every family whose loaders read through the GGUF
/// tensor index (parakeet-ctc, parakeet-tdt, funasr-nano encoder half).
///
/// In-crate unit tests only; not part of the cross-crate `testing` surface.
#[cfg(test)]
pub(crate) fn assert_trace_matches_descriptor_set(
    trace: &[crate::ggml_runtime::GgufTensorAccessRecord],
    descriptors: &[TensorBindingDescriptor],
) {
    let mut required: std::collections::BTreeMap<&str, &TensorBindingDescriptorRequirement> =
        std::collections::BTreeMap::new();
    for descriptor in descriptors {
        if required
            .insert(descriptor.tensor_name.as_str(), &descriptor.requirement)
            .is_some()
        {
            panic!(
                "descriptor names must be unique: {}",
                descriptor.tensor_name
            );
        }
    }
    let mut traced: std::collections::BTreeMap<String, Vec<u64>> =
        std::collections::BTreeMap::new();
    for record in trace {
        if let Some(previous) = traced.get(&record.name) {
            assert_eq!(
                previous, &record.dims,
                "traced dims for '{}' must be stable across reads",
                record.name
            );
        } else {
            traced.insert(record.name.clone(), record.dims.clone());
        }
    }
    let missing: Vec<&&str> = required
        .keys()
        .filter(|name| !traced.contains_key(**name))
        .collect();
    let extra: Vec<&String> = traced
        .keys()
        .filter(|name| !required.contains_key(name.as_str()))
        .collect();
    assert!(
        missing.is_empty() && extra.is_empty(),
        "loader read set and contract descriptor set must be equal; \
         required-but-never-read={missing:?} read-but-not-required={extra:?}"
    );
    for (name, requirement) in &required {
        let dims = &traced[*name];
        assert!(
            requirement_matches_dims(requirement, dims),
            "loader read '{name}' with dims {dims:?}, but the contract requires {requirement:?}"
        );
    }
}

/// Check stored dims against one descriptor requirement at the precision
/// it declares. Mirrors [`validate_tensor_binding`] for u64 dims.
///
/// Helper for in-crate unit tests only.
#[cfg(test)]
fn requirement_matches_dims(
    requirement: &TensorBindingDescriptorRequirement,
    dims: &[u64],
) -> bool {
    let spec = TensorBindingSpec {
        tensor_name: "trace",
        requirement: requirement.as_requirement(),
        reason: "",
    };
    validate_tensor_binding(dims, spec, |_, _, _| ()).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        read_gguf_tensor_index,
        testing::{TinyGgufFixtureSpec, write_tiny_gguf_runtime_source},
    };
    use tempfile::NamedTempFile;

    fn binding_error(name: &str, dims: &[u64], reason: String) -> String {
        format!("{name}:{dims:?}:{reason}")
    }

    fn missing_error(name: &str) -> String {
        format!("missing:{name}")
    }

    #[test]
    fn validates_rank2_either_dims() {
        let spec = TensorBindingSpec {
            tensor_name: "weight",
            requirement: TensorBindingRequirement::Rank2EitherDims(16, 32),
            reason: "expected dims",
        };
        validate_tensor_binding(&[16, 32], spec, binding_error).expect("canonical dims");
        validate_tensor_binding(&[32, 16], spec, binding_error).expect("transposed dims");
    }

    #[test]
    fn exact_dims_accepts_unit_axis_padding_but_rejects_transpose() {
        let spec = TensorBindingSpec {
            tensor_name: "enc.blk.0.conv.pw1.weight",
            requirement: TensorBindingRequirement::ExactDims(&[1024, 2048]),
            reason: "pw1 GLU",
        };
        validate_tensor_binding(&[1024, 2048], spec, binding_error).expect("exact");
        validate_tensor_binding(&[1, 1024, 2048], spec, binding_error)
            .expect("leading unit axis is layout-equivalent");
        validate_tensor_binding(&[1024, 2048, 1], spec, binding_error)
            .expect("trailing unit axis is layout-equivalent");
        validate_tensor_binding(&[2048, 1024], spec, binding_error)
            .expect_err("transposed non-unit extents must fail");
    }

    #[test]
    fn rejects_mismatched_rank_at_axis() {
        let spec = TensorBindingSpec {
            tensor_name: "conv_out",
            requirement: TensorBindingRequirement::RankAtLeastWithDimAt {
                min_rank: 2,
                axis: 1,
                dim: 64,
            },
            reason: "expected rank>=2 and dims[1]=64",
        };
        let error = validate_tensor_binding(&[32, 32], spec, binding_error)
            .expect_err("axis mismatch must fail");
        assert!(error.contains("conv_out"));
    }

    #[test]
    fn validates_descriptor_batches() {
        let file = NamedTempFile::new().expect("temp file");
        let spec = TinyGgufFixtureSpec::new(Default::default())
            .with_tensor_shape("audio.mel_window", [400_u64])
            .with_tensor_shape("output.weight", [32_u64, 64_u64]);
        write_tiny_gguf_runtime_source(file.path(), &spec).expect("write fixture");
        let index = read_gguf_tensor_index(file.path()).expect("read tensor index");
        let bindings = vec![
            TensorBindingDescriptor {
                tensor_name: "audio.mel_window".to_string(),
                requirement: TensorBindingDescriptorRequirement::VectorLen(400),
                reason: "expected mel window".to_string(),
            },
            TensorBindingDescriptor {
                tensor_name: "output.weight".to_string(),
                requirement: TensorBindingDescriptorRequirement::Rank2EitherDims(32, 64),
                reason: "expected output projection matrix".to_string(),
            },
        ];

        validate_tensor_binding_descriptors(&index, &bindings, missing_error, binding_error)
            .expect("descriptor batch validation should succeed");
    }
}
