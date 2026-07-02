use crate::{GgufTensorIndex, GgufTensorMetadata};

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
        TensorBindingRequirement::ExactDims(expected) => {
            dims.len() == expected.len()
                && dims
                    .iter()
                    .copied()
                    .map(|value| value as usize)
                    .eq(expected.iter().copied())
        }
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
