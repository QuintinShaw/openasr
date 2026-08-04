use std::fmt::Display;

const OUTDATED_PACK_HINT: &str = "this pack was likely produced by an outdated or incompatible conversion pipeline; re-convert or re-pull the model pack";

pub(crate) fn metadata_validation_error(label: &str, error: impl Display) -> String {
    format!("{label} runtime metadata contract validation failed: {error} ({OUTDATED_PACK_HINT})")
}

pub(crate) fn tensor_validation_error(error: impl Display) -> String {
    format!("runtime tensor contract validation failed: {error}")
}
