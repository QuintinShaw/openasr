use std::path::Path;

use super::{ModelCard, RegistryError};

mod common;
mod id;
mod variant;

pub(super) fn validate_card(path: &Path, card: &ModelCard) -> Result<(), RegistryError> {
    id::validate_identity(path, card)?;
    variant::validate_variant_metadata(path, card)
}

pub(super) fn validate_unique_ids(cards: &[ModelCard]) -> Result<(), RegistryError> {
    id::validate_unique_ids(cards)
}

pub(super) fn validate_variant_index(cards: &[ModelCard]) -> Result<(), RegistryError> {
    variant::validate_variant_index(cards)
}
