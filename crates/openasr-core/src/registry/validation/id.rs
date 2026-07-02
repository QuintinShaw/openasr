use std::path::Path;

use crate::BackendKind;

use super::{
    ModelCard, RegistryError,
    common::{require_allowed, require_non_empty},
};

pub(super) fn validate_identity(path: &Path, card: &ModelCard) -> Result<(), RegistryError> {
    let file_id = path.file_stem().and_then(|stem| stem.to_str());
    if file_id != Some(card.id.as_str()) {
        return Err(super::common::invalid_card(
            path,
            format!("id '{}' must match the file name", card.id),
        ));
    }

    require_non_empty(path, "id", &card.id)?;
    require_non_empty(path, "display_name", &card.display_name)?;
    require_allowed(path, "backend", &card.backend, BackendKind::ALL)?;
    require_allowed(path, "task", &card.task, &["transcription"])?;
    if card.languages.is_empty() {
        return Err(super::common::invalid_card(
            path,
            "languages must list at least one language code",
        ));
    }
    for language in &card.languages {
        require_non_empty(path, "languages[]", language)?;
    }
    require_non_empty(path, "size", &card.size)?;
    require_non_empty(path, "recommended_hardware", &card.recommended_hardware)?;
    require_non_empty(path, "license", &card.license)?;
    require_non_empty(path, "quality_profile", &card.quality_profile)?;
    require_non_empty(path, "source", &card.source)?;

    if card.features.is_empty() {
        return Err(super::common::invalid_card(
            path,
            "features must not be empty",
        ));
    }
    for feature in &card.features {
        require_non_empty(path, "features[]", feature)?;
    }

    Ok(())
}

pub(super) fn validate_unique_ids(cards: &[ModelCard]) -> Result<(), RegistryError> {
    for pair in cards.windows(2) {
        if pair[0].id == pair[1].id {
            return Err(RegistryError::DuplicateModelId {
                model_id: pair[0].id.clone(),
            });
        }
    }
    Ok(())
}
