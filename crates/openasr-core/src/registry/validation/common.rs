use std::path::Path;

use super::RegistryError;

pub(super) fn invalid_card(path: &Path, message: impl Into<String>) -> RegistryError {
    RegistryError::ValidateCard {
        path: path.to_path_buf(),
        message: message.into(),
    }
}

pub(super) fn require_non_empty(
    path: &Path,
    field: &str,
    value: &str,
) -> Result<(), RegistryError> {
    if value.trim().is_empty() {
        return Err(invalid_card(path, format!("{field} is required")));
    }
    Ok(())
}

pub(super) fn require_allowed(
    path: &Path,
    field: &str,
    value: &str,
    allowed: &[&str],
) -> Result<(), RegistryError> {
    require_non_empty(path, field, value)?;
    if !allowed.contains(&value) {
        return Err(invalid_card(
            path,
            format!("{field} '{value}' is not supported"),
        ));
    }
    Ok(())
}
