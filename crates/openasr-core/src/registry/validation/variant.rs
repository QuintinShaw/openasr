use std::path::Path;

use super::{
    ModelCard, RegistryError,
    common::{require_allowed, require_non_empty},
};

pub(super) fn validate_variant_metadata(
    path: &Path,
    card: &ModelCard,
) -> Result<(), RegistryError> {
    if let Some(family) = &card.family {
        require_non_empty(path, "family", family)?;
    }
    if let Some(default_variant) = &card.default_variant {
        require_non_empty(path, "default_variant", default_variant)?;
    }
    if let Some(variant) = &card.variant {
        require_non_empty(path, "variant.tag", &variant.tag)?;
        require_allowed(path, "variant.format", &variant.format, &["oasr"])?;
        if let Some(quantization) = &variant.quantization {
            require_non_empty(path, "variant.quantization", quantization)?;
        }
        if let Some(role) = &variant.role {
            require_non_empty(path, "variant.role", role)?;
        }
    }
    Ok(())
}

pub(super) fn validate_variant_index(cards: &[ModelCard]) -> Result<(), RegistryError> {
    let mut variants: Vec<(String, String)> = cards
        .iter()
        .filter_map(|card| {
            let variant = card.variant.as_ref()?;
            Some((card.family_name().to_string(), variant.tag.clone()))
        })
        .collect();
    variants.sort();
    for pair in variants.windows(2) {
        if pair[0] == pair[1] {
            return Err(RegistryError::DuplicateVariant {
                family: pair[0].0.clone(),
                tag: pair[0].1.clone(),
            });
        }
    }

    let defaults = collect_defaults(cards)?;
    for (family, default_variant) in defaults {
        let found = cards.iter().any(|card| {
            card.family_name() == family
                && card
                    .variant
                    .as_ref()
                    .is_some_and(|variant| variant.tag == default_variant)
        });
        if !found {
            return Err(RegistryError::MissingDefaultVariant {
                family,
                default_variant,
            });
        }
    }

    Ok(())
}

fn collect_defaults(cards: &[ModelCard]) -> Result<Vec<(String, String)>, RegistryError> {
    let mut defaults: Vec<(String, String)> = Vec::new();
    for card in cards {
        let Some(default_variant) = &card.default_variant else {
            continue;
        };
        let family = card.family_name().to_string();
        if let Some((_, existing)) = defaults
            .iter()
            .find(|(existing_family, _)| existing_family == &family)
        {
            if existing != default_variant {
                return Err(RegistryError::ConflictingDefaultVariant {
                    family,
                    left: existing.clone(),
                    right: default_variant.clone(),
                });
            }
        } else {
            defaults.push((family, default_variant.clone()));
        }
    }
    Ok(defaults)
}
