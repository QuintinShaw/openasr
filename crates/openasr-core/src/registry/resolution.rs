use super::{ModelCard, ModelRef, ModelResolutionError, ResolvedModel};
use crate::catalog_series::family_aliases_match;

pub(super) fn parse_model_ref(value: &str) -> Result<ModelRef, ModelResolutionError> {
    let value = value.trim();
    if value.is_empty() {
        return Err(ModelResolutionError::InvalidRef(value.to_string()));
    }
    let mut parts = value.split(':');
    let family = parts.next().unwrap_or_default();
    let tag = parts.next();
    if parts.next().is_some()
        || family.trim().is_empty()
        || tag.is_some_and(|tag| tag.trim().is_empty())
    {
        return Err(ModelResolutionError::InvalidRef(value.to_string()));
    }

    Ok(ModelRef {
        family: family.to_string(),
        tag: tag.map(ToOwned::to_owned),
    })
}

pub(super) fn resolve_registry_model_ref<'a>(
    cards: &'a [ModelCard],
    model_ref: &str,
) -> Result<ResolvedModel<'a>, ModelResolutionError> {
    let parsed = parse_model_ref(model_ref)?;
    if let Some(tag) = parsed.tag.as_deref() {
        return resolve_tagged_model_ref(cards, model_ref, &parsed.family, tag);
    }

    resolve_untagged_model_ref(cards, model_ref, &parsed.family)
}

fn resolve_tagged_model_ref<'a>(
    cards: &'a [ModelCard],
    requested: &str,
    family: &str,
    tag: &str,
) -> Result<ResolvedModel<'a>, ModelResolutionError> {
    let family_cards = family_cards_v0(cards, family, false);
    if family_cards.is_empty() {
        return Err(ModelResolutionError::UnknownModel(family.to_string()));
    }

    let matches: Vec<_> = family_cards
        .iter()
        .copied()
        .filter(|card| card.variant_tag() == Some(tag))
        .collect();
    resolve_match_or_error_v0(
        &matches,
        requested,
        Some(tag.to_string()),
        || ModelResolutionError::UnknownVariantTag {
            family: family.to_string(),
            tag: tag.to_string(),
            available_tags: available_tags(&family_cards),
        },
        || ModelResolutionError::AmbiguousModelRef {
            model_ref: requested.to_string(),
            available_refs: available_refs(&family_cards),
        },
    )
}

fn resolve_untagged_model_ref<'a>(
    cards: &'a [ModelCard],
    requested: &str,
    model: &str,
) -> Result<ResolvedModel<'a>, ModelResolutionError> {
    let family_cards = family_cards_v0(cards, model, true);
    if !family_cards.is_empty() {
        if let Some(default_variant) = family_default_variant(&family_cards)? {
            let matches: Vec<_> = family_cards
                .iter()
                .copied()
                .filter(|card| card.variant_tag() == Some(default_variant))
                .collect();
            let default_variant = default_variant.to_string();
            return resolve_match_or_error_v0(
                &matches,
                requested,
                Some(default_variant.clone()),
                || ModelResolutionError::MissingDefaultVariant {
                    family: model.to_string(),
                    default_variant: default_variant.clone(),
                },
                || ModelResolutionError::AmbiguousModelRef {
                    model_ref: requested.to_string(),
                    available_refs: available_refs(&family_cards),
                },
            );
        }
        if family_cards.len() == 1 {
            let card = family_cards[0];
            return Ok(resolved_model(
                card,
                requested,
                card.variant_tag().map(ToOwned::to_owned),
            ));
        }
        return Err(ModelResolutionError::AmbiguousModelRef {
            model_ref: requested.to_string(),
            available_refs: available_refs(&family_cards),
        });
    }

    let exact_matches: Vec<_> = cards.iter().filter(|card| card.id == model).collect();
    match exact_matches.as_slice() {
        [card] => Ok(resolved_model(
            card,
            requested,
            card.variant_tag().map(ToOwned::to_owned),
        )),
        [] => Err(ModelResolutionError::UnknownModel(model.to_string())),
        _ => Err(ModelResolutionError::AmbiguousModelRef {
            model_ref: requested.to_string(),
            available_refs: exact_matches
                .iter()
                .map(|card| card.id.as_str())
                .collect::<Vec<_>>()
                .join(", "),
        }),
    }
}

fn family_cards_v0<'a>(
    cards: &'a [ModelCard],
    family: &str,
    only_with_variant: bool,
) -> Vec<&'a ModelCard> {
    cards
        .iter()
        .filter(|card| {
            family_aliases_match(card.family_name(), family)
                && (!only_with_variant || card.variant.is_some())
        })
        .collect()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ResolveSingleMatchErrorV0 {
    NotFound,
    Ambiguous,
}

fn resolve_single_match_v0<'a>(
    matches: &[&'a ModelCard],
) -> Result<&'a ModelCard, ResolveSingleMatchErrorV0> {
    match matches {
        [card] => Ok(*card),
        [] => Err(ResolveSingleMatchErrorV0::NotFound),
        _ => Err(ResolveSingleMatchErrorV0::Ambiguous),
    }
}

fn resolve_match_or_error_v0<'a>(
    matches: &[&'a ModelCard],
    requested: &str,
    tag: Option<String>,
    not_found: impl FnOnce() -> ModelResolutionError,
    ambiguous: impl FnOnce() -> ModelResolutionError,
) -> Result<ResolvedModel<'a>, ModelResolutionError> {
    match resolve_single_match_v0(matches) {
        Ok(card) => Ok(resolved_model(card, requested, tag)),
        Err(ResolveSingleMatchErrorV0::NotFound) => Err(not_found()),
        Err(ResolveSingleMatchErrorV0::Ambiguous) => Err(ambiguous()),
    }
}

fn family_default_variant<'a>(
    family_cards: &[&'a ModelCard],
) -> Result<Option<&'a str>, ModelResolutionError> {
    let mut default = None;
    for card in family_cards {
        let Some(card_default) = card.default_variant.as_deref() else {
            continue;
        };
        if let Some(existing) = default {
            if existing != card_default {
                return Err(ModelResolutionError::AmbiguousModelRef {
                    model_ref: card.family_name().to_string(),
                    available_refs: available_refs(family_cards),
                });
            }
        } else {
            default = Some(card_default);
        }
    }
    Ok(default)
}

fn resolved_model<'a>(
    card: &'a ModelCard,
    requested: &str,
    tag: Option<String>,
) -> ResolvedModel<'a> {
    ResolvedModel {
        card,
        requested: requested.to_string(),
        resolved_id: card.id.clone(),
        family: card.family_name().to_string(),
        tag,
        is_default_variant: card.is_default_variant(),
    }
}

fn available_tags(cards: &[&ModelCard]) -> String {
    let mut tags: Vec<_> = cards.iter().filter_map(|card| card.variant_tag()).collect();
    tags.sort_unstable();
    tags.dedup();
    if tags.is_empty() {
        "<none>".to_string()
    } else {
        tags.join(", ")
    }
}

fn available_refs(cards: &[&ModelCard]) -> String {
    let mut refs: Vec<_> = cards
        .iter()
        .map(|card| {
            card.variant_tag().map_or_else(
                || card.id.clone(),
                |tag| format!("{}:{tag}", card.family_name()),
            )
        })
        .collect();
    refs.sort();
    refs.dedup();
    refs.join(", ")
}
