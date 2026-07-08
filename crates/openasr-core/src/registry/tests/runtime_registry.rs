use std::collections::BTreeSet;

use super::*;

fn bundled_public_catalog() -> ModelCatalog {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../model-registry/catalog.public.json");
    let contents = fs::read_to_string(&path).unwrap();
    parse_model_catalog(&contents, &path.display().to_string()).unwrap()
}

/// P0 regression: a deployed binary has no source-tree `model-registry/`
/// directory. The old `load_registry(default_registry_dir())` path fails closed
/// there; the runtime registry must instead derive from the signed catalog (here
/// the embedded snapshot) with no filesystem registry dir, no override, and no
/// network. Fails on the pre-fix code path, passes on the fix.
#[test]
fn deployed_layout_resolves_runtime_registry_from_signed_catalog() {
    let temp = tempfile::tempdir().unwrap();

    // The deployed failure mode the P0 bug hit: no on-disk registry directory.
    let deployed_dir = temp.path().join("model-registry/models");
    assert!(matches!(
        load_registry(&deployed_dir),
        Err(RegistryError::MissingDirectory(_))
    ));

    // The fix: derive the registry from the signed catalog. No dev override is set,
    // so the derive path (not the filesystem path) is exercised.
    assert!(std::env::var_os(OPENASR_REGISTRY_DIR_ENV).is_none());
    let embedded = load_embedded_signed_catalog(temp.path()).unwrap();
    let cards = runtime_registry(Some(&embedded)).expect("derive registry from signed catalog");

    assert!(!cards.is_empty());
    // Model resolution goes through with no filesystem registry present, and each
    // model stays its own family (no AmbiguousModelRef collapse).
    let resolved = resolve_registry_model_ref(&cards, "whisper-small").unwrap();
    assert_eq!(resolved.card.id, "whisper-small");
    assert_eq!(resolved.family, "whisper-small");
    assert_eq!(resolved.card.family_name(), "whisper-small");
}

/// The embedded snapshot is an epoch-max floor: it wins only when strictly newer
/// than the network/cache tier (local preview), never downgrades a newer catalog
/// (release), and a tie keeps the freshly fetched network catalog. The security
/// rollback guard lives in `load_embedded_signed_catalog`; this only decides
/// freshness.
#[test]
fn epoch_max_prefers_only_strictly_newer_embedded() {
    // Local preview: embedded ahead of production -> embedded wins.
    assert_eq!(
        choose_runtime_catalog(Some(3), Some(5)),
        RuntimeCatalogChoice::Embedded
    );
    // Release: embedded is the floor (<= production) -> network wins (latest models).
    assert_eq!(
        choose_runtime_catalog(Some(5), Some(3)),
        RuntimeCatalogChoice::Network
    );
    // Tie: keep the freshly fetched network catalog.
    assert_eq!(
        choose_runtime_catalog(Some(4), Some(4)),
        RuntimeCatalogChoice::Network
    );
    // Unknown network epoch: never displace with embedded here (load_model_catalog
    // already handled the offline embedded fallback).
    assert_eq!(
        choose_runtime_catalog(None, Some(9)),
        RuntimeCatalogChoice::Network
    );
}

/// The catalog-derived cards are the verified 1:1 projection of the committed
/// public registry cards: family falls back to id (no `whisper-*` collapse),
/// quantization mirrors `recommended_quant`, and every derived id has a matching
/// on-disk card with an equivalent runtime projection.
#[test]
fn derived_cards_match_committed_public_registry() {
    let catalog = bundled_public_catalog();
    let derived = model_cards_from_catalog(&catalog).unwrap();
    let on_disk = load_registry(test_model_registry_dir()).unwrap();
    let on_disk_ids: BTreeSet<&str> = on_disk.iter().map(|card| card.id.as_str()).collect();

    assert!(!derived.is_empty());
    for card in &derived {
        // family_name() falls back to id exactly like the committed cards.
        assert_eq!(card.family, None);
        assert_eq!(card.family_name(), card.id);
        assert_eq!(card.variant_tag(), Some("published"));
        assert_eq!(card.variant_format(), Some("oasr"));
        assert_eq!(card.backend, "native");
        assert!(card.is_default_variant());

        let model = catalog
            .models
            .iter()
            .find(|model| model.id == card.id)
            .unwrap();
        assert_eq!(
            card.variant_quantization(),
            Some(model.recommended_quant.as_str())
        );

        // Public entries are a subset of the on-disk cards (which also carry the
        // staged, non-public forced-aligner); each derived card matches its
        // on-disk twin on the runtime-relevant projection.
        assert!(
            on_disk_ids.contains(card.id.as_str()),
            "derived id {} has no committed card",
            card.id
        );
        let disk = on_disk.iter().find(|disk| disk.id == card.id).unwrap();
        assert_eq!(disk.family_name(), card.family_name());
        assert_eq!(disk.variant_quantization(), card.variant_quantization());
        assert_eq!(disk.is_default_variant(), card.is_default_variant());
    }

    // Several whisper models, each its own family: no collapse into one family.
    let whisper: Vec<&ModelCard> = derived
        .iter()
        .filter(|card| card.id.starts_with("whisper-"))
        .collect();
    assert!(whisper.len() >= 2, "expected multiple whisper models");
    for card in whisper {
        assert_eq!(card.family_name(), card.id);
    }
}
