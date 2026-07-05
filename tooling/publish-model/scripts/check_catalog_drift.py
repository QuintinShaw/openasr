#!/usr/bin/env python3
"""CI-safe drift checks for OpenASR publish model metadata.

This intentionally avoids tmp/publish evidence. It verifies that the split
tooling/publish-model TOML source still agrees with committed registry cards
and with any already committed machine-catalog entries.
"""
from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(SCRIPT_DIR))

from _catalog import (  # noqa: E402
    QUANT_METADATA,
    language_labels_wire,
    language_mode_for_model,
    languages_for_model,
    load as load_publish_catalog,
    validate_all_card_prose_locales,
)
from _manifest import prose_locales_block, read_prose  # noqa: E402


def load_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def load_toml(path: Path) -> dict:
    return tomllib.loads(path.read_text())


def expected_catalog_quant(registry_id: str, quant: str, recommended: str) -> dict:
    meta = QUANT_METADATA[quant]
    return {
        "quant": quant,
        "suffix": meta.suffix,
        "pull": f"{registry_id}:{meta.suffix}",
        "recommended": quant == recommended,
    }


def check_registry_card(model: str, entry: dict, errors: list[str]) -> None:
    registry_id = entry["registry_id"]
    path = REPO_ROOT / "model-registry" / "models" / f"{registry_id}.toml"
    if not path.exists():
        errors.append(f"{model}: missing registry card {path.relative_to(REPO_ROOT)}")
        return
    card = load_toml(path)
    expected = {
        "id": registry_id,
        "display_name": f"{entry['display_name']} (OpenASR pack)",
        "languages": languages_for_model(entry),
        "size": entry["size"],
        "license": entry["license_name"],
    }
    for key, value in expected.items():
        if card.get(key) != value:
            errors.append(f"{model}: registry {key} drifted: got {card.get(key)!r}, expected {value!r}")
    if card.get("family", registry_id) != registry_id:
        errors.append(
            f"{model}: registry family drifted: got {card.get('family')!r}, expected omitted or {registry_id!r}"
        )
    variant = card.get("variant", {})
    if variant.get("tag", "published") != "published":
        errors.append(
            f"{model}: registry variant.tag drifted: got {variant.get('tag')!r}, expected omitted or 'published'"
        )
    if variant.get("quantization") != entry["recommended_quant"]:
        errors.append(
            f"{model}: registry variant.quantization drifted: "
            f"got {variant.get('quantization')!r}, expected {entry['recommended_quant']!r}"
        )


def check_machine_catalog_entry(model: str, entry: dict, machine_model: dict, errors: list[str]) -> None:
    registry_id = entry["registry_id"]
    languages = languages_for_model(entry)
    expected_scalars = {
        "id": registry_id,
        "kind": entry["kind"],
        "display_name": entry["display_name"],
        "family": entry["family"],
        "pull_alias": entry["pull_alias"],
        "size": entry["size"],
        "languages": languages,
        "license": entry["license_name"],
        "license_url": entry["license_source"],
        "license_class": entry["license_class"],
        "hf_repo": entry["hf_repo"],
        "recommended_quant": entry["recommended_quant"],
        "pull_recommended": f"{registry_id}:{QUANT_METADATA[entry['recommended_quant']].suffix}",
    }
    # language_mode/language_default are omitted entirely (not just falsy) for
    # kinds language_mode_for_model() returns {} for; compare against None so a
    # spuriously-added field on those entries is still caught as drift.
    expected_language_mode = language_mode_for_model(entry, languages)
    expected_scalars["language_mode"] = expected_language_mode.get("language_mode")
    expected_scalars["language_default"] = expected_language_mode.get("language_default")
    for key, value in expected_scalars.items():
        if machine_model.get(key) != value:
            errors.append(f"{model}: catalog {key} drifted: got {machine_model.get(key)!r}, expected {value!r}")
    if machine_model.get("aliases") != entry["aliases"]:
        errors.append(f"{model}: catalog aliases drifted")
    if machine_model.get("capability") != entry.get("capability"):
        errors.append(
            f"{model}: catalog capability drifted: got {machine_model.get('capability')!r}, "
            f"expected {entry.get('capability')!r}"
        )
    expected_sort_weight = entry.get("sort_weight", 0)
    if machine_model.get("sort_weight", 0) != expected_sort_weight:
        errors.append(
            f"{model}: catalog sort_weight drifted: got {machine_model.get('sort_weight', 0)!r}, "
            f"expected {expected_sort_weight!r}"
        )
    expected_recommended = entry.get("recommended") is True
    if (machine_model.get("recommended") is True) != expected_recommended:
        errors.append(
            f"{model}: catalog recommended drifted: got {machine_model.get('recommended')!r}, "
            f"expected {expected_recommended!r}"
        )
    expected_prose_locales = prose_locales_block(model, read_prose(model))
    if machine_model.get("prose_locales") != expected_prose_locales:
        errors.append(
            f"{model}: catalog prose_locales drifted from cards/{model}.toml "
            "(re-run regenerate_all.sh to pick up card changes)"
        )
    for key in (
        "experimental",
        "source_langs",
        "target_langs",
        "upstream_base_repo",
        "upstream_base_revision",
        "upstream_gguf_repo",
        "upstream_gguf_revision",
        "license_files",
        "upstream_release_date",
    ):
        if key in entry and machine_model.get(key) != entry[key]:
            errors.append(
                f"{model}: catalog {key} drifted: got {machine_model.get(key)!r}, "
                f"expected {entry[key]!r}"
            )
    quants = machine_model.get("quants")
    if not isinstance(quants, list):
        errors.append(f"{model}: catalog quants is not a list")
        return
    observed = {
        quant.get("quant"): {
            "quant": quant.get("quant"),
            "suffix": quant.get("suffix"),
            "pull": quant.get("pull"),
            "recommended": quant.get("recommended"),
        }
        for quant in quants
        if isinstance(quant, dict)
    }
    expected = {
        quant: expected_catalog_quant(registry_id, quant, entry["recommended_quant"])
        for quant in entry["quants"]
    }
    if observed != expected:
        errors.append(f"{model}: catalog quants drifted: got {observed!r}, expected {expected!r}")


def main(argv: list[str]) -> int:
    publish_catalog = load_publish_catalog()
    selected = argv or sorted(publish_catalog)
    unknown = sorted(set(selected) - set(publish_catalog))
    if unknown:
        raise SystemExit(f"unknown model(s): {', '.join(unknown)}")

    machine_catalog = load_json(REPO_ROOT / "model-registry" / "catalog.json")
    machine_by_id = {
        model.get("id"): model
        for model in machine_catalog.get("models", [])
        if isinstance(model, dict)
    }
    errors: list[str] = []
    skipped: list[str] = []

    translated_cards: list[str] = []
    try:
        translated_cards = validate_all_card_prose_locales()
    except KeyError as error:
        errors.append(str(error))

    # The signed catalog's top-level language/dialect label map is generated
    # data (source = _catalog.LANGUAGE_DISPLAY_LABELS, itself pinned to Rust's
    # language_display_label by a drift test). A hand-edit or a stale map that
    # no longer matches the Python source fails the drift gate loudly. Only
    # checked once a language_labels map exists so a label-less catalog is fine.
    expected_language_labels = language_labels_wire()
    actual_language_labels = machine_catalog.get("language_labels")
    if actual_language_labels is not None and actual_language_labels != expected_language_labels:
        errors.append(
            "catalog language_labels drifted from _catalog.LANGUAGE_DISPLAY_LABELS "
            "(re-run: _catalog.py write-language-labels model-registry/catalog.json)"
        )
    for model in selected:
        entry = publish_catalog[model]
        machine_model = machine_by_id.get(entry["registry_id"])
        if machine_model is not None:
            check_registry_card(model, entry, errors)
            check_machine_catalog_entry(model, entry, machine_model, errors)
        else:
            skipped.append(f"{model} (registry_id={entry['registry_id']})")

    for skip in skipped:
        # A source model with no catalog.json entry is not necessarily an error
        # (staged-publish lag is legitimate), but it must be visible so a dropped
        # or mistyped entry does not pass silently.
        print(
            f"warning: {skip} has no catalog.json entry; drift not checked",
            file=sys.stderr,
        )

    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print(f"catalog drift check passed for {len(selected)} model(s)")
    print(f"prose_locales check passed for {len(translated_cards)} model(s): {', '.join(translated_cards)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
