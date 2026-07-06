#!/usr/bin/env python3
"""Upsert one published model into model-registry/catalog.json.

  _manifest.py <model-id> [--hf-revision <commit-sha>] [--hf-repo <owner/name>] [--public]

This is the machine catalog writer for the model-distribution branch. It reads
the publish catalog, per-quant conversion sidecars, benchmark metrics, resolved
HF repo id, and prose card, then atomically rewrites catalog.json. The script is
deliberately strict: missing artifact sidecars, missing metrics, non-HTTPS URLs,
non-hex sha256 values, and failed public-listing gates fail before any catalog
is written.
"""
from __future__ import annotations

import argparse
import re
import sys
from datetime import date, datetime, timezone
from pathlib import Path

from _catalog import (
    CATALOG_SCHEMA_VERSION,
    CATALOG_URL,
    DEFAULT_MIN_CLI_VERSION,
    QUANT_METADATA,
    language_mode_for_model,
    languages_for_model,
    load as load_publish_catalog,
    punctuation_for_model,
    validate_card_prose_locales,
)
from _file_loaders import atomic_write_json, load_required_json, load_toml
from _pathlib_helpers import repo_root
from public_gate import DEFAULT_PUBLIC_NAMESPACE, PublicGateError, validate_public_model

SCRIPT_DIR = Path(__file__).resolve().parent
TOOLING_ROOT = SCRIPT_DIR.parent
HF_REPO_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*")
GIT_REVISION_RE = re.compile(r"[0-9a-fA-F]{40}")
LICENSE_CLASS_BY_NAME = {
    "mit": "permissive",
    "apache-2.0": "permissive",
    "apache 2.0": "permissive",
    "cc-by": "permissive",
    "cc-by-4.0": "permissive",
    "cc-by-nc": "noncommercial",
    "cc-by-nc-4.0": "noncommercial",
}


REPO_ROOT = repo_root(SCRIPT_DIR)


def work_root(model: str) -> Path:
    return REPO_ROOT / "tmp" / "publish" / model


def result_json(model: str, quant: str) -> Path:
    return work_root(model) / "packs" / f"{model}.{quant}.result.json"


def read_resolved_repo(model: str, catalog_entry: dict, explicit: str | None) -> str:
    if explicit:
        repo = explicit.strip()
    else:
        path = work_root(model) / "hf_repo.txt"
        repo = path.read_text().strip() if path.exists() else catalog_entry["hf_repo"]
    if not HF_REPO_RE.fullmatch(repo):
        raise SystemExit(f"HF repo must be '<owner>/<name>', got: {repo}")
    return repo


def read_git_revision(model: str, explicit: str | None, sidecar: str, label: str) -> str:
    if explicit:
        revision = explicit.strip()
    else:
        path = work_root(model) / sidecar
        revision = path.read_text().strip() if path.exists() else ""
    if not GIT_REVISION_RE.fullmatch(revision):
        raise SystemExit(
            f"{label} revision must be an immutable 40-hex commit sha. Pass --{label.lower()}-revision "
            f"or write tmp/publish/<model>/{sidecar} after publishing."
        )
    return revision


def read_hf_revision(model: str, explicit: str | None) -> str:
    return read_git_revision(model, explicit, "hf_revision.txt", "HF")


def read_prose(model: str) -> dict:
    path = TOOLING_ROOT / "cards" / f"{model}.toml"
    return load_toml(path) if path.exists() else {}


def validate_public_prose(model: str, prose: dict) -> None:
    """`--public` entries must ship real marketing prose, not a silently empty
    card. A missing card, or a card missing tagline/intro, degrades quietly to
    empty strings for staging/private entries (see `prose_block`) -- that
    fallback is not acceptable once a model is public-facing.
    """
    path = TOOLING_ROOT / "cards" / f"{model}.toml"
    if not path.exists():
        raise SystemExit(f"{model}: --public requires a prose card at {path.relative_to(REPO_ROOT)}")
    tagline = prose.get("tagline", "")
    if not isinstance(tagline, str) or not tagline.strip():
        raise SystemExit(f"{model}: --public requires a non-empty 'tagline' in {path.relative_to(REPO_ROOT)}")
    intro = prose.get("intro", "")
    if not isinstance(intro, str) or not intro.strip():
        raise SystemExit(f"{model}: --public requires a non-empty 'intro' in {path.relative_to(REPO_ROOT)}")


def prose_block(prose: dict) -> dict:
    intro = prose.get("intro", "").strip()
    overview = [intro] if intro else []
    return {
        "tagline": prose.get("tagline", ""),
        "overview": overview,
        "highlights": prose.get("highlights", []),
    }


def prose_locales_block(model: str, prose: dict) -> dict | None:
    """Build the catalog `prose_locales` field from a card's authored
    `[prose_locales."<bcp47>"]` tables. First iteration: tagline + highlights
    only (no overview). Validates format + staleness before emitting, so a
    stale/malformed translation fails the regen rather than shipping silently.
    """
    locales = prose.get("prose_locales")
    if not locales:
        return None
    validate_card_prose_locales(model, prose)
    return {
        locale: {
            "tagline": block.get("tagline"),
            "highlights": block.get("highlights", []),
        }
        for locale, block in sorted(locales.items())
    }


def license_class(entry: dict) -> str:
    if "license_class" in entry:
        value = entry["license_class"]
    else:
        value = LICENSE_CLASS_BY_NAME.get(entry["license_name"].strip().lower(), "gated")
    if value not in {"permissive", "noncommercial", "gated"}:
        raise SystemExit(f"unsupported license_class '{value}'")
    return value


def quant_entry(
    model: str,
    registry_id: str,
    quant: str,
    hf_repo: str,
    hf_revision: str,
    metrics: dict,
) -> dict:
    suffix = QUANT_METADATA[quant].suffix
    result = load_required_json(result_json(model, quant))
    metric = metrics.get("quants", {}).get(quant)
    if metric is None:
        raise SystemExit(f"metrics.json missing quant '{quant}'")
    sha = result.get("sha256", "")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", sha):
        raise SystemExit(f"{result_json(model, quant)} has invalid sha256")
    size = result.get("size_bytes")
    if not isinstance(size, int) or size <= 0:
        raise SystemExit(f"{result_json(model, quant)} has invalid size_bytes")
    metric_size = metric.get("size_bytes")
    if metric_size is not None and metric_size != size:
        raise SystemExit(
            f"metrics.json size_bytes for '{quant}' ({metric_size}) does not match result.json ({size})"
        )
    # Bench-time binding: metrics.json must record the sha256 of the exact pack
    # it benchmarked. No back-compat fallback for older metrics.json files
    # without a sha256 -- a missing or mismatched value means the numbers may
    # not describe the pack being published, so bench must be re-run.
    metric_sha = metric.get("sha256")
    if not isinstance(metric_sha, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", metric_sha):
        raise SystemExit(
            f"metrics.json for '{quant}' is missing a valid pack sha256; re-run bench to record it "
            f"({work_root(model) / 'metrics.json'})"
        )
    if metric_sha.lower() != sha.lower():
        raise SystemExit(
            f"metrics.json sha256 for '{quant}' ({metric_sha}) does not match result.json sha256 "
            f"({sha}); the pack changed since the last bench run -- re-run bench before publishing"
        )
    filename = Path(result.get("pack", f"{model}-{quant}.oasr")).name
    if not filename.endswith(".oasr"):
        raise SystemExit(f"{result_json(model, quant)} has invalid pack filename")
    entry = {
        "quant": quant,
        "suffix": suffix,
        "pull": f"{registry_id}:{suffix}",
        "filename": filename,
        "url": f"https://huggingface.co/{hf_repo}/resolve/{hf_revision}/{filename}",
        "sha256": sha.lower(),
        "size_bytes": size,
        "recommended": False,
        "perf": {
            "rtf_cpu": metric.get("rtf_cpu"),
            "rtf_metal": metric.get("rtf_metal"),
            "peak_rss_bytes": metric.get("peak_rss_bytes"),
            "jfk_wer_vs_fp16": metric.get("jfk_wer_vs_fp16"),
        },
    }
    return entry


def build_catalog_model(model: str, entry: dict, args: argparse.Namespace) -> dict:
    registry_id = entry["registry_id"]
    hf_repo = read_resolved_repo(model, entry, args.hf_repo)
    hf_revision = read_hf_revision(model, args.hf_revision)
    metrics = load_required_json(work_root(model) / "metrics.json")
    prose = read_prose(model)
    if args.public:
        validate_public_prose(model, prose)
    quants = [
        quant_entry(
            model,
            registry_id,
            quant,
            hf_repo,
            hf_revision,
            metrics,
        )
        for quant in entry["quants"]
    ]
    recommended = entry["recommended_quant"]
    # Denormalized signed-catalog wire fields. The authoring source stays
    # models-core.toml:recommended_quant; `recommended_quant`,
    # `pull_recommended`, and each quant's `recommended` flag are generated
    # together for distinct Rust/TS/display consumers.
    for quant in quants:
        quant["recommended"] = quant["quant"] == recommended
    if not any(quant["recommended"] for quant in quants):
        raise SystemExit(f"recommended_quant '{recommended}' was not emitted for {model}")
    recommended_suffix = QUANT_METADATA[recommended].suffix

    model_entry = {
        "id": registry_id,
        "kind": entry["kind"],
    }
    if "capability" in entry:
        model_entry["capability"] = dict(entry["capability"])
    languages = languages_for_model(entry)
    model_entry.update({
        "display_name": entry["display_name"],
        "family": entry["family"],
        "aliases": entry.get("aliases", []),
        "pull_alias": entry.get("pull_alias"),
        "size": entry["size"],
        "languages": languages,
        "vendor": entry["upstream_repo"].split("/", 1)[0],
        "license": entry["license_name"],
        "license_url": entry["license_source"],
        "license_class": license_class(entry),
        "hf_repo": hf_repo,
        "hf_revision": hf_revision,
        "public": bool(args.public),
        "min_cli_version": args.min_cli_version,
        "recommended_quant": recommended,
        "pull_recommended": f"{registry_id}:{recommended_suffix}",
        "prose": prose_block(prose),
        "quants": quants,
    })
    # Per-model source-language parameter policy, derived from core's
    # LanguageMode for this family (asr-model only; see
    # language_mode_for_model()'s docstring for why translation-model /
    # capability-pack entries omit it).
    model_entry.update(language_mode_for_model(entry, languages))
    # Whether this model's transcripts include punctuation, derived from its
    # family (asr-model only; see punctuation_for_model()'s docstring for why
    # translation-model / capability-pack entries omit it).
    model_entry.update(punctuation_for_model(entry))
    if entry.get("experimental") is True:
        model_entry["experimental"] = True
    # Explicit, author-set display hints (models-core.toml `sort_weight`/
    # `recommended`; no threshold inference). Only serialized when set so
    # unmarked models keep the Rust-side serde defaults (0 / false).
    if "sort_weight" in entry:
        model_entry["sort_weight"] = entry["sort_weight"]
    if entry.get("recommended") is True:
        model_entry["recommended"] = True
    prose_locales = prose_locales_block(model, prose)
    if prose_locales is not None:
        model_entry["prose_locales"] = prose_locales
    for key in (
        "source_langs",
        "target_langs",
        "upstream_base_repo",
        "upstream_base_revision",
        "upstream_gguf_repo",
        "upstream_gguf_revision",
        "license_files",
        "upstream_release_date",
        "min_core_version",
    ):
        if key in entry:
            model_entry[key] = entry[key]
    return model_entry


def ensure_release_public_allowed(model: str, entry: dict, public: bool) -> None:
    if public and entry.get("release_public") is not True:
        raise SystemExit(
            f"{model} is not marked release_public=true in models-core.toml; refusing --public"
        )


def load_catalog(path: Path) -> dict:
    if not path.exists():
        return {
            "schema_version": CATALOG_SCHEMA_VERSION,
            "generated_at": "",
            "catalog_url": CATALOG_URL,
            "models": [],
        }
    data = load_required_json(path)
    if data.get("schema_version") != CATALOG_SCHEMA_VERSION:
        raise SystemExit(f"unsupported catalog schema_version in {path}")
    if not isinstance(data.get("models"), list):
        raise SystemExit(f"catalog models must be a list in {path}")
    return data


def model_sort_key(item: dict) -> tuple:
    """Display-order sort key for catalog `models[]`.

    Primary key is `sort_weight` (higher first). Within an equal `sort_weight`,
    a newer `upstream_release_date` sorts first and models with no date sort
    after any dated model; `id` breaks any remaining tie for determinism. The
    date is a TIEBREAKER only and never overrides `sort_weight`. ISO
    `yyyy-mm-dd` ordinals negate so a newer (larger) date sorts earlier, and the
    `has_date` flag keeps nulls last regardless of the (then-unused) ordinal.
    """
    raw_date = item.get("upstream_release_date")
    has_date = raw_date is not None
    ordinal = date.fromisoformat(raw_date).toordinal() if has_date else 0
    return (-item.get("sort_weight", 0), not has_date, -ordinal, item["id"])


def upsert_model(catalog: dict, model_entry: dict) -> dict:
    models = [model for model in catalog["models"] if model.get("id") != model_entry["id"]]
    models.append(model_entry)
    # Catalog array order is the display order consumers inherit for free
    # (market/desktop listings render models[] as-is). See model_sort_key.
    models.sort(key=model_sort_key)
    catalog["schema_version"] = CATALOG_SCHEMA_VERSION
    catalog["generated_at"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    catalog["catalog_url"] = CATALOG_URL
    catalog["models"] = models
    return catalog


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--hf-revision")
    parser.add_argument("--hf-repo")
    parser.add_argument("--public", action="store_true")
    parser.add_argument("--public-namespace", default=DEFAULT_PUBLIC_NAMESPACE)
    parser.add_argument("--public-gate-timeout", type=float, default=60.0)
    parser.add_argument("--min-cli-version", default=DEFAULT_MIN_CLI_VERSION)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=REPO_ROOT / "model-registry" / "catalog.json",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    catalog_entries = load_publish_catalog()
    if args.model not in catalog_entries:
        raise SystemExit(f"unknown model '{args.model}'")
    ensure_release_public_allowed(args.model, catalog_entries[args.model], args.public)
    model_entry = build_catalog_model(args.model, catalog_entries[args.model], args)
    if args.public:
        try:
            validate_public_model(
                model_entry,
                expected_namespace=args.public_namespace,
                timeout=args.public_gate_timeout,
            )
        except PublicGateError as error:
            raise SystemExit(f"public-listing gate failed: {error}") from None
    catalog = upsert_model(load_catalog(args.catalog), model_entry)
    atomic_write_json(args.catalog, catalog)
    print(args.catalog)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
