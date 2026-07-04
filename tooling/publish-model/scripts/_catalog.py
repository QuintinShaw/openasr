#!/usr/bin/env python3
"""Catalog reader for the OpenASR publishing harness.

Single source of truth = tooling/publish-model/models-core.toml plus
tooling/publish-model/models-publish.toml. Bash scripts shell out to this for
field lookups and quant-token mapping so the catalog is parsed in one place
(Python 3.11+ stdlib tomllib) rather than re-implemented in fragile shell.

Usage:
  _catalog.py field   <model> <key>     # print one catalog value (lists -> space-joined)
  _catalog.py quants  <model>           # print the quant ids, one per line
  _catalog.py token   <quant_id>        # internal quant id -> CLI --quantization token
  _catalog.py suffix  <quant_id>        # internal quant id -> pull-grammar suffix (fp16/q8/q4)
  _catalog.py models                    # list all model ids
  _catalog.py json    <model>           # full entry as JSON (with id injected)
  _catalog.py prose-locale-hash <model> # compute source_sha256 for cards/<model>.toml's EN tagline+highlights
  _catalog.py check-prose-locales       # validate every card's prose_locales block (format + staleness)
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from _file_loaders import load_toml
from _pathlib_helpers import repo_root

PUB = Path(__file__).resolve().parent
REPO_ROOT = repo_root(PUB)
CATALOG_CORE = REPO_ROOT / "tooling" / "publish-model" / "models-core.toml"
CATALOG_PUBLISH = REPO_ROOT / "tooling" / "publish-model" / "models-publish.toml"
CATALOG_SERIES = REPO_ROOT / "crates" / "openasr-core" / "catalog-series.toml"
CARDS_DIR = REPO_ROOT / "tooling" / "publish-model" / "cards"
CATALOG = CATALOG_CORE
CATALOG_URL = "https://catalog.openasr.org/v1/catalog.json"
CATALOG_SCHEMA_VERSION = 1
DEFAULT_MIN_CLI_VERSION = "0.1.0"
REGISTRY_CARD_DEFAULTS = {
    "default_variant": "published",
}
DEFAULT_CATALOG_MODEL_KIND = "asr-model"
SUPPORTED_CATALOG_MODEL_KINDS = {"asr-model", "capability-pack", "translation-model"}
SUPPORTED_CAPABILITY_ROLES = {"speaker-embedder", "speaker-segmenter"}
GIT_REVISION_RE = re.compile(r"[0-9a-fA-F]{40}")
TRANSLATION_REQUIRED_LICENSE_FILES = {"LICENSE.txt", "NOTICE.openasr.txt"}


@dataclass(frozen=True)
class QuantMetadata:
    cli_token: str
    suffix: str
    label: str


QUANT_METADATA = {
    # Raw f32 remains a catalog-declared variant for published diarization
    # support packs. The Rust canonical_quant_tag passes unknown tags through
    # unchanged, so "f32" needs no new match arm there.
    "f32": QuantMetadata(cli_token="f32", suffix="f32", label="f32"),
    "fp16": QuantMetadata(cli_token="fp16", suffix="fp16", label="fp16"),
    "q8_0": QuantMetadata(cli_token="q8-0", suffix="q8", label="q8_0"),
    "q4_k": QuantMetadata(cli_token="q4-k", suffix="q4", label="q4_k"),
    # Product quant name for mixed-tensor GGUF files such as Hy-MT2 Q4_K_M.
    # This is catalog/pack metadata only: the runtime still sees ordinary GGUF
    # tensor types (Q4_K, Q6_K, F32) and does not gain a new matmul type.
    "q4_k_m": QuantMetadata(cli_token="q4-k-m", suffix="q4km", label="Q4_K_M"),
    "q3_k": QuantMetadata(cli_token="q3-k", suffix="q3", label="q3_k"),
}
# Per-family list of the natural languages a model officially supports, as
# ISO 639-1 two-letter codes (ISO 639-3 where no 639-1 code exists), sorted.
# RULE: LANGUAGES ONLY, NOT DIALECTS/ACCENTS. Chinese is a single language "zh"
# (Mandarin/Cantonese/Wu/Min and regional dialects all collapse into "zh");
# English is "en" (US/UK/etc. are not split). A card that advertises "30 languages
# and 22 Chinese dialects" yields the 30 languages, with the dialects folded into
# the single "zh". If a model supports N languages, list all N. See SKILL.md.
LANG_BY_FAMILY = {
    # Qwen3-ASR card lists 30 languages; Cantonese folds into zh -> 29 ISO langs.
    "qwen": [
        "ar", "cs", "da", "de", "el", "en", "es", "fa", "fi", "fil", "fr", "hi",
        "hu", "id", "it", "ja", "ko", "mk", "ms", "nl", "pl", "pt", "ro", "ru",
        "sv", "th", "tr", "vi", "zh",
    ],
    # CohereLabs cohere-transcribe card: trained on 14 languages.
    "cohere": [
        "ar", "de", "el", "en", "es", "fr", "it", "ja", "ko", "nl", "pl", "pt",
        "vi", "zh",
    ],
    # OpenAI Whisper tokenizer LANGUAGES dict; Cantonese->zh, Nynorsk->no,
    # jw->jv normalized -> 98 distinct ISO languages (haw is ISO 639-3).
    "whisper": [
        "af", "am", "ar", "as", "az", "ba", "be", "bg", "bn", "bo", "br", "bs",
        "ca", "cs", "cy", "da", "de", "el", "en", "es", "et", "eu", "fa", "fi",
        "fo", "fr", "gl", "gu", "ha", "haw", "he", "hi", "hr", "ht", "hu", "hy",
        "id", "is", "it", "ja", "jv", "ka", "kk", "km", "kn", "ko", "la", "lb",
        "ln", "lo", "lt", "lv", "mg", "mi", "mk", "ml", "mn", "mr", "ms", "mt",
        "my", "ne", "nl", "no", "oc", "pa", "pl", "ps", "pt", "ro", "ru", "sa",
        "sd", "si", "sk", "sl", "sn", "so", "sq", "sr", "su", "sv", "sw", "ta",
        "te", "tg", "th", "tk", "tl", "tr", "tt", "uk", "ur", "uz", "vi", "yi",
        "yo", "zh",
    ],
    # X-ASR-zh-en: bilingual Chinese + English.
    "xasr-zipformer": ["en", "zh"],
    "moonshine": ["en"],
    "parakeet": ["en"],
    "wav2vec2": ["en"],
}


def load() -> dict:
    core = load_toml(CATALOG_CORE)
    publish = load_toml(CATALOG_PUBLISH)
    series = load_catalog_series()
    unknown_publish_models = sorted(set(publish) - set(core))
    if unknown_publish_models:
        raise KeyError(
            "publish-only model(s) missing from models-core.toml: "
            + ", ".join(unknown_publish_models)
        )
    merged = {model: dict(entry) for model, entry in core.items()}
    for model, entry in publish.items():
        overlap = sorted(set(merged[model]) & set(entry))
        if overlap:
            raise KeyError(
                f"publish-only entry '{model}' duplicates core key(s): {', '.join(overlap)}"
            )
        merged[model].update(entry)
    for model, entry in merged.items():
        apply_catalog_series_defaults(model, entry, series)
    return merged


def load_catalog_series() -> dict:
    return load_toml(CATALOG_SERIES)


def apply_catalog_series_defaults(model: str, entry: dict, series: dict) -> None:
    kind = entry.get("kind", DEFAULT_CATALOG_MODEL_KIND)
    if kind not in SUPPORTED_CATALOG_MODEL_KINDS:
        raise KeyError(
            f"model '{model}' has unsupported kind '{kind}'. "
            f"Known kinds: {', '.join(sorted(SUPPORTED_CATALOG_MODEL_KINDS))}"
        )
    entry["kind"] = kind
    validate_capability(model, entry)
    validate_translation_model(model, entry)
    validate_display_ranking(model, entry)

    spec = series.get(entry["family"])
    if spec is not None and entry["size"] not in spec["member_sizes"]:
        raise KeyError(
            f"model '{model}' size '{entry['size']}' is not listed in "
            f"catalog-series.toml family '{entry['family']}'"
        )
    if "aliases" not in entry:
        entry["aliases"] = list(spec.get("catalog_aliases", [])) if spec is not None else []
    if "pull_alias" not in entry:
        entry["pull_alias"] = spec.get("catalog_pull_alias") if spec is not None else None


def validate_capability(model: str, entry: dict) -> None:
    capability = entry.get("capability")
    if entry["kind"] == "capability-pack":
        if not isinstance(capability, dict):
            raise KeyError(f"model '{model}' is kind=capability-pack but has no capability table")
        feature = capability.get("feature")
        role = capability.get("role")
        if not isinstance(feature, str) or not feature.strip():
            raise KeyError(f"model '{model}' capability.feature must be a non-empty string")
        if role not in SUPPORTED_CAPABILITY_ROLES:
            raise KeyError(
                f"model '{model}' capability.role '{role}' is unsupported. "
                f"Known roles: {', '.join(sorted(SUPPORTED_CAPABILITY_ROLES))}"
            )
    elif capability is not None:
        raise KeyError(f"model '{model}' has capability metadata but kind is not capability-pack")


def validate_translation_model(model: str, entry: dict) -> None:
    if entry["kind"] != "translation-model":
        if "source_langs" in entry or "target_langs" in entry:
            raise KeyError(
                f"model '{model}' has translation metadata but kind is not translation-model"
            )
        return

    validate_lang_list(model, "source_langs", entry.get("source_langs"))
    validate_lang_list(model, "target_langs", entry.get("target_langs"))
    overlap = sorted(set(entry["source_langs"]) & set(entry["target_langs"]))
    if overlap:
        raise KeyError(
            f"model '{model}' source_langs and target_langs must not overlap: {', '.join(overlap)}"
        )

    if entry.get("license_name") != "Apache-2.0":
        raise KeyError(f"model '{model}' translation model license_name must be Apache-2.0")
    if entry.get("license_class") != "permissive":
        raise KeyError(f"model '{model}' translation model license_class must be permissive")

    license_files = entry.get("license_files")
    if not isinstance(license_files, list):
        raise KeyError(f"model '{model}' translation model must declare license_files")
    missing_license_files = sorted(TRANSLATION_REQUIRED_LICENSE_FILES - set(license_files))
    if missing_license_files:
        raise KeyError(
            f"model '{model}' translation model license_files missing: "
            + ", ".join(missing_license_files)
        )

    notice_file = entry.get("notice_file")
    if not isinstance(notice_file, str) or not notice_file.strip():
        raise KeyError(f"model '{model}' translation model must declare notice_file")
    notice_path = REPO_ROOT / notice_file
    if not notice_path.is_file():
        raise KeyError(f"model '{model}' notice_file does not exist: {notice_file}")
    notice = notice_path.read_text(encoding="utf-8")
    for required in ("repackaged", ".oasr", "LICENSE.txt", "NOTICE.openasr.txt"):
        if required not in notice:
            raise KeyError(
                f"model '{model}' notice_file must mention {required!r}: {notice_file}"
            )

    for field in ("upstream_base_repo", "upstream_gguf_repo"):
        value = entry.get(field)
        if not isinstance(value, str) or "/" not in value:
            raise KeyError(f"model '{model}' translation model must declare {field}")
    for field in ("upstream_base_revision", "upstream_gguf_revision"):
        value = entry.get(field)
        if not isinstance(value, str) or GIT_REVISION_RE.fullmatch(value) is None:
            raise KeyError(f"model '{model}' translation model {field} must be a 40-hex revision")

    source_revision = entry.get("source_revision")
    if source_revision != entry["upstream_gguf_revision"]:
        raise KeyError(
            f"model '{model}' source_revision must equal upstream_gguf_revision "
            f"({entry['upstream_gguf_revision']})"
        )
    if entry.get("upstream_repo") != entry["upstream_gguf_repo"]:
        raise KeyError(
            f"model '{model}' upstream_repo must equal upstream_gguf_repo "
            f"({entry['upstream_gguf_repo']})"
        )


def validate_display_ranking(model: str, entry: dict) -> None:
    """`sort_weight`/`recommended` are explicit, author-set display hints (no
    threshold inference from perf/WER data). Both are optional; the catalog
    defaults are sort_weight=0, recommended=false (see registry.rs CatalogModel).
    """
    if "sort_weight" in entry:
        value = entry["sort_weight"]
        if isinstance(value, bool) or not isinstance(value, int):
            raise KeyError(f"model '{model}' sort_weight must be an int, got {value!r}")
    if "recommended" in entry:
        value = entry["recommended"]
        if not isinstance(value, bool):
            raise KeyError(f"model '{model}' recommended must be a bool, got {value!r}")


def validate_lang_list(model: str, field: str, value: object) -> None:
    if not isinstance(value, list) or not value:
        raise KeyError(f"model '{model}' {field} must be a non-empty list")
    if value != sorted(set(value)):
        raise KeyError(f"model '{model}' {field} must be sorted and de-duplicated")
    for code in value:
        if not isinstance(code, str) or re.fullmatch(r"[a-z]{2,3}", code) is None:
            raise KeyError(
                f"model '{model}' {field} contains invalid ISO language code: {code!r}"
            )


def languages_for_family(family: str) -> list[str]:
    try:
        return list(LANG_BY_FAMILY[family])
    except KeyError as error:
        known = ", ".join(sorted(LANG_BY_FAMILY))
        raise KeyError(f"unknown model family '{family}'. Known language mappings: {known}") from error


def languages_for_model(entry: dict) -> list[str]:
    """Resolve the languages a specific model supports.

    Language support is a per-MODEL property: a model may support fewer languages
    than its family default (e.g. Whisper's English-only `*.en` checkpoints support
    only `en` even though the multilingual Whisper family supports ~98). A model
    that needs to differ from the family default declares an explicit `languages`
    list in models-core.toml; otherwise it inherits `LANG_BY_FAMILY[family]`.
    """
    override = entry.get("languages")
    if override is not None:
        if not isinstance(override, list) or not override:
            raise ValueError(
                f"model '{entry.get('id', '?')}' has an invalid 'languages' override; "
                "expected a non-empty list of ISO language codes"
            )
        # De-dup + sort so the override obeys the same invariant as family lists.
        return sorted(set(override))
    return languages_for_family(entry["family"])


# --- prose_locales machine checks -------------------------------------------
#
# First-iteration scope is tagline + highlights only (no `overview`/intro
# translation yet). Each locale block is authored in tooling/publish-model/
# cards/<id>.toml under a `[prose_locales."<bcp47>"]` table (e.g.
# `[prose_locales."zh-CN"]`) alongside the canonical English `tagline` /
# `highlights`. These checks are deliberately mechanical (formatting +
# staleness), not a translation-quality gate: a human still reviews the prose.

BOLD_MARKER = "**"
# Loosely "a number-shaped token": digits, then digit-ish punctuation
# (.,/exponent/multiply/percent), then a trailing unit-ish letter run (27M,
# 680k, 1.55B, 7e-5, ...). Good enough to catch a translator dropping or
# changing a figure; it is a drift detector, not a strict tokenizer.
NUMBER_TOKEN_RE = re.compile(r"[0-9][0-9.,eE×xX%]*[A-Za-z]*")
PROSE_LOCALE_OPTIONAL_FIELDS = {"tagline", "highlights", "source_sha256"}


def _leading_emoji(text: str) -> str:
    stripped = text.strip()
    return stripped[:1] if stripped else ""


def _number_tokens(text: str) -> list[str]:
    return NUMBER_TOKEN_RE.findall(text)


def prose_locale_source_text(tagline: str, highlights: list[str]) -> str:
    """Normalized English source text a locale's `source_sha256` is over."""
    parts = [tagline.strip()] + [item.strip() for item in highlights]
    return "\n".join(parts)


def prose_locale_source_sha256(tagline: str, highlights: list[str]) -> str:
    text = prose_locale_source_text(tagline, highlights)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _validate_prose_line_pair(
    model: str,
    locale: str,
    label: str,
    en_text: str,
    translated_text: str,
    *,
    check_leading_emoji: bool = True,
) -> None:
    if en_text.count(BOLD_MARKER) != translated_text.count(BOLD_MARKER):
        raise KeyError(
            f"model '{model}' prose_locales.{locale} {label}: '**' bold-marker count drifted from English"
        )
    if en_text.count("`") != translated_text.count("`"):
        raise KeyError(
            f"model '{model}' prose_locales.{locale} {label}: backtick count drifted from English"
        )
    # Only highlight lines carry a leading emoji by convention; the tagline is
    # plain prose, so its leading-character check is skipped.
    if check_leading_emoji and _leading_emoji(en_text) != _leading_emoji(translated_text):
        raise KeyError(
            f"model '{model}' prose_locales.{locale} {label}: leading emoji drifted from English "
            f"(expected {_leading_emoji(en_text)!r}, got {_leading_emoji(translated_text)!r})"
        )
    en_numbers = sorted(_number_tokens(en_text))
    translated_numbers = sorted(_number_tokens(translated_text))
    if en_numbers != translated_numbers:
        raise KeyError(
            f"model '{model}' prose_locales.{locale} {label}: numeric tokens drifted from English "
            f"(expected {en_numbers!r}, got {translated_numbers!r})"
        )


def validate_prose_locale_block(
    model: str,
    locale: str,
    en_tagline: str,
    en_highlights: list[str],
    block: dict,
) -> None:
    if "overview" in block:
        raise KeyError(
            f"model '{model}' prose_locales.{locale} must not include 'overview' "
            "(first iteration only translates tagline + highlights)"
        )
    unknown = sorted(set(block) - PROSE_LOCALE_OPTIONAL_FIELDS)
    if unknown:
        raise KeyError(f"model '{model}' prose_locales.{locale} has unknown field(s): {', '.join(unknown)}")

    translated_tagline = block.get("tagline")
    if not isinstance(translated_tagline, str) or not translated_tagline.strip():
        raise KeyError(f"model '{model}' prose_locales.{locale} tagline must be a non-empty string")
    _validate_prose_line_pair(
        model, locale, "tagline", en_tagline, translated_tagline, check_leading_emoji=False
    )

    translated_highlights = block.get("highlights")
    if not isinstance(translated_highlights, list):
        raise KeyError(f"model '{model}' prose_locales.{locale} highlights must be a list")
    if len(translated_highlights) != len(en_highlights):
        raise KeyError(
            f"model '{model}' prose_locales.{locale} highlights count {len(translated_highlights)} "
            f"does not match English count {len(en_highlights)}"
        )
    for index, (en_item, translated_item) in enumerate(zip(en_highlights, translated_highlights)):
        if not isinstance(translated_item, str) or not translated_item.strip():
            raise KeyError(f"model '{model}' prose_locales.{locale} highlight[{index}] must be a non-empty string")
        _validate_prose_line_pair(model, locale, f"highlight[{index}]", en_item, translated_item)

    expected_hash = prose_locale_source_sha256(en_tagline, en_highlights)
    actual_hash = block.get("source_sha256")
    if actual_hash != expected_hash:
        raise KeyError(
            f"model '{model}' prose_locales.{locale} translation stale: source_sha256 mismatch "
            f"(expected {expected_hash}, got {actual_hash!r}); English tagline/highlights changed since "
            "the translation was authored -- re-translate and update source_sha256 "
            f"(see: _catalog.py prose-locale-hash {model})"
        )


def validate_card_prose_locales(model: str, card: dict) -> None:
    locales = card.get("prose_locales")
    if not locales:
        return
    if not isinstance(locales, dict):
        raise KeyError(f"model '{model}' prose_locales must be a table of locale -> {{tagline, highlights}}")
    en_tagline = card.get("tagline", "")
    en_highlights = card.get("highlights", [])
    for locale, block in sorted(locales.items()):
        if not isinstance(block, dict):
            raise KeyError(f"model '{model}' prose_locales.{locale} must be a table")
        validate_prose_locale_block(model, locale, en_tagline, en_highlights, block)


def read_card(model: str) -> dict:
    path = CARDS_DIR / f"{model}.toml"
    return load_toml(path) if path.exists() else {}


def validate_all_card_prose_locales() -> list[str]:
    """Validate every authored card's prose_locales block. Returns the sorted
    list of model ids that declare at least one locale (for reporting)."""
    translated: list[str] = []
    for path in sorted(CARDS_DIR.glob("*.toml")):
        model = path.stem
        card = load_toml(path)
        if card.get("prose_locales"):
            translated.append(model)
        validate_card_prose_locales(model, card)
    return translated


def entry(model: str) -> dict:
    data = load()
    if model not in data:
        sys.exit(f"unknown model '{model}'. Known: {', '.join(sorted(data))}")
    e = dict(data[model])
    e["id"] = model
    return e


def main(argv: list[str]) -> int:
    if not argv:
        sys.exit(__doc__)
    cmd = argv[0]
    if cmd == "models":
        print("\n".join(sorted(load())))
    elif cmd == "field":
        val = entry(argv[1]).get(argv[2])
        if val is None:
            sys.exit(f"no key '{argv[2]}' for model '{argv[1]}'")
        if isinstance(val, bool):
            print("true" if val else "false")  # shell-friendly, not Python's True/False
        elif isinstance(val, list):
            print(" ".join(val))
        else:
            print(val)
    elif cmd == "quants":
        print("\n".join(entry(argv[1])["quants"]))
    elif cmd == "token":
        print(QUANT_METADATA[argv[1]].cli_token)
    elif cmd == "suffix":
        print(QUANT_METADATA[argv[1]].suffix)
    elif cmd == "json":
        print(json.dumps(entry(argv[1]), indent=2))
    elif cmd == "prose-locale-hash":
        card = read_card(argv[1])
        print(prose_locale_source_sha256(card.get("tagline", ""), card.get("highlights", [])))
    elif cmd == "check-prose-locales":
        translated = validate_all_card_prose_locales()
        print(f"prose_locales check passed for {len(translated)} model(s): {', '.join(translated)}")
    else:
        sys.exit(f"unknown command '{cmd}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
