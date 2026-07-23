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
  _catalog.py language-labels           # print the curated language/dialect label map as JSON
  _catalog.py write-language-labels <catalog.json>  # refresh the catalog's top-level language_labels map
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from dataclasses import dataclass
from datetime import date
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
SUPPORTED_CAPABILITY_ROLES = {
    "speaker-embedder",
    "speaker-segmenter",
    "forced-aligner",
    "punctuation-restorer",
}
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
# --- Recognition dialect codes + curated display labels ---------------------
#
# Python mirror of crate::models::language (Rust). REGISTERED_DIALECT_CODES and
# LANGUAGE_DISPLAY_LABELS are the single Python source of truth here; the label
# map is emitted into the signed catalog's top-level `language_labels`, and a
# Rust drift test (bundled_catalog_language_labels_match_rust_display_table)
# pins it back to `language_display_label` so the two languages cannot diverge
# (like the canonical quant-tag contract).

# Base-language codes carrying a `-region` subtag that a dialect-capable model
# may advertise as a selectable source language (the Chinese province tags
# Dolphin recognizes). Kept sorted + de-duplicated; must match Rust's
# REGISTERED_DIALECT_CODES exactly.
REGISTERED_DIALECT_CODES = [
    "zh-anhui",
    "zh-dongbei",
    "zh-fujian",
    "zh-gansu",
    "zh-guangdong",
    "zh-guizhou",
    "zh-hebei",
    "zh-henan",
    "zh-hubei",
    "zh-hunan",
    "zh-jiangsu",
    "zh-jiangxi",
    "zh-ningxia",
    "zh-shaanxi",
    "zh-shandong",
    "zh-shanghai",
    "zh-shanxi",
    "zh-sichuan",
    "zh-tianjin",
    "zh-tw",
    "zh-yunnan",
    "zh-zhejiang",
]

# code -> (English, Simplified-Chinese) display label. The Sinitic base codes
# whose ISO naming is unhelpful (`zh`/`yue`/`wuu`) plus every province dialect
# code, matching `language_display_label()` in crate::models::language 1:1.
LANGUAGE_DISPLAY_LABELS = {
    "zh": ("Chinese", "中文"),
    "yue": ("Cantonese", "粤语"),
    "wuu": ("Wu Chinese", "吴语"),
    "nan": ("Min Nan Chinese", "闽南语"),
    "zh-anhui": ("Chinese (Anhui)", "中文（安徽话）"),
    "zh-guangdong": ("Chinese (Guangdong)", "中文（广东话）"),
    "zh-hebei": ("Chinese (Hebei)", "中文（河北话）"),
    "zh-hubei": ("Chinese (Hubei)", "中文（湖北话）"),
    "zh-jiangsu": ("Chinese (Jiangsu)", "中文（江苏话）"),
    "zh-ningxia": ("Chinese (Ningxia)", "中文（宁夏话）"),
    "zh-shaanxi": ("Chinese (Shaanxi)", "中文（陕西话）"),
    "zh-shandong": ("Chinese (Shandong)", "中文（山东话）"),
    "zh-shanghai": ("Chinese (Shanghainese)", "中文（上海话）"),
    "zh-shanxi": ("Chinese (Shanxi)", "中文（山西话）"),
    "zh-sichuan": ("Chinese (Sichuanese)", "中文（四川话）"),
    "zh-tianjin": ("Chinese (Tianjin)", "中文（天津话）"),
    "zh-tw": ("Chinese (Taiwan)", "中文（台湾）"),
    "zh-henan": ("Chinese (Henan)", "中文（河南话）"),
    "zh-hunan": ("Chinese (Hunan)", "中文（湖南话）"),
    "zh-jiangxi": ("Chinese (Jiangxi)", "中文（江西话）"),
    "zh-fujian": ("Chinese (Fujian)", "中文（福建话）"),
    "zh-gansu": ("Chinese (Gansu)", "中文（甘肃话）"),
    "zh-guizhou": ("Chinese (Guizhou)", "中文（贵州话）"),
    "zh-yunnan": ("Chinese (Yunnan)", "中文（云南话）"),
    "zh-dongbei": ("Chinese (Northeastern)", "中文（东北话）"),
    "zh-zhejiang": ("Chinese (Zhejiang)", "中文（浙江话）"),
}

# Families that MAY enumerate dialect recognition codes in `languages`
# (selective dialect collapse). Every other family collapses regional dialects
# into the base language (`zh`), so a stray dialect code fails loudly instead
# of shipping. Two distinct routes earn membership here:
#  - dolphin: its executor ships a concrete code->prompt map, so a dialect
#    code is actually SELECTABLE per request.
#  - firered-aed / qwen: not selectable (firered-aed is a fixed bilingual
#    vocab with no language prompt; qwen self-detects and rejects an explicit
#    hint), but each has upstream-published, benchmark-verified per-dialect
#    recognition coverage (FireRedASR2 README + arXiv:2603.10420 Table 2's 19
#    dialect test sets; Qwen3-ASR's 22-dialect enumeration), so the dialect
#    codes describe RECOGNIZED capability rather than a selectable parameter
#    -- the same "recognizes whatever it hears" semantics FixedMultilingual
#    families like parakeet-tdt already use for their base language list.
DIALECT_CAPABLE_FAMILIES = {"dolphin", "firered-aed", "qwen"}

# Shape of a recognition-language code: a lowercase ISO 639 base (2-3 letters)
# with an OPTIONAL single `-region` subtag. Deliberately broader than the
# translation-only `[a-z]{2,3}` check (validate_lang_list), matching Rust's
# `validate_language_code` regex `^[a-z]{2,3}(-[a-z0-9]+)?$`.
RECOGNITION_LANGUAGE_CODE_RE = re.compile(r"[a-z]{2,3}(?:-[a-z0-9]+)?")


def validate_recognition_language_code(model: str, code: str) -> None:
    """Validate one advertised recognition-language code, Rust-parity.

    Accepts a plain lowercase ISO base code (`en`, `zh`, `yue`) OR a REGISTERED
    `-region` dialect code (`zh-sichuan`); rejects a malformed shape or an
    unregistered `-region` subtag so a typo (`zh-sichaun`) ships loudly rather
    than landing in a signed catalog.
    """
    if not isinstance(code, str) or RECOGNITION_LANGUAGE_CODE_RE.fullmatch(code) is None:
        raise KeyError(
            f"model '{model}' languages contains malformed recognition code {code!r} "
            "(expected a lowercase ISO base code with an optional -region subtag)"
        )
    if "-" in code and code not in REGISTERED_DIALECT_CODES:
        raise KeyError(
            f"model '{model}' languages dialect code {code!r} is not in the registered dialect-code set"
        )


def validate_recognition_languages(model: str, family: str, languages: list[str]) -> None:
    """Validate a resolved recognition `languages` list and enforce SELECTIVE
    dialect collapse: only a dialect-capable family may enumerate `-region`
    dialect codes; every other family must fold regional dialects into `zh`.
    """
    for code in languages:
        validate_recognition_language_code(model, code)
    dialects = sorted(code for code in languages if "-" in code)
    if dialects and family not in DIALECT_CAPABLE_FAMILIES:
        raise KeyError(
            f"model '{model}' family '{family}' advertises dialect code(s) "
            f"{', '.join(dialects)} but is not dialect-capable; regional dialects "
            "collapse into the base language unless the executor ships a code->prompt map"
        )


def language_labels_wire() -> dict:
    """The catalog's top-level `language_labels` map: code -> {en, zh-CN},
    sorted by code (BTreeMap order on the Rust side). Source of truth for the
    signed catalog; a Rust drift test pins it to `language_display_label`.
    """
    return {
        code: {"en": en, "zh-CN": zh_cn}
        for code, (en, zh_cn) in sorted(LANGUAGE_DISPLAY_LABELS.items())
    }


# The exact set of Chinese-dialect codes the `dolphin-cn-dialect-*` pack
# family can actually SELECT via its `<REGION>` prompt token (the domain of
# `dolphin_region_token_for_code` in crate::models::dolphin::language -- Rust
# mirror, kept in sync by hand). Distinct from (and a strict subset of) the
# model-agnostic REGISTERED_DIALECT_CODES above, which is now a cross-family
# typo-guard union: firered-aed and qwen also register dialect codes for
# their own benchmark-verified (not selectable) recognition coverage, so the
# global registry has grown codes Dolphin's own vocab has no region token
# for. LANG_BY_FAMILY["dolphin"] must be built from THIS list, never the
# global one, or a future cross-family dialect addition would silently make
# Dolphin's family default advertise a region it cannot honor.
DOLPHIN_CN_DIALECT_CODES = [
    "zh-anhui",
    "zh-guangdong",
    "zh-hebei",
    "zh-hubei",
    "zh-jiangsu",
    "zh-ningxia",
    "zh-shaanxi",
    "zh-shandong",
    "zh-shanghai",
    "zh-shanxi",
    "zh-sichuan",
    "zh-tianjin",
    "zh-tw",
]

# Per-family list of the natural languages a model officially supports, as
# ISO 639-1 two-letter codes (ISO 639-3 where no 639-1 code exists), sorted.
# RULE (SELECTIVE dialect collapse): by DEFAULT list LANGUAGES ONLY, NOT
# dialects/accents -- Chinese is a single language "zh" (Mandarin/Cantonese/Wu/
# Min and regional dialects fold into "zh"), English is "en" (US/UK not split),
# and a card advertising "30 languages and 22 Chinese dialects" yields the 30
# languages with the dialects folded into "zh". EXCEPTION: a dialect-capable
# family (DIALECT_CAPABLE_FAMILIES -- see its own doc comment for the two ways
# a family earns membership) MAY enumerate REGISTERED_DIALECT_CODES
# (`zh-sichuan`, ...) alongside the base `zh`. If a model supports N
# languages, list all N. See SKILL.md.
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
    # Dolphin cn-dialect small: Mandarin + 22 Chinese dialects (Sichuan, Wu,
    # Minnan, ...). Dolphin is dialect-capable (its executor ships a
    # code->region-prompt map), so it advertises the base `zh` plus every
    # dialect code ITS OWN region-prompt map supports as selectable source
    # languages -- DOLPHIN_CN_DIALECT_CODES, not the (now broader, cross-
    # family) REGISTERED_DIALECT_CODES.
    "dolphin": sorted(["zh", *DOLPHIN_CN_DIALECT_CODES]),
    "moonshine": ["en"],
    "parakeet": ["en"],
    # parakeet-tdt-0.6b-v3 card: 25 European languages (bg hr cs da nl en et fi
    # fr de el hu it lv lt mt pl pt ro sk sl es sv ru uk).
    "parakeet-tdt": [
        "bg", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr", "hr", "hu",
        "it", "lt", "lv", "mt", "nl", "pl", "pt", "ro", "ru", "sk", "sl", "sv",
        "uk",
    ],
    "wav2vec2": ["en"],
    # SenseVoiceSmall: zh, yue (Cantonese), en, ja, ko with model-side LID.
    "sensevoice": ["en", "ja", "ko", "yue", "zh"],
    # FireRedASR-AED-L: fixed bilingual Mandarin + English char/SPM vocab, no
    # language-selection prompt token (mirrors LanguageFamilyHint::
    # FixedMultilingual { languages: &["zh", "en"] } in
    # ggml_family_adapter.rs / arch/mod.rs).
    "firered-aed": ["en", "zh"],
    # FireRedASR2-LLM: fixed bilingual Mandarin + English Qwen2 BPE vocab, no
    # language-selection prompt token (FixedMultilingual { languages:
    # &["zh", "en"] } arch descriptor).
    "firered2-llm": ["en", "zh"],
    # MiMo-V2.5-ASR: fixed Mandarin + English + Cantonese Qwen2 BPE vocab, no
    # language-selection prompt token (FixedMultilingual { languages:
    # &["zh", "en", "yue"] } arch descriptor). "yue" is a base ISO 639-3 code
    # (Cantonese), not a dialect subtag.
    "mimo-asr": ["en", "yue", "zh"],
    # moss-transcribe-diarize is intentionally NOT listed here (its entry
    # carries an explicit `languages` override instead) so the user-facing "N
    # model families" doc counts (check_family_count_strings) do not move
    # before this family's public flip -- same staging convention used for
    # firered2-llm/mimo-asr in c847ae2.
}

# Per-family source-language parameter policy for the catalog's `language_mode`
# field, mirroring crates/openasr-core/src/models/ggml_family_adapter.rs's
# `LanguageFamilyHint` (and the `LanguageMode` it resolves to in
# crates/openasr-core/src/models/language.rs) 1:1. This is deliberately NOT an
# authored per-model TOML field: whether a family accepts/requires/rejects an
# explicit source-language selection is an architecture property owned by
# core, not a per-release editorial choice, so it is derived here from the
# same family the runtime dispatches on. "whisper" is intentionally absent --
# WhisperVocabGated resolves per-PACK from the pack's own vocab (multilingual
# vs English-only), so it is derived from the model's resolved `languages`
# list in `language_mode_for_model()` below instead of a fixed per-family value.
#
# Values are the exact LanguageMode wire tags core already serializes on
# `/v1/capabilities` (`LanguageCapability::mode` in
# crates/openasr-core/src/api/backend/mod.rs), reused verbatim so the catalog
# and the running-model capability surface never drift into two vocabularies
# for the same axis.
LANGUAGE_MODE_BY_FAMILY = {
    # Qwen3-ASR: SelfDetectsRejectsHint -- self-detects, explicit hint rejected.
    "qwen": "detect_implicit",
    # Cohere transcribe: SelectsViaPrompt -- always conditions on a language
    # token (its own default when the request omits one), never a true
    # decode-time auto-detect.
    "cohere": "specify_only",
    # X-ASR zh-en: FixedMultilingual -- built-in bilingual set, no per-request
    # language selection at all.
    "xasr-zipformer": "fixed_multilingual",
    # Dolphin: SelectsViaPrompt -- the dialect/region is chosen through prompt
    # tokens (<sos> <zh> <SICHUAN> <asr> <notimestamp>), never a decode-time
    # auto-detect, so it conditions on its default when the request omits one.
    "dolphin": "specify_only",
    # CTC / Moonshine: FixedMonolingual -- intrinsically a single language.
    "moonshine": "fixed_monolingual",
    "parakeet": "fixed_monolingual",
    # parakeet-tdt: FixedMultilingual -- built-in 25-language set, decodes
    # whatever it hears, no per-request language selection.
    "parakeet-tdt": "fixed_multilingual",
    "wav2vec2": "fixed_monolingual",
    # SenseVoice: DetectAndSelectsViaPrompt -- explicit zh/yue/en/ja/ko selection
    # via the 4-token prompt, decode-time LID (readable <|lang|> tag) when unset.
    "sensevoice": "detect_and_specify",
    # FireRedASR-AED: FixedMultilingual -- fixed zh+en char/SPM vocab, no
    # per-request language selection or decode-time LID token at all.
    "firered-aed": "fixed_multilingual",
    # FireRedASR2-LLM / MiMo-V2.5-ASR: FixedMultilingual -- the Qwen2 BPE decoder
    # has no language-selection prompt token and no decode-time LID token, so
    # neither exposes a per-request source-language axis (mirrors the
    # FixedMultilingual LanguageFamilyHint on both arch descriptors).
    "firered2-llm": "fixed_multilingual",
    "mimo-asr": "fixed_multilingual",
    # MOSS-Transcribe-Diarize: SelfDetectsRejectsHint -- the Qwen3-style
    # decoder auto-detects/produces the transcript language through free-text
    # instruction-following (no dedicated language token), same shape as qwen.
    "moss-transcribe-diarize": "detect_implicit",
}

# SpecifyOnly's conditioned default, mirroring the `default_language` literal
# on each family's `LanguageFamilyHint::SelectsViaPrompt` in
# ggml_family_adapter.rs. Not derivable from `languages` (English is not
# `languages[0]` alphabetically for cohere), so kept as an explicit
# same-source-of-truth constant instead of guessed.
LANGUAGE_MODE_DEFAULT_BY_FAMILY = {
    "cohere": "en",
    "dolphin": "zh",
}


def language_mode_for_model(entry: dict, languages: list[str]) -> dict:
    """Per-model `language_mode` (+ `language_default` where applicable) for
    the catalog, mirroring core's resolved `LanguageMode` for this model's
    family (see module docstring on `LANGUAGE_MODE_BY_FAMILY`).

    Returns {} for any kind other than asr-model: translation models (hymt2)
    and capability packs (wespeaker/pyannote-segmentation) are not
    GgmlFamilyAdapterDescriptor ASR families in core and have no per-request
    source-language axis, so the field is omitted rather than guessed.
    """
    if entry.get("kind", DEFAULT_CATALOG_MODEL_KIND) != "asr-model":
        return {}

    family = entry["family"]
    if family == "whisper":
        # WhisperVocabGated resolves per-pack from the pack's own vocab; the
        # catalog mirrors that via the model's resolved `languages` list
        # (English-only *.en checkpoints declare a single-element override).
        if len(languages) == 1:
            return {"language_mode": "fixed_monolingual", "language_default": languages[0]}
        return {"language_mode": "detect_and_specify"}

    mode = LANGUAGE_MODE_BY_FAMILY.get(family)
    if mode is None:
        known = ", ".join(sorted({*LANGUAGE_MODE_BY_FAMILY, "whisper"}))
        raise KeyError(
            f"model '{entry.get('id', '?')}' family '{family}' has no language_mode mapping. "
            f"Known families: {known}"
        )

    if mode == "fixed_monolingual":
        if len(languages) != 1:
            raise KeyError(
                f"model '{entry.get('id', '?')}' language_mode fixed_monolingual requires "
                f"exactly one language, got {languages!r}"
            )
        return {"language_mode": mode, "language_default": languages[0]}

    if mode == "specify_only":
        default_language = LANGUAGE_MODE_DEFAULT_BY_FAMILY.get(family)
        if default_language is None:
            raise KeyError(
                f"model '{entry.get('id', '?')}' family '{family}' has no "
                "LANGUAGE_MODE_DEFAULT_BY_FAMILY entry"
            )
        if default_language not in languages:
            raise KeyError(
                f"model '{entry.get('id', '?')}' language_mode specify_only "
                f"default_language {default_language!r} is not in languages {languages!r}"
            )
        return {"language_mode": mode, "language_default": default_language}

    # detect_implicit / fixed_multilingual: no default_language (core's
    # LanguageCapability leaves it unset for both -- there is either nothing to
    # default (self-detected, unexposed) or no per-request selection at all).
    return {"language_mode": mode}


# Per-family whether the model's transcripts include punctuation, mirroring
# whether the family's decoder/tokenizer can ever produce a punctuation token at
# all (an architecture/training-corpus property, not a per-release editorial
# choice -- like LANGUAGE_MODE_BY_FAMILY, derived here from the same family core
# dispatches on). `wav2vec2`/`parakeet` (character/BPE CTC, no catalog entries
# yet) are deliberately absent: whether a BYO-imported checkpoint's vocab
# includes punctuation depends on that specific checkpoint, not the family, so
# it cannot be stated as a fixed fact here.
#
# dolphin: DataoceanAI's cn-dialect-small training corpus is transcribed WITHOUT
# punctuation and the model has no punctuation-prediction head/token to enable,
# so its output is honestly unpunctuated -- product-decided (2026-07) to
# disclose this in the model card and market UI rather than leave it
# unexplained. Every other current asr-model family's training data/tokenizer
# supports punctuation and its transcripts include it.
#
# Conceptual single source of truth: the Rust engine's own declaration of this
# fact is `OpenAsrArchitectureDescriptor::emits_punctuation` in
# crates/openasr-core/src/arch/mod.rs (one field per builtin architecture,
# looked up via `emits_punctuation_for_model_architecture`). There is no
# Rust<->Python codegen bridge yet -- and this dict is keyed by the catalog's
# `family` string (e.g. "qwen", "cohere"), a different vocabulary from the
# engine's `model_family`/`model_architecture` constants (e.g. "qwen3-asr",
# "cohere-transcribe-conformer-transformer") -- so this table is hand-kept in
# lockstep with the Rust descriptor rather than generated from it. Rust's
# `registry/tests/catalog.rs::embedded_catalog_emits_punctuation_matches_family`
# cross-checks the shipped catalog's values against the descriptor for every
# family it can map, so a hand-edit here that drifts from the engine fact
# fails that test. Keep any change to a family's punctuation behavior
# synchronized on both sides.
PUNCTUATION_BY_FAMILY = {
    "qwen": True,
    # parakeet-tdt: trained on transcripts that preserve punctuation and
    # capitalization (verified on the imported pack: JFK decodes with full
    # punctuation).
    "parakeet-tdt": True,
    "cohere": True,
    "whisper": True,
    "xasr-zipformer": True,
    "dolphin": False,
    "moonshine": True,
    "sensevoice": True,
    # firered-aed: the reference tokenizer's dict.txt has no punctuation/
    # <space> entries (char + SPM vocab trained on unpunctuated Mandarin ASR
    # corpora), so the raw decode is honestly punctuation-free (verified on
    # the golden-diff fixture transcript).
    "firered-aed": False,
    # firered2-llm / mimo-asr: their Qwen2 ChatML decode is a plain
    # transcription completion with no characterized punctuation-suppression
    # behavior yet (arch descriptors declare emits_punctuation: None -- see
    # arch/mod.rs). None means "unclaimed": the catalog omits the field rather
    # than assert an unverified True/False (mirrors Option<bool>::None).
    "firered2-llm": None,
    "mimo-asr": None,
    # moss-transcribe-diarize: the fixed instruction prompts for full
    # punctuation-bearing prose, but this has not been verified against
    # enough real transcripts to assert as a capability yet (arch descriptor
    # declares emits_punctuation: None -- see arch/mod.rs) -- unclaimed.
    "moss-transcribe-diarize": None,
}


def punctuation_for_model(entry: dict) -> dict:
    """Per-model `emits_punctuation` for the catalog, mirroring whether this
    model's family ever predicts punctuation tokens (see module docstring on
    `PUNCTUATION_BY_FAMILY`).

    Returns {} for any kind other than asr-model: translation models (hymt2)
    and capability packs (wespeaker/pyannote-segmentation) don't produce an ASR
    transcript, so the field is omitted rather than guessed.
    """
    if entry.get("kind", DEFAULT_CATALOG_MODEL_KIND) != "asr-model":
        return {}

    family = entry["family"]
    if family not in PUNCTUATION_BY_FAMILY:
        known = ", ".join(sorted(PUNCTUATION_BY_FAMILY))
        raise KeyError(
            f"model '{entry.get('id', '?')}' family '{family}' has no emits_punctuation "
            f"mapping. Known families: {known}"
        )
    emits_punctuation = PUNCTUATION_BY_FAMILY[family]
    # A mapped None mirrors the arch descriptor's Option<bool>::None ("this
    # family's punctuation behavior is unclaimed"): omit the catalog field
    # entirely rather than assert an unverified value. Distinct from an ABSENT
    # family, which is a typo/onboarding gap and still fails loudly above.
    if emits_punctuation is None:
        return {}
    return {"emits_punctuation": emits_punctuation}


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
    validate_upstream_release_date(model, entry)
    validate_min_core_version(model, entry)

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


UPSTREAM_RELEASE_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


def validate_upstream_release_date(model: str, entry: dict) -> None:
    """`upstream_release_date` is the upstream model's original release date
    (ISO `yyyy-mm-dd`), an explicit author-set field distinct from our repack
    `generated_at`. Optional (nullable); when present it must be a real calendar
    date in `yyyy-mm-dd` form and not in the future.
    """
    value = entry.get("upstream_release_date")
    if value is None:
        return
    if not isinstance(value, str) or UPSTREAM_RELEASE_DATE_RE.fullmatch(value) is None:
        raise KeyError(
            f"model '{model}' upstream_release_date must be an ISO yyyy-mm-dd string, got {value!r}"
        )
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise KeyError(
            f"model '{model}' upstream_release_date is not a valid calendar date: {value!r}"
        ) from error
    if parsed > date.today():
        raise KeyError(
            f"model '{model}' upstream_release_date {value!r} is in the future"
        )


MIN_CORE_VERSION_RE = re.compile(r"\d+\.\d+\.\d+")


def validate_min_core_version(model: str, entry: dict) -> None:
    """`min_core_version` is the optional, author-set minimum core RUNTIME version
    a model needs (distinct from the publish-time `min_cli_version` floor). It
    lets a model be forward-published before older builds can execute it: those
    builds surface it as "update to use" and refuse the pull (see registry.rs
    CatalogModel::availability). Optional; when present it must be a plain
    `major.minor.patch` semver triplet. The value is NEVER derived from the
    current build -- it is set by hand per model.
    """
    value = entry.get("min_core_version")
    if value is None:
        return
    if not isinstance(value, str) or MIN_CORE_VERSION_RE.fullmatch(value) is None:
        raise KeyError(
            f"model '{model}' min_core_version must be a major.minor.patch semver string, "
            f"got {value!r}"
        )


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
        languages = sorted(set(override))
    else:
        languages = languages_for_family(entry["family"])
    # Validate the resolved codes (shape + registered-dialect membership) and
    # enforce selective dialect collapse, so a malformed/typo'd/unauthorized
    # dialect code fails loudly here rather than shipping in a signed catalog.
    validate_recognition_languages(
        entry.get("id", "?"), entry.get("family", "?"), languages
    )
    return languages


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

# A half-width ASCII punctuation mark sandwiched directly between two CJK
# (Han) characters is almost always a stray Western-keyboard artifact in
# otherwise full-width Chinese prose (e.g. "...E-Branchformer(CTC + 注意力),
# 覆盖..." should read "...（CTC + 注意力），覆盖..."). Requiring CJK on *both*
# sides keeps this from firing on legitimate ASCII usage: English clauses,
# code/backtick spans, markdown link syntax, and thousands separators like
# "400,000" all have a non-CJK neighbor on at least one side of the mark.
_CJK_CHAR = "一-鿿"
ZH_HALFWIDTH_PUNCT_BETWEEN_CJK_RE = re.compile(f"[{_CJK_CHAR}][,.!?;:][{_CJK_CHAR}]")


def _check_no_halfwidth_punct_between_cjk(model: str, locale: str, label: str, text: str) -> None:
    if not locale.lower().startswith("zh"):
        return
    match = ZH_HALFWIDTH_PUNCT_BETWEEN_CJK_RE.search(text)
    if match:
        raise KeyError(
            f"model '{model}' prose_locales.{locale} {label}: half-width punctuation "
            f"{match.group()[1]!r} directly between CJK characters ({match.group()!r}); "
            "use the full-width equivalent (， 。 ！ ？ ； ：) in Chinese prose"
        )


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
    _check_no_halfwidth_punct_between_cjk(model, locale, label, translated_text)
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
    elif cmd == "language-labels":
        print(json.dumps(language_labels_wire(), indent=2, ensure_ascii=False))
    elif cmd == "write-language-labels":
        from _file_loaders import atomic_write_json

        path = Path(argv[1])
        data = json.loads(path.read_text(encoding="utf-8"))
        # Refresh (or add) the top-level map in place, preserving key order so a
        # per-model regenerate that only touches models[] stays a minimal diff.
        data["language_labels"] = language_labels_wire()
        atomic_write_json(path, data)
        print(f"wrote language_labels ({len(data['language_labels'])} codes) to {path}")
    else:
        sys.exit(f"unknown command '{cmd}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
