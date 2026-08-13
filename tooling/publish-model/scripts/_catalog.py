#!/usr/bin/env python3
"""Catalog reader for the OpenASR publishing harness.

Catalog authoring source = tooling/publish-model/models-core.toml plus
tooling/publish-model/models-publish.toml. Runtime family capabilities are
read from the generated tooling/model-family-inventory.v1.json; this module
strictly validates that inventory before exposing any capability projection.
Bash scripts shell out to this for field lookups and quant-token mapping so the
catalog is parsed in one place (Python 3.11+ stdlib tomllib) rather than
re-implemented in fragile shell.

Usage:
  _catalog.py field   <model> <key>     # print one catalog value (lists -> space-joined)
  _catalog.py field-lines <model> <key> # print a list-valued key one item per line (empty if absent)
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
MODEL_FAMILY_INVENTORY = REPO_ROOT / "tooling" / "model-family-inventory.v1.json"
MODEL_FAMILY_INVENTORY_SCHEMA = "openasr.model-family-inventory.v1"
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
MODULE_SLUG_RE = re.compile(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*")
TRANSLATION_REQUIRED_LICENSE_FILES = {"LICENSE.txt", "NOTICE.openasr.txt"}


_INVENTORY_TOP_LEVEL_KEYS = {"schema", "families"}
_INVENTORY_FAMILY_KEYS = {
    "catalog_family_id",
    "model_family",
    "model_architecture",
    "runtime_architecture_aliases",
    "adapter_id",
    "module_slug",
    "language",
    "pack",
    "execution",
    "topology",
    "optimization",
    "quantization",
    "conformance",
}
_INVENTORY_LANGUAGE_KEYS = {
    "policy",
    "default_language",
    "reject_reason",
    "languages",
    "dialect_mode",
    "selectable_dialect_codes",
}
_INVENTORY_PACK_KEYS = {
    "audio_frontend_id",
    "decode_policy_id",
    "runtime_tensor_contract_id",
    "tokenizer_id",
    "hparam_schema",
    "importer",
}
_INVENTORY_IMPORTER_KEYS = {"kind", "symbol", "relative_path"}
_INVENTORY_EXECUTION_KEYS = {
    "executor_component_id",
    "executor",
    "execution_capabilities",
    "streaming_partial_granularity",
    "speaker_segmentation",
    "emits_punctuation",
    "supports_phrase_bias",
    "phrase_bias_strategy",
    "phrase_bias_required_tensor",
    "supports_translation_task",
    "supports_source_language_hint",
    "adapter_binding",
    "prepared_runtime",
    "word_timestamp_strategy",
    "invocation_span",
}
_INVENTORY_EXECUTION_CAPABILITIES_KEYS = {"cpu", "providers"}
_INVENTORY_PROVIDER_KEYS = {"provider", "full_device", "hybrid"}
_INVENTORY_INVOCATION_KEYS = {"policy", "max_seconds"}
_INVENTORY_TOPOLOGY_KEYS = {
    "decode_driver",
    "decode_driver_reason",
    "block_stack",
    "block_stack_reason",
    "decoder_state",
}
_INVENTORY_OPTIMIZATION_KEYS = {
    "prefer_cpu_decoder_for_multichunk_metal",
    "auto_gpu_policy",
    "encoder_attention_span",
    "encoder_attention_max_safe_chunk_seconds",
}
_INVENTORY_QUANTIZATION_KEYS = {"tensor_classification", "quantized_axis"}
_INVENTORY_TENSOR_CLASSIFICATIONS = {
    "semantic-roles-v1",
    "entire-acoustic-pack",
    "not-applicable",
}
_INVENTORY_QUANTIZED_AXES = {"first", "last"}
_INVENTORY_ADAPTER_BINDINGS = {
    "unsupported",
    "qwen3-asr-lora-v1",
    "moonshine-lora-v1",
}
_INVENTORY_CONFORMANCE_KEYS = {"profile_id", "reference_dumper_source"}
_INVENTORY_PHRASE_BIAS_STRATEGIES = {"unsupported", "always", "requires-tensor"}
_INVENTORY_WORD_TIMESTAMP_STRATEGIES = {"decode-invariant", "decode-sensitive"}
_INVENTORY_PREPARED_RUNTIME_SHARED_RE = re.compile(r"shared-[a-z0-9-]+-v[1-9][0-9]*")
_INVENTORY_LANGUAGE_POLICIES = {
    "selects-via-prompt",
    "fixed-monolingual",
    "fixed-multilingual",
    "self-detects-rejects-hint",
    "detect-and-selects-via-prompt",
    "whisper-vocab-gated",
}
_INVENTORY_DIALECT_MODES = {
    "not-advertised",
    "recognizes-catalog-declared",
    "selects-via-prompt",
}
_LANGUAGE_POLICY_TO_WIRE_MODE = {
    "selects-via-prompt": "specify_only",
    "fixed-monolingual": "fixed_monolingual",
    "fixed-multilingual": "fixed_multilingual",
    "self-detects-rejects-hint": "detect_implicit",
    "detect-and-selects-via-prompt": "detect_and_specify",
}


def _inventory_error(path: str, message: str) -> ValueError:
    return ValueError(f"model family inventory {path}: {message}")


def _inventory_object(value: object, path: str) -> dict:
    if not isinstance(value, dict):
        raise _inventory_error(path, f"must be an object, got {type(value).__name__}")
    return value


def _inventory_exact_keys(value: dict, expected: set[str], path: str) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append(f"missing {', '.join(missing)}")
        if unknown:
            details.append(f"unknown {', '.join(unknown)}")
        raise _inventory_error(path, "; ".join(details))


def _inventory_string(value: object, path: str, *, allow_empty: bool = False) -> None:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise _inventory_error(path, "must be a non-empty string")


def _inventory_nullable_string(value: object, path: str) -> None:
    if value is not None:
        _inventory_string(value, path)


def _inventory_string_list(value: object, path: str, *, allow_empty: bool = False) -> None:
    if not isinstance(value, list) or (not allow_empty and not value):
        expected = "a string list" if allow_empty else "a non-empty string list"
        raise _inventory_error(path, f"must be {expected}")
    for index, item in enumerate(value):
        _inventory_string(item, f"{path}[{index}]")


def _inventory_base_language_list(value: object, path: str) -> None:
    """Validate the canonical Rust-projected ISO 639 base-language list."""
    _inventory_string_list(value, path)
    assert isinstance(value, list)
    if value != sorted(set(value)):
        raise _inventory_error(path, "must be sorted and unique")
    for index, code in enumerate(value):
        if re.fullmatch(r"[a-z]{2,3}", code) is None:
            raise _inventory_error(
                f"{path}[{index}]",
                "must be a lowercase ISO 639 base code (2-3 letters)",
            )


def _inventory_nullable_number(value: object, path: str, *, positive: bool = False) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _inventory_error(path, "must be a number or null")
    if positive and value <= 0:
        raise _inventory_error(path, "must be greater than zero")


def _validate_model_family_inventory(payload: object) -> None:
    root = _inventory_object(payload, "root")
    _inventory_exact_keys(root, _INVENTORY_TOP_LEVEL_KEYS, "root")
    if root["schema"] != MODEL_FAMILY_INVENTORY_SCHEMA:
        raise _inventory_error(
            "schema",
            f"must be {MODEL_FAMILY_INVENTORY_SCHEMA!r}, got {root['schema']!r}",
        )
    families = root["families"]
    if not isinstance(families, list) or not families:
        raise _inventory_error("families", "must be a non-empty list")

    seen_catalog_family_ids: set[str] = set()
    seen_module_slugs: set[str] = set()
    for index, raw_family in enumerate(families):
        path = f"families[{index}]"
        family = _inventory_object(raw_family, path)
        _inventory_exact_keys(family, _INVENTORY_FAMILY_KEYS, path)
        for field in (
            "catalog_family_id",
            "model_family",
            "model_architecture",
            "adapter_id",
            "module_slug",
        ):
            _inventory_string(family[field], f"{path}.{field}")
        catalog_family_id = family["catalog_family_id"]
        if catalog_family_id in seen_catalog_family_ids:
            raise _inventory_error(
                f"{path}.catalog_family_id",
                f"duplicate catalog_family_id {catalog_family_id!r}",
            )
        seen_catalog_family_ids.add(catalog_family_id)
        module_slug = family["module_slug"]
        if MODULE_SLUG_RE.fullmatch(module_slug) is None:
            raise _inventory_error(
                f"{path}.module_slug",
                "must be a non-empty snake_case identifier",
            )
        if module_slug in seen_module_slugs:
            raise _inventory_error(
                f"{path}.module_slug",
                f"duplicate module_slug {module_slug!r}",
            )
        seen_module_slugs.add(module_slug)
        _inventory_string_list(
            family["runtime_architecture_aliases"],
            f"{path}.runtime_architecture_aliases",
        )

        language = _inventory_object(family["language"], f"{path}.language")
        _inventory_exact_keys(language, _INVENTORY_LANGUAGE_KEYS, f"{path}.language")
        policy = language["policy"]
        if policy not in _INVENTORY_LANGUAGE_POLICIES:
            raise _inventory_error(f"{path}.language.policy", f"unsupported policy {policy!r}")
        _inventory_nullable_string(language["default_language"], f"{path}.language.default_language")
        _inventory_nullable_string(language["reject_reason"], f"{path}.language.reject_reason")
        _inventory_base_language_list(
            family["language"]["languages"], f"{path}.language.languages"
        )
        dialect_mode = language["dialect_mode"]
        if dialect_mode not in _INVENTORY_DIALECT_MODES:
            raise _inventory_error(
                f"{path}.language.dialect_mode",
                f"unsupported dialect mode {dialect_mode!r}",
            )
        selectable_dialect_codes = language["selectable_dialect_codes"]
        _inventory_string_list(
            selectable_dialect_codes,
            f"{path}.language.selectable_dialect_codes",
            allow_empty=True,
        )
        if selectable_dialect_codes != sorted(set(selectable_dialect_codes)):
            raise _inventory_error(
                f"{path}.language.selectable_dialect_codes",
                "must be sorted and unique",
            )
        if dialect_mode == "selects-via-prompt" and not selectable_dialect_codes:
            raise _inventory_error(
                f"{path}.language.selectable_dialect_codes",
                "must be non-empty for selects-via-prompt",
            )
        if dialect_mode != "selects-via-prompt" and selectable_dialect_codes:
            raise _inventory_error(
                f"{path}.language.selectable_dialect_codes",
                f"must be empty for {dialect_mode}",
            )
        default_language = language["default_language"]
        reject_reason = language["reject_reason"]
        languages = language["languages"]
        if policy in {"selects-via-prompt", "fixed-monolingual"} and default_language is None:
            raise _inventory_error(f"{path}.language.default_language", f"required for {policy}")
        if policy in {"fixed-multilingual", "self-detects-rejects-hint", "detect-and-selects-via-prompt", "whisper-vocab-gated"} and default_language is not None:
            raise _inventory_error(f"{path}.language.default_language", f"must be null for {policy}")
        if policy == "self-detects-rejects-hint" and not reject_reason:
            raise _inventory_error(f"{path}.language.reject_reason", "required for self-detects-rejects-hint")
        if policy != "self-detects-rejects-hint" and reject_reason is not None:
            raise _inventory_error(f"{path}.language.reject_reason", f"must be null for {policy}")
        if policy == "fixed-monolingual" and len(languages) != 1:
            raise _inventory_error(f"{path}.language.languages", "fixed-monolingual requires one language")
        if policy == "fixed-multilingual" and not languages:
            raise _inventory_error(f"{path}.language.languages", "fixed-multilingual requires languages")
        if default_language is not None and default_language not in languages:
            raise _inventory_error(
                f"{path}.language.default_language",
                f"must be present in languages for {policy}",
            )

        pack = _inventory_object(family["pack"], f"{path}.pack")
        _inventory_exact_keys(pack, _INVENTORY_PACK_KEYS, f"{path}.pack")
        for field in ("audio_frontend_id", "decode_policy_id", "runtime_tensor_contract_id", "tokenizer_id"):
            _inventory_string(pack[field], f"{path}.pack.{field}")
        _inventory_string_list(pack["hparam_schema"], f"{path}.pack.hparam_schema", allow_empty=True)
        importer = _inventory_object(pack["importer"], f"{path}.pack.importer")
        _inventory_exact_keys(importer, _INVENTORY_IMPORTER_KEYS, f"{path}.pack.importer")
        importer_kind = importer["kind"]
        if importer_kind == "core-convert":
            _inventory_string(importer["symbol"], f"{path}.pack.importer.symbol")
            if importer["relative_path"] is not None:
                raise _inventory_error(f"{path}.pack.importer.relative_path", "must be null for core-convert")
        elif importer_kind == "external-tooling":
            if importer["symbol"] is not None:
                raise _inventory_error(f"{path}.pack.importer.symbol", "must be null for external-tooling")
            _inventory_string(importer["relative_path"], f"{path}.pack.importer.relative_path")
        else:
            raise _inventory_error(f"{path}.pack.importer.kind", f"unsupported kind {importer_kind!r}")

        execution = _inventory_object(family["execution"], f"{path}.execution")
        _inventory_exact_keys(execution, _INVENTORY_EXECUTION_KEYS, f"{path}.execution")
        for field in ("executor_component_id", "executor", "streaming_partial_granularity"):
            _inventory_string(execution[field], f"{path}.execution.{field}")
        execution_capabilities = _inventory_object(
            execution["execution_capabilities"], f"{path}.execution.execution_capabilities"
        )
        _inventory_exact_keys(
            execution_capabilities,
            _INVENTORY_EXECUTION_CAPABILITIES_KEYS,
            f"{path}.execution.execution_capabilities",
        )
        if not isinstance(execution_capabilities["cpu"], bool):
            raise _inventory_error(f"{path}.execution.execution_capabilities.cpu", "must be boolean")
        providers = execution_capabilities["providers"]
        if not isinstance(providers, list) or not providers:
            raise _inventory_error(
                f"{path}.execution.execution_capabilities.providers",
                "must be a non-empty list",
            )
        seen_providers: set[str] = set()
        for provider_index, raw_provider in enumerate(providers):
            provider_path = f"{path}.execution.execution_capabilities.providers[{provider_index}]"
            provider = _inventory_object(raw_provider, provider_path)
            _inventory_exact_keys(provider, _INVENTORY_PROVIDER_KEYS, provider_path)
            _inventory_string(provider["provider"], f"{provider_path}.provider")
            if provider["provider"] in seen_providers:
                raise _inventory_error(f"{provider_path}.provider", "duplicate provider")
            seen_providers.add(provider["provider"])
            for field in ("full_device", "hybrid"):
                if not isinstance(provider[field], bool):
                    raise _inventory_error(f"{provider_path}.{field}", "must be boolean")
        if execution["speaker_segmentation"] not in {"external", "in-decoder"}:
            raise _inventory_error(f"{path}.execution.speaker_segmentation", "must be external or in-decoder")
        if execution["emits_punctuation"] is not None and not isinstance(execution["emits_punctuation"], bool):
            raise _inventory_error(f"{path}.execution.emits_punctuation", "must be boolean or null")
        for field in (
            "supports_phrase_bias",
            "supports_translation_task",
            "supports_source_language_hint",
        ):
            if not isinstance(execution[field], bool):
                raise _inventory_error(f"{path}.execution.{field}", "must be boolean")
        adapter_binding = execution["adapter_binding"]
        _inventory_string(adapter_binding, f"{path}.execution.adapter_binding")
        if adapter_binding not in _INVENTORY_ADAPTER_BINDINGS:
            raise _inventory_error(
                f"{path}.execution.adapter_binding",
                f"unsupported binding {adapter_binding!r}",
            )
        phrase_bias_strategy = execution["phrase_bias_strategy"]
        _inventory_string(phrase_bias_strategy, f"{path}.execution.phrase_bias_strategy")
        if phrase_bias_strategy not in _INVENTORY_PHRASE_BIAS_STRATEGIES:
            raise _inventory_error(
                f"{path}.execution.phrase_bias_strategy",
                f"unsupported strategy {phrase_bias_strategy!r}; expected unsupported, always, or requires-tensor",
            )
        if execution["supports_phrase_bias"] != (phrase_bias_strategy != "unsupported"):
            raise _inventory_error(
                f"{path}.execution.supports_phrase_bias",
                "must be equivalent to phrase_bias_strategy != 'unsupported'",
            )
        phrase_bias_required_tensor = execution["phrase_bias_required_tensor"]
        if phrase_bias_strategy == "requires-tensor":
            _inventory_string(
                phrase_bias_required_tensor,
                f"{path}.execution.phrase_bias_required_tensor",
            )
        elif phrase_bias_required_tensor is not None:
            raise _inventory_error(
                f"{path}.execution.phrase_bias_required_tensor",
                "must be null unless phrase_bias_strategy is 'requires-tensor'",
            )
        word_timestamp_strategy = execution["word_timestamp_strategy"]
        _inventory_string(word_timestamp_strategy, f"{path}.execution.word_timestamp_strategy")
        if word_timestamp_strategy not in _INVENTORY_WORD_TIMESTAMP_STRATEGIES:
            raise _inventory_error(
                f"{path}.execution.word_timestamp_strategy",
                f"unsupported strategy {word_timestamp_strategy!r}; expected decode-invariant or decode-sensitive",
            )
        prepared_runtime = execution["prepared_runtime"]
        _inventory_string(prepared_runtime, f"{path}.execution.prepared_runtime")
        if prepared_runtime != "family-owned" and _INVENTORY_PREPARED_RUNTIME_SHARED_RE.fullmatch(prepared_runtime) is None:
            raise _inventory_error(
                f"{path}.execution.prepared_runtime",
                "must be family-owned or match shared-[a-z0-9-]+-v[1-9][0-9]*",
            )
        invocation_span = _inventory_object(execution["invocation_span"], f"{path}.execution.invocation_span")
        _inventory_exact_keys(invocation_span, _INVENTORY_INVOCATION_KEYS, f"{path}.execution.invocation_span")
        if invocation_span["policy"] not in {"elastic", "bounded"}:
            raise _inventory_error(f"{path}.execution.invocation_span.policy", "must be elastic or bounded")
        _inventory_nullable_number(invocation_span["max_seconds"], f"{path}.execution.invocation_span.max_seconds", positive=True)
        if invocation_span["policy"] == "elastic" and invocation_span["max_seconds"] is not None:
            raise _inventory_error(f"{path}.execution.invocation_span.max_seconds", "must be null for elastic")
        if invocation_span["policy"] == "bounded" and invocation_span["max_seconds"] is None:
            raise _inventory_error(f"{path}.execution.invocation_span.max_seconds", "required for bounded")

        topology = _inventory_object(family["topology"], f"{path}.topology")
        _inventory_exact_keys(topology, _INVENTORY_TOPOLOGY_KEYS, f"{path}.topology")
        for field in ("decode_driver", "block_stack", "decoder_state"):
            _inventory_string(topology[field], f"{path}.topology.{field}")
        for field in ("decode_driver_reason", "block_stack_reason"):
            _inventory_nullable_string(topology[field], f"{path}.topology.{field}")
        if topology["decode_driver"] == "dedicated" and not topology["decode_driver_reason"]:
            raise _inventory_error(f"{path}.topology.decode_driver_reason", "required for dedicated")
        if topology["decode_driver"] != "dedicated" and topology["decode_driver_reason"] is not None:
            raise _inventory_error(f"{path}.topology.decode_driver_reason", "must be null for shared driver")
        if topology["block_stack"] == "architecture-graph" and not topology["block_stack_reason"]:
            raise _inventory_error(f"{path}.topology.block_stack_reason", "required for architecture-graph")
        if topology["block_stack"] != "architecture-graph" and topology["block_stack_reason"] is not None:
            raise _inventory_error(f"{path}.topology.block_stack_reason", "must be null for shared block stack")

        optimization = _inventory_object(family["optimization"], f"{path}.optimization")
        _inventory_exact_keys(optimization, _INVENTORY_OPTIMIZATION_KEYS, f"{path}.optimization")
        if not isinstance(optimization["prefer_cpu_decoder_for_multichunk_metal"], bool):
            raise _inventory_error(f"{path}.optimization.prefer_cpu_decoder_for_multichunk_metal", "must be boolean")
        for field in ("auto_gpu_policy", "encoder_attention_span"):
            _inventory_string(optimization[field], f"{path}.optimization.{field}")
        _inventory_nullable_number(
            optimization["encoder_attention_max_safe_chunk_seconds"],
            f"{path}.optimization.encoder_attention_max_safe_chunk_seconds",
            positive=True,
        )

        quantization = _inventory_object(family["quantization"], f"{path}.quantization")
        _inventory_exact_keys(quantization, _INVENTORY_QUANTIZATION_KEYS, f"{path}.quantization")
        tensor_classification = quantization["tensor_classification"]
        _inventory_string(tensor_classification, f"{path}.quantization.tensor_classification")
        if tensor_classification not in _INVENTORY_TENSOR_CLASSIFICATIONS:
            raise _inventory_error(
                f"{path}.quantization.tensor_classification",
                f"unsupported classification {tensor_classification!r}",
            )
        quantized_axis = quantization["quantized_axis"]
        if quantized_axis is not None:
            _inventory_string(quantized_axis, f"{path}.quantization.quantized_axis")
            if quantized_axis not in _INVENTORY_QUANTIZED_AXES:
                raise _inventory_error(
                    f"{path}.quantization.quantized_axis",
                    f"unsupported axis {quantized_axis!r}; expected null, 'first', or 'last'",
                )
        if tensor_classification == "semantic-roles-v1" and quantized_axis not in _INVENTORY_QUANTIZED_AXES:
            raise _inventory_error(
                f"{path}.quantization.quantized_axis",
                "semantic-roles-v1 requires 'first' or 'last'",
            )
        if tensor_classification != "semantic-roles-v1" and quantized_axis is not None:
            raise _inventory_error(
                f"{path}.quantization.quantized_axis",
                f"{tensor_classification} requires null",
            )

        conformance = _inventory_object(family["conformance"], f"{path}.conformance")
        _inventory_exact_keys(conformance, _INVENTORY_CONFORMANCE_KEYS, f"{path}.conformance")
        _inventory_string(conformance["profile_id"], f"{path}.conformance.profile_id")
        _inventory_nullable_string(conformance["reference_dumper_source"], f"{path}.conformance.reference_dumper_source")


def load_model_family_inventory(path: Path | None = None) -> dict[str, dict]:
    """Load and strictly validate the generated Rust family inventory."""
    inventory_path = MODEL_FAMILY_INVENTORY if path is None else Path(path)
    try:
        payload = json.loads(inventory_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"model family inventory cannot be read: {inventory_path}: {error}") from error
    _validate_model_family_inventory(payload)
    return {family["catalog_family_id"]: family for family in payload["families"]}


MODEL_FAMILY_CAPABILITIES = load_model_family_inventory()


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
    # Product quant name for mixed-tensor Q4_K_M GGUF files.
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
    dialect collapse from the generated family inventory: only a family whose
    dialect mode advertises recognition may enumerate `-region` dialect codes;
    every other family must fold regional dialects into `zh`.
    """
    for code in languages:
        validate_recognition_language_code(model, code)
    dialects = sorted(code for code in languages if "-" in code)
    family_inventory = MODEL_FAMILY_CAPABILITIES.get(family)
    dialect_mode = (
        family_inventory["language"]["dialect_mode"]
        if family_inventory is not None
        else "not-advertised"
    )
    if dialects and dialect_mode == "not-advertised":
        raise KeyError(
            f"model '{model}' family '{family}' advertises dialect code(s) "
            f"{', '.join(dialects)} but its inventory dialect_mode is "
            "'not-advertised'; regional dialects collapse into the base language"
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


# Convert the inventory's language-policy names to the existing catalog wire tags.
# Whisper remains pack-dependent: its vocabulary gate is resolved from the
# model's effective language list below rather than from a fixed inventory row.


def _inventory_family_for_capability(family: str, capability: str) -> dict:
    try:
        return MODEL_FAMILY_CAPABILITIES[family]
    except KeyError as error:
        known = ", ".join(sorted(MODEL_FAMILY_CAPABILITIES))
        raise KeyError(
            f"model family '{family}' has no {capability} mapping in the model family inventory. "
            f"Known families: {known}"
        ) from error


def language_mode_for_model(entry: dict, languages: list[str]) -> dict:
    """Resolve the catalog language mode from the generated family inventory."""
    if entry.get("kind", DEFAULT_CATALOG_MODEL_KIND) != "asr-model":
        return {}

    family = entry["family"]
    language = _inventory_family_for_capability(family, "language_mode")["language"]
    policy = language["policy"]
    if policy == "whisper-vocab-gated":
        # WhisperVocabGated resolves per-pack from the pack's own vocab; the
        # catalog mirrors that via the model's effective language list.
        if len(languages) == 1:
            return {"language_mode": "fixed_monolingual", "language_default": languages[0]}
        return {"language_mode": "detect_and_specify"}

    mode = _LANGUAGE_POLICY_TO_WIRE_MODE[policy]
    if mode == "fixed_monolingual":
        if len(languages) != 1:
            raise KeyError(
                f"model '{entry.get('id', '?')}' language_mode fixed_monolingual requires "
                f"exactly one language, got {languages!r}"
            )
        default_language = language["default_language"]
        if default_language not in languages:
            raise KeyError(
                f"model '{entry.get('id', '?')}' inventory default_language "
                f"{default_language!r} is not in languages {languages!r}"
            )
        return {"language_mode": mode, "language_default": default_language}

    if mode == "specify_only":
        default_language = language["default_language"]
        if default_language not in languages:
            raise KeyError(
                f"model '{entry.get('id', '?')}' language_mode specify_only "
                f"default_language {default_language!r} is not in languages {languages!r}"
            )
        return {"language_mode": mode, "language_default": default_language}

    # detect_implicit / fixed_multilingual / detect_and_specify: no default.
    return {"language_mode": mode}


def punctuation_for_model(entry: dict) -> dict:
    """Resolve ``emits_punctuation`` from the generated family inventory."""
    if entry.get("kind", DEFAULT_CATALOG_MODEL_KIND) != "asr-model":
        return {}

    family = entry["family"]
    emits_punctuation = _inventory_family_for_capability(family, "emits_punctuation")[
        "execution"
    ]["emits_punctuation"]
    # Null is an explicit unclaimed capability: omit the catalog field.
    if emits_punctuation is None:
        return {}
    return {"emits_punctuation": emits_punctuation}


def speaker_source_for_model(entry: dict) -> dict:
    """Resolve the catalog Voice-ID source from the generated family inventory."""
    if entry.get("kind", DEFAULT_CATALOG_MODEL_KIND) != "asr-model":
        return {}

    family = entry["family"]
    source = _inventory_family_for_capability(family, "speaker_source")["execution"][
        "speaker_segmentation"
    ]
    # Inventory uses architecture terminology; catalog uses the public wire tag.
    return {"speaker_source": "native" if source == "in-decoder" else "external"}


# Source of the word anchors needed to project transcript text onto an
# external speaker timeline. This mirrors
# `OpenAsrArchitectureDescriptor::word_timestamp_source`; keeping it in the
# signed catalog lets clients pre-install the forced aligner without a
# model-id allowlist or a late transcription failure.
WORD_TIMESTAMP_SOURCE_BY_FAMILY = {
    "qwen": "native",
    "parakeet-tdt": "native",
    "cohere": "native",
    "whisper": "native",
    "xasr-zipformer": "native",
    "moonshine": "native",
    "dolphin": "forced_aligner",
    "sensevoice": "forced_aligner",
    "firered-aed": "forced_aligner",
    "firered2-llm": "forced_aligner",
    "mimo-asr": "forced_aligner",
    "moss-transcribe-diarize": "forced_aligner",
    "funasr-nano": "forced_aligner",
    "granite-speech": "forced_aligner",
}


def word_timestamp_source_for_model(entry: dict) -> dict:
    """Return the architecture's usable word-anchor source for ASR models."""
    if entry.get("kind", DEFAULT_CATALOG_MODEL_KIND) != "asr-model":
        return {}
    family = entry["family"]
    source = WORD_TIMESTAMP_SOURCE_BY_FAMILY.get(family)
    if source is None:
        known = ", ".join(sorted(WORD_TIMESTAMP_SOURCE_BY_FAMILY))
        raise KeyError(
            f"model '{entry.get('id', '?')}' family '{family}' has no "
            f"word_timestamp_source mapping. Known families: {known}"
        )
    return {"word_timestamp_source": source}


def apply_word_timestamp_sources_to_catalog(
    catalog: dict, catalog_entries: dict | None = None
) -> int:
    """Refresh architecture-derived word timestamp sources in-place."""
    entries = catalog_entries if catalog_entries is not None else load()
    by_registry_id = {entry["registry_id"]: entry for entry in entries.values()}
    models = catalog.get("models")
    if not isinstance(models, list):
        raise KeyError("catalog models must be a list")

    updated = 0
    for model in models:
        if not isinstance(model, dict) or not isinstance(model.get("id"), str):
            raise KeyError("catalog models must contain object entries with string ids")
        source = by_registry_id.get(model["id"])
        if source is None:
            raise KeyError(
                f"catalog model '{model['id']}' has no models-core.toml source entry"
            )
        expected = word_timestamp_source_for_model(source).get("word_timestamp_source")
        previous = model.get("word_timestamp_source")
        if expected is None:
            model.pop("word_timestamp_source", None)
        else:
            model["word_timestamp_source"] = expected
        if previous != expected:
            updated += 1
    return updated


def apply_speaker_sources_to_catalog(
    catalog: dict, catalog_entries: dict | None = None
) -> int:
    """Refresh generated ``speaker_source`` fields without rebuilding packs.

    A catalog-wide capability migration must not require old publish evidence
    (metrics and result sidecars) merely to add one architecture-derived
    scalar. Every catalog model still has to resolve to the authoring TOML, so
    an unknown id fails closed instead of being guessed.
    """
    entries = catalog_entries if catalog_entries is not None else load()
    by_registry_id = {entry["registry_id"]: entry for entry in entries.values()}
    models = catalog.get("models")
    if not isinstance(models, list):
        raise KeyError("catalog models must be a list")

    updated = 0
    for model in models:
        if not isinstance(model, dict) or not isinstance(model.get("id"), str):
            raise KeyError("catalog models must contain object entries with string ids")
        source = by_registry_id.get(model["id"])
        if source is None:
            raise KeyError(
                f"catalog model '{model['id']}' has no models-core.toml source entry"
            )
        expected = speaker_source_for_model(source).get("speaker_source")
        previous = model.get("speaker_source")
        if expected is None:
            model.pop("speaker_source", None)
        else:
            model["speaker_source"] = expected
        if previous != expected:
            updated += 1
    return updated


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
        language = MODEL_FAMILY_CAPABILITIES[family]["language"]
    except KeyError as error:
        known = ", ".join(sorted(MODEL_FAMILY_CAPABILITIES))
        raise KeyError(f"unknown model family '{family}'. Known language mappings: {known}") from error
    languages = list(language["languages"])
    if language["dialect_mode"] == "selects-via-prompt":
        languages = sorted(set(languages + language["selectable_dialect_codes"]))
    return languages


def languages_for_model(entry: dict) -> list[str]:
    """Resolve the languages a specific model supports.

    Language support is a per-MODEL property: a model may support fewer languages
    than its family default (e.g. Whisper's English-only `*.en` checkpoints support
    only `en` even though the multilingual Whisper family supports ~98). A model
    that needs to differ from the family default declares an explicit `languages`
    list in models-core.toml; otherwise it inherits the generated Rust family
    inventory (including prompt-selectable dialect codes where declared).
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
    elif cmd == "field-lines":
        # One list item per line (empty output when the key is absent): the
        # shell uses mapfile over lists whose items carry spaces (prep
        # scripts, import command templates), where `field`'s space-join
        # would corrupt the entries.
        val = entry(argv[1]).get(argv[2])
        if val is None:
            return 0
        if not isinstance(val, list) or not all(isinstance(item, str) for item in val):
            sys.exit(f"key '{argv[2]}' for model '{argv[1]}' is not a string list")
        print("\n".join(val))
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
    elif cmd == "write-speaker-sources":
        from _file_loaders import atomic_write_json

        path = Path(argv[1])
        data = json.loads(path.read_text(encoding="utf-8"))
        updated = apply_speaker_sources_to_catalog(data)
        atomic_write_json(path, data)
        print(f"wrote speaker_source for {updated} changed catalog model(s) to {path}")
    elif cmd == "write-word-timestamp-sources":
        from _file_loaders import atomic_write_json

        path = Path(argv[1])
        data = json.loads(path.read_text(encoding="utf-8"))
        updated = apply_word_timestamp_sources_to_catalog(data)
        atomic_write_json(path, data)
        print(
            f"wrote word_timestamp_source for {updated} changed catalog model(s) to {path}"
        )
    elif cmd == "prune-catalog-models":
        from _file_loaders import atomic_write_json

        path = Path(argv[1])
        data = json.loads(path.read_text(encoding="utf-8"))
        allowed_registry_ids = {
            model_entry["registry_id"] for model_entry in load().values()
        }
        before = len(data.get("models", []))
        data["models"] = [
            model
            for model in data.get("models", [])
            if model.get("id") in allowed_registry_ids
        ]
        removed = before - len(data["models"])
        atomic_write_json(path, data)
        print(f"pruned {removed} stale catalog model(s) from {path}")
    elif cmd == "write-public-projection":
        from _file_loaders import atomic_write_json

        source = Path(argv[1])
        target = Path(argv[2])
        data = json.loads(source.read_text(encoding="utf-8"))
        projection = {
            "schema_version": data["schema_version"],
            "generated_at": data["generated_at"],
            "catalog_url": data["catalog_url"],
            "models": [model for model in data.get("models", []) if model.get("public") is True],
        }
        if data.get("language_labels"):
            projection["language_labels"] = data["language_labels"]
        atomic_write_json(target, projection)
        print(f"wrote {len(projection['models'])}-model public projection to {target}")
    else:
        sys.exit(f"unknown command '{cmd}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
