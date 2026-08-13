#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

from _catalog import (
    LANGUAGE_DISPLAY_LABELS,
    MODEL_FAMILY_INVENTORY,
    MODEL_FAMILY_INVENTORY_SCHEMA,
    REGISTERED_DIALECT_CODES,
    _check_no_halfwidth_punct_between_cjk,
    apply_catalog_series_defaults,
    apply_speaker_sources_to_catalog,
    apply_word_timestamp_sources_to_catalog,
    language_labels_wire,
    language_mode_for_model,
    languages_for_model,
    load_model_family_inventory,
    prose_locale_source_sha256,
    punctuation_for_model,
    speaker_source_for_model,
    validate_all_card_prose_locales,
    validate_card_prose_locales,
    validate_display_ranking,
    validate_prose_locale_block,
    validate_recognition_language_code,
    validate_min_core_version,
    validate_recognition_languages,
    validate_upstream_release_date,
    word_timestamp_source_for_model,
)


EN_TAGLINE = "Dedicated 2B ASR for 14-language transcription"
EN_HIGHLIGHTS = [
    "🎙️ **Dedicated ASR** — audio-in, text-out model built specifically for transcription",
    "🌍 **14 languages** — covers a wide range of scripts",
]


class ModelFamilyInventoryTest(unittest.TestCase):
    def _write_payload(self, payload: dict) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "inventory.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def _committed_payload(self) -> dict:
        return json.loads(MODEL_FAMILY_INVENTORY.read_text(encoding="utf-8"))

    def test_committed_inventory_is_strictly_valid(self) -> None:
        inventory = load_model_family_inventory()
        payload = self._committed_payload()
        self.assertEqual(payload["schema"], MODEL_FAMILY_INVENTORY_SCHEMA)
        self.assertEqual(len(inventory), len(set(inventory)))
        self.assertEqual(set(inventory), {family["catalog_family_id"] for family in payload["families"]})

    def test_rejects_wrong_schema(self) -> None:
        payload = self._committed_payload()
        payload["schema"] = "openasr.model-family-inventory.v0"
        with self.assertRaisesRegex(ValueError, "schema"):
            load_model_family_inventory(self._write_payload(payload))

    def test_rejects_non_list_families(self) -> None:
        payload = self._committed_payload()
        payload["families"] = {}
        with self.assertRaisesRegex(ValueError, "families.*non-empty list"):
            load_model_family_inventory(self._write_payload(payload))

    def test_rejects_non_object_family_entry(self) -> None:
        payload = self._committed_payload()
        payload["families"][0] = "not-an-object"
        with self.assertRaisesRegex(ValueError, r"families\[0\].*object"):
            load_model_family_inventory(self._write_payload(payload))

    def test_rejects_duplicate_catalog_family_id(self) -> None:
        payload = self._committed_payload()
        payload["families"][1]["catalog_family_id"] = payload["families"][0]["catalog_family_id"]
        with self.assertRaisesRegex(ValueError, "duplicate catalog_family_id"):
            load_model_family_inventory(self._write_payload(payload))

    def test_rejects_missing_required_family_field(self) -> None:
        payload = self._committed_payload()
        del payload["families"][0]["execution"]
        with self.assertRaisesRegex(ValueError, r"families\[0\].*missing execution"):
            load_model_family_inventory(self._write_payload(payload))

    def test_execution_capability_projection_requires_booleans(self) -> None:
        payload = self._committed_payload()
        payload["families"][0]["execution"]["supports_translation_task"] = "yes"
        with self.assertRaisesRegex(
            ValueError,
            r"families\[0\]\.execution\.supports_translation_task.*boolean",
        ):
            load_model_family_inventory(self._write_payload(payload))

    def test_execution_typed_capability_projection(self) -> None:
        inventory = load_model_family_inventory()
        phrase_bias_strategies = {
            family["execution"]["phrase_bias_strategy"]
            for family in inventory.values()
        }
        word_timestamp_strategies = {
            family["execution"]["word_timestamp_strategy"]
            for family in inventory.values()
        }
        prepared_runtime_strategies = {
            family["execution"]["prepared_runtime"]
            for family in inventory.values()
        }

        self.assertEqual(
            phrase_bias_strategies,
            {"unsupported", "always", "requires-tensor"},
        )
        self.assertEqual(
            word_timestamp_strategies,
            {"decode-invariant", "decode-sensitive"},
        )
        self.assertIn("family-owned", prepared_runtime_strategies)
        self.assertTrue(
            any(
                prepared_runtime.startswith("shared-")
                for prepared_runtime in prepared_runtime_strategies
            )
        )

        for family in inventory.values():
            execution = family["execution"]
            self.assertIn(
                execution["adapter_binding"],
                {"unsupported", "qwen3-asr-lora-v1", "moonshine-lora-v1"},
            )
            self.assertEqual(
                execution["supports_phrase_bias"],
                execution["phrase_bias_strategy"] != "unsupported",
            )
            if execution["phrase_bias_strategy"] == "requires-tensor":
                self.assertIsInstance(execution["phrase_bias_required_tensor"], str)
                self.assertTrue(execution["phrase_bias_required_tensor"])
            else:
                self.assertIsNone(execution["phrase_bias_required_tensor"])
            self.assertIn(
                execution["word_timestamp_strategy"],
                {"decode-invariant", "decode-sensitive"},
            )
            prepared_runtime = execution["prepared_runtime"]
            self.assertTrue(
                prepared_runtime == "family-owned"
                or re.fullmatch(
                    r"shared-[a-z0-9-]+-v[1-9][0-9]*", prepared_runtime
                )
            )

    def test_accepts_new_shared_prepared_runtime_component_without_allowlist(self) -> None:
        payload = self._committed_payload()
        payload["families"][0]["execution"]["prepared_runtime"] = "shared-future-component-v17"
        load_model_family_inventory(self._write_payload(payload))

    def test_rejects_unknown_adapter_binding(self) -> None:
        payload = self._committed_payload()
        payload["families"][0]["execution"]["adapter_binding"] = "future-lora-v1"
        with self.assertRaisesRegex(
            ValueError,
            r"families\[0\]\.execution\.adapter_binding.*unsupported binding",
        ):
            load_model_family_inventory(self._write_payload(payload))

    def test_rejects_phrase_bias_strategy_mismatch(self) -> None:
        payload = self._committed_payload()
        execution = payload["families"][0]["execution"]
        execution["supports_phrase_bias"] = not execution["supports_phrase_bias"]
        with self.assertRaisesRegex(
            ValueError,
            r"supports_phrase_bias.*equivalent to phrase_bias_strategy",
        ):
            load_model_family_inventory(self._write_payload(payload))

    def test_rejects_invalid_phrase_bias_strategy(self) -> None:
        payload = self._committed_payload()
        payload["families"][0]["execution"]["phrase_bias_strategy"] = "sometimes"
        with self.assertRaisesRegex(ValueError, "unsupported strategy 'sometimes'"):
            load_model_family_inventory(self._write_payload(payload))

    def test_rejects_phrase_bias_required_tensor_for_non_tensor_strategy(self) -> None:
        payload = self._committed_payload()
        execution = payload["families"][0]["execution"]
        execution["phrase_bias_required_tensor"] = "unexpected.tensor"
        with self.assertRaisesRegex(
            ValueError,
            r"phrase_bias_required_tensor.*must be null unless",
        ):
            load_model_family_inventory(self._write_payload(payload))

    def test_rejects_missing_phrase_bias_required_tensor(self) -> None:
        payload = self._committed_payload()
        execution = next(
            family["execution"]
            for family in payload["families"]
            if family["execution"]["phrase_bias_strategy"] == "requires-tensor"
        )
        execution["phrase_bias_required_tensor"] = None
        with self.assertRaisesRegex(
            ValueError,
            r"phrase_bias_required_tensor.*non-empty string",
        ):
            load_model_family_inventory(self._write_payload(payload))

    def test_rejects_invalid_word_timestamp_strategy(self) -> None:
        payload = self._committed_payload()
        payload["families"][0]["execution"]["word_timestamp_strategy"] = "sometimes"
        with self.assertRaisesRegex(ValueError, "unsupported strategy 'sometimes'"):
            load_model_family_inventory(self._write_payload(payload))

    def test_rejects_invalid_prepared_runtime_strategy(self) -> None:
        for prepared_runtime in ("shared-future-component-v0", "shared-Future-v1", "runtime-owned"):
            payload = self._committed_payload()
            payload["families"][0]["execution"]["prepared_runtime"] = prepared_runtime
            with self.subTest(prepared_runtime=prepared_runtime):
                with self.assertRaisesRegex(
                    ValueError,
                    r"prepared_runtime.*family-owned or match shared-",
                ):
                    load_model_family_inventory(self._write_payload(payload))

    def test_quantization_contract_declares_axis_by_classification(self) -> None:
        inventory = load_model_family_inventory()
        for family in inventory.values():
            quantization = family["quantization"]
            if quantization["tensor_classification"] == "semantic-roles-v1":
                self.assertIn(quantization["quantized_axis"], {"first", "last"})
            else:
                self.assertIn(
                    quantization["tensor_classification"],
                    {"entire-acoustic-pack", "not-applicable"},
                )
                self.assertIsNone(quantization["quantized_axis"])

    def test_dialect_projection_declares_all_three_strategies(self) -> None:
        inventory = load_model_family_inventory()
        self.assertEqual(
            inventory["dolphin"]["language"]["dialect_mode"],
            "selects-via-prompt",
        )
        self.assertEqual(
            inventory["qwen"]["language"]["dialect_mode"],
            "recognizes-catalog-declared",
        )
        self.assertEqual(
            inventory["firered-aed"]["language"]["dialect_mode"],
            "recognizes-catalog-declared",
        )
        self.assertTrue(
            inventory["dolphin"]["language"]["selectable_dialect_codes"]
        )
        self.assertEqual(
            inventory["dolphin"]["language"]["selectable_dialect_codes"],
            sorted(set(inventory["dolphin"]["language"]["selectable_dialect_codes"])),
        )
        self.assertTrue(
            all(
                family["language"]["dialect_mode"] == "not-advertised"
                and not family["language"]["selectable_dialect_codes"]
                for catalog_family_id, family in inventory.items()
                if catalog_family_id not in {"dolphin", "qwen", "firered-aed"}
            )
        )

    def test_rejects_invalid_dialect_projection(self) -> None:
        payload = self._committed_payload()
        dolphin = next(family for family in payload["families"] if family["catalog_family_id"] == "dolphin")
        dolphin["language"]["dialect_mode"] = "unknown"
        with self.assertRaisesRegex(ValueError, "unsupported dialect mode"):
            load_model_family_inventory(self._write_payload(payload))

        payload = self._committed_payload()
        dolphin = next(family for family in payload["families"] if family["catalog_family_id"] == "dolphin")
        dolphin["language"]["selectable_dialect_codes"] = []
        with self.assertRaisesRegex(ValueError, "must be non-empty for selects-via-prompt"):
            load_model_family_inventory(self._write_payload(payload))

        payload = self._committed_payload()
        qwen = next(family for family in payload["families"] if family["catalog_family_id"] == "qwen")
        qwen["language"]["selectable_dialect_codes"] = ["zh-sichuan"]
        with self.assertRaisesRegex(ValueError, "must be empty for recognizes-catalog-declared"):
            load_model_family_inventory(self._write_payload(payload))

        payload = self._committed_payload()
        dolphin = next(family for family in payload["families"] if family["catalog_family_id"] == "dolphin")
        codes = dolphin["language"]["selectable_dialect_codes"]
        dolphin["language"]["selectable_dialect_codes"] = [codes[1], codes[0], codes[0]]
        with self.assertRaisesRegex(ValueError, "must be sorted and unique"):
            load_model_family_inventory(self._write_payload(payload))

    def test_rejects_invalid_language_projection(self) -> None:
        payload = self._committed_payload()
        family = payload["families"][0]
        family["language"]["languages"] = []
        with self.assertRaisesRegex(ValueError, r"language\.languages.*non-empty"):
            load_model_family_inventory(self._write_payload(payload))

        payload = self._committed_payload()
        family = payload["families"][0]
        family["language"]["languages"] = ["en", "en"]
        with self.assertRaisesRegex(ValueError, r"language\.languages.*sorted and unique"):
            load_model_family_inventory(self._write_payload(payload))

        payload = self._committed_payload()
        family = payload["families"][0]
        family["language"]["languages"] = ["EN"]
        with self.assertRaisesRegex(ValueError, r"language\.languages.*ISO 639"):
            load_model_family_inventory(self._write_payload(payload))

        payload = self._committed_payload()
        family = next(
            family for family in payload["families"] if family["language"]["policy"] == "selects-via-prompt"
        )
        family["language"]["default_language"] = "xx"
        with self.assertRaisesRegex(ValueError, r"default_language.*present in languages"):
            load_model_family_inventory(self._write_payload(payload))

    def test_rejects_invalid_module_slug(self) -> None:
        payload = self._committed_payload()
        payload["families"][0]["module_slug"] = "not-a-slug"
        with self.assertRaisesRegex(ValueError, r"module_slug.*snake_case"):
            load_model_family_inventory(self._write_payload(payload))

        payload = self._committed_payload()
        payload["families"][1]["module_slug"] = payload["families"][0]["module_slug"]
        with self.assertRaisesRegex(ValueError, r"module_slug.*duplicate"):
            load_model_family_inventory(self._write_payload(payload))

    def test_rejects_missing_quantized_axis(self) -> None:
        payload = self._committed_payload()
        del payload["families"][0]["quantization"]["quantized_axis"]
        with self.assertRaisesRegex(ValueError, r"families\[0\]\.quantization.*missing quantized_axis"):
            load_model_family_inventory(self._write_payload(payload))

    def test_rejects_semantic_roles_without_quantized_axis(self) -> None:
        payload = self._committed_payload()
        family = next(
            family
            for family in payload["families"]
            if family["quantization"]["tensor_classification"] == "semantic-roles-v1"
        )
        family["quantization"]["quantized_axis"] = None
        with self.assertRaisesRegex(ValueError, "semantic-roles-v1 requires 'first' or 'last'"):
            load_model_family_inventory(self._write_payload(payload))

    def test_accepts_nonsemantic_quantization_contracts_without_axis(self) -> None:
        payload = self._committed_payload()
        for classification in ("entire-acoustic-pack", "not-applicable"):
            candidate = json.loads(json.dumps(payload))
            candidate["families"][0]["quantization"] = {
                "tensor_classification": classification,
                "quantized_axis": None,
            }
            with self.subTest(classification=classification):
                load_model_family_inventory(self._write_payload(candidate))

    def test_rejects_nonsemantic_quantization_with_quantized_axis(self) -> None:
        payload = self._committed_payload()
        payload["families"][0]["quantization"] = {
            "tensor_classification": "entire-acoustic-pack",
            "quantized_axis": "first",
        }
        with self.assertRaisesRegex(
            ValueError,
            "entire-acoustic-pack requires null",
        ):
            load_model_family_inventory(self._write_payload(payload))

    def test_rejects_unknown_quantized_axis(self) -> None:
        payload = self._committed_payload()
        payload["families"][0]["quantization"]["quantized_axis"] = "middle"
        with self.assertRaisesRegex(ValueError, "unsupported axis 'middle'"):
            load_model_family_inventory(self._write_payload(payload))

    def test_unknown_family_capability_fails_closed(self) -> None:
        with self.assertRaisesRegex(KeyError, "no emits_punctuation mapping"):
            punctuation_for_model({"kind": "asr-model", "family": "unknown"})


def valid_locale_block() -> dict:
    return {
        "tagline": "面向转写打造的 2B 专用语音识别模型，覆盖 14 种语言",
        "highlights": [
            "🎙️ **专用语音识别** — 面向转写任务、音频输入文本输出的模型",
            "🌍 **14 种语言** — 覆盖广泛的文字体系",
        ],
        "source_sha256": prose_locale_source_sha256(EN_TAGLINE, EN_HIGHLIGHTS),
    }


class DisplayRankingTest(unittest.TestCase):
    def test_sort_weight_and_recommended_default_to_absent(self) -> None:
        entry: dict = {"family": "whisper"}
        validate_display_ranking("m", entry)
        self.assertNotIn("sort_weight", entry)
        self.assertNotIn("recommended", entry)

    def test_sort_weight_must_be_int_not_bool(self) -> None:
        with self.assertRaises(KeyError):
            validate_display_ranking("m", {"sort_weight": True})

    def test_sort_weight_rejects_non_int(self) -> None:
        with self.assertRaises(KeyError):
            validate_display_ranking("m", {"sort_weight": "920"})

    def test_recommended_must_be_bool(self) -> None:
        with self.assertRaises(KeyError):
            validate_display_ranking("m", {"recommended": "true"})

    def test_valid_values_pass(self) -> None:
        entry = {"sort_weight": 920, "recommended": True}
        validate_display_ranking("m", entry)  # must not raise

    def test_apply_catalog_series_defaults_accepts_valid_ranking(self) -> None:
        entry = {"family": "whisper", "size": "tiny", "sort_weight": 10, "recommended": False}
        apply_catalog_series_defaults("m", entry, {})
        self.assertEqual(entry["sort_weight"], 10)
        self.assertFalse(entry["recommended"])


class UpstreamReleaseDateTest(unittest.TestCase):
    def test_absent_field_is_a_noop(self) -> None:
        validate_upstream_release_date("m", {"family": "whisper"})  # must not raise

    def test_explicit_none_is_a_noop(self) -> None:
        validate_upstream_release_date("m", {"upstream_release_date": None})  # must not raise

    def test_valid_past_date_passes(self) -> None:
        validate_upstream_release_date("m", {"upstream_release_date": "2022-09-21"})

    def test_today_passes(self) -> None:
        validate_upstream_release_date("m", {"upstream_release_date": date.today().isoformat()})

    def test_rejects_wrong_format(self) -> None:
        with self.assertRaisesRegex(KeyError, "ISO yyyy-mm-dd"):
            validate_upstream_release_date("m", {"upstream_release_date": "2022/09/21"})

    def test_rejects_non_string(self) -> None:
        with self.assertRaisesRegex(KeyError, "ISO yyyy-mm-dd"):
            validate_upstream_release_date("m", {"upstream_release_date": 20220921})

    def test_rejects_impossible_calendar_date(self) -> None:
        with self.assertRaisesRegex(KeyError, "not a valid calendar date"):
            validate_upstream_release_date("m", {"upstream_release_date": "2022-13-40"})

    def test_rejects_future_date(self) -> None:
        future = (date.today() + timedelta(days=1)).isoformat()
        with self.assertRaisesRegex(KeyError, "in the future"):
            validate_upstream_release_date("m", {"upstream_release_date": future})

    def test_apply_catalog_series_defaults_runs_the_check(self) -> None:
        future = (date.today() + timedelta(days=1)).isoformat()
        entry = {"family": "whisper", "size": "tiny", "upstream_release_date": future}
        with self.assertRaisesRegex(KeyError, "in the future"):
            apply_catalog_series_defaults("m", entry, {})


class MinCoreVersionTest(unittest.TestCase):
    def test_absent_field_is_a_noop(self) -> None:
        validate_min_core_version("m", {"family": "whisper"})  # must not raise

    def test_explicit_none_is_a_noop(self) -> None:
        validate_min_core_version("m", {"min_core_version": None})  # must not raise

    def test_valid_triplet_passes(self) -> None:
        validate_min_core_version("m", {"min_core_version": "0.1.3"})

    def test_rejects_two_component_version(self) -> None:
        with self.assertRaisesRegex(KeyError, "major.minor.patch"):
            validate_min_core_version("m", {"min_core_version": "0.1"})

    def test_rejects_prerelease_suffix(self) -> None:
        with self.assertRaisesRegex(KeyError, "major.minor.patch"):
            validate_min_core_version("m", {"min_core_version": "0.1.3-rc.1"})

    def test_rejects_non_string(self) -> None:
        with self.assertRaisesRegex(KeyError, "major.minor.patch"):
            validate_min_core_version("m", {"min_core_version": 13})

    def test_apply_catalog_series_defaults_runs_the_check(self) -> None:
        entry = {"family": "whisper", "size": "tiny", "min_core_version": "0.1"}
        with self.assertRaisesRegex(KeyError, "major.minor.patch"):
            apply_catalog_series_defaults("m", entry, {})


class ProseLocaleValidationTest(unittest.TestCase):
    def test_valid_block_passes(self) -> None:
        validate_prose_locale_block("m", "zh-CN", EN_TAGLINE, EN_HIGHLIGHTS, valid_locale_block())

    def test_rejects_overview_field(self) -> None:
        block = valid_locale_block()
        block["overview"] = ["not allowed"]
        with self.assertRaisesRegex(KeyError, "must not include 'overview'"):
            validate_prose_locale_block("m", "zh-CN", EN_TAGLINE, EN_HIGHLIGHTS, block)

    def test_rejects_unknown_field(self) -> None:
        block = valid_locale_block()
        block["intro"] = "not allowed either"
        with self.assertRaisesRegex(KeyError, "unknown field"):
            validate_prose_locale_block("m", "zh-CN", EN_TAGLINE, EN_HIGHLIGHTS, block)

    def test_rejects_highlight_count_mismatch(self) -> None:
        block = valid_locale_block()
        block["highlights"] = block["highlights"][:1]
        with self.assertRaisesRegex(KeyError, "highlights count"):
            validate_prose_locale_block("m", "zh-CN", EN_TAGLINE, EN_HIGHLIGHTS, block)

    def test_rejects_bold_marker_count_drift(self) -> None:
        block = valid_locale_block()
        block["highlights"][0] = block["highlights"][0].replace("**", "", 1)  # drop one of two markers
        with self.assertRaisesRegex(KeyError, "'\\*\\*' bold-marker count drifted"):
            validate_prose_locale_block("m", "zh-CN", EN_TAGLINE, EN_HIGHLIGHTS, block)

    def test_rejects_backtick_count_drift(self) -> None:
        en_highlights = ["🦀 **Native** — `.oasr` packs run with no Python"]
        block = {
            "tagline": EN_TAGLINE,
            "highlights": ["🦀 **原生运行** — .oasr 包无需 Python"],  # backticks dropped
            "source_sha256": prose_locale_source_sha256(EN_TAGLINE, en_highlights),
        }
        with self.assertRaisesRegex(KeyError, "backtick count drifted"):
            validate_prose_locale_block("m", "zh-CN", EN_TAGLINE, en_highlights, block)

    def test_rejects_leading_emoji_drift_on_highlight(self) -> None:
        block = valid_locale_block()
        block["highlights"][0] = "🌍" + block["highlights"][0][2:]  # swap emoji
        with self.assertRaisesRegex(KeyError, "leading emoji drifted"):
            validate_prose_locale_block("m", "zh-CN", EN_TAGLINE, EN_HIGHLIGHTS, block)

    def test_tagline_does_not_require_leading_emoji_match(self) -> None:
        # Taglines are plain prose (no emoji prefix by convention); only
        # highlight lines are checked for a matching leading emoji.
        block = valid_locale_block()
        block["tagline"] = "面向转写打造的 2B 专用语音识别模型，覆盖 14 种语言"
        validate_prose_locale_block("m", "zh-CN", EN_TAGLINE, EN_HIGHLIGHTS, block)  # must not raise

    def test_rejects_numeric_token_drift(self) -> None:
        block = valid_locale_block()
        block["highlights"][1] = block["highlights"][1].replace("14", "15")
        with self.assertRaisesRegex(KeyError, "numeric tokens drifted"):
            validate_prose_locale_block("m", "zh-CN", EN_TAGLINE, EN_HIGHLIGHTS, block)

    def test_rejects_stale_source_hash(self) -> None:
        block = valid_locale_block()
        block["source_sha256"] = "0" * 64
        with self.assertRaisesRegex(KeyError, "translation stale"):
            validate_prose_locale_block("m", "zh-CN", EN_TAGLINE, EN_HIGHLIGHTS, block)

    def test_source_hash_changes_when_english_changes(self) -> None:
        original = prose_locale_source_sha256(EN_TAGLINE, EN_HIGHLIGHTS)
        changed = prose_locale_source_sha256(EN_TAGLINE + " updated", EN_HIGHLIGHTS)
        self.assertNotEqual(original, changed)

    def test_rejects_halfwidth_comma_between_cjk_chars(self) -> None:
        # Regression case for the market-page bug: a half-width ASCII comma
        # sandwiched between two CJK characters, as previously shipped in
        # dolphin-cn-dialect-small's highlights_zh ("...表现突出,同时覆盖...").
        block = valid_locale_block()
        block["highlights"][0] = (
            "🀄 **22 种中文方言** — 四川话（川话）表现突出,同时覆盖吴语、粤语"
        )
        with self.assertRaisesRegex(KeyError, "half-width punctuation"):
            validate_prose_locale_block("m", "zh-CN", EN_TAGLINE, EN_HIGHLIGHTS, block)

    def test_rejects_halfwidth_comma_between_cjk_chars_sensevoice_case(self) -> None:
        # Regression case for sensevoice-small's highlights_zh
        # ("...并支持自动语种识别" followed by a halfwidth comma before the
        # next clause, e.g. "...日语、韩语,并支持自动语种识别").
        block = valid_locale_block()
        block["highlights"][0] = "🌏 **多语言、中文优先** — 高精度普通话、粤语,并支持自动语种识别"
        with self.assertRaisesRegex(KeyError, "half-width punctuation"):
            validate_prose_locale_block("m", "zh-CN", EN_TAGLINE, EN_HIGHLIGHTS, block)

    def test_allows_halfwidth_comma_as_english_thousands_separator(self) -> None:
        # A halfwidth comma is fine when at least one neighbor is not CJK,
        # e.g. inside a number run like "400,000" embedded in otherwise
        # full-width Chinese prose (sensevoice-small's real highlight text).
        _check_no_halfwidth_punct_between_cjk(
            "m", "zh-CN", "highlight[0]",
            "🀄 **中文基准表现突出** — 超过 400,000 小时语音训练；上游报告优于 Whisper",
        )  # must not raise

    def test_ignores_non_zh_locales(self) -> None:
        # The check is zh-specific; other locales are free to use halfwidth
        # ASCII punctuation between non-Latin scripts without tripping it.
        block = {
            "tagline": EN_TAGLINE,
            "highlights": EN_HIGHLIGHTS,
            "source_sha256": prose_locale_source_sha256(EN_TAGLINE, EN_HIGHLIGHTS),
        }
        validate_prose_locale_block("m", "en-US", EN_TAGLINE, EN_HIGHLIGHTS, block)  # must not raise

    def test_card_with_no_prose_locales_is_a_noop(self) -> None:
        validate_card_prose_locales("m", {"tagline": EN_TAGLINE, "highlights": EN_HIGHLIGHTS})

    def test_card_prose_locales_must_be_a_table(self) -> None:
        with self.assertRaisesRegex(KeyError, "must be a table"):
            validate_card_prose_locales(
                "m",
                {"tagline": EN_TAGLINE, "highlights": EN_HIGHLIGHTS, "prose_locales": ["not-a-table"]},
            )


class LanguageModeForModelTest(unittest.TestCase):
    def test_qwen_is_detect_implicit(self) -> None:
        entry = {"kind": "asr-model", "family": "qwen"}
        self.assertEqual(
            language_mode_for_model(entry, ["en", "zh"]), {"language_mode": "detect_implicit"}
        )

    def test_xasr_zipformer_is_fixed_multilingual(self) -> None:
        entry = {"kind": "asr-model", "family": "xasr-zipformer"}
        self.assertEqual(
            language_mode_for_model(entry, ["en", "zh"]), {"language_mode": "fixed_multilingual"}
        )

    def test_moonshine_is_fixed_monolingual_with_default(self) -> None:
        entry = {"kind": "asr-model", "family": "moonshine"}
        self.assertEqual(
            language_mode_for_model(entry, ["en"]),
            {"language_mode": "fixed_monolingual", "language_default": "en"},
        )

    def test_cohere_is_specify_only_with_en_default(self) -> None:
        entry = {"kind": "asr-model", "family": "cohere"}
        self.assertEqual(
            language_mode_for_model(entry, ["ar", "en", "zh"]),
            {"language_mode": "specify_only", "language_default": "en"},
        )

    def test_multilingual_whisper_is_detect_and_specify(self) -> None:
        entry = {"kind": "asr-model", "family": "whisper"}
        self.assertEqual(
            language_mode_for_model(entry, ["en", "zh", "ja"]),
            {"language_mode": "detect_and_specify"},
        )

    def test_english_only_whisper_is_fixed_monolingual(self) -> None:
        entry = {"kind": "asr-model", "family": "whisper"}
        self.assertEqual(
            language_mode_for_model(entry, ["en"]),
            {"language_mode": "fixed_monolingual", "language_default": "en"},
        )

    def test_translation_model_is_omitted(self) -> None:
        entry = {"kind": "translation-model", "family": "translator-test"}
        self.assertEqual(language_mode_for_model(entry, ["en", "zh"]), {})


class PunctuationForModelTest(unittest.TestCase):
    def test_dolphin_does_not_emit_punctuation(self) -> None:
        entry = {"kind": "asr-model", "family": "dolphin"}
        self.assertEqual(punctuation_for_model(entry), {"emits_punctuation": False})

    def test_other_asr_families_emit_punctuation(self) -> None:
        for family in ("qwen", "cohere", "whisper", "xasr-zipformer", "moonshine", "sensevoice"):
            entry = {"kind": "asr-model", "family": family}
            self.assertEqual(
                punctuation_for_model(entry),
                {"emits_punctuation": True},
                f"family {family!r} should emit punctuation",
            )

    def test_null_inventory_punctuation_is_omitted(self) -> None:
        for family in ("funasr-nano", "firered2-llm", "mimo-asr", "moss-transcribe-diarize"):
            with self.subTest(family=family):
                self.assertEqual(
                    punctuation_for_model({"kind": "asr-model", "family": family}), {}
                )

    def test_translation_model_is_omitted(self) -> None:
        entry = {"kind": "translation-model", "family": "translator-test"}
        self.assertEqual(punctuation_for_model(entry), {})

    def test_capability_pack_is_omitted(self) -> None:
        entry = {"kind": "capability-pack", "family": "redimnet2"}
        self.assertEqual(punctuation_for_model(entry), {})

    def test_unknown_family_raises(self) -> None:
        entry = {"kind": "asr-model", "family": "made-up-family", "id": "m"}
        with self.assertRaisesRegex(KeyError, "no emits_punctuation mapping"):
            punctuation_for_model(entry)


class SpeakerSourceForModelTest(unittest.TestCase):
    def test_moss_uses_native_tracks(self) -> None:
        entry = {"kind": "asr-model", "family": "moss-transcribe-diarize"}
        self.assertEqual(speaker_source_for_model(entry), {"speaker_source": "native"})

    def test_other_asr_families_use_external_tracks(self) -> None:
        for family in ("qwen", "cohere", "whisper", "firered2-llm", "granite-speech"):
            with self.subTest(family=family):
                entry = {"kind": "asr-model", "family": family}
                self.assertEqual(
                    speaker_source_for_model(entry), {"speaker_source": "external"}
                )

    def test_non_asr_kinds_omit_speaker_source(self) -> None:
        self.assertEqual(
            speaker_source_for_model({"kind": "capability-pack", "family": "redimnet2"}),
            {},
        )

    def test_unknown_family_raises(self) -> None:
        with self.assertRaisesRegex(KeyError, "no speaker_source mapping"):
            speaker_source_for_model(
                {"kind": "asr-model", "family": "made-up-family", "id": "m"}
            )

    def test_catalog_wide_refresh_adds_asr_and_removes_non_asr_field(self) -> None:
        catalog = {
            "models": [
                {"id": "moss"},
                {"id": "qwen"},
                {"id": "segmenter", "speaker_source": "external"},
            ]
        }
        entries = {
            "moss-src": {
                "registry_id": "moss",
                "kind": "asr-model",
                "family": "moss-transcribe-diarize",
            },
            "qwen-src": {
                "registry_id": "qwen",
                "kind": "asr-model",
                "family": "qwen",
            },
            "seg-src": {
                "registry_id": "segmenter",
                "kind": "capability-pack",
                "family": "pyannote-segmentation",
            },
        }
        self.assertEqual(apply_speaker_sources_to_catalog(catalog, entries), 3)
        self.assertEqual(catalog["models"][0]["speaker_source"], "native")
        self.assertEqual(catalog["models"][1]["speaker_source"], "external")
        self.assertNotIn("speaker_source", catalog["models"][2])

    def test_catalog_wide_refresh_rejects_unknown_id(self) -> None:
        with self.assertRaisesRegex(KeyError, "no models-core.toml source"):
            apply_speaker_sources_to_catalog({"models": [{"id": "unknown"}]}, {})


class WordTimestampSourceForModelTest(unittest.TestCase):
    def test_native_anchor_families_are_declared(self) -> None:
        for family in ("qwen", "cohere", "whisper", "parakeet-tdt", "xasr-zipformer", "moonshine"):
            with self.subTest(family=family):
                self.assertEqual(
                    word_timestamp_source_for_model({"kind": "asr-model", "family": family}),
                    {"word_timestamp_source": "native"},
                )

    def test_forced_aligner_families_are_declared(self) -> None:
        for family in ("dolphin", "sensevoice", "firered-aed", "firered2-llm", "funasr-nano", "mimo-asr", "moss-transcribe-diarize", "granite-speech"):
            with self.subTest(family=family):
                self.assertEqual(
                    word_timestamp_source_for_model({"kind": "asr-model", "family": family}),
                    {"word_timestamp_source": "forced_aligner"},
                )

    def test_non_asr_kinds_omit_word_timestamp_source(self) -> None:
        self.assertEqual(
            word_timestamp_source_for_model(
                {"kind": "capability-pack", "family": "qwen3-forced-aligner"}
            ),
            {},
        )

    def test_catalog_wide_refresh_is_fail_closed(self) -> None:
        catalog = {
            "models": [
                {"id": "qwen"},
                {"id": "funasr"},
                {"id": "aligner", "word_timestamp_source": "native"},
            ]
        }
        entries = {
            "qwen-src": {"registry_id": "qwen", "kind": "asr-model", "family": "qwen"},
            "funasr-src": {
                "registry_id": "funasr",
                "kind": "asr-model",
                "family": "funasr-nano",
            },
            "aligner-src": {
                "registry_id": "aligner",
                "kind": "capability-pack",
                "family": "qwen3-forced-aligner",
            },
        }
        self.assertEqual(apply_word_timestamp_sources_to_catalog(catalog, entries), 3)
        self.assertEqual(catalog["models"][0]["word_timestamp_source"], "native")
        self.assertEqual(
            catalog["models"][1]["word_timestamp_source"], "forced_aligner"
        )
        self.assertNotIn("word_timestamp_source", catalog["models"][2])

        with self.assertRaisesRegex(KeyError, "no models-core.toml source"):
            apply_word_timestamp_sources_to_catalog({"models": [{"id": "unknown"}]}, {})


class RecognitionLanguageValidatorTest(unittest.TestCase):
    def test_accepts_plain_iso_and_registered_dialects(self) -> None:
        for code in ("en", "zh", "yue", "fil", "haw", "zh-sichuan", "zh-tw"):
            validate_recognition_language_code("m", code)  # must not raise

    def test_rejects_typo_and_unregistered_region(self) -> None:
        # A typo'd region ships loudly.
        with self.assertRaisesRegex(KeyError, "registered dialect-code set"):
            validate_recognition_language_code("m", "zh-sichaun")
        # Well-formed but unregistered region is rejected (must be registered).
        with self.assertRaisesRegex(KeyError, "registered dialect-code set"):
            validate_recognition_language_code("m", "zh-cn")

    def test_rejects_malformed_shape(self) -> None:
        for bad in ("EN", "e", "abcd", "zh-", "-zh", "zh-a-b", "zh_sichuan"):
            with self.assertRaises(KeyError):
                validate_recognition_language_code("m", bad)

    def test_selective_collapse_blocks_dialect_on_non_dialect_family(self) -> None:
        # A non-dialect-capable family may not enumerate dialect codes.
        with self.assertRaisesRegex(KeyError, "not-advertised"):
            validate_recognition_languages("cohere-transcribe", "cohere", ["zh", "zh-sichuan"])
        # Dolphin (dialect-capable via a code->prompt map) may.
        validate_recognition_languages(
            "dolphin-cn-dialect-small", "dolphin", ["zh", "zh-sichuan"]
        )
        # firered-aed and qwen advertise catalog-declared recognition coverage,
        # not a selectable prompt, so they may enumerate dialects too.
        validate_recognition_languages(
            "firered-aed-l-v2", "firered-aed", ["en", "zh", "zh-sichuan"]
        )
        validate_recognition_languages("qwen3-asr-1.7b", "qwen", ["zh", "zh-sichuan"])

    def test_dolphin_family_advertises_base_plus_its_own_dialect_codes(self) -> None:
        # Dolphin's family default is built from its inventory selectable codes,
        # not the broader cross-family REGISTERED_DIALECT_CODES set.
        inventory = load_model_family_inventory()
        codes = inventory["dolphin"]["language"]["selectable_dialect_codes"]
        expected = sorted(["zh", *codes])
        self.assertLess(set(codes), set(REGISTERED_DIALECT_CODES))
        # Resolving through the public seam validates + returns the same set.
        resolved = languages_for_model({"id": "dolphin-cn-dialect-small", "family": "dolphin"})
        self.assertEqual(resolved, expected)


class LanguageLabelsWireTest(unittest.TestCase):
    def test_wire_shape_is_code_to_en_and_zh_cn(self) -> None:
        wire = language_labels_wire()
        # Every curated code is present with exactly {en, zh-CN}.
        self.assertEqual(set(wire), set(LANGUAGE_DISPLAY_LABELS))
        for code, entry in wire.items():
            self.assertEqual(set(entry), {"en", "zh-CN"})
            en, zh_cn = LANGUAGE_DISPLAY_LABELS[code]
            self.assertEqual(entry["en"], en)
            self.assertEqual(entry["zh-CN"], zh_cn)

    def test_wire_is_sorted_by_code(self) -> None:
        wire = language_labels_wire()
        self.assertEqual(list(wire), sorted(wire))

    def test_every_registered_dialect_has_a_label(self) -> None:
        for code in REGISTERED_DIALECT_CODES:
            self.assertIn(code, LANGUAGE_DISPLAY_LABELS)
        # Registered dialect set is sorted + de-duplicated (catalog invariant).
        self.assertEqual(REGISTERED_DIALECT_CODES, sorted(set(REGISTERED_DIALECT_CODES)))

    def test_capability_pack_is_omitted(self) -> None:
        entry = {"kind": "capability-pack", "family": "redimnet2"}
        self.assertEqual(language_mode_for_model(entry, ["en", "zh"]), {})

    def test_unknown_family_raises(self) -> None:
        entry = {"kind": "asr-model", "family": "made-up-family", "id": "m"}
        with self.assertRaisesRegex(KeyError, "no language_mode mapping"):
            language_mode_for_model(entry, ["en"])

    def test_fixed_monolingual_rejects_multiple_languages(self) -> None:
        entry = {"kind": "asr-model", "family": "moonshine", "id": "m"}
        with self.assertRaisesRegex(KeyError, "exactly one language"):
            language_mode_for_model(entry, ["en", "fr"])


class AllCardsProseLocalesTest(unittest.TestCase):
    def test_every_authored_card_prose_locale_is_valid_and_fresh(self) -> None:
        # Exercises the same check regenerate_all.sh --check runs: every card's
        # prose_locales block (if any) must be internally consistent with its
        # English tagline/highlights and not stale.
        translated = validate_all_card_prose_locales()
        self.assertIsInstance(translated, list)
        self.assertIn("qwen3-asr-1.7b", translated)
        self.assertEqual(len(translated), len(set(translated)))


if __name__ == "__main__":
    unittest.main()
