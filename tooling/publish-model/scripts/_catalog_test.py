#!/usr/bin/env python3
from __future__ import annotations

import unittest

from _catalog import (
    apply_catalog_series_defaults,
    language_mode_for_model,
    prose_locale_source_sha256,
    validate_all_card_prose_locales,
    validate_card_prose_locales,
    validate_display_ranking,
    validate_prose_locale_block,
)


EN_TAGLINE = "Dedicated 2B ASR for 14-language transcription"
EN_HIGHLIGHTS = [
    "🎙️ **Dedicated ASR** — audio-in, text-out model built specifically for transcription",
    "🌍 **14 languages** — covers a wide range of scripts",
]


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
        entry = {"kind": "translation-model", "family": "hymt2"}
        self.assertEqual(language_mode_for_model(entry, ["en", "zh"]), {})

    def test_capability_pack_is_omitted(self) -> None:
        entry = {"kind": "capability-pack", "family": "wespeaker"}
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
