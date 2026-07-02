#!/usr/bin/env python3
from __future__ import annotations

import unittest

from _catalog import (
    CATALOG,
    DEFAULT_CATALOG_MODEL_KIND,
    LANG_BY_FAMILY,
    QUANT_METADATA,
    apply_catalog_series_defaults,
    languages_for_family,
    languages_for_model,
    load,
    load_catalog_series,
)
from _file_loaders import load_toml


EXPECTED_CAPABILITY_PACKS = {
    "pyannote-segmentation-3.0": "speaker-segmenter",
    "wespeaker-voxceleb-resnet34-lm": "speaker-embedder",
}
EXPECTED_TRANSLATION_MODELS = {
    "hymt2-1.8b": (["zh"], ["en"]),
}


class PublishHelpersTest(unittest.TestCase):
    def test_quant_metadata_covers_every_catalog_quant(self) -> None:
        catalog = load()
        observed = sorted({quant for entry in catalog.values() for quant in entry["quants"]})

        for quant in observed:
            self.assertIn(quant, QUANT_METADATA)
            self.assertTrue(QUANT_METADATA[quant].cli_token)
            self.assertTrue(QUANT_METADATA[quant].suffix)
            self.assertTrue(QUANT_METADATA[quant].label)

        self.assertEqual(QUANT_METADATA["q8_0"].cli_token, "q8-0")
        self.assertEqual(QUANT_METADATA["q8_0"].suffix, "q8")
        self.assertEqual(QUANT_METADATA["q4_k_m"].cli_token, "q4-k-m")
        self.assertEqual(QUANT_METADATA["q4_k_m"].suffix, "q4km")

    def test_every_catalog_model_resolves_a_language_mapping(self) -> None:
        catalog = load()

        for model, entry in sorted(catalog.items()):
            family = entry["family"]
            if family not in LANG_BY_FAMILY:
                # Language-agnostic support packs (e.g. the speaker-diarization
                # embedder/segmenter) have no family-wide ASR language list;
                # each such model must then declare an explicit override.
                self.assertTrue(
                    entry.get("languages"),
                    f"model '{model}' (family '{family}') has neither a family "
                    "language mapping nor an explicit 'languages' override",
                )
            langs = languages_for_model({**entry, "id": model})
            self.assertIsInstance(langs, list)
            self.assertTrue(langs, f"model '{model}' has no languages")
            for code in langs:
                # ISO 639-1 (2 letters) or 639-3 (3 letters); languages only.
                self.assertRegex(
                    code, r"^[a-z]{2,3}$", f"model '{model}' code {code!r} is not an ISO 639 code"
                )
            self.assertEqual(
                langs, sorted(set(langs)), f"model '{model}' languages must be sorted + de-duped"
            )

    def test_languages_for_model_prefers_explicit_override(self) -> None:
        # Language support is per-model: a model (e.g. a Whisper English-only
        # *.en checkpoint) may declare fewer languages than its family default.
        self.assertEqual(
            languages_for_model({"id": "whisper-tiny.en", "family": "whisper", "languages": ["en"]}),
            ["en"],
        )
        # Without an override the model inherits the family default.
        self.assertEqual(
            languages_for_model({"id": "whisper-small", "family": "whisper"}),
            languages_for_family("whisper"),
        )

    def test_catalog_path_points_to_models_toml(self) -> None:
        self.assertEqual(CATALOG.name, "models-core.toml")

    def test_catalog_kinds_and_capabilities_are_explicitly_resolved(self) -> None:
        source = load_toml(CATALOG)
        catalog = load()
        explicit_capability_packs = {
            model: entry["capability"]["role"]
            for model, entry in source.items()
            if entry.get("kind") == "capability-pack"
        }
        explicit_translation_models = {
            model: (entry["source_langs"], entry["target_langs"])
            for model, entry in source.items()
            if entry.get("kind") == "translation-model"
        }

        self.assertEqual(explicit_capability_packs, EXPECTED_CAPABILITY_PACKS)
        self.assertEqual(explicit_translation_models, EXPECTED_TRANSLATION_MODELS)

        for model, entry in sorted(catalog.items()):
            self.assertIn("kind", entry, model)
            if model in explicit_capability_packs:
                self.assertEqual(entry["kind"], "capability-pack")
                self.assertEqual(entry["capability"]["feature"], "speaker-diarization")
                self.assertEqual(entry["capability"]["role"], explicit_capability_packs[model])
            elif model in explicit_translation_models:
                self.assertEqual(entry["kind"], "translation-model")
                self.assertEqual(
                    (entry["source_langs"], entry["target_langs"]),
                    explicit_translation_models[model],
                )
                self.assertNotIn("capability", entry)
            else:
                self.assertEqual(entry["kind"], DEFAULT_CATALOG_MODEL_KIND)
                self.assertNotIn("capability", entry)

        explicit_non_default = set(explicit_capability_packs) | set(explicit_translation_models)
        for model in sorted(set(catalog) - explicit_non_default):
            self.assertNotIn("kind", source[model], f"{model} should exercise the default kind")

    def test_missing_kind_defaults_to_asr_model_without_family_inference(self) -> None:
        entry = {"family": "wespeaker"}

        apply_catalog_series_defaults("synthetic-wespeaker", entry, {})

        self.assertEqual(entry["kind"], DEFAULT_CATALOG_MODEL_KIND)
        self.assertNotIn("capability", entry)

    def test_qwen_catalog_aliases_are_series_owned(self) -> None:
        catalog = load()
        series = load_catalog_series()

        self.assertEqual(catalog["qwen3-asr-0.6b"]["aliases"], ["qwen3", "qwen3-asr"])
        self.assertEqual(catalog["qwen3-asr-0.6b"]["pull_alias"], "qwen3")
        self.assertEqual(catalog["qwen3-asr-1.7b"]["aliases"], ["qwen3", "qwen3-asr"])
        self.assertEqual(catalog["qwen3-asr-1.7b"]["pull_alias"], "qwen3")
        self.assertIn("qwen-asr", series["qwen"]["aliases"])

    def test_xasr_catalog_aliases_are_series_owned(self) -> None:
        catalog = load()
        series = load_catalog_series()

        self.assertEqual(catalog["xasr-zh-en"]["aliases"], ["xasr", "x-asr"])
        self.assertEqual(catalog["xasr-zh-en"]["pull_alias"], "xasr")
        self.assertIn("xasr-zipformer", series["xasr-zipformer"]["aliases"])


if __name__ == "__main__":
    unittest.main()
