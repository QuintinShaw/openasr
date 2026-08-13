#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("render_card.py")
sys.path.insert(0, str(SCRIPT.parent))

from render_card import card_type_for_catalog, pipeline_tag_for_catalog, pull_command  # noqa: E402


class RenderCardTest(unittest.TestCase):
    def test_renderer_uses_repo_owned_template_and_prose(self) -> None:
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "moonshine-tiny"],
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("# Moonshine Tiny · OpenASR", result.stdout)
        self.assertIn("Tiny 27M-parameter English ASR", result.stdout)
        self.assertIn("Native in OpenASR", result.stdout)
        self.assertNotIn("{{", result.stdout)

    def test_card_type_and_pipeline_are_derived_from_catalog_semantics(self) -> None:
        self.assertEqual(card_type_for_catalog({"kind": "asr-model"}), "asr")
        with self.assertRaisesRegex(ValueError, "catalog kind is reserved"):
            card_type_for_catalog(
                {
                    "kind": "translation-model",
                    "source_langs": ["zh"],
                    "target_langs": ["en"],
                }
            )
        self.assertEqual(
            card_type_for_catalog(
                {
                    "kind": "capability-pack",
                    "capability": {
                        "feature": "speaker-diarization",
                        "role": "speaker-segmenter",
                    },
                }
            ),
            "diarize",
        )
        self.assertEqual(
            card_type_for_catalog(
                {
                    "kind": "capability-pack",
                    "capability": {"feature": "punctuation", "role": "punctuation-restorer"},
                }
            ),
            "capability",
        )

        self.assertEqual(
            pipeline_tag_for_catalog(
                {
                    "kind": "capability-pack",
                    "capability": {
                        "feature": "speaker-diarization",
                        "role": "speaker-embedder",
                    },
                },
                {},
            ),
            "feature-extraction",
        )
        self.assertEqual(
            pipeline_tag_for_catalog(
                {
                    "kind": "capability-pack",
                    "capability": {"feature": "word-timestamps", "role": "forced-aligner"},
                },
                {},
            ),
            "automatic-speech-recognition",
        )
        with self.assertRaisesRegex(ValueError, "catalog kind is reserved"):
            pipeline_tag_for_catalog({"kind": "translation-model"}, {})

    def test_renderer_fails_closed_without_semantic_catalog_fields(self) -> None:
        with self.assertRaisesRegex(ValueError, "supported catalog kind"):
            card_type_for_catalog({"family": "redimnet2"})

        with self.assertRaisesRegex(ValueError, "missing capability metadata"):
            card_type_for_catalog({"kind": "capability-pack"})

        with self.assertRaisesRegex(ValueError, "requires feature"):
            card_type_for_catalog(
                {
                    "kind": "capability-pack",
                    "capability": {"feature": "punctuation", "role": "speaker-segmenter"},
                }
            )

    def test_semantic_diarization_pack_uses_diarize_card(self) -> None:
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "diarizen-large-s80-v2"],
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("pipeline_tag: voice-activity-detection", result.stdout)
        self.assertIn("Speaker-diarization support pack", result.stdout)
        self.assertIn(
            "openasr pull diarizen-large-s80-v2:fp16 --accept-license",
            result.stdout,
        )
        self.assertIn("python3 tooling/diarizen/convert_diarizen.py", result.stdout)
        self.assertNotIn("openasr model-pack external:", result.stdout)
        self.assertNotIn("pipeline_tag: automatic-speech-recognition", result.stdout)

    def test_pull_command_keeps_permissive_simple_and_requires_restricted_consent(self) -> None:
        self.assertEqual(
            pull_command({"license_class": "permissive"}, "segmentation:fp16"),
            "openasr pull segmentation:fp16",
        )
        self.assertEqual(
            pull_command({"license_class": "noncommercial"}, "diarizen:fp16"),
            "openasr pull diarizen:fp16 --accept-license",
        )

    def test_forced_aligner_card_uses_the_public_importer(self) -> None:
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "qwen3-forced-aligner-0.6b"],
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "openasr model-pack import qwen-forced-aligner ...", result.stdout
        )
        self.assertNotIn("not yet wired", result.stdout)


if __name__ == "__main__":
    unittest.main()
