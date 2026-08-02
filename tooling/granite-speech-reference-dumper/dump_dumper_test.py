#!/usr/bin/env python3
"""Weight-free self-tests for the granite-speech reference dumper helpers.

No checkpoint, no network, no torch model load -- just the pure argparse /
prompt-assembly / npy round-trip logic, so `python3 -m unittest` stays fast
and hermetic in CI. Mirrors `../firered2-reference-dumper/dump_reference_test.py`
in spirit (that dumper has enough standalone arithmetic to unit-test; this one
tests the pieces that do not need weights).
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

import dump_golden as golden
import dump_intermediate as intermediate


class SampleArgParsingTest(unittest.TestCase):
    def test_parses_name_and_relative_path(self) -> None:
        self.assertEqual(golden.parse_sample_arg("jfk=jfk.wav"), ("jfk", "jfk.wav"))
        self.assertEqual(
            golden.parse_sample_arg("en_short=samples/en_short.wav"),
            ("en_short", "samples/en_short.wav"),
        )

    def test_rejects_missing_separator(self) -> None:
        with self.assertRaises(Exception):
            golden.parse_sample_arg("jfk.wav")

    def test_rejects_empty_sides(self) -> None:
        with self.assertRaises(Exception):
            golden.parse_sample_arg("=jfk.wav")
        with self.assertRaises(Exception):
            golden.parse_sample_arg("jfk=")


class DtypeParsingTest(unittest.TestCase):
    def test_accepts_aliases(self) -> None:
        self.assertEqual(golden.parse_dtype("bf16"), torch.bfloat16)
        self.assertEqual(golden.parse_dtype("bfloat16"), torch.bfloat16)
        self.assertEqual(golden.parse_dtype("float32"), torch.float32)
        self.assertEqual(golden.parse_dtype("fp16"), torch.float16)

    def test_rejects_unknown(self) -> None:
        with self.assertRaises(Exception):
            golden.parse_dtype("float8")


class PromptAssemblyTest(unittest.TestCase):
    def test_default_question_matches_executor(self) -> None:
        self.assertEqual(
            golden.DEFAULT_QUESTION,
            "can you transcribe the speech into a written format?",
        )

    def test_build_user_prompt_prefixes_audio_token(self) -> None:
        prompt = golden.build_user_prompt(golden.DEFAULT_QUESTION)
        self.assertTrue(prompt.startswith(golden.AUDIO_TOKEN))
        self.assertEqual(
            prompt, f"{golden.AUDIO_TOKEN}{golden.DEFAULT_QUESTION}"
        )
        # No intervening space -- matches the HF model-card / executor shape.
        self.assertFalse(prompt.startswith(f"{golden.AUDIO_TOKEN} "))

    def test_build_user_prompt_does_not_double_placeholder(self) -> None:
        already = f"{golden.AUDIO_TOKEN}transcribe the speech to text."
        self.assertEqual(golden.build_user_prompt(already), already)

    def test_kwb_question_shape(self) -> None:
        q = "transcribe the speech to text. Keywords: OpenASR, Granite"
        prompt = golden.build_user_prompt(q)
        self.assertEqual(prompt, f"{golden.AUDIO_TOKEN}{q}")


class DecoderProbeTest(unittest.TestCase):
    def test_default_decoder_ids_match_legacy_dump(self) -> None:
        # Bit-stable with the original tmp/granite-work/dump_decoder_golden.py
        # probe so regenerating fixtures stays comparable to existing goldens.
        self.assertEqual(
            intermediate.DEFAULT_DECODER_INPUT_IDS,
            [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200],
        )
        self.assertEqual(len(intermediate.DEFAULT_DECODER_INPUT_IDS), 12)
        self.assertTrue(
            all(0 < i < 100353 for i in intermediate.DEFAULT_DECODER_INPUT_IDS)
        )


class NpyRoundTripTest(unittest.TestCase):
    def test_save_f32_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.npy"
            src = torch.tensor([[[1.0, -2.5, 3.25]]], dtype=torch.float32)
            intermediate.save_f32(path, src)
            loaded = np.load(path)
            self.assertEqual(loaded.dtype, np.float32)
            np.testing.assert_array_equal(loaded, src.numpy())

    def test_save_i64_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ids.npy"
            src = torch.tensor([intermediate.DEFAULT_DECODER_INPUT_IDS], dtype=torch.long)
            intermediate.save_i64(path, src)
            loaded = np.load(path)
            self.assertEqual(loaded.dtype, np.int64)
            np.testing.assert_array_equal(loaded, src.numpy())


class ArgParserSmokeTest(unittest.TestCase):
    def test_golden_parser_requires_weights_and_samples(self) -> None:
        parser = golden.build_arg_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args([])

    def test_golden_parser_accepts_minimal(self) -> None:
        parser = golden.build_arg_parser()
        args = parser.parse_args(
            [
                "--weights-dir",
                "/tmp/weights",
                "--samples-dir",
                "/tmp/wavs",
                "--sample",
                "jfk=jfk.wav",
                "--out-dir",
                "/tmp/out",
            ]
        )
        self.assertEqual(args.samples, [("jfk", "jfk.wav")])
        self.assertEqual(args.question, golden.DEFAULT_QUESTION)
        self.assertEqual(args.dtype, torch.bfloat16)

    def test_intermediate_parser_accepts_minimal(self) -> None:
        parser = intermediate.build_arg_parser()
        args = parser.parse_args(
            [
                "--weights-dir",
                "/tmp/weights",
                "--wav",
                "/tmp/en_short.wav",
                "--out-dir",
                "/tmp/out",
                "--sample-name",
                "en_short",
            ]
        )
        self.assertEqual(args.sample_name, "en_short")
        self.assertFalse(args.skip_decoder)
        self.assertFalse(args.skip_audio_prefill)


if __name__ == "__main__":
    unittest.main()
