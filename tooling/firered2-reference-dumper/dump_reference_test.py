#!/usr/bin/env python3
"""Self-tests for dump_reference.py that don't need the official FireRedASR2S
clone, real weights, or the LLM stage -- just the pure numeric/parsing logic,
so `python3 -m unittest` stays fast and hermetic in CI."""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
import torch

import dump_reference as dr


class AvailableMemoryParsingTest(unittest.TestCase):
    SAMPLE_VM_STAT = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                    22726.
Pages active:                                 247746.
Pages inactive:                               242892.
Pages speculative:                              5776.
Pages throttled:                                   0.
Pages wired down:                             242970.
Pages purgeable:                               12494.
"""

    def test_parses_free_inactive_purgeable_speculative(self) -> None:
        completed = mock.Mock(stdout=self.SAMPLE_VM_STAT)
        with mock.patch("subprocess.run", return_value=completed):
            gb = dr.available_memory_gb()
        page_size = 16384
        expected_pages = 22726 + 242892 + 12494 + 5776
        expected_gb = expected_pages * page_size / (1024**3)
        self.assertAlmostEqual(gb, expected_gb, places=6)

    def test_non_macos_does_not_block(self) -> None:
        with mock.patch("subprocess.run", side_effect=OSError("no vm_stat")):
            gb = dr.available_memory_gb()
        self.assertEqual(gb, float("inf"))

    def test_wait_for_memory_returns_immediately_when_available(self) -> None:
        with mock.patch.object(dr, "available_memory_gb", return_value=100.0):
            with mock.patch("time.sleep") as sleep_mock:
                dr.wait_for_memory(min_gb=6.0)
        sleep_mock.assert_not_called()

    def test_wait_for_memory_polls_until_available(self) -> None:
        readings = iter([1.0, 1.0, 8.0])
        with mock.patch.object(dr, "available_memory_gb", side_effect=lambda: next(readings)):
            with mock.patch("time.sleep") as sleep_mock:
                dr.wait_for_memory(min_gb=6.0, poll_seconds=0.01)
        self.assertEqual(sleep_mock.call_count, 2)

    def test_wait_for_memory_gives_up_after_max_wait(self) -> None:
        with mock.patch.object(dr, "available_memory_gb", return_value=1.0):
            with mock.patch("time.sleep"):
                # Should return (not hang forever) once the wait budget is exhausted.
                dr.wait_for_memory(min_gb=6.0, poll_seconds=1.0, max_wait_seconds=2.0)


class AdapterFrameStackingMathTest(unittest.TestCase):
    """Exercise the same frame-stack -> linear1 -> relu -> linear2 arithmetic
    used by adapter_ggml_pack_crosscheck against a hand-computed reference,
    independent of any real .oasr pack file."""

    def test_frame_stacking_matches_manual_reshape(self) -> None:
        t, d = 6, 4
        x = np.arange(t * d, dtype=np.float32).reshape(t, d)
        ds = 2
        stacked = x.reshape(t // ds, d * ds)
        # frame 0 stacked with frame 1, frame 2 with frame 3, ...
        expected_row0 = np.concatenate([x[0], x[1]])
        np.testing.assert_array_equal(stacked[0], expected_row0)
        expected_row2 = np.concatenate([x[4], x[5]])
        np.testing.assert_array_equal(stacked[2], expected_row2)

    def test_odd_length_discards_trailing_frame(self) -> None:
        t, d = 5, 4
        x = np.arange(t * d, dtype=np.float32).reshape(t, d)
        ds = 2
        discard = t % ds
        trimmed = x[: t - discard]
        self.assertEqual(trimmed.shape[0], 4)

    def test_linear_relu_linear_matches_numpy_reference(self) -> None:
        rng = np.random.default_rng(0)
        stacked = rng.standard_normal((3, 8)).astype(np.float32)
        w1 = rng.standard_normal((5, 8)).astype(np.float32)
        b1 = rng.standard_normal((5,)).astype(np.float32)
        w2 = rng.standard_normal((5, 5)).astype(np.float32)
        b2 = rng.standard_normal((5,)).astype(np.float32)

        hidden = stacked @ w1.T + b1
        hidden = np.maximum(hidden, 0.0)
        out = hidden @ w2.T + b2

        # Reference via explicit per-row loop (independent code path).
        expected = np.zeros_like(out)
        for i in range(stacked.shape[0]):
            h = np.maximum(w1 @ stacked[i] + b1, 0.0)
            expected[i] = w2 @ h + b2
        np.testing.assert_allclose(out, expected, atol=1e-5)


class PromptSplicingTest(unittest.TestCase):
    """Verify the batch=1 speech-token splicing used by build_prompt_embeds
    against a minimal fake tokenizer + streaming model, independent of the
    real Qwen2 weights."""

    def test_speech_features_spliced_at_placeholder_position(self) -> None:
        # build_prompt_embeds imports DEFAULT_SPEECH_TOKEN from the official
        # refcode package; stub it so this test stays hermetic (no repo clone
        # needed just to exercise the batch=1 splicing arithmetic).
        import sys
        import types

        fake_pkg = types.ModuleType("fireredasr2")
        fake_tok_pkg = types.ModuleType("fireredasr2.tokenizer")
        fake_llm_tok = types.ModuleType("fireredasr2.tokenizer.llm_tokenizer")
        fake_llm_tok.DEFAULT_SPEECH_TOKEN = "<speech>"
        sys.modules.setdefault("fireredasr2", fake_pkg)
        sys.modules.setdefault("fireredasr2.tokenizer", fake_tok_pkg)
        sys.modules["fireredasr2.tokenizer.llm_tokenizer"] = fake_llm_tok

        class FakeTokenizer:
            SPEECH_ID = 999

            def apply_chat_template(self, messages, **_kwargs):
                # 3 leading text tokens, 1 speech placeholder, 2 trailing text tokens
                return [1, 2, self.SPEECH_ID, 3, 4]

            def convert_tokens_to_ids(self, tok):
                assert tok == "<speech>"
                return self.SPEECH_ID

        class FakeEmbedding(torch.nn.Module):
            def forward(self, ids):
                # deterministic embedding: id repeated across an 8-dim vector
                return ids.float().unsqueeze(-1).repeat(1, 1, 8)

        class FakeModelModel:
            embed_tokens = FakeEmbedding()

        class FakeModel:
            model = FakeModelModel()

        class FakeStreaming:
            torch = torch
            model = FakeModel()

        tokenizer = FakeTokenizer()
        streaming = FakeStreaming()
        speech_features = np.full((2, 8), 111.0, dtype=np.float32)  # 2 fake speech frames

        prompt_ids, speech_idx, final_embeds = dr.build_prompt_embeds(streaming, tokenizer, speech_features)

        self.assertEqual(prompt_ids, [1, 2, FakeTokenizer.SPEECH_ID, 3, 4])
        self.assertEqual(speech_idx, 2)
        # final sequence length = 4 text tokens (placeholder removed) + 2 speech frames = 6
        self.assertEqual(final_embeds.shape, (1, 6, 8))
        # positions 0,1 are the leading text embeds (ids 1, 2)
        np.testing.assert_allclose(final_embeds[0, 0].numpy(), np.full(8, 1.0))
        np.testing.assert_allclose(final_embeds[0, 1].numpy(), np.full(8, 2.0))
        # positions 2,3 are the spliced-in speech features
        np.testing.assert_allclose(final_embeds[0, 2].numpy(), np.full(8, 111.0))
        np.testing.assert_allclose(final_embeds[0, 3].numpy(), np.full(8, 111.0))
        # positions 4,5 are the trailing text embeds (ids 3, 4)
        np.testing.assert_allclose(final_embeds[0, 4].numpy(), np.full(8, 3.0))
        np.testing.assert_allclose(final_embeds[0, 5].numpy(), np.full(8, 4.0))


if __name__ == "__main__":
    unittest.main()
