#!/usr/bin/env python3
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from _require_files import (
    DEFAULT_MIN_BYTES,
    STATS_MIN_BYTES,
    WEIGHT_MIN_BYTES,
    check_required_files,
    looks_like_error_page,
    min_bytes_for,
)

# The real 2026-07 incident: a tokenless fetch of a private HF repo landed a
# 29-byte captured error body at the filename the pipeline expected, and a
# plain glob-existence check recorded it as "ok" during the original
# qwen3-forced-aligner-0.6b import. Reconstruct that exact size here as a JSON
# error envelope, the shape hf-mirror/HF actually return.
TWENTY_NINE_BYTE_ERROR_PAGE = b'{"error":"no such repo!!"}'.ljust(29, b" ")
assert len(TWENTY_NINE_BYTE_ERROR_PAGE) == 29


class MinBytesForTests(unittest.TestCase):
    def test_weight_extensions_get_the_megabyte_floor(self) -> None:
        for name in (
            "model.safetensors",
            "model-00001-of-00002.safetensors",
            "small.cn.pt",
            "b6-vb2+vox2+cnc2_v0-lm.pt",
            "model.onnx",
            "Hy-MT2-1.8B-Q4_K_M.gguf",
        ):
            self.assertEqual(min_bytes_for(name), WEIGHT_MIN_BYTES, name)

    def test_compound_pth_tar_suffix_gets_the_megabyte_floor(self) -> None:
        self.assertEqual(min_bytes_for("model.pth.tar"), WEIGHT_MIN_BYTES)
        self.assertEqual(min_bytes_for("asr_encoder.pth.tar"), WEIGHT_MIN_BYTES)

    def test_stats_extensions_get_the_stats_floor(self) -> None:
        for name in ("feats_stats.npz", "cmvn.ark", "am.mvn"):
            self.assertEqual(min_bytes_for(name), STATS_MIN_BYTES, name)

    def test_config_and_text_files_get_the_low_default_floor(self) -> None:
        for name in (
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "model.safetensors.index.json",
            "units.txt",
            "dict.txt",
            "bpe.model",
            "config.yaml",
            "LICENSE.txt",
            "chinese-lert-base/vocab.txt",
        ):
            self.assertEqual(min_bytes_for(name), DEFAULT_MIN_BYTES, name)


class LooksLikeErrorPageTests(unittest.TestCase):
    def test_detects_html_error_page(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_bytes(b"<!DOCTYPE html><html><body>404</body></html>")
            self.assertIsNotNone(looks_like_error_page(str(path)))

    def test_detects_json_error_envelope(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_bytes(b'{"error":"Repository Not Found"}')
            self.assertIsNotNone(looks_like_error_page(str(path)))

    def test_detects_auth_rejection_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.safetensors"
            path.write_bytes(b"Invalid username or password.")
            self.assertIsNotNone(looks_like_error_page(str(path)))

    def test_real_looking_json_config_is_not_flagged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_bytes(
                b'{"architectures": ["Wav2Vec2ForCTC"], "model_type": "wav2vec2", '
                b'"vocab_size": 32, "hidden_size": 768}'
            )
            self.assertIsNone(looks_like_error_page(str(path)))


class CheckRequiredFilesTests(unittest.TestCase):
    def test_passes_when_every_pattern_matches_a_real_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_bytes(
                b'{"architectures": ["Wav2Vec2ForCTC"], "model_type": "wav2vec2"}'
            )
            (root / "model.safetensors").write_bytes(b"\x00" * (WEIGHT_MIN_BYTES + 1))
            problems = check_required_files(str(root), ["config.json", "model.safetensors"])
            self.assertEqual(problems, [])

    def test_legitimately_tiny_text_file_is_not_rejected(self) -> None:
        # False-positive guard sized against this repo's own staged inputs:
        # a 20-byte hf_repo.txt is a real, complete file. An earlier 32-byte
        # default floor rejected it -- that floor had been reverse-engineered
        # from the incident's 29 bytes rather than from what a legitimate
        # small file looks like.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tiny = b"OpenASR/whisper-tiny"
            self.assertLess(len(tiny), 64)
            (root / "hf_repo.txt").write_bytes(tiny)
            problems = check_required_files(str(root), ["hf_repo.txt"])
            self.assertEqual(problems, [])

    def test_missing_pattern_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            problems = check_required_files(tmp, ["model.safetensors"])
            self.assertEqual(len(problems), 1)
            self.assertIn("missing", problems[0])
            self.assertIn("model.safetensors", problems[0])

    def test_29_byte_captured_error_page_on_a_weight_pattern_is_rejected(self) -> None:
        # The core regression case: a tokenless fetch of a private repo lands
        # a 29-byte error page at the filename a weight glob expects. The
        # size floor alone (33 bytes < 1 MiB) must fail this closed.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "model.safetensors").write_bytes(TWENTY_NINE_BYTE_ERROR_PAGE)
            problems = check_required_files(str(root), ["model.safetensors"])
            self.assertEqual(len(problems), 1)
            self.assertIn("too small", problems[0])
            self.assertIn("model.safetensors", problems[0])

    def test_29_byte_captured_error_page_on_a_config_pattern_is_rejected(self) -> None:
        # Same incident shape, but landing on a text/JSON category, where a
        # size floor cannot help: this repo stages legitimate 21- and 29-byte
        # hf_repo.txt files, so 29 bytes is not by itself a defect. The
        # content sniff is what has to catch it here.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_bytes(TWENTY_NINE_BYTE_ERROR_PAGE)
            problems = check_required_files(str(root), ["config.json"])
            self.assertEqual(len(problems), 1)
            self.assertIn("looks like an error page", problems[0])

    def test_oversized_error_page_on_a_config_pattern_is_caught_by_content_sniff(self) -> None:
        # A larger captured error body clears the low default size floor but
        # must still be caught by the content sniff -- this is the case the
        # size floor alone cannot handle.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            body = b"<!DOCTYPE html><html><head><title>401</title></head>" \
                   b"<body>Invalid username or password</body></html>"
            self.assertGreater(len(body), DEFAULT_MIN_BYTES)
            (root / "config.json").write_bytes(body)
            problems = check_required_files(str(root), ["config.json"])
            self.assertEqual(len(problems), 1)
            self.assertIn("looks like an error page", problems[0])

    def test_one_bad_shard_among_several_fails_the_whole_pattern(self) -> None:
        # A sharded weight glob (model-*.safetensors) must reject the whole
        # requirement if even one matched shard is corrupt/replaced, not just
        # skip the bad one because a sibling shard is fine.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "model-00001-of-00002.safetensors").write_bytes(b"\x00" * (WEIGHT_MIN_BYTES + 1))
            (root / "model-00002-of-00002.safetensors").write_bytes(TWENTY_NINE_BYTE_ERROR_PAGE)
            problems = check_required_files(str(root), ["model-*.safetensors"])
            self.assertEqual(len(problems), 1)
            self.assertIn("00002-of-00002", problems[0])

    def test_directory_match_is_not_treated_as_a_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "Qwen2-7B-Instruct").mkdir()
            problems = check_required_files(str(root), ["Qwen2-7B-Instruct"])
            self.assertEqual(problems, [])


if __name__ == "__main__":
    unittest.main()
