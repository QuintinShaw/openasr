import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import compare  # noqa: E402


class NormalizeTests(unittest.TestCase):
    def test_casefold_punctuation_whitespace(self):
        self.assertEqual(
            compare.normalize("And so,  my Fellow-Americans!"),
            "and so my fellow americans",
        )

    def test_cjk_untouched_by_casefold(self):
        self.assertEqual(compare.normalize("你好，世界。"), "你好 世界")


class TokenizeTests(unittest.TestCase):
    def test_cjk_chars_are_individual_tokens(self):
        self.assertEqual(compare.tokenize("今天weather好"), ["今", "天", "weather", "好"])

    def test_plain_english_words(self):
        self.assertEqual(compare.tokenize("Ask not!"), ["ask", "not"])


class WerTests(unittest.TestCase):
    def test_identical_is_zero(self):
        self.assertEqual(compare.wer("ask not", "ask not"), 0.0)

    def test_one_substitution(self):
        self.assertAlmostEqual(compare.wer("ask nod what", "ask not what"), 1 / 3)

    def test_empty_reference(self):
        self.assertEqual(compare.wer("something", ""), 1.0)
        self.assertEqual(compare.wer("", ""), 0.0)


class CliTests(unittest.TestCase):
    def run_compare(self, transcript: str, goldens: dict[str, str], strategy: str) -> int:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            transcript_path = root / "transcript.txt"
            transcript_path.write_text(transcript, encoding="utf-8")
            golden_dir = root / "goldens"
            golden_dir.mkdir()
            for name, text in goldens.items():
                (golden_dir / name).write_text(text, encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_DIR / "compare.py"),
                    "--transcript",
                    str(transcript_path),
                    "--golden-dir",
                    str(golden_dir),
                    "--strategy",
                    strategy,
                ],
                capture_output=True,
                text=True,
            )
            return result.returncode

    def test_exact_pass_and_fail(self):
        self.assertEqual(self.run_compare("a b c", {"golden.txt": "a b c\n"}, "exact"), 0)
        self.assertEqual(self.run_compare("a b d", {"golden.txt": "a b c"}, "exact"), 1)

    def test_any_variant_passes(self):
        goldens = {"golden.txt": "mac text", "golden.linux-x86_64.txt": "linux text"}
        self.assertEqual(self.run_compare("linux text", goldens, "exact"), 0)

    def test_normalized_pass(self):
        self.assertEqual(
            self.run_compare("And so, my fellow", {"golden.txt": "and so my Fellow!"}, "normalized"),
            0,
        )

    def test_wer_threshold(self):
        goldens = {"golden.txt": "ask not what your country"}
        self.assertEqual(self.run_compare("ask nod what your country", goldens, "wer:0.2"), 0)
        self.assertEqual(self.run_compare("completely different words here now", goldens, "wer:0.2"), 1)

    def test_missing_golden_dir_is_config_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            transcript_path = root / "transcript.txt"
            transcript_path.write_text("x", encoding="utf-8")
            empty = root / "empty"
            empty.mkdir()
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_DIR / "compare.py"),
                    "--transcript",
                    str(transcript_path),
                    "--golden-dir",
                    str(empty),
                    "--strategy",
                    "exact",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 2)


if __name__ == "__main__":
    unittest.main()
