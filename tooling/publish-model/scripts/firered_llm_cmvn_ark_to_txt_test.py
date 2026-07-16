#!/usr/bin/env python3
from __future__ import annotations

import struct
import unittest
from pathlib import Path

from firered_llm_cmvn_ark_to_txt import (
    CmvnArkFormatError,
    format_kaldi_text_matrix,
    parse_cmvn_ark,
)

TESTDATA_DIR = Path(__file__).parent / "testdata"
REAL_CMVN_ARK = TESTDATA_DIR / "firered2_llm_cmvn.ark"

# Recorded during stage-1 reconnaissance (scratchpad/fr2/T1-findings.md) by
# hand-parsing the real 1311-byte cmvn.ark with an independent script; used
# here as the oracle for this module's parser.
EXPECTED_MEAN_HEAD = [10.499, 10.949, 11.889, 12.635, 13.397]
EXPECTED_ISTD_HEAD = [0.252, 0.237, 0.232, 0.233, 0.232]
EXPECTED_FRAME_COUNT = 1_183_022_220


def build_synthetic_ark(rows: list[list[float]]) -> bytes:
    """Hand-construct a minimal Kaldi binary double-matrix payload matching
    the byte layout `firered_llm_cmvn_ark_to_txt.parse_cmvn_ark` expects, so
    the round-trip test does not depend on the real fixture file."""
    n_rows = len(rows)
    n_cols = len(rows[0])
    out = bytearray()
    out += b"\x00B"
    out += b"DM "
    out += bytes([0x04]) + struct.pack("<i", n_rows)
    out += bytes([0x04]) + struct.pack("<i", n_cols)
    for row in rows:
        assert len(row) == n_cols
        out += struct.pack(f"<{n_cols}d", *row)
    return bytes(out)


class ParseCmvnArkTests(unittest.TestCase):
    def test_parses_synthetic_hand_built_matrix(self) -> None:
        # dim=2, count=4: sums [8, 4], sumsq [32, 8] (same numbers as the
        # firered_aed Rust test fixture, so the two parsers are cross-checked
        # against the identical hand-computed case).
        data = build_synthetic_ark([[8.0, 4.0, 4.0], [32.0, 8.0, 0.0]])
        rows = parse_cmvn_ark(data)
        self.assertEqual(rows, [[8.0, 4.0, 4.0], [32.0, 8.0, 0.0]])

    def test_rejects_bad_binary_marker(self) -> None:
        data = bytearray(build_synthetic_ark([[1.0, 2.0], [3.0, 4.0]]))
        data[0:2] = b"XX"
        with self.assertRaises(CmvnArkFormatError):
            parse_cmvn_ark(bytes(data))

    def test_rejects_truncated_payload(self) -> None:
        data = build_synthetic_ark([[1.0, 2.0], [3.0, 4.0]])
        with self.assertRaises(CmvnArkFormatError):
            parse_cmvn_ark(data[:-1])

    def test_rejects_non_double_matrix_token(self) -> None:
        data = bytearray(build_synthetic_ark([[1.0, 2.0], [3.0, 4.0]]))
        data[2:5] = b"FM "
        with self.assertRaises(CmvnArkFormatError):
            parse_cmvn_ark(bytes(data))


class FormatKaldiTextMatrixTests(unittest.TestCase):
    def test_round_trips_through_the_bracket_delimited_text_format(self) -> None:
        rows = [[8.0, 4.0, 4.0], [32.0, 8.0, 0.0]]
        text = format_kaldi_text_matrix(rows)
        self.assertTrue(text.strip().startswith("["))
        self.assertTrue(text.strip().endswith("]"))
        # Re-parse with the same whitespace/bracket convention the Rust
        # importer's parse_kaldi_cmvn_stats uses, to prove the emitted text
        # is actually consumable (not just superficially bracketed).
        body = text[text.index("[") + 1 : text.rindex("]")]
        parsed_rows = [
            [float(token) for token in line.split()]
            for line in body.splitlines()
            if line.strip()
        ]
        self.assertEqual(parsed_rows, rows)

    def test_preserves_float64_precision(self) -> None:
        # A value that is not exactly representable in fewer significant
        # digits must still round-trip exactly through %.17g.
        rows = [[1183022220.0, 0.1 + 0.2], [3.0, 4.0]]
        text = format_kaldi_text_matrix(rows)
        body = text[text.index("[") + 1 : text.rindex("]")]
        parsed_rows = [
            [float(token) for token in line.split()]
            for line in body.splitlines()
            if line.strip()
        ]
        self.assertEqual(parsed_rows, rows)


class RealCmvnArkFixtureTests(unittest.TestCase):
    """Parses the real (tiny, 1311-byte, non-sensitive statistics-only)
    `cmvn.ark` checked in under testdata/ and asserts the derived mean/istd
    head values and frame count against the independently hand-verified
    numbers recorded in T1-findings.md."""

    def setUp(self) -> None:
        if not REAL_CMVN_ARK.is_file():
            self.skipTest(f"real cmvn.ark fixture missing: {REAL_CMVN_ARK}")
        self.rows = parse_cmvn_ark(REAL_CMVN_ARK.read_bytes())

    def test_matrix_shape(self) -> None:
        self.assertEqual(len(self.rows), 2)
        self.assertEqual(len(self.rows[0]), 81)  # feature_dim(80) + count column
        self.assertEqual(len(self.rows[1]), 81)

    def test_frame_count_matches_recorded_value(self) -> None:
        dim = len(self.rows[0]) - 1
        count = self.rows[0][dim]
        self.assertAlmostEqual(count, EXPECTED_FRAME_COUNT, delta=1.0)
        # Row 1's trailing column is unused by the mean/var formula (only
        # `sums[dim]` from row 0 is read as the frame count -- see
        # `parse_kaldi_cmvn_stats` in the Rust importer, which never reads
        # `sum_squares[dim]`); it is 0.0 in the real accumulator, not a
        # redundant copy of the count.
        self.assertEqual(self.rows[1][dim], 0.0)

    def test_mean_and_inv_stddev_head_values_match_hand_verified_reference(self) -> None:
        dim = len(self.rows[0]) - 1
        count = self.rows[0][dim]
        sums, sum_squares = self.rows[0][:dim], self.rows[1][:dim]
        for i in range(5):
            mean = sums[i] / count
            variance = max(sum_squares[i] / count - mean * mean, 1e-20)
            istd = 1.0 / (variance**0.5)
            self.assertAlmostEqual(mean, EXPECTED_MEAN_HEAD[i], places=3)
            self.assertAlmostEqual(istd, EXPECTED_ISTD_HEAD[i], places=3)

    def test_serialized_text_survives_round_trip_through_rust_style_parser(self) -> None:
        text = format_kaldi_text_matrix(self.rows)
        body = text[text.index("[") + 1 : text.rindex("]")]
        parsed_rows = [
            [float(token) for token in line.split()]
            for line in body.splitlines()
            if line.strip()
        ]
        self.assertEqual(len(parsed_rows), 2)
        self.assertEqual(parsed_rows, self.rows)


if __name__ == "__main__":
    unittest.main()
