from __future__ import annotations

import hashlib
import tempfile
import time
import unittest
import zipfile
from pathlib import Path

from deterministic_zip import (
    FIXED_DATE_TIME,
    DeterministicZipError,
    create_deterministic_zip,
)


class CreateDeterministicZipTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.src_dir = self.root / "src"
        self.src_dir.mkdir()

    def _write_tree(self, src_dir: Path) -> None:
        (src_dir / "amdhip64_6.dll").write_bytes(b"fake-amdhip64-bytes")
        (src_dir / "rocblas.dll").write_bytes(b"fake-rocblas-bytes")
        (src_dir / "rocblas" / "library").mkdir(parents=True)
        (src_dir / "rocblas" / "library" / "TensileLibrary.dat").write_bytes(b"fake-tensile-bytes")

    def test_same_content_produces_byte_identical_zips_regardless_of_write_order_or_mtime(self) -> None:
        src_a = self.root / "src-a"
        src_a.mkdir()
        self._write_tree(src_a)

        # Same content, different physical write order/mtimes -- a second
        # source dir built by writing the SAME three files in reverse order,
        # with an artificially different mtime on one file, must still
        # produce a byte-identical zip: that is the entire point of
        # deterministic packaging (content-addressed dedup on the sha256 of
        # the OUTPUT zip, not of the source tree).
        src_b = self.root / "src-b"
        (src_b / "rocblas" / "library").mkdir(parents=True)
        (src_b / "rocblas" / "library" / "TensileLibrary.dat").write_bytes(b"fake-tensile-bytes")
        (src_b / "rocblas.dll").write_bytes(b"fake-rocblas-bytes")
        (src_b / "amdhip64_6.dll").write_bytes(b"fake-amdhip64-bytes")
        old_time = time.time() - 86400
        import os

        os.utime(src_b / "amdhip64_6.dll", (old_time, old_time))

        out_a = self.root / "a.zip"
        out_b = self.root / "b.zip"
        create_deterministic_zip(out_a, src_a)
        create_deterministic_zip(out_b, src_b)

        self.assertEqual(
            hashlib.sha256(out_a.read_bytes()).hexdigest(),
            hashlib.sha256(out_b.read_bytes()).hexdigest(),
        )

    def test_rebuilding_from_the_same_source_dir_is_byte_identical(self) -> None:
        self._write_tree(self.src_dir)
        out_1 = self.root / "1.zip"
        out_2 = self.root / "2.zip"
        create_deterministic_zip(out_1, self.src_dir)
        create_deterministic_zip(out_2, self.src_dir)
        self.assertEqual(out_1.read_bytes(), out_2.read_bytes())

    def test_entries_are_sorted_by_posix_relative_path(self) -> None:
        self._write_tree(self.src_dir)
        out = self.root / "out.zip"
        create_deterministic_zip(out, self.src_dir)
        with zipfile.ZipFile(out) as archive:
            names = archive.namelist()
        self.assertEqual(names, sorted(names))
        self.assertIn("rocblas/library/TensileLibrary.dat", names)

    def test_entries_use_a_fixed_date_time_not_the_real_mtime(self) -> None:
        self._write_tree(self.src_dir)
        out = self.root / "out.zip"
        create_deterministic_zip(out, self.src_dir)
        with zipfile.ZipFile(out) as archive:
            for info in archive.infolist():
                self.assertEqual(info.date_time, FIXED_DATE_TIME)

    def test_all_file_bytes_are_preserved(self) -> None:
        self._write_tree(self.src_dir)
        out = self.root / "out.zip"
        create_deterministic_zip(out, self.src_dir)
        with zipfile.ZipFile(out) as archive:
            self.assertEqual(archive.read("amdhip64_6.dll"), b"fake-amdhip64-bytes")
            self.assertEqual(archive.read("rocblas.dll"), b"fake-rocblas-bytes")
            self.assertEqual(
                archive.read("rocblas/library/TensileLibrary.dat"), b"fake-tensile-bytes"
            )

    def test_missing_source_dir_fails_loudly(self) -> None:
        with self.assertRaises(DeterministicZipError):
            create_deterministic_zip(self.root / "out.zip", self.root / "does-not-exist")

    def test_empty_source_dir_fails_loudly(self) -> None:
        with self.assertRaises(DeterministicZipError):
            create_deterministic_zip(self.root / "out.zip", self.src_dir)


if __name__ == "__main__":
    unittest.main()
