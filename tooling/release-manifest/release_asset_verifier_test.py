from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import release_asset_verifier


class ReleaseAssetVerifierTest(unittest.TestCase):
    def write_fixture(self, checksums_text: str) -> tuple[Path, Path, tempfile.TemporaryDirectory[str]]:
        temp = tempfile.TemporaryDirectory()
        root = Path(temp.name)
        asset = root / "openasr-0.1.35-linux-x86_64.tar.gz"
        asset.write_bytes(b"verified release bytes")
        checksums = root / "SHA256SUMS"
        checksums.write_text(checksums_text, encoding="utf-8")
        return asset, checksums, temp

    def test_accepts_exactly_one_matching_digest(self) -> None:
        digest = hashlib.sha256(b"verified release bytes").hexdigest()
        asset, checksums, temp = self.write_fixture(f"{digest}  openasr-0.1.35-linux-x86_64.tar.gz\n")
        self.addCleanup(temp.cleanup)

        self.assertEqual(release_asset_verifier.verify_release_asset(asset, checksums), digest)

    def test_rejects_a_digest_mismatch(self) -> None:
        asset, checksums, temp = self.write_fixture(
            f"{'0' * 64}  openasr-0.1.35-linux-x86_64.tar.gz\n"
        )
        self.addCleanup(temp.cleanup)

        with self.assertRaisesRegex(
            release_asset_verifier.ReleaseAssetVerificationError, "sha256 mismatch"
        ):
            release_asset_verifier.verify_release_asset(asset, checksums)

    def test_rejects_a_missing_manifest_entry(self) -> None:
        asset, checksums, temp = self.write_fixture(f"{'0' * 64}  another.tar.gz\n")
        self.addCleanup(temp.cleanup)

        with self.assertRaisesRegex(
            release_asset_verifier.ReleaseAssetVerificationError, "exactly one entry"
        ):
            release_asset_verifier.verify_release_asset(asset, checksums)

    def test_rejects_duplicate_manifest_entries(self) -> None:
        digest = hashlib.sha256(b"verified release bytes").hexdigest()
        line = f"{digest}  openasr-0.1.35-linux-x86_64.tar.gz\n"
        asset, checksums, temp = self.write_fixture(line + line)
        self.addCleanup(temp.cleanup)

        with self.assertRaisesRegex(
            release_asset_verifier.ReleaseAssetVerificationError, "found 2"
        ):
            release_asset_verifier.verify_release_asset(asset, checksums)


if __name__ == "__main__":
    unittest.main()
