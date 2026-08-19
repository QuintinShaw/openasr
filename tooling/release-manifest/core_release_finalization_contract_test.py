from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class CoreReleaseFinalizationContractTests(unittest.TestCase):
    def test_core_release_stays_draft_until_signed_backend_catalog_is_live(self) -> None:
        release = (ROOT / ".github/workflows/release-core.yml").read_text(encoding="utf-8")
        prepare = (ROOT / "scripts/prepare-windows-backend-catalog-release.sh").read_text(
            encoding="utf-8"
        )
        finalize = (ROOT / "scripts/finalize-core-release.sh").read_text(encoding="utf-8")
        publish = (
            ROOT / "tooling/publish-model/scripts/publish_catalog.sh"
        ).read_text(encoding="utf-8")

        self.assertIn("gh release create", release)
        self.assertIn("--draft", release)
        self.assertIn("verify-assets", prepare)
        self.assertIn("publish_catalog.sh", prepare)
        self.assertIn("verify-catalog", prepare)
        self.assertIn("backend_hardware_evidence.py", prepare)
        self.assertIn("tr -d '\\r'", prepare)
        self.assertIn('source.read_text(encoding="utf-8")', publish)
        self.assertIn("path.write_bytes", prepare)
        self.assertIn("target.write_bytes", publish)
        self.assertLess(
            prepare.index("preflighting local catalog signer toolchain"),
            prepare.index("downloading backend entries"),
        )
        self.assertIn("catalog.openasr.org/v1/catalog.json", finalize)
        self.assertNotIn("backends-manifest", finalize)
        self.assertIn("verify-catalog", finalize)
        self.assertIn("backend_hardware_evidence.py", finalize)
        self.assertIn("tr -d '\\r'", finalize)
        self.assertIn('gh release edit "$tag" --draft=false --latest', finalize)

    def test_finalizer_never_publishes_before_all_cuda_and_hip_target_entries(self) -> None:
        finalize = (ROOT / "scripts/finalize-core-release.sh").read_text(encoding="utf-8")
        self.assertIn("backend-pack-*.json", finalize)
        self.assertIn('"${#cuda_entries[@]}" -ne 6', finalize)
        self.assertIn('"${#hip_entries[@]}" -ne 14', finalize)
        self.assertLess(finalize.index("verify-catalog"), finalize.index("--draft=false"))


if __name__ == "__main__":
    unittest.main()
