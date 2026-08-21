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
        sync = (ROOT / "scripts/sync-windows-backend-cdn.sh").read_text(encoding="utf-8")
        deploy = (ROOT / ".github/workflows/deploy-catalog.yml").read_text(encoding="utf-8")
        publish = (
            ROOT / "tooling/publish-model/scripts/publish_catalog.sh"
        ).read_text(encoding="utf-8")

        self.assertIn("gh release create", release)
        self.assertIn("--draft", release)
        self.assertIn("verify-assets", prepare)
        self.assertIn("publish_catalog.sh", prepare)
        self.assertIn("verify-catalog", prepare)
        self.assertIn("verify-cdn", prepare)
        self.assertIn("backend_hardware_evidence.py", prepare)
        self.assertIn("tr -d '\\r'", prepare)
        self.assertNotIn("mapfile -t", prepare)
        self.assertNotIn("mapfile -t", finalize)
        self.assertIn('source.read_text(encoding="utf-8")', publish)
        self.assertIn("path.write_bytes", prepare)
        self.assertIn("target.write_bytes", publish)
        self.assertLess(
            prepare.index("preflighting local catalog signer toolchain"),
            prepare.index("downloading backend entries"),
        )
        self.assertLess(prepare.index("verify-cdn"), prepare.index("publish_catalog.sh"))
        self.assertLess(
            prepare.index("verify-cdn"),
            prepare.index('old_epoch="$(tr -d'),
        )
        self.assertIn("prepare-windows-backend-catalog-release.sh", sync)
        self.assertIn("verify-cdn", deploy)
        self.assertIn("check_catalog_consistency.py", deploy)
        self.assertIn("regenerate_all.sh --check", deploy)
        self.assertLess(deploy.index("verify-cdn"), deploy.index("Deploy to Cloudflare"))
        self.assertIn("catalog.openasr.org/v1/catalog.json", finalize)
        self.assertNotIn("backends-manifest", finalize)
        self.assertIn("verify-catalog", finalize)
        self.assertIn("verify-cdn", finalize)
        self.assertIn("backend_hardware_evidence.py", finalize)
        self.assertIn("tr -d '\\r'", finalize)
        self.assertIn('gh release edit "$tag" --draft=false --latest', finalize)
        self.assertLess(finalize.index("verify-cdn"), finalize.index("--draft=false"))

    def test_finalizer_never_publishes_before_all_cuda_and_hip_target_entries(self) -> None:
        finalize = (ROOT / "scripts/finalize-core-release.sh").read_text(encoding="utf-8")
        self.assertIn("backend-pack-*.json", finalize)
        self.assertIn('"${#cuda_entries[@]}" -ne 6', finalize)
        self.assertIn('"${#hip_entries[@]}" -ne 14', finalize)
        self.assertLess(finalize.index("verify-catalog"), finalize.index("--draft=false"))

    def test_release_matrix_has_one_formal_entrypoint_and_channels_wait_for_publish(self) -> None:
        release = (ROOT / ".github/workflows/release-core.yml").read_text(encoding="utf-8")
        binaries = (ROOT / ".github/workflows/release-binaries.yml").read_text(encoding="utf-8")
        channels = (ROOT / ".github/workflows/publish-core-channels.yml").read_text(
            encoding="utf-8"
        )

        self.assertNotIn("push:\n    tags:", binaries)
        self.assertIn("uses: ./.github/workflows/release-binaries.yml", release)
        self.assertNotIn("docker-images:", release)
        self.assertNotIn("update-homebrew-tap:", release)
        self.assertIn("types: [published]", channels)
        self.assertIn("uses: ./.github/workflows/docker-release.yml", channels)
        self.assertIn("releases/latest", channels)
        self.assertIn("distribution-gate:", channels)
        self.assertIn("backend_hardware_evidence.py", channels)
        self.assertIn("gate-catalog-against-released-binary.sh", channels)
        self.assertIn("verify-catalog", channels)
        self.assertIn("verify-cdn", channels)
        self.assertIn("needs: [resolve, distribution-gate]", channels)
        self.assertIn("git push origin main", channels)

    def test_family_regression_reuses_published_assets_instead_of_racing_raw_tag(self) -> None:
        family = (ROOT / ".github/workflows/family-regression.yml").read_text(
            encoding="utf-8"
        )

        self.assertIn("types: [published]", family)
        self.assertNotIn('tags: ["v*"]', family)
        self.assertIn("releases/latest", family)
        self.assertIn("refusing a duplicate local build", family)


if __name__ == "__main__":
    unittest.main()
