from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "release-binaries.yml"


class WindowsBackendReleaseContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.workflow = WORKFLOW.read_text(encoding="utf-8")

    def assert_matrix_leg(
        self,
        target: str,
        *,
        asset: str,
        features: str,
        provider: str,
        distribution: str,
    ) -> None:
        leg = re.search(
            rf"(?ms)^\s*- os: windows[^\n]*\n"
            rf"\s*target: {re.escape(target)}\n"
            rf"(?P<body>.*?)(?=^\s*- os:|^\s*steps:)",
            self.workflow,
        )
        self.assertIsNotNone(leg, f"missing Windows release leg {target}")
        body = leg.group("body")
        for key, value in (
            ("asset", asset),
            ("features", features),
            ("provider", provider),
            ("distribution", distribution),
        ):
            self.assertRegex(body, rf"(?m)^\s*{key}: {re.escape(value)}\s*$")
        self.assertNotRegex(body, r"(?m)^\s*experimental:\s*true\s*$")

    def test_terminal_host_and_optional_provider_legs_are_release_blocking(self) -> None:
        self.assert_matrix_leg(
            "x86_64-pc-windows-msvc-neutral",
            asset="windows-x86_64-neutral",
            features="vulkan",
            provider="vulkan",
            distribution="host",
        )
        self.assert_matrix_leg(
            "x86_64-pc-windows-msvc-cuda-plugin",
            asset="windows-x86_64-cuda-plugin",
            features="cuda",
            provider="cuda",
            distribution="plugin",
        )
        self.assert_matrix_leg(
            "x86_64-pc-windows-msvc-hip-plugin",
            asset="windows-x86_64-rocm-plugin",
            features="hip",
            provider="hip",
            distribution="plugin",
        )

    def test_migration_sidecars_remain_explicit_legacy_builds(self) -> None:
        self.assert_matrix_leg(
            "x86_64-pc-windows-msvc-cuda",
            asset="windows-x86_64-cuda-sidecar",
            features="cuda,legacy-windows-static-sidecar",
            provider="cuda",
            distribution="legacy",
        )
        self.assert_matrix_leg(
            "x86_64-pc-windows-msvc-hip",
            asset="windows-x86_64-rocm-sidecar",
            features="hip,legacy-windows-static-sidecar",
            provider="hip",
            distribution="legacy",
        )

    def test_both_optional_plugins_feed_one_catalog_and_update_hint(self) -> None:
        required = (
            "--entry dist/backend-pack-cuda.json",
            "--entry dist/backend-pack-hip.json",
            "--out dist/catalog.backends.candidate.json",
            "--out dist/backend-plugin-hints.json",
            "windows-x86_64-cuda-plugin:dll",
            "windows-x86_64-rocm-plugin:dll",
            'echo "backend-pack-cuda.json"',
            'echo "backend-pack-hip.json"',
            'echo "backend-plugin-hints.json"',
        )
        for fragment in required:
            self.assertIn(fragment, self.workflow)

    def test_plugin_vendor_and_signing_steps_cover_cuda_and_hip(self) -> None:
        self.assertIn(
            "matrix.distribution == 'plugin'", self.workflow
        )
        self.assertIn('$provider -eq "cuda"', self.workflow)
        self.assertIn('$provider -eq "hip"', self.workflow)
        self.assertIn("VENDOR_LAYER_KEY=cuda-runtime", self.workflow)
        self.assertIn("VENDOR_LAYER_KEY=rocm-runtime", self.workflow)
        self.assertIn("Resolve Windows binaries to sign", self.workflow)
        self.assertIn("$env:PLUGIN_ASSET_PATH", self.workflow)
        self.assertIn("Attest backend catalog metadata", self.workflow)
        self.assertIn("Verify optional backend PE contract", self.workflow)
        for symbol in (
            "ggml_backend_init",
            "openasr_ggml_backend_abi_v1",
            "openasr_ggml_backend_probe_v1",
            "openasr_ggml_backend_provider_v1",
            "ggml-base\\.dll",
            "cudart64_",
            "cublas64_",
            "nvcuda.dll",
            "amdhip64",
            "libhipblas",
            "rocblas",
        ):
            self.assertIn(symbol, self.workflow)


if __name__ == "__main__":
    unittest.main()
