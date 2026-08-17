from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "release-binaries.yml"
MATRIX = ROOT / "tooling" / "release-manifest" / "release_binaries_matrix.json"
CORE_BUILD_RS = ROOT / "crates" / "openasr-core" / "build.rs"


class WindowsBackendReleaseContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.workflow = WORKFLOW.read_text(encoding="utf-8")
        cls.core_build_rs = CORE_BUILD_RS.read_text(encoding="utf-8")
        cls.matrix = json.loads(MATRIX.read_text(encoding="utf-8"))

    def test_backend_abi_is_independent_of_git_checkout_newlines(self) -> None:
        self.assertIn(
            "let bytes = normalize_abi_source_newlines(&bytes);",
            self.core_build_rs,
        )
        normalizer = self.core_build_rs.split(
            "fn normalize_abi_source_newlines", 1
        )[1].split("fn sha256_hex", 1)[0]
        self.assertIn("bytes[index] == b'\\r'", normalizer)
        self.assertIn("Some(&b'\\n')", normalizer)
        self.assertIn("normalized.push(b'\\n')", normalizer)

    def assert_matrix_leg(
        self,
        target: str,
        *,
        asset: str,
        features: str,
        provider: str,
        distribution: str,
    ) -> None:
        matches = [row for row in self.matrix if row.get("target") == target]
        self.assertEqual(len(matches), 1, f"missing Windows release leg {target}")
        body = matches[0]
        self.assertTrue(str(body.get("os", "")).startswith("windows"), target)
        self.assertEqual(body.get("asset"), asset)
        self.assertEqual(body.get("features"), features)
        self.assertEqual(body.get("provider"), provider)
        self.assertEqual(body.get("distribution"), distribution)
        self.assertIsNot(body.get("experimental"), True)

    def test_terminal_host_and_target_scoped_provider_legs_are_release_blocking(self) -> None:
        self.assert_matrix_leg(
            "x86_64-pc-windows-msvc-neutral",
            asset="windows-x86_64-neutral",
            features="vulkan",
            provider="vulkan",
            distribution="host",
        )
        for sm in ("75", "80", "86", "89", "90"):
            self.assert_matrix_leg(
                f"x86_64-pc-windows-msvc-cuda-sm_{sm}-plugin",
                asset=f"windows-x86_64-cuda-sm_{sm}-plugin",
                features="cuda",
                provider="cuda",
                distribution="plugin",
            )
        for gfx in (
            "gfx1030", "gfx1031", "gfx1032", "gfx1035", "gfx1100", "gfx1101",
            "gfx1102", "gfx1103", "gfx1150", "gfx1151", "gfx1152", "gfx1153",
            "gfx1200", "gfx1201",
        ):
            self.assert_matrix_leg(
                f"x86_64-pc-windows-msvc-hip-{gfx}-plugin",
                asset=f"windows-x86_64-rocm-{gfx}-plugin",
                features="hip",
                provider="hip",
                distribution="plugin",
            )

    def test_terminal_release_has_no_legacy_windows_sidecar_rail(self) -> None:
        for obsolete in (
            "x86_64-pc-windows-msvc-vulkan",
            "x86_64-pc-windows-msvc-cuda\n",
            "x86_64-pc-windows-msvc-hip\n",
            "windows-x86_64-vulkan",
            "windows-x86_64-cuda-sidecar",
            "windows-x86_64-rocm-sidecar",
            "legacy-windows-static-sidecar",
            "distribution: legacy",
            "Generate backends-manifest.json",
            "verify-backends-manifest-signature",
        ):
            self.assertNotIn(obsolete, self.workflow)

    def test_target_scoped_optional_plugins_feed_one_catalog_and_update_hint(self) -> None:
        required = (
            "backend-pack-cuda-sm_*.json",
            "backend-pack-hip-gfx*.json",
            "--require-single-target",
            "--out dist/catalog.backends.candidate.json",
            "--out dist/backend-plugin-hints.json",
            'echo "backend-pack-cuda-sm_${sm}.json"',
            'echo "backend-pack-hip-${gfx}.json"',
            'echo "backend-plugin-hints.json"',
            'echo "catalog.backends.candidate.json"',
            "staging/catalog.backends.candidate.json",
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
        self.assertIn("VENDOR_OWNER", self.workflow)
        self.assertIn("env.VENDOR_OWNER == 'true'", self.workflow)
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

    def test_windows_cuda_release_remains_compatible_with_cuda_12_drivers(self) -> None:
        sm86 = next(
            row
            for row in self.matrix
            if row.get("target") == "x86_64-pc-windows-msvc-cuda-sm_86-plugin"
        )
        self.assertEqual(sm86.get("os"), "windows-2022")
        self.assertIn("matrix.cuda_toolkit || '12.6.3'", self.workflow)
        self.assertIn('min_driver_api="12.0.0"', self.workflow)
        self.assertNotIn('min_driver_api="13.0.0"', self.workflow)
        sm120 = next(
            row
            for row in self.matrix
            if row.get("target") == "x86_64-pc-windows-msvc-cuda-sm_120-plugin"
        )
        self.assertEqual(sm120.get("cuda_toolkit"), "12.8.1")
        self.assertEqual(sm120.get("min_driver_api"), "12.8.0")
        self.assertTrue(sm120.get("experimental"))

    def test_dynamic_matrix_is_selected_before_build_jobs_instantiate(self) -> None:
        self.assertIn("\n  select-matrix:\n", self.workflow)
        self.assertIn("needs: [select-matrix]", self.workflow)
        self.assertIn(
            "include: ${{ fromJSON(needs.select-matrix.outputs.include) }}",
            self.workflow,
        )
        self.assertIn("select_release_matrix.py", self.workflow)
        self.assertNotIn("LEG_SELECTED", self.workflow)
        self.assertNotIn("uses: Jimver/cuda-toolkit", self.workflow)
        self.assertNotIn("uses: ggml-org/free-disk-space", self.workflow)
        self.assertNotIn("uses: azure/trusted-signing-action", self.workflow)
        self.assertNotIn("uses: Swatinem/rust-cache@v2", self.workflow)
        self.assertIn("uses: ./.github/actions/install-cuda-toolkit-windows", self.workflow)
        self.assertIn("uses: ./.github/actions/free-disk-space", self.workflow)
        self.assertNotIn("uses: actions/attest-build-provenance@v4", self.workflow)
        self.assertIn("uses: ./.github/actions/attest-build-provenance", self.workflow)
        self.assertIn("uses: ./.github/actions/rust-cache", self.workflow)

    def test_full_matrix_has_exactly_one_vendor_owner_per_optional_vendor(self) -> None:
        cuda_owners = [
            row["target"]
            for row in self.matrix
            if row.get("provider") == "cuda" and row.get("vendor_owner") is True
        ]
        hip_owners = [
            row["target"]
            for row in self.matrix
            if row.get("provider") == "hip" and row.get("vendor_owner") is True
        ]
        self.assertEqual(cuda_owners, ["x86_64-pc-windows-msvc-cuda-sm_75-plugin"])
        self.assertEqual(hip_owners, ["x86_64-pc-windows-msvc-hip-gfx1030-plugin"])

    def test_diagnostic_only_target_temporarily_owns_vendor_assets(self) -> None:
        self.assertIn(
            "VENDOR_OWNER: ${{ matrix.distribution == 'plugin' && "
            "((github.event_name == 'workflow_dispatch' && inputs.only_target != '' "
            "&& matrix.target == inputs.only_target) || matrix.vendor_owner) }}",
            self.workflow,
        )

    def test_hip_pe_gate_requires_only_direct_runtime_imports(self) -> None:
        self.assertIn(
            "foreach ($requiredImport in @('amdhip64', 'libhipblas'))",
            self.workflow,
        )
        self.assertNotIn(
            "foreach ($requiredImport in @('amdhip64', 'libhipblas', 'rocblas'))",
            self.workflow,
        )
        self.assertIn("rocblas\\library", self.workflow)

    def test_neutral_hosts_and_optional_plugins_install_the_vulkan_sdk(self) -> None:
        self.assertIn(
            "NEEDS_WINDOWS_VULKAN_SDK: ${{ matrix.target == "
            "'x86_64-pc-windows-msvc' || contains(matrix.features, 'vulkan') || "
            "matrix.distribution == 'host' || matrix.distribution == 'plugin' }}",
            self.workflow,
        )
        self.assertIn("env.NEEDS_WINDOWS_VULKAN_SDK == 'true'", self.workflow)

    def test_only_host_archives_bundle_the_vulkan_loader(self) -> None:
        self.assertIn(
            "BUNDLES_WINDOWS_VULKAN_LOADER: ${{ matrix.target == "
            "'x86_64-pc-windows-msvc' || contains(matrix.features, 'vulkan') || "
            "matrix.distribution == 'host' }}",
            self.workflow,
        )
        self.assertEqual(
            self.workflow.count("env.BUNDLES_WINDOWS_VULKAN_LOADER == 'true'"),
            2,
        )

    def test_windows_cuda_uses_only_cuda_12_6_component_names(self) -> None:
        self.assertIn(
            "sub-packages: '[\"nvcc\", \"cudart\", \"cublas\", "
            "\"cublas_dev\", \"thrust\"]'",
            self.workflow,
        )
        windows_cuda = self.workflow.split(
            "- name: Install CUDA toolkit (Windows)", 1
        )[1].split("- name: Install Rust toolchain", 1)[0]
        self.assertNotIn('"crt"', windows_cuda)
        self.assertNotIn('"nvvm"', windows_cuda)

    def test_single_target_dispatch_does_not_build_the_xcframework(self) -> None:
        xcframework = self.workflow.split("\n  xcframework:\n", 1)[1].split(
            "\n  checksums:\n", 1
        )[0]
        self.assertIn(
            "if: ${{ github.event_name != 'workflow_dispatch' || "
            "inputs.only_target == '' }}",
            xcframework,
        )

    def test_windows_arm64_cross_build_disables_openmp(self) -> None:
        openmp_contract = self.core_build_rs.split(
            "let openmp_unsupported_target =", 1
        )[1].split(";", 1)[0]
        self.assertIn("is_windows_arm64", openmp_contract)


if __name__ == "__main__":
    unittest.main()
