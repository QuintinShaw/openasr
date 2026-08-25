from __future__ import annotations

import json
import os
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "release-binaries.yml"
MATRIX = ROOT / "tooling" / "release-manifest" / "release_binaries_matrix.json"
CORE_BUILD_RS = ROOT / "crates" / "openasr-core" / "build.rs"
QUALIFICATION_SIGN = ROOT / "scripts" / "sign-and-verify-qualification-manifests.sh"
QUALIFICATION_LOCK = ROOT / "scripts" / "qualification-release-lock.sh"


class WindowsBackendReleaseContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.workflow = WORKFLOW.read_text(encoding="utf-8")
        cls.core_build_rs = CORE_BUILD_RS.read_text(encoding="utf-8")
        cls.matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
        cls.qualification_sign = QUALIFICATION_SIGN.read_text(encoding="utf-8")
        cls.qualification_lock = QUALIFICATION_LOCK.read_text(encoding="utf-8")

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
        features: str | None,
        provider: str | None,
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
            features=None,
            provider=None,
            distribution="host",
        )
        self.assert_matrix_leg(
            "x86_64-pc-windows-msvc-vulkan-generic-plugin",
            asset="windows-x86_64-vulkan-generic-plugin",
            features="vulkan",
            provider="vulkan",
            distribution="plugin",
        )
        for sm in ("75", "80", "86", "89", "90", "120"):
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

    def test_neutral_host_build_stages_only_the_cpu_rescue_provider(self) -> None:
        self.assertIn("let build_vulkan = feat_vulkan;", self.core_build_rs)
        self.assertNotIn("feat_vulkan || use_backend_dl", self.core_build_rs)
        self.assertIn('"schema_version": 4', self.core_build_rs)
        self.assertNotIn("OPENASR_BUNDLED_VULKAN_CONTRACT_SHA256", self.core_build_rs)

    def test_target_scoped_optional_plugins_feed_one_catalog_and_update_hint(self) -> None:
        required = (
            "backend-pack-cuda-sm_*.json",
            "backend-pack-hip-gfx*.json",
            "backend-pack-vulkan-generic.json",
            "--require-single-target",
            "--out dist/catalog.backends.candidate.json",
            "--out dist/backend-plugin-hints.json",
            'names.append(f"backend-pack-cuda-sm_{row[\'cuda_gpu_target\']}.json")',
            'names.append(f"backend-pack-hip-{row[\'hip_gpu_target\']}.json")',
            'names.append("backend-pack-vulkan-generic.json")',
            'echo "backend-plugin-hints.json"',
            'echo "catalog.backends.candidate.json"',
            "staging/catalog.backends.candidate.json",
        )
        for fragment in required:
            self.assertIn(fragment, self.workflow)

    def test_plugin_vendor_and_signing_steps_cover_all_gpu_providers(self) -> None:
        self.assertIn(
            "matrix.distribution == 'plugin'", self.workflow
        )
        self.assertIn('$provider -eq "cuda"', self.workflow)
        self.assertIn('$provider -eq "hip"', self.workflow)
        self.assertIn("$provider -eq 'vulkan'", self.workflow)
        self.assertIn("VENDOR_LAYER_KEY=cuda-runtime", self.workflow)
        self.assertIn("VENDOR_LAYER_KEY=rocm-runtime", self.workflow)
        self.assertIn("VENDOR_OWNER", self.workflow)
        self.assertIn('[ "${VENDOR_OWNER:-false}" = "true" ]', self.workflow)
        self.assertIn("Resolve Windows binaries to sign", self.workflow)
        self.assertIn("$env:PLUGIN_ASSET_PATH", self.workflow)
        self.assertIn("Attest release subjects (attempt 1)", self.workflow)
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
            "vulkan-1\\.dll",
            "VENDOR_LAYER_KEY=vulkan-loader",
        ):
            self.assertIn(symbol, self.workflow)

    def test_provider_probe_driver_evidence_fails_closed_on_missing_or_truncated_output(self) -> None:
        ggml = ROOT / "crates/openasr-core/third_party/openasr-ggml/src"
        for source_path in (
            ggml / "ggml-cuda/ggml-cuda.cu",
            ggml / "ggml-vulkan/ggml-vulkan.cpp",
        ):
            source = source_path.read_text(encoding="utf-8")
            self.assertIn(
                "driver_out == nullptr || driver_out_capacity == 0", source
            )
            self.assertIn(
                "static_cast<size_t>(driver_length) >= driver_out_capacity", source
            )
            self.assertIn("driver_out[0] = '\\0';", source)
            self.assertIn("catch (...)", source)

    def test_vulkan_exported_init_and_graph_compute_keep_exceptions_inside_status_boundaries(self) -> None:
        source = (
            ROOT
            / "crates/openasr-core/third_party/openasr-ggml/src/ggml-vulkan/ggml-vulkan.cpp"
        ).read_text(encoding="utf-8")
        init = source.split("ggml_backend_t ggml_backend_vk_init", 1)[1].split(
            "bool ggml_backend_is_vk", 1
        )[0]
        self.assertLess(init.index("try {"), init.index("VK_LOG_DEBUG"))
        self.assertIn("catch (const vk::SystemError & error)", init)
        self.assertIn("catch (...)", init)
        self.assertIn("return nullptr;", init)

        graph = source.split("static ggml_status ggml_backend_vk_graph_compute", 1)[
            1
        ].split("static void ggml_vk_graph_optimize", 1)[0]
        self.assertIn("try {", graph)
        self.assertIn("catch (const vk::SystemError & error)", graph)
        self.assertIn("catch (const std::bad_alloc &)", graph)
        self.assertIn("catch (...)", graph)
        self.assertIn("GGML_STATUS_EXECUTION_FAILED", graph)

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
        self.assertIsNot(sm120.get("experimental"), True)
        self.assertTrue(sm120.get("vendor_owner"))

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
        self.assertNotIn("uses: ./.github/actions/attest-build-provenance", self.workflow)
        self.assertEqual(
            self.workflow.count("uses: actions/attest-build-provenance@v4"), 3
        )
        self.assertIn("uses: ./.github/actions/rust-cache", self.workflow)

    def test_release_provenance_is_aggregated_and_retryable(self) -> None:
        build = self.workflow.split("\n  build:\n", 1)[1].split(
            "\n  xcframework:\n", 1
        )[0]
        xcframework = self.workflow.split("\n  xcframework:\n", 1)[1].split(
            "\n  checksums:\n", 1
        )[0]
        checksums = self.workflow.split("\n  checksums:\n", 1)[1].split(
            "\n  upload-to-release:\n", 1
        )[0]

        self.assertNotIn("actions/attest", build)
        self.assertNotIn("actions/attest", xcframework)
        self.assertIn("subject-checksums: dist/SHA256SUMS", checksums)
        self.assertEqual(checksums.count("subject-checksums: dist/SHA256SUMS"), 3)
        self.assertIn("continue-on-error: true", checksums)
        self.assertIn("run: sleep 30", checksums)
        self.assertIn("run: sleep 90", checksums)
        self.assertIn("needs.build.result == 'success'", checksums)

    def test_manual_dispatch_cannot_recover_or_mutate_release_assets(self) -> None:
        dispatch_inputs = self.workflow.split("  workflow_dispatch:\n", 1)[1].split(
            "  workflow_call:\n", 1
        )[0]
        call_inputs = self.workflow.split("  workflow_call:\n", 1)[1].split(
            "permissions:\n", 1
        )[0]
        self.assertNotIn("formal_release:", dispatch_inputs)
        self.assertIn("formal_release:", call_inputs)
        self.assertNotIn("source_run_id:", self.workflow)
        self.assertNotIn("supplemental_source_run_id:", self.workflow)
        self.assertNotIn("promote_cuda_targets:", self.workflow)
        self.assertNotIn("Upload recovered assets to release", self.workflow)
        self.assertIn("formal_release:", self.workflow)
        self.assertIn("manual release-binaries runs require one diagnostic only_target", self.workflow)
        self.assertIn("inputs.formal_release == true", self.workflow)
        self.assertIn('CALLER_WORKFLOW: ${{ github.workflow }}', self.workflow)
        self.assertIn('[ "$CALLER_WORKFLOW" = "Release core" ]', self.workflow)
        self.assertIn('[ "$CALLER_REF" = "refs/heads/main" ]', self.workflow)
        upload_job = self.workflow.split("\n  upload-to-release:\n", 1)[1].split(
            "\n  verify-completeness:\n", 1
        )[0]
        self.assertIn("inputs.formal_release == true", upload_job)
        self.assertIn("refusing to overwrite assets on a public release", self.workflow)
        self.assertIn("release tag and checked-out source commit differ", self.workflow)
        self.assertIn("formal release assets may be uploaded only to an existing draft", self.workflow)
        self.assertIn("contains unexpected asset(s)", self.workflow)
        self.assertIn("staging/*.sha256", self.workflow)
    def test_qualification_manifests_bind_the_successful_attestation_bundle(self) -> None:
        checksums = self.workflow.split("\n  checksums:\n", 1)[1].split(
            "\n  upload-to-release:\n", 1
        )[0]
        self.assertIn("id: attest_release_3", checksums)
        for attempt in (1, 2, 3):
            self.assertIn(
                f"steps.attest_release_{attempt}.outputs.bundle-path", checksums
            )
            self.assertIn(f'ATTEST_{attempt}_OUTCOME:', checksums)
        self.assertNotIn("\n          gh attestation download", checksums)
        self.assertIn("qualification_manifest.py", checksums)
        self.assertIn(
            "QUALIFICATION_SOURCE_DIGEST: ${{ needs.select-matrix.outputs.source_digest }}",
            checksums,
        )
        self.assertIn('--source-digest "$QUALIFICATION_SOURCE_DIGEST"', checksums)
        self.assertIn("if actual != expected:", checksums)
        self.assertIn("dist/backend-pack-vulkan-generic.json", checksums)
        self.assertIn("compiler.artifact_cell", checksums)
        self.assertIn("compiler.expected_artifact_cells", checksums)
        self.assertNotIn("--bundled-vulkan-target", checksums)
        self.assertNotIn("vulkan-windows-x86_64", checksums)
        self.assertIn("backend-qualification-assets", checksums)
        self.assertIn("openasr-*-build-provenance.bundle.json", checksums)
        self.assertIn("openasr-*-qualification-*.json", checksums)
        self.assertLess(
            checksums.index("subject-checksums: dist/SHA256SUMS"),
            checksums.index("Compile inert backend qualification manifests"),
        )
        compile_step = checksums.split(
            "- name: Compile inert backend qualification manifests", 1
        )[1].split("- name: Upload inert backend qualification assets", 1)[0]
        for forbidden in ("activation_mode", "active.json", "catalog.backends.candidate"):
            self.assertNotIn(forbidden, compile_step)

    def test_qualification_signing_is_local_exact_tag_and_round_trip_verified(self) -> None:
        checksums = self.workflow.split("\n  checksums:\n", 1)[1].split(
            "\n  upload-to-release:\n", 1
        )[0]
        self.assertNotIn("OPENASR_CATALOG_SIGNING_KEY_SEED_HEX", checksums)
        for fragment in (
            'if [ "${CI:-}" = "true" ] || [ "${GITHUB_ACTIONS:-}" = "true" ]',
            'git status --porcelain --untracked-files=normal',
            'git rev-parse "${tag}^{commit}"',
            '[ "$head_commit" = "$tag_commit" ]',
            'git/ref/tags/${tag}',
            '[ "$remote_tag_object" = "$local_tag_object" ]',
            '[ "$remote_tag_commit" = "$tag_commit" ]',
            '[ "$is_draft" = "true" ]',
            'source_digest != tag_commit',
            'cells != expected_cells',
            'compiler.expected_artifact_cells',
            'compiler.SCHEMA_VERSION',
            'compiler._safe_basename',
            'expected_signature_count=',
            '--promote-cuda-targets',
            'unknown promoted CUDA target(s)',
            'qualification-release-lock.sh acquire "$tag" "$lock_token"',
            'qualification-release-lock.sh release "$tag" "$lock_token"',
            "stopped being a draft before signature upload",
            "gh attestation verify",
            '--signer-workflow "${repository}/.github/workflows/release-binaries.yml"',
            '--source-digest "$source_digest"',
            "--deny-self-hosted-runners",
            "__openasr-sign-qualification-manifest",
            "__openasr-verify-qualification-manifest",
            'gh release upload "$tag"',
            "QUALIFICATION-MANIFESTS-SIGNED-AND-VERIFIED",
        ):
            self.assertIn(fragment, self.qualification_sign)
        self.assertLess(
            self.qualification_sign.index(
                "unset OPENASR_CATALOG_SIGNING_KEY_SEED_HEX"
            ),
            self.qualification_sign.index("cargo build --quiet -p openasr-cli"),
        )
        self.assertIn(
            'OPENASR_CATALOG_SIGNING_KEY_SEED_HEX="$signing_key_seed"',
            self.qualification_sign,
        )
        self.assertNotIn("cargo run", self.qualification_sign)

    def test_qualification_release_mutations_share_an_atomic_remote_lock(self) -> None:
        self.assertIn(
            'asset_name="openasr-${version}-qualification-mutation.lock"',
            self.qualification_lock,
        )
        self.assertIn(
            'gh release upload "$tag" "$temporary/$asset_name"',
            self.qualification_lock,
        )
        acquire = self.qualification_lock.split("  acquire)", 1)[1].split(
            "  release)", 1
        )[0]
        self.assertNotIn("--clobber", acquire)
        self.assertIn(
            'cmp -s -- "$token_file" "$temporary/$asset_name"',
            self.qualification_lock,
        )
        self.assertIn("gh release delete-asset", self.qualification_lock)
        self.assertEqual(
            self.workflow.count(
                'scripts/qualification-release-lock.sh acquire "$tag" "$lock_token"'
            ),
            1,
        )
        self.assertEqual(
            self.workflow.count(
                'scripts/qualification-release-lock.sh release "$tag" "$lock_token"'
            ),
            2,
        )
        self.assertIn(
            'lock_asset="openasr-${version}-qualification-mutation.lock"',
            self.workflow,
        )
        self.assertIn("stopped being a draft before asset upload", self.workflow)

    def test_qualification_release_lock_is_exclusive_and_nonce_owned(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            binary_dir = root / "bin"
            remote = root / "release-assets"
            binary_dir.mkdir()
            remote.mkdir()
            fake_gh = binary_dir / "gh"
            fake_gh.write_text(
                textwrap.dedent(
                    """\
                    #!/usr/bin/env python3
                    import os
                    import shutil
                    import sys
                    from pathlib import Path

                    args = sys.argv[1:]
                    remote = Path(os.environ["FAKE_GH_REMOTE"])
                    if args[:2] == ["release", "upload"]:
                        source = Path(args[3])
                        target = remote / source.name
                        if target.exists():
                            raise SystemExit(1)
                        shutil.copyfile(source, target)
                    elif args[:2] == ["release", "download"]:
                        asset = args[args.index("-p") + 1]
                        destination = Path(args[args.index("-D") + 1]) / asset
                        shutil.copyfile(remote / asset, destination)
                    elif args[:2] == ["release", "delete-asset"]:
                        (remote / args[3]).unlink()
                    else:
                        raise SystemExit(2)
                    """
                ),
                encoding="utf-8",
            )
            fake_gh.chmod(0o755)
            environment = dict(os.environ)
            environment["FAKE_GH_REMOTE"] = str(remote)
            environment["PATH"] = f"{binary_dir}:{environment['PATH']}"
            first = root / "first.token"
            second = root / "second.token"
            wrong = root / "wrong.token"
            lock_asset = remote / "openasr-1.2.3-qualification-mutation.lock"

            subprocess.run(
                [QUALIFICATION_LOCK, "acquire", "v1.2.3", first],
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertTrue(lock_asset.is_file())
            refused = subprocess.run(
                [QUALIFICATION_LOCK, "acquire", "v1.2.3", second],
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(refused.returncode, 0)
            self.assertFalse(second.exists())
            wrong.write_text("0" * 64 + "\n", encoding="ascii")
            refused = subprocess.run(
                [QUALIFICATION_LOCK, "release", "v1.2.3", wrong],
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(refused.returncode, 0)
            self.assertTrue(lock_asset.is_file())
            subprocess.run(
                [QUALIFICATION_LOCK, "release", "v1.2.3", first],
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertFalse(lock_asset.exists())
            self.assertFalse(first.exists())

    def test_release_jobs_pin_one_event_tag_and_checkout_source_digest(self) -> None:
        select = self.workflow.split("\n  select-matrix:\n", 1)[1].split(
            "\n  build:\n", 1
        )[0]
        self.assertIn('checkout_commit="$(git rev-parse HEAD^{commit})"', select)
        self.assertIn('[ "$checkout_commit" = "$GITHUB_SHA" ]', select)
        self.assertIn('[ "$tag_commit" = "$checkout_commit" ]', select)
        self.assertIn('echo "source_digest=$checkout_commit"', select)
        self.assertGreaterEqual(
            self.workflow.count("ref: ${{ needs.select-matrix.outputs.source_digest }}"),
            3,
        )
        self.assertGreaterEqual(
            self.workflow.count("ref: ${{ needs.checksums.outputs.source_digest }}"),
            2,
        )

    def test_catalog_candidate_uses_only_release_blocking_plugin_targets(self) -> None:
        required_cuda = [
            row
            for row in self.matrix
            if row.get("provider") == "cuda" and not row.get("experimental", False)
        ]
        required_hip = [
            row
            for row in self.matrix
            if row.get("provider") == "hip" and not row.get("experimental", False)
        ]
        required_vulkan = [
            row
            for row in self.matrix
            if row.get("provider") == "vulkan" and not row.get("experimental", False)
        ]
        self.assertEqual(len(required_cuda), 6)
        self.assertEqual(len(required_hip), 14)
        self.assertEqual(len(required_vulkan), 1)
        self.assertEqual(
            [f'backend-pack-cuda-sm_{row["cuda_gpu_target"]}.json' for row in required_cuda],
            [
                "backend-pack-cuda-sm_75.json",
                "backend-pack-cuda-sm_80.json",
                "backend-pack-cuda-sm_86.json",
                "backend-pack-cuda-sm_89.json",
                "backend-pack-cuda-sm_90.json",
                "backend-pack-cuda-sm_120.json",
            ],
        )
        self.assertIn('not row.get("experimental", False)', self.workflow)
        self.assertIn('entry="dist/backend-pack-cuda-sm_${target}.json"', self.workflow)
        self.assertIn('entry="dist/backend-pack-hip-${target}.json"', self.workflow)

    def test_full_matrix_has_one_vendor_owner_per_distinct_runtime(self) -> None:
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
        vulkan_owners = [
            row["target"]
            for row in self.matrix
            if row.get("provider") == "vulkan" and row.get("vendor_owner") is True
        ]
        self.assertEqual(
            cuda_owners,
            [
                "x86_64-pc-windows-msvc-cuda-sm_75-plugin",
                "x86_64-pc-windows-msvc-cuda-sm_120-plugin",
            ],
        )
        self.assertEqual(hip_owners, ["x86_64-pc-windows-msvc-hip-gfx1030-plugin"])
        self.assertEqual(
            vulkan_owners,
            ["x86_64-pc-windows-msvc-vulkan-generic-plugin"],
        )

    def test_diagnostic_only_target_temporarily_owns_vendor_assets(self) -> None:
        self.assertIn(
            "VENDOR_OWNER: ${{ matrix.distribution == 'plugin' && "
            "((inputs.only_target != '' && matrix.target == inputs.only_target) "
            "|| matrix.vendor_owner) }}",
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

    def test_only_optional_vulkan_plugin_installs_the_vulkan_sdk(self) -> None:
        self.assertIn(
            "NEEDS_WINDOWS_VULKAN_SDK: ${{ contains(matrix.features, 'vulkan') }}",
            self.workflow,
        )
        self.assertIn("env.NEEDS_WINDOWS_VULKAN_SDK == 'true'", self.workflow)

    def test_only_optional_vulkan_pack_owns_the_vulkan_loader(self) -> None:
        self.assertIn(
            "BUNDLES_WINDOWS_VULKAN_LOADER: ${{ matrix.distribution == 'plugin' && matrix.provider == 'vulkan' }}",
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
        self.assertIn("if: ${{ inputs.formal_release == true }}", xcframework)

    def test_windows_arm64_cross_build_disables_openmp(self) -> None:
        openmp_contract = self.core_build_rs.split(
            "let openmp_unsupported_target =", 1
        )[1].split(";", 1)[0]
        self.assertIn("is_windows_arm64", openmp_contract)

    def test_plugin_legs_build_openasr_core_not_cli(self) -> None:
        build = self.workflow.split("\n  build:\n", 1)[1].split(
            "\n  xcframework:\n", 1
        )[0]
        self.assertIn('[ "${{ matrix.distribution }}" = "plugin" ]', build)
        self.assertIn('crate="openasr-core"', build)
        self.assertIn('crate="openasr-cli"', build)
        self.assertIn('-p "${crate}"', build)
        self.assertNotIn("cargo build --release -p openasr-cli", build)
        self.assertNotIn("cargo zigbuild --release -p openasr-cli", build)
        self.assertIn("Verify optional backend PE contract", build)
        self.assertIn("openasr-backend-packs\\$provider\\ggml-$provider.dll", build)


if __name__ == "__main__":
    unittest.main()
