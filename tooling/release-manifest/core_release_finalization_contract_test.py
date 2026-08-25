from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class CoreReleaseFinalizationContractTests(unittest.TestCase):
    def test_retired_parallel_release_binding_authority_is_absent(self) -> None:
        self.assertFalse(
            (ROOT / "tooling/release-manifest/release_correctness_binding.py").exists(),
            "deploy-catalog-binding.json is the sole release/deploy binding",
        )

    def test_release_caller_grants_every_permission_requested_by_reusable_jobs(self) -> None:
        release = (ROOT / ".github/workflows/release-core.yml").read_text(encoding="utf-8")
        binaries = (ROOT / ".github/workflows/release-binaries.yml").read_text(
            encoding="utf-8"
        )
        caller = release.split("\n  binaries:\n", maxsplit=1)[1].split(
            "\n  prepublication-family:\n", maxsplit=1
        )[0]
        caller_permissions = dict(
            re.findall(r"(?m)^      ([a-z-]+): (read|write)$", caller)
        )
        requested_permissions = re.findall(
            r"(?m)^(?:  |      )([a-z-]+): (read|write)$", binaries
        )
        permission_rank = {"read": 1, "write": 2}

        for scope, requested in requested_permissions:
            granted = caller_permissions.get(scope)
            self.assertIsNotNone(
                granted,
                f"release-core reusable caller does not grant requested {scope}: {requested}",
            )
            self.assertGreaterEqual(
                permission_rank[granted],
                permission_rank[requested],
                f"release-core reusable caller grants {scope}: {granted}, below {requested}",
            )
    def test_reusable_release_declares_every_referenced_input(self) -> None:
        binaries = (ROOT / ".github/workflows/release-binaries.yml").read_text(
            encoding="utf-8"
        )
        workflow_call = re.search(
            r"(?ms)^  workflow_call:\n(?P<body>.*?)(?=^  # Formal releases)",
            binaries,
        )
        self.assertIsNotNone(workflow_call)
        declared = set(
            re.findall(r"(?m)^      ([a-z][a-z0-9_]*):$", workflow_call.group("body"))
        )
        referenced = set(re.findall(r"\binputs\.([a-z][a-z0-9_]*)", binaries))

        self.assertEqual(
            referenced - declared,
            set(),
            "workflow_call must declare every inputs.* value used by the reusable workflow",
        )
        self.assertNotIn(
            "github.event.inputs.",
            binaries,
            "shared dispatch/call inputs must use the typed inputs context",
        )

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
        self.assertNotRegex(release, r"(?m)^  push:")
        self.assertIn("workflow_dispatch:", release)
        self.assertIn("WANT: ${{ inputs.version }}", release)
        self.assertNotIn('want="${{ inputs.version }}"', release)
        self.assertIn("should_build", release)
        self.assertIn("should_finalize", release)
        self.assertIn('tag_commit="$(git rev-parse "${tag}^{}")"', release)
        self.assertIn('[ "$tag_commit" = "${GITHUB_SHA}" ]', release)
        self.assertIn('refs/remotes/origin/main', release)
        self.assertIn('git merge-base --is-ancestor "$tag_commit"', release)
        self.assertIn("release ${tag} is a draft", release)
        self.assertIn("uses: ./.github/workflows/family-regression.yml", release)
        self.assertIn("uses: ./.github/workflows/deploy-catalog.yml", release)
        self.assertNotIn("correctness_sources_artifact", release)
        self.assertNotIn("correctness_matrix_artifact", release)
        self.assertIn("orchestrator_run_id: ${{ github.run_id }}", release)
        self.assertIn("activation_transition: published-inert", release)
        self.assertIn("needs: [resolve, prepublication-family]", release)
        self.assertIn("verify-assets", prepare)
        self.assertIn("publish_catalog.sh", prepare)
        self.assertIn("verify-catalog", prepare)
        self.assertIn("verify-cdn", prepare)
        self.assertIn("backend_hardware_evidence.py", prepare)
        self.assertNotIn("backend-hardware-audit-*.json", prepare)
        self.assertNotIn("--raw-audit", prepare)
        self.assertIn("qualification is post-publication", prepare)
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
        self.assertNotIn("backend-hardware-evidence-*.json", sync)
        self.assertNotIn("backend-hardware-audit-*.json", sync)
        self.assertNotIn("backend_hardware_evidence.py", sync)
        self.assertIn("hardware/token qualification is post-publication", sync)
        self.assertIn("verify-cdn", deploy)
        self.assertIn("workflow_call:", deploy)
        self.assertNotIn("push:\n", deploy)
        self.assertIn("gate-activation:", deploy)
        self.assertNotIn("actions/download-artifact@v8", deploy)
        self.assertIn("needs: [gate-activation, gate-released-binary-compat]", deploy)
        self.assertIn("Verify PublishedInert state", deploy)
        self.assertIn("verify-catalog-transition", deploy)
        self.assertIn("verify-revocation-transition", deploy)
        self.assertIn("qualification-signer-workflow", deploy)
        self.assertGreaterEqual(
            deploy.count("check_catalog_consistency.py"),
            4,
            "candidate and live catalogs must be checked under production trust roots",
        )
        self.assertIn("Verify candidate public catalog", deploy)
        self.assertIn("Record immutable deploy binding", deploy)
        self.assertIn("deploy-catalog-binding-${{ github.run_id }}", deploy)
        self.assertIn('"release_tag": os.environ["RELEASE_TAG"]', deploy)
        self.assertIn('"source_commit": os.environ["GITHUB_SHA"]', deploy)
        self.assertIn('"catalog_signature_sha256"', deploy)
        self.assertLess(deploy.index("verify-cdn"), deploy.index("Deploy to Cloudflare"))
        self.assertIn("catalog.openasr.org/v1/catalog.json", finalize)
        self.assertNotIn("backends-manifest", finalize)
        self.assertIn("verify-catalog", finalize)
        self.assertIn("verify-cdn", finalize)
        self.assertIn("backend_hardware_evidence.py", finalize)
        self.assertNotIn("backend-hardware-audit-*.json", finalize)
        self.assertNotIn("gpu-correctness-matrix.v1.json", finalize)
        self.assertIn("resolve_tag_commit", finalize)
        self.assertIn("git/tags/${object_sha}", finalize)
        self.assertIn("gh attestation verify", finalize)
        self.assertIn("--source-digest \"$tag_commit\"", finalize)
        self.assertIn("OPENASR_DEPLOY_CATALOG_RUN_ID", finalize)
        self.assertIn("gh run view", finalize)
        self.assertIn('"Deploy PublishedInert candidate catalog"', finalize)
        self.assertIn('value.get("headSha") != sys.argv[2]', finalize)
        self.assertIn('deploy-catalog-binding-${deploy_run_id}', finalize)
        self.assertIn('"release_tag": tag', finalize)
        self.assertIn('"activation_transition": "published-inert"', finalize)
        self.assertIn('"source_commit": commit', finalize)
        self.assertIn('"catalog_signature_sha256": digest(signature_path)', finalize)
        self.assertIn("check_catalog_consistency.py", finalize)
        self.assertIn("live catalog bytes differ", finalize)
        self.assertIn('gh release edit "$tag" --repo "$repository" --draft=false --latest', finalize)
        self.assertIn("RELEASE-PUBLISHED-INERT", finalize)
        self.assertLess(finalize.index("verify-cdn"), finalize.index("--draft=false"))

    def test_finalizer_never_publishes_before_all_gpu_provider_entries(self) -> None:
        finalize = (ROOT / "scripts/finalize-core-release.sh").read_text(encoding="utf-8")
        self.assertIn("backend-pack-*.json", finalize)
        self.assertIn('"${#cuda_entries[@]}" -ne 6', finalize)
        self.assertIn('"${#hip_entries[@]}" -ne 14', finalize)
        self.assertIn('"${#vulkan_entries[@]}" -ne 1', finalize)
        self.assertIn('"${#backend_entries[@]}" -ne 21', finalize)
        self.assertLess(finalize.index("verify-catalog"), finalize.index("--draft=false"))

    def test_release_matrix_has_one_formal_entrypoint_and_channels_wait_for_publish(self) -> None:
        release = (ROOT / ".github/workflows/release-core.yml").read_text(encoding="utf-8")
        binaries = (ROOT / ".github/workflows/release-binaries.yml").read_text(encoding="utf-8")
        channels = (ROOT / ".github/workflows/publish-core-channels.yml").read_text(
            encoding="utf-8"
        )

        self.assertNotIn("push:\n    tags:", binaries)
        self.assertIn("uses: ./.github/workflows/release-binaries.yml", release)
        self.assertIn("formal_release: true", release)
        self.assertNotIn("docker-images:", release)
        self.assertNotIn("update-homebrew-tap:", release)
        self.assertIn("types: [published]", channels)
        self.assertIn("uses: ./.github/workflows/docker-release.yml", channels)
        self.assertIn("releases/latest", channels)
        self.assertIn("distribution-gate:", channels)
        self.assertIn("backend_hardware_evidence.py", channels)
        self.assertNotIn("backend-hardware-audit-*.json", channels)
        self.assertNotIn("--raw-audit", channels)
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
        self.assertEqual(family.count("release_asset_verifier.py"), 3)
        self.assertEqual(family.count("--pattern SHA256SUMS"), 2)
        self.assertEqual(family.count("gh attestation verify"), 3)
        self.assertEqual(family.count("--signer-workflow"), 3)
        self.assertIn("attestations: read", family)
        self.assertIn("workflow_call:", family)
        self.assertIn("pre_publication", family)
        self.assertIn("release_tag", family)
        self.assertNotIn("candidate_cli_artifact", family)
        self.assertNotIn("correctness_matrix_artifact", family)
        self.assertIn("CPU-only/post-release", family)
        qwen = (ROOT / ".github/workflows/qwen-gpu-parity.yml").read_text(encoding="utf-8")
        self.assertIn("expected_provider", qwen)
        self.assertNotIn("candidate_plugin_artifact", qwen)
        self.assertNotIn("correctness_matrix_artifact", qwen)
        self.assertNotIn("correctness_receipt_artifact", qwen)
        self.assertNotIn("gpu-correctness-receipt-", qwen)
        self.assertNotIn("gpu-correctness-trace-", qwen)
        self.assertIn("qwen-gpu-parity-diagnostic-output", qwen)
        self.assertIn("not release authority", qwen)
        self.assertIn("$candidates.Count -ne 1", qwen)
        self.assertIn("$packs.Count -ne 1", qwen)
        self.assertIn("$LASTEXITCODE -ne 0", qwen)
        self.assertNotIn("no-op", qwen)
        self.assertNotIn("exit 0", qwen)

    def test_post_publication_activation_is_explicit_attested_and_two_step(self) -> None:
        qualify = (ROOT / ".github/workflows/qualify-windows-backend.yml").read_text(
            encoding="utf-8"
        )
        activate = (ROOT / ".github/workflows/activate-backend-catalog.yml").read_text(
            encoding="utf-8"
        )
        prepare = (
            ROOT / "scripts/activate-windows-backend-catalog-release.sh"
        ).read_text(encoding="utf-8")
        gate = (ROOT / "tooling/release-manifest/gpu_correctness_gate.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("GITHUB_REF", qualify)
        self.assertIn("gh release view $tag", qualify)
        self.assertIn("--json isDraft,isPrerelease,tagName,publishedAt", qualify)
        self.assertIn("$release.isDraft", qualify)
        self.assertIn("$release.isPrerelease", qualify)
        self.assertIn("$release.tagName -ne $tag", qualify)
        self.assertIn("-not $release.publishedAt", qualify)
        self.assertIn(
            "qualification may consume only already-public PublishedInert release bytes",
            qualify,
        )
        self.assertIn("gh release download $tag", qualify)
        self.assertLess(
            qualify.index("qualification may consume only already-public PublishedInert release bytes"),
            qualify.index("gh release download $tag"),
        )
        self.assertIn('"backend-pack-*.json"', qualify)
        self.assertIn('"catalog.backends.candidate.json"', qualify)
        self.assertIn('"SHA256SUMS"', qualify)
        self.assertIn('"openasr-$version-windows-x86_64-neutral.zip"', qualify)
        self.assertIn("catalog.openasr.org/v1/catalog.json", qualify)
        self.assertIn("actions/attest-build-provenance@v4", qualify)
        self.assertIn("subject-checksums", qualify)
        self.assertIn("generate_backend_hardware_evidence.py", qualify)
        self.assertIn("--release-preflight-only", qualify)
        self.assertLess(
            qualify.index("--release-preflight-only"),
            qualify.index("Expand-Archive"),
        )
        self.assertIn("windows-backend-qualification/run.ps1", qualify)
        self.assertNotIn("gh release upload", qualify)
        self.assertIn("authorization", activate)
        self.assertIn('activate:${BACKEND_ID}', activate)
        self.assertIn("activation_transition: activated", activate)
        self.assertIn("qualify-catalog", prepare)
        self.assertIn("activate-catalog", prepare)
        self.assertIn("verify-catalog-transition", prepare)
        self.assertLess(prepare.index("qualify-catalog"), prepare.index("activate-catalog"))
        self.assertIn("--json isDraft,isPrerelease,tagName,publishedAt", prepare)
        self.assertIn("already-public stable PublishedInert bytes", prepare)
        self.assertLess(prepare.index("already-public stable PublishedInert bytes"), prepare.index("gh run download"))
        self.assertIn('gh release download "$tag" --repo "$repository"', prepare)
        self.assertIn("qualification-signer-workflow", prepare)
        self.assertIn("must be independently qualified", gate)

        revoke_workflow = (
            ROOT / ".github/workflows/revoke-backend-catalog.yml"
        ).read_text(encoding="utf-8")
        revoke_prepare = (
            ROOT / "scripts/revoke-windows-backend-catalog-release.sh"
        ).read_text(encoding="utf-8")
        self.assertIn('revoke:${BACKEND_ID}', revoke_workflow)
        self.assertIn("activation_transition: revoked", revoke_workflow)
        self.assertIn("revoke-catalog", revoke_prepare)
        self.assertIn("verify-revocation-transition", revoke_prepare)
        self.assertIn("preserving its former qualification bindings", revoke_prepare)

    def test_family_regression_ignores_non_core_release_tags(self) -> None:
        family = (ROOT / ".github/workflows/family-regression.yml").read_text(
            encoding="utf-8"
        )
        jobs = family.split("\njobs:\n", maxsplit=1)[1]
        starts = list(re.finditer(r"(?m)^  ([a-z0-9-]+):\n", jobs))
        self.assertGreater(len(starts), 0)

        guard = (
            "github.event_name != 'release' || "
            "startsWith(github.event.release.tag_name, 'v')"
        )
        for index, match in enumerate(starts):
            end = starts[index + 1].start() if index + 1 < len(starts) else len(jobs)
            block = jobs[match.start() : end]
            self.assertIn(
                guard,
                block,
                f"family-regression job {match.group(1)!r} accepts desktop-v* releases",
            )

    def test_push_ci_cannot_be_bypassed_with_a_commit_message_prefix(self) -> None:
        ci = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")

        self.assertNotIn("github.event.head_commit.message", ci)


if __name__ == "__main__":
    unittest.main()
