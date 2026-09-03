from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import qualification_manifest
import release_completeness as completeness


ROOT = Path(__file__).resolve().parents[2]
MATRIX = completeness.load_matrix()
WORKFLOW = (ROOT / ".github" / "workflows" / "release-binaries.yml").read_text(
    encoding="utf-8"
)


def _pack(*, vendor: str, targets: list[str], files: list[str], ident: str) -> dict:
    return {
        "id": ident,
        "vendor": vendor,
        "targets": targets,
        "files": [{"filename": name, "sha256": "a" * 64, "size": 1} for name in files],
    }


class ReleaseCompletenessTests(unittest.TestCase):
    def test_required_cli_archives_match_non_experimental_non_plugin_matrix_rows(self) -> None:
        names = completeness.required_cli_archives("0.1.37", MATRIX)
        self.assertIn("openasr-0.1.37-linux-x86_64.tar.gz", names)
        self.assertIn("openasr-0.1.37-windows-x86_64-neutral.zip", names)
        self.assertIn("openasr-0.1.37-linux-x86_64-vulkan.tar.gz", names)
        self.assertIn("openasr-0.1.37-xcframework.zip", names)
        self.assertIn("openasr-0.1.37-xcframework.zip.sha256", names)
        self.assertNotIn("openasr-0.1.37-linux-x86_64-musl.tar.gz", names)
        self.assertNotIn("openasr-0.1.37-windows-arm64.zip", names)
        self.assertNotIn("openasr-0.1.37-windows-x86_64-vulkan-generic-plugin.dll", names)

    def test_optional_experimental_archives_cover_musl_and_windows_arm64(self) -> None:
        names = completeness.optional_experimental_archives("0.1.37", MATRIX)
        self.assertEqual(
            names,
            {
                "openasr-0.1.37-linux-x86_64-musl.tar.gz",
                "openasr-0.1.37-linux-arm64-musl.tar.gz",
                "openasr-0.1.37-windows-arm64.zip",
            },
        )

    def test_qualification_names_use_artifact_cell_including_vulkan_generic(self) -> None:
        entries = [
            _pack(
                vendor="cuda",
                targets=["sm_75"],
                files=["openasr-0.1.37-windows-x86_64-cuda-sm_75-plugin.dll"],
                ident="cuda-sm-75",
            ),
            _pack(
                vendor="vulkan",
                targets=[],
                files=["openasr-0.1.37-windows-x86_64-vulkan-generic-plugin.dll"],
                ident="vulkan-generic",
            ),
        ]
        names = completeness.required_from_backend_packs("0.1.37", entries)
        self.assertEqual(qualification_manifest.artifact_cell(entries[1]), ("vulkan", "generic"))
        self.assertIn("openasr-0.1.37-qualification-cuda-sm_75.json", names)
        self.assertIn("openasr-0.1.37-qualification-vulkan-generic.json", names)
        self.assertNotIn("openasr-0.1.37-qualification-vulkan-windows-x86_64.json", names)

    def test_compare_allows_successful_experimental_archives_and_rejects_unknown_extras(self) -> None:
        entries = [
            _pack(
                vendor="vulkan",
                targets=[],
                files=["plugin.dll", "vendor.zip"],
                ident="vulkan-generic",
            )
        ]
        required = completeness.required_assets("0.1.37", MATRIX, entries)
        actual = set(required)
        actual.add("openasr-0.1.37-windows-arm64.zip")
        missing, extra, lock = completeness.compare_assets(
            version="0.1.37",
            actual=actual,
            matrix=MATRIX,
            pack_entries=entries,
        )
        self.assertEqual(missing, [])
        self.assertEqual(extra, [])
        self.assertIsNone(lock)

        actual.add("unexpected.bin")
        missing, extra, lock = completeness.compare_assets(
            version="0.1.37",
            actual=actual,
            matrix=MATRIX,
            pack_entries=entries,
        )
        self.assertEqual(extra, ["unexpected.bin"])

    def test_compare_accepts_detached_qualification_signatures_for_exact_cells_only(self) -> None:
        entries = [
            _pack(
                vendor="cuda",
                targets=["sm_75"],
                files=["openasr-0.1.39-windows-x86_64-cuda-sm_75-plugin.dll"],
                ident="cuda-sm-75",
            )
        ]
        required = completeness.required_assets("0.1.39", MATRIX, entries)
        self.assertNotIn("openasr-0.1.39-qualification-cuda-sm_75.signature.json", required)

        actual = set(required)
        actual.add("openasr-0.1.39-qualification-cuda-sm_75.signature.json")
        missing, extra, lock = completeness.compare_assets(
            version="0.1.39",
            actual=actual,
            matrix=MATRIX,
            pack_entries=entries,
        )
        self.assertEqual((missing, extra, lock), ([], [], None))

        actual.add("openasr-0.1.39-qualification-cuda-sm_120.signature.json")
        _, extra, _ = completeness.compare_assets(
            version="0.1.39",
            actual=actual,
            matrix=MATRIX,
            pack_entries=entries,
        )
        self.assertEqual(extra, ["openasr-0.1.39-qualification-cuda-sm_120.signature.json"])

    def test_compare_rejects_qualification_mutation_lock(self) -> None:
        entries = [
            _pack(
                vendor="vulkan",
                targets=[],
                files=["plugin.dll"],
                ident="vulkan-generic",
            )
        ]
        required = completeness.required_assets("0.1.37", MATRIX, entries)
        actual = set(required)
        actual.add("openasr-0.1.37-qualification-mutation.lock")
        missing, extra, lock = completeness.compare_assets(
            version="0.1.37",
            actual=actual,
            matrix=MATRIX,
            pack_entries=entries,
        )
        self.assertEqual(lock, "openasr-0.1.37-qualification-mutation.lock")
        self.assertEqual(extra, [])
        self.assertEqual(missing, [])

    def test_pack_names_cli_emits_the_21_required_backend_entries(self) -> None:
        names = completeness.backend_pack_names(MATRIX)
        self.assertEqual(len(names), 21)
        self.assertIn("backend-pack-vulkan-generic.json", names)
        self.assertEqual(len([name for name in names if name.startswith("backend-pack-cuda-")]), 6)
        self.assertEqual(len([name for name in names if name.startswith("backend-pack-hip-")]), 14)

    def test_compare_cli_accepts_a_complete_actual_list(self) -> None:
        entries = [
            _pack(
                vendor="cuda",
                targets=["sm_75"],
                files=["cuda.dll"],
                ident="cuda-sm-75",
            ),
            _pack(
                vendor="vulkan",
                targets=[],
                files=["vulkan.dll"],
                ident="vulkan-generic",
            ),
        ]
        required = completeness.required_assets("0.1.37", MATRIX, entries)
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "backend-pack-cuda-sm_75.json").write_text(
                json.dumps(entries[0]), encoding="utf-8"
            )
            (root / "backend-pack-vulkan-generic.json").write_text(
                json.dumps(entries[1]), encoding="utf-8"
            )
            actual = root / "actual.txt"
            actual.write_text("\n".join(sorted(required)) + "\n", encoding="utf-8")
            missing, extra, lock = completeness.compare_assets(
                version="0.1.37",
                actual=required,
                matrix=MATRIX,
                pack_entries=completeness.load_pack_entries(root),
            )
            self.assertEqual((missing, extra, lock), ([], [], None))


class CompletenessWorkflowContractTests(unittest.TestCase):
    def test_completeness_job_can_see_drafts_and_uses_shared_expected_set(self) -> None:
        job = WORKFLOW.split("\n  verify-completeness:\n", 1)[1]
        header = job.split("\n    steps:", 1)[0]
        self.assertIn("contents: write", header)
        self.assertIn("scripts/verify-release-completeness.sh", job)
        self.assertNotIn("qualification-vulkan-windows-x86_64", job)
        self.assertNotIn("is not an exact CUDA/HIP target entry", job)

    def test_workflow_does_not_expect_the_retired_vulkan_windows_qualification_name(self) -> None:
        self.assertNotIn("qualification-vulkan-windows-x86_64", WORKFLOW)
        self.assertNotIn("vulkan-windows-x86_64.json", WORKFLOW)


if __name__ == "__main__":
    unittest.main()
