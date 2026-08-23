from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

PATH = Path(__file__).with_name("release_correctness_binding.py")
SPEC = importlib.util.spec_from_file_location("release_correctness_binding", PATH)
assert SPEC and SPEC.loader
BINDING = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BINDING)


class ReleaseCorrectnessBindingTests(unittest.TestCase):
    def test_build_and_verify_rehashes_every_external_anchor(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            inventory = root / "inventory.json"
            catalog = root / "catalog.json"
            backend = root / "backend.json"
            inventory.write_text("{\"inventory\": true}")
            catalog.write_text("{\"models\": []}")
            backend.write_text("{\"backends\": []}")
            matrix = root / "matrix.json"
            source_digests = {
                "architecture_inventory_sha256": BINDING.sha256(inventory),
                "model_catalog_sha256": BINDING.sha256(catalog),
                "backend_catalog_sha256": BINDING.sha256(backend),
            }
            matrix.write_text(
                json.dumps(
                    {
                        "schema": "openasr.gpu-correctness-matrix.v1",
                        "matrix_sha256": "a" * 64,
                        "source_digests": source_digests,
                    }
                )
            )
            public = root / "public.json"
            signature = root / "public.signature.json"
            public.write_text("{\"models\": []}")
            signature.write_text(json.dumps({"catalog_sha256": BINDING.sha256(public)}))
            asset = root / "openasr-linux.tar.gz"
            asset.write_bytes(b"candidate")
            sums = root / "SHA256SUMS"
            sums.write_text(f"{BINDING.sha256(asset)}  {asset.name}\n")
            deploy = root / "deploy.json"
            deploy.write_text(
                json.dumps(
                    {
                        "workflow_name": "Deploy catalog",
                        "conclusion": "success",
                        "event": "workflow_call",
                        "caller_run_id": "123",
                        "release_tag": "v0.1.37",
                        "head_sha": "b" * 40,
                    }
                )
            )
            args = type(
                "Args",
                (),
                {
                    "matrix": matrix,
                    "inventory": inventory,
                    "model_catalog": catalog,
                    "backend_catalog": backend,
                    "public_catalog": public,
                    "public_signature": signature,
                    "sha256sums": sums,
                    "deploy_run": deploy,
                    "asset": [asset],
                    "tag": "v0.1.37",
                    "tag_commit": "b" * 40,
                    "orchestrator_run_id": "123",
                    "plugin_sha256": "c" * 64,
                },
            )()
            binding = BINDING.build_binding(args)
            binding_path = root / "binding.json"
            binding_path.write_text(json.dumps(binding))
            BINDING.verify_binding(binding, args)
            public.write_text("tampered")
            with self.assertRaisesRegex(BINDING.BindingError, "matrix model catalog digest|binding does not match"):
                BINDING.verify_binding(binding, args)

    def test_family_regression_is_candidate_blocker_not_post_publish_monitor(self) -> None:
        root = Path(__file__).resolve().parents[2]
        family = (root / ".github/workflows/family-regression.yml").read_text(encoding="utf-8")
        release = (root / ".github/workflows/release-core.yml").read_text(encoding="utf-8")
        deploy = (root / ".github/workflows/deploy-catalog.yml").read_text(encoding="utf-8")
        self.assertIn("workflow_call:", family)
        self.assertIn("pre_publication:", family)
        self.assertIn("required: true", family.split("pre_publication:", maxsplit=1)[1])
        self.assertIn("candidate_cli_artifact:", family)
        self.assertIn("correctness_matrix_artifact:", family)
        self.assertIn("uses: ./.github/workflows/family-regression.yml", release)
        self.assertIn("pre_publication: true", release)
        self.assertIn("needs: [resolve, binaries, prepublication-family]", release)
        self.assertLess(release.index("prepublication-family:"), release.index("deploy-catalog:"))
        self.assertLess(release.index("prepublication-family:"), release.index("finalize-notes:"))
        self.assertNotIn("It is not the publication blocker", family)
        self.assertIn("gate-correctness:", deploy)
        self.assertIn("gpu_correctness_gate.py validate", deploy)
        self.assertIn("needs: [gate-correctness, gate-released-binary-compat]", deploy)
        self.assertLess(deploy.index("gate-correctness:"), deploy.index("\n  deploy:"))

    def test_binding_cannot_activate_without_the_deploy_catalog_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            deploy = root / "deploy.json"
            deploy.write_text(
                json.dumps(
                    {
                        "workflow_name": "Deploy catalog",
                        "conclusion": "success",
                        "event": "release",
                        "caller_run_id": "123",
                        "release_tag": "v0.1.37",
                        "head_sha": "b" * 40,
                    }
                )
            )
            with self.assertRaisesRegex(BINDING.BindingError, "release orchestrator"):
                BINDING.validate_deploy_run(
                    {
                        "workflow_name": "Deploy catalog",
                        "conclusion": "success",
                        "event": "release",
                        "caller_run_id": "123",
                        "release_tag": "v0.1.37",
                        "head_sha": "b" * 40,
                    },
                    "123",
                    "v0.1.37",
                    "b" * 40,
                )

    def test_unrelated_successful_run_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            deploy = root / "deploy.json"
            deploy.write_text(
                json.dumps(
                    {
                        "workflow_name": "Unrelated workflow",
                        "conclusion": "success",
                        "event": "workflow_call",
                        "caller_run_id": "123",
                    }
                )
            )
            with self.assertRaisesRegex(BINDING.BindingError, "Deploy catalog"):
                BINDING.validate_deploy_run(
                    {"workflow_name": "Unrelated workflow", "conclusion": "success", "event": "workflow_call", "caller_run_id": "123"},
                    "123",
                    "v0.1.37",
                    "b" * 40,
                )


if __name__ == "__main__":
    unittest.main()
