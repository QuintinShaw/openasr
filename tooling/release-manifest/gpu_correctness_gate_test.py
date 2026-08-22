from __future__ import annotations

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


PATH = Path(__file__).with_name("gpu_correctness_gate.py")
SPEC = importlib.util.spec_from_file_location("gpu_correctness_gate", PATH)
assert SPEC and SPEC.loader
GATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GATE)


class GpuCorrectnessGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.inventory = {
            "schema": "openasr.model-family-inventory.v1",
            "families": [
                {
                    "catalog_family_id": "qwen",
                    "execution": {
                        "execution_capabilities": {
                            "cpu": True,
                            "providers": [
                                {"provider": "cuda", "full_device": True, "hybrid": False},
                                {"provider": "metal", "full_device": True, "hybrid": False},
                            ],
                        }
                    },
                    "optimization": {"auto_gpu_policy": "all-backends"},
                    "topology": {
                        "decode_driver": "shared-seq2seq-greedy",
                        "decoder_state": "causal-self-attention-kv",
                        "block_stack": "shared",
                    },
                    "quantization": {"tensor_classification": "semantic-roles-v1"},
                }
            ],
        }
        self.catalog = {
            "models": [
                {
                    "id": "qwen3-asr-0.6b",
                    "family": "qwen",
                    "kind": "asr-model",
                    "public": True,
                    "recommended_quant": "q8_0",
                    "quants": [{"quant": "q8_0"}, {"quant": "q4_k"}],
                }
            ]
        }
        self.backends = {
            "backends": [{"id": "cuda-test", "vendor": "cuda"}],
        }

    def test_projection_is_inventory_and_catalog_derived(self) -> None:
        matrix = GATE.project_matrix(self.inventory, self.catalog, self.backends)
        self.assertEqual(matrix["schema"], GATE.SCHEMA)
        cells = {(cell["provider"], tuple(cell["activation_modes"])) for cell in matrix["cells"]}
        self.assertEqual(
            cells,
            {
                ("cpu", ("explicit", "auto")),
                ("cuda", ("explicit", "auto")),
                ("metal", ("explicit", "auto")),
            },
        )
        cuda = next(cell for cell in matrix["cells"] if cell["provider"] == "cuda")
        self.assertEqual(cuda["backend_catalog_ids"], ["cuda-test"])
        self.assertEqual(cuda["reuse_modes"], ["cold", "reuse"])
        self.assertEqual(cuda["status"], "pending")
        self.assertNotIn("pass", json.dumps(matrix))

    def test_public_family_missing_from_inventory_fails_closed(self) -> None:
        catalog = copy.deepcopy(self.catalog)
        catalog["models"][0]["family"] = "missing"
        with self.assertRaisesRegex(GATE.MatrixError, "missing from architecture"):
            GATE.project_matrix(self.inventory, catalog, self.backends)

    def test_incomplete_receipts_fail_without_fabricating_hardware(self) -> None:
        matrix = GATE.project_matrix(self.inventory, self.catalog, self.backends)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "receipt.json"
            path.write_text(json.dumps({"schema": GATE.RECEIPT_SCHEMA}))
            with self.assertRaisesRegex(GATE.MatrixError, "no versioned correctness evidence"):
                GATE.validate_matrix(matrix, [path])

    def test_complete_synthetic_receipts_close_every_cell(self) -> None:
        matrix = GATE.project_matrix(self.inventory, self.catalog, self.backends)
        with tempfile.TemporaryDirectory() as temp:
            paths: list[Path] = []
            for index, cell in enumerate(matrix["cells"]):
                for evidence_class in ("placement_resource", "token_transcript"):
                    for mode in ("cold", "reuse"):
                        evidence = {
                            "schema": GATE.EVIDENCE_SCHEMA,
                            "evidence_class": evidence_class,
                            "family": cell["family"],
                            "provider": cell["provider"],
                            "device": f"test-{cell['provider']}",
                            "placement": cell["placement"],
                            "result": "pass",
                            "execution": {"mode": mode},
                            "artifacts": {
                                "binary": {"label": "binary", "sha256": "a" * 64},
                                "pack": {"label": "pack", "sha256": "b" * 64},
                                "fixture": {"label": "fixture", "sha256": "c" * 64},
                            },
                        }
                        if evidence_class == "token_transcript":
                            evidence.update(
                                {
                                    "output_plan": {
                                        "kind": "full_logits",
                                        "logits_or_scores": "complete",
                                        "tie_policy": "first_maximum",
                                    },
                                    "family_oracle": {
                                        "family": cell["family"],
                                        "tie_policy": "first_maximum",
                                    },
                                    "trace": {"token_trace_sha256": "d" * 64},
                                }
                            )
                        document = {
                            "schema": GATE.RECEIPT_SCHEMA,
                            "pack": {"model_id": cell["model_ids"][0]},
                            "evidence": evidence,
                        }
                        path = Path(temp) / f"{index}-{evidence_class}-{mode}.json"
                        path.write_text(json.dumps(document))
                        paths.append(path)
            build = Path(temp) / "build.json"
            build.write_text(
                json.dumps(
                    {
                        "schema": GATE.RECEIPT_SCHEMA,
                        "evidence": {
                            "schema": GATE.EVIDENCE_SCHEMA,
                            "evidence_class": "build_packaging",
                            "family": "release",
                            "provider": "release",
                            "device": "release",
                            "placement": "not-applicable",
                            "result": "pass",
                            "artifacts": {
                                "binary": {"label": "binary", "sha256": "a" * 64},
                                "pack": {"label": "pack", "sha256": "b" * 64},
                                "fixture": {"label": "fixture", "sha256": "c" * 64},
                            },
                        },
                    }
                )
            )
            paths.append(build)
            GATE.validate_matrix(matrix, paths)


if __name__ == "__main__":
    unittest.main()
