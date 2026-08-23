from __future__ import annotations

import copy
import hashlib
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
        self.backends = {"backends": [{"id": "cuda-test", "vendor": "cuda"}]}
        self.source_digests = {
            "architecture_inventory_sha256": "1" * 64,
            "model_catalog_sha256": "2" * 64,
            "backend_catalog_sha256": "3" * 64,
        }
        self.candidate = {
            "release_subject": "v0.1.36-test",
            "core_commit": "0123456789abcdef0123456789abcdef01234567",
            "binary_sha256": "4" * 64,
            "plugin_sha256": "5" * 64,
        }

    def project(self) -> dict:
        return GATE.project_matrix(
            self.inventory,
            self.catalog,
            self.backends,
            source_digests=self.source_digests,
            candidate=self.candidate,
        )

    def test_projection_is_inventory_and_catalog_derived(self) -> None:
        matrix = self.project()
        self.assertEqual(matrix["schema"], GATE.SCHEMA)
        cells = {(cell["provider"], cell["model_id"], cell["quant"]) for cell in matrix["cells"]}
        self.assertEqual(len(cells), 6)
        cuda = next(cell for cell in matrix["cells"] if cell["provider"] == "cuda")
        self.assertEqual(cuda["backend_catalog_ids"], ["cuda-test"])
        self.assertEqual(cuda["kernel_coverage_bucket"]["members"], ["qwen3-asr-0.6b:q4_k"])
        self.assertEqual(cuda["output_plan"]["tie_policy"], "first_maximum")
        self.assertNotIn('"status": "pass"', json.dumps(matrix))

    def test_projection_requires_candidate_and_source_digests(self) -> None:
        with self.assertRaisesRegex(GATE.MatrixError, "candidate contract"):
            GATE.project_matrix(self.inventory, self.catalog, self.backends)
        with self.assertRaisesRegex(GATE.MatrixError, "canonical source digests"):
            GATE.project_matrix(self.inventory, self.catalog, self.backends, candidate=self.candidate)

    def test_public_family_missing_from_inventory_fails_closed(self) -> None:
        catalog = copy.deepcopy(self.catalog)
        catalog["models"][0]["family"] = "missing"
        with self.assertRaisesRegex(GATE.MatrixError, "missing from architecture"):
            GATE.project_matrix(
                self.inventory,
                catalog,
                self.backends,
                source_digests=self.source_digests,
                candidate=self.candidate,
            )

    def test_incomplete_receipts_fail_without_fabricating_hardware(self) -> None:
        matrix = self.project()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "receipt.json"
            path.write_text(json.dumps({"schema": GATE.RECEIPT_SCHEMA}))
            trace = Path(temp) / "trace.jsonl"
            trace.write_text(
                json.dumps({"schema": "openasr.gpu-correctness-trace.v1", "event": "header", "mode": "cold", "provider": "cpu", "device": "test-cpu"})
                + "\n"
                + json.dumps({"schema": "openasr.gpu-correctness-trace.v1", "event": "token", "step_index": 0, "token_id": 1})
                + "\n"
                + json.dumps({"schema": "openasr.gpu-correctness-trace.v1", "event": "top_k", "step_index": 0, "items": [{"token_id": 1, "value": 1.0}]})
                + "\n"
            )
            with self.assertRaisesRegex(GATE.MatrixError, "no versioned correctness evidence"):
                GATE.validate_matrix(
                    matrix,
                    [path],
                    inventory=self.inventory,
                    catalog=self.catalog,
                    backend_catalog=self.backends,
                    source_digests=self.source_digests,
                    trace_paths=[trace],
                )

    def receipt(self, cell: dict, evidence_class: str, mode: str) -> dict:
        token_content = (
            json.dumps({"schema": "openasr.gpu-correctness-trace.v1", "event": "header", "mode": mode, "provider": cell["provider"], "device": f"test-{cell['provider']}"})
            + "\n"
            + json.dumps({"schema": "openasr.gpu-correctness-trace.v1", "event": "token", "step_index": 0, "token_id": 7, "is_eot": 0})
            + "\n"
            + json.dumps({"schema": "openasr.gpu-correctness-trace.v1", "event": "top_k", "step_index": 0, "items": [{"token_id": 7, "value": 1.25}, {"token_id": 8, "value": 0.75}]})
            + "\n"
        )
        token_hash = hashlib.sha256(token_content.encode()).hexdigest()
        evidence = {
            "schema": GATE.EVIDENCE_SCHEMA,
            "contract": "openasr.gpu-correctness-artifact.v1",
            "evidence_class": evidence_class,
            "matrix_sha256": self.project()["matrix_sha256"],
            "candidate_release_subject": self.candidate["release_subject"],
            "core_commit": self.candidate["core_commit"],
            "catalog_digests": self.source_digests,
            "family": cell["family"],
            "model_id": cell["model_id"],
            "quant": cell["quant"],
            "topology": cell["topology"]["decoder_state"],
            "provider": cell["provider"],
            "device": f"test-{cell['provider']}",
            "placement": cell["placement"],
            "capture_mode": cell["capture_mode"],
            "scheduler_mode": cell["scheduler_mode"],
            "result": "pass",
            "artifacts": {
                "binary": {"label": "binary", "sha256": self.candidate["binary_sha256"]},
                "plugin": {"label": "plugin", "sha256": self.candidate["plugin_sha256"]},
                "pack": {"label": "pack", "sha256": "6" * 64},
                "fixture": {"label": "fixture", "sha256": "7" * 64},
            },
            "execution": {"mode": mode},
        }
        if evidence_class == "token_transcript":
            evidence.update(
                {
                    "output_plan": cell["output_plan"],
                    "family_oracle": {
                        "family": cell["family"],
                        "tie_policy": cell["output_plan"]["tie_policy"],
                    },
                    "trace": {
                        "token_trace": {"label": f"token-{cell['provider']}-{mode}.jsonl", "sha256": token_hash},
                        "logits": {"label": f"logits-{cell['provider']}-{mode}.jsonl", "sha256": token_hash},
                        "top_k": [{"token_id": 7, "value": 1.25}],
                        "top1_top2_margin": 0.5,
                    },
                }
            )
        return {
            "schema": GATE.RECEIPT_SCHEMA,
            "pack": {
                "model_id": f"{cell['model_id']}:{cell['quant']}",
                "quant": cell["quant"],
                "content_sha256": "6" * 64,
            },
            "audio": {"sha256": "7" * 64},
            "evidence": evidence,
        }

    def test_complete_receipts_close_every_exact_cell(self) -> None:
        matrix = self.project()
        with tempfile.TemporaryDirectory() as temp:
            paths: list[Path] = []
            for index, cell in enumerate(matrix["cells"]):
                for evidence_class in ("placement_resource", "token_transcript"):
                    for mode in ("cold", "reuse"):
                        path = Path(temp) / f"{index}-{evidence_class}-{mode}.json"
                        path.write_text(json.dumps(self.receipt(cell, evidence_class, mode)))
                        paths.append(path)
            build = Path(temp) / "build.json"
            build.write_text(
                json.dumps(
                    {
                        "schema": GATE.RECEIPT_SCHEMA,
                        "evidence": {
                            "schema": GATE.EVIDENCE_SCHEMA,
                            "contract": "openasr.gpu-correctness-artifact.v1",
                            "evidence_class": "build_packaging",
                            "matrix_sha256": matrix["matrix_sha256"],
                            "candidate_release_subject": self.candidate["release_subject"],
                            "core_commit": self.candidate["core_commit"],
                            "catalog_digests": self.source_digests,
                            "family": "release",
                            "model_id": "release",
                            "quant": "release",
                            "topology": "release",
                            "provider": "release",
                            "device": "release",
                            "placement": "not-applicable",
                            "capture_mode": "disabled",
                            "scheduler_mode": "disabled",
                            "result": "pass",
                            "artifacts": {
                                "binary": {"label": "binary", "sha256": self.candidate["binary_sha256"]},
                                "plugin": {"label": "plugin", "sha256": self.candidate["plugin_sha256"]},
                                "pack": {"label": "pack", "sha256": "6" * 64},
                                "fixture": {"label": "fixture", "sha256": "7" * 64},
                            },
                        },
                    }
                )
            )
            paths.append(build)
            trace_paths: list[Path] = []
            providers = sorted({cell["provider"] for cell in matrix["cells"]})
            for provider in providers:
                for mode in ("cold", "reuse"):
                    token_content = (
                        json.dumps({"schema": "openasr.gpu-correctness-trace.v1", "event": "header", "mode": mode, "provider": provider, "device": f"test-{provider}"})
                        + "\n"
                        + json.dumps({"schema": "openasr.gpu-correctness-trace.v1", "event": "token", "step_index": 0, "token_id": 7, "is_eot": 0})
                        + "\n"
                        + json.dumps({"schema": "openasr.gpu-correctness-trace.v1", "event": "top_k", "step_index": 0, "items": [{"token_id": 7, "value": 1.25}, {"token_id": 8, "value": 0.75}]})
                        + "\n"
                    )
                    token_trace = Path(temp) / f"token-{provider}-{mode}.jsonl"
                    token_trace.write_text(token_content)
                    trace_paths.append(token_trace)
                    logits_trace = Path(temp) / f"logits-{provider}-{mode}.jsonl"
                    logits_trace.write_text(token_content)
                    trace_paths.append(logits_trace)
            GATE.validate_matrix(
                matrix,
                paths,
                inventory=self.inventory,
                catalog=self.catalog,
                backend_catalog=self.backends,
                source_digests=self.source_digests,
                trace_paths=trace_paths,
            )

    def test_trace_semantic_forgery_is_rejected(self) -> None:
        matrix = self.project()
        cell = matrix["cells"][0]
        receipt = self.receipt(cell, "token_transcript", "cold")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "forged.json"
            path.write_text(json.dumps(receipt))
            trace = Path(temp) / receipt["evidence"]["trace"]["token_trace"]["label"]
            trace.write_text(
                json.dumps({"schema": "openasr.gpu-correctness-trace.v1", "event": "header", "mode": "cold", "provider": cell["provider"], "device": f"test-{cell['provider']}"})
                + "\n"
                + json.dumps({"schema": "openasr.gpu-correctness-trace.v1", "event": "token", "step_index": 0, "token_id": 7})
                + "\n"
            )
            with self.assertRaisesRegex(GATE.MatrixError, "matching per-step"):
                GATE.validate_matrix(
                    matrix,
                    [path],
                    inventory=self.inventory,
                    catalog=self.catalog,
                    backend_catalog=self.backends,
                    source_digests=self.source_digests,
                    trace_paths=[trace],
                )

        matrix = self.project()
        cell = matrix["cells"][0]
        receipt = self.receipt(cell, "token_transcript", "cold")
        receipt["evidence"]["matrix_sha256"] = "f" * 64
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "stale.json"
            path.write_text(json.dumps(receipt))
            trace = Path(temp) / "trace.jsonl"
            trace.write_text(
                json.dumps({"schema": "openasr.gpu-correctness-trace.v1", "event": "header", "mode": "cold", "provider": "cpu", "device": "test-cpu"})
                + "\n"
                + json.dumps({"schema": "openasr.gpu-correctness-trace.v1", "event": "token", "step_index": 0, "token_id": 1})
                + "\n"
                + json.dumps({"schema": "openasr.gpu-correctness-trace.v1", "event": "top_k", "step_index": 0, "items": [{"token_id": 1, "value": 1.0}]})
                + "\n"
            )
            with self.assertRaisesRegex(GATE.MatrixError, "stale"):
                GATE.validate_matrix(
                    matrix,
                    [path],
                    inventory=self.inventory,
                    catalog=self.catalog,
                    backend_catalog=self.backends,
                    source_digests=self.source_digests,
                    trace_paths=[trace],
                )

    def test_cpu_receipt_cannot_close_gpu_cell(self) -> None:
        matrix = self.project()
        cuda = next(cell for cell in matrix["cells"] if cell["provider"] == "cuda")
        cpu = next(cell for cell in matrix["cells"] if cell["provider"] == "cpu")
        self.assertEqual((cuda["model_id"], cuda["quant"]), (cpu["model_id"], cpu["quant"]))
        with tempfile.TemporaryDirectory() as temp:
            receipt = self.receipt(cpu, "token_transcript", "cold")
            path = Path(temp) / "cpu-as-gpu.json"
            path.write_text(json.dumps(receipt))
            token_label = receipt["evidence"]["trace"]["token_trace"]["label"]
            token_content = (
                json.dumps({"schema": "openasr.gpu-correctness-trace.v1", "event": "header", "mode": "cold", "provider": "cpu", "device": "test-cpu"})
                + "\n"
                + json.dumps({"schema": "openasr.gpu-correctness-trace.v1", "event": "token", "step_index": 0, "token_id": 7, "is_eot": 0})
                + "\n"
                + json.dumps({"schema": "openasr.gpu-correctness-trace.v1", "event": "top_k", "step_index": 0, "items": [{"token_id": 7, "value": 1.25}, {"token_id": 8, "value": 0.75}]})
                + "\n"
            )
            token_trace = Path(temp) / token_label
            token_trace.write_text(token_content)
            logits_label = receipt["evidence"]["trace"]["logits"]["label"]
            logits_trace = Path(temp) / logits_label
            logits_trace.write_text(token_content)
            with self.assertRaisesRegex(GATE.MatrixError, "incomplete|missing receipts"):
                GATE.validate_matrix(
                    matrix,
                    [path],
                    inventory=self.inventory,
                    catalog=self.catalog,
                    backend_catalog=self.backends,
                    source_digests=self.source_digests,
                    trace_paths=[token_trace, logits_trace],
                )

    def test_placement_receipt_cannot_close_token_cell(self) -> None:
        matrix = self.project()
        cell = next(item for item in matrix["cells"] if item["provider"] == "cuda")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "placement-only.json"
            path.write_text(json.dumps(self.receipt(cell, "placement_resource", "cold")))
            trace = Path(temp) / "trace.jsonl"
            trace.write_text(
                json.dumps({"schema": "openasr.gpu-correctness-trace.v1", "event": "header", "mode": "cold", "provider": "cuda", "device": "test-cuda"})
                + "\n"
                + json.dumps({"schema": "openasr.gpu-correctness-trace.v1", "event": "token", "step_index": 0, "token_id": 1})
                + "\n"
                + json.dumps({"schema": "openasr.gpu-correctness-trace.v1", "event": "top_k", "step_index": 0, "items": [{"token_id": 1, "value": 1.0}]})
                + "\n"
            )
            with self.assertRaisesRegex(GATE.MatrixError, "incomplete|missing receipts"):
                GATE.validate_matrix(
                    matrix,
                    [path],
                    inventory=self.inventory,
                    catalog=self.catalog,
                    backend_catalog=self.backends,
                    source_digests=self.source_digests,
                    trace_paths=[trace],
                )


if __name__ == "__main__":
    unittest.main()
