from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

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
                                {"provider": "hip", "full_device": True, "hybrid": False},
                                {"provider": "vulkan", "full_device": True, "hybrid": False},
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
        self.backends = {"backends": [self.backend("cuda", "sm_75", "5")]}
        self.source_digests = {
            "architecture_inventory_sha256": "1" * 64,
            "model_catalog_sha256": "2" * 64,
            "backend_catalog_sha256": "3" * 64,
        }
        self.candidate = {
            "release_subject": "v0.1.36-test",
            "release_version": "0.1.36",
            "core_commit": "0123456789abcdef0123456789abcdef01234567",
            "binary_sha256": "4" * 64,
        }

    @staticmethod
    def backend(provider: str, target: str, sha_char: str) -> dict:
        value = {
            "id": f"{provider}-test-{target}",
            "vendor": provider,
            "version": "0.1.36",
            "targets": [] if provider == "vulkan" else [target],
            "host_abi": {"fingerprint": "8" * 64},
            "files": [
                {
                    "filename": f"ggml-{provider}-{target}.dll",
                    "role": "plugin",
                    "sha256": sha_char * 64,
                    "size_bytes": 10,
                }
            ],
        }
        if provider != "vulkan":
            value["min_driver_api"] = "12.0.0"
        return value

    def project(self, *, vulkan_targets: dict[str, str] | None = None) -> dict:
        return GATE.project_matrix(
            self.inventory,
            self.catalog,
            self.backends,
            source_digests=self.source_digests,
            candidate=self.candidate,
            vulkan_targets=vulkan_targets,
        )

    def test_projection_rejects_cross_provider_or_malformed_targets(self) -> None:
        for provider, target in (
            ("cuda", "gfx1100"),
            ("cuda", "sm_gfx1100"),
            ("hip", "sm_86"),
            ("hip", "gfx_sm86"),
        ):
            with self.subTest(provider=provider, target=target):
                self.backends = {"backends": [self.backend(provider, target, "5")]}
                with self.assertRaisesRegex(GATE.MatrixError, "non-canonical"):
                    self.project()

    @staticmethod
    def trace_label(cell: dict, mode: str, kind: str) -> str:
        return (
            f"{kind}-{cell['family']}-{cell['model_id']}-{cell['quant']}-"
            f"{cell['provider']}-{cell['device_target']}-{cell['backend_id']}-{mode}.jsonl"
        )

    @staticmethod
    def trace_content(
        cell: dict,
        mode: str,
        *,
        provider: str | None = None,
        device_target: str | None = None,
        backend_id: str | None = None,
        artifact_fingerprint: str | None = None,
    ) -> str:
        return (
            json.dumps(
                {
                    "schema": "openasr.gpu-correctness-trace.v1",
                    "event": "header",
                    "graph_mode": cell["graph_mode"],
                    "provider": provider or cell["provider"],
                    "device_target": device_target or cell["device_target"],
                    "backend_id": backend_id or cell["backend_id"],
                    "driver_version": "12.7.0",
                    "artifact_fingerprint": artifact_fingerprint
                    or cell["artifact_fingerprint"],
                    "device": f"test-{cell['provider']}",
                }
            )
            + "\n"
            + json.dumps(
                {
                    "schema": "openasr.gpu-correctness-trace.v1",
                    "event": "logits_digest",
                    "step_index": 0,
                    "element_count": 2,
                    "sha256": "a" * 64,
                    "non_finite_count": 0,
                }
            )
            + "\n"
            + json.dumps(
                {
                    "schema": "openasr.gpu-correctness-trace.v1",
                    "event": "token",
                    "step_index": 0,
                    "token_id": 7,
                    "is_eot": 0,
                }
            )
            + "\n"
            + json.dumps(
                {
                    "schema": "openasr.gpu-correctness-trace.v1",
                    "event": "top_k",
                    "step_index": 0,
                    "items": [
                        {"token_id": 7, "value": 1.25},
                        {"token_id": 8, "value": 0.75},
                    ],
                    "top1_top2_margin": 0.5,
                }
            )
            + "\n"
        )

    def receipt(self, cell: dict, evidence_class: str, mode: str) -> dict:
        token_content = self.trace_content(cell, mode)
        token_hash = hashlib.sha256(token_content.encode()).hexdigest()
        evidence = {
            "schema": GATE.EVIDENCE_SCHEMA,
            "contract": "openasr.gpu-correctness-artifact.v1",
            "evidence_class": evidence_class,
            "matrix_sha256": self.project()["matrix_sha256"],
            "candidate_release_subject": self.candidate["release_subject"],
            "core_commit": self.candidate["core_commit"],
            "catalog_digests": {
                "inventory_sha256": self.source_digests[
                    "architecture_inventory_sha256"
                ],
                "model_catalog_sha256": self.source_digests["model_catalog_sha256"],
                "backend_catalog_sha256": self.source_digests[
                    "backend_catalog_sha256"
                ],
            },
            "family": cell["family"],
            "model_id": cell["model_id"],
            "quant": cell["quant"],
            "topology": cell["topology"]["decoder_state"],
            "provider": cell["provider"],
            "device_target": cell["device_target"],
            "backend_id": cell["backend_id"],
            "driver_version": "12.7.0",
            "artifact_fingerprint": cell["artifact_fingerprint"],
            "device": f"test-{cell['provider']}",
            "placement": cell["placement"],
            "capture_mode": cell["capture_mode"],
            "scheduler_mode": cell["scheduler_mode"],
            "result": "pass",
            "artifacts": {
                "binary": {
                    "label": "binary",
                    "sha256": self.candidate["binary_sha256"],
                },
                "plugin": {"label": "plugin", "sha256": cell["plugin_sha256"]},
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
                        "token_trace": {
                            "label": self.trace_label(cell, mode, "token"),
                            "sha256": token_hash,
                        },
                        "logits": {
                            "label": self.trace_label(cell, mode, "logits"),
                            "sha256": token_hash,
                        },
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
            "run": {
                "warmup": "cold" if mode == "cold" else "warm",
                "cache_state": "empty" if mode == "cold" else "populated",
            },
            "decode_diagnostics": {
                "output_plan": cell["output_plan"]["kind"],
                "reuse_mode": cell["graph_mode"],
                "capability_evidence_revision": 1,
            },
            "evidence": evidence,
        }

    def build_receipt(self, matrix: dict) -> dict:
        return {
            "schema": GATE.RECEIPT_SCHEMA,
            "evidence": {
                "schema": GATE.EVIDENCE_SCHEMA,
                "contract": "openasr.gpu-correctness-artifact.v1",
                "evidence_class": "build_packaging",
                "matrix_sha256": matrix["matrix_sha256"],
                "candidate_release_subject": self.candidate["release_subject"],
                "core_commit": self.candidate["core_commit"],
                "catalog_digests": {
                    "inventory_sha256": self.source_digests[
                        "architecture_inventory_sha256"
                    ],
                    "model_catalog_sha256": self.source_digests[
                        "model_catalog_sha256"
                    ],
                    "backend_catalog_sha256": self.source_digests[
                        "backend_catalog_sha256"
                    ],
                },
                "family": "release",
                "model_id": "release",
                "quant": "release",
                "topology": "release",
                "provider": "release",
                "device_target": "release",
                "backend_id": "release",
                "driver_version": "0",
                "artifact_fingerprint": "8" * 64,
                "device": "release",
                "placement": "not-applicable",
                "capture_mode": "disabled",
                "scheduler_mode": "disabled",
                "result": "pass",
                "artifacts": {
                    "binary": {
                        "label": "binary",
                        "sha256": self.candidate["binary_sha256"],
                    },
                    "plugin": {"label": "plugin", "sha256": "5" * 64},
                    "pack": {"label": "pack", "sha256": "6" * 64},
                    "fixture": {"label": "fixture", "sha256": "7" * 64},
                },
            },
        }

    def bind_kwargs(self, traces: list[Path]) -> dict:
        return {
            "inventory": self.inventory,
            "catalog": self.catalog,
            "backend_catalog": self.backends,
            "source_digests": self.source_digests,
            "trace_paths": traces,
            "qualification_validator": lambda _paths: None,
        }

    def write_traces(self, root: Path, cell: dict, mode: str) -> list[Path]:
        content = self.trace_content(cell, mode)
        result = []
        for kind in ("token", "logits"):
            path = root / self.trace_label(cell, mode, kind)
            path.write_text(content)
            result.append(path)
        return result

    def test_projection_is_exact_target_and_catalog_derived(self) -> None:
        matrix = self.project()
        self.assertEqual(matrix["schema"], GATE.SCHEMA)
        self.assertEqual(len(matrix["cells"]), 2)
        cell = matrix["cells"][0]
        self.assertEqual(cell["provider"], "cuda")
        self.assertEqual(cell["device_target"], "sm_75")
        self.assertEqual(cell["backend_id"], "cuda-test-sm_75")
        self.assertEqual(cell["plugin_sha256"], "5" * 64)
        self.assertTrue(GATE._hex_digest(cell["artifact_fingerprint"]))
        self.assertNotIn("backend_catalog_ids", cell)
        self.assertNotIn('"status": "pass"', json.dumps(matrix))

    def test_projection_requires_candidate_sources_and_release_version(self) -> None:
        with self.assertRaisesRegex(GATE.MatrixError, "candidate contract"):
            GATE.project_matrix(self.inventory, self.catalog, self.backends)
        candidate = dict(self.candidate)
        candidate.pop("release_version")
        with self.assertRaisesRegex(GATE.MatrixError, "candidate contract"):
            GATE.project_matrix(
                self.inventory,
                self.catalog,
                self.backends,
                source_digests=self.source_digests,
                candidate=candidate,
            )
        with self.assertRaisesRegex(GATE.MatrixError, "canonical source digests"):
            GATE.project_matrix(
                self.inventory,
                self.catalog,
                self.backends,
                candidate=self.candidate,
            )

    def test_generic_vulkan_requires_one_explicit_capability_class(self) -> None:
        target = "vk_caps_00001002_0000744c_0123456789abcdef0123456789abcdef"
        self.backends["backends"].append(self.backend("vulkan", "generic", "a"))
        inert = self.project()
        self.assertFalse(any(cell["provider"] == "vulkan" for cell in inert["cells"]))
        self.assertEqual(inert["artifact_contract"]["vulkan_qualification_targets"], {})

        projected = self.project(
            vulkan_targets={"vulkan-test-generic": target}
        )
        vulkan_cells = [
            cell for cell in projected["cells"] if cell["provider"] == "vulkan"
        ]
        self.assertEqual(len(vulkan_cells), 2)
        self.assertEqual({cell["device_target"] for cell in vulkan_cells}, {target})
        self.assertEqual(
            inert["artifact_contract"]["backend_candidates_sha256"],
            projected["artifact_contract"]["backend_candidates_sha256"],
        )

        with self.assertRaisesRegex(GATE.MatrixError, "canonical vk_caps"):
            self.project(
                vulkan_targets={"vulkan-test-generic": "vk_caps_wrong"}
            )

    def test_activated_vulkan_target_must_exist_in_the_matrix(self) -> None:
        target = "vk_caps_00001002_0000744c_0123456789abcdef0123456789abcdef"
        self.backends["backends"].append(self.backend("vulkan", "generic", "a"))
        matrix = self.project()
        activation_catalog = copy.deepcopy(self.backends)
        activation_catalog["backends"][1]["activation"] = {
            "state": "activated",
            "qualification_source_catalog_sha256": self.source_digests[
                "backend_catalog_sha256"
            ],
            "hardware_evidence_sha256": "b" * 64,
            "qualified_device_target": target,
            "qualified_driver_version": "305419896",
            "correctness_matrix_sha256": matrix["matrix_sha256"],
            "correctness_receipts_sha256": GATE._canonical_sha([]),
        }
        with tempfile.TemporaryDirectory() as temp:
            build = Path(temp) / "build.json"
            build.write_text(json.dumps(self.build_receipt(matrix)))
            with self.assertRaisesRegex(GATE.MatrixError, "not projected"):
                GATE.validate_matrix(
                    matrix,
                    [build],
                    activation_catalog=activation_catalog,
                    **self.bind_kwargs([]),
                )

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

    def test_complete_receipts_close_every_exact_cell(self) -> None:
        matrix = self.project()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            receipts: list[Path] = []
            traces: list[Path] = []
            for index, cell in enumerate(matrix["cells"]):
                for evidence_class in ("placement_resource", "token_transcript"):
                    for mode in ("cold", "reuse"):
                        path = root / f"{index}-{evidence_class}-{mode}.json"
                        path.write_text(json.dumps(self.receipt(cell, evidence_class, mode)))
                        receipts.append(path)
                        if evidence_class == "token_transcript":
                            traces.extend(self.write_traces(root, cell, mode))
            build = root / "build.json"
            build.write_text(json.dumps(self.build_receipt(matrix)))
            receipts.append(build)
            activation_catalog = copy.deepcopy(self.backends)
            candidate = matrix["cells"][0]
            activation_catalog["backends"][0]["activation"] = {
                "state": "activated",
                "qualification_source_catalog_sha256": self.source_digests[
                    "backend_catalog_sha256"
                ],
                "hardware_evidence_sha256": "b" * 64,
                "qualified_device_target": candidate["device_target"],
                "qualified_driver_version": "12.7.0",
                "correctness_matrix_sha256": matrix["matrix_sha256"],
                "correctness_receipts_sha256": GATE.correctness_receipt_set_sha256(
                    receipts,
                    provider=candidate["provider"],
                    device_target=candidate["device_target"],
                    backend_id=candidate["backend_id"],
                ),
            }
            GATE.validate_matrix(
                matrix,
                receipts,
                activation_catalog=activation_catalog,
                **self.bind_kwargs(traces),
            )

    def test_published_inert_candidate_needs_build_receipt_but_not_runtime_receipts(self) -> None:
        matrix = self.project()
        with tempfile.TemporaryDirectory() as temp:
            build = Path(temp) / "build.json"
            build.write_text(json.dumps(self.build_receipt(matrix)))
            GATE.validate_matrix(
                matrix,
                [build],
                inventory=self.inventory,
                catalog=self.catalog,
                backend_catalog=self.backends,
                source_digests=self.source_digests,
                trace_paths=[],
                qualification_validator=lambda _paths: None,
            )
            cell = matrix["cells"][0]
            with self.assertRaisesRegex(GATE.MatrixError, "not selectable"):
                GATE.require_activation(
                    matrix,
                    set(),
                    provider=cell["provider"],
                    device_target=cell["device_target"],
                    backend_id=cell["backend_id"],
                    mode="auto",
                )

    def test_qualification_and_activation_are_two_exact_immutable_transitions(self) -> None:
        matrix = self.project()
        candidate = matrix["cells"][0]
        activation_catalog = copy.deepcopy(self.catalog)
        activation_catalog["backends"] = copy.deepcopy(self.backends["backends"])
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            receipts: list[Path] = []
            traces: list[Path] = []
            for index, cell in enumerate(matrix["cells"]):
                for evidence_class in ("placement_resource", "token_transcript"):
                    for mode in ("cold", "reuse"):
                        path = root / f"{index}-{evidence_class}-{mode}.json"
                        path.write_text(
                            json.dumps(self.receipt(cell, evidence_class, mode))
                        )
                        receipts.append(path)
                        if evidence_class == "token_transcript":
                            traces.extend(self.write_traces(root, cell, mode))
            build = root / "build.json"
            build.write_text(json.dumps(self.build_receipt(matrix)))
            receipts.append(build)

            entry = root / "backend-entry.json"
            entry.write_text(json.dumps(self.backends["backends"][0]))
            hardware_evidence = root / "backend-hardware-evidence-test.json"
            hardware_document = {
                "schema_version": 1,
                "result": "pass",
                "driver_version": "12.7.0",
            }
            hardware_evidence.write_text(json.dumps(hardware_document))
            raw = root / "backend-hardware-audit-test.json"
            raw.write_text("{}")
            expected_identity = (
                candidate["provider"],
                candidate["device_target"],
                candidate["backend_id"],
                candidate["artifact_fingerprint"],
                candidate["plugin_sha256"],
                self.candidate["release_version"],
                "12.7.0",
            )
            with (
                mock.patch(
                    "backend_hardware_evidence.approved_entry_paths",
                    return_value=[entry],
                ),
                mock.patch(
                    "backend_hardware_evidence._entry_identity",
                    return_value=(self.backends["backends"][0], mock.Mock(
                        backend_id=candidate["backend_id"], tuple=expected_identity
                    )),
                ),
                mock.patch(
                    "backend_hardware_evidence._common_evidence_identity",
                    return_value=(hardware_document, expected_identity),
                ),
                mock.patch("backend_hardware_evidence.verify_catalog_policy"),
            ):
                qualified = GATE.qualify_catalog_backend(
                    matrix=matrix,
                    source_backend_catalog=self.backends,
                    current_activation_catalog=activation_catalog,
                    source_digests=self.source_digests,
                    backend_id=candidate["backend_id"],
                    entry_paths=[entry],
                    hardware_evidence_paths=[hardware_evidence],
                    hardware_raw_audit_paths=[raw],
                )
                qualification = qualified["backends"][0]["activation"]
                self.assertEqual(qualification["state"], "qualified")
                self.assertNotIn("correctness_matrix_sha256", qualification)
                with self.assertRaisesRegex(
                    GATE.MatrixError, "must be independently qualified"
                ):
                    GATE.activate_catalog_backend(
                        matrix=matrix,
                        inventory=self.inventory,
                        model_catalog=self.catalog,
                        source_backend_catalog=self.backends,
                        current_activation_catalog=activation_catalog,
                        source_digests=self.source_digests,
                        backend_id=candidate["backend_id"],
                        entry_paths=[entry],
                        hardware_evidence_paths=[hardware_evidence],
                        hardware_raw_audit_paths=[raw],
                        receipt_paths=receipts,
                        trace_paths=traces,
                        qualification_validator=lambda _paths: None,
                    )
                tampered_qualified = copy.deepcopy(qualified)
                tampered_qualified["backends"][0]["activation"][
                    "qualified_driver_version"
                ] = "12.8.0"
                with self.assertRaisesRegex(
                    GATE.MatrixError, "qualified state has different bindings"
                ):
                    GATE.activate_catalog_backend(
                        matrix=matrix,
                        inventory=self.inventory,
                        model_catalog=self.catalog,
                        source_backend_catalog=self.backends,
                        current_activation_catalog=tampered_qualified,
                        source_digests=self.source_digests,
                        backend_id=candidate["backend_id"],
                        entry_paths=[entry],
                        hardware_evidence_paths=[hardware_evidence],
                        hardware_raw_audit_paths=[raw],
                        receipt_paths=receipts,
                        trace_paths=traces,
                        qualification_validator=lambda _paths: None,
                    )
                activated = GATE.activate_catalog_backend(
                    matrix=matrix,
                    inventory=self.inventory,
                    model_catalog=self.catalog,
                    source_backend_catalog=self.backends,
                    current_activation_catalog=qualified,
                    source_digests=self.source_digests,
                    backend_id=candidate["backend_id"],
                    entry_paths=[entry],
                    hardware_evidence_paths=[hardware_evidence],
                    hardware_raw_audit_paths=[raw],
                    receipt_paths=receipts,
                    trace_paths=traces,
                    qualification_validator=lambda _paths: None,
                )
                changed_models = copy.deepcopy(qualified)
                changed_models["models"].append(
                    {
                        "id": "untested-model",
                        "family": "qwen",
                        "kind": "asr-model",
                        "public": True,
                    }
                )
                with self.assertRaisesRegex(
                    GATE.MatrixError, "current activation catalog model projection"
                ):
                    GATE.activate_catalog_backend(
                        matrix=matrix,
                        inventory=self.inventory,
                        model_catalog=self.catalog,
                        source_backend_catalog=self.backends,
                        current_activation_catalog=changed_models,
                        source_digests=self.source_digests,
                        backend_id=candidate["backend_id"],
                        entry_paths=[entry],
                        hardware_evidence_paths=[hardware_evidence],
                        hardware_raw_audit_paths=[raw],
                        receipt_paths=receipts,
                        trace_paths=traces,
                        qualification_validator=lambda _paths: None,
                    )
                GATE.verify_catalog_backend_transition(
                    matrix=matrix,
                    inventory=self.inventory,
                    model_catalog=self.catalog,
                    source_backend_catalog=self.backends,
                    current_activation_catalog=activation_catalog,
                    candidate_activation_catalog=activated,
                    source_digests=self.source_digests,
                    backend_id=candidate["backend_id"],
                    entry_paths=[entry],
                    hardware_evidence_paths=[hardware_evidence],
                    hardware_raw_audit_paths=[raw],
                    receipt_paths=receipts,
                    trace_paths=traces,
                    qualification_validator=lambda _paths: None,
                )
                untested_candidate = copy.deepcopy(activated)
                untested_candidate["models"].append(
                    {
                        "id": "untested-model",
                        "family": "qwen",
                        "kind": "asr-model",
                        "public": True,
                    }
                )
                with self.assertRaisesRegex(
                    GATE.MatrixError, "candidate activation catalog model projection"
                ):
                    GATE.verify_catalog_backend_transition(
                        matrix=matrix,
                        inventory=self.inventory,
                        model_catalog=self.catalog,
                        source_backend_catalog=self.backends,
                        current_activation_catalog=activation_catalog,
                        candidate_activation_catalog=untested_candidate,
                        source_digests=self.source_digests,
                        backend_id=candidate["backend_id"],
                        entry_paths=[entry],
                        hardware_evidence_paths=[hardware_evidence],
                        hardware_raw_audit_paths=[raw],
                        receipt_paths=receipts,
                        trace_paths=traces,
                        qualification_validator=lambda _paths: None,
                    )
                unrelated = copy.deepcopy(activated)
                unrelated["backends"][0]["description"] = "unreviewed mutation"
                with self.assertRaisesRegex(
                    GATE.MatrixError, "exact qualify-then-activate replay"
                ):
                    GATE.verify_catalog_backend_transition(
                        matrix=matrix,
                        inventory=self.inventory,
                        model_catalog=self.catalog,
                        source_backend_catalog=self.backends,
                        current_activation_catalog=activation_catalog,
                        candidate_activation_catalog=unrelated,
                        source_digests=self.source_digests,
                        backend_id=candidate["backend_id"],
                        entry_paths=[entry],
                        hardware_evidence_paths=[hardware_evidence],
                        hardware_raw_audit_paths=[raw],
                        receipt_paths=receipts,
                        trace_paths=traces,
                        qualification_validator=lambda _paths: None,
                    )
            activation = activated["backends"][0]["activation"]
            self.assertEqual(activation["state"], "activated")
            self.assertEqual(
                activation["qualification_source_catalog_sha256"],
                self.source_digests["backend_catalog_sha256"],
            )
            self.assertEqual(
                activation["correctness_receipts_sha256"],
                GATE.correctness_receipt_set_sha256(
                    receipts,
                    provider=candidate["provider"],
                    device_target=candidate["device_target"],
                    backend_id=candidate["backend_id"],
                ),
            )

    def test_revocation_is_one_way_exact_and_preserves_audit_bindings(self) -> None:
        current = copy.deepcopy(self.catalog)
        current["backends"] = copy.deepcopy(self.backends["backends"])
        bindings = {
            "state": "activated",
            "qualification_source_catalog_sha256": "1" * 64,
            "hardware_evidence_sha256": "2" * 64,
            "qualified_device_target": "sm_75",
            "qualified_driver_version": "12.7.0",
            "correctness_matrix_sha256": "3" * 64,
            "correctness_receipts_sha256": "4" * 64,
        }
        current["backends"][0]["activation"] = bindings

        revoked = GATE.revoke_catalog_backend(
            current_activation_catalog=current,
            backend_id="cuda-test-sm_75",
        )
        self.assertEqual(revoked["backends"][0]["activation"]["state"], "revoked")
        self.assertEqual(
            revoked["backends"][0]["activation"]["correctness_receipts_sha256"],
            bindings["correctness_receipts_sha256"],
        )
        GATE.verify_catalog_backend_revocation(
            current_activation_catalog=current,
            candidate_activation_catalog=revoked,
            backend_id="cuda-test-sm_75",
        )
        GATE.verify_catalog_backend_revocation(
            current_activation_catalog=revoked,
            candidate_activation_catalog=revoked,
            backend_id="cuda-test-sm_75",
        )

        unrelated = copy.deepcopy(revoked)
        unrelated["models"].append({"id": "unreviewed"})
        with self.assertRaisesRegex(GATE.MatrixError, "exact revocation replay"):
            GATE.verify_catalog_backend_revocation(
                current_activation_catalog=current,
                candidate_activation_catalog=unrelated,
                backend_id="cuda-test-sm_75",
            )
        partial = copy.deepcopy(revoked)
        partial["backends"][0]["activation"].pop("correctness_receipts_sha256")
        with self.assertRaisesRegex(GATE.MatrixError, "bindings are incomplete"):
            GATE.revoke_catalog_backend(
                current_activation_catalog=partial,
                backend_id="cuda-test-sm_75",
            )
        with self.assertRaisesRegex(GATE.MatrixError, "cannot be requalified"):
            with mock.patch(
                "backend_hardware_evidence.approved_entry_paths", return_value=[]
            ):
                GATE.qualify_catalog_backend(
                    matrix=self.project(),
                    source_backend_catalog=self.backends,
                    current_activation_catalog=revoked,
                    source_digests=self.source_digests,
                    backend_id="cuda-test-sm_75",
                    entry_paths=[],
                    hardware_evidence_paths=[],
                    hardware_raw_audit_paths=[],
                )

    def test_bind_cell_derives_identity_from_runtime_and_cpu_oracle(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            binary = root / "openasr.exe"
            plugin = root / "ggml-cuda.dll"
            pack = root / "qwen.oasr"
            fixture = root / "jfk.wav"
            for path, contents in (
                (binary, b"binary"),
                (plugin, b"plugin"),
                (pack, b"pack"),
                (fixture, b"audio"),
            ):
                path.write_bytes(contents)
            self.candidate["binary_sha256"] = hashlib.sha256(b"binary").hexdigest()
            self.backends["backends"][0]["files"][0]["sha256"] = hashlib.sha256(
                b"plugin"
            ).hexdigest()
            matrix = self.project()
            cell = matrix["cells"][0]

            gpu_trace = root / self.trace_label(cell, "cold", "gpu-correctness-trace")
            gpu_trace.write_text(self.trace_content(cell, "cold"))
            cpu_lines = [
                json.loads(line)
                for line in self.trace_content(cell, "cold").splitlines()
            ]
            cpu_lines[0].update(
                {
                    "provider": "cpu",
                    "device_target": "unqualified",
                    "backend_id": "unqualified",
                    "driver_version": "unqualified",
                    "artifact_fingerprint": "unqualified",
                    "device": "cpu0",
                }
            )
            cpu_trace = root / "cpu-oracle.jsonl"
            cpu_trace.write_text("\n".join(json.dumps(line) for line in cpu_lines) + "\n")

            gpu = self.receipt(cell, "token_transcript", "cold")
            gpu["evidence"] = None
            gpu["core_commit"] = self.candidate["core_commit"]
            gpu["pack"]["content_sha256"] = hashlib.sha256(b"pack").hexdigest()
            gpu["audio"]["sha256"] = hashlib.sha256(b"audio").hexdigest()
            gpu["transcript"] = {"text": "hello", "text_sha256": hashlib.sha256(b"hello").hexdigest()}
            gpu["placement"] = "cuda"
            gpu["run"].update({"backend": "native", "device": "cuda", "os": "windows"})
            gpu["observed_placement"] = {
                "direct_graph_computes": 1,
                "scheduler_graph_computes": 0,
                "observed_compute_nodes_by_backend": {"CUDA0": 10},
                "fallback_node_samples_by_backend": {},
            }
            cpu = copy.deepcopy(gpu)
            cpu["run"]["device"] = "cpu"
            cpu["placement"] = "cpu"
            gpu_receipt = root / "gpu.json"
            cpu_receipt = root / "cpu.json"
            gpu_receipt.write_text(json.dumps(gpu))
            cpu_receipt.write_text(json.dumps(cpu))

            placement, token = GATE.bind_runtime_cell_receipts(
                matrix=matrix,
                inventory=self.inventory,
                model_catalog=self.catalog,
                backend_catalog=self.backends,
                source_digests=self.source_digests,
                backend_id=cell["backend_id"],
                process_mode="cold",
                gpu_receipt_path=gpu_receipt,
                gpu_trace_path=gpu_trace,
                cpu_receipt_path=cpu_receipt,
                cpu_trace_path=cpu_trace,
                binary_path=binary,
                plugin_path=plugin,
                pack_path=pack,
                fixture_path=fixture,
            )
            self.assertEqual(placement["evidence"]["evidence_class"], "placement_resource")
            self.assertEqual(token["evidence"]["evidence_class"], "token_transcript")
            self.assertEqual(token["evidence"]["device_target"], "sm_75")
            self.assertEqual(token["evidence"]["trace"]["logits"]["label"], gpu_trace.name)

            cpu_lines[2]["token_id"] = 99
            cpu_trace.write_text("\n".join(json.dumps(line) for line in cpu_lines) + "\n")
            with self.assertRaisesRegex(GATE.MatrixError, "CPU family oracle"):
                GATE.bind_runtime_cell_receipts(
                    matrix=matrix,
                    inventory=self.inventory,
                    model_catalog=self.catalog,
                    backend_catalog=self.backends,
                    source_digests=self.source_digests,
                    backend_id=cell["backend_id"],
                    process_mode="cold",
                    gpu_receipt_path=gpu_receipt,
                    gpu_trace_path=gpu_trace,
                    cpu_receipt_path=cpu_receipt,
                    cpu_trace_path=cpu_trace,
                    binary_path=binary,
                    plugin_path=plugin,
                    pack_path=pack,
                    fixture_path=fixture,
                )

    def test_sm89_receipts_cannot_close_sm75_lane(self) -> None:
        self.backends["backends"].append(self.backend("cuda", "sm_89", "9"))
        matrix = self.project()
        sm75 = next(cell for cell in matrix["cells"] if cell["device_target"] == "sm_75")
        sm89 = next(
            cell
            for cell in matrix["cells"]
            if cell["device_target"] == "sm_89"
            and cell["model_id"] == sm75["model_id"]
            and cell["quant"] == sm75["quant"]
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            receipt = self.receipt(sm89, "token_transcript", "cold")
            path = root / "sm89.json"
            path.write_text(json.dumps(receipt))
            traces = self.write_traces(root, sm89, "cold")
            closed, _ = GATE.closed_receipt_keys(
                matrix, [path], **self.bind_kwargs(traces)
            )
            self.assertTrue(any(key[4:6] == ("sm_89", sm89["backend_id"]) for key in closed))
            self.assertFalse(any(key[4:6] == ("sm_75", sm75["backend_id"]) for key in closed))
            modes = GATE.lane_activation_modes(matrix, closed)
            sm75_lane = (
                sm75["family"], sm75["model_id"], sm75["quant"], sm75["provider"],
                sm75["device_target"], sm75["backend_id"],
            )
            self.assertEqual(modes[sm75_lane], ())
            with self.assertRaisesRegex(GATE.MatrixError, "not selectable"):
                GATE.require_activation(
                    matrix, closed, provider="cuda", device_target="sm_75",
                    backend_id=sm75["backend_id"], mode="explicit",
                )

    def test_trace_target_or_backend_relabel_is_rejected(self) -> None:
        matrix = self.project()
        cell = matrix["cells"][0]
        receipt = self.receipt(cell, "token_transcript", "cold")
        forged = self.trace_content(cell, "cold", device_target="sm_89")
        forged_sha = hashlib.sha256(forged.encode()).hexdigest()
        receipt["evidence"]["trace"]["token_trace"]["sha256"] = forged_sha
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            receipt_path = root / "receipt.json"
            receipt_path.write_text(json.dumps(receipt))
            token = root / receipt["evidence"]["trace"]["token_trace"]["label"]
            token.write_text(forged)
            logits = root / receipt["evidence"]["trace"]["logits"]["label"]
            logits.write_text(self.trace_content(cell, "cold"))
            with self.assertRaisesRegex(GATE.MatrixError, "trace header does not match"):
                GATE.closed_receipt_keys(
                    matrix, [receipt_path], **self.bind_kwargs([token, logits])
                )

    def test_wrong_artifact_fingerprint_is_rejected(self) -> None:
        matrix = self.project()
        cell = matrix["cells"][0]
        receipt = self.receipt(cell, "placement_resource", "cold")
        receipt["evidence"]["artifact_fingerprint"] = "f" * 64
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "receipt.json"
            path.write_text(json.dumps(receipt))
            trace = root / "dummy.jsonl"
            trace.write_text(self.trace_content(cell, "cold"))
            with self.assertRaisesRegex(GATE.MatrixError, "artifact fingerprint"):
                GATE.closed_receipt_keys(matrix, [path], **self.bind_kwargs([trace]))

    def test_placement_receipt_cannot_close_token_cell(self) -> None:
        matrix = self.project()
        cell = matrix["cells"][0]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            receipt = self.receipt(cell, "placement_resource", "cold")
            path = root / "placement.json"
            path.write_text(json.dumps(receipt))
            trace = root / "dummy.jsonl"
            trace.write_text(self.trace_content(cell, "cold"))
            closed, _ = GATE.closed_receipt_keys(
                matrix, [path], **self.bind_kwargs([trace])
            )
            self.assertTrue(any(key[7] == "placement_resource" for key in closed))
            self.assertFalse(any(key[7] == "token_transcript" for key in closed))

    def test_core_qualification_rejection_precedes_matrix_closure(self) -> None:
        matrix = self.project()
        cell = matrix["cells"][0]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "receipt.json"
            path.write_text(json.dumps(self.receipt(cell, "placement_resource", "cold")))
            trace = root / "dummy.jsonl"
            trace.write_text(self.trace_content(cell, "cold"))

            def reject(_paths: list[Path]) -> None:
                raise GATE.MatrixError("strict core predicate rejected incomplete timing")

            kwargs = self.bind_kwargs([trace])
            kwargs["qualification_validator"] = reject
            with self.assertRaisesRegex(GATE.MatrixError, "incomplete timing"):
                GATE.closed_receipt_keys(matrix, [path], **kwargs)

    def test_default_validator_invokes_core_owned_cli_once(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            receipt = Path(temp) / "receipt.json"
            receipt.write_text("{}")
            completed = subprocess.CompletedProcess([], 0, b"ok", b"")
            with mock.patch.object(GATE.subprocess, "run", return_value=completed) as run:
                GATE._core_validate_qualification_receipts([receipt])
            command = run.call_args.args[0]
            self.assertIn("openasr-cli", command)
            self.assertIn("validate-qualification", command)
            self.assertEqual(command.count("--receipt"), 1)

    def test_incomplete_receipt_never_reaches_python_only_acceptance(self) -> None:
        matrix = self.project()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "receipt.json"
            path.write_text(json.dumps({"schema": GATE.RECEIPT_SCHEMA}))
            trace = root / "trace.jsonl"
            trace.write_text(self.trace_content(matrix["cells"][0], "cold"))
            with self.assertRaisesRegex(GATE.MatrixError, "no versioned correctness evidence"):
                GATE.validate_matrix(matrix, [path], **self.bind_kwargs([trace]))


if __name__ == "__main__":
    unittest.main()
