"""Fail-closed host and desktop evidence for the GPU decode correctness contract.

Missing CUDA/Vulkan/HIP hosts and a failed desktop plugin-switch log are not
passes. These tests drive the shipped gate functions against the committed
records.
"""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

PATH = Path(__file__).with_name("gpu_correctness_gate.py")
SPEC = importlib.util.spec_from_file_location("gpu_correctness_gate", PATH)
assert SPEC and SPEC.loader
GATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GATE)

REPO = Path(__file__).resolve().parents[2]
EVIDENCE = REPO / "docs/design/gpu-decode-correctness-evidence"
HARDWARE = EVIDENCE / "hardware-unavailable.txt"
DESKTOP = EVIDENCE / "desktop-plugin-switch.fail.txt"


class FailClosedEvidenceTests(unittest.TestCase):
    def test_committed_hardware_record_marks_cuda_vulkan_hip_unavailable(self) -> None:
        statuses = GATE.parse_hardware_unavailable(HARDWARE)
        self.assertEqual(
            statuses,
            {"cuda": "unavailable", "vulkan": "unavailable", "hip": "unavailable"},
        )

    def test_hardware_record_rejecting_a_fabricated_pass(self) -> None:
        with TemporaryDirectory() as temp:
            path = Path(temp) / "hardware-unavailable.txt"
            path.write_text(
                "windows_cuda: pass\nvulkan: unavailable\nhip: unavailable\nabsence is not a pass\n"
            )
            with self.assertRaisesRegex(GATE.MatrixError, "cuda"):
                GATE.parse_hardware_unavailable(path)

    def test_committed_desktop_plugin_switch_is_fail_not_skip(self) -> None:
        text = DESKTOP.read_text(encoding="utf-8")
        self.assertIn("result=FAIL", text)
        self.assertIn("skipped=false", text)
        with self.assertRaisesRegex(GATE.MatrixError, "not selectable"):
            GATE.require_desktop_plugin_switch(DESKTOP)

    def test_desktop_plugin_switch_skip_is_not_a_pass(self) -> None:
        with TemporaryDirectory() as temp:
            path = Path(temp) / "desktop-plugin-switch.log"
            path.write_text("result=FAIL\nskipped=true\nreason=no host\n")
            with self.assertRaisesRegex(GATE.MatrixError, "skip is not a pass"):
                GATE.require_desktop_plugin_switch(path)

    def test_untested_cuda_vulkan_hip_and_desktop_cells_are_not_selectable(self) -> None:
        inventory = {
            "schema": "openasr.model-family-inventory.v1",
            "families": [
                {
                    "catalog_family_id": "qwen",
                    "execution": {
                        "execution_capabilities": {
                            "cpu": True,
                            "providers": [
                                {"provider": "cuda", "full_device": True, "hybrid": False},
                                {"provider": "vulkan", "full_device": True, "hybrid": False},
                                {"provider": "hip", "full_device": True, "hybrid": False},
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
        catalog = {
            "models": [
                {
                    "id": "qwen3-asr-0.6b",
                    "family": "qwen",
                    "kind": "asr-model",
                    "public": True,
                    "recommended_quant": "q8_0",
                    "quants": [{"quant": "q8_0"}],
                }
            ]
        }
        backends = {
            "backends": [
                {
                    "id": "cuda-test",
                    "vendor": "cuda",
                    "version": "0.1.36",
                    "targets": ["sm_89"],
                    "host_abi": {"fingerprint": "8" * 64},
                    "files": [{"filename": "cuda.dll", "role": "plugin", "sha256": "5" * 64, "size_bytes": 1}],
                },
                {
                    "id": "vulkan-test",
                    "vendor": "vulkan",
                    "version": "0.1.36",
                    "targets": [],
                    "qualification_target": "vulkan-pci-10de-2820",
                    "host_abi": {"fingerprint": "8" * 64},
                    "files": [{"filename": "vulkan.dll", "role": "plugin", "sha256": "6" * 64, "size_bytes": 1}],
                },
                {
                    "id": "hip-test",
                    "vendor": "hip",
                    "version": "0.1.36",
                    "targets": ["gfx1200"],
                    "host_abi": {"fingerprint": "8" * 64},
                    "files": [{"filename": "hip.dll", "role": "plugin", "sha256": "7" * 64, "size_bytes": 1}],
                },
            ]
        }
        matrix = GATE.project_matrix(
            inventory,
            catalog,
            backends,
            source_digests={
                "architecture_inventory_sha256": "1" * 64,
                "model_catalog_sha256": "2" * 64,
                "backend_catalog_sha256": "3" * 64,
            },
            candidate={
                "release_subject": "v0.1.36-test",
                "release_version": "0.1.36",
                "core_commit": "0123456789abcdef0123456789abcdef01234567",
                "binary_sha256": "4" * 64,
            },
        )
        closed: set[GATE.ReceiptKey] = set()
        with self.assertRaisesRegex(GATE.MatrixError, "not selectable"):
            GATE.require_untested_hosts_not_activatable(matrix, closed, HARDWARE)
        with self.assertRaisesRegex(GATE.MatrixError, "not selectable"):
            GATE.require_desktop_plugin_switch(DESKTOP)


if __name__ == "__main__":
    unittest.main()
