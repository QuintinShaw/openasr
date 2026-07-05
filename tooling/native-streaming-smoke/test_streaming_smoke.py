#!/usr/bin/env python3
"""Unit tests for the native streaming smoke helper.

The real smoke still has to run against local runtime packs. These tests keep
the evidence writer and capability gates deterministic without model I/O.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).with_name("streaming_smoke.py")
SPEC = importlib.util.spec_from_file_location("streaming_smoke", SCRIPT_PATH)
assert SPEC is not None
streaming_smoke = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules["streaming_smoke"] = streaming_smoke
SPEC.loader.exec_module(streaming_smoke)


class NativeStreamingSmokeTests(unittest.TestCase):
    def test_parse_args_accepts_summary_outputs(self) -> None:
        args = streaming_smoke.parse_args(
            ["--summary-json", "tmp/summary.json", "--summary-md", "tmp/summary.md"]
        )

        self.assertEqual(args.summary_json, "tmp/summary.json")
        self.assertEqual(args.summary_md, "tmp/summary.md")

    def test_selected_families_accepts_subset_and_rejects_empty(self) -> None:
        selected = streaming_smoke.selected_families("qwen, moonshine")

        self.assertEqual([family.name for family in selected], ["qwen", "moonshine"])
        with self.assertRaisesRegex(SystemExit, "--families must name"):
            streaming_smoke.selected_families(" , ")

    def test_parse_pack_overrides_requires_known_family_mapping(self) -> None:
        self.assertEqual(
            streaming_smoke.parse_pack_overrides(["qwen=/tmp/qwen.oasr"]),
            {"qwen": "/tmp/qwen.oasr"},
        )
        with self.assertRaisesRegex(SystemExit, "FAMILY=PATH"):
            streaming_smoke.parse_pack_overrides(["qwen"])
        with self.assertRaisesRegex(SystemExit, "unknown family"):
            streaming_smoke.parse_pack_overrides(["unknown=/tmp/x.oasr"])
        with self.assertRaisesRegex(SystemExit, "more than once"):
            streaming_smoke.parse_pack_overrides(["qwen=/tmp/a.oasr", "qwen=/tmp/b.oasr"])

    def test_strict_release_evidence_requires_summary_and_selected_packs(self) -> None:
        [qwen] = streaming_smoke.selected_families("qwen")
        missing_summary_json = streaming_smoke.parse_args(
            [
                "--families",
                "qwen",
                "--pack",
                "qwen=/tmp/qwen.oasr",
                "--summary-md",
                "tmp/evidence.md",
                "--strict-release-evidence",
            ]
        )
        with self.assertRaisesRegex(SystemExit, "--summary-json is required"):
            streaming_smoke.validate_strict_release_evidence_args(
                missing_summary_json,
                [qwen],
                {"qwen": "/tmp/qwen.oasr"},
            )

        missing_pack = streaming_smoke.parse_args(
            [
                "--families",
                "qwen",
                "--summary-json",
                "tmp/evidence.json",
                "--strict-release-evidence",
            ]
        )
        with self.assertRaisesRegex(SystemExit, "requires --pack"):
            streaming_smoke.validate_strict_release_evidence_args(missing_pack, [qwen], {})

        complete = streaming_smoke.parse_args(
            [
                "--families",
                "qwen",
                "--pack",
                "qwen=/tmp/qwen.oasr",
                "--summary-json",
                "tmp/evidence.json",
                "--build-id",
                "build-123",
                "--strict-release-evidence",
            ]
        )
        streaming_smoke.validate_strict_release_evidence_args(
            complete,
            [qwen],
            {"qwen": "/tmp/qwen.oasr"},
        )

        missing_build = streaming_smoke.parse_args(
            [
                "--families",
                "qwen",
                "--pack",
                "qwen=/tmp/qwen.oasr",
                "--summary-json",
                "tmp/evidence.json",
                "--strict-release-evidence",
            ]
        )
        with self.assertRaisesRegex(SystemExit, "--build-id is required"):
            streaming_smoke.validate_strict_release_evidence_args(
                missing_build,
                [qwen],
                {"qwen": "/tmp/qwen.oasr"},
            )

        temporary_pack = streaming_smoke.parse_args(
            [
                "--families",
                "qwen",
                "--pack",
                "qwen=/tmp/qwen3-asr-0.6b-q4_k.streaming.oasr",
                "--summary-json",
                "tmp/evidence.json",
                "--build-id",
                "build-123",
                "--strict-release-evidence",
            ]
        )
        with self.assertRaisesRegex(SystemExit, "temporary local streaming packs"):
            streaming_smoke.validate_strict_release_evidence_args(
                temporary_pack,
                [qwen],
                {"qwen": "/tmp/qwen3-asr-0.6b-q4_k.streaming.oasr"},
            )

    def test_check_inspect_output_requires_true_streaming_capability(self) -> None:
        output = "\n".join(
            [
                "Model identity: qwen3-asr-0.6b:q4_k (openasr.model.id)",
                "- openasr.model.family: qwen3-asr",
                "- mode: true_streaming",
                "- supports_partial_results: true",
                "- is_true_streaming: true",
            ]
        )

        streaming_smoke.check_inspect_output(output, Path("pack.oasr"))
        self.assertEqual(
            streaming_smoke.check_runtime_family(
                output,
                streaming_smoke.FAMILIES[0],
                Path("qwen.oasr"),
            ),
            ("qwen3-asr-0.6b:q4_k", "qwen3-asr"),
        )
        with self.assertRaisesRegex(SystemExit, "did not advertise true-streaming"):
            streaming_smoke.check_inspect_output("- mode: file_per_utterance", Path("pack.oasr"))
        with self.assertRaisesRegex(SystemExit, "reported runtime family"):
            streaming_smoke.check_runtime_family(
                output.replace("qwen3-asr", "whisper"),
                streaming_smoke.FAMILIES[0],
                Path("qwen.oasr"),
            )
        with self.assertRaisesRegex(SystemExit, "did not report Model identity"):
            streaming_smoke.check_runtime_family(
                "- openasr.model.family: qwen3-asr",
                streaming_smoke.FAMILIES[0],
                Path("qwen.oasr"),
            )

    def test_final_text_from_line_strips_smoke_prefix(self) -> None:
        self.assertEqual(
            streaming_smoke.final_text_from_line(
                "native streaming smoke final text (4000 ms): And so my fellow Americans asked"
            ),
            "And so my fellow Americans asked",
        )

    def test_relative_path_or_name_redacts_external_absolute_paths(self) -> None:
        repo_root = Path("/repo")

        self.assertEqual(
            streaming_smoke.relative_path_or_name(repo_root, Path("/repo/tmp/native")),
            "tmp/native",
        )
        self.assertEqual(
            streaming_smoke.relative_path_or_name(repo_root, Path("/private/audio")),
            "audio",
        )

    def test_build_summary_is_redacted_and_structured(self) -> None:
        repo_root = Path("/repo")
        family = streaming_smoke.Family(
            name="qwen",
            source="tmp/models/qwen3-asr/Qwen-source",
            output_name="qwen.oasr",
            import_args=(),
        )
        result = streaming_smoke.FamilySmokeResult(
            family="qwen",
            source=family.source,
            pack_file="qwen.oasr",
            pack_sha256="a" * 64,
            pack_size_bytes=1234,
            model_identity="qwen3-asr-0.6b:q4_k",
            runtime_family="qwen3-asr",
            final_text_line="native streaming smoke final text: hello",
            final_text="hello",
        )

        summary = streaming_smoke.build_summary(
            repo_root=repo_root,
            audio=Path("/private/openasr-secret-audio/jfk.wav"),
            workdir=Path("/repo/tmp/native-streaming-smoke"),
            families=[family],
            max_ms=4000,
            skip_import=True,
            build_id="build-123",
            results=[result],
        )
        rendered = json.dumps(summary, sort_keys=True)

        self.assertEqual(summary["probe"], "native_streaming_smoke")
        self.assertEqual(summary["audio_file"], "jfk.wav")
        self.assertEqual(summary["workdir"], "tmp/native-streaming-smoke")
        self.assertFalse(summary["strict_release_evidence"])
        self.assertEqual(summary["build"], {"runner": "build-123"})
        self.assertEqual(summary["families_requested"], ["qwen"])
        self.assertEqual(summary["results"][0]["final_text_chars"], 5)
        self.assertEqual(summary["results"][0]["pack_origin"], "generated")
        self.assertEqual(summary["results"][0]["pack_sha256"], "a" * 64)
        self.assertEqual(summary["results"][0]["pack_size_bytes"], 1234)
        self.assertEqual(summary["results"][0]["model_identity"], "qwen3-asr-0.6b:q4_k")
        self.assertEqual(summary["results"][0]["runtime_family"], "qwen3-asr")
        self.assertTrue(summary["results"][0]["inspect_true_streaming"])
        self.assertNotIn("openasr-secret-audio", rendered)
        self.assertNotIn("/private", rendered)

    def test_validation_summary_markdown_is_copyable_and_redacted(self) -> None:
        repo_root = Path("/repo")
        family = streaming_smoke.Family(
            name="qwen",
            source="tmp/models/qwen3-asr/Qwen-source",
            output_name="qwen.oasr",
            import_args=(),
        )
        result = streaming_smoke.FamilySmokeResult(
            family="qwen",
            source=family.source,
            pack_file="qwen.oasr",
            pack_sha256="a" * 64,
            pack_size_bytes=1234,
            model_identity="qwen3-asr-0.6b:q4_k",
            runtime_family="qwen3-asr",
            final_text_line="native streaming smoke final text: hello",
            final_text="hello `world`\nnext line",
        )
        summary = streaming_smoke.build_summary(
            repo_root=repo_root,
            audio=Path("/private/openasr-secret-audio/jfk.wav"),
            workdir=Path("/repo/tmp/native-streaming-smoke"),
            families=[family],
            max_ms=4000,
            skip_import=True,
            build_id="build-123",
            results=[result],
        )

        markdown = streaming_smoke.build_validation_markdown(summary)

        self.assertIn("Native streaming release-pack smoke evidence", markdown)
        self.assertIn("- audio fixture: `jfk.wav`", markdown)
        self.assertIn("- workdir: `tmp/native-streaming-smoke`", markdown)
        self.assertIn("- strict release evidence: `False`", markdown)
        self.assertIn("- runner build: `build-123`", markdown)
        self.assertIn("- families requested: `qwen`", markdown)
        self.assertIn("model `qwen3-asr-0.6b:q4_k`", markdown)
        self.assertIn("runtime family `qwen3-asr`", markdown)
        self.assertIn(
            "sha256 `aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa`",
            markdown,
        )
        self.assertIn("size `1234` bytes", markdown)
        self.assertIn("origin `generated`", markdown)
        self.assertIn("final 23 chars `hello 'world' next line`", markdown)
        self.assertNotIn("openasr-secret-audio", markdown)
        self.assertNotIn("/private", markdown)

    def test_smoke_family_can_use_provided_pack_without_source_or_import(self) -> None:
        inspect_output = "\n".join(
            [
                "Model identity: qwen3-asr-0.6b:q4_k (openasr.model.id)",
                "- openasr.model.family: qwen3-asr",
                "- mode: true_streaming",
                "- supports_partial_results: true",
                "- is_true_streaming: true",
            ]
        )
        family = streaming_smoke.Family(
            name="qwen",
            source="missing/source",
            output_name="unused.oasr",
            import_args=("model-pack", "import", "qwen", "{source}", "{output}"),
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pack = root / "release" / "qwen-release.oasr"
            pack.parent.mkdir()
            pack.write_bytes(b"pack")
            calls: list[list[str]] = []
            original = streaming_smoke.run_streaming

            def fake_run_streaming(command: list[str], **kwargs) -> str:
                calls.append(command)
                if command[1:2] == ["show"]:
                    return inspect_output
                if command[1:2] == ["verify"]:
                    return ""
                if command[:4] == ["cargo", "test", "-p", "openasr-core"]:
                    return "native streaming smoke final text (4000 ms): release text\n"
                raise AssertionError(f"unexpected command: {command}")

            try:
                streaming_smoke.run_streaming = fake_run_streaming
                with contextlib.redirect_stdout(io.StringIO()):
                    result = streaming_smoke.smoke_family(
                        repo_root=root,
                        openasr_bin=Path("openasr"),
                        workdir=root / "work",
                        audio=root / "audio.wav",
                        max_ms=4000,
                        family=family,
                        skip_import=False,
                        pack_override=str(pack),
                    )
            finally:
                streaming_smoke.run_streaming = original

            rendered_calls = " ".join(" ".join(command) for command in calls)
            self.assertEqual(result.pack_origin, "provided")
            self.assertEqual(result.source, "provided-pack")
            self.assertEqual(result.pack_file, "qwen-release.oasr")
            self.assertEqual(
                result.pack_sha256,
                "4862f447f2c7f272fa2f4aaf89dadb3b1ac09105bd5864f8d1a0c9452bb0a226",
            )
            self.assertEqual(result.pack_size_bytes, 4)
            self.assertEqual(result.model_identity, "qwen3-asr-0.6b:q4_k")
            self.assertEqual(result.runtime_family, "qwen3-asr")
            self.assertEqual(result.final_text, "release text")
            self.assertNotIn("import qwen", rendered_calls)
            self.assertIn(str(pack), rendered_calls)

    def test_write_summary_json_creates_parent_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "nested" / "summary.json"

            with contextlib.redirect_stdout(io.StringIO()):
                streaming_smoke.write_summary_json(str(path), {"schema_version": 1})

            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"schema_version": 1})

    def test_write_summary_markdown_creates_parent_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "nested" / "summary.md"

            with contextlib.redirect_stdout(io.StringIO()):
                streaming_smoke.write_summary_markdown(
                    str(path),
                    {
                        "schema_version": 1,
                        "probe": "native_streaming_smoke",
                        "audio_file": "jfk.wav",
                        "workdir": "tmp/native-streaming-smoke",
                        "max_ms": 4000,
                        "skip_import": True,
                        "families_requested": ["qwen"],
                        "results": [],
                    },
                )

            self.assertIn(
                "Native streaming release-pack smoke evidence",
                path.read_text(encoding="utf-8"),
            )


if __name__ == "__main__":
    unittest.main()
