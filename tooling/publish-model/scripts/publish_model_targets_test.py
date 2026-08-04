#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import publish_model_targets as publish


def _fake_run_with_receipts(commands: list[list[str]]):
    def fake_run(
        args: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None
    ) -> str:
        del cwd, env
        commands.append(args)
        if args[1:3] == ["model-pack", "preflight"]:
            source = Path(args[3])
            return json.dumps(
                {
                    "schema": publish.PREFLIGHT_RECEIPT_SCHEMA,
                    "content_id": "sha256:" + "a" * 64,
                    "size_bytes": source.stat().st_size,
                    "route": "asr",
                    "catalog_family_id": "qwen",
                    "model_family": "qwen3-asr",
                    "model_architecture": "qwen3-asr",
                    "build_commit": "d" * 40,
                }
            )
        if args == ["git", "rev-parse", "HEAD"]:
            return "d" * 40
        return ""

    return fake_run


class PublishModelTargetsTest(unittest.TestCase):
    def test_scope_is_limited_to_release_lane_and_full_quant_sets(self) -> None:
        publish.validate_scope(
            "qwen3-asr-0.6b", ["fp16", "q8_0", "q4_k"], ["fp16", "q8_0", "q4_k"]
        )
        # Published diarization support packs keep their catalog-declared variants.
        publish.validate_scope(
            "redimnet2-b6-cn", ["fp16", "q8_0", "f32"], ["fp16", "q8_0", "f32"]
        )
        publish.validate_scope("pyannote-segmentation-3.0", ["f32"], ["f32"])
        publish.validate_scope("diarizen-large-s80-v2", ["fp16"], ["fp16"])

        # Published support packs (moonshine, diarization, ...) are in the lane;
        # a model the lane does not list is rejected before any quant check.
        publish.validate_scope("moonshine-tiny", ["fp16", "q8_0"], ["fp16", "q8_0"])
        with self.assertRaisesRegex(SystemExit, "only publishes qwen3-asr-0.6b"):
            publish.validate_scope(
                "parakeet-ctc-0.6b", ["fp16", "q8_0", "q4_k"], ["fp16", "q8_0", "q4_k"]
            )
        # Partial publishes of the catalog-declared quant set are refused.
        with self.assertRaisesRegex(SystemExit, "must be exactly"):
            publish.validate_scope("qwen3-asr-0.6b", ["q8_0"], ["fp16", "q8_0", "q4_k"])
        with self.assertRaisesRegex(SystemExit, "must be exactly"):
            publish.validate_scope("redimnet2-b6-cn", ["fp16"], ["fp16", "q8_0", "f32"])

    def test_dry_run_builds_stage_without_writing_revision_sidecars(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            packs = root / "tmp" / "publish" / "qwen3-asr-0.6b" / "packs"
            repo = root / "tmp" / "publish" / "qwen3-asr-0.6b" / "repo"
            packs.mkdir(parents=True)
            repo.mkdir(parents=True)
            (repo / "README.md").write_text("HF README\n")
            for quant in publish.DEFAULT_QUANTS:
                pack = packs / f"qwen3-asr-0.6b-{quant}.oasr"
                pack.write_bytes(b"general.architecture\0qwen3-asr\0" + f"{quant}\n".encode())
                (packs / f"qwen3-asr-0.6b.{quant}.result.json").write_text(
                    json.dumps(
                        {
                            "pack": str(pack),
                            "size_bytes": pack.stat().st_size,
                            "sha256": "a" * 64,
                        }
                    )
                )

            entry = {
                "hf_repo": "OpenASR/qwen3-asr-0.6b",
                "family": "qwen",
                "kind": "asr-model",
                "release_public": True,
                "license_name": "Apache-2.0",
            }
            old_root = publish.REPO_ROOT
            commands: list[list[str]] = []
            try:
                publish.REPO_ROOT = root
                with (
                    mock.patch.object(
                        publish, "openasr_release_binary", return_value=root / "openasr"
                    ),
                    mock.patch.object(
                        publish, "run", side_effect=_fake_run_with_receipts(commands)
                    ),
                ):
                    publish.publish_hf(
                        "qwen3-asr-0.6b", entry, list(publish.DEFAULT_QUANTS), True
                    )
            finally:
                publish.REPO_ROOT = old_root

            work = root / "tmp" / "publish" / "qwen3-asr-0.6b"
            self.assertFalse((work / "hf_revision.txt").exists())

    def test_push_git_uses_explicit_remote_branch(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            stage = Path(temp)
            commands: list[list[str]] = []

            def fake_run(args: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> str:
                del cwd, env
                commands.append(args)
                if args == ["git", "rev-parse", "HEAD"]:
                    return "b" * 40
                return ""

            with mock.patch.object(publish, "run", side_effect=fake_run):
                self.assertEqual(publish.push_git(stage, "https://example.invalid/repo.git", False), "b" * 40)
                self.assertEqual(
                    publish.push_git(stage, "https://example.invalid/repo.git", False, branch="master"),
                    "b" * 40,
                )

            self.assertIn(["git", "push", "--force", "origin", "HEAD:main"], commands)
            self.assertIn(["git", "push", "--force", "origin", "HEAD:master"], commands)

    def test_commit_stage_can_skip_lfs_for_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            stage = Path(temp)
            commands: list[list[str]] = []

            def fake_run(args: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> str:
                del cwd, env
                commands.append(args)
                if args == ["git", "rev-parse", "HEAD"]:
                    return "c" * 40
                return ""

            with mock.patch.object(publish, "run", side_effect=fake_run):
                self.assertEqual(
                    publish.commit_stage(stage, "publish dry run", use_lfs=False),
                    "c" * 40,
                )

            self.assertNotIn(["git", "lfs", "install", "--local"], commands)

    def test_ensure_hf_repo_always_creates_private(self) -> None:
        # Fail-closed visibility: repo creation never takes a `public` bit.
        # Making a model's HF repo public is a separate, explicit step taken
        # only after the `_manifest.py --public` catalog-listing gate has
        # passed -- publish_hf must never do it implicitly.
        commands: list[list[str]] = []

        def fake_run(args: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> str:
            del cwd, env
            commands.append(args)
            return ""

        with mock.patch.object(publish, "run", side_effect=fake_run):
            publish.ensure_hf_repo("OpenASR/some-model", "tok", False)

        self.assertEqual(len(commands), 1)
        self.assertIn("--private", commands[0])
        self.assertNotIn("--public", commands[0])

    def test_ensure_hf_repo_signature_has_no_public_visibility_parameter(self) -> None:
        # Regression guard: `ensure_hf_repo` must not regain a `public`
        # parameter that a caller could wire back up to `release_public`.
        import inspect

        params = list(inspect.signature(publish.ensure_hf_repo).parameters)
        self.assertEqual(params, ["repo", "token", "dry_run"])

    def test_publish_hf_never_requests_a_public_repo_even_when_release_public_is_true(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            packs = root / "tmp" / "publish" / "qwen3-asr-0.6b" / "packs"
            repo = root / "tmp" / "publish" / "qwen3-asr-0.6b" / "repo"
            packs.mkdir(parents=True)
            repo.mkdir(parents=True)
            (repo / "README.md").write_text("HF README\n")
            for quant in publish.DEFAULT_QUANTS:
                pack = packs / f"qwen3-asr-0.6b-{quant}.oasr"
                pack.write_bytes(b"general.architecture\0qwen3-asr\0" + f"{quant}\n".encode())
                (packs / f"qwen3-asr-0.6b.{quant}.result.json").write_text(
                    json.dumps(
                        {
                            "pack": str(pack),
                            "size_bytes": pack.stat().st_size,
                            "sha256": "a" * 64,
                        }
                    )
                )

            entry = {
                "hf_repo": "OpenASR/qwen3-asr-0.6b",
                "family": "qwen",
                "kind": "asr-model",
                "release_public": True,
                "license_name": "Apache-2.0",
            }
            calls: list[list[str]] = []

            def fake_ensure_hf_repo(repo: str, token: str, dry_run: bool) -> None:
                calls.append([repo, token, str(dry_run)])

            old_root = publish.REPO_ROOT
            commands: list[list[str]] = []
            try:
                publish.REPO_ROOT = root
                with (
                    mock.patch.object(publish, "ensure_hf_repo", side_effect=fake_ensure_hf_repo),
                    mock.patch.object(
                        publish, "openasr_release_binary", return_value=root / "openasr"
                    ),
                    mock.patch.object(
                        publish, "run", side_effect=_fake_run_with_receipts(commands)
                    ),
                ):
                    publish.publish_hf(
                        "qwen3-asr-0.6b", entry, list(publish.DEFAULT_QUANTS), True
                    )
            finally:
                publish.REPO_ROOT = old_root

            # release_public=True on the catalog entry must not leak into any
            # HF-visibility argument -- ensure_hf_repo takes no such argument.
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][0], "OpenASR/qwen3-asr-0.6b")

    def test_copy_stage_uses_cli_receipt_and_does_not_copy_pack_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            packs = root / "tmp" / "publish" / "qwen3-asr-0.6b" / "packs"
            packs.mkdir(parents=True)
            quants = ["q8_0"]
            pack = packs / "qwen3-asr-0.6b-q8_0.oasr"
            pack.write_bytes(b"not scanned in Python")
            (packs / "qwen3-asr-0.6b.q8_0.result.json").write_text(
                json.dumps(
                    {
                        "pack": str(pack),
                        "size_bytes": pack.stat().st_size,
                        "sha256": "a" * 64,
                    }
                )
            )

            old_root = publish.REPO_ROOT
            commands: list[list[str]] = []
            try:
                publish.REPO_ROOT = root
                stage = root / "stage"
                with (
                    mock.patch.object(
                        publish, "openasr_release_binary", return_value=root / "openasr"
                    ),
                    mock.patch.object(
                        publish, "run", side_effect=_fake_run_with_receipts(commands)
                    ),
                ):
                    publish.copy_stage(
                        "qwen3-asr-0.6b",
                        {"family": "qwen"},
                        quants,
                        "README\n",
                        stage,
                    )
            finally:
                publish.REPO_ROOT = old_root

            self.assertEqual(len(commands), 1)
            self.assertEqual(
                commands[0][1:],
                [
                    "model-pack",
                    "preflight",
                    str(pack),
                    "--stage",
                    str(stage / pack.name),
                    "--json",
                ],
            )
            self.assertFalse((stage / pack.name).exists())
            self.assertEqual((stage / "README.md").read_text(), "README\n")
            self.assertTrue((stage / ".gitattributes").exists())

    def test_copy_stage_rejects_receipt_content_id_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            packs = root / "tmp" / "publish" / "qwen3-asr-0.6b" / "packs"
            packs.mkdir(parents=True)
            pack = packs / "qwen3-asr-0.6b-fp16.oasr"
            pack.write_bytes(b"receipt mismatch")
            (packs / "qwen3-asr-0.6b.fp16.result.json").write_text(
                json.dumps(
                    {
                        "pack": str(pack),
                        "size_bytes": pack.stat().st_size,
                        "sha256": "a" * 64,
                    }
                )
            )

            def mismatched_receipt(
                args: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None
            ) -> str:
                del cwd, env
                if args[1:3] == ["model-pack", "preflight"]:
                    destination = Path(args[5])
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(pack.read_bytes())
                    destination.chmod(0o444)
                    return json.dumps(
                        {
                            "schema": publish.PREFLIGHT_RECEIPT_SCHEMA,
                            "content_id": "sha256:" + "b" * 64,
                            "size_bytes": pack.stat().st_size,
                            "route": "asr",
                            "catalog_family_id": "qwen",
                            "build_commit": "d" * 40,
                        }
                    )
                return ""

            old_root = publish.REPO_ROOT
            try:
                publish.REPO_ROOT = root
                with (
                    mock.patch.object(
                        publish, "openasr_release_binary", return_value=root / "openasr"
                    ),
                    mock.patch.object(publish, "run", side_effect=mismatched_receipt),
                    self.assertRaisesRegex(SystemExit, "content_id mismatch"),
                ):
                    publish.copy_stage(
                        "qwen3-asr-0.6b",
                        {"family": "qwen"},
                        ["fp16"],
                        "README\n",
                        root / "stage",
                    )
            finally:
                publish.REPO_ROOT = old_root
            self.assertFalse((root / "stage" / pack.name).exists())

    def test_copy_stage_rejects_a_valid_pack_from_another_catalog_family(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "other-family.oasr"
            source.write_bytes(b"verified elsewhere")
            result = {
                "pack_path": source,
                "size_bytes": source.stat().st_size,
                "sha256": "a" * 64,
            }
            receipt = json.dumps(
                {
                    "schema": publish.PREFLIGHT_RECEIPT_SCHEMA,
                    "content_id": "sha256:" + "a" * 64,
                    "size_bytes": source.stat().st_size,
                    "route": "asr",
                    "catalog_family_id": "whisper",
                    "build_commit": "d" * 40,
                }
            )

            def wrong_family_receipt(
                args: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None
            ) -> str:
                del cwd, env
                destination = Path(args[5])
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(source.read_bytes())
                destination.chmod(0o444)
                return receipt

            stage = root / "stage"
            with (
                mock.patch.object(publish, "pack_result", return_value=result),
                mock.patch.object(publish, "openasr_release_binary", return_value=root / "openasr"),
                mock.patch.object(publish, "run", side_effect=wrong_family_receipt),
                self.assertRaisesRegex(SystemExit, "family mismatch"),
            ):
                publish.copy_stage(
                    "qwen3-asr-0.6b",
                    {"family": "qwen", "kind": "asr-model"},
                    ["fp16"],
                    "README\n",
                    stage,
                )
            self.assertFalse((stage / source.name).exists())


if __name__ == "__main__":
    unittest.main()
