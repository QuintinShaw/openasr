#!/usr/bin/env python3
"""Regenerate local true-streaming packs and run the real-runtime smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Family:
    name: str
    source: str
    output_name: str
    import_args: tuple[str, ...]


@dataclass(frozen=True)
class FamilySmokeResult:
    family: str
    source: str
    pack_file: str
    pack_sha256: str
    pack_size_bytes: int
    model_identity: str
    runtime_family: str
    final_text_line: str
    final_text: str
    pack_origin: str = "generated"
    inspect_true_streaming: bool = True
    validated: bool = True
    smoke_passed: bool = True


FAMILIES: tuple[Family, ...] = (
    Family(
        name="qwen",
        source="tmp/models/qwen3-asr/Qwen-source",
        output_name="qwen3-asr-0.6b-q4_k.streaming.oasr",
        import_args=(
            "model-pack",
            "import",
            "qwen",
            "{source}",
            "{output}",
            "--package-id",
            "qwen3-asr-0.6b",
            "--package-variant",
            "q4_k",
            "--source-revision",
            "local",
            "--license-source",
            "https://huggingface.co/Qwen/Qwen3-ASR-0.6B",
            "--quantization",
            "q4-k",
        ),
    ),
    Family(
        name="whisper",
        source="tmp/models/whisper/hf/whisper-tiny.en",
        output_name="whisper-tiny-en-q4_k.streaming.oasr",
        import_args=(
            "model-pack",
            "import",
            "whisper",
            "{source}",
            "{output}",
            "--package-id",
            "whisper-tiny.en",
            "--package-variant",
            "q4_k",
            "--source-revision",
            "local",
            "--quantization",
            "q4-k",
        ),
    ),
    Family(
        name="cohere",
        source="tmp/models/cohere-transcribe-03-2026/CohereLabs-source",
        output_name="cohere-transcribe-03-2026-q4_k.streaming.oasr",
        import_args=(
            "model-pack",
            "import",
            "cohere",
            "{source}",
            "{output}",
            "--package-id",
            "cohere-transcribe-03-2026",
            "--package-variant",
            "q4_k",
            "--source-revision",
            "local",
            "--license-source",
            "https://huggingface.co/CohereLabs/cohere-transcribe-03-2026",
            "--quantization",
            "q4-k",
        ),
    ),
    Family(
        name="moonshine",
        source="tmp/models/moonshine-tiny-source",
        output_name="moonshine-tiny-q4_k.streaming.oasr",
        import_args=(
            "model-pack",
            "import",
            "moonshine",
            "{source}",
            "{output}",
            "--package-id",
            "moonshine-tiny",
            "--package-variant",
            "q4_k",
            "--quantization",
            "q4-k",
        ),
    ),
    Family(
        name="parakeet",
        source="tmp/models/parakeet-ctc-0.6b",
        output_name="parakeet-ctc-0.6b-q4_k.streaming.oasr",
        import_args=(
            "model-pack",
            "import",
            "parakeet-ctc",
            "{source}",
            "{output}",
            "--package-id",
            "parakeet-ctc-0.6b",
            "--quantization",
            "q4-k",
        ),
    ),
    Family(
        name="wav2vec2",
        source="tmp/models/wav2vec2-base-960h-source",
        output_name="wav2vec2-base-960h-q4_k.streaming.oasr",
        import_args=(
            "model-pack",
            "import",
            "wav2vec2-ctc",
            "{source}",
            "{output}",
            "--package-id",
            "wav2vec2-base-960h",
            "--quantization",
            "q4-k",
        ),
    ),
    Family(
        name="xasr",
        source="tmp/models/xasr-zh-en",
        output_name="xasr-zh-en-q4_k.streaming.oasr",
        import_args=(
            "model-pack",
            "import",
            "xasr-zipformer",
            "{source}",
            "{output}",
            "--package-id",
            "xasr-zh-en",
            "--quantization",
            "q4-k",
        ),
    ),
)
TEMP_STREAMING_PACK_SUFFIX = ".streaming.oasr"
EXPECTED_RUNTIME_FAMILIES_BY_SMOKE_FAMILY = {
    "qwen": {"qwen3-asr"},
    "whisper": {"whisper"},
    "cohere": {"cohere-transcribe"},
    "moonshine": {"moonshine"},
    "parakeet": {"parakeet-ctc"},
    "wav2vec2": {"wav2vec2-ctc"},
    "xasr": {"xasr-zipformer"},
}


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate local true-streaming packs and run ignored native smoke tests."
    )
    parser.add_argument("--repo-root", default=".", help=argparse.SUPPRESS)
    parser.add_argument(
        "--families",
        default="all",
        help="Comma-separated family names or 'all'.",
    )
    parser.add_argument(
        "--audio",
        default=os.environ.get("OPENASR_STREAMING_SMOKE_AUDIO", "fixtures/jfk.wav"),
    )
    parser.add_argument(
        "--workdir",
        default=os.environ.get("OPENASR_STREAMING_SMOKE_WORKDIR", "tmp/native-streaming-smoke"),
    )
    parser.add_argument(
        "--bin",
        default=os.environ.get("OPENASR_STREAMING_SMOKE_BIN", "target/debug/openasr"),
        help="openasr CLI binary to use. Built automatically when missing.",
    )
    parser.add_argument("--max-ms", type=int, default=4000)
    parser.add_argument("--skip-import", action="store_true", help="Reuse existing packs in --workdir.")
    parser.add_argument(
        "--pack",
        action="append",
        default=[],
        metavar="FAMILY=PATH",
        help=(
            "Use an existing release/candidate pack for one family instead of "
            "importing or reusing the default workdir pack. Repeat for multiple families."
        ),
    )
    parser.add_argument(
        "--summary-json",
        help="Write a redacted JSON evidence summary to this path.",
    )
    parser.add_argument(
        "--summary-md",
        help="Write a redacted Markdown evidence block for a local validation evidence log.",
    )
    parser.add_argument(
        "--strict-release-evidence",
        action="store_true",
        help=(
            "Require final release-pack evidence shape: every selected family "
            "must use --pack FAMILY=PATH, --build-id, and --summary-json must be set."
        ),
    )
    parser.add_argument(
        "--build-id",
        help="OpenASR build or commit identifier to include in final release-pack evidence.",
    )
    return parser.parse_args(list(argv))


def resolve(repo_root: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = repo_root / path
    return path


def is_release_pack_filename(value: str) -> bool:
    name = Path(value).name
    return name.endswith(".oasr") and not name.endswith(TEMP_STREAMING_PACK_SUFFIX)


def run(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    rendered = " ".join(command)
    print(f"$ {rendered}", flush=True)
    return subprocess.run(
        command,
        cwd=str(cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )


def run_streaming(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> str:
    completed = run(command, cwd=cwd, env=env)
    if completed.stdout:
        print(completed.stdout, end="")
    return completed.stdout


def selected_families(value: str) -> list[Family]:
    by_name = {family.name: family for family in FAMILIES}
    if value.strip().lower() == "all":
        return list(FAMILIES)
    names = [part.strip().lower() for part in value.split(",") if part.strip()]
    unknown = [name for name in names if name not in by_name]
    if unknown:
        raise SystemExit(f"unknown families: {', '.join(unknown)}")
    if not names:
        raise SystemExit("--families must name at least one family")
    return [by_name[name] for name in names]


def parse_pack_overrides(values: list[str]) -> dict[str, str]:
    by_name = {family.name for family in FAMILIES}
    overrides: dict[str, str] = {}
    for raw in values:
        family, separator, path = raw.partition("=")
        family = family.strip().lower()
        path = path.strip()
        if not separator or not family or not path:
            raise SystemExit("--pack must use FAMILY=PATH")
        if family not in by_name:
            raise SystemExit(f"--pack names unknown family: {family}")
        if family in overrides:
            raise SystemExit(f"--pack specified more than once for family: {family}")
        overrides[family] = path
    return overrides


def validate_strict_release_evidence_args(
    args: argparse.Namespace,
    families: list[Family],
    pack_overrides: dict[str, str],
) -> None:
    if not args.strict_release_evidence:
        return
    errors: list[str] = []
    if not args.summary_json:
        errors.append("--summary-json is required with --strict-release-evidence")
    if not args.build_id:
        errors.append("--build-id is required with --strict-release-evidence")
    selected_names = {family.name for family in families}
    missing = sorted(family.name for family in families if family.name not in pack_overrides)
    if missing:
        errors.append(
            "--strict-release-evidence requires --pack for selected families: "
            + ", ".join(missing)
        )
    temporary_pack_overrides = sorted(
        f"{family}={path}"
        for family, path in pack_overrides.items()
        if family in selected_names and not is_release_pack_filename(path)
    )
    if temporary_pack_overrides:
        errors.append(
            "--strict-release-evidence requires final release-pack filenames; "
            "temporary local streaming packs are not allowed: "
            + ", ".join(temporary_pack_overrides)
        )
    if errors:
        raise SystemExit("; ".join(errors))


def ensure_cli(repo_root: Path, openasr_bin: Path) -> None:
    if openasr_bin.is_file() and os.access(openasr_bin, os.X_OK):
        return
    run_streaming(["cargo", "build", "-p", "openasr-cli"], cwd=repo_root)
    if not openasr_bin.is_file() or not os.access(openasr_bin, os.X_OK):
        raise SystemExit(f"openasr binary is not executable: {openasr_bin}")


def check_inspect_output(output: str, pack: Path) -> None:
    required = [
        "- mode: true_streaming",
        "- supports_partial_results: true",
        "- is_true_streaming: true",
    ]
    missing = [line for line in required if line not in output]
    if missing:
        raise SystemExit(f"{pack} did not advertise true-streaming capability; missing {missing}")


def parse_inspect_metadata(output: str) -> tuple[str, str]:
    model_identity = ""
    runtime_family = ""
    for line in output.splitlines():
        if line.startswith("Model identity: "):
            model_identity = line.partition(": ")[2].partition(" (")[0].strip()
        if line.startswith("- openasr.model.family: "):
            runtime_family = line.partition(": ")[2].strip()
    return model_identity, runtime_family


def check_runtime_family(output: str, family: Family, pack: Path) -> tuple[str, str]:
    model_identity, runtime_family = parse_inspect_metadata(output)
    expected = EXPECTED_RUNTIME_FAMILIES_BY_SMOKE_FAMILY.get(family.name, set())
    if not model_identity:
        raise SystemExit(f"{pack} did not report Model identity in show output")
    if runtime_family not in expected:
        expected_text = ", ".join(sorted(expected)) or family.name
        raise SystemExit(
            f"{pack} reported runtime family '{runtime_family or '<missing>'}' "
            f"for smoke family '{family.name}', expected {expected_text}"
        )
    return model_identity, runtime_family


def final_text_from_line(line: str) -> str:
    prefix = "native streaming smoke final text"
    if line.startswith(prefix):
        return line.partition(":")[2].strip()
    return line.strip()


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def smoke_family(
    repo_root: Path,
    openasr_bin: Path,
    workdir: Path,
    audio: Path,
    max_ms: int,
    family: Family,
    skip_import: bool,
    pack_override: str | None = None,
) -> FamilySmokeResult:
    source = resolve(repo_root, family.source)
    output = resolve(repo_root, pack_override) if pack_override else workdir / family.output_name
    pack_origin = "provided" if pack_override else "generated"
    if pack_override:
        if not output.is_file():
            raise SystemExit(f"{family.name} pack override is missing: {output}")
        print(f"Using provided {family.name} pack: {output}", flush=True)
    else:
        if not source.exists():
            raise SystemExit(f"{family.name} source path is missing: {source}")
    if not pack_override and (not skip_import or not output.exists()):
        command = [
            str(openasr_bin),
            *[
                part.format(source=str(source), output=str(output))
                for part in family.import_args
            ],
        ]
        run_streaming(command, cwd=repo_root)
    inspect_output = run_streaming(
        [
            str(openasr_bin),
            "show",
            str(output),
        ],
        cwd=repo_root,
    )
    check_inspect_output(inspect_output, output)
    model_identity, runtime_family = check_runtime_family(inspect_output, family, output)
    run_streaming([str(openasr_bin), "verify", str(output)], cwd=repo_root)
    env = os.environ.copy()
    env["OPENASR_NATIVE_STREAMING_SMOKE_PACK"] = str(output)
    env["OPENASR_NATIVE_STREAMING_SMOKE_WAV"] = str(audio)
    env["OPENASR_NATIVE_STREAMING_SMOKE_MAX_MS"] = str(max_ms)
    env.setdefault("OPENASR_GGML_BACKEND", "cpu")
    smoke_output = run_streaming(
        [
            "cargo",
            "test",
            "-p",
            "openasr-core",
            "native_streaming_real_runtime_smoke_from_env",
            "--",
            "--ignored",
            "--nocapture",
        ],
        cwd=repo_root,
        env=env,
    )
    final_line = next(
        (
            line
            for line in smoke_output.splitlines()
            if line.startswith("native streaming smoke final text")
        ),
        None,
    )
    if final_line is None:
        raise SystemExit(f"{family.name} smoke did not print the final transcript line")
    return FamilySmokeResult(
        family=family.name,
        source="provided-pack" if pack_override else family.source,
        pack_file=output.name,
        pack_sha256=file_sha256(output),
        pack_size_bytes=output.stat().st_size,
        model_identity=model_identity,
        runtime_family=runtime_family,
        final_text_line=final_line,
        final_text=final_text_from_line(final_line),
        pack_origin=pack_origin,
    )


def relative_path_or_name(repo_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.name


def build_summary(
    *,
    repo_root: Path,
    audio: Path,
    workdir: Path,
    families: list[Family],
    max_ms: int,
    skip_import: bool,
    build_id: str | None,
    results: list[FamilySmokeResult],
) -> dict:
    return {
        "schema_version": 1,
        "probe": "native_streaming_smoke",
        "audio_file": audio.name,
        "workdir": relative_path_or_name(repo_root, workdir),
        "max_ms": max_ms,
        "skip_import": skip_import,
        "strict_release_evidence": False,
        "build": {"runner": build_id or "not-recorded"},
        "families_requested": [family.name for family in families],
        "results": [
            {
                "family": result.family,
                "source": result.source,
                "pack_file": result.pack_file,
                "pack_sha256": result.pack_sha256,
                "pack_size_bytes": result.pack_size_bytes,
                "model_identity": result.model_identity,
                "runtime_family": result.runtime_family,
                "pack_origin": result.pack_origin,
                "inspect_true_streaming": result.inspect_true_streaming,
                "validated": result.validated,
                "smoke_passed": result.smoke_passed,
                "final_text": result.final_text,
                "final_text_chars": len(result.final_text),
            }
            for result in results
        ],
    }


def write_summary_json(path: str, summary: dict) -> None:
    destination = Path(path)
    if destination.parent != Path("."):
        destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote redacted native streaming smoke summary JSON: {destination}")


def write_summary_markdown(path: str, summary: dict) -> None:
    destination = Path(path)
    if destination.parent != Path("."):
        destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(build_validation_markdown(summary), encoding="utf-8")
    print(f"Wrote redacted native streaming smoke summary Markdown: {destination}")


def markdown_inline(value: object) -> str:
    return str(value).replace("`", "'").replace("\r", " ").replace("\n", " ")


def build_validation_markdown(summary: dict) -> str:
    if summary.get("probe") != "native_streaming_smoke":
        raise RuntimeError(f"unsupported validation summary probe: {summary.get('probe')!r}")
    results = summary.get("results", [])
    result_lines = []
    for result in results:
        result_lines.append(
            "- "
            f"`{markdown_inline(result.get('family'))}`: "
            f"pack `{markdown_inline(result.get('pack_file'))}`, "
            f"model `{markdown_inline(result.get('model_identity'))}`, "
            f"runtime family `{markdown_inline(result.get('runtime_family'))}`, "
            f"sha256 `{markdown_inline(result.get('pack_sha256'))}`, "
            f"size `{markdown_inline(result.get('pack_size_bytes'))}` bytes, "
            f"origin `{markdown_inline(result.get('pack_origin'))}`, "
            f"inspect true-streaming `{markdown_inline(result.get('inspect_true_streaming'))}`, "
            f"validated `{markdown_inline(result.get('validated'))}`, "
            f"smoke `{markdown_inline(result.get('smoke_passed'))}`, "
            f"final {result.get('final_text_chars')} chars "
            f"`{markdown_inline(result.get('final_text'))}`"
        )
    rendered_results = "\n".join(result_lines) if result_lines else "- no results recorded"
    return (
        "### YYYY-MM-DD — Native streaming release-pack smoke evidence\n\n"
        "Scope:\n\n"
        "- Ran the native streaming smoke helper against runtime packs whose "
        "family registers a built-in streaming executor.\n"
        "- Verified inspect capability gates, `verify`, and the "
        "ignored real-runtime streaming smoke final transcript.\n\n"
        "Validation:\n\n"
        f"- audio fixture: `{markdown_inline(summary.get('audio_file'))}`\n"
        f"- workdir: `{markdown_inline(summary.get('workdir'))}`\n"
        f"- max duration: `{markdown_inline(summary.get('max_ms'))} ms`\n"
        f"- skip import: `{markdown_inline(summary.get('skip_import'))}`\n"
        f"- strict release evidence: `{markdown_inline(summary.get('strict_release_evidence', False))}`\n"
        f"- runner build: `{markdown_inline((summary.get('build') or {}).get('runner') or 'not-recorded')}`\n"
        f"- families requested: `{markdown_inline(', '.join(summary.get('families_requested', [])))}`\n"
        f"{rendered_results}\n"
        "\n"
        "Notes:\n\n"
        "- Replace `YYYY-MM-DD` with the actual run date.\n"
        "- Keep generated packs under ignored `tmp/`; do not commit model weights "
        "or generated runtime packs.\n"
        "- Add release artifact URLs or commit ids only after the packs have gone "
        "through the signed publication flow.\n"
    )


def print_summary(results: list[FamilySmokeResult]) -> None:
    print("\nNative streaming smoke summary:")
    for result in results:
        print(f"- {result.family}: {result.final_text_line}")


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    if args.max_ms <= 0:
        raise SystemExit("--max-ms must be positive")
    repo_root = Path(args.repo_root).resolve()
    openasr_bin = resolve(repo_root, args.bin)
    audio = resolve(repo_root, args.audio)
    workdir = resolve(repo_root, args.workdir)
    if not audio.is_file():
        raise SystemExit(f"audio fixture not found: {audio}")
    workdir.mkdir(parents=True, exist_ok=True)
    ensure_cli(repo_root, openasr_bin)

    families = selected_families(args.families)
    pack_overrides = parse_pack_overrides(args.pack)
    unselected_overrides = sorted(set(pack_overrides) - {family.name for family in families})
    if unselected_overrides:
        raise SystemExit(
            "--pack specified families not selected by --families: "
            + ", ".join(unselected_overrides)
        )
    validate_strict_release_evidence_args(args, families, pack_overrides)
    results: list[FamilySmokeResult] = []
    for family in families:
        print(f"\n== {family.name} ==", flush=True)
        results.append(
            smoke_family(
                repo_root,
                openasr_bin,
                workdir,
                audio,
                args.max_ms,
                family,
                args.skip_import,
                pack_overrides.get(family.name),
            )
        )

    print_summary(results)
    summary = build_summary(
        repo_root=repo_root,
        audio=audio,
        workdir=workdir,
        families=families,
        max_ms=args.max_ms,
        skip_import=args.skip_import,
        build_id=args.build_id,
        results=results,
    )
    summary["strict_release_evidence"] = bool(args.strict_release_evidence)
    if args.summary_json:
        write_summary_json(args.summary_json, summary)
    if args.summary_md:
        write_summary_markdown(args.summary_md, summary)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except KeyboardInterrupt:
        raise SystemExit(130)
