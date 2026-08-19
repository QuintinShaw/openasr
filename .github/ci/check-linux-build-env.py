#!/usr/bin/env python3
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
LOCK = ROOT / ".github/ci/linux-build-env.lock"
IMAGE_PATTERN = re.compile(
    r"ghcr\.io/quintinshaw/openasr-ci-linux@sha256:[0-9a-f]{64}"
)
CONSUMERS = {
    ".github/workflows/ci.yml": 2,
    ".github/workflows/family-regression.yml": 1,
    ".github/workflows/public-hf-e2e.yml": 1,
    ".github/workflows/release-core.yml": 1,
    ".github/workflows/serve-batch-parity.yml": 1,
}


def main() -> None:
    expected = LOCK.read_text(encoding="utf-8").strip()
    if IMAGE_PATTERN.fullmatch(expected) is None:
        raise SystemExit(f"invalid Linux CI image lock: {expected!r}")

    failures: list[str] = []
    for relative, expected_count in CONSUMERS.items():
        text = (ROOT / relative).read_text(encoding="utf-8")
        references = IMAGE_PATTERN.findall(text)
        if references != [expected] * expected_count:
            failures.append(
                f"{relative}: expected {expected_count} reference(s) to {expected}, "
                f"found {references}"
            )
        if "apt-get" in text:
            failures.append(f"{relative}: routine consumer must not run apt-get")

    if failures:
        raise SystemExit("\n".join(failures))
    print(f"Linux CI image consumers match {expected}")


if __name__ == "__main__":
    main()
