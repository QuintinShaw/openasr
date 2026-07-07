#!/usr/bin/env python3
"""Replace the Install & Verify section of a release body in place.

Usage:
  splice-install-verify.py <body-file> <new-section-file>

Prints the spliced body to stdout. <new-section-file> must contain the
`<!-- install-verify:start -->` ... `<!-- install-verify:end -->` markers
(scripts/render-install-verify.sh emits them); everything between the same
markers in <body-file> is replaced with it.
"""
import sys

START_MARKER = "<!-- install-verify:start -->"
END_MARKER = "<!-- install-verify:end -->"


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 1
    body_path, new_section_path = sys.argv[1], sys.argv[2]
    body = open(body_path, encoding="utf-8").read()
    new_section = open(new_section_path, encoding="utf-8").read().rstrip("\n")

    try:
        start = body.index(START_MARKER)
        end = body.index(END_MARKER) + len(END_MARKER)
    except ValueError:
        print(
            f"error: {body_path} does not contain both {START_MARKER!r} and "
            f"{END_MARKER!r} markers -- was it created by the release job's "
            "render-install-verify.sh stub?",
            file=sys.stderr,
        )
        return 1

    sys.stdout.write(body[:start] + new_section + body[end:])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
