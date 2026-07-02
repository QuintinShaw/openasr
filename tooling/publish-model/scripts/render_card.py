#!/usr/bin/env python3
"""Stage 4 — render the HF model card (README.md) for one model.

  render_card.py <model-id>   > tmp/publish/<model>/repo/README.md

Fills tooling/publish-model/template/MODEL_CARD.md.tmpl from three sources:
  - the publish catalog (tooling/publish-model/*.toml)   — identity, license, pull UX
  - measured metrics (tmp/publish/<model>/metrics.json)  — size / RAM peak / RTF table
  - optional prose (tooling/publish-model/cards/<model>.toml)
    — intro / tagline / highlights / acknowledgement

Models without a prose file get a generic intro + acknowledgement generated from
catalog fields, so a brand-new `发布 <x>` still produces a complete card.
"""
from __future__ import annotations

import sys
from pathlib import Path

from _catalog import QUANT_METADATA, load as load_publish_catalog
from _file_loaders import load_json, load_toml
from _pathlib_helpers import repo_root

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = repo_root(SCRIPT_DIR)
TOOLING_ROOT = REPO_ROOT / "tooling" / "publish-model"
TEMPLATE = TOOLING_ROOT / "template" / "MODEL_CARD.md.tmpl"
DIARIZE_TEMPLATE = TOOLING_ROOT / "template" / "DIARIZE_CARD.md.tmpl"
TRANSLATION_TEMPLATE = TOOLING_ROOT / "template" / "TRANSLATION_CARD.md.tmpl"
# Diarization support packs render the diarize card: no ASR pipeline tags,
# no transcribe quickstart.
DIARIZE_FAMILIES = {"wespeaker", "pyannote-segmentation"}
# Translation models render the translation card: translation pipeline tag,
# realtime-translation quickstart, no ASR bench columns.
TRANSLATION_FAMILIES = {"hymt2"}
# HF YAML pipeline tag per diarize family (the prose card may override).
DIARIZE_PIPELINE_TAG_BY_FAMILY = {
    "wespeaker": "feature-extraction",
    "pyannote-segmentation": "voice-activity-detection",
}
OPENASR_NATIVE_HIGHLIGHT = (
    "🦀 **Native in OpenASR** — `.oasr` packs run with no Python at inference, "
    "engineered for peak performance on CPU & GPU"
)


def human_bytes(n: int | None) -> str:
    if not n:
        return "n/a"
    gb = n / 1e9
    return f"{gb:.2f} GB" if gb >= 1 else f"{n / 1e6:.0f} MB"


def rtf(v) -> str:
    return f"{v:.2f}×" if isinstance(v, (int, float)) else "n/a"


def pct(v) -> str:
    return f"{v * 100:.1f}%" if isinstance(v, (int, float)) else "n/a"


def main(argv: list[str]) -> int:
    model = argv[0]
    catalog = load_publish_catalog()[model]
    metrics_path = REPO_ROOT / "tmp" / "publish" / model / "metrics.json"
    metrics = load_json(metrics_path) if metrics_path.exists() else {"quants": {}}
    prose_path = TOOLING_ROOT / "cards" / f"{model}.toml"
    prose = load_toml(prose_path) if prose_path.exists() else {}

    upstream = catalog["upstream_repo"]
    upstream_link = f"https://huggingface.co/{upstream}"
    registry_id = catalog["registry_id"]

    diarize = catalog["family"] in DIARIZE_FAMILIES
    translation = catalog["family"] in TRANSLATION_FAMILIES

    # Perf table rows + pull lines, one per built quant (catalog order). The
    # diarize card's table carries only quant/file/size — ASR bench columns
    # (RTF/WER) do not apply to support packs.
    rows, pulls = [], []
    qm = metrics.get("quants", {})
    for q in catalog["quants"]:
        meta = QUANT_METADATA[q]
        m = qm.get(q, {})
        if diarize or translation:
            rows.append(f"| {meta.label} | `{model}-{q}.oasr` | {human_bytes(m.get('size_bytes'))} |")
        else:
            rows.append(
                f"| {meta.label} | `{model}-{q}.oasr` | {human_bytes(m.get('size_bytes'))} | "
                f"{human_bytes(m.get('peak_rss_bytes'))} | {rtf(m.get('rtf_cpu'))} | "
                f"{rtf(m.get('rtf_metal'))} | {pct(m.get('jfk_wer_vs_fp16'))} |"
            )
        pulls.append(f"openasr pull {registry_id}:{meta.suffix}")

    intro = (prose.get("intro") or generic_intro(catalog, upstream_link)).strip()
    ack = (prose.get("acknowledgement") or generic_ack(catalog, upstream_link)).strip()
    aliases = " · ".join(f"`{a}`" for a in catalog["aliases"])
    rec = catalog["recommended_quant"]
    rec_suffix = QUANT_METADATA[rec].suffix

    tagline = (prose.get("tagline") or generic_tagline(catalog)).strip()
    highlights = with_openasr_native_highlight(
        prose.get("highlights") or generic_highlights(catalog, qm)
    )

    if diarize:
        template = DIARIZE_TEMPLATE
    elif translation:
        template = TRANSLATION_TEMPLATE
    else:
        template = TEMPLATE
    text = template.read_text()
    repl = {
        "pipeline_tag": prose.get("pipeline_tag")
        or ("translation" if translation else None)
        or DIARIZE_PIPELINE_TAG_BY_FAMILY.get(catalog["family"], "automatic-speech-recognition"),
        "upstream_license_id": catalog["license_name"],
        # HF requires the YAML `license:` to be a lowercase SPDX id from its
        # allowed list; the body keeps the display-cased form.
        "license_yaml": catalog["license_name"].lower(),
        "license_badge": badge_text(catalog["license_name"]),
        "upstream_badge": badge_text(upstream.split("/")[-1]),
        "upstream_repo": upstream,
        "pull_alias": catalog["pull_alias"],
        "openasr_repo": catalog["hf_repo"],
        "registry_id": registry_id,
        "tagline": tagline,
        "highlights_block": "\n".join(f"- {h}" for h in highlights),
        "intro": intro,
        "model_display_name": catalog["display_name"],
        "perf_table_rows": "\n".join(rows),
        "recommended_quant": rec,
        "pull_recommended": f"openasr pull {registry_id}:{rec_suffix}",
        "pull_lines": "\n".join(pulls),
        "aliases_inline": aliases,
        "upstream_link": upstream_link,
        "import_subcommand": catalog["import_subcommand"],
        "upstream_license_link": catalog["license_source"],
        "acknowledgement_block": ack,
    }
    for k, v in repl.items():
        text = text.replace("{{" + k + "}}", str(v))
    sys.stdout.write(text)
    return 0


def badge_text(s: str) -> str:
    """shields.io escaping: '-' -> '--', '_' -> '__', spaces -> '_'."""
    return s.replace("-", "--").replace("_", "__").replace(" ", "_")


def generic_tagline(c: dict) -> str:
    return f"{c['display_name']} speech recognition, packaged for the OpenASR runtime"


def generic_highlights(c: dict, qm: dict) -> list[str]:
    h = []
    metal = [v.get("rtf_metal") for v in qm.values() if isinstance(v.get("rtf_metal"), (int, float))]
    if metal:
        h.append(f"⚡ **Real-time on Apple Silicon** — down to {min(metal):.2f}× RTF on the M1 GPU (Metal)")
    sizes = [v.get("size_bytes") for v in qm.values() if v.get("size_bytes")]
    if sizes:
        h.append(
            f"🪶 **Three builds** from {human_bytes(min(sizes))} (q4_k) to full-fidelity fp16 — "
            f"`{c['recommended_quant']}` recommended"
        )
    else:
        h.append(f"🪶 **Three builds** (fp16 · q8_0 · q4_k) — `{c['recommended_quant']}` recommended")
    h.append(f"🔓 **{c['license_name']}** — same license as the upstream model")
    return h


def with_openasr_native_highlight(highlights: list[str]) -> list[str]:
    """Keep the OpenASR runtime promise as the final README highlight."""
    kept = [
        h for h in highlights
        if not (
            "Native in OpenASR" in h
            or "Native Rust runtime" in h
            or "no Python at inference" in h
        )
    ]
    return [*kept, OPENASR_NATIVE_HIGHLIGHT]


def generic_intro(c: dict, link: str) -> str:
    return (
        f"{c['display_name']} packaged for the OpenASR runtime as `.oasr` packs — no "
        f"Python at inference time. Repackaged from [{c['upstream_repo']}]({link}); the "
        f"{c['recommended_quant']} build is the recommended default, with fp16 for "
        f"maximum fidelity and q4_k for tight-memory deployments."
    )


def generic_ack(c: dict, link: str) -> str:
    return (
        f"This pack is a redistribution of **{c['display_name']}** "
        f"([{c['upstream_repo']}]({link})). All credit for the original architecture, "
        f"training, and weights belongs to the upstream authors; the license is inherited "
        f"from and identical to the upstream model ({c['license_name']})."
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
