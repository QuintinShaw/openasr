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

import re
import sys
from pathlib import Path

from _catalog import (
    QUANT_METADATA,
    SUPPORTED_CAPABILITY_ROLES,
    load as load_publish_catalog,
)
from _file_loaders import load_json, load_toml
from _pathlib_helpers import repo_root

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = repo_root(SCRIPT_DIR)
TOOLING_ROOT = REPO_ROOT / "tooling" / "publish-model"
TEMPLATE = TOOLING_ROOT / "template" / "MODEL_CARD.md.tmpl"
DIARIZE_TEMPLATE = TOOLING_ROOT / "template" / "DIARIZE_CARD.md.tmpl"
TRANSLATION_TEMPLATE = TOOLING_ROOT / "template" / "TRANSLATION_CARD.md.tmpl"
CAPABILITY_TEMPLATE = TOOLING_ROOT / "template" / "CAPABILITY_CARD.md.tmpl"
# Capability semantics are part of the catalog contract. Keep the small
# presentation mapping keyed by role rather than duplicating family ids here.
_CAPABILITY_FEATURE_BY_ROLE = {
    "speaker-embedder": "speaker-diarization",
    "speaker-segmenter": "speaker-diarization",
    "forced-aligner": "word-timestamps",
    "punctuation-restorer": "punctuation",
}
_CAPABILITY_PIPELINE_TAG_BY_ROLE = {
    "speaker-embedder": "feature-extraction",
    "speaker-segmenter": "voice-activity-detection",
    "forced-aligner": "automatic-speech-recognition",
    "punctuation-restorer": "automatic-speech-recognition",
}
# SPDX ids (lowercased) that HF's YAML `license:` field accepts directly. A
# license outside this set (e.g. the FunASR Model License) must use the HF
# `license: other` convention with `license_name` + `license_link` instead of
# an unrecognized bare value.
HF_SPDX_LICENSE_IDS = {"apache-2.0", "mit", "cc-by-4.0"}
OPENASR_NATIVE_HIGHLIGHT = (
    "🦀 **Native in OpenASR** — `.oasr` packs run with no Python at inference, "
    "engineered for peak performance on CPU & GPU"
)


def human_bytes(n: int | None) -> str:
    if not n:
        return "n/a"
    gb = n / 1e9
    return f"{gb:.2f} GB" if gb >= 1 else f"{n / 1e6:.0f} MB"


def pack_quant_note(quants: list[str]) -> str:
    """Footnote under the diarize pack table, driven by the catalog quant set."""
    if quants == ["f32"]:
        return (
            "<sub>Single raw-**f32** build: the pure-Rust forward pass consumes f32 directly and the\n"
            "parity gates assert bit-exact outputs vs the upstream weights, so no integer\n"
            "quantization is produced.</sub>"
        )
    if quants == ["fp16"]:
        return (
            "<sub>Single **fp16** build: projection weights ship as fp16; norms/biases and other\n"
            "parity-sensitive tensors stay f32 inside the pack. No extra public quant tiers.</sub>"
        )
    labels = " · ".join(QUANT_METADATA[q].label for q in quants)
    return f"<sub>Shipped quant tiers: **{labels}**.</sub>"


def pack_storage_note(quants: list[str]) -> str:
    """Body note under the importer snippet for diarize packs."""
    if quants == ["f32"]:
        return (
            "The `.oasr` container is GGUF-backed; every tensor is stored as raw f32 so the\n"
            "pack round-trips bit-identically against the source weights."
        )
    if quants == ["fp16"]:
        return (
            "The `.oasr` container is GGUF-backed; projection weights are stored as fp16 while\n"
            "norms/biases and other parity-sensitive tensors remain f32."
        )
    return (
        "The `.oasr` container is GGUF-backed; each shipped quant stores weights at the\n"
        "requested precision while parity-sensitive tensors stay f32 where required."
    )


def rtf(v) -> str:
    return f"{v:.2f}×" if isinstance(v, (int, float)) else "n/a"


def pct(v) -> str:
    return f"{v * 100:.1f}%" if isinstance(v, (int, float)) else "n/a"


def pull_command(catalog: dict, model_ref: str) -> str:
    """Render the exact consent-bearing CLI command for a catalog entry."""
    command = f"openasr pull {model_ref}"
    if catalog.get("license_class") in {"noncommercial", "gated"}:
        command += " --accept-license"
    return command


def _catalog_kind(catalog: dict) -> str:
    """Return a validated semantic catalog kind; never infer it from family."""
    kind = catalog.get("kind")
    if kind not in {"asr-model", "translation-model", "capability-pack"}:
        raise ValueError(
            "render_card requires a supported catalog kind; "
            f"got {kind!r}"
        )
    return kind


def _capability_semantics(catalog: dict) -> tuple[str, str]:
    """Validate and return ``(feature, role)`` for a capability pack."""
    if _catalog_kind(catalog) != "capability-pack":
        raise ValueError("render_card capability semantics requested for a non-capability model")
    capability = catalog.get("capability")
    if not isinstance(capability, dict):
        raise ValueError("render_card capability-pack entry is missing capability metadata")
    feature = capability.get("feature")
    role = capability.get("role")
    if not isinstance(feature, str) or not feature.strip():
        raise ValueError("render_card capability.feature must be a non-empty string")
    if role not in SUPPORTED_CAPABILITY_ROLES:
        raise ValueError(f"render_card capability.role is unsupported: {role!r}")
    expected_feature = _CAPABILITY_FEATURE_BY_ROLE.get(role)
    if expected_feature is None:
        raise ValueError(f"render_card has no semantics for capability role: {role!r}")
    if feature != expected_feature:
        raise ValueError(
            f"render_card capability role {role!r} requires feature {expected_feature!r}, "
            f"got {feature!r}"
        )
    return feature, role


def card_type_for_catalog(catalog: dict) -> str:
    """Select the card template from catalog semantics, not a family allowlist."""
    kind = _catalog_kind(catalog)
    if kind == "asr-model":
        if catalog.get("capability") is not None:
            raise ValueError("render_card asr-model entry must not carry capability metadata")
        return "asr"
    if kind == "translation-model":
        if catalog.get("capability") is not None:
            raise ValueError("render_card translation-model entry must not carry capability metadata")
        return "translation"

    feature, role = _capability_semantics(catalog)
    if feature == "speaker-diarization" and role in {"speaker-embedder", "speaker-segmenter"}:
        return "diarize"
    return "capability"


def pipeline_tag_for_catalog(catalog: dict, prose: dict) -> str:
    """Resolve the HF pipeline tag from prose or catalog kind/capability role."""
    kind = _catalog_kind(catalog)
    if kind == "capability-pack":
        _feature, role = _capability_semantics(catalog)
    else:
        role = None
        if catalog.get("capability") is not None:
            raise ValueError(f"render_card {kind} entry must not carry capability metadata")

    explicit = prose.get("pipeline_tag")
    if explicit:
        return explicit

    if kind == "asr-model":
        return "automatic-speech-recognition"
    if kind == "translation-model":
        return "translation"

    try:
        assert role is not None
        return _CAPABILITY_PIPELINE_TAG_BY_ROLE[role]
    except KeyError as exc:  # pragma: no cover - guarded by _capability_semantics
        raise ValueError(f"render_card has no pipeline tag for capability role: {role!r}") from exc


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

    card_type = card_type_for_catalog(catalog)
    diarize = card_type == "diarize"
    translation = card_type == "translation"
    capability = card_type == "capability"

    # Perf table rows + pull lines, one per built quant (catalog order). The
    # diarize/capability card's table carries only quant/file/size — ASR bench
    # columns (RTF/WER) do not apply to support packs.
    rows, pulls = [], []
    qm = metrics.get("quants", {})
    for q in catalog["quants"]:
        meta = QUANT_METADATA[q]
        m = qm.get(q, {})
        if diarize or translation or capability:
            rows.append(f"| {meta.label} | `{model}-{q}.oasr` | {human_bytes(m.get('size_bytes'))} |")
        else:
            rows.append(
                f"| {meta.label} | `{model}-{q}.oasr` | {human_bytes(m.get('size_bytes'))} | "
                f"{human_bytes(m.get('peak_rss_bytes'))} | {rtf(m.get('rtf_cpu'))} | "
                f"{rtf(m.get('rtf_metal'))} | {pct(m.get('jfk_wer_vs_fp16'))} |"
            )
        pulls.append(pull_command(catalog, f"{registry_id}:{meta.suffix}"))

    intro = (prose.get("intro") or generic_intro(catalog, upstream_link)).strip()
    ack = (prose.get("acknowledgement") or generic_ack(catalog, upstream_link)).strip()
    aliases = " · ".join(f"`{a}`" for a in catalog["aliases"])
    rec = catalog["recommended_quant"]
    rec_suffix = QUANT_METADATA[rec].suffix

    tagline = (prose.get("tagline") or generic_tagline(catalog)).strip()
    highlights = with_openasr_native_highlight(
        prose.get("highlights") or generic_highlights(catalog, qm)
    )

    if card_type == "diarize":
        template = DIARIZE_TEMPLATE
    elif card_type == "translation":
        template = TRANSLATION_TEMPLATE
    elif card_type == "capability":
        template = CAPABILITY_TEMPLATE
    else:
        template = TEMPLATE
    text = template.read_text()
    repl = {
        "pipeline_tag": pipeline_tag_for_catalog(catalog, prose),
        "upstream_license_id": catalog["license_name"],
        # HF requires the YAML `license:` to be a lowercase SPDX id from its
        # allowed list; the body keeps the display-cased form. Non-SPDX
        # licenses use the HF `other` convention with name + link.
        "license_yaml": license_yaml(catalog),
        "license_badge": badge_text(catalog["license_name"]),
        "upstream_badge": badge_text(upstream.split("/")[-1]),
        "upstream_repo": upstream,
        # Falls back to the registry id when a model has no short pull alias
        # (e.g. no `catalog-series.toml` entry for its family, and none set
        # per-model): every model has a registry id, so the YAML tags list
        # never renders the literal string "None".
        "pull_alias": catalog["pull_alias"] or registry_id,
        "openasr_repo": catalog["hf_repo"],
        "registry_id": registry_id,
        "tagline": tagline,
        "highlights_block": "\n".join(f"- {h}" for h in highlights),
        "intro": intro,
        "model_display_name": catalog["display_name"],
        # Benchmark-clip wording. Defaults describe the fixed 11s JFK ruler used
        # for every card; a prose file overrides them when the model is measured
        # on a different clip (e.g. an in-language clip for a language whose
        # audio JFK does not represent), so the caption never misstates the clip.
        "bench_clip_phrase": prose.get("bench_clip_phrase") or "the fixed 11s JFK clip",
        "bench_clip_short": prose.get("bench_clip_short") or "JFK",
        "drift_metric_label": prose.get("drift_metric_label") or "JFK ΔWER",
        "perf_table_rows": "\n".join(rows),
        "recommended_quant": rec,
        "pull_recommended": pull_command(catalog, f"{registry_id}:{rec_suffix}"),
        "pull_lines": "\n".join(pulls),
        "aliases_inline": aliases,
        "upstream_link": upstream_link,
        "import_subcommand": catalog["import_subcommand"],
        "import_command": prose.get("import_command")
        or (
            f"openasr model-pack {catalog['import_subcommand']} <src>.safetensors <out>.oasr \\\n"
            f"  --package-id {registry_id}"
        ),
        "upstream_license_link": catalog["license_source"],
        "acknowledgement_block": ack,
        "pack_quant_note": pack_quant_note(list(catalog["quants"])),
        "pack_storage_note": pack_storage_note(list(catalog["quants"])),
    }
    for k, v in repl.items():
        text = text.replace("{{" + k + "}}", str(v))
    sys.stdout.write(text)
    return 0


def license_yaml(c: dict) -> str:
    """The YAML `license:` value (plus companions for non-SPDX licenses)."""
    lowered = c["license_name"].lower()
    if lowered in HF_SPDX_LICENSE_IDS:
        return lowered
    slug = re.sub(r"[^a-z0-9.]+", "-", lowered).strip("-")
    return f"other\nlicense_name: {slug}\nlicense_link: {c['license_source']}"


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
