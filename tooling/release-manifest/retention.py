#!/usr/bin/env python3
"""Hot-cache retention planner for Official (B2) and China (CNB) releases.

GitHub keeps the full archive. B2 keeps the last 30 stables + 5 prereleases;
CNB keeps 15 + 3. The live latest stable is pinned so a prune cannot delete
the objects the current pointer still names.

Default is log-only. Real deletes require OPENASR_RELEASE_PRUNE=1 and a
caller-supplied delete_keys callback. This module does not talk to the
network.
"""
from __future__ import annotations

import os
import re
from functools import cmp_to_key
from typing import Callable, Iterable

OFFICIAL_RETENTION = {"keep_stable": 30, "keep_prerelease": 5}
CHINA_RETENTION = {"keep_stable": 15, "keep_prerelease": 3}

_SEMVER = re.compile(
    r"^v?(\d+)\.(\d+)\.(\d+)(?:-([0-9A-Za-z.-]+))?(?:\+[0-9A-Za-z.-]+)?$"
)
_VERSIONED_KEY = re.compile(r"^(?:desktop/releases|core|cli)/v([^/]+)/")


class RetentionError(Exception):
    pass


def parse_semver(version: str) -> tuple[int, int, int, list[str]] | None:
    match = _SEMVER.match((version or "").strip())
    if not match:
        return None
    prerelease = match.group(4).split(".") if match.group(4) else []
    return int(match.group(1)), int(match.group(2)), int(match.group(3)), prerelease


def is_prerelease_version(version: str) -> bool:
    parsed = parse_semver(version)
    return bool(parsed and parsed[3])


def compare_semver(left: str, right: str) -> int:
    """Return -1 / 0 / 1. Twin of apps/desktop/scripts/release-publish.mjs."""
    pa = parse_semver(left)
    pb = parse_semver(right)
    if pa is None or pb is None:
        raise RetentionError(f"Cannot compare non-semver versions: {left!r} vs {right!r}")
    for index in range(3):
        if pa[index] != pb[index]:
            return -1 if pa[index] < pb[index] else 1
    a_pre, b_pre = pa[3], pb[3]
    if not a_pre and not b_pre:
        return 0
    if not a_pre:
        return 1
    if not b_pre:
        return -1
    length = max(len(a_pre), len(b_pre))
    for index in range(length):
        if index >= len(a_pre):
            return -1
        if index >= len(b_pre):
            return 1
        ia, ib = a_pre[index], b_pre[index]
        na = int(ia) if ia.isdigit() else None
        nb = int(ib) if ib.isdigit() else None
        if na is not None and nb is not None:
            if na != nb:
                return -1 if na < nb else 1
        elif na is not None:
            return -1
        elif nb is not None:
            return 1
        elif ia != ib:
            return -1 if ia < ib else 1
    return 0


def version_from_object_key(key: str) -> str | None:
    match = _VERSIONED_KEY.match(key)
    return match.group(1) if match else None


def plan_release_retention(
    versions: Iterable[str],
    latest_stable: str | None,
    keep_stable: int,
    keep_prerelease: int,
) -> dict[str, list[str]]:
    unique = list(dict.fromkeys(version for version in versions if version))
    by_semver = cmp_to_key(compare_semver)
    stable = sorted(
        (version for version in unique if not is_prerelease_version(version)),
        key=by_semver,
        reverse=True,
    )
    prerelease = sorted(
        (version for version in unique if is_prerelease_version(version)),
        key=by_semver,
        reverse=True,
    )
    keep: set[str] = set()
    if latest_stable:
        keep.add(latest_stable)
    keep.update(stable[:keep_stable])
    keep.update(prerelease[:keep_prerelease])
    prune = sorted((version for version in unique if version not in keep), key=by_semver, reverse=True)
    kept = sorted((version for version in unique if version in keep), key=by_semver, reverse=True)
    return {"keep": kept, "prune": prune}


def plan_retention_for_keys(
    keys: Iterable[str],
    latest_stable: str | None,
    keep_stable: int,
    keep_prerelease: int,
) -> dict[str, list[str]]:
    key_list = list(keys)
    versions = list(dict.fromkeys(version for version in (version_from_object_key(key) for key in key_list) if version))
    plan = plan_release_retention(versions, latest_stable, keep_stable, keep_prerelease)
    prune_set = set(plan["prune"])
    plan["prune_keys"] = [key for key in key_list if version_from_object_key(key) in prune_set]
    return plan


def apply_release_retention(
    *,
    profile: str = "official",
    latest_stable: str | None = None,
    keys: Iterable[str] | None = None,
    prune: bool | None = None,
    delete_keys: Callable[[list[str]], None] | None = None,
    log: Callable[[str], None] = print,
) -> dict:
    limits = CHINA_RETENTION if profile == "china" else OFFICIAL_RETENTION
    if prune is None:
        prune = os.environ.get("OPENASR_RELEASE_PRUNE") == "1"
    log(
        f"release retention ({profile}): keepStable={limits['keep_stable']} "
        f"keepPrerelease={limits['keep_prerelease']} pin={latest_stable or '(none)'}"
    )
    key_list = list(keys or [])
    if not key_list:
        log("release retention: no object list supplied; skip would-delete")
        return {"keep": [latest_stable] if latest_stable else [], "prune": [], "prune_keys": [], "applied": False}
    plan = plan_retention_for_keys(key_list, latest_stable, limits["keep_stable"], limits["keep_prerelease"])
    if not plan["prune"]:
        log("release retention: nothing to prune")
        plan["applied"] = False
        return plan
    if not prune:
        log(
            f"release retention: would delete {', '.join(plan['prune'])} "
            f"({len(plan['prune_keys'])} objects); set OPENASR_RELEASE_PRUNE=1 to apply"
        )
        plan["applied"] = False
        return plan
    if delete_keys is None:
        raise RetentionError("OPENASR_RELEASE_PRUNE=1 requires a delete_keys callback")
    log(f"release retention: deleting {', '.join(plan['prune'])} ({len(plan['prune_keys'])} objects)")
    delete_keys(plan["prune_keys"])
    plan["applied"] = True
    return plan
