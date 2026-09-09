from __future__ import annotations

import unittest

from retention import (
    CHINA_RETENTION,
    OFFICIAL_RETENTION,
    apply_release_retention,
    is_prerelease_version,
    plan_release_retention,
    plan_retention_for_keys,
    version_from_object_key,
)


class PrereleaseTest(unittest.TestCase):
    def test_classifies_preview_and_stable(self) -> None:
        self.assertFalse(is_prerelease_version("0.1.22"))
        self.assertTrue(is_prerelease_version("0.1.22-preview.9"))
        self.assertTrue(is_prerelease_version("0.1.36-rc.1"))


class PlanRetentionTest(unittest.TestCase):
    def test_pins_latest_even_when_outside_the_window(self) -> None:
        stables = [f"0.1.{i}" for i in range(1, 41)]
        plan = plan_release_retention(stables, "0.1.1", **OFFICIAL_RETENTION)
        self.assertIn("0.1.1", plan["keep"])
        self.assertIn("0.1.40", plan["keep"])
        self.assertEqual(len(plan["keep"]), 31)
        self.assertIn("0.1.2", plan["prune"])
        self.assertNotIn("0.1.1", plan["prune"])

    def test_prereleases_do_not_squeeze_stables(self) -> None:
        versions = [f"0.1.{i}" for i in range(1, 17)] + [f"0.2.0-preview.{i}" for i in range(1, 9)]
        plan = plan_release_retention(versions, "0.1.16", **CHINA_RETENTION)
        kept_stable = [version for version in plan["keep"] if not is_prerelease_version(version)]
        kept_pre = [version for version in plan["keep"] if is_prerelease_version(version)]
        self.assertEqual(len(kept_stable), 15)
        self.assertEqual(kept_pre, ["0.2.0-preview.8", "0.2.0-preview.7", "0.2.0-preview.6"])
        self.assertIn("0.2.0-preview.1", plan["prune"])
        self.assertIn("0.1.1", plan["prune"])


class PlanKeysTest(unittest.TestCase):
    def test_never_proposes_deleting_latest_json(self) -> None:
        keys = [
            "desktop/releases/v0.1.1/OpenASR-Desktop-0.1.1-aarch64.dmg",
            "desktop/releases/v0.1.2/OpenASR-Desktop-0.1.2-aarch64.dmg",
            "desktop/stable/latest.json",
            "core/v0.1.1/openasr-windows-cuda.zip",
        ]
        plan = plan_retention_for_keys(keys, "0.1.2", keep_stable=1, keep_prerelease=0)
        self.assertEqual(plan["prune"], ["0.1.1"])
        self.assertEqual(
            plan["prune_keys"],
            [
                "desktop/releases/v0.1.1/OpenASR-Desktop-0.1.1-aarch64.dmg",
                "core/v0.1.1/openasr-windows-cuda.zip",
            ],
        )
        self.assertNotIn("desktop/stable/latest.json", plan["prune_keys"])

    def test_parses_prefixes(self) -> None:
        self.assertEqual(
            version_from_object_key("desktop/releases/v0.1.22-preview.9/x.app.tar.gz"),
            "0.1.22-preview.9",
        )
        self.assertEqual(version_from_object_key("core/v0.1.36/openasr.zip"), "0.1.36")
        self.assertIsNone(version_from_object_key("desktop/stable/latest.json"))


class ApplyRetentionTest(unittest.TestCase):
    def test_dry_run_does_not_delete(self) -> None:
        deleted: list[str] = []
        logs: list[str] = []
        keys = [f"desktop/releases/v0.1.{i}/a.dmg" for i in range(1, 33)]
        result = apply_release_retention(
            profile="official",
            latest_stable="0.1.32",
            keys=keys,
            prune=False,
            delete_keys=deleted.extend,
            log=logs.append,
        )
        self.assertFalse(result["applied"])
        self.assertEqual(deleted, [])
        self.assertTrue(any("would delete" in line and "0.1.1" in line for line in logs))

    def test_prune_deletes_old_keys(self) -> None:
        deleted: list[str] = []
        keys = [f"desktop/releases/v0.1.{i}/a.dmg" for i in range(1, 18)]
        result = apply_release_retention(
            profile="china",
            latest_stable="0.1.17",
            keys=keys,
            prune=True,
            delete_keys=deleted.extend,
            log=lambda _line: None,
        )
        self.assertTrue(result["applied"])
        self.assertEqual(deleted, ["desktop/releases/v0.1.1/a.dmg", "desktop/releases/v0.1.2/a.dmg"])


if __name__ == "__main__":
    unittest.main()
