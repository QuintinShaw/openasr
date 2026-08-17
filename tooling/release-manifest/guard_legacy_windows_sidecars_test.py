import unittest

from guard_legacy_windows_sidecars import legacy_sidecar_allowed


class LegacyWindowsSidecarGuardTest(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = {
            "schema_version": 1,
            "legacy_static_sidecar_last_core_version": "0.1.33",
        }

    def test_allows_only_the_frozen_migration_window(self) -> None:
        self.assertTrue(legacy_sidecar_allowed("0.1.32", self.policy))
        self.assertTrue(legacy_sidecar_allowed("v0.1.33", self.policy))
        self.assertFalse(legacy_sidecar_allowed("0.1.34", self.policy))

    def test_rejects_invalid_policy_and_versions(self) -> None:
        with self.assertRaises(ValueError):
            legacy_sidecar_allowed("preview", self.policy)
        with self.assertRaises(ValueError):
            legacy_sidecar_allowed("0.1.34", {"schema_version": 2})


if __name__ == "__main__":
    unittest.main()
