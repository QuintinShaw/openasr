#!/usr/bin/env python3
"""Tests for the catalog consistency gate (check_catalog_consistency.py)."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import _catalog  # noqa: E402
from _pathlib_helpers import repo_root  # noqa: E402
from check_catalog_consistency import (  # noqa: E402
    CANONICAL_CATALOG_URL,
    LOCAL_DEV_KEY_ID,
    LOCAL_DEV_PUBLIC_KEY_HEX,
    PRODUCTION_KEY_ID,
    PRODUCTION_PUBLIC_KEY_HEX,
    GateFailure,
    catalog_signing_payload,
    check_public_projection,
    ed25519_public_key,
    ed25519_sign,
    ed25519_verify,
    run_gate,
    sha256_hex,
)

REPO_ROOT = repo_root(SCRIPT_DIR)
REGISTRY = REPO_ROOT / "model-registry"
RUST_SECURITY = REPO_ROOT / "crates" / "openasr-core" / "src" / "catalog_security.rs"

# The public, deterministic local-dev signing seed (catalog_security.rs /
# sign_local_catalog.sh). Public by design; used here to build dev-signed
# fixtures without touching any production key material.
LOCAL_DEV_SEED = bytes.fromhex(
    "7181d685f3c226e1c111574368512b603d67964c057165ad004683b84998960e"
)


def copy_registry() -> Path:
    destination = Path(tempfile.mkdtemp(prefix="openasr-catalog-gate-"))
    shutil.rmtree(destination)
    shutil.copytree(REGISTRY, destination)
    return destination


def sign_manifest(
    *,
    seed: bytes,
    key_id: str,
    catalog_bytes: bytes,
    epoch: int,
) -> dict:
    sha = sha256_hex(catalog_bytes)
    payload = catalog_signing_payload(
        key_id=key_id,
        catalog_url=CANONICAL_CATALOG_URL,
        catalog_sha256=sha,
        catalog_epoch=epoch,
    )
    return {
        "schema_version": 1,
        "catalog_url": CANONICAL_CATALOG_URL,
        "catalog_sha256": sha,
        "catalog_epoch": epoch,
        "signature": {
            "algorithm": "ed25519",
            "key_id": key_id,
            "value": ed25519_sign(seed, payload).hex(),
        },
    }


class Ed25519PrimitivesTest(unittest.TestCase):
    def test_dev_seed_derives_the_documented_dev_public_key(self) -> None:
        # RFC 8032 round trip anchored on the repo's own documented dev key:
        # the pure-Python implementation derives the exact public key the
        # Rust trust root commits to.
        self.assertEqual(ed25519_public_key(LOCAL_DEV_SEED).hex(), LOCAL_DEV_PUBLIC_KEY_HEX)

    def test_sign_verify_round_trip_and_tamper_detection(self) -> None:
        message = b"openasr catalog payload"
        signature = ed25519_sign(LOCAL_DEV_SEED, message)
        public = bytes.fromhex(LOCAL_DEV_PUBLIC_KEY_HEX)
        self.assertTrue(ed25519_verify(public, message, signature))
        self.assertFalse(ed25519_verify(public, message + b"x", signature))
        tampered = bytearray(signature)
        tampered[10] ^= 0x01
        self.assertFalse(ed25519_verify(public, message, bytes(tampered)))
        self.assertFalse(ed25519_verify(public, message, signature[:-1]))


class CommittedRegistryGateTest(unittest.TestCase):
    def test_committed_registry_passes(self) -> None:
        # The committed state must ALWAYS be consistent; this is the same
        # invariant CI and the pre-commit hook enforce.
        failures = run_gate(registry_dir=REGISTRY, allow_dev_key=False)
        self.assertEqual(failures, [])

    def test_catalog_pair_cli_uses_the_production_signature_trust_root(self) -> None:
        command = [
            sys.executable,
            str(SCRIPT_DIR / "check_catalog_consistency.py"),
            "--catalog-pair",
            str(REGISTRY / "catalog.public.json"),
            str(REGISTRY / "catalog.public.signature.json"),
        ]
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)

        with tempfile.TemporaryDirectory() as temp:
            tampered = Path(temp) / "catalog.json"
            tampered.write_bytes((REGISTRY / "catalog.public.json").read_bytes() + b" ")
            rejected = subprocess.run(
                [*command[:3], str(tampered), command[-1]],
                check=False,
                capture_output=True,
                text=True,
            )
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("content hash mismatch", rejected.stderr)

    def test_tampered_catalog_fails_on_hash_binding(self) -> None:
        registry = copy_registry()
        try:
            catalog = registry / "catalog.json"
            catalog.write_text(catalog.read_text() + " \n", encoding="utf-8")
            failures = run_gate(registry_dir=registry, allow_dev_key=False)
            self.assertTrue(
                any("content hash mismatch" in failure for failure in failures),
                failures,
            )
        finally:
            shutil.rmtree(registry)

    def test_epoch_drift_fails(self) -> None:
        registry = copy_registry()
        try:
            epoch_file = registry / "catalog.epoch"
            bumped = int(epoch_file.read_text().strip()) + 1
            epoch_file.write_text(f"{bumped}\n", encoding="utf-8")
            failures = run_gate(registry_dir=registry, allow_dev_key=False)
            self.assertTrue(
                any("catalog.epoch" in failure for failure in failures), failures
            )
        finally:
            shutil.rmtree(registry)

    def test_projection_drift_fails(self) -> None:
        registry = copy_registry()
        try:
            public_path = registry / "catalog.public.json"
            public = json.loads(public_path.read_text(encoding="utf-8"))
            public["models"] = public["models"][:-1]
            public_path.write_text(
                json.dumps(public, indent=2) + "\n", encoding="utf-8"
            )
            failures = run_gate(registry_dir=registry, allow_dev_key=False)
            self.assertTrue(
                any("public projection" in failure for failure in failures), failures
            )
        finally:
            shutil.rmtree(registry)

    def test_dev_signed_manifest_is_rejected_unless_allowed(self) -> None:
        registry = copy_registry()
        try:
            catalog_bytes = (registry / "catalog.json").read_bytes()
            epoch = int((registry / "catalog.epoch").read_text().strip())
            dev_manifest = sign_manifest(
                seed=LOCAL_DEV_SEED,
                key_id=LOCAL_DEV_KEY_ID,
                catalog_bytes=catalog_bytes,
                epoch=epoch,
            )
            (registry / "catalog.signature.json").write_text(
                json.dumps(dev_manifest, indent=2) + "\n", encoding="utf-8"
            )

            failures = run_gate(registry_dir=registry, allow_dev_key=False)
            self.assertTrue(
                any("LOCAL DEV key" in failure for failure in failures), failures
            )

            # With the dev key explicitly allowed, the same manifest verifies
            # (signature + hash + epoch all bind) -- only the policy check
            # stood in the way above.
            failures = run_gate(registry_dir=registry, allow_dev_key=True)
            self.assertEqual(failures, [])
        finally:
            shutil.rmtree(registry)

    def test_forged_signature_fails_verification(self) -> None:
        registry = copy_registry()
        try:
            # A manifest that RECORDS the right hash but is signed with the
            # dev key while CLAIMING the production key id: the hash check
            # passes, the signature check must not.
            catalog_bytes = (registry / "catalog.json").read_bytes()
            epoch = int((registry / "catalog.epoch").read_text().strip())
            forged = sign_manifest(
                seed=LOCAL_DEV_SEED,
                key_id=PRODUCTION_KEY_ID,
                catalog_bytes=catalog_bytes,
                epoch=epoch,
            )
            (registry / "catalog.signature.json").write_text(
                json.dumps(forged, indent=2) + "\n", encoding="utf-8"
            )
            failures = run_gate(registry_dir=registry, allow_dev_key=True)
            self.assertTrue(
                any("does not verify" in failure for failure in failures), failures
            )
        finally:
            shutil.rmtree(registry)


    def test_projection_entry_without_id_fails_closed(self) -> None:
        # A projected model that lost its id must surface a named gate
        # failure, not a KeyError traceback.
        registry = copy_registry()
        try:
            public_path = registry / "catalog.public.json"
            public = json.loads(public_path.read_text(encoding="utf-8"))
            del public["models"][0]["id"]
            public_path.write_text(json.dumps(public, indent=2) + "\n", encoding="utf-8")
            failures = run_gate(registry_dir=registry, allow_dev_key=False)
            self.assertTrue(
                any("has no string 'id' field" in failure for failure in failures),
                failures,
            )
        finally:
            shutil.rmtree(registry)

    def test_projection_comparison_is_canonical_bytes(self) -> None:
        registry = copy_registry()
        try:
            public_path = registry / "catalog.public.json"
            public = json.loads(public_path.read_text(encoding="utf-8"))

            # Key order and indentation are NOT content: the same objects
            # re-serialized with sorted keys and different indent must still
            # pass (canonical byte comparison, not raw-text comparison).
            public_path.write_text(
                json.dumps(public, indent=4, sort_keys=True) + "\n", encoding="utf-8"
            )
            check_public_projection(registry_dir=registry)

            # A changed field IS content: caught byte-precisely.
            public["models"][0]["display_name"] = "tampered"
            public_path.write_text(
                json.dumps(public, indent=2) + "\n", encoding="utf-8"
            )
            with self.assertRaises(GateFailure) as caught:
                check_public_projection(registry_dir=registry)
            self.assertIn("canonical byte comparison", str(caught.exception))
        finally:
            shutil.rmtree(registry)


class TrustRootDriftTest(unittest.TestCase):
    """The Python gate's hardcoded trust roots must equal catalog_security.rs.

    The gate deliberately avoids a cargo dependency (it runs pre-commit), so
    the constants are duplicated; this test is the drift lock.
    """

    def test_canonical_catalog_url_matches_every_authoring_site(self) -> None:
        # The canonical URL is authored in three places that must never
        # drift: the gate's identity check, the catalog generator's constant
        # (baked into catalog.json, which the signer then binds into both
        # manifests), and the committed artifacts themselves.
        committed_catalog = json.loads((REGISTRY / "catalog.json").read_text(encoding="utf-8"))
        sources = {
            "gate constant": CANONICAL_CATALOG_URL,
            "_catalog.CATALOG_URL": _catalog.CATALOG_URL,
            "catalog.json": committed_catalog["catalog_url"],
            "catalog.signature.json": json.loads(
                (REGISTRY / "catalog.signature.json").read_text(encoding="utf-8")
            )["catalog_url"],
            "catalog.public.signature.json": json.loads(
                (REGISTRY / "catalog.public.signature.json").read_text(encoding="utf-8")
            )["catalog_url"],
        }
        self.assertEqual(
            len(set(sources.values())),
            1,
            f"canonical catalog URL drifted across sources: {sources}",
        )

    def test_public_keys_and_key_ids_match_rust_source(self) -> None:
        source = RUST_SECURITY.read_text(encoding="utf-8")

        def rust_const(name: str) -> str:
            match = re.search(rf'{name}:\s*&str\s*=\s*\n?\s*"([^"]+)"', source)
            self.assertIsNotNone(match, f"{name} not found in catalog_security.rs")
            return match.group(1)

        self.assertEqual(rust_const("const CATALOG_SIGNATURE_KEY_ID"), PRODUCTION_KEY_ID)
        self.assertEqual(
            rust_const("const OPENASR_CATALOG_V1_PUBLIC_KEY_HEX"),
            PRODUCTION_PUBLIC_KEY_HEX,
        )
        self.assertEqual(
            rust_const("pub const CATALOG_SIGNATURE_LOCAL_DEV_KEY_ID"),
            LOCAL_DEV_KEY_ID,
        )
        self.assertEqual(
            rust_const("const OPENASR_CATALOG_LOCAL_DEV_PUBLIC_KEY_HEX"),
            LOCAL_DEV_PUBLIC_KEY_HEX,
        )
        self.assertIn(
            f'const CATALOG_SIGNATURE_DOMAIN: &str = "openasr.catalog_manifest.v1"',
            source,
        )


class GateCliTest(unittest.TestCase):
    def test_cli_exits_zero_on_committed_registry(self) -> None:
        result = subprocess.run(
            [sys.executable, str(SCRIPT_DIR / "check_catalog_consistency.py")],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("catalog consistency gate passed", result.stdout)


if __name__ == "__main__":
    unittest.main()
