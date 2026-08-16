#!/usr/bin/env python3
"""Catalog consistency gate: fail closed BEFORE a catalog edit commits when
the committed catalog JSONs and their signed manifests disagree.

The trust boundary problem this gate exists to close: model-registry/
catalog.json can be regenerated without re-signing, and nothing used to
notice until the (much later) Rust test suite verified the embedded
signatures. A catalog edit that lands without its re-signature ships a
hash mismatch to every client. This gate runs in seconds, needs no cargo,
and belongs in CI and the pre-commit hook:

  1. BINDING -- each manifest's recorded catalog_sha256 equals the sha256 of
     its own catalog file (catalog.json <-> catalog.signature.json,
     catalog.public.json <-> catalog.public.signature.json).
  2. EPOCH -- both manifests carry the epoch recorded in catalog.epoch.
  3. IDENTITY -- every catalog_url agrees (manifests, catalog files, and the
     canonical production URL for the committed files).
  4. SIGNATURE -- the ed25519 signature verifies (pure-stdlib RFC 8032, no
     deps) against the trust root its key_id names. Committed files must be
     PRODUCTION-signed; a dev-key manifest in model-registry/ is itself an
     incident (sign_local_catalog.sh's output is a never-commit preview).
  5. PROJECTION -- catalog.public.json is exactly the public:true subset of
     catalog.json (canonically serialized byte-equal model objects -- sorted
     keys, minimal separators -- plus matching header fields), so the
     embedded/served catalog cannot silently drift from the source.

Usage:
  check_catalog_consistency.py [--registry-dir DIR] [--allow-dev-key]

Exit code is the number of failed checks (capped at 255); zero is a fully
consistent, production-signed registry state.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from _pathlib_helpers import repo_root

SCRIPT_DIR = Path(__file__).resolve().parent

# --- trust roots (mirror crates/openasr-core/src/catalog_security.rs) -------
#
# These are the PUBLIC keys; the single source of truth is catalog_security.rs
# and check_catalog_consistency_test.py fails if the two drift.
CATALOG_SIGNATURE_DOMAIN = "openasr.catalog_manifest.v1"
CATALOG_SIGNATURE_ALGORITHM = "ed25519"
PRODUCTION_KEY_ID = "openasr-catalog-v1"
PRODUCTION_PUBLIC_KEY_HEX = "92331f1048a70b70fb00818f263b4f532ff536f21b7e86df2eb11c175105c2ad"
LOCAL_DEV_KEY_ID = "openasr-catalog-local-dev-v1"
LOCAL_DEV_PUBLIC_KEY_HEX = "bc1306d4cc4a1cbc817a862ee0223713ff79208c39bc8ce732da851db3c6b6a1"
TRUST_ROOTS = {
    PRODUCTION_KEY_ID: PRODUCTION_PUBLIC_KEY_HEX,
    LOCAL_DEV_KEY_ID: LOCAL_DEV_PUBLIC_KEY_HEX,
}
CANONICAL_CATALOG_URL = "https://catalog.openasr.org/v1/catalog.json"


# --- minimal RFC 8032 Ed25519 (verify + sign), stdlib only ------------------

_ED_P = 2**255 - 19
_ED_L = 2**252 + 27742317777372353535851937790883648493
_ED_D = -121665 * pow(121666, _ED_P - 2, _ED_P) % _ED_P
_ED_I = pow(2, (_ED_P - 1) // 4, _ED_P)


def _ed_xrecover(y: int) -> int:
    xx = (y * y - 1) * pow(_ED_D * y * y + 1, _ED_P - 2, _ED_P)
    x = pow(xx, (_ED_P + 3) // 8, _ED_P)
    if (x * x - xx) % _ED_P != 0:
        x = (x * _ED_I) % _ED_P
    if x % 2 != 0:
        x = _ED_P - x
    return x


_ED_BY = 4 * pow(5, _ED_P - 2, _ED_P) % _ED_P
_ED_BX = _ed_xrecover(_ED_BY)
_ED_B = (_ED_BX % _ED_P, _ED_BY % _ED_P)


def _ed_add(point_a, point_b):
    # Twisted Edwards addition on -x^2 + y^2 = 1 + d x^2 y^2 (the RFC 8032
    # curve), via the standard unified formula with modular inverse.
    x1, y1 = point_a
    x2, y2 = point_b
    x3 = (x1 * y2 + x2 * y1) * pow(1 + _ED_D * x1 * x2 * y1 * y2, _ED_P - 2, _ED_P)
    y3 = (y1 * y2 + x1 * x2) * pow(1 - _ED_D * x1 * x2 * y1 * y2, _ED_P - 2, _ED_P)
    return (x3 % _ED_P, y3 % _ED_P)


def _ed_scalar_mult(point, scalar):
    result = (0, 1)
    addend = point
    while scalar:
        if scalar & 1:
            result = _ed_add(result, addend)
        addend = _ed_add(addend, addend)
        scalar >>= 1
    return result


def _ed_bit(h: bytes, position: int) -> int:
    return (h[position // 8] >> (position % 8)) & 1


def _ed_on_curve(point) -> bool:
    x, y = point
    return (-x * x + y * y - 1 - _ED_D * x * x * y * y) % _ED_P == 0


def _ed_decode_point(encoded: bytes):
    y = int.from_bytes(encoded, "little")
    sign = y >> 255
    y &= (1 << 255) - 1
    x = _ed_xrecover(y)
    if x & 1 != sign:
        x = _ED_P - x
    point = (x, y)
    if not _ed_on_curve(point):
        raise ValueError("point is not on the ed25519 curve")
    return point


def _ed_encode_point(point) -> bytes:
    x, y = point
    encoded = (y | ((x & 1) << 255)).to_bytes(32, "little")
    return encoded


def _ed_secret_scalar(seed: bytes) -> int:
    digest = hashlib.sha512(seed).digest()
    scalar = int.from_bytes(digest[:32], "little")
    scalar &= (1 << 254) - 8
    scalar |= 1 << 254
    return scalar


def ed25519_public_key(seed: bytes) -> bytes:
    return _ed_encode_point(_ed_scalar_mult(_ED_B, _ed_secret_scalar(seed)))


def ed25519_sign(seed: bytes, message: bytes) -> bytes:
    digest = hashlib.sha512(seed).digest()
    scalar = _ed_secret_scalar(seed)
    public = _ed_encode_point(_ed_scalar_mult(_ED_B, scalar))
    nonce = int.from_bytes(
        hashlib.sha512(digest[32:64] + message).digest(), "little"
    ) % _ED_L
    r_point = _ed_scalar_mult(_ED_B, nonce)
    k = int.from_bytes(
        hashlib.sha512(_ed_encode_point(r_point) + public + message).digest(), "little"
    ) % _ED_L
    s = (nonce + k * scalar) % _ED_L
    return _ed_encode_point(r_point) + s.to_bytes(32, "little")


def ed25519_verify(public_key: bytes, message: bytes, signature: bytes) -> bool:
    if len(signature) != 64 or len(public_key) != 32:
        return False
    try:
        r_point = _ed_decode_point(signature[:32])
        a_point = _ed_decode_point(public_key)
    except ValueError:
        return False
    s = int.from_bytes(signature[32:], "little")
    if s >= _ED_L:
        return False
    k = int.from_bytes(
        hashlib.sha512(signature[:32] + public_key + message).digest(), "little"
    ) % _ED_L
    left = _ed_scalar_mult(_ED_B, s)
    right = _ed_add(r_point, _ed_scalar_mult(a_point, k))
    return left == right


# --- signing payload (mirror catalog_security.rs::catalog_signing_payload) --


def catalog_signing_payload(
    *,
    key_id: str,
    catalog_url: str,
    catalog_sha256: str,
    catalog_epoch: int,
) -> bytes:
    return (
        f"{CATALOG_SIGNATURE_DOMAIN}\n"
        f"algorithm:{CATALOG_SIGNATURE_ALGORITHM}\n"
        f"key_id:{key_id}\n"
        f"catalog_url:{catalog_url}\n"
        f"catalog_sha256:{catalog_sha256}\n"
        f"catalog_epoch:{catalog_epoch}\n"
    ).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --- the gate ----------------------------------------------------------------


class GateFailure(Exception):
    """One failed consistency check (message is the diagnostic)."""


def load_manifest(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise GateFailure(f"signature manifest missing: {path}") from error
    except json.JSONDecodeError as error:
        raise GateFailure(f"signature manifest is not valid JSON: {path}: {error}") from error
    if not isinstance(data, dict):
        raise GateFailure(f"signature manifest is not a JSON object: {path}")
    return data


def check_manifest_binding(
    *,
    label: str,
    catalog_path: Path,
    manifest_path: Path,
    allow_dev_key: bool,
) -> None:
    """Checks 1-4 for one (catalog, manifest) pair."""
    try:
        catalog_bytes = catalog_path.read_bytes()
    except FileNotFoundError as error:
        raise GateFailure(f"{label}: catalog file missing: {catalog_path}") from error
    manifest = load_manifest(manifest_path)

    signature = manifest.get("signature")
    if not isinstance(signature, dict):
        raise GateFailure(f"{label}: manifest has no signature object: {manifest_path}")

    algorithm = signature.get("algorithm")
    if algorithm != CATALOG_SIGNATURE_ALGORITHM:
        raise GateFailure(f"{label}: unsupported signature algorithm {algorithm!r}")

    key_id = signature.get("key_id")
    if key_id not in TRUST_ROOTS:
        raise GateFailure(f"{label}: unknown signing key id {key_id!r}")
    if key_id == LOCAL_DEV_KEY_ID and not allow_dev_key:
        raise GateFailure(
            f"{label}: committed manifest is signed with the LOCAL DEV key; "
            "restore the production-signed manifest (rerun publish_catalog.sh) "
            "before committing"
        )

    recorded_sha = manifest.get("catalog_sha256")
    actual_sha = sha256_hex(catalog_bytes)
    if recorded_sha != actual_sha:
        raise GateFailure(
            f"{label}: catalog content hash mismatch -- the catalog changed "
            f"without a re-signature (recorded {recorded_sha}, actual {actual_sha}); "
            "rerun publish_catalog.sh"
        )

    try:
        catalog_doc = json.loads(catalog_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GateFailure(f"{label}: catalog is not valid JSON: {catalog_path}: {error}") from error

    catalog_url = manifest.get("catalog_url")
    catalog_doc_url = catalog_doc.get("catalog_url") if isinstance(catalog_doc, dict) else None
    if catalog_url != CANONICAL_CATALOG_URL:
        raise GateFailure(
            f"{label}: manifest catalog_url {catalog_url!r} is not the canonical "
            f"production identity {CANONICAL_CATALOG_URL!r}"
        )
    if catalog_doc_url != catalog_url:
        raise GateFailure(
            f"{label}: catalog file catalog_url {catalog_doc_url!r} disagrees with "
            f"the manifest's {catalog_url!r}"
        )

    epoch = manifest.get("catalog_epoch")
    if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch <= 0:
        raise GateFailure(f"{label}: manifest catalog_epoch must be a positive integer, got {epoch!r}")

    signature_value = signature.get("value")
    if not isinstance(signature_value, str):
        raise GateFailure(f"{label}: manifest signature value must be a hex string")
    try:
        signature_bytes = bytes.fromhex(signature_value)
    except ValueError as error:
        raise GateFailure(f"{label}: manifest signature value is not hex: {error}") from error

    payload = catalog_signing_payload(
        key_id=key_id,
        catalog_url=catalog_url,
        catalog_sha256=recorded_sha,
        catalog_epoch=epoch,
    )
    public_key = bytes.fromhex(TRUST_ROOTS[key_id])
    if not ed25519_verify(public_key, payload, signature_bytes):
        raise GateFailure(
            f"{label}: ed25519 signature does not verify under {key_id}; "
            "the manifest or its catalog was tampered with or mis-signed"
        )


def check_epoch_file(*, registry_dir: Path, manifests: list[Path]) -> None:
    """Check 2: catalog.epoch agrees with every manifest's catalog_epoch."""
    epoch_path = registry_dir / "catalog.epoch"
    try:
        epoch_text = epoch_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError as error:
        raise GateFailure(f"catalog epoch file missing: {epoch_path}") from error
    if not epoch_text.isdigit() or int(epoch_text) <= 0:
        raise GateFailure(f"catalog.epoch must be a positive integer, got {epoch_text!r}")
    recorded_epoch = int(epoch_text)
    for manifest_path in manifests:
        manifest = load_manifest(manifest_path)
        manifest_epoch = manifest.get("catalog_epoch")
        if manifest_epoch != recorded_epoch:
            raise GateFailure(
                f"{manifest_path.name}: catalog_epoch {manifest_epoch!r} disagrees with "
                f"catalog.epoch ({recorded_epoch}); bump the epoch and re-sign"
            )


def canonical_json_bytes(value: object) -> bytes:
    """Canonical serialization for byte comparison: sorted keys, minimal
    separators. Two catalog entries compare byte-equal iff they carry the
    same content -- on-disk indentation/key order never matters, but a
    dropped/changed/added field always flips a byte."""
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def models_by_id(catalog_doc: object, label: str, *, require_public: bool) -> dict:
    """Index a catalog's models[] by id, failing closed (named GateFailure,
    never KeyError) on any malformed entry: a projection entry without an id
    is exactly the kind of corruption this gate must name, not traceback."""
    if not isinstance(catalog_doc, dict):
        raise GateFailure(f"{label}: catalog is not a JSON object")
    models = catalog_doc.get("models")
    if not isinstance(models, list):
        raise GateFailure(f"{label}: catalog 'models' must be a list")
    by_id: dict = {}
    for index, model in enumerate(models):
        if not isinstance(model, dict):
            raise GateFailure(f"{label}: models[{index}] is not a JSON object")
        model_id = model.get("id")
        if not isinstance(model_id, str) or not model_id:
            raise GateFailure(f"{label}: models[{index}] has no string 'id' field")
        if require_public and model.get("public") is not True:
            continue
        if model_id in by_id:
            raise GateFailure(f"{label}: duplicate model id {model_id!r}")
        by_id[model_id] = model
    return by_id


def check_public_projection(*, registry_dir: Path) -> None:
    """Check 5: catalog.public.json is exactly catalog.json's public:true subset."""
    full = json.loads((registry_dir / "catalog.json").read_text(encoding="utf-8"))
    public = json.loads((registry_dir / "catalog.public.json").read_text(encoding="utf-8"))

    for field in ("schema_version", "generated_at", "catalog_url"):
        if full.get(field) != public.get(field):
            raise GateFailure(
                f"public projection top-level field {field!r} drifted from the full "
                "catalog; rerun publish_catalog.sh"
            )
    if full.get("language_labels") != public.get("language_labels"):
        raise GateFailure(
            "public projection language_labels drifted from the full catalog; "
            "rerun publish_catalog.sh"
        )
    if canonical_json_bytes(full.get("backends", [])) != canonical_json_bytes(
        public.get("backends", [])
    ):
        raise GateFailure(
            "public projection backends drifted from the full catalog; "
            "rerun publish_catalog.sh"
        )

    full_public_models = models_by_id(full, "catalog.json", require_public=True)
    projected_models = models_by_id(
        public, "catalog.public.json", require_public=False
    )
    if set(projected_models) != set(full_public_models):
        missing = sorted(set(full_public_models) - set(projected_models))
        extra = sorted(set(projected_models) - set(full_public_models))
        raise GateFailure(
            "public projection model set drifted from the full catalog's "
            f"public:true set (missing={missing}, extra={extra}); rerun publish_catalog.sh"
        )
    for model_id, full_model in sorted(full_public_models.items()):
        projected = projected_models[model_id]
        if canonical_json_bytes(projected) != canonical_json_bytes(full_model):
            raise GateFailure(
                f"public projection model {model_id!r} differs (canonical byte "
                "comparison) from its full-catalog entry; rerun publish_catalog.sh"
            )


def run_gate(*, registry_dir: Path, allow_dev_key: bool) -> list[str]:
    """Run every check; returns the list of failure messages (empty = green)."""
    failures: list[str] = []
    pairs = [
        ("catalog.json", "catalog.signature.json"),
        ("catalog.public.json", "catalog.public.signature.json"),
    ]
    for catalog_name, manifest_name in pairs:
        try:
            check_manifest_binding(
                label=manifest_name,
                catalog_path=registry_dir / catalog_name,
                manifest_path=registry_dir / manifest_name,
                allow_dev_key=allow_dev_key,
            )
        except GateFailure as failure:
            failures.append(str(failure))

    try:
        check_epoch_file(
            registry_dir=registry_dir,
            manifests=[registry_dir / name for _, name in pairs],
        )
    except GateFailure as failure:
        failures.append(str(failure))

    if not (registry_dir / "catalog.public.json").exists():
        failures.append(f"public projection missing: {registry_dir / 'catalog.public.json'}")
    else:
        try:
            check_public_projection(registry_dir=registry_dir)
        except GateFailure as failure:
            failures.append(str(failure))

    return failures


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry-dir",
        type=Path,
        default=None,
        help="model-registry directory (default: <repo>/model-registry)",
    )
    parser.add_argument(
        "--allow-dev-key",
        action="store_true",
        help="accept the local-dev signing key (preview workflows only; "
        "committed state must be production-signed)",
    )
    args = parser.parse_args(argv)

    registry_dir = args.registry_dir
    if registry_dir is None:
        registry_dir = repo_root(SCRIPT_DIR) / "model-registry"
    registry_dir = registry_dir.resolve()

    failures = run_gate(registry_dir=registry_dir, allow_dev_key=args.allow_dev_key)
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        print(
            f"catalog consistency gate: {len(failures)} check(s) failed in {registry_dir}",
            file=sys.stderr,
        )
        return min(len(failures), 255)

    print(f"catalog consistency gate passed: {registry_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
