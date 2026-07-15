#!/usr/bin/env python3
"""Sync core release assets to Backblaze B2 behind https://dl.openasr.org.

  b2_sync.py sync --version <semver> \\
      dist/openasr-<v>-windows-x86_64-vulkan.zip \\
      dist/openasr-<v>-windows-x86_64-cuda.zip \\
      dist/openasr-<v>-windows-x86_64-rocm.zip \\
      dist/backends-manifest.json \\
      [dist/backends-manifest.signature.json]

Uploads each given file to `core/v<version>/<basename>` in the SAME B2
bucket/Cloudflare-Worker setup `openasr-app`'s desktop installers already use
(see that repo's `apps/desktop/scripts/release-publish.mjs` and
`b2-s3-client.mjs`, which this script's SigV4 signer is a Python port of --
`desktop/releases/v<version>/...` there, `core/v<version>/...` here). Uses
THE SAME environment variable names as that script:

  B2_S3_ENDPOINT        Required. e.g. https://s3.us-east-005.backblazeb2.com
  B2_APPLICATION_KEY_ID Required.
  B2_APPLICATION_KEY    Required. Never logged.
  B2_BUCKET             Default: openasr-releases
  B2_S3_REGION          Override if it cannot be inferred from B2_S3_ENDPOINT.

Immutability, same policy as the desktop publish script: before uploading a
key, HEAD it. If an object already exists there with a DIFFERENT sha256, this
aborts rather than silently overwriting a shipped release asset -- bump the
version instead. Re-uploading byte-identical content is a no-op.

This is deliberately NOT wired into any GitHub Actions workflow yet -- see
tooling/release-manifest/README.md's "dl.openasr.org sync" section for why
(credential/bucket-sharing decision, not a technical blocker).
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import os
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, urlparse

ALGORITHM = "AWS4-HMAC-SHA256"
SERVICE = "s3"
DEFAULT_BUCKET = "openasr-releases"
DEFAULT_DL_BASE_URL = "https://dl.openasr.org"


class B2SyncError(Exception):
    pass


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hmac(key: bytes, data: str) -> bytes:
    return hmac.new(key, data.encode("utf-8"), hashlib.sha256).digest()


def amz_date_parts(date: datetime) -> tuple[str, str]:
    amz_date = date.strftime("%Y%m%dT%H%M%SZ")
    return amz_date, amz_date[:8]


def derive_signing_key(secret_access_key: str, date_stamp: str, region: str, service: str = SERVICE) -> bytes:
    k_date = _hmac(f"AWS4{secret_access_key}".encode("utf-8"), date_stamp)
    k_region = _hmac(k_date, region)
    k_service = _hmac(k_region, service)
    return _hmac(k_service, "aws4_request")


# SigV4 requires RFC 3986 percent-encoding, which differs from
# urllib.parse.quote's default `safe` set -- keep `!'()*` encoded too.
def _encode_rfc3986(component: str) -> str:
    return quote(component, safe="")


def canonical_uri(pathname: str) -> str:
    return "/".join(_encode_rfc3986(segment) for segment in pathname.split("/"))


def build_canonical_request(
    method: str, pathname: str, headers: dict[str, str], payload_hash: str
) -> tuple[str, str]:
    signed_header_names = sorted(name.lower() for name in headers)
    canonical_headers = "".join(f"{name}:{str(headers[name]).strip()}\n" for name in signed_header_names)
    signed_headers = ";".join(signed_header_names)
    canonical_request = "\n".join(
        [method, canonical_uri(pathname), "", canonical_headers, signed_headers, payload_hash]
    )
    return canonical_request, signed_headers


@dataclass
class SignedRequest:
    authorization: str
    amz_date: str


def sign_request(
    *,
    method: str,
    host: str,
    pathname: str,
    headers: dict[str, str],
    payload_hash: str,
    access_key_id: str,
    secret_access_key: str,
    region: str,
    date: datetime,
) -> SignedRequest:
    amz_date, date_stamp = amz_date_parts(date)
    headers_to_sign = {**headers, "host": host, "x-amz-date": amz_date, "x-amz-content-sha256": payload_hash}

    canonical_request, signed_headers = build_canonical_request(method, pathname, headers_to_sign, payload_hash)

    credential_scope = f"{date_stamp}/{region}/{SERVICE}/aws4_request"
    string_to_sign = "\n".join([ALGORITHM, amz_date, credential_scope, sha256_hex(canonical_request.encode("utf-8"))])
    signing_key = derive_signing_key(secret_access_key, date_stamp, region)
    signature = hmac.new(signing_key, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()

    authorization = (
        f"{ALGORITHM} Credential={access_key_id}/{credential_scope}, "
        f"SignedHeaders={signed_headers}, Signature={signature}"
    )
    return SignedRequest(authorization=authorization, amz_date=amz_date)


def region_from_endpoint(endpoint: str, override: str | None) -> str:
    if override:
        return override
    host = urlparse(endpoint).netloc
    # e.g. s3.us-east-005.backblazeb2.com -> us-east-005
    match = re.match(r"s3\.([^.]+)\.", host)
    if match:
        return match.group(1)
    raise B2SyncError(
        f"Could not infer the B2/S3 region from endpoint host '{host}'. Set B2_S3_REGION explicitly."
    )


class Transport:
    """Minimal HTTP transport seam so tests can inject a fake instead of
    making real network calls."""

    def request(self, method: str, url: str, headers: dict[str, str], body: bytes) -> tuple[int, dict[str, str]]:
        request = urllib.request.Request(url, data=body if method != "HEAD" else None, method=method, headers=headers)
        try:
            with urllib.request.urlopen(request) as response:  # noqa: S310 (fixed https B2 endpoint)
                return response.status, dict(response.headers)
        except urllib.error.HTTPError as error:
            return error.code, dict(error.headers or {})


@dataclass
class B2Credentials:
    endpoint: str
    access_key_id: str
    secret_access_key: str
    bucket: str = DEFAULT_BUCKET
    region: str | None = None

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "B2Credentials":
        env = env if env is not None else os.environ
        endpoint = env.get("B2_S3_ENDPOINT")
        access_key_id = env.get("B2_APPLICATION_KEY_ID")
        secret_access_key = env.get("B2_APPLICATION_KEY")
        if not endpoint:
            raise B2SyncError(
                "B2_S3_ENDPOINT is not set. The bucket's region/cluster is not known until it's "
                "created; set this to the bucket's S3-compatible endpoint, e.g. "
                "https://s3.us-east-005.backblazeb2.com."
            )
        if not access_key_id or not secret_access_key:
            raise B2SyncError("B2_APPLICATION_KEY_ID and B2_APPLICATION_KEY must both be set to sync.")
        return cls(
            endpoint=endpoint,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            bucket=env.get("B2_BUCKET", DEFAULT_BUCKET),
            region=env.get("B2_S3_REGION"),
        )


class B2Client:
    def __init__(self, credentials: B2Credentials, transport: Transport | None = None):
        self.credentials = credentials
        self.transport = transport or Transport()
        self.region = region_from_endpoint(credentials.endpoint, credentials.region)
        endpoint_url = urlparse(credentials.endpoint)
        # Virtual-hosted-style (`https://<bucket>.<endpoint-host>/<key>`), NOT
        # path-style (`https://<endpoint-host>/<bucket>/<key>`) -- matches
        # openasr-app's b2-s3-client.mjs `s3ObjectRequest`/`virtualHostedUrl`,
        # which this is a Python port of. The bucket is part of the signed
        # `host`, so getting this wrong produces a signature B2 rejects, not
        # just a wrong-looking URL.
        self.scheme = endpoint_url.scheme
        self.host = f"{credentials.bucket}.{endpoint_url.netloc}"

    def _object_url(self, key: str) -> str:
        return f"{self.scheme}://{self.host}/{canonical_uri(key)}"

    def _signed_headers(
        self, method: str, key: str, payload_hash: str, extra_headers: dict[str, str]
    ) -> dict[str, str]:
        signed = sign_request(
            method=method,
            host=self.host,
            pathname=f"/{key}",
            headers=extra_headers,
            payload_hash=payload_hash,
            access_key_id=self.credentials.access_key_id,
            secret_access_key=self.credentials.secret_access_key,
            region=self.region,
            date=datetime.now(timezone.utc),
        )
        return {
            **extra_headers,
            "Authorization": signed.authorization,
            "x-amz-date": signed.amz_date,
            "x-amz-content-sha256": payload_hash,
        }

    def head_object(self, key: str) -> dict[str, str] | None:
        """Returns response headers if the object exists, None on 404, raises
        on any other non-2xx status."""
        payload_hash = sha256_hex(b"")
        headers = self._signed_headers("HEAD", key, payload_hash, {})
        status, response_headers = self.transport.request("HEAD", self._object_url(key), headers, b"")
        if status == 404:
            return None
        if not 200 <= status < 300:
            raise B2SyncError(f"HEAD {key} failed with status {status}")
        return response_headers

    def put_object(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
        payload_hash = sha256_hex(data)
        # content-type/content-length are part of the SIGNED headers here
        # (passed as extra_headers into sign_request), matching
        # b2-s3-client.mjs's s3ObjectRequest -- not just attached afterward.
        extra_headers = {"content-type": content_type, "content-length": str(len(data))}
        headers = self._signed_headers("PUT", key, payload_hash, extra_headers)
        status, _ = self.transport.request("PUT", self._object_url(key), headers, data)
        if not 200 <= status < 300:
            raise B2SyncError(f"PUT {key} failed with status {status}")

    def public_url(self, key: str, dl_base_url: str = DEFAULT_DL_BASE_URL) -> str:
        return f"{dl_base_url.rstrip('/')}/{key}"


def _normalize_etag(etag: str | None) -> str:
    return (etag or "").replace('"', "").lower()


def decide_immutability_action(remote: dict[str, object] | None, local_size: int, local_md5_hex: str) -> str:
    """Mirrors release-publish.mjs's `decideImmutabilityAction`: "put" if
    nothing is there yet, "skip" if an identical object already is (same
    size + ETag/md5 -- an idempotent re-run), or raises if a DIFFERENT
    object already occupies the key (never silently overwrite a shipped
    release asset; the caller must bump the version instead)."""
    if remote is None:
        return "put"
    remote_size = remote.get("size")
    remote_etag = _normalize_etag(remote.get("etag"))  # type: ignore[arg-type]
    if remote_size == local_size and remote_etag == local_md5_hex.lower():
        return "skip"
    raise B2SyncError(
        f"refusing to overwrite existing object with different content "
        f"(remote size={remote_size} etag={remote_etag!r} vs local size={local_size} "
        f"md5={local_md5_hex!r}); bump the version instead of re-publishing under the same one"
    )


def sync_files(
    client: B2Client, version: str, files: list[Path], dl_base_url: str = DEFAULT_DL_BASE_URL
) -> list[str]:
    """Uploads each file to `core/v<version>/<basename>`, honoring the same
    immutability gate release-publish.mjs uses (abort on a differing existing
    object; allow an idempotent re-upload of identical bytes). Returns the
    public dl.openasr.org URLs, in input order."""
    urls = []
    for path in files:
        if not path.is_file():
            raise B2SyncError(f"file not found: {path}")
        data = path.read_bytes()
        key = f"core/v{version}/{path.name}"

        head = client.head_object(key)
        remote = None
        if head is not None:
            content_length = head.get("Content-Length") or head.get("content-length")
            remote = {
                "size": int(content_length) if content_length is not None else None,
                "etag": head.get("ETag") or head.get("etag"),
            }
        local_md5_hex = hashlib.md5(data).hexdigest()  # noqa: S324 (B2/S3 ETag comparison, not a security use)
        action = decide_immutability_action(remote, len(data), local_md5_hex)
        if action == "put":
            client.put_object(key, data)

        urls.append(client.public_url(key, dl_base_url))
    return urls


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    sync = subparsers.add_parser("sync", help="Upload files to core/v<version>/ on B2")
    sync.add_argument("--version", required=True, help="Release semver, e.g. 0.1.10")
    sync.add_argument("--dl-base-url", default=DEFAULT_DL_BASE_URL)
    sync.add_argument("files", nargs="+", type=Path)

    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.command == "sync":
        try:
            credentials = B2Credentials.from_env()
            client = B2Client(credentials)
            urls = sync_files(client, args.version, args.files, args.dl_base_url)
        except B2SyncError as error:
            print(f"b2_sync.py: {error}", file=sys.stderr)
            return 1
        for url in urls:
            print(url)
        return 0
    raise SystemExit(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
