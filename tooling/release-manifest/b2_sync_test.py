from __future__ import annotations

import hashlib
import unittest
from datetime import datetime, timezone

from b2_sync import (
    B2Client,
    B2Credentials,
    B2SyncError,
    amz_date_parts,
    build_canonical_request,
    canonical_uri,
    decide_immutability_action,
    region_from_endpoint,
    sha256_hex,
    sign_request,
    sync_files,
    sync_vendor_files,
)

# AWS's own published worked example for SigV4 header-based auth (GET Object,
# examplebucket/test.txt) -- the same canonical cross-implementation vector
# openasr-app's b2-s3-client.test.mjs pins for its JS SigV4 signer (this
# module is a Python port of that signer). Matching the identical published
# Authorization header is real evidence the signing math is correct,
# independent of both implementations and of B2 itself.
# https://docs.aws.amazon.com/AmazonS3/latest/API/sig-v4-header-based-auth.html
AWS_EXAMPLE = {
    "access_key_id": "AKIAIOSFODNN7EXAMPLE",
    "secret_access_key": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
    "region": "us-east-1",
    "date": datetime(2013, 5, 24, tzinfo=timezone.utc),
    "host": "examplebucket.s3.amazonaws.com",
    "pathname": "/test.txt",
    "payload_hash": sha256_hex(b""),
}


class Sha256Md5Test(unittest.TestCase):
    def test_sha256_of_empty_payload(self) -> None:
        self.assertEqual(
            sha256_hex(b""), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        )


class AmzDatePartsTest(unittest.TestCase):
    def test_formats_amz_date_and_date_stamp(self) -> None:
        self.assertEqual(amz_date_parts(AWS_EXAMPLE["date"]), ("20130524T000000Z", "20130524"))


class CanonicalUriTest(unittest.TestCase):
    def test_preserves_slashes_and_percent_encodes_spaces(self) -> None:
        self.assertEqual(
            canonical_uri("/core/v0.1.10/openasr-0.1.10-windows-x86_64-vulkan.zip"),
            "/core/v0.1.10/openasr-0.1.10-windows-x86_64-vulkan.zip",
        )
        self.assertEqual(canonical_uri("/a b/c"), "/a%20b/c")


class BuildCanonicalRequestTest(unittest.TestCase):
    def test_matches_aws_published_canonical_request(self) -> None:
        canonical_request, signed_headers = build_canonical_request(
            "GET",
            AWS_EXAMPLE["pathname"],
            {
                "host": AWS_EXAMPLE["host"],
                "range": "bytes=0-9",
                "x-amz-content-sha256": AWS_EXAMPLE["payload_hash"],
                "x-amz-date": "20130524T000000Z",
            },
            AWS_EXAMPLE["payload_hash"],
        )

        self.assertEqual(signed_headers, "host;range;x-amz-content-sha256;x-amz-date")
        self.assertEqual(
            canonical_request,
            "\n".join(
                [
                    "GET",
                    "/test.txt",
                    "",
                    "host:examplebucket.s3.amazonaws.com",
                    "range:bytes=0-9",
                    f"x-amz-content-sha256:{AWS_EXAMPLE['payload_hash']}",
                    "x-amz-date:20130524T000000Z",
                    "",
                    "host;range;x-amz-content-sha256;x-amz-date",
                    AWS_EXAMPLE["payload_hash"],
                ]
            ),
        )


class SignRequestTest(unittest.TestCase):
    def test_reproduces_aws_published_authorization_header(self) -> None:
        signed = sign_request(
            method="GET",
            host=AWS_EXAMPLE["host"],
            pathname=AWS_EXAMPLE["pathname"],
            headers={"range": "bytes=0-9"},
            payload_hash=AWS_EXAMPLE["payload_hash"],
            access_key_id=AWS_EXAMPLE["access_key_id"],
            secret_access_key=AWS_EXAMPLE["secret_access_key"],
            region=AWS_EXAMPLE["region"],
            date=AWS_EXAMPLE["date"],
        )

        self.assertEqual(signed.amz_date, "20130524T000000Z")
        self.assertEqual(
            signed.authorization,
            "AWS4-HMAC-SHA256 Credential=AKIAIOSFODNN7EXAMPLE/20130524/us-east-1/s3/aws4_request, "
            "SignedHeaders=host;range;x-amz-content-sha256;x-amz-date, "
            "Signature=f0e8bdb87c964420e857bd35b5d6ed310bd44f0170aba48dd91039c6036bdb41",
        )

    def test_deterministic_for_a_fixed_date_and_changes_with_the_secret(self) -> None:
        base = dict(
            method="HEAD",
            host="openasr-releases.s3.us-east-005.backblazeb2.com",
            pathname="/core/v0.1.10/backends-manifest.json",
            payload_hash=sha256_hex(b""),
            access_key_id="keyid",
            region="us-east-005",
            date=datetime(2026, 7, 3, tzinfo=timezone.utc),
        )
        first = sign_request(headers={}, secret_access_key="secret-a", **base)
        again = sign_request(headers={}, secret_access_key="secret-a", **base)
        different_key = sign_request(headers={}, secret_access_key="secret-b", **base)

        self.assertEqual(first.authorization, again.authorization)
        self.assertNotEqual(first.authorization, different_key.authorization)
        self.assertTrue(
            first.authorization.startswith(
                "AWS4-HMAC-SHA256 Credential=keyid/20260703/us-east-005/s3/aws4_request"
            )
        )


class RegionFromEndpointTest(unittest.TestCase):
    def test_infers_region_from_backblaze_endpoint(self) -> None:
        self.assertEqual(
            region_from_endpoint("https://s3.us-east-005.backblazeb2.com", None), "us-east-005"
        )

    def test_override_wins(self) -> None:
        self.assertEqual(
            region_from_endpoint("https://s3.us-east-005.backblazeb2.com", "eu-central-003"),
            "eu-central-003",
        )

    def test_raises_when_it_cannot_infer(self) -> None:
        with self.assertRaises(B2SyncError):
            region_from_endpoint("https://example.com", None)


class DecideImmutabilityActionTest(unittest.TestCase):
    def test_put_when_nothing_remote(self) -> None:
        self.assertEqual(decide_immutability_action(None, 10, "abc"), "put")

    def test_skip_when_identical(self) -> None:
        md5 = hashlib.md5(b"same-bytes").hexdigest()
        remote = {"size": len(b"same-bytes"), "etag": f'"{md5}"'}
        self.assertEqual(decide_immutability_action(remote, len(b"same-bytes"), md5), "skip")

    def test_raises_when_different(self) -> None:
        remote = {"size": 999, "etag": '"deadbeef"'}
        with self.assertRaises(B2SyncError):
            decide_immutability_action(remote, 10, "abc123")


class FakeTransport:
    """In-memory stand-in for b2_sync.Transport -- lets sync_files be tested
    without any real network access."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.requests: list[tuple[str, str]] = []

    def request(self, method, url, headers, body):
        self.requests.append((method, url))
        # Key is whatever comes after the host in the URL path.
        key = url.split("/", 3)[-1]
        if method == "HEAD":
            if key not in self.objects:
                return 404, {}
            data = self.objects[key]
            return 200, {
                "Content-Length": str(len(data)),
                "ETag": f'"{hashlib.md5(data).hexdigest()}"',
            }
        if method == "PUT":
            self.objects[key] = body
            return 200, {}
        raise AssertionError(f"unexpected method: {method}")


def _fake_client() -> tuple[B2Client, FakeTransport]:
    transport = FakeTransport()
    credentials = B2Credentials(
        endpoint="https://s3.us-east-005.backblazeb2.com",
        access_key_id="keyid",
        secret_access_key="secret",
        bucket="openasr-releases",
    )
    return B2Client(credentials, transport=transport), transport


class SyncFilesTest(unittest.TestCase):
    def setUp(self) -> None:
        import tempfile
        from pathlib import Path

        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dist_dir = Path(self._tmp.name)

    def _write(self, name: str, content: bytes):
        path = self.dist_dir / name
        path.write_bytes(content)
        return path

    def test_uploads_new_files_under_core_v_version_prefix(self) -> None:
        client, transport = _fake_client()
        path = self._write("backends-manifest.json", b'{"schema_version":1}')

        urls = sync_files(client, "0.1.10", [path])

        self.assertEqual(urls, ["https://dl.openasr.org/core/v0.1.10/backends-manifest.json"])
        self.assertIn("core/v0.1.10/backends-manifest.json", transport.objects)
        self.assertEqual(
            transport.objects["core/v0.1.10/backends-manifest.json"], b'{"schema_version":1}'
        )

    def test_idempotent_reupload_of_identical_bytes_is_a_no_op(self) -> None:
        client, transport = _fake_client()
        path = self._write("backends-manifest.json", b"same-bytes")
        sync_files(client, "0.1.10", [path])
        put_count_after_first = sum(1 for method, _ in transport.requests if method == "PUT")

        sync_files(client, "0.1.10", [path])
        put_count_after_second = sum(1 for method, _ in transport.requests if method == "PUT")

        self.assertEqual(put_count_after_first, 1)
        self.assertEqual(put_count_after_second, 1)  # no second PUT

    def test_refuses_to_overwrite_a_different_object_at_the_same_key(self) -> None:
        client, transport = _fake_client()
        path = self._write("backends-manifest.json", b"version-one-bytes")
        sync_files(client, "0.1.10", [path])

        path.write_bytes(b"version-one-bytes-but-different")
        with self.assertRaises(B2SyncError):
            sync_files(client, "0.1.10", [path])

    def test_missing_local_file_fails_loudly(self) -> None:
        client, _ = _fake_client()
        with self.assertRaises(B2SyncError):
            sync_files(client, "0.1.10", [self.dist_dir / "does-not-exist.zip"])


class SyncVendorFilesTest(unittest.TestCase):
    def setUp(self) -> None:
        import tempfile
        from pathlib import Path

        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dist_dir = Path(self._tmp.name)

    def _write(self, name: str, content: bytes):
        path = self.dist_dir / name
        path.write_bytes(content)
        return path

    def test_uploads_under_content_addressed_vendor_prefix_not_a_version(self) -> None:
        client, transport = _fake_client()
        content = b"fake-cuda-runtime-bytes"
        sha256 = hashlib.sha256(content).hexdigest()
        path = self._write(f"openasr-vendor-cuda-runtime-{sha256[:12]}.zip", content)

        urls = sync_vendor_files(client, [path])

        expected_key = f"core/vendor/{sha256}/{path.name}"
        self.assertEqual(urls, [f"https://dl.openasr.org/{expected_key}"])
        self.assertIn(expected_key, transport.objects)
        self.assertEqual(transport.objects[expected_key], content)

    def test_idempotent_reupload_of_identical_bytes_is_a_no_op(self) -> None:
        client, transport = _fake_client()
        path = self._write("openasr-vendor-rocm-runtime-aaaaaaaaaaaa.zip", b"same-vendor-bytes")
        sync_vendor_files(client, [path])
        put_count_after_first = sum(1 for method, _ in transport.requests if method == "PUT")

        sync_vendor_files(client, [path])
        put_count_after_second = sum(1 for method, _ in transport.requests if method == "PUT")

        self.assertEqual(put_count_after_first, 1)
        self.assertEqual(put_count_after_second, 1)  # no second PUT: same bytes, same content-addressed key

    def test_two_different_vendor_archives_land_at_two_different_keys(self) -> None:
        # Distinct content addresses to distinct keys -- there is no shared
        # "current version" key for two different vendor archives to collide
        # under, unlike sync_files' per-version prefix.
        client, transport = _fake_client()
        cuda_path = self._write("openasr-vendor-cuda-runtime-000000000000.zip", b"cuda-bytes")
        rocm_path = self._write("openasr-vendor-rocm-runtime-000000000000.zip", b"rocm-bytes")

        sync_vendor_files(client, [cuda_path, rocm_path])

        cuda_sha = hashlib.sha256(b"cuda-bytes").hexdigest()
        rocm_sha = hashlib.sha256(b"rocm-bytes").hexdigest()
        self.assertIn(f"core/vendor/{cuda_sha}/{cuda_path.name}", transport.objects)
        self.assertIn(f"core/vendor/{rocm_sha}/{rocm_path.name}", transport.objects)

    def test_missing_local_file_fails_loudly(self) -> None:
        client, _ = _fake_client()
        with self.assertRaises(B2SyncError):
            sync_vendor_files(client, [self.dist_dir / "does-not-exist.zip"])


if __name__ == "__main__":
    unittest.main()
