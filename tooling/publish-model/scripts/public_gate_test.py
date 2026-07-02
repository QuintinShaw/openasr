#!/usr/bin/env python3
from __future__ import annotations

import ssl
import socket
import unittest
import urllib.error

from public_gate import HeadResult, PublicGateError, PublicProbe, validate_public_model


REVISION = "b" * 40
SHA256 = "a" * 64
URL = f"https://huggingface.co/OpenASR/moonshine-tiny/resolve/{REVISION}/moonshine-tiny-q8_0.oasr"


def valid_model() -> dict:
    return {
        "id": "moonshine-tiny",
        "hf_repo": "OpenASR/moonshine-tiny",
        "hf_revision": REVISION,
        "public": True,
        "min_cli_version": "0.1.0",
        "quants": [
            {
                "quant": "q8_0",
                "filename": "moonshine-tiny-q8_0.oasr",
                "url": URL,
                "sha256": SHA256,
                "size_bytes": 3,
                "perf": {
                    "rtf_cpu": 0.1,
                    "rtf_metal": None,
                    "peak_rss_bytes": 1,
                    "jfk_wer_vs_fp16": 0.0,
                },
            }
        ],
    }


class FakeProbe:
    def __init__(
        self,
        *,
        info: dict | None = None,
        head: HeadResult | None = None,
        download: tuple[str, int] | None = None,
    ) -> None:
        self.info = info if info is not None else {"id": "OpenASR/moonshine-tiny", "private": False}
        self.head_result = head or HeadResult(
            status=200,
            headers={"content-length": "3", "x-repo-commit": REVISION},
            final_url=URL,
        )
        self.download = download or (SHA256, 3)
        self.model_info_calls: list[str] = []
        self.head_calls: list[str] = []
        self.sha256_calls: list[str] = []

    def model_info(self, repo_id: str) -> dict:
        self.model_info_calls.append(repo_id)
        return self.info

    def head(self, url: str) -> HeadResult:
        self.head_calls.append(url)
        return self.head_result

    def sha256(self, url: str) -> tuple[str, int]:
        self.sha256_calls.append(url)
        return self.download


class FakeResponse:
    def __init__(self, *, status: int, headers: dict[str, str], url: str, body: bytes = b"x") -> None:
        self.status = status
        self.headers = headers
        self._url = url
        self.body = body
        self.read_calls: list[int] = []

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:  # type: ignore[no-untyped-def]
        return None

    def geturl(self) -> str:
        return self._url

    def read(self, size: int = -1) -> bytes:
        self.read_calls.append(size)
        if size == -1:
            return self.body
        if size == 0:
            return b""
        return self.body[:size] or b"x"


class TransportProbe(PublicProbe):
    def __init__(self, events: list[BaseException | FakeResponse], *, head_attempts: int = 1) -> None:
        super().__init__(timeout=0.01, head_attempts=head_attempts)
        self.events = events
        self.requests: list[tuple[str, str | None]] = []
        self.sleep_calls: list[float] = []

    def _open(self, _opener, request):  # type: ignore[no-untyped-def]
        self.requests.append((request.get_method(), request.get_header("Range")))
        event = self.events.pop(0)
        if isinstance(event, BaseException):
            raise event
        return event

    def _sleep(self, seconds: float) -> None:
        self.sleep_calls.append(seconds)


class ModelInfoProbe(PublicProbe):
    def __init__(self, events: list[BaseException | FakeResponse], *, attempts: int = 2) -> None:
        super().__init__(timeout=0.01, head_attempts=attempts)
        self.events = events
        self.urls: list[str] = []
        self.sleep_calls: list[float] = []

    def _urlopen(self, request):  # type: ignore[no-untyped-def]
        self.urls.append(request.full_url)
        event = self.events.pop(0)
        if isinstance(event, BaseException):
            raise event
        return event

    def _sleep(self, seconds: float) -> None:
        self.sleep_calls.append(seconds)


def http_error(status: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(URL, status, "simulated", {}, None)


class PublicGateTest(unittest.TestCase):
    def test_valid_public_entry_checks_each_remote_surface(self) -> None:
        probe = FakeProbe()

        validate_public_model(valid_model(), probe=probe)

        self.assertEqual(probe.model_info_calls, ["OpenASR/moonshine-tiny"])
        self.assertEqual(probe.head_calls, [URL])
        self.assertEqual(probe.sha256_calls, [URL])

    def test_rejects_mirror_metadata(self) -> None:
        model = valid_model()
        model["quants"][0]["mirrors"] = [
            {
                "source": "modelscope",
                "url": f"https://modelscope.cn/models/openasr/moonshine-tiny/resolve/{REVISION}/moonshine-tiny-q8_0.oasr",
            }
        ]

        with self.assertRaisesRegex(PublicGateError, "mirror sources are not supported"):
            validate_public_model(model, probe=FakeProbe())

    def test_rejects_wrong_namespace(self) -> None:
        model = valid_model()
        model["hf_repo"] = "not-the-canonical-org/moonshine-tiny"
        model["quants"][0]["url"] = f"https://huggingface.co/not-the-canonical-org/moonshine-tiny/resolve/{REVISION}/moonshine-tiny-q8_0.oasr"

        with self.assertRaisesRegex(PublicGateError, "canonical namespace"):
            validate_public_model(model, probe=FakeProbe())

    def test_rejects_private_repo(self) -> None:
        with self.assertRaisesRegex(PublicGateError, "private"):
            validate_public_model(valid_model(), probe=FakeProbe(info={"private": True}))

    def test_rejects_probe_failure(self) -> None:
        probe = FakeProbe(head=HeadResult(status=404, headers={}, final_url=URL))

        with self.assertRaisesRegex(PublicGateError, "pack probe returned HTTP 404"):
            validate_public_model(valid_model(), probe=probe)

    def test_head_transport_error_falls_back_to_ranged_get(self) -> None:
        transport = TransportProbe(
            [
                socket.timeout("simulated timeout"),
                FakeResponse(
                    status=206,
                    headers={
                        "content-length": "1",
                        "content-range": "bytes 0-0/3",
                        "x-repo-commit": REVISION,
                        "x-linked-etag": f'"{SHA256}"',
                    },
                    url=URL,
                ),
            ]
        )

        result = transport.head(URL)

        self.assertEqual(result.status, 206)
        self.assertEqual(result.headers["content-range"], "bytes 0-0/3")
        self.assertEqual(transport.requests, [("HEAD", None), ("GET", "bytes=0-0")])

    def test_head_503_backs_off_then_retries(self) -> None:
        transport = TransportProbe(
            [
                http_error(503),
                FakeResponse(
                    status=200,
                    headers={
                        "content-length": "3",
                        "x-repo-commit": REVISION,
                        "x-linked-etag": f'"{SHA256}"',
                    },
                    url=URL,
                ),
            ],
            head_attempts=2,
        )

        result = transport.head(URL)

        self.assertEqual(result.status, 200)
        self.assertEqual(transport.requests, [("HEAD", None), ("HEAD", None)])
        self.assertEqual(transport.sleep_calls, [0.25])

    def test_model_info_socket_timeout_backs_off_then_retries(self) -> None:
        probe = ModelInfoProbe(
            [
                socket.timeout("slow model_info"),
                FakeResponse(
                    status=200,
                    headers={},
                    url=URL,
                    body=b'{"id": "OpenASR/moonshine-tiny", "private": false}',
                ),
            ],
            attempts=2,
        )

        info = probe.model_info("OpenASR/moonshine-tiny")

        self.assertEqual(info["private"], False)
        self.assertEqual(len(probe.urls), 2)
        self.assertEqual(probe.sleep_calls, [0.25])

    def test_model_info_503_backs_off_then_retries(self) -> None:
        probe = ModelInfoProbe(
            [
                http_error(503),
                FakeResponse(
                    status=200,
                    headers={},
                    url=URL,
                    body=b'{"id": "OpenASR/moonshine-tiny", "private": false}',
                ),
            ],
            attempts=2,
        )

        info = probe.model_info("OpenASR/moonshine-tiny")

        self.assertEqual(info["private"], False)
        self.assertEqual(len(probe.urls), 2)
        self.assertEqual(probe.sleep_calls, [0.25])

    def test_model_info_persistent_404_fails_closed_without_retry(self) -> None:
        probe = ModelInfoProbe([http_error(404)], attempts=3)

        with self.assertRaisesRegex(PublicGateError, "HTTP 404"):
            probe.model_info("OpenASR/moonshine-tiny")

        self.assertEqual(len(probe.urls), 1)
        self.assertEqual(probe.sleep_calls, [])

    def test_ranged_get_metadata_can_satisfy_public_gate(self) -> None:
        probe = FakeProbe(
            head=HeadResult(
                status=206,
                headers={
                    "content-length": "1",
                    "content-range": "bytes 0-0/3",
                    "x-repo-commit": REVISION,
                    "x-linked-etag": f'"{SHA256}"',
                },
                final_url=URL,
            )
        )

        validate_public_model(valid_model(), probe=probe)

        self.assertEqual(probe.sha256_calls, [])

    def test_rejects_sha_mismatch(self) -> None:
        probe = FakeProbe(download=("c" * 64, 3))

        with self.assertRaisesRegex(PublicGateError, "HF sha256"):
            validate_public_model(valid_model(), probe=probe)

    def test_uses_lfs_head_sha_without_downloading(self) -> None:
        probe = FakeProbe(
            head=HeadResult(
                status=200,
                headers={
                    "content-length": "3",
                    "x-repo-commit": REVISION,
                    "x-linked-etag": f'"{SHA256}"',
                },
                final_url=URL,
            )
        )

        validate_public_model(valid_model(), probe=probe)

        self.assertEqual(probe.sha256_calls, [])

    def test_does_not_trust_plain_etag_as_pack_sha(self) -> None:
        probe = FakeProbe(
            head=HeadResult(
                status=200,
                headers={
                    "content-length": "3",
                    "x-repo-commit": REVISION,
                    "etag": '"' + SHA256 + '"',
                },
                final_url=URL,
            )
        )

        validate_public_model(valid_model(), probe=probe)

        self.assertEqual(probe.sha256_calls, [URL])

    def test_rejects_lfs_head_sha_mismatch(self) -> None:
        probe = FakeProbe(
            head=HeadResult(
                status=200,
                headers={
                    "content-length": "3",
                    "x-repo-commit": REVISION,
                    "x-linked-etag": '"' + ("c" * 64) + '"',
                },
                final_url=URL,
            )
        )

        with self.assertRaisesRegex(PublicGateError, "HF sha256"):
            validate_public_model(valid_model(), probe=probe)

    def test_rejects_non_ga_min_cli(self) -> None:
        model = valid_model()
        model["min_cli_version"] = "0.1.0-rc.1"

        with self.assertRaisesRegex(PublicGateError, "GA semver"):
            validate_public_model(model, probe=FakeProbe())

    def test_rejects_missing_metrics(self) -> None:
        model = valid_model()
        model["quants"][0]["perf"] = {
            "rtf_cpu": None,
            "rtf_metal": None,
            "peak_rss_bytes": None,
            "jfk_wer_vs_fp16": None,
        }

        with self.assertRaisesRegex(PublicGateError, "perf metrics"):
            validate_public_model(model, probe=FakeProbe())


if __name__ == "__main__":
    unittest.main()
