#!/usr/bin/env python3
"""Unit tests for ModelScope URL mapping used by the sync script."""
from __future__ import annotations

import unittest

from sync_models_to_modelscope import (
    adopt_prefetch,
    download,
    iter_packs,
    modelscope_resolve_url,
    part_path,
    prefetch_pack,
    remote_sha256,
    ssl_context,
)


class ModelscopeMappingTest(unittest.TestCase):
    def test_maps_hf_resolve_to_lowercase_owner(self) -> None:
        url = (
            "https://huggingface.co/OpenASR/moonshine-tiny/resolve/"
            "0123456789abcdef0123456789abcdef01234567/moonshine-tiny-q8_0.oasr"
        )
        self.assertEqual(
            modelscope_resolve_url(url),
            "https://www.modelscope.cn/models/openasr/moonshine-tiny/resolve/"
            "master/moonshine-tiny-q8_0.oasr",
        )

    def test_rejects_non_resolve_and_traversal(self) -> None:
        self.assertIsNone(modelscope_resolve_url("https://catalog.openasr.org/v1/catalog.json"))
        self.assertIsNone(
            modelscope_resolve_url(
                "https://huggingface.co/OpenASR/evil/resolve/abc/../secrets"
            )
        )

    def test_iter_packs_skips_non_openasr_repos(self) -> None:
        packs = iter_packs(
            {
                "models": [
                    {
                        "id": "moonshine-tiny",
                        "public": True,
                        "hf_repo": "OpenASR/moonshine-tiny",
                        "hf_revision": "abc",
                        "quants": [
                            {
                                "filename": "moonshine-tiny-q8_0.oasr",
                                "url": "https://huggingface.co/OpenASR/moonshine-tiny/resolve/abc/moonshine-tiny-q8_0.oasr",
                                "sha256": "a" * 64,
                                "size_bytes": 10,
                            }
                        ],
                    },
                    {
                        "id": "private",
                        "public": False,
                        "hf_repo": "OpenASR/secret",
                        "quants": [
                            {
                                "filename": "x.oasr",
                                "url": "https://huggingface.co/OpenASR/secret/resolve/abc/x.oasr",
                                "sha256": "b" * 64,
                                "size_bytes": 10,
                            }
                        ],
                    },
                ]
            }
        )
        self.assertEqual(len(packs), 1)
        self.assertEqual(packs[0]["repo"], "moonshine-tiny")

    def test_ssl_context_uses_certifi_when_present(self) -> None:
        ctx = ssl_context()
        self.assertTrue(ctx.check_hostname)

    def test_download_rejects_content_length_mismatch(self) -> None:
        import tempfile
        import threading
        from http.server import BaseHTTPRequestHandler, HTTPServer
        from pathlib import Path

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                body = b"short"
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_args: object) -> None:
                return

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            dest = Path(tempfile.mkdtemp()) / "pack.oasr"
            with self.assertRaises(RuntimeError):
                download(f"http://127.0.0.1:{server.server_port}/pack.oasr", dest, 99)
        finally:
            server.shutdown()
            server.server_close()

    def test_download_resumes_from_partial_with_range(self) -> None:
        import tempfile
        import threading
        from http.server import BaseHTTPRequestHandler, HTTPServer
        from pathlib import Path

        payload = b"abcdefghij"

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                rng = self.headers.get("Range")
                if rng == "bytes=4-":
                    body = payload[4:]
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes 4-9/{len(payload)}")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                self.send_response(200)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *_args: object) -> None:
                return

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            dest = Path(tempfile.mkdtemp()) / "pack.oasr"
            part_path(dest).write_bytes(payload[:4])
            download(f"http://127.0.0.1:{server.server_port}/pack.oasr", dest, len(payload))
            self.assertEqual(dest.read_bytes(), payload)
            self.assertFalse(part_path(dest).exists())
        finally:
            server.shutdown()
            server.server_close()

    def test_remote_skip_uses_range_when_head_has_no_length(self) -> None:
        import threading
        from http.server import BaseHTTPRequestHandler, HTTPServer

        class Handler(BaseHTTPRequestHandler):
            def do_HEAD(self):
                self.send_response(200)
                self.end_headers()

            def do_GET(self):
                if self.headers.get("Range") != "bytes=0-0":
                    self.send_response(400)
                    self.end_headers()
                    return
                self.send_response(206)
                self.send_header("Content-Range", "bytes 0-0/1490916416")
                self.send_header("Content-Length", "1")
                self.end_headers()
                self.wfile.write(b"x")

            def log_message(self, *_args: object) -> None:
                return

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_port}/qwen3-asr-1.7b-q4_k.oasr"
            self.assertEqual(remote_sha256(url, 1490916416), "size-match")
            self.assertIsNone(remote_sha256(url, 1))
        finally:
            server.shutdown()
            server.server_close()

    def test_remote_skip_missing_object_is_none(self) -> None:
        import threading
        from http.server import BaseHTTPRequestHandler, HTTPServer

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(404)
                self.end_headers()

            def log_message(self, *_args: object) -> None:
                return

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_port}/missing.oasr"
            self.assertIsNone(remote_sha256(url, 100))
        finally:
            server.shutdown()
            server.server_close()

    def test_prefetch_skips_uploader_part_file(self) -> None:
        import tempfile
        from pathlib import Path

        dest = Path(tempfile.mkdtemp()) / "pack.oasr"
        part_path(dest).write_bytes(b"busy")
        result = prefetch_pack(
            {
                "url": "http://127.0.0.1:1/pack.oasr",
                "sha256": "a" * 64,
                "size_bytes": 10,
                "ms_url": None,
            },
            dest,
        )
        self.assertEqual(result, "skip-busy")

    def test_adopt_prefetch_renames_completed_sidecar(self) -> None:
        import hashlib
        import tempfile
        from pathlib import Path

        dest = Path(tempfile.mkdtemp()) / "pack.oasr"
        payload = b"prefetch-bytes"
        sidecar = dest.with_suffix(dest.suffix + ".prefetch")
        sidecar.write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        self.assertTrue(adopt_prefetch(dest, digest, wait_seconds=1))
        self.assertEqual(dest.read_bytes(), payload)
        self.assertFalse(sidecar.exists())

    def test_cache_root_honors_env_and_avoids_workstation_paths(self) -> None:
        import os
        import tempfile
        from pathlib import Path

        from sync_models_to_modelscope import cache_root

        with tempfile.TemporaryDirectory() as td:
            os.environ["OPENASR_MODELSCOPE_CACHE"] = td
            try:
                self.assertEqual(cache_root(), Path(td))
            finally:
                del os.environ["OPENASR_MODELSCOPE_CACHE"]


if __name__ == "__main__":
    unittest.main()
