#!/usr/bin/env python3
"""Dump Hy-MT2 tokenizer parity goldens from the pinned HF tokenizer.json."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tokenizers import Tokenizer


CASES = {
    "zh_subtitle_fast_path": "我们需要保持流式路径很快。",
    "zh_with_digits": "第12集将在2026年6月上线。",
    "ja_clause": "これは字幕のテストです。",
    "en_punct_spacing": "Keep the streaming path fast, please.",
    "mixed_emoji": "OpenASR v2.1：延迟 < 300ms 🙂",
    "role_literal": "<｜hy_User｜>不要翻译成角色名。",
    "zwj_format_char": "zwj ‍abc",
}

BOS_ID = 120000
USER_ID = 120006
ASSISTANT_ID = 120007


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tokenizer_json", type=Path)
    parser.add_argument(
        "--revision",
        default="9a341cd1b679d3efd23b46e847b01745a71ed792",
    )
    args = parser.parse_args()

    tokenizer_bytes = args.tokenizer_json.read_bytes()
    tokenizer_payload = json.loads(tokenizer_bytes)
    tokenizer = Tokenizer.from_file(str(args.tokenizer_json))
    content_tokenizer = tokenizer_without_special_added_tokens(tokenizer_payload)
    payload = {
        "tokenizer_repo": "tencent/Hy-MT2-1.8B",
        "tokenizer_revision": args.revision,
        "tokenizer_json_sha256": hashlib.sha256(tokenizer_bytes).hexdigest(),
        "cases": [],
    }
    for name, text in CASES.items():
        token_ids = tokenizer.encode(text, add_special_tokens=False).ids
        content_token_ids = content_tokenizer.encode(text, add_special_tokens=False).ids
        payload["cases"].append(
            {
                "name": name,
                "text": text,
                "token_ids": token_ids,
                "chat_token_ids": [BOS_ID, USER_ID, *content_token_ids, ASSISTANT_ID],
            }
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def tokenizer_without_special_added_tokens(tokenizer_payload: dict) -> Tokenizer:
    content_payload = dict(tokenizer_payload)
    content_payload["added_tokens"] = [
        token
        for token in tokenizer_payload.get("added_tokens", [])
        if not token.get("special", False)
    ]
    return Tokenizer.from_str(json.dumps(content_payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
