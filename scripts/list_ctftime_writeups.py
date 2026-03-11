#!/usr/bin/env python3
"""Enumerate CTFtime writeup URLs into a reusable target list."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urljoin


CTFTIME_WRITEUPS_URL = "https://ctftime.org/writeups"
DEFAULT_USER_AGENT = (
    "qwen-ctfer-model/1.0 (+research enumeration for authorized CTF dataset building)"
)
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
WRITEUP_LINK_RE = re.compile(r'href="(/writeup/\d+)"')
ORIGINAL_LINK_RE = re.compile(
    r'Original writeup(?:</a>)?\s*\((?:<a[^>]+href=")?(https?://[^")<\s]+)',
    re.IGNORECASE,
)
CANONICAL_LINK_RE = re.compile(r'<link rel="canonical" href="([^"]+)"')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Enumerate CTFtime writeup page URLs and optionally resolve their "
            "original external writeup links."
        )
    )
    parser.add_argument("--start-page", type=int, default=1)
    parser.add_argument("--max-pages", type=int, default=1)
    parser.add_argument("--include-original", action="store_true")
    parser.add_argument(
        "--target-source",
        choices=("ctftime", "original", "both"),
        default="ctftime",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=3,
        help="Maximum number of parallel detail-page fetches when resolving original URLs.",
    )
    parser.add_argument("--output-txt", required=True)
    parser.add_argument("--output-jsonl", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    txt_path = Path(args.output_txt).resolve()
    jsonl_path = Path(args.output_jsonl).resolve() if args.output_jsonl else None
    for output_path in [txt_path, jsonl_path]:
        if output_path is None:
            continue
        if output_path.exists() and not args.overwrite:
            raise SystemExit(f"Refusing to overwrite existing file: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)

    client = build_http_client()
    writeup_urls = enumerate_ctftime_writeup_pages(
        client=client,
        start_page=args.start_page,
        max_pages=args.max_pages,
    )
    rows: list[dict[str, Any]] = []
    target_urls: list[str] = []
    seen_targets: set[str] = set()

    rows = build_rows(
        client=client,
        writeup_urls=writeup_urls,
        include_original=args.include_original,
        max_concurrency=args.max_concurrency,
    )

    for row in rows:
        for target_url in select_target_urls(row, args.target_source):
            if target_url not in seen_targets:
                seen_targets.add(target_url)
                target_urls.append(target_url)

    txt_path.write_text("\n".join(target_urls) + ("\n" if target_urls else ""), encoding="utf-8")
    if jsonl_path:
        with jsonl_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "start_page": args.start_page,
                "max_pages": args.max_pages,
                "writeup_pages": len(rows),
                "target_urls": len(target_urls),
                "original_errors": sum(1 for row in rows if row.get("original_error")),
                "output_txt": str(txt_path),
                "output_jsonl": str(jsonl_path) if jsonl_path else None,
                "target_source": args.target_source,
            },
            indent=2,
        )
    )
    return 0


def build_http_client():
    try:
        import requests

        return ("requests", requests)
    except ImportError:
        import urllib.request

        return ("urllib", urllib.request)


def fetch_text(client: tuple[str, Any], url: str, *, attempts: int = 4) -> str:
    mode, module = client
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            if mode == "requests":
                response = module.get(url, timeout=30, headers={"User-Agent": DEFAULT_USER_AGENT})
                if response.status_code in RETRYABLE_STATUS_CODES and attempt < attempts:
                    time.sleep(attempt)
                    continue
                response.raise_for_status()
                return response.text

            request = module.Request(url, headers={"User-Agent": DEFAULT_USER_AGENT})
            with module.urlopen(request, timeout=30) as response:
                return response.read().decode("utf-8", errors="replace")
        except Exception as exc:
            last_error = exc
            status_code = getattr(exc, "code", None)
            if status_code in RETRYABLE_STATUS_CODES and attempt < attempts:
                time.sleep(attempt)
                continue
            raise
    assert last_error is not None
    raise last_error


def build_rows(
    *,
    client: tuple[str, Any],
    writeup_urls: list[str],
    include_original: bool,
    max_concurrency: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [
        {
            "index": index,
            "ctftime_writeup_url": writeup_url,
            "original_url": None,
            "original_error": None,
        }
        for index, writeup_url in enumerate(writeup_urls, start=1)
    ]
    if not include_original or not rows:
        return rows

    max_workers = max(1, min(max_concurrency, len(rows)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(resolve_original_url, client, row["ctftime_writeup_url"]): row["index"]
            for row in rows
        }
        for future in concurrent.futures.as_completed(future_to_index):
            row_index = future_to_index[future] - 1
            rows[row_index].update(future.result())
    return rows


def resolve_original_url(client: tuple[str, Any], writeup_url: str) -> dict[str, Any]:
    try:
        html = fetch_text(client, writeup_url)
    except Exception as exc:
        return {
            "original_url": None,
            "original_error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "original_url": extract_original_writeup_url(html),
        "original_error": None,
    }


def enumerate_ctftime_writeup_pages(
    *, client: tuple[str, Any], start_page: int, max_pages: int
) -> list[str]:
    discovered: list[str] = []
    seen: set[str] = set()
    for page in range(start_page, start_page + max_pages):
        page_url = CTFTIME_WRITEUPS_URL if page == 1 else f"{CTFTIME_WRITEUPS_URL}?page={page}"
        html = fetch_text(client, page_url)
        for relative_url in WRITEUP_LINK_RE.findall(html):
            absolute_url = urljoin(CTFTIME_WRITEUPS_URL, relative_url)
            if absolute_url not in seen:
                seen.add(absolute_url)
                discovered.append(absolute_url)
    return discovered


def extract_original_writeup_url(html: str) -> str | None:
    canonical_match = CANONICAL_LINK_RE.search(html)
    canonical_url = canonical_match.group(1) if canonical_match else None
    match = ORIGINAL_LINK_RE.search(html)
    if not match:
        return None
    original_url = match.group(1)
    if canonical_url and original_url == canonical_url:
        return None
    return original_url


def select_target_urls(row: dict[str, Any], target_source: str) -> list[str]:
    ctftime_url = row["ctftime_writeup_url"]
    original_url = row.get("original_url")
    if target_source == "ctftime":
        return [ctftime_url]
    if target_source == "original":
        return [original_url] if original_url else []
    targets = [ctftime_url]
    if original_url:
        targets.append(original_url)
    return targets


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130)
