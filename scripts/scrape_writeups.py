#!/usr/bin/env python3
"""Fetch writeup URLs into the local scrape cache and emit per-file metadata."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


DEFAULT_TIMEOUT = 20
DEFAULT_USER_AGENT = (
    "qwen-ctfer-model/1.0 (+local research cache; manual provenance review required)"
)

EXTENSION_BY_CONTENT_TYPE = {
    "text/html": ".html",
    "text/plain": ".txt",
    "text/markdown": ".md",
    "application/json": ".json",
    "application/pdf": ".pdf",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch URLs into data/raw/scraped and write a per-batch metadata manifest. "
            "This is a local cache helper, not a training-data importer."
        )
    )
    parser.add_argument(
        "--url",
        action="append",
        default=[],
        help="URL to fetch. Can be repeated.",
    )
    parser.add_argument(
        "--url-file",
        default=None,
        help="Optional newline-delimited file of URLs to fetch.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/raw/scraped",
        help="Base directory for scrape-cache batches.",
    )
    parser.add_argument(
        "--batch-name",
        default=None,
        help="Optional batch directory name. Defaults to scrape_<UTC timestamp>.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--user-agent",
        default=DEFAULT_USER_AGENT,
        help="HTTP User-Agent header.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing batch directory.",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable TLS certificate verification. Use only when the local CA store is broken or the target is trusted.",
    )
    parser.add_argument(
        "--save-body",
        action="store_true",
        help="Save fetched bodies. Disabled only if metadata-only probing is desired.",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Fetch headers/body for status and metadata but do not write body files.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=10,
        help="Maximum number of parallel fetches. Default: 10",
    )
    parser.add_argument(
        "--cache-dir",
        default="data/raw/scraped/.cache",
        help="Shared cache directory used across scrape batches.",
    )
    parser.add_argument(
        "--cache-mode",
        choices=("reuse", "refresh", "off"),
        default="reuse",
        help="Cache behavior. Default: reuse",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    urls = unique(read_urls(args.url, args.url_file))
    if not urls:
        raise SystemExit("No URLs provided. Use --url or --url-file.")

    repo_root = Path.cwd()
    output_root = resolve_path(repo_root, Path(args.output_dir))
    cache_root = resolve_path(repo_root, Path(args.cache_dir))
    batch_name = args.batch_name or default_batch_name()
    batch_dir = output_root / batch_name
    body_dir = batch_dir / "files"
    metadata_path = batch_dir / "fetch_manifest.json"
    cache_body_dir = cache_root / "files"
    cache_index_path = cache_root / "index.json"

    if batch_dir.exists() and not args.overwrite:
        raise SystemExit(
            f"Batch directory already exists: {batch_dir}. Use --overwrite or a new --batch-name."
        )
    if args.max_concurrency < 1:
        raise SystemExit("--max-concurrency must be at least 1")

    batch_dir.mkdir(parents=True, exist_ok=True)
    if not args.metadata_only and args.save_body:
        body_dir.mkdir(parents=True, exist_ok=True)
    if args.cache_mode != "off":
        cache_body_dir.mkdir(parents=True, exist_ok=True)

    client = build_http_client()
    cache_index = load_cache_index(cache_index_path) if args.cache_mode != "off" else {}
    records = fetch_all(
        client=client,
        urls=urls,
        body_dir=body_dir,
        cache_index=cache_index,
        cache_body_dir=cache_body_dir,
        timeout=args.timeout,
        user_agent=args.user_agent,
        write_body=bool(args.save_body and not args.metadata_only),
        insecure=args.insecure,
        max_concurrency=args.max_concurrency,
        cache_mode=args.cache_mode,
    )
    if args.cache_mode != "off":
        save_cache_index(cache_index_path, cache_index)

    metadata = {
        "created_at": iso_now(),
        "batch_name": batch_name,
        "output_dir": str(batch_dir),
        "write_body": bool(args.save_body and not args.metadata_only),
        "max_concurrency": args.max_concurrency,
        "cache_mode": args.cache_mode,
        "cache_dir": str(cache_root),
        "records": records,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote metadata manifest: {metadata_path}")
    return 0


def build_http_client():
    try:
        import requests

        return ("requests", requests)
    except ImportError:
        import urllib.request

        return ("urllib", urllib.request)


def fetch_one(
    *,
    client,
    url: str,
    index: int,
    body_dir: Path,
    cache_index: dict[str, Any] | None,
    cache_body_dir: Path | None,
    timeout: float,
    user_agent: str,
    write_body: bool,
    insecure: bool,
    cache_mode: str,
) -> dict[str, Any]:
    if cache_mode == "reuse" and cache_index is not None:
        cached = get_cached_record(
            url=url,
            index=index,
            body_dir=body_dir,
            write_body=write_body,
            cache_index=cache_index,
            cache_body_dir=cache_body_dir,
        )
        if cached is not None:
            return cached

    mode, module = client
    fetched_at = iso_now()
    record: dict[str, Any] = {
        "url": url,
        "fetched_at": fetched_at,
        "index": index,
        "status": "error",
    }

    try:
        if mode == "requests":
            response = module.get(
                url,
                headers={"User-Agent": user_agent},
                timeout=timeout,
                allow_redirects=True,
                verify=not insecure,
            )
            content = response.content
            status_code = response.status_code
            headers = dict(response.headers)
            final_url = response.url
        else:
            if insecure:
                import ssl

                ssl_context = ssl._create_unverified_context()
            else:
                ssl_context = None
            request = module.Request(url, headers={"User-Agent": user_agent})
            with module.urlopen(request, timeout=timeout, context=ssl_context) as response:
                content = response.read()
                status_code = getattr(response, "status", response.getcode())
                headers = dict(response.headers.items())
                final_url = response.geturl()
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        return record

    content_type = normalize_content_type(headers.get("Content-Type"))
    sha256 = hashlib.sha256(content).hexdigest()
    filename = build_filename(index=index, final_url=final_url, content_type=content_type)

    record.update(
        {
            "status": "ok",
            "status_code": int(status_code),
            "final_url": final_url,
            "content_type": content_type,
            "content_length": len(content),
            "sha256": sha256,
            "headers": {
                "content_type": headers.get("Content-Type"),
                "content_length": headers.get("Content-Length"),
                "last_modified": headers.get("Last-Modified"),
                "etag": headers.get("ETag"),
            },
            "body_path": str((body_dir / filename)) if write_body else None,
        }
    )

    if write_body:
        (body_dir / filename).write_bytes(content)

    if cache_mode != "off" and cache_index is not None and cache_body_dir is not None:
        cache_payload_body(
            url=url,
            final_url=final_url,
            record=record,
            content=content,
            cache_index=cache_index,
            cache_body_dir=cache_body_dir,
        )

    return record


def fetch_all(
    *,
    client,
    urls: list[str],
    body_dir: Path,
    cache_index: dict[str, Any] | None,
    cache_body_dir: Path | None,
    timeout: float,
    user_agent: str,
    write_body: bool,
    insecure: bool,
    max_concurrency: int,
    cache_mode: str,
) -> list[dict[str, Any]]:
    total = len(urls)
    if total == 0:
        return []
    if max_concurrency == 1 or total == 1:
        records: list[dict[str, Any]] = []
        for index, url in enumerate(urls, start=1):
            record = fetch_one(
                client=client,
                url=url,
                index=index,
                body_dir=body_dir,
                cache_index=cache_index,
                cache_body_dir=cache_body_dir,
                timeout=timeout,
                user_agent=user_agent,
                write_body=write_body,
                insecure=insecure,
                cache_mode=cache_mode,
            )
            records.append(record)
            print(f"[{index}/{total}] {record['status']} {url}")
        return records

    records_by_index: dict[int, dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(max_concurrency, total)
    ) as executor:
        future_to_meta = {
            executor.submit(
                fetch_one,
                client=client,
                url=url,
                index=index,
                body_dir=body_dir,
                cache_index=cache_index,
                cache_body_dir=cache_body_dir,
                timeout=timeout,
                user_agent=user_agent,
                write_body=write_body,
                insecure=insecure,
                cache_mode=cache_mode,
            ): (index, url)
            for index, url in enumerate(urls, start=1)
        }
        for future in concurrent.futures.as_completed(future_to_meta):
            index, url = future_to_meta[future]
            record = future.result()
            records_by_index[index] = record
            print(f"[{index}/{total}] {record['status']} {url}")

    return [records_by_index[index] for index in sorted(records_by_index)]


def read_urls(cli_urls: list[str], url_file: str | None) -> list[str]:
    urls = [item.strip().lstrip("\ufeff") for item in cli_urls if item.strip()]
    if url_file:
        path = Path(url_file)
        for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
            line = raw_line.strip().lstrip("\ufeff")
            if not line or line.startswith("#"):
                continue
            urls.append(line)
    return urls


def unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def load_cache_index(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def save_cache_index(path: Path, cache_index: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache_index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def make_cache_key(url: str) -> str:
    return url.strip()


def get_cached_record(
    *,
    url: str,
    index: int,
    body_dir: Path,
    write_body: bool,
    cache_index: dict[str, Any],
    cache_body_dir: Path | None,
) -> dict[str, Any] | None:
    entry = cache_index.get(make_cache_key(url))
    if not isinstance(entry, dict):
        return None
    if entry.get("status") != "ok":
        return None

    record = {
        "url": url,
        "fetched_at": entry.get("fetched_at"),
        "index": index,
        "status": "cached",
        "status_code": entry.get("status_code"),
        "final_url": entry.get("final_url"),
        "content_type": entry.get("content_type"),
        "content_length": entry.get("content_length"),
        "sha256": entry.get("sha256"),
        "headers": entry.get("headers", {}),
        "body_path": None,
        "cache_hit": True,
        "cached_from": entry.get("cached_at"),
    }

    cached_body_name = entry.get("cached_body_name")
    if write_body and cached_body_name and cache_body_dir is not None:
        cached_body_path = cache_body_dir / cached_body_name
        if cached_body_path.exists():
            target_name = build_filename(
                index=index,
                final_url=str(entry.get("final_url") or url),
                content_type=str(entry.get("content_type") or "application/octet-stream"),
            )
            target_path = body_dir / target_name
            shutil.copy2(cached_body_path, target_path)
            record["body_path"] = str(target_path)
    return record


def cache_payload_body(
    *,
    url: str,
    final_url: str,
    record: dict[str, Any],
    content: bytes,
    cache_index: dict[str, Any],
    cache_body_dir: Path,
) -> None:
    sha256 = str(record["sha256"])
    extension = EXTENSION_BY_CONTENT_TYPE.get(str(record["content_type"]), ".bin")
    cached_body_name = f"{sha256}{extension}"
    cached_body_path = cache_body_dir / cached_body_name
    if not cached_body_path.exists():
        cached_body_path.write_bytes(content)

    entry = {
        "url": url,
        "status": "ok",
        "fetched_at": record.get("fetched_at"),
        "cached_at": iso_now(),
        "status_code": record.get("status_code"),
        "final_url": final_url,
        "content_type": record.get("content_type"),
        "content_length": record.get("content_length"),
        "sha256": sha256,
        "headers": record.get("headers", {}),
        "cached_body_name": cached_body_name,
    }
    cache_index[make_cache_key(url)] = entry


def resolve_path(repo_root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def default_batch_name() -> str:
    return "scrape_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalize_content_type(value: str | None) -> str:
    if not value:
        return "application/octet-stream"
    return value.split(";", 1)[0].strip().lower()


def build_filename(*, index: int, final_url: str, content_type: str) -> str:
    parsed = urlparse(final_url)
    base = Path(parsed.path).name or parsed.netloc or "page"
    base = sanitize_name(base)
    if "." not in base:
        base += EXTENSION_BY_CONTENT_TYPE.get(content_type, ".bin")
    return f"{index:03d}_{base}"


def sanitize_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    cleaned = cleaned.strip("._-")
    return cleaned or "file"


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130)
