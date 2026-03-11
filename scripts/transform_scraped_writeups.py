#!/usr/bin/env python3
"""Build transformed text records from approved scraped writeup sources."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


DEFAULT_USER_AGENT = (
    "qwen-ctfer-model/1.0 (+authorized scraped writeup transformation for local model training)"
)
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
GITHUB_HOSTS = {"github.com", "www.github.com", "raw.githubusercontent.com"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read a scrape review ledger and emit transformed source records using "
            "GitHub raw URLs when possible and r.jina.ai for other sites."
        )
    )
    parser.add_argument("--review-ledger", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-concurrency", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--include-unapproved",
        action="store_true",
        help="Transform all successful rows instead of only allowed_for_training=true rows.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    review_ledger_path = Path(args.review_ledger).resolve()
    if not review_ledger_path.exists():
        raise SystemExit(f"Review ledger not found: {review_ledger_path}")

    output_path = (
        Path(args.output).resolve()
        if args.output
        else review_ledger_path.parent / "transformed_writeups.jsonl"
    )
    if output_path.exists() and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite existing output file: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = list(iter_jsonl(review_ledger_path))
    selected_rows = select_rows(
        rows,
        include_unapproved=args.include_unapproved,
        limit=args.limit,
    )

    client = build_http_client()
    records = transform_rows(
        client=client,
        rows=selected_rows,
        max_concurrency=max(1, args.max_concurrency),
    )
    write_jsonl(output_path, records)

    summary = {
        "review_ledger": str(review_ledger_path),
        "output": str(output_path),
        "input_rows": len(rows),
        "selected_rows": len(selected_rows),
        "successes": sum(1 for row in records if row["transform_status"] == "ok"),
        "errors": sum(1 for row in records if row["transform_status"] == "error"),
        "methods": count_values(records, "transform_method"),
    }
    print(json.dumps(summary, indent=2))
    return 0


def select_rows(
    rows: list[dict[str, Any]],
    *,
    include_unapproved: bool,
    limit: int | None,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in rows:
        fetch_status = str(row.get("fetch_status") or "")
        if fetch_status not in {"ok", "cached"}:
            continue
        if not include_unapproved and row.get("allowed_for_training") is not True:
            continue
        selected.append(row)
        if limit is not None and len(selected) >= limit:
            break
    return selected


def build_http_client():
    try:
        import requests

        return ("requests", requests)
    except ImportError:
        import urllib.request

        return ("urllib", urllib.request)


def transform_rows(
    *,
    client,
    rows: list[dict[str, Any]],
    max_concurrency: int,
) -> list[dict[str, Any]]:
    if not rows:
        return []
    if len(rows) == 1 or max_concurrency == 1:
        return [transform_one(client, row) for row in rows]

    records_by_index: dict[int, dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(max_concurrency, len(rows))
    ) as executor:
        future_to_index = {
            executor.submit(transform_one, client, row): index
            for index, row in enumerate(rows)
        }
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            records_by_index[index] = future.result()
    return [records_by_index[index] for index in sorted(records_by_index)]


def transform_one(client, row: dict[str, Any]) -> dict[str, Any]:
    final_url = str(row.get("final_url") or row.get("source_url") or "")
    source_url = str(row.get("source_url") or final_url)
    target_url = final_url or source_url
    transform_method, transform_url = choose_transform_target(row)
    record = {
        "review_id": row.get("review_id"),
        "source_url": row.get("source_url"),
        "final_url": row.get("final_url"),
        "permission_basis": row.get("permission_basis", ""),
        "transform_method": transform_method,
        "transform_url": transform_url,
        "transform_status": "error",
        "transform_error": None,
        "content_sha256": None,
        "content": None,
    }
    try:
        if transform_method == "local_cached_body":
            content = read_local_text(Path(transform_url))
        else:
            content = fetch_text(client, transform_url)
        record["transform_status"] = "ok"
        record["content"] = normalize_text(content)
        record["content_sha256"] = hashlib.sha256(record["content"].encode("utf-8")).hexdigest()
    except Exception as exc:
        record["transform_error"] = f"{type(exc).__name__}: {exc}"
    return record


def choose_transform_target(row: dict[str, Any]) -> tuple[str, str]:
    url = str(row.get("final_url") or row.get("source_url") or "")
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    body_path = row.get("body_path")
    if host.endswith("ctftime.org") and isinstance(body_path, str) and body_path.strip():
        return ("local_cached_body", str(Path(body_path).resolve()))
    if host == "raw.githubusercontent.com":
        return ("github_raw", url)
    if host in {"github.com", "www.github.com"}:
        raw_url = to_github_raw_url(url)
        if raw_url:
            return ("github_raw", raw_url)
    stripped = url.removeprefix("https://").removeprefix("http://")
    return ("jina", f"https://r.jina.ai/http://{stripped}")


def to_github_raw_url(url: str) -> str | None:
    parsed = urlparse(url)
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 5:
        return None
    owner, repo, mode = parts[0], parts[1], parts[2]
    ref = parts[3]
    rest = parts[4:]
    if mode == "blob" and rest:
        return f"https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{'/'.join(rest)}"
    if mode == "tree":
        joined = "/".join(rest)
        return f"https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{joined}/README.md"
    return None


def fetch_text(client, url: str, *, attempts: int = 6) -> str:
    mode, module = client
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            if mode == "requests":
                response = module.get(url, timeout=45, headers={"User-Agent": DEFAULT_USER_AGENT})
                if response.status_code in RETRYABLE_STATUS_CODES and attempt < attempts:
                    time.sleep(attempt * 2)
                    continue
                response.raise_for_status()
                return response.text

            request = module.Request(url, headers={"User-Agent": DEFAULT_USER_AGENT})
            with module.urlopen(request, timeout=45) as response:
                return response.read().decode("utf-8", errors="replace")
        except Exception as exc:
            last_error = exc
            status_code = getattr(exc, "code", None)
            if status_code in RETRYABLE_STATUS_CODES and attempt < attempts:
                time.sleep(attempt * 2)
                continue
            raise
    assert last_error is not None
    raise last_error


def normalize_text(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n").strip()


def read_local_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                yield json.loads(stripped)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def count_values(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key) or "")
        counts[value] = counts.get(value, 0) + 1
    return counts


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130)
