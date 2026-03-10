#!/usr/bin/env python3
"""Turn a scrape batch fetch manifest into a conservative review ledger."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any


SUCCESS_FETCH_STATUSES = {"ok", "cached"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read a scrape batch fetch_manifest.json and emit a review ledger with "
            "per-source approval fields for later derivative creation."
        )
    )
    parser.add_argument(
        "--fetch-manifest",
        required=True,
        help="Path to data/raw/scraped/<batch>/fetch_manifest.json",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Path for the review ledger JSONL. Defaults to "
            "<batch>/review_ledger.jsonl beside the fetch manifest."
        ),
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help=(
            "Optional path for a CSV export of the review ledger. Defaults to "
            "<batch>/review_ledger.csv beside the fetch manifest."
        ),
    )
    parser.add_argument(
        "--summary-output",
        default=None,
        help=(
            "Optional path for a compact summary JSON. Defaults to "
            "<batch>/review_summary.json beside the fetch manifest."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing output files.",
    )
    parser.add_argument(
        "--approve-training",
        action="store_true",
        help="Initialize allowed_for_training=true for successful fetches.",
    )
    parser.add_argument(
        "--approve-redistribution",
        action="store_true",
        help="Initialize allowed_for_redistribution=true for successful fetches.",
    )
    parser.add_argument(
        "--review-status",
        default=None,
        help="Optional review_status to assign to successful fetches.",
    )
    parser.add_argument(
        "--redistribution-status",
        default=None,
        help="Optional redistribution_status to assign to successful fetches.",
    )
    parser.add_argument(
        "--permission-basis",
        default="",
        help="Optional note describing the permission basis for this batch.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def default_output_path(fetch_manifest_path: Path, filename: str) -> Path:
    return fetch_manifest_path.parent / filename


def slugify(value: str) -> str:
    compact = re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip())
    compact = compact.strip("._-")
    return compact or "record"


def build_review_record(
    batch_name: str,
    index: int,
    record: dict[str, Any],
    *,
    approve_training: bool,
    approve_redistribution: bool,
    review_status_override: str | None,
    redistribution_status_override: str | None,
    permission_basis: str,
) -> dict[str, Any]:
    final_url = str(record.get("final_url") or record.get("url") or "")
    status_code = record.get("status_code")
    content_type = record.get("content_type")
    fetched_at = record.get("fetched_at")
    sha256 = record.get("sha256")
    body_path = record.get("body_path")
    fetch_status = str(record.get("status") or "error")
    fetch_error = record.get("error")
    fetch_succeeded = fetch_status in SUCCESS_FETCH_STATUSES
    review_status = (
        review_status_override.strip()
        if review_status_override and fetch_succeeded
        else ("pending_review" if fetch_succeeded else "fetch_error")
    )
    redistribution_status = (
        redistribution_status_override.strip()
        if redistribution_status_override and fetch_succeeded
        else "review_required"
    )

    return {
        "review_id": f"{slugify(batch_name)}_{index:04d}",
        "batch_name": batch_name,
        "source_url": record.get("url"),
        "final_url": final_url or None,
        "fetched_at": fetched_at,
        "http_status": status_code,
        "fetch_status": fetch_status,
        "fetch_error": fetch_error,
        "content_type": content_type,
        "sha256": sha256,
        "body_path": body_path,
        "author_if_known": None,
        "license_if_known": "",
        "permission_basis": permission_basis,
        "review_status": review_status,
        "allowed_for_training": bool(approve_training and fetch_succeeded),
        "allowed_for_redistribution": bool(approve_redistribution and fetch_succeeded),
        "redistribution_status": redistribution_status,
        "transformation_status": "not_started",
        "notes": "",
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "review_id",
        "batch_name",
        "source_url",
        "final_url",
        "fetched_at",
        "http_status",
        "fetch_status",
        "fetch_error",
        "content_type",
        "sha256",
        "body_path",
        "author_if_known",
        "license_if_known",
        "permission_basis",
        "review_status",
        "allowed_for_training",
        "allowed_for_redistribution",
        "redistribution_status",
        "transformation_status",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    fetch_manifest_path = Path(args.fetch_manifest).resolve()
    if not fetch_manifest_path.exists():
        raise SystemExit(f"Fetch manifest not found: {fetch_manifest_path}")

    payload = load_json(fetch_manifest_path)
    batch_name = str(payload.get("batch_name") or fetch_manifest_path.parent.name)
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise SystemExit(f"Fetch manifest has no records: {fetch_manifest_path}")

    output_path = (
        Path(args.output).resolve()
        if args.output
        else default_output_path(fetch_manifest_path, "review_ledger.jsonl")
    )
    csv_path = (
        Path(args.output_csv).resolve()
        if args.output_csv
        else default_output_path(fetch_manifest_path, "review_ledger.csv")
    )
    summary_path = (
        Path(args.summary_output).resolve()
        if args.summary_output
        else default_output_path(fetch_manifest_path, "review_summary.json")
    )

    for target in (output_path, csv_path, summary_path):
        if target.exists() and not args.overwrite:
            raise SystemExit(f"Refusing to overwrite existing file: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)

    review_rows = [
        build_review_record(
            batch_name=batch_name,
            index=index,
            record=record,
            approve_training=args.approve_training,
            approve_redistribution=args.approve_redistribution,
            review_status_override=args.review_status,
            redistribution_status_override=args.redistribution_status,
            permission_basis=args.permission_basis.strip(),
        )
        for index, record in enumerate(records, start=1)
    ]
    write_jsonl(output_path, review_rows)
    write_csv(csv_path, review_rows)

    summary = {
        "batch_name": batch_name,
        "fetch_manifest": str(fetch_manifest_path),
        "review_ledger": str(output_path),
        "review_ledger_csv": str(csv_path),
        "records": len(review_rows),
        "pending_reviews": sum(
            1 for row in review_rows if row["review_status"] == "pending_review"
        ),
        "approved_for_training": sum(1 for row in review_rows if row["allowed_for_training"]),
        "approved_for_redistribution": sum(
            1 for row in review_rows if row["allowed_for_redistribution"]
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130)
