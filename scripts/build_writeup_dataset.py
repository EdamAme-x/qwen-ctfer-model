#!/usr/bin/env python3
"""Convert transformed scraped writeups into train/eval chat datasets."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


DEFAULT_SYSTEM_PROMPT = (
    "You are a CTF assistant working in an authorized sandbox. "
    "Be concise, evidence-first, and technical."
)
DEFAULT_USER_PROMPT_BY_CATEGORY = {
    "pwn": "Walk through the shortest exploitation path for this authorized pwn challenge. Keep the steps concrete and operator-focused.",
    "rev": "Explain the shortest reverse-engineering path for this authorized CTF challenge. Focus on the decisive observations and any code needed.",
    "web": "Give a compact operator-focused walkthrough for this authorized web challenge. Include the key requests, bugs, and exploit flow.",
    "crypto": "Summarize the decisive cryptographic weakness and the solve path for this authorized crypto challenge.",
    "forensics": "Provide a concise forensic solve path for this authorized CTF task. Include the key artifacts, tools, and extraction steps.",
    "osint": "Provide a concise OSINT solve path for this authorized CTF task. Include the key pivots, sources, and validation steps.",
    "misc": "Provide a concise operator-focused walkthrough for this authorized CTF challenge. Keep only the decisive steps, code, and commands.",
}
FLAG_PATTERNS = (
    re.compile(r"(?i)\b[a-z0-9_]{0,32}\{[^{}\n]{4,200}\}"),
    re.compile(r"(?i)\bflag\s*[:=]\s*[^\s`]+"),
)
CATEGORY_RULES = {
    "pwn": ("pwntools", "rop", "libc", "canary", "pie", "gdb", "heap", "format string", "shellcode", "elf"),
    "rev": ("ghidra", "ida", "decompiler", "decompile", "disassembly", "disasm", "angr", "binary ninja"),
    "web": ("http", "request", "sql", "xss", "csrf", "ssrf", "jwt", "cookie", "flask", "php", "endpoint"),
    "crypto": ("rsa", "aes", "ecb", "cbc", "xor", "cipher", "lattice", "curve", "hash", "padding oracle"),
    "forensics": ("pcap", "wireshark", "memory", "disk", "registry", "exif", "metadata", "zip2john", "volatility"),
    "osint": ("osint", "whois", "dns", "geolocation", "social media", "satellite", "reverse image", "google dork"),
}
LOW_SIGNAL_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"}
MEDIUM_PROFILE_RE = re.compile(r"^/@[^/]+/?$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transformed-file", required=True)
    parser.add_argument("--enumeration-jsonl", default=None)
    parser.add_argument("--train-output", default="data/processed/train.jsonl")
    parser.add_argument("--eval-output", default="data/processed/eval.jsonl")
    parser.add_argument("--summary-output", default="data/processed/writeup_dataset_summary.json")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-assistant-chars", type=int, default=12000)
    parser.add_argument("--min-content-chars", type=int, default=400)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    transformed_path = Path(args.transformed_file).resolve()
    if not transformed_path.exists():
        raise SystemExit(f"Transformed file not found: {transformed_path}")

    train_output = Path(args.train_output).resolve()
    eval_output = Path(args.eval_output).resolve()
    summary_output = Path(args.summary_output).resolve()
    for path in (train_output, eval_output, summary_output):
        if path.exists() and not args.overwrite:
            raise SystemExit(f"Refusing to overwrite existing output file: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

    ctftime_to_original, original_to_ctftime = load_enumeration_links(args.enumeration_jsonl)
    transformed_rows = [row for row in iter_jsonl(transformed_path) if row.get("transform_status") == "ok"]
    selected_rows = select_preferred_rows(transformed_rows, ctftime_to_original, original_to_ctftime)

    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    dropped_short = 0
    dropped_low_signal = 0

    for row in selected_rows:
        content = clean_content(row["content"], row["transform_method"])
        content = redact_flags(content)
        content = content[: args.max_assistant_chars].strip()
        if len(content) < args.min_content_chars:
            dropped_short += 1
            continue
        if not should_keep_record(row, content):
            dropped_low_signal += 1
            continue

        sample = build_sample(row, content, ctftime_to_original, original_to_ctftime)
        if sample["split"] == "eval":
            eval_rows.append(sample)
        else:
            train_rows.append(sample)

    write_jsonl(train_output, train_rows)
    write_jsonl(eval_output, eval_rows)

    summary = {
        "transformed_file": str(transformed_path),
        "train_output": str(train_output),
        "eval_output": str(eval_output),
        "input_ok_rows": len(transformed_rows),
        "selected_rows": len(selected_rows),
        "dropped_short": dropped_short,
        "dropped_low_signal": dropped_low_signal,
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "categories": count_values([*train_rows, *eval_rows], "category"),
    }
    summary_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def load_enumeration_links(
    enumeration_jsonl: str | None,
) -> tuple[dict[str, str], dict[str, str]]:
    if not enumeration_jsonl:
        return {}, {}
    path = Path(enumeration_jsonl).resolve()
    if not path.exists():
        raise SystemExit(f"Enumeration JSONL not found: {path}")

    ctftime_to_original: dict[str, str] = {}
    original_to_ctftime: dict[str, str] = {}
    for row in iter_jsonl(path):
        ctftime_url = str(row.get("ctftime_writeup_url") or "").strip()
        original_url = str(row.get("original_url") or "").strip()
        if not ctftime_url:
            continue
        if original_url:
            ctftime_to_original[ctftime_url] = original_url
            original_to_ctftime[original_url] = ctftime_url
    return ctftime_to_original, original_to_ctftime


def select_preferred_rows(
    rows: list[dict[str, Any]],
    ctftime_to_original: dict[str, str],
    original_to_ctftime: dict[str, str],
) -> list[dict[str, Any]]:
    best_by_group: dict[str, dict[str, Any]] = {}
    for row in rows:
        group_key = canonical_group_key(row, ctftime_to_original, original_to_ctftime)
        current = best_by_group.get(group_key)
        if current is None or row_priority(row) > row_priority(current):
            best_by_group[group_key] = row
    return list(best_by_group.values())


def canonical_group_key(
    row: dict[str, Any],
    ctftime_to_original: dict[str, str],
    original_to_ctftime: dict[str, str],
) -> str:
    url = str(row.get("final_url") or row.get("source_url") or "")
    if url in original_to_ctftime:
        return original_to_ctftime[url]
    if url in ctftime_to_original:
        return url
    return url


def row_priority(row: dict[str, Any]) -> tuple[int, int]:
    url = str(row.get("final_url") or row.get("source_url") or "")
    host = urlparse(url).netloc.lower()
    is_ctftime = host.endswith("ctftime.org")
    content_length = len(str(row.get("content") or ""))
    return (0 if is_ctftime else 1, content_length)


def clean_content(content: str, transform_method: str) -> str:
    cleaned = content.strip()
    if transform_method == "local_cached_body" or looks_like_html(cleaned):
        cleaned = html_to_text(cleaned)
    cleaned = strip_jina_preamble(cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def strip_jina_preamble(value: str) -> str:
    marker = "Markdown Content:"
    if marker in value:
        value = value.split(marker, 1)[1]
    return value.strip()


def looks_like_html(value: str) -> bool:
    prefix = value[:512].lower()
    return "<html" in prefix or "<!doctype html" in prefix


def html_to_text(value: str) -> str:
    # Local cached HTML is a fallback source; strip markup aggressively so the
    # later dataset does not memorize page chrome.
    value = re.sub(r"(?is)<script.*?>.*?</script>", " ", value)
    value = re.sub(r"(?is)<style.*?>.*?</style>", " ", value)
    value = re.sub(r"(?i)<br\s*/?>", "\n", value)
    value = re.sub(r"(?i)</p\s*>", "\n\n", value)
    value = re.sub(r"(?i)</div\s*>", "\n", value)
    value = re.sub(r"(?is)<[^>]+>", " ", value)
    value = html.unescape(value)
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n[ \t]+", "\n", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def redact_flags(value: str) -> str:
    redacted = value
    for pattern in FLAG_PATTERNS:
        redacted = pattern.sub("<FLAG>", redacted)
    return redacted


def build_sample(
    row: dict[str, Any],
    content: str,
    ctftime_to_original: dict[str, str],
    original_to_ctftime: dict[str, str],
) -> dict[str, Any]:
    url = str(row.get("final_url") or row.get("source_url") or "")
    group_key = canonical_group_key(row, ctftime_to_original, original_to_ctftime)
    category = infer_category(url, content)
    user_prompt = DEFAULT_USER_PROMPT_BY_CATEGORY.get(category, DEFAULT_USER_PROMPT_BY_CATEGORY["misc"])
    split = "eval" if int(short_hash(group_key), 16) % 10 == 0 else "train"
    sample_id = f"{category}_{short_hash(group_key)}"
    return {
        "id": sample_id,
        "category": category,
        "difficulty": "unknown",
        "source": "transformed_scraped",
        "license": "internal_or_permitted",
        "challenge_family": category,
        "split": split,
        "messages": [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": content},
        ],
        "artifacts": {
            "has_code": bool(re.search(r"`{3}|pwntools|def |import |SELECT |curl ", content)),
            "source_url": url,
        },
    }


def infer_category(url: str, content: str) -> str:
    haystack = f"{url}\n{content}".lower()
    best_category = "misc"
    best_score = 0
    for category, keywords in CATEGORY_RULES.items():
        score = sum(1 for keyword in keywords if keyword in haystack)
        if score > best_score:
            best_category = category
            best_score = score
    return best_category


def should_keep_record(row: dict[str, Any], content: str) -> bool:
    url = str(row.get("final_url") or row.get("source_url") or "")
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    path = parsed.path or "/"
    if host in LOW_SIGNAL_HOSTS:
        return False
    if host.endswith("medium.com") and MEDIUM_PROFILE_RE.fullmatch(path):
        return False
    if host in {"github.com", "www.github.com"} and "/blob/" not in path and "/tree/" not in path:
        return False

    lower = content[:3000].lower()
    low_signal_hits = sum(
        marker in lower
        for marker in (
            "open in app",
            "sign up",
            "sign in",
            "followers",
            "sitemap",
            "member-only story",
            "latest stories",
        )
    )
    if host.endswith("medium.com") and low_signal_hits >= 3:
        return False
    return True


def short_hash(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:10]


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
