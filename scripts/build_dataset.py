#!/usr/bin/env python3
"""Build a processed CTF SFT dataset from manifest-described raw records."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Pattern, Tuple


DEFAULT_SYSTEM_PROMPT = (
    "You are a CTF assistant working in an authorized sandbox. "
    "Be concise, evidence-first, and technical."
)

ROLE_ALIASES = {
    "human": "user",
    "prompt": "user",
    "input": "user",
    "question": "user",
    "bot": "assistant",
    "output": "assistant",
    "answer": "assistant",
    "response": "assistant",
}

REQUIRED_SAMPLE_KEYS = (
    "id",
    "category",
    "difficulty",
    "source",
    "license",
    "challenge_family",
    "messages",
)

DEFAULT_REDACTION_DROP_KEYS = (
    "contest",
    "contest_name",
    "event_name",
    "challenge",
    "challenge_name",
    "problem",
    "problem_name",
    "problem_title",
    "title",
)

REGEX_FLAG_MAP = {
    "IGNORECASE": re.IGNORECASE,
    "MULTILINE": re.MULTILINE,
    "DOTALL": re.DOTALL,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize raw CTF records described by JSON manifests into "
            "training-ready JSONL files grouped by split."
        )
    )
    parser.add_argument(
        "--manifest-dir",
        default="data/raw/manifests",
        help="Directory that contains manifest JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Directory to receive processed JSONL files and the summary report.",
    )
    parser.add_argument(
        "--summary-path",
        default=None,
        help="Optional explicit path for the summary JSON report.",
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root used to resolve relative manifest paths.",
    )
    parser.add_argument(
        "--default-system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt injected when a sample has no system message.",
    )
    parser.add_argument(
        "--allow-missing-paths",
        action="store_true",
        help="Skip missing source files instead of failing.",
    )
    parser.add_argument(
        "--redaction-rules",
        default=None,
        help=(
            "Optional JSON file with regex replacements and metadata keys to drop "
            "for stronger anonymization."
        ),
    )
    parser.add_argument(
        "--strict-anonymize",
        action="store_true",
        help=(
            "Drop provenance and title-like metadata keys from emitted samples. "
            "Use with --redaction-rules for regex masking of contest/problem names."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    manifest_dir = resolve_path(repo_root, Path(args.manifest_dir))
    output_dir = resolve_path(repo_root, Path(args.output_dir))
    summary_path = (
        resolve_path(repo_root, Path(args.summary_path))
        if args.summary_path
        else output_dir / "summary.json"
    )
    redaction_policy = (
        load_redaction_policy(resolve_path(repo_root, Path(args.redaction_rules)))
        if args.redaction_rules
        else None
    )

    manifests = sorted(manifest_dir.glob("*.json"))
    if not manifests:
        raise SystemExit(f"No manifest JSON files found under {manifest_dir}")

    seen_ids: set[str] = set()
    grouped_samples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    manifest_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()

    for manifest_path in manifests:
        if redaction_policy and manifest_path.resolve() == Path(redaction_policy["path"]).resolve():
            continue
        manifest = load_json(manifest_path)
        if not isinstance(manifest, dict):
            raise ValueError(f"Manifest must be a JSON object: {manifest_path}")
        if manifest.get("enabled", True) is False:
            continue

        for sample in iter_manifest_samples(
            repo_root=repo_root,
            manifest_path=manifest_path,
            manifest=manifest,
            default_system_prompt=args.default_system_prompt,
            allow_missing_paths=args.allow_missing_paths,
            redaction_policy=redaction_policy,
            strict_anonymize=args.strict_anonymize,
        ):
            sample_id = sample["id"]
            if sample_id in seen_ids:
                raise ValueError(f"Duplicate sample id detected: {sample_id}")
            seen_ids.add(sample_id)

            split = sample.pop("split", None) or "dataset"
            grouped_samples[split].append(sample)
            manifest_counts[manifest_path.name] += 1
            category_counts[sample["category"]] += 1
            source_counts[sample["source"]] += 1
            split_counts[split] += 1

    output_dir.mkdir(parents=True, exist_ok=True)
    for split, samples in grouped_samples.items():
        write_jsonl(output_dir / f"{split}.jsonl", samples)

    summary = {
        "manifest_dir": str(manifest_dir),
        "output_dir": str(output_dir),
        "total_samples": sum(split_counts.values()),
        "splits": dict(sorted(split_counts.items())),
        "categories": dict(sorted(category_counts.items())),
        "sources": dict(sorted(source_counts.items())),
        "manifests": dict(sorted(manifest_counts.items())),
        "files_written": sorted(f"{split}.jsonl" for split in grouped_samples),
        "redaction_rules": str(resolve_path(repo_root, Path(args.redaction_rules)))
        if args.redaction_rules
        else None,
        "strict_anonymize": args.strict_anonymize,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))
    return 0


def iter_manifest_samples(
    *,
    repo_root: Path,
    manifest_path: Path,
    manifest: dict[str, Any],
    default_system_prompt: str,
    allow_missing_paths: bool,
    redaction_policy: dict[str, Any] | None,
    strict_anonymize: bool,
) -> Iterator[dict[str, Any]]:
    source_id = manifest.get("source_id") or manifest_path.stem
    manifest_defaults = normalize_defaults(
        {
            "source": manifest.get("source", "unknown"),
            "license": manifest.get("license", "unknown"),
            **(manifest.get("defaults") or {}),
        }
    )

    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError(f"Manifest has no records: {manifest_path}")

    for record_index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Record must be an object: {manifest_path}#{record_index}")
        record_defaults = normalize_defaults(
            merge_dicts(manifest_defaults, record.get("defaults") or {})
        )
        record_format = record.get("format", "jsonl_messages")
        record_paths = extract_record_paths(record)

        for source_path in record_paths:
            resolved_path = resolve_record_path(
                repo_root=repo_root,
                manifest_path=manifest_path,
                source_path=source_path,
            )
            if not resolved_path.exists():
                if allow_missing_paths:
                    continue
                raise FileNotFoundError(f"Source path does not exist: {resolved_path}")

            loaded_samples = load_record_samples(
                resolved_path=resolved_path,
                record_format=record_format,
                defaults=record_defaults,
                source_id=source_id,
                record_index=record_index,
            )
            for sample_index, sample in enumerate(loaded_samples):
                normalized = normalize_sample(
                    sample=sample,
                    defaults=record_defaults,
                    source_id=source_id,
                    source_path=resolved_path,
                    sample_index=sample_index,
                    default_system_prompt=default_system_prompt,
                    redaction_policy=redaction_policy,
                    strict_anonymize=strict_anonymize,
                )
                yield normalized


def load_record_samples(
    *,
    resolved_path: Path,
    record_format: str,
    defaults: dict[str, Any],
    source_id: str,
    record_index: int,
) -> Iterator[dict[str, Any]]:
    loaders = {
        "jsonl_qa": load_jsonl_qa,
        "json_qa": load_json_qa,
        "jsonl_messages": load_jsonl_messages,
        "json_messages": load_json_messages,
    }
    try:
        loader = loaders[record_format]
    except KeyError as exc:
        raise ValueError(f"Unsupported record format: {record_format}") from exc
    yield from loader(
        resolved_path=resolved_path,
        defaults=defaults,
        source_id=source_id,
        record_index=record_index,
    )


def load_jsonl_qa(
    *,
    resolved_path: Path,
    defaults: dict[str, Any],
    source_id: str,
    record_index: int,
) -> Iterator[dict[str, Any]]:
    with resolved_path.open("r", encoding="utf-8") as handle:
        for line_index, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected object in {resolved_path}:{line_index}")
            prompt = first_value(payload, "prompt", "question", "user")
            answer = first_value(payload, "answer", "assistant", "completion", "response")
            if prompt is None or answer is None:
                raise ValueError(
                    f"Missing prompt/answer fields in {resolved_path}:{line_index}"
                )
            sample = {
                **payload,
                "id": payload.get("id")
                or f"{source_id}_{resolved_path.stem}_{record_index}_{line_index}",
                "messages": build_messages_from_qa(
                    prompt=prompt,
                    answer=answer,
                    system_prompt=payload.get("system_prompt")
                    or defaults.get("system_prompt"),
                ),
            }
            yield sample


def load_json_qa(
    *,
    resolved_path: Path,
    defaults: dict[str, Any],
    source_id: str,
    record_index: int,
) -> Iterator[dict[str, Any]]:
    payload = load_json(resolved_path)
    items = payload if isinstance(payload, list) else [payload]
    for item_index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"Expected object in {resolved_path} at index {item_index}")
        prompt = first_value(item, "prompt", "question", "user")
        answer = first_value(item, "answer", "assistant", "completion", "response")
        if prompt is None or answer is None:
            raise ValueError(f"Missing prompt/answer fields in {resolved_path}")
        sample = {
            **item,
            "id": item.get("id")
            or f"{source_id}_{resolved_path.stem}_{record_index}_{item_index}",
            "messages": build_messages_from_qa(
                prompt=prompt,
                answer=answer,
                system_prompt=item.get("system_prompt") or defaults.get("system_prompt"),
            ),
        }
        yield sample


def load_jsonl_messages(
    *,
    resolved_path: Path,
    defaults: dict[str, Any],
    source_id: str,
    record_index: int,
) -> Iterator[dict[str, Any]]:
    with resolved_path.open("r", encoding="utf-8") as handle:
        for line_index, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            sample = normalize_message_record(
                payload=payload,
                source_id=source_id,
                stem=resolved_path.stem,
                record_index=record_index,
                item_index=line_index,
            )
            yield sample


def load_json_messages(
    *,
    resolved_path: Path,
    defaults: dict[str, Any],
    source_id: str,
    record_index: int,
) -> Iterator[dict[str, Any]]:
    payload = load_json(resolved_path)
    items = payload if isinstance(payload, list) else [payload]
    for item_index, item in enumerate(items):
        sample = normalize_message_record(
            payload=item,
            source_id=source_id,
            stem=resolved_path.stem,
            record_index=record_index,
            item_index=item_index,
        )
        yield sample


def normalize_message_record(
    *,
    payload: Any,
    source_id: str,
    stem: str,
    record_index: int,
    item_index: int,
) -> dict[str, Any]:
    if isinstance(payload, dict):
        if "messages" in payload:
            messages = payload["messages"]
            sample = dict(payload)
        else:
            raise ValueError("Message record object must include a 'messages' field")
    elif isinstance(payload, list):
        messages = payload
        sample = {}
    else:
        raise ValueError("Message record must be a list or object")
    sample.setdefault("id", f"{source_id}_{stem}_{record_index}_{item_index}")
    sample["messages"] = messages
    return sample


def normalize_sample(
    *,
    sample: dict[str, Any],
    defaults: dict[str, Any],
    source_id: str,
    source_path: Path,
    sample_index: int,
    default_system_prompt: str,
    redaction_policy: dict[str, Any] | None,
    strict_anonymize: bool,
) -> dict[str, Any]:
    merged = merge_dicts(defaults, sample)
    original_identifier = str(
        merged.get("id") or f"{source_id}_{source_path.stem}_{sample_index}"
    )
    if strict_anonymize:
        merged = drop_metadata_keys(merged, set(DEFAULT_REDACTION_DROP_KEYS))
    if redaction_policy:
        merged = apply_redaction_policy(merged, redaction_policy)
    if strict_anonymize:
        merged["id"] = build_anonymized_id(
            original_identifier=original_identifier,
            category=str(merged.get("category", "sample")),
        )
    else:
        merged["id"] = slugify(merged.get("id") or original_identifier)
        if redaction_policy:
            merged["id"] = f"{merged['id']}_{short_hash(original_identifier)}"
    merged["messages"] = normalize_messages(
        merged.get("messages"),
        default_system_prompt=default_system_prompt,
        sample_system_prompt=merged.get("system_prompt"),
    )
    merged.setdefault("category", "unknown")
    merged.setdefault("difficulty", "unknown")
    merged.setdefault("source", defaults.get("source", "unknown"))
    merged.setdefault("license", defaults.get("license", "unknown"))
    merged.setdefault("challenge_family", "unknown")
    merged.setdefault("artifacts", {})
    merged["provenance"] = {
        "source_id": source_id,
        "source_path": str(source_path),
    }
    merged.pop("system_prompt", None)
    if strict_anonymize:
        merged.pop("provenance", None)

    for key in REQUIRED_SAMPLE_KEYS:
        if key not in merged:
            raise ValueError(f"Sample is missing required key '{key}': {merged['id']}")
    return merged


def normalize_messages(
    payload: Any,
    *,
    default_system_prompt: str,
    sample_system_prompt: str | None,
) -> list[dict[str, str]]:
    if not isinstance(payload, list) or not payload:
        raise ValueError("messages must be a non-empty list")

    normalized: list[dict[str, str]] = []
    for entry in payload:
        if isinstance(entry, dict):
            role = ROLE_ALIASES.get(str(entry.get("role", "")).strip().lower(), None)
            if role is None:
                role = str(entry.get("role", "")).strip().lower()
            content = entry.get("content")
        elif isinstance(entry, str):
            raise ValueError("String-only message entries are not supported")
        else:
            raise ValueError("Message entry must be an object")
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"Unsupported role: {role}")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Message content must be a non-empty string")
        normalized.append({"role": role, "content": content.strip()})

    if normalized[0]["role"] != "system":
        system_prompt = sample_system_prompt or default_system_prompt
        normalized.insert(0, {"role": "system", "content": system_prompt})
    if not any(message["role"] == "assistant" for message in normalized):
        raise ValueError("At least one assistant message is required")
    return normalized


def build_messages_from_qa(
    *, prompt: str, answer: str, system_prompt: str | None
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt.strip()})
    messages.append({"role": "assistant", "content": answer.strip()})
    return messages


def extract_record_paths(record: dict[str, Any]) -> list[str]:
    if "path" in record:
        return [record["path"]]
    if "paths" in record and isinstance(record["paths"], list):
        return [str(item) for item in record["paths"]]
    raise ValueError("Record must include 'path' or 'paths'")


def resolve_record_path(*, repo_root: Path, manifest_path: Path, source_path: str) -> Path:
    candidate = Path(source_path)
    if candidate.is_absolute():
        return candidate
    manifest_relative = (manifest_path.parent / candidate).resolve()
    if manifest_relative.exists():
        return manifest_relative
    return (repo_root / candidate).resolve()


def resolve_path(repo_root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, samples: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")


def normalize_defaults(defaults: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(defaults)
    if "artifacts" in normalized and not isinstance(normalized["artifacts"], dict):
        raise ValueError("artifacts default must be an object")
    return normalized


def load_redaction_policy(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Redaction rules must be a JSON object: {path}")

    drop_keys = payload.get("drop_keys", list(DEFAULT_REDACTION_DROP_KEYS))
    if not isinstance(drop_keys, list) or not all(isinstance(item, str) for item in drop_keys):
        raise ValueError("redaction drop_keys must be a list of strings")

    forbidden = set(drop_keys) & set(REQUIRED_SAMPLE_KEYS)
    if forbidden:
        raise ValueError(
            "redaction drop_keys must not remove required sample keys: "
            + ", ".join(sorted(forbidden))
        )

    compiled_rules: list[tuple[Pattern[str], str]] = []
    for index, rule in enumerate(payload.get("rules", [])):
        if not isinstance(rule, dict):
            raise ValueError(f"redaction rule #{index} must be an object")
        pattern = rule.get("pattern")
        replacement = rule.get("replacement", "")
        if not isinstance(pattern, str):
            raise ValueError(f"redaction rule #{index} is missing a string pattern")
        flags = 0
        for flag_name in rule.get("flags", []):
            try:
                flags |= REGEX_FLAG_MAP[flag_name]
            except KeyError as exc:
                raise ValueError(f"Unsupported regex flag: {flag_name}") from exc
        compiled_rules.append((re.compile(pattern, flags), str(replacement)))

    return {
        "drop_keys": set(drop_keys),
        "rules": compiled_rules,
        "path": str(path),
    }


def drop_metadata_keys(value: Any, drop_keys: set[str]) -> Any:
    if isinstance(value, list):
        return [drop_metadata_keys(item, drop_keys) for item in value]
    if isinstance(value, dict):
        reduced: dict[str, Any] = {}
        for key, item in value.items():
            if key in drop_keys:
                continue
            reduced[key] = drop_metadata_keys(item, drop_keys)
        return reduced
    return value


def apply_redaction_policy(value: Any, policy: dict[str, Any]) -> Any:
    if isinstance(value, str):
        redacted = value
        for pattern, replacement in policy["rules"]:
            redacted = pattern.sub(replacement, redacted)
        return redacted
    if isinstance(value, list):
        return [apply_redaction_policy(item, policy) for item in value]
    if isinstance(value, dict):
        redacted_dict: dict[str, Any] = {}
        for key, item in value.items():
            if key in policy["drop_keys"]:
                continue
            redacted_dict[key] = apply_redaction_policy(item, policy)
        return redacted_dict
    return value


def build_anonymized_id(*, original_identifier: str, category: str) -> str:
    category_slug = slugify(category.lower()) or "sample"
    return f"{category_slug}_{short_hash(original_identifier)}"


def short_hash(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:10]


def merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def first_value(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return None


def slugify(value: str) -> str:
    compact = re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip())
    compact = compact.strip("._-")
    return compact or "sample"


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130)
