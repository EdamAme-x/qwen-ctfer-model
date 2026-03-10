#!/usr/bin/env python3
"""Upload a local release folder to Hugging Face with repo-safe defaults."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_IGNORE_PATTERNS = (
    ".git",
    ".git/**",
    "__pycache__",
    "**/__pycache__/**",
    "*.pyc",
    "*.pyo",
    "*.log",
    ".DS_Store",
    "events.out.tfevents*",
    "optimizer.pt",
    "optimizer.bin",
    "scheduler.pt",
    "trainer_state.json",
    "rng_state*.pth",
    "checkpoint-*/**",
    "outputs/**",
    "data/raw/**",
    "data/interim/**",
    "*.tmp",
    "*.swp",
)


DEFAULT_REQUIRED_BY_KIND = {
    "adapter": ("README.md", "adapter_config.json"),
    "merged": ("README.md", "config.json"),
    "dataset": ("README.md",),
    "other": tuple(),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload a local folder to the Hugging Face Hub with validation."
    )
    parser.add_argument("--local-dir", required=True, help="Folder to upload.")
    parser.add_argument("--repo-id", required=True, help="Destination repo id, e.g. user/name.")
    parser.add_argument(
        "--repo-type",
        default="model",
        choices=("model", "dataset", "space"),
        help="Hub repository type.",
    )
    parser.add_argument(
        "--release-kind",
        default="adapter",
        choices=tuple(DEFAULT_REQUIRED_BY_KIND),
        help="Used to select required metadata files.",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
        help="Hugging Face token. Defaults to HF_TOKEN or HUGGING_FACE_HUB_TOKEN.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the remote repo as private if it does not already exist.",
    )
    parser.add_argument(
        "--commit-message",
        help="Commit message for the upload. Defaults to a deterministic message.",
    )
    parser.add_argument(
        "--ignore-pattern",
        action="append",
        default=[],
        help="Extra path/glob patterns to exclude. Can be repeated.",
    )
    parser.add_argument(
        "--allow-pattern",
        action="append",
        default=[],
        help="Optional allowlist patterns passed to upload_folder. Can be repeated.",
    )
    parser.add_argument(
        "--require-file",
        action="append",
        default=[],
        help="Extra required files relative to --local-dir. Can be repeated.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print the resolved settings without uploading.",
    )
    return parser.parse_args()


def validate_local_dir(local_dir: Path, required_files: Sequence[str]) -> None:
    if not local_dir.exists():
        raise FileNotFoundError(f"Local directory does not exist: {local_dir}")
    if not local_dir.is_dir():
        raise NotADirectoryError(f"Local path is not a directory: {local_dir}")

    missing = [rel for rel in required_files if not (local_dir / rel).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required files for upload: " + ", ".join(sorted(missing))
        )

    if not any(local_dir.iterdir()):
        raise RuntimeError(f"Local directory is empty: {local_dir}")


def build_commit_message(repo_type: str, release_kind: str) -> str:
    return f"Upload {release_kind} artifact to {repo_type} repo"


def unique(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def main() -> None:
    args = parse_args()
    local_dir = Path(args.local_dir).resolve()
    required_files = unique(
        [*DEFAULT_REQUIRED_BY_KIND[args.release_kind], *args.require_file]
    )
    ignore_patterns = unique([*DEFAULT_IGNORE_PATTERNS, *args.ignore_pattern])
    commit_message = args.commit_message or build_commit_message(
        args.repo_type, args.release_kind
    )

    if not args.token and not args.dry_run:
        raise RuntimeError(
            "No Hugging Face token provided. Set --token or HF_TOKEN."
        )

    validate_local_dir(local_dir, required_files)

    if args.dry_run:
        print(f"local_dir={local_dir}")
        print(f"repo_id={args.repo_id}")
        print(f"repo_type={args.repo_type}")
        print(f"release_kind={args.release_kind}")
        print(f"required_files={required_files}")
        print(f"ignore_patterns={ignore_patterns}")
        print(f"allow_patterns={args.allow_pattern}")
        print(f"commit_message={commit_message}")
        return

    from huggingface_hub import HfApi

    api = HfApi(token=args.token)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        private=args.private,
        exist_ok=True,
    )
    commit_info = api.upload_folder(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        folder_path=str(local_dir),
        allow_patterns=args.allow_pattern or None,
        ignore_patterns=ignore_patterns,
        commit_message=commit_message,
    )
    print(f"Uploaded {local_dir} to https://huggingface.co/{args.repo_id}")
    print(commit_info)


if __name__ == "__main__":
    main()
