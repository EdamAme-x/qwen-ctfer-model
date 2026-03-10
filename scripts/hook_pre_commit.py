#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


MAX_FILE_BYTES = 5 * 1024 * 1024


def choose_python() -> str:
    candidate = os.environ.get("PYTHON")
    if candidate:
        return candidate
    return sys.executable or "python"


def git_capture(repo_root: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=check,
    )
    return result.stdout


def main() -> int:
    repo_root = Path(git_capture(Path.cwd(), "rev-parse", "--show-toplevel").strip())
    staged_output = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    ).stdout
    staged_files = [item.decode("utf-8") for item in staged_output.split(b"\x00") if item]

    if not staged_files:
        return 0

    reject_path = False
    large_file = False
    python_files: list[str] = []

    for relative_path in staged_files:
        path = Path(relative_path)
        path_str = path.as_posix()

        if path_str != "data/raw/scraped/.gitkeep" and (
            path_str.startswith("data/raw/scraped/") or path_str.startswith("outputs/")
        ):
            print(
                f"pre-commit: refusing to commit generated/raw artifact path: {path_str}",
                file=sys.stderr,
            )
            reject_path = True

        absolute_path = repo_root / path
        if absolute_path.is_file():
            size_bytes = absolute_path.stat().st_size
            if size_bytes > MAX_FILE_BYTES:
                print(
                    f"pre-commit: refusing to commit file larger than 5 MiB: "
                    f"{path_str} ({size_bytes} bytes)",
                    file=sys.stderr,
                )
                large_file = True

        if path.suffix == ".py":
            python_files.append(path_str)

    whitespace_check = subprocess.run(
        ["git", "diff", "--cached", "--check"],
        cwd=repo_root,
        check=False,
    )
    if whitespace_check.returncode != 0:
        print("pre-commit: whitespace or conflict-marker issues detected", file=sys.stderr)
        return 1

    if reject_path or large_file:
        return 1

    if python_files:
        subprocess.run(
            [choose_python(), "-m", "py_compile", *python_files],
            cwd=repo_root,
            check=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
