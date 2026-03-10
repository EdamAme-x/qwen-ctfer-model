#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


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
    python_files = [
        line.strip()
        for line in git_capture(repo_root, "ls-files", "*.py").splitlines()
        if line.strip()
    ]

    if python_files:
        subprocess.run(
            [choose_python(), "-m", "py_compile", *python_files],
            cwd=repo_root,
            check=True,
        )

    tracked_scraped = git_capture(
        repo_root,
        "ls-files",
        "data/raw/scraped/*",
        ":!data/raw/scraped/.gitkeep",
        check=False,
    ).strip()
    if tracked_scraped:
        print("pre-push: tracked files under data/raw/scraped are not allowed", file=sys.stderr)
        return 1

    outputs_tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "outputs/*"],
        cwd=repo_root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if outputs_tracked.returncode == 0:
        print("pre-push: tracked files under outputs are not allowed", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
