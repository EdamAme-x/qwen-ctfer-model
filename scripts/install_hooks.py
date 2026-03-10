#!/usr/bin/env python3
from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path


def main() -> int:
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        print(
            "Usage:\n"
            "  python scripts/install_hooks.py\n\n"
            "Configures git to use the repository-local hooks under .githooks/."
        )
        return 0

    repo_root = Path(__file__).resolve().parent.parent
    hook_names = ("pre-commit", "pre-push")

    for hook_name in hook_names:
        hook_path = repo_root / ".githooks" / hook_name
        if not hook_path.exists():
            print(f"Missing hook file: {hook_path}", file=sys.stderr)
            return 1

    current_hooks_path = subprocess.run(
        ["git", "config", "--get", "core.hooksPath"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    if current_hooks_path.returncode != 0 or current_hooks_path.stdout.strip() != ".githooks":
        subprocess.run(
            ["git", "config", "core.hooksPath", ".githooks"],
            cwd=repo_root,
            check=True,
        )

    if os.name != "nt":
        for hook_name in hook_names:
            hook_path = repo_root / ".githooks" / hook_name
            current_mode = hook_path.stat().st_mode
            hook_path.chmod(current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    print("Configured core.hooksPath to .githooks")
    print("Hooks ready:")
    for hook_name in hook_names:
        print(f"  - {hook_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
