#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create or attach the default worker worktrees.",
    )
    parser.add_argument(
        "--base-dir",
        help="Parent directory that will contain the worktrees. Default: ../qwen-ctfer-model-worktrees",
    )
    parser.add_argument(
        "--base-ref",
        default="main",
        help="Base ref used when creating a new feature branch. Default: main",
    )
    return parser.parse_args()


def run_capture(repo_root: Path, *args: str) -> str:
    result = subprocess.run(
        list(args),
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout


def run_passthrough(repo_root: Path, *args: str) -> None:
    subprocess.run(list(args), cwd=repo_root, check=True)


def branch_exists(repo_root: Path, branch_name: str) -> bool:
    result = subprocess.run(
        ["git", "show-ref", "--verify", "--quiet", f"refs/heads/{branch_name}"],
        cwd=repo_root,
        check=False,
    )
    return result.returncode == 0


def normalize_path(value: Path) -> str:
    resolved = str(value.resolve())
    return resolved.lower() if os_name_is_windows() else resolved


def os_name_is_windows() -> bool:
    return sys.platform.startswith("win")


def looks_like_git_worktree(target_path: Path) -> bool:
    git_path = target_path / ".git"
    return git_path.is_file() or git_path.is_dir()


def main() -> int:
    args = parse_args()
    fallback_repo_root = Path(__file__).resolve().parent.parent
    repo_root = Path(
        run_capture(fallback_repo_root, "git", "rev-parse", "--show-toplevel").strip()
    )
    default_base_dir = repo_root.parent / "qwen-ctfer-model-worktrees"
    base_dir = (repo_root / args.base_dir).resolve() if args.base_dir else default_base_dir.resolve()
    base_dir.mkdir(parents=True, exist_ok=True)

    worktrees = (
        ("feat/dataset", base_dir / "dataset"),
        ("feat/train", base_dir / "train"),
        ("feat/eval", base_dir / "eval"),
        ("feat/release", base_dir / "release"),
        ("feat/docs", base_dir / "docs"),
    )

    worktree_list = run_capture(repo_root, "git", "worktree", "list", "--porcelain")
    attached_paths = {
        normalize_path(Path(line[len("worktree ") :]))
        for line in worktree_list.splitlines()
        if line.startswith("worktree ")
    }

    for branch_name, target_path in worktrees:
        target_path = target_path.resolve()
        if normalize_path(target_path) in attached_paths:
            print(f"[skip] {branch_name} already attached at {target_path}")
            continue

        if target_path.exists() and not looks_like_git_worktree(target_path):
            print(f"[error] path exists but is not a git worktree: {target_path}", file=sys.stderr)
            return 1

        if branch_exists(repo_root, branch_name):
            print(f"[reuse] attaching existing branch {branch_name} -> {target_path}")
            run_passthrough(repo_root, "git", "worktree", "add", str(target_path), branch_name)
        else:
            print(f"[create] {branch_name} from {args.base_ref} -> {target_path}")
            run_passthrough(
                repo_root,
                "git",
                "worktree",
                "add",
                "-b",
                branch_name,
                str(target_path),
                args.base_ref,
            )

    print()
    print(f"Created or reused worktrees under: {base_dir}")
    sys.stdout.write(run_capture(repo_root, "git", "worktree", "list"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
