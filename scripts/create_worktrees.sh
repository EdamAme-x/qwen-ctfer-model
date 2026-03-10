#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/create_worktrees.sh [--base-dir PATH] [--base-ref REF]

Options:
  --base-dir PATH  Parent directory that will contain the worktrees.
                   Default: ../qwen-ctfer-model-worktrees
  --base-ref REF   Base ref used when creating a new feature branch.
                   Default: main
EOF
}

repo_root="$(git rev-parse --show-toplevel)"
default_base_dir="$(dirname "$repo_root")/qwen-ctfer-model-worktrees"
base_dir="$default_base_dir"
base_ref="main"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-dir)
      base_dir="$2"
      shift 2
      ;;
    --base-ref)
      base_ref="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

mkdir -p "$base_dir"

declare -a branch_names=(
  "feat/dataset"
  "feat/train"
  "feat/eval"
  "feat/release"
  "feat/docs"
)

declare -A branch_paths=(
  ["feat/dataset"]="$base_dir/dataset"
  ["feat/train"]="$base_dir/train"
  ["feat/eval"]="$base_dir/eval"
  ["feat/release"]="$base_dir/release"
  ["feat/docs"]="$base_dir/docs"
)

for branch in "${branch_names[@]}"; do
  path="${branch_paths[$branch]}"

  if git worktree list --porcelain | grep -Fqx "worktree $path"; then
    echo "[skip] $branch already attached at $path"
    continue
  fi

  if [[ -e "$path" ]] && [[ ! -d "$path/.git" ]]; then
    echo "[error] path exists but is not a git worktree: $path" >&2
    exit 1
  fi

  if git show-ref --verify --quiet "refs/heads/$branch"; then
    echo "[reuse] attaching existing branch $branch -> $path"
    git worktree add "$path" "$branch"
  else
    echo "[create] $branch from $base_ref -> $path"
    git worktree add -b "$branch" "$path" "$base_ref"
  fi
done

echo
echo "Created or reused worktrees under: $base_dir"
git worktree list
