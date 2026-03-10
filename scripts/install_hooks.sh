#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

git config core.hooksPath .githooks
chmod +x .githooks/pre-commit .githooks/pre-push

echo "Configured core.hooksPath to .githooks"
echo "Hooks ready:"
echo "  - pre-commit"
echo "  - pre-push"
