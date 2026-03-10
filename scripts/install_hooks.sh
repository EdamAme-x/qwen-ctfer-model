#!/bin/sh
set -eu

repo_root="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
exec bun --cwd "$repo_root" run install-hooks "$@"
