#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ ! -d ".githooks" ]]; then
  echo "Missing .githooks directory. Run from repository root." >&2
  exit 1
fi

git config core.hooksPath .githooks
chmod +x .githooks/commit-msg .githooks/pre-push || true

echo "Configured git hooks path: .githooks"
echo "Main push policy is enabled. To bypass once: ALLOW_MAIN_PUSH=1 git push origin main"

