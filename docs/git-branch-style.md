# Git Branch Style

This repository uses a branch-first workflow to keep a readable, auditable git tree.

## Core Rules

- `main` is protected release history.
- No direct commits to `main`.
- Every change must go through branch + merge commit.
- Use merge commits (`--no-ff`) as default.
- Rebase/squash merge is not the default workflow.

## Branch Types

Allowed naming patterns:

- `feat/backend-<slug>`
- `feat/frontend-<slug>`
- `feat/deploy-<slug>`
- `fix/backend-<slug>`
- `fix/frontend-<slug>`
- `fix/deploy-<slug>`
- `chore/<slug>`
- `docs/<slug>`
- `perf/<slug>`
- `test/<slug>`

Rules:

- Use lowercase only.
- Use kebab-case only.
- Keep `<slug>` short and explicit (2-6 words).
- Avoid date prefixes and personal prefixes.

Examples:

- `feat/backend-search-budget`
- `perf/backend-objective-fastpath`
- `fix/deploy-systemd-paths`
- `docs/branch-governance`

## Commit Message Sanity

Local hooks enforce baseline commit hygiene:

- Subject cannot be empty.
- Subject length must stay within a practical range.
- Subject cannot end with a period.
- Merge and revert commits are allowed.

## Merge Message Format

Use this merge message pattern:

- `Merge <branch>: <human summary>`

Example:

- `Merge feat/backend-search-budget: adaptive search budget profile`

## Area Branch Workflow

- Work on area branches (`backend`, `frontend`, `deploy`) across related tasks.
- Keep commits scoped and readable inside the area branch.
- Merge area branch into `main` with `--no-ff`.
- Remote feature branches are retained after merge.

## Local Setup

Run one of:

- `powershell -ExecutionPolicy Bypass -File scripts/setup-git-hooks.ps1`
- `bash scripts/setup-git-hooks.sh`

This configures:

- `core.hooksPath=.githooks`

