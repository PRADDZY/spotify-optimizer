# GitHub Repo Settings Checklist

Use this checklist to enforce branch governance in GitHub settings.

## Branch Protection (`main`)

- Require a pull request before merging.
- Require status checks to pass before merging.
- Require branches to be up to date before merging (optional, recommended).
- Restrict force pushes.
- Restrict deletion.

## Merge Methods

- Enable merge commits.
- Disable squash merge (if strict tree style is required).
- Disable rebase merge (if strict tree style is required).
- Set default merge message policy to preserve branch context.

## Status Checks

Required checks should include at minimum:

- `backend-tests`
- `frontend-build`
- `branch-name-policy`
- `main-merge-commit-policy`

## Pull Request Rules

- Require at least one reviewer (recommended).
- Resolve all conversations before merge (recommended).
- Require linear history: disabled (because `--no-ff` merges are used).

## Repository Ruleset Notes

If using rulesets instead of legacy branch protection:

- Apply ruleset to `refs/heads/main`.
- Block direct pushes except for admins only if absolutely needed.
- Keep an explicit exception policy documented in this file.

