$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repoRoot

if (-not (Test-Path ".githooks")) {
  throw "Missing .githooks directory. Run from repository root."
}

git config core.hooksPath .githooks

Write-Host "Configured git hooks path: .githooks"
Write-Host "Main push policy is enabled. To bypass once: `$env:ALLOW_MAIN_PUSH='1'; git push origin main"

