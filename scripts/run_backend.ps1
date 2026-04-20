$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$py = (Get-Command py -ErrorAction SilentlyContinue).Source
if (-not $py) { $py = (Get-Command python -ErrorAction SilentlyContinue).Source }
if (-not $py) { throw "Python launcher not found. Install Python or ensure 'py'/'python' is on PATH." }

& $py -m uvicorn backend.main:app --reload --port 8001
