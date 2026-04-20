$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$py = (Get-Command py -ErrorAction SilentlyContinue).Source
if (-not $py) { $py = (Get-Command python -ErrorAction SilentlyContinue).Source }
if (-not $py) { throw "Python launcher not found. Install Python or ensure 'py'/'python' is on PATH." }

Start-Process -FilePath $py -ArgumentList @("-m","uvicorn","backend.main:app","--reload","--port","8001")
Start-Process -FilePath $py -ArgumentList @("-m","streamlit","run","frontend/app.py","--server.port","8501")

Write-Host "Backend:  http://127.0.0.1:8001"
Write-Host "Frontend: http://127.0.0.1:8501"
