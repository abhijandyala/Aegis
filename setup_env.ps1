$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$Python = if ($env:PYTHON) { $env:PYTHON } else { "python" }
& $Python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
& .\.venv\Scripts\python.exe -m pytest tests\ -q

Write-Host "Aegis is ready. Run: .\.venv\Scripts\python.exe web\server.py"
