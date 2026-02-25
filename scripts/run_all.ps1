<#
PowerShell helper to start backend and frontend in separate windows.

Usage:
  .\scripts\run_all.ps1 [-AutoYes] [-OpenBrowser]

#>
param(
    [switch]$AutoYes,
    [switch]$OpenBrowser = $true
)

if ($AutoYes) { $env:AUTO_YES = '1' }

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition

Write-Host "Starting backend (uvicorn) in new PowerShell window..."
Start-Process powershell -ArgumentList "-NoExit","-Command","python -m uvicorn api.main:app --reload --port 8000" -WorkingDirectory $repoRoot

Write-Host "Starting frontend (static server) in new PowerShell window..."
Start-Process powershell -ArgumentList "-NoExit","-Command","cd web_ui; python -m http.server 5500" -WorkingDirectory $repoRoot

if ($OpenBrowser) {
    Start-Process "http://localhost:5500"
}

Write-Host "Launched backend and frontend. Check new windows for logs."
