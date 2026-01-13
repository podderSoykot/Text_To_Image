# PowerShell script to configure pip to use F drive
# Run this script before installing packages

# Create directories on F drive if they don't exist
if (-not (Test-Path "F:\pip_cache")) {
    New-Item -ItemType Directory -Path "F:\pip_cache" -Force | Out-Null
    Write-Host "Created F:\pip_cache directory"
}

if (-not (Test-Path "F:\tmp")) {
    New-Item -ItemType Directory -Path "F:\tmp" -Force | Out-Null
    Write-Host "Created F:\tmp directory"
}

# Set environment variables for current session
$env:PIP_CACHE_DIR = "F:\pip_cache"
$env:TMP = "F:\tmp"
$env:TEMP = "F:\tmp"

Write-Host "Environment variables set:"
Write-Host "  PIP_CACHE_DIR = $env:PIP_CACHE_DIR"
Write-Host "  TMP = $env:TMP"
Write-Host "  TEMP = $env:TEMP"
Write-Host ""
Write-Host "Now run: pip install -r requirements.txt"



