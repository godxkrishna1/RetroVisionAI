<#
.SYNOPSIS
    Builds RetroVisionAI.exe on Windows.

.DESCRIPTION
    One-click wrapper around build_windows.bat. Run this from PowerShell:

        powershell -ExecutionPolicy Bypass -File make_exe.ps1

    Requirements: Windows 10/11 x64, Python 3.10 or 3.11 (64-bit) on PATH.
#>
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $root

Write-Host ""
Write-Host "  RetroVision AI - Windows executable builder" -ForegroundColor Magenta
Write-Host "  -------------------------------------------" -ForegroundColor DarkGray
Write-Host ""

$py = Get-Command python -ErrorAction SilentlyContinue
if (-not $py) {
    Write-Host "  [X] Python not found on PATH." -ForegroundColor Red
    Write-Host "      Install Python 3.10/3.11 x64 from python.org and tick" -ForegroundColor Yellow
    Write-Host "      'Add python.exe to PATH', then re-run this script." -ForegroundColor Yellow
    exit 1
}

$version = (& python --version) -replace "Python ", ""
Write-Host "  Python $version" -ForegroundColor Green
if ($version -notmatch "^3\.(10|11)\.") {
    Write-Host "  [!] Tested on 3.10/3.11. $version may still work." -ForegroundColor Yellow
}

$gpu = "CPU only"
try {
    $null = nvidia-smi 2>$null
    if ($LASTEXITCODE -eq 0) { $gpu = "NVIDIA GPU detected" }
} catch { }
Write-Host "  $gpu" -ForegroundColor Green
Write-Host ""

# build_windows.bat does the venv, CUDA torch, tests and PyInstaller run
& cmd /c "build_windows.bat"
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "  [X] Build failed. Scroll up for the first error." -ForegroundColor Red
    exit $LASTEXITCODE
}

$exe = Join-Path $root "dist\RetroVisionAI\RetroVisionAI.exe"
if (Test-Path $exe) {
    $size = [math]::Round((Get-ChildItem (Split-Path $exe) -Recurse |
             Measure-Object -Property Length -Sum).Sum / 1GB, 2)
    Write-Host ""
    Write-Host "  ============================================" -ForegroundColor Green
    Write-Host "   BUILD OK" -ForegroundColor Green
    Write-Host "   $exe" -ForegroundColor White
    Write-Host "   Folder size: $size GB" -ForegroundColor DarkGray
    Write-Host "  ============================================" -ForegroundColor Green
    Write-Host ""
    $open = Read-Host "  Open the output folder now? (y/N)"
    if ($open -eq "y") { Start-Process explorer.exe (Split-Path $exe) }
} else {
    Write-Host "  [X] Expected executable was not produced." -ForegroundColor Red
    exit 1
}
