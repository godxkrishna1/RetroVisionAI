@echo off
REM ===========================================================================
REM  RetroVision AI - portable launcher
REM
REM  For people who do not want to run build_windows.bat themselves.
REM  On first run it creates a local venv, installs the CPU build of PyTorch
REM  and the app dependencies (~2.5 GB, one time), then starts the app.
REM  Later runs just start it.
REM
REM  For GPU acceleration install CUDA torch manually first:
REM    .venv\Scripts\pip install torch==2.4.1 torchvision==0.19.1 ^
REM        --index-url https://download.pytorch.org/whl/cu121
REM ===========================================================================
setlocal EnableDelayedExpansion
cd /d "%~dp0"
title RetroVision AI

if not exist ".venv\Scripts\python.exe" goto :bootstrap

REM ---------------- fast path: venv already exists -----------------------
if not exist ".venv\.rv_ready" goto :bootstrap
".venv\Scripts\python.exe" -c "import torch, PySide6, cv2" >nul 2>nul
if errorlevel 1 (
  echo [setup] Something is missing from the environment - repairing...
  goto :bootstrap
)
echo Starting RetroVision AI...
".venv\Scripts\python.exe" main.py %*
goto :end

:bootstrap
echo ============================================================
echo   RetroVision AI - first-run setup
echo   This installs Python packages once. Expect a few minutes.
echo   Download size is roughly 2.5 GB.
echo ============================================================
echo.

where python >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Python was not found on PATH.
  echo.
  echo  1. Install Python 3.10 or 3.11 (64-bit) from https://python.org
  echo  2. Tick "Add python.exe to PATH" during installation
  echo  3. Run this file again
  echo.
  pause
  goto :fail
)

for /f "tokens=2" %%v in ('python --version 2^>^&1') do set "PYVER=%%v"
echo [1/4] Python %PYVER% found. Creating local environment...
if not exist ".venv\Scripts\python.exe" (
  python -m venv .venv || goto :fail
)

echo [2/4] Upgrading pip...
".venv\Scripts\python.exe" -m pip install --upgrade pip wheel setuptools >nul 2>nul

echo [3/4] Installing PyTorch (CPU build - works everywhere, slower)
".venv\Scripts\python.exe" -m pip install torch==2.4.1 torchvision==0.19.1 ^
  --index-url https://download.pytorch.org/whl/cpu || goto :pipfail

echo [4/4] Installing application dependencies...
".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :pipfail

> ".venv\.rv_ready" echo ok
echo.
echo Setup complete. Starting RetroVision AI...
".venv\Scripts\python.exe" main.py %*
goto :end

:pipfail
echo.
echo [ERROR] Package installation failed.
echo   * Check your internet connection (a proxy may be blocking PyPI).
echo   * If you are on an NVIDIA GPU and want CUDA, install torch first:
echo       .venv\Scripts\pip install torch==2.4.1 torchvision==0.19.1 ^
echo         --index-url https://download.pytorch.org/whl/cu121
echo     then run this file again.
echo.
pause
goto :fail

:fail
echo RetroVision AI could not start.
pause
exit /b 1

:end
endlocal
