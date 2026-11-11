@echo off
REM ===========================================================================
REM  RetroVision AI - Windows build script
REM  Creates dist\RetroVisionAI\RetroVisionAI.exe (one-folder build)
REM
REM  Requirements:
REM    * Windows 10/11 x64
REM    * Python 3.10 or 3.11 (64-bit) - https://www.python.org/downloads/
REM    * An NVIDIA GPU + driver (optional, CUDA wheels used for GPU speed)
REM
REM  Usage:
REM    build_windows.bat            full build
REM    build_windows.bat clean      remove previous build artefacts
REM ===========================================================================
setlocal EnableDelayedExpansion
cd /d "%~dp0"

echo.
echo  ####################################################
echo  #   RetroVision AI - Windows packaging             #
echo  ####################################################
echo.

if /I "%1"=="clean" goto :clean

REM ---------------------------------------------------------------- python
where python >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Python was not found on PATH.
  echo         Install Python 3.10 or 3.11 x64 and re-open this window.
  goto :fail
)
for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo [1/7] Python %PYVER% detected

REM --------------------------------------------------------------- venv
if not exist ".venv" (
  echo [2/7] Creating virtual environment .venv
  python -m venv .venv || goto :fail
) else (
  echo [2/7] Using existing .venv
)
call .venv\Scripts\activate.bat || goto :fail

echo [3/7] Upgrading packaging tools
python -m pip install --upgrade pip wheel setuptools >nul || goto :fail

REM --------------------------------------------------------------- torch
REM  Pick the CUDA build. cu121 wheels are the safest choice for GTX 10xx
REM  through RTX 40xx with recent drivers.
echo [4/7] Installing PyTorch (CUDA 12.1 build; use cu118 for very old drivers)
python -m pip install torch==2.4.1 torchvision==0.19.1 ^
  --index-url https://download.pytorch.org/whl/cu121 || goto :fail

echo [5/7] Installing application dependencies
python -m pip install -r requirements.txt "pyinstaller>=6.14,<7" || goto :fail

REM --------------------------------------------------------------- icons
echo [6/7] Generating icon assets
python scripts\make_icons.py || goto :fail

REM --------------------------------------------------------------- tests
echo      Running test-suite before packaging (failures abort the build)
python -m pytest tests -q
if errorlevel 1 (
  echo [WARNING] Tests reported failures.
  choice /M "Continue with packaging anyway"
  if errorlevel 2 goto :fail
)

REM --------------------------------------------------------------- package
echo [7/7] Packaging with PyInstaller (this takes a few minutes)
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
pyinstaller RetroVisionAI.spec --noconfirm --clean || goto :fail

if not exist "dist\RetroVisionAI\RetroVisionAI.exe" goto :fail

REM Ship editable config alongside the EXE and create the folder skeleton
if exist "dist\RetroVisionAI\config.json" del "dist\RetroVisionAI\config.json"
copy /y config.json "dist\RetroVisionAI\config.json" >nul
if not exist "dist\RetroVisionAI\models"  mkdir "dist\RetroVisionAI\models"
if not exist "dist\RetroVisionAI\output"  mkdir "dist\RetroVisionAI\output"
if not exist "dist\RetroVisionAI\temp"    mkdir "dist\RetroVisionAI\temp"
if not exist "dist\RetroVisionAI\logs"    mkdir "dist\RetroVisionAI\logs"

echo.
echo  ====================================================
echo   BUILD COMPLETE
echo   Executable : dist\RetroVisionAI\RetroVisionAI.exe
echo   Models     : downloaded on demand (AI Model Manager)
echo   FFmpeg     : put ffmpeg.exe + ffprobe.exe in
echo                dist\RetroVisionAI\bin\  or install on PATH
echo  ====================================================
echo.
start "" "dist\RetroVisionAI"
goto :end

:clean
echo Removing build artefacts...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist "*.spec.bak" del "*.spec.bak"
echo Done.
goto :end

:fail
echo.
echo [BUILD FAILED] See the messages above.
exit /b 1

:end
endlocal
