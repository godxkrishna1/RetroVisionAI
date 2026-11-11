# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller build specification for RetroVision AI (Windows).

Build:   pyinstaller RetroVisionAI.spec --noconfirm

NOTE: use PyInstaller >= 6.14.  Versions before that ship a numpy hook that
crashes on numpy >= 2.3 ("you should not try to import numpy from its source
directory"), which breaks the frozen app even though `python main.py` works.
Result:  dist/RetroVisionAI/RetroVisionAI.exe     GUI (no console window)
         dist/RetroVisionAI/retrovision-cli.exe   console build for --check/--quick

Two executables are produced from one shared Analysis.  A windowed build has no
stdout, so --check / --quick / --version would print nothing there; the console
build exists so those flags actually produce output (and so the CI verification
step can assert on it).

A one-folder build starts noticeably faster than one-file and lets users
manage the models/ folder right next to the executable. AI model weights are
NOT bundled - they are downloaded on demand from the official repositories
(see the AI Model Manager) which keeps the installer small.

FFmpeg is *not* bundled either: we ship a `bin/` folder convention and fall
back to PATH. See build_windows.bat for how to vendor ffmpeg.exe if you want a
fully self-contained installer (mind the FFmpeg licence).
"""
import os
import sys
from pathlib import Path

ROOT = Path(SPECPATH).resolve()

# Windows uses the .ico; the same spec builds and is testable on Linux/macOS.
if sys.platform == "win32":
    ICON_NAME = "app_icon.ico"
    BIN_NAMES = ("ffmpeg.exe", "ffprobe.exe")
else:
    ICON_NAME = "app_icon.png"
    BIN_NAMES = ("ffmpeg", "ffprobe")

# ---- optional vendor folders -------------------------------------------------
datas = [
    (str(ROOT / "assets"), "assets"),
    (str(ROOT / "config.json"), "."),
]
binaries = []

for name in BIN_NAMES:
    candidate = ROOT / "bin" / name
    if candidate.exists():
        binaries.append((str(candidate), "bin"))

# ---- torch / torchvision hidden imports (kept minimal) ----------------------
hiddenimports = [
    "torch",
    "torchvision",
    "cv2",
    "PIL.Image",
    "numpy",
    "psutil",
    "requests",
    "packaging",
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
]

# Gui-only multimedia is optional: include it when PySide6 provides it so that
# the preview player works in the packaged build.
try:
    import PySide6  # noqa: F401
    for module in ("PySide6.QtMultimedia", "PySide6.QtMultimediaWidgets"):
        try:
            __import__(module)
            hiddenimports.append(module)
        except Exception:
            pass
except Exception:
    pass

# torch ships CUDA DLLs; the hook collects them automatically. We exclude the
# training-only pieces to keep the folder size sane.
# Keep this list limited to genuinely unused top-level packages.
# Do NOT exclude torch submodules (e.g. torch.testing): torch imports them
# internally and the frozen app then fails with "No module named 'torch.testing'".
excludes = [
    "torch.utils.tensorboard",
    "matplotlib",
    "tkinter",
    "IPython",
    "notebook",
    "pytest",
    "pytest_qt",
    "PySide6.QtWebEngineCore",
    "PySide6.QtWebEngineWidgets",
    "PySide6.QtQuick",
    "PySide6.QtQml",
    "PySide6.Qt3DCore",
]

block_cipher = None

a = Analysis(
    [str(ROOT / "main.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="RetroVisionAI",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,                     # UPX can break torch DLLs - keep it off
    console=False,                 # GUI app, no console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ROOT / "assets" / ICON_NAME),
    version=None,
)

# Console sibling: same frozen code, but stdout/stderr are real. Use it for
# `--check`, `--quick` and `--version`, or when troubleshooting a user's machine.
cli_exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="retrovision-cli",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,                  # <- the only difference that matters
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ROOT / "assets" / ICON_NAME),
    version=None,
)

coll = COLLECT(
    exe,
    cli_exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="RetroVisionAI",
)
