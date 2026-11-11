# Getting RetroVisionAI.exe

The `.exe` has to be produced on Windows — PyInstaller cannot cross-compile, and
the finished bundle is 1.5–4 GB (PyTorch runtime included), which is larger than
what can be handed to you from this sandbox.

Below are three ways to get it, easiest first.

---

## Option 1 — Let GitHub build it (no Windows machine needed)

The repo already contains `.github/workflows/build-windows.yml`, which builds on
a real `windows-latest` runner and hands you the finished bundle.

1. Push this folder to a GitHub repository (public or private).
2. Open the **Actions** tab → **Build Windows EXE** → **Run workflow**.
3. Wait ~15–25 minutes.
4. On the finished run, scroll to **Artifacts** →
   download **`RetroVisionAI-windows`** (a zip of `dist\RetroVisionAI\`).
5. Unzip anywhere and double-click `RetroVisionAI.exe`.

Tagging a release (`git tag v1.0.0 && git push --tags`) additionally attaches
`RetroVisionAI-v1.0.0-windows-x64.zip` to the GitHub release automatically.

The workflow already:

* installs Python 3.11 + CUDA 12.1 PyTorch
* runs the test-suite before packaging
* runs `RetroVisionAI.exe --version` and `--check` to prove the frozen build works
* creates `models/`, `output/`, `temp/`, `logs/`, `bin/` next to the EXE
* copies `config.json`, `README.md`, `DELIVERY.md` and `docs/` into the bundle

---

## Option 2 — Build it on your own Windows PC

Requirements: Windows 10/11 x64, Python 3.10 or 3.11 (64-bit), ~10 GB free disk.

**PowerShell (recommended — has pre-flight checks):**

```powershell
powershell -ExecutionPolicy Bypass -File make_exe.ps1
```

**Command Prompt:**

```bat
build_windows.bat
```

Either way you get `dist\RetroVisionAI\RetroVisionAI.exe`. Zip that folder to
distribute it; no installer technology is required.

What the script does, in order:

1. Creates `.venv` and upgrades pip
2. Installs `torch==2.4.1 torchvision==0.19.1` from the **CUDA 12.1** index
3. Installs `requirements.txt` plus `pyinstaller>=6.14,<7`
4. Regenerates icons
5. Runs the pytest suite (warns, and lets you abort, if anything fails)
6. Freezes with `RetroVisionAI.spec`
7. Creates the runtime folder skeleton next to the EXE

> **Do not downgrade PyInstaller below 6.14.** Versions ≤ 6.10 ship a numpy hook
> that crashes the frozen app with *"you should not try to import numpy from its
> source directory"*. This is reproducible with a three-line test app, so it is a
> PyInstaller bug rather than an app bug — but it will bite you.

---

## Option 3 — Skip the EXE entirely

If you just want to *use* the app and don't care about a single-file executable:

```bat
Run-RetroVisionAI.bat
```

On first run it builds a local virtual environment, installs the **CPU** build of
PyTorch plus the app dependencies (~2.5 GB, one time), then starts RetroVision
AI. Every later run starts instantly. Add `--check` to verify the environment:

```bat
Run-RetroVisionAI.bat --check
```

For GPU acceleration, install CUDA PyTorch once and rerun:

```bat
.venv\Scripts\pip install torch==2.4.1 torchvision==0.19.1 ^
    --index-url https://download.pytorch.org/whl/cu121
```

---

## What is *not* in the bundle (by design)

| | |
|---|---|
| Model weights (~455 MB for all 10) | Downloaded from official upstream releases via **AI Model Manager**, with SHA-256 verification |
| FFmpeg | Uses `bin\ffmpeg.exe` if present, otherwise `PATH`. Drop the two binaries into `bin\` before building and the spec collects them automatically |

Shipping weights inside the installer would mean redistributing third-party
binaries (and their licences) on the user's behalf. Keeping them as an explicit,
verified download is both cleaner legally and keeps the download small.

---

## Verifying your build

```bat
dist\RetroVisionAI\RetroVisionAI.exe --version
dist\RetroVisionAI\RetroVisionAI.exe --check
```

`--check` prints Python/PySide6/PyTorch/CUDA/OpenCV/FFmpeg versions, the encoder
list, detected GPU, CPU/RAM, and the install state of every model. It must end
with `RESULT : ready`.

This exact flow was validated here on a frozen build (Linux target, same spec):
`RESULT: ready`, a real 50-frame 720p super-resolution run through the packaged
binary, and a clean GUI launch. See `docs/PACKAGING.md` for the details.
