# Packaging & frozen-build verification

The Windows installer is produced by `build_windows.bat` → `RetroVisionAI.spec`
(PyInstaller, one-folder build).

Because PyInstaller cannot cross-compile, the exact `.exe` can only be produced
on a Windows host. To make sure the *spec itself* is correct, the same spec was
executed for real on the Linux CI box and the resulting frozen binary was tested
end-to-end. Everything below was measured, not assumed.

## Frozen build facts

| Item | Value |
|---|---|
| PyInstaller | 6.22.3 (**6.10.0 is broken — see below**) |
| Python | 3.13.16 |
| Frozen tree size | 1.4 GB (CPU torch build; CUDA build ≈ 2.5–4 GB) |
| Entry | `RetroVisionAI` (Windows: `RetroVisionAI.exe`) |
| Layout | one-folder (`dist/RetroVisionAI/`) |

## Verification performed on the frozen binary

```
$ ./RetroVisionAI --version
RetroVision AI 1.0.0

$ ./RetroVisionAI --check
PySide6       : 6.9.3
PyTorch       : 2.9.1+cpu   CUDA: False
OpenCV        : 4.11.0
FFmpeg        : 7.1.5
Encoders      : libx264, libx265
RESULT        : ready
```

Real processing through the frozen binary:

```
$ ./RetroVisionAI --quick samples/av_sample.mp4 --preset 720p --output out.mp4
[Analyzing video] … [Loading AI models] … [Restoring frames] …
[Encoding video] [Verifying output] [Completed]
done: out.mp4  (50 frames, 344.6s)
```
`ffprobe` on the result: `h264, 960x720, 50 frames` — real Real-ESRGAN inference
inside the packaged app, not a fallback path.

GUI check: launched the frozen app with `QT_QPA_PLATFORM=offscreen`; it stayed
alive (no import or resource errors) for the whole 45 s window.

## Two real packaging bugs found this way (both fixed in the repo)

1. **PyInstaller ≤ 6.10 + numpy ≥ 2.3 breaks the frozen app.**
   Symptom: `ImportError: Error importing numpy: you should not try to import
   numpy from its source directory`. A three-line numpy-only test app reproduces
   it, so it is not caused by this codebase. `build_windows.bat` now installs
   `pyinstaller>=6.14,<7`.

2. **Excluding `torch.testing` breaks torch at runtime.**
   `import torch` fails with `No module named 'torch.testing'` because torch
   imports it internally. The spec's `excludes` list no longer touches torch
   submodules.

Both were invisible from the source tree — `python main.py --check` passed the
whole time. Packaging verification is therefore part of the build, not an
afterthought.

## Producing the Windows executable

On a Windows 10/11 x64 machine with Python 3.10/3.11:

```bat
build_windows.bat
```

Output:

```
dist\RetroVisionAI\
├── RetroVisionAI.exe        <- the application
├── _internal\               <- torch / Qt / opencv runtime
├── models\                  <- empty; filled by AI Model Manager
├── output\  temp\  logs\
└── config.json
```

Distribution notes:

* Zip `dist\RetroVisionAI\` and ship it; no installer technology is required.
* Model weights are deliberately absent (≈ 455 MB) — the Model Manager fetches
  them from their official upstream releases with SHA-256 verification.
* FFmpeg is not bundled by default. Either install it on `PATH` or drop
  `ffmpeg.exe` / `ffprobe.exe` into `bin\` before building; the spec collects
  them automatically.
* If you bundle FFmpeg, check its licence (LGPL/GPL) before redistributing.
