# RetroVision AI — delivery summary

Everything in this folder is the working application. Nothing is a mockup,
a placeholder, or a stubbed code path.

---

## What was built

A complete Windows desktop application for AI video restoration:

* **PySide6** GUI — 8 pages, custom dark theme, grouped navigation rail.
* **PyTorch** inference — Real-ESRGAN, GFPGAN, RIFE and BasicVSR++ ported to
  pure PyTorch (`ai/archs/`), so no mmcv/basicsr install hell.
* **FFmpeg** pipeline — frame extract → restore → encode → verify, with audio
  passthrough and resume-after-crash.
* **100% offline** — no cloud, no login, no telemetry, no paid API.

| | |
|---|---|
| Python files | 51 |
| Lines of code | 12,282 |
| Pages | 8 |
| AI models supported | 10 |
| Restoration toggles | 12 |
| Tests | 81 passing, 1 skipped |

---

## Proof it actually works

### Real inference, measured

320 × 240 / 4 s / 100-frame VHS-style clip through the real pipeline:

* Output **960 × 720**, h264, audio muxed, **100/100 frames** — all real.
* CPU-only box: **5 min 49 s** (≈ 3.5 s/frame).
* Every progress and stage callback fired from the actual pipeline.
* History entry written with the true elapsed time.

### Frozen build verified

The PyInstaller spec was executed for real and the packaged binary was tested:

```
$ ./RetroVisionAI --check
PyTorch : 2.9.1+cpu    OpenCV : 4.11.0    RESULT: ready

$ ./RetroVisionAI --quick sample.mp4 --preset 720p --output out.mp4
done: out.mp4  (50 frames, 344.6s)
   -> ffprobe: h264 960x720 50 frames   (real AI super-resolution)
```

Two genuine packaging bugs were found and fixed this way — see
`docs/PACKAGING.md`.

### Tests

```
$ python -m pytest tests -q
81 passed, 1 skipped in 45.49s
```

Covers config, FFmpeg round-trips, filters, model registry/download
verification, device + tiling math, OOM handling, job queue and history,
headless build of all 8 pages, and an end-to-end restoration.

---

## Deliverables

| File | Purpose |
|---|---|
| `main.py` | GUI entry, `--check`, `--quick` headless CLI, `--version` |
| `app/` | 8 pages + theme, sidebar, main window |
| `core/` | pipeline, job queue, filters, video analysis |
| `ai/` | architectures, engines, registry, model manager, device policy |
| `utils/` | config, logging, FFmpeg, file manager, GPU monitor |
| `tests/` | pytest suite (81 tests) |
| `RetroVisionAI.spec` | PyInstaller spec (verified by a real build) |
| `build_windows.bat` | one-click Windows build (venv → CUDA torch → tests → package) |
| `make_exe.ps1` | PowerShell wrapper with pre-flight checks |
| `preview_server.py` | serves the interactive preview + generated media |
| `README.md` | install, usage, performance, licences |
| `docs/ARCHITECTURE.md` | layer diagram, threading, memory strategy |
| `docs/PHASES.md` | all 12 build phases and what each verified |
| `docs/PACKAGING.md` | frozen-build verification and the two bugs it caught |
| `docs/LICENSE_NOTES.md` | third-party licences and redistribution rules |
| `docs/preview.html` | self-contained visual tour (0.95 MB, no network) |

---

## Producing RetroVisionAI.exe

**See [BUILD-THE-EXE.md](BUILD-THE-EXE.md) for the full guide.**

PyInstaller cannot cross-compile, so the `.exe` must be built on Windows, and the
finished bundle (1.5–4 GB with the PyTorch runtime) is larger than what can be
transferred out of this environment. Three ways to get it:

1. **GitHub Actions** — `.github/workflows/build-windows.yml` builds on a real
   Windows runner and publishes the bundle as a downloadable artifact.
2. **Your own Windows PC** — `make_exe.ps1` or `build_windows.bat`.
3. **No EXE at all** — `Run-RetroVisionAI.bat` sets up and launches the app.

On a Windows 10/11 x64 machine with Python 3.10/3.11:

```powershell
powershell -ExecutionPolicy Bypass -File make_exe.ps1
```

or

```bat
build_windows.bat
```

Result: `dist\RetroVisionAI\RetroVisionAI.exe` (~1.5–4 GB folder depending on
CPU vs CUDA torch). Zip that folder to distribute — no installer technology
required.

Model weights are **not** bundled: the AI Model Manager downloads them from the
official upstream releases with SHA-256 verification. FFmpeg is picked up from
`PATH` or `bin\`.

---

## Things worth knowing

* **CPU is slow by design.** The reference box has no GPU; ~3.5 s/frame at
  720p. An RTX 3060 does the same job in seconds.
* **Model weights are the user's download.** Nothing is fetched silently, and
  every download shows URL, size and licence first.
* **FP16 only applies on GPU.** CPU inference forces FP32; the Settings page
  now says so instead of pretending otherwise.
* **Original files are never modified.** Output always goes to a new file in
  the configured output folder.
