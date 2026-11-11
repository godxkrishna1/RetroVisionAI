# RetroVision AI

**Restore old and low-quality video to crisp HD — 100% offline, 100% free, open-source AI.**

RetroVision AI is a Windows desktop application built with PySide 6 and PyTorch that
takes VHS captures, phone clips, DVD rips and compressed web video and rebuilds them
with modern deep-learning restoration models that run **entirely on your own machine**.
No cloud, no accounts, no subscriptions, no telemetry, no per-video fees.

![RetroVision AI](assets/banner.png)

---

## What it does

| Capability | Model / engine | Licence |
|---|---|---|
| Super-resolution upscaling (2× / 4×) | Real-ESRGAN (RRDBNet, SRVGGNetCompact) | BSD-3-Clause |
| Face repair & enhancement | GFPGAN v1.4 (StyleGAN2 + CSFT) | Apache-2.0 |
| Frame interpolation (24 → 48/60 fps) | RIFE 4.7 / 4.9 (IFNet) | MIT |
| Temporal restoration (flicker, noise across frames) | BasicVSR++ + SPyNet | Apache-2.0 |
| Denoise, deblur, sharpen, colour restore, brightness/contrast, grain & scratch reduction | Classical OpenCV / NumPy pipelines | MIT / BSD |

Target presets: **Original · 720p · 1080p · 2K · 4K** (aspect ratio always preserved).

---

## Requirements

| | Minimum | Recommended |
|---|---|---|
| OS | Windows 10 x64 | Windows 11 x64 |
| CPU | Any 64-bit, 4 threads | 6+ cores |
| RAM | 8 GB | 16–32 GB |
| GPU | — (CPU works, slowly) | NVIDIA GTX 1060 6 GB or better, CUDA 12.1 driver |
| Disk | 3 GB free (app) | +3 GB for all model weights |
| FFmpeg | Required | Bundled in `bin/` folder or on `PATH` |

> **macOS / Linux** users can run from source: `pip install -r requirements.txt` then
> `python main.py` (PyTorch CPU or CUDA build of your choice).

---

## Installation

### Option A — Pre-built Windows package

1. Download and extract `RetroVisionAI.zip`.
2. Run `RetroVisionAI\RetroVisionAI.exe`.
3. Open **AI Model Manager** and download the models you need (one click each,
   pulled from the official upstream GitHub/OpenMMLab releases).
4. Drop a video on the dashboard and press **Enhance**.

### Option B — From source

```bat
git clone https://github.com/your-org/RetroVisionAI.git
cd RetroVisionAI

python -m venv .venv
.venv\Scripts\activate

:: GPU build (recommended)
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121

:: dependencies
pip install -r requirements.txt

:: verify the environment
python main.py --check

:: run
python main.py
```

CPU-only installs use `--index-url https://download.pytorch.org/whl/cpu`.

---

## Building the Windows executable

```bat
build_windows.bat            :: full build  -> dist\RetroVisionAI\RetroVisionAI.exe
build_windows.bat clean      :: remove build/ and dist/
```

The script creates a virtual environment, installs CUDA PyTorch, runs the test-suite,
and packages with PyInstaller using `RetroVisionAI.spec`.  Model weights are **not**
bundled — the app downloads them on demand.

To make a fully self-contained installer, drop `ffmpeg.exe` and `ffprobe.exe` into
`bin/` before building; the spec file picks them up automatically.

---

## Using the app

### 1. Dashboard
Drop a video, or click one of the **Quick enhancement** recipes:

| Recipe | Stack |
|---|---|
| VHS Cleanup | Denoise + colour restore + 1080p |
| Concert Footage | Face restoration + colour + 2K |
| Archive Master | 4K + interpolation + temporal restore |
| Quick Preview | Fast 720p pass to check settings |

### 2. Video Enhancer
- **Left column** — source preview, target resolution, AI model, and the full
  restoration stack (12 toggles, each with an intensity selector).
- **Right column** — before/after preview with a draggable splitter, live metrics,
  and the output settings (codec, encoder, quality/CRF, audio handling).
- **Enhance Video** — queues the job; progress, stage text, ETA and per-frame
  counters are reported by the real pipeline (never simulated).

### 3. Batch Processing
Add a folder or individual files, apply the same stack to all of them, reorder,
and let the queue run overnight. Each file produces its own output and history
entry; failures do not stop the queue.

### 4. AI Model Manager
Shows every supported model with size, licence, author and install status.
Download with progress and SHA-256 verification, delete unused weights, and
enable/disable individual models.

### 5. GPU Monitor
Live utilisation, VRAM, temperature and encoder availability, plus recommended
tile size / batch-size for your VRAM. Falls back gracefully to CPU telemetry.

### 6. Processing History
Every finished job with the settings used — filter by name, open the output,
reveal it in Explorer, or clear the log.

### 7. Settings
FFmpeg paths (with auto-detect), model/output/temp folders, device and precision,
tile size, batch size, resume-after-crash, keep-temp-frames, log management and a
factory reset.

### 8. About
Application version, every bundled engine with its upstream author and licence,
and live environment information.

---

## Command line

```bat
python main.py                       :: launch the GUI
python main.py --check               :: environment diagnostics (no GUI)
python main.py --version
python main.py "D:\tapes\holiday.avi"  :: open a file directly in the enhancer
python main.py --quick "in.mp4" --preset 1080p --output out.mp4
```

`--check` prints Python/PySide6/PyTorch/CUDA/OpenCV/FFmpeg versions, available
encoders, detected GPU, CPU/RAM, and the install status of every model.

---

## Project layout

```
RetroVisionAI/
├── main.py                  entry point + --check diagnostics
├── config.json              user settings (hand-editable)
├── RetroVisionAI.spec       PyInstaller build spec
├── build_windows.bat        one-click Windows build
├── requirements.txt
├── app/                     PySide6 UI layer
│   ├── theme.py             palette, cards, gauges, stylesheet
│   ├── sidebar.py           grouped navigation rail
│   ├── main_window.py       shell, routing, job wiring
│   ├── dashboard.py  video_enhancer.py  video_preview.py
│   ├── batch_page.py  model_page.py  gpu_page.py
│   └── history_page.py  settings_page.py  about_page.py
├── core/                    processing layer (no Qt imports)
│   ├── pipeline.py          frame-by-frame restoration + encode
│   ├── job.py  job_manager.py   queue, workers, history
│   ├── filters.py           classical filter chain
│   └── video_analyzer.py    source analysis and recommendations
├── ai/                      model layer (pure PyTorch, no mmcv)
│   ├── archs/               RRDBNet, SRVGGNet, StyleGAN2/GFPGAN, IFNet, BasicVSR++
│   ├── realesrgan_engine.py  gfpgan_engine.py  rife_engine.py
│   ├── registry.py          model catalogue (URLs, SHA-256, licences)
│   ├── model_manager.py     download / verify / resolve
│   └── device_manager.py    device, precision, tiling, OOM guard
├── utils/                   config, logging, ffmpeg, file manager, GPU monitor
├── tests/                   pytest suite (unit + real-inference integration)
├── scripts/make_icons.py    icon generation
├── models/                  model weights (downloaded, not shipped)
├── assets/  samples/  docs/
```

---

## Performance

Measured on the reference CPU-only test box (2 vCPU, no GPU) with
Real-ESRGAN general-x4v3 + denoise at 320×240 → 960×720:

| Stage | Time |
|---|---|
| 100 frames, CPU, tile 192 | ≈ 5 min 49 s |
| Per frame | ≈ 3.5 s |

With an RTX 3060 the same job takes **10–20 seconds** (≈ 130–200 fps at 720p).
Approximate GPU throughput for 1080p output:

| GPU | Real-ESRGAN 2× | + GFPGAN | + RIFE |
|---|---|---|---|
| GTX 1060 6 GB | 6–10 fps | 2–4 fps | 12–20 fps |
| RTX 3060 12 GB | 15–25 fps | 6–10 fps | 30–50 fps |
| RTX 4090 24 GB | 40–60 fps | 15–25 fps | 70–110 fps |

The pipeline adapts: it slices frames into tiles sized to your VRAM, batches where
possible, and catches CUDA OOM to retry with smaller tiles instead of crashing.

---

## Model weights

Weights are **never** downloaded silently and never bundled in the installer.
The Model Manager shows the exact upstream URL and SHA-256 before you start, then
verifies the hash after download. If a hash does not match, the file is discarded.

| Key | File | Source |
|---|---|---|
| `realesrgan_x4plus` | `RealESRGAN_x4plus.pth` | xinntao/Real-ESRGAN v0.2.1 |
| `realesrgan_x2plus` | `RealESRGAN_x2plus.pth` | xinntao/Real-ESRGAN v0.2.1 |
| `realesrgan_anime6b` | `RealESRGAN_x4plus_anime_6B.pth` | xinntao/Real-ESRGAN v0.2.2.4 |
| `realesr_animevideo` | `realesr-animevideov3.pth` | xinntao/Real-ESRGAN v0.2.5.0 |
| `realesr_general_x4v3` | `realesr-general-x4v3.pth` | xinntao/Real-ESRGAN v0.2.5.0 |
| `gfpgan_v14` | `GFPGANv1.4.pth` | TencentARC/GFPGAN v1.3.4 |
| `rife47` / `rife49` | `rife47.pth`, `rife49.pth` | Fannovel16/ComfyUI-Frame-Interpolation `models` |
| `basicvsr_pp` | `basicvsr_plusplus_c64n7_8x1_600k_reds4_20210217-db622b2f.pth` | OpenMMLab mmediting |
| `spynet` | `spynet_20210409-c6c1bd09.pth` | OpenMMLab mmediting |

Manual install: drop the file into `models/` with the exact filename above and
press **Rescan** in the Model Manager.

---

## Testing

```bash
python -m pytest tests -q                 # full suite
python -m pytest tests -q -k "not real"   # skip weight-dependent tests
```

The suite covers configuration, FFmpeg probe/extract/rebuild round-trips, filter
behaviour, model registry integrity and download verification, device/tiling math,
OOM handling, job queue and history persistence, every UI page building headlessly,
and a real end-to-end restoration (super-resolution + audio passthrough + encode).

---

## Licences

RetroVision AI is released under the **MIT licence**.
Third-party components: Real-ESRGAN (BSD-3-Clause), GFPGAN (Apache-2.0),
RIFE (MIT), BasicVSR++/MMEditing (Apache-2.0), PyTorch (BSD-3-Clause),
PySide6 (LGPL-3.0), OpenCV (Apache-2.0), FFmpeg (LGPL/GPL depending on build —
if you redistribute FFmpeg binaries, comply with its licence).

### Disclaimer

Restoration output is generated by machine-learning models and may differ from the
original footage; it can introduce artefacts or invent detail. Always keep your
original files — RetroVision AI never modifies or deletes source videos. The
authors are not liable for any use of restored material that infringes
third-party rights.
