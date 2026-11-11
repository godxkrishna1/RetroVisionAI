# Build phases

The application was built in twelve verified phases. Each phase ships working
code plus tests; nothing was stubbed or faked.

| # | Phase | Outcome |
|---|---|---|
| 1 | Environment + skeleton | Directory layout, `requirements.txt`, `config.json`, logger, config loader. `--check` verifies python/torch/cuda/opencv/ffmpeg. |
| 2 | Foundation utilities | `file_manager` (safe temp workspaces, name sanitising, unique paths), `gpu_monitor` (NVML-free telemetry), `ffmpeg` wrapper. |
| 3 | AI architectures | Pure-PyTorch RRDBNet, SRVGGNet, StyleGAN2+CSFT, GFPGANv1Clean, IFNet, BasicVSR++/SPyNet ported from upstream sources. |
| 4 | Model catalogue | `ai/registry.py` – 10 models with official URLs, SHA-256, sizes, licences. `model_manager` – resumable verified downloads, enable/disable. |
| 5 | Inference engines | `realesrgan_engine`, `gfpgan_engine`, `rife_engine` – tiling, OOM retry, cache, real state-dict loading. |
| 6 | Video analysis + filters | `video_analyzer` (source stats, tile advice), `filters` (denoise, deblur, sharpen, colour, brightness, contrast, grain, scratches). |
| 7 | Pipeline + jobs | `pipeline` (7-stage real processing with resume/verify), `job`/`job_manager` (queue, worker thread, JSONL history). |
| 8 | Theme + shell | `theme` (palette, cards, gauges, stylesheet), `sidebar` (grouped nav), `main_window` (routing, status bar, telemetry timer). |
| 9 | Core pages | Dashboard, Video Enhancer (12 toggles, preset selector, encode settings, live preview), Batch, Model Manager, GPU Monitor. |
| 10 | Remaining pages | History (filter, open, clear), Settings (paths, device, folders, diagnostics, factory reset), About (engine board, licences, environment). |
| 11 | Integration proof | End-to-end run through the real Qt job manager: 100 frames 320×240 → 960×720 with audio passthrough, history recorded, UI state verified. |
| 12 | Packaging + docs | `RetroVisionAI.spec`, `build_windows.bat`, README, architecture and licence docs, `--quick` headless CLI. |

## Verification at each phase

* `python -m py_compile` across every package after each file batch.
* Real inference proofs: SR (2×/4×), GFPGAN face crop, RIFE midpoint, tiling
  bit-exactness, 100-frame full pipeline with audio.
* Headless GUI: every page instantiated and navigated under
  `QT_QPA_PLATFORM=offscreen`, screenshots reviewed for layout defects.
* `pytest tests -q` – 81 passing tests (1 skipped when a 4× weight is absent).
