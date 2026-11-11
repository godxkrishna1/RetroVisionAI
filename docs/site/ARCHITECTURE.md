# Architecture

RetroVision AI is split into four layers with a strict one-way dependency rule:
`app` → `core` → `ai` → `utils`.  No layer below imports anything from a layer
above it, and `core`/`ai`/`utils` never import Qt.

```
┌─────────────────────────────────────────────────────────────┐
│ app/     PySide6 UI: pages, theme, sidebar, main window     │
│          owns nothing but presentation + user intent         │
└──────────────────────────┬──────────────────────────────────┘
                           │ JobSpec / signals
┌──────────────────────────▼──────────────────────────────────┐
│ core/    orchestration: pipeline, job queue, filters,        │
│          video analysis.  Pure Python + OpenCV + FFmpeg.     │
└──────────────────────────┬──────────────────────────────────┘
                           │ numpy frames
┌──────────────────────────▼──────────────────────────────────┐
│ ai/      inference: engine wrappers over pure-torch archs,   │
│          model registry/manager, device + tiling policy.     │
└──────────────────────────┬──────────────────────────────────┘
                           │ paths, logs, config
┌──────────────────────────▼──────────────────────────────────┐
│ utils/   config, logging, FFmpeg wrapper, file manager,      │
│          GPU monitor.  Zero domain knowledge.                │
└─────────────────────────────────────────────────────────────┘
```

## Threading model

* The GUI thread only ever touches widgets and signal emission.
* `core.job_manager.JobManager` runs a `QThread` worker per job, serialised by a
  queue, so batch processing is naturally sequential (no GPU contention).
* Progress crosses the thread boundary as Qt signals (`job_progress`,
  `job_stage`, `job_finished`, …), never as shared mutable state.
* `pipeline.VideoPipeline` calls back with a `JobProgress` dataclass; the worker
  throttles these to ~10 Hz before emitting so the UI stays responsive.

## The pipeline

1. **Analyze** – FFprobe metadata → `VideoInfo`, source histogram check.
2. **Extract audio** – stream-copy to a temp file when the container allows it,
   so the original soundtrack survives untouched.
3. **Extract frames** – FFmpeg decodes to a numbered PNG sequence. PNG (not JPG)
   because the restore stage must not see compression artefacts it would then
   amplify.
4. **Load models** – only the engines the settings actually ask for; each is
   cached across frames and released on unload.
5. **Restore frames** – per frame: temporal restore → super-resolution →
   face restoration → classical filter chain.  Optional interpolation pass emits
   `factor-1` synthetic frames between each pair.
6. **Encode** – NVENC when the driver really supports it (functional probe), else
   libx264/libx265. Audio is muxed back or re-encoded per settings.
7. **Verify** – probe the output, compare frame count/duration, report a real
   result rather than assuming success.

Resume state is written to `temp/<job_id>/resume.json` after each stage so a
crashed or cancelled job continues where it stopped instead of restarting.

## Memory strategy

`ai.device_manager` decides tile size and batch size from reported VRAM:

| VRAM | tile | batch |
|---|---|---|
| ≥ 16 GB | untiled (0) | 8 |
| 8–16 GB | 512 | 4 |
| 4–8 GB | 384 | 2 |
| 2–4 GB | 192 | 1 |
| CPU / unknown | 192 | 1 |

Every inference call runs inside `OOMGuard`, which catches CUDA OOM, halves the
tile, empties the cache and retries — a 4K job on a 4 GB card finishes slowly
instead of dying.

## Why the architectures are inlined

Upstream Real-ESRGAN/GFPGAN/RIFE/BasicVSR++ all depend on `basicsr` + `mmcv`,
which pull compiled CUDA extensions and pin old torch versions.  Re-implementing
the networks (`ai/archs/`) from their published sources removes that entire
class of install failure on a user's Windows machine.  The state-dict key
layouts were matched against the official checkpoints and are covered by
`tests/test_ai_engines.py`.
