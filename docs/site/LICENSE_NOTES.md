# Third-party licences and attribution

RetroVision AI itself: **MIT**.

## AI models (weights are downloaded by the user, never bundled)

| Component | Licence | Upstream |
|---|---|---|
| Real-ESRGAN (RRDBNet, SRVGGNetCompact, weights) | BSD-3-Clause | https://github.com/xinntao/Real-ESRGAN |
| GFPGAN v1.4 (StyleGAN2 + CSFT, weights) | Apache-2.0 | https://github.com/TencentARC/GFPGAN |
| RIFE 4.7 / 4.9 (IFNet, weights) | MIT | https://github.com/hzwer/RIFE · packaging: Fannovel16/ComfyUI-Frame-Interpolation |
| BasicVSR++ / SPyNet (weights) | Apache-2.0 | https://github.com/open-mmlab/mmagic |

## Libraries

| Library | Licence |
|---|---|
| PySide6 / Qt 6 | LGPL-3.0 |
| PyTorch, torchvision | BSD-3-Clause |
| OpenCV | Apache-2.0 |
| NumPy | BSD-3-Clause |
| Pillow | HPND |
| psutil | BSD-3-Clause |
| requests | Apache-2.0 |
| FFmpeg | LGPL-2.1+ or GPL-2.0+ depending on build |

## Redistribution notes

* PySide6 is LGPL: the packaged build links it dynamically and ships the Qt
  DLLs unmodified, satisfying the relinking requirement.
* FFmpeg is **not** shipped with the default build. If you bundle `ffmpeg.exe`,
  use an LGPL build if you redistribute commercially, and include its licence
  text and source offer.
* Model weights keep their upstream licence. RetroVision AI does not relicense
  them; the Model Manager displays the licence next to every download.
* If you publish restored footage, you are responsible for the rights to the
  underlying material.

## Attribution shown in-app

The About page lists every engine with author and licence, and the Model Manager
shows the same information per model, together with its source URL and SHA-256.
