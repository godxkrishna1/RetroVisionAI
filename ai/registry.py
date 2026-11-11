"""Official model catalogue.

Every entry records the *official* upstream URL, the exact file size and the
SHA-256 digest of the released weights (computed from the official files) so
the Model Manager can verify downloads and refuse corrupted ones.

Nothing is ever downloaded without explicit user action.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class ModelSpec:
    key: str                  # stable id used in configs / filenames
    title: str                # human readable name
    category: str             # upscale | face | interpolation | temporal | auxiliary
    filename: str             # file name inside models/
    url: str                  # official release URL
    size_bytes: int
    sha256: str
    version: str
    license: str
    author: str
    description: str
    scale: int = 4            # upscale factor (0 for non-SR)
    arch: str = ""            # arch id used by the engines
    default_for: str = ""     # "general" | "anime" | "compact" | ...
    homepage: str = ""
    requires: List[str] = field(default_factory=list)
    recommended_vram_mb: int = 0
    short_name: str = ""
    quality: str = ""
    speed: str = ""

    @property
    def display_size(self) -> str:
        gb = self.size_bytes / (1024 ** 3)
        if gb >= 1:
            return f"{gb:.2f} GB"
        return f"{self.size_bytes / (1024 ** 2):.1f} MB"


_CATALOGUE: List[ModelSpec] = [
    # ----------------------------------------------------------- Real-ESRGAN
    ModelSpec(
        key="realesrgan_x4plus",
        title="Real-ESRGAN x4plus",
        category="upscale",
        filename="RealESRGAN_x4plus.pth",
        url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        size_bytes=67040989,
        sha256="4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1",
        version="1.0",
        license="BSD-3-Clause",
        author="Xintao Wang / Tencent ARC",
        description="General-purpose 4x super-resolution for real-world footage. "
                    "Best all-round choice for 80s/90s music videos and films.",
        scale=4, arch="rrdb", default_for="general",
        homepage="https://github.com/xinntao/Real-ESRGAN",
        recommended_vram_mb=4096, short_name="x4plus",
        quality="highest", speed="slow"),
    ModelSpec(
        key="realesrgan_x2plus",
        title="Real-ESRGAN x2plus",
        category="upscale",
        filename="RealESRGAN_x2plus.pth",
        url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        size_bytes=67061725,
        sha256="49fafd45f8fd7aa8d31ab2a22d14d91b536c34494a5cfe31eb5d89c2fa266abb",
        version="2.0",
        license="BSD-3-Clause",
        author="Xintao Wang / Tencent ARC",
        description="2x super-resolution - ideal for DVD/VHS sources going to "
                    "720p/1080p, with fewer artefacts than a 4x pass.",
        scale=2, arch="rrdb", default_for="general",
        homepage="https://github.com/xinntao/Real-ESRGAN",
        recommended_vram_mb=4096, short_name="x2plus",
        quality="high", speed="medium"),
    ModelSpec(
        key="realesrgan_anime6b",
        title="Real-ESRGAN x4plus anime 6B",
        category="upscale",
        filename="RealESRGAN_x4plus_anime_6B.pth",
        url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        size_bytes=17938799,
        sha256="f872d837d3c90ed2e05227bed711af5671a6fd1c9f7d7e91c911a61f155e99da",
        version="1.0",
        license="BSD-3-Clause",
        author="Xintao Wang / Tencent ARC",
        description="Compact 4x model tuned for animation. Excellent for 80s "
                    "cartoons and animated music videos.",
        scale=4, arch="rrdb6", default_for="anime",
        homepage="https://github.com/xinntao/Real-ESRGAN",
        recommended_vram_mb=2048, short_name="anime6B",
        quality="high", speed="fast"),
    ModelSpec(
        key="realesr_animevideo",
        title="Real-ESRGAN anime video v3",
        category="upscale",
        filename="realesr-animevideov3.pth",
        url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth",
        size_bytes=2504012,
        sha256="b8a8376811077954d82ca3fcf476f1ac3da3e8a68a4f4d71363008000a18b75d",
        version="3.0",
        license="BSD-3-Clause",
        author="Xintao Wang / Tencent ARC",
        description="Tiny, very fast 4x model for anime video. Low VRAM usage.",
        scale=4, arch="srvgg", default_for="anime-fast",
        homepage="https://github.com/xinntao/Real-ESRGAN",
        recommended_vram_mb=1024, short_name="anime-v3",
        quality="medium", speed="very fast"),
    ModelSpec(
        key="realesr_general_x4v3",
        title="Real-ESRGAN general x4v3",
        category="upscale",
        filename="realesr-general-x4v3.pth",
        url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
        size_bytes=4885111,
        sha256="8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292",
        version="3.0",
        license="BSD-3-Clause",
        author="Xintao Wang / Tencent ARC",
        description="Small general 4x model - a good balance of quality and "
                    "speed for quick previews and lower-end GPUs.",
        scale=4, arch="srvgg", default_for="general-fast",
        homepage="https://github.com/xinntao/Real-ESRGAN",
        recommended_vram_mb=1024, short_name="general-v3",
        quality="medium-high", speed="fast"),
    # --------------------------------------------------------------- GFPGAN
    ModelSpec(
        key="gfpgan_v14",
        title="GFPGAN v1.4",
        category="face",
        filename="GFPGANv1.4.pth",
        url="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth",
        size_bytes=348632874,
        sha256="e2cd4703ab14f4d01fd1383a8a8b266f9a5833dacee8e6a79d3bf21a1b6be5ad",
        version="1.4",
        license="Apache-2.0",
        author="Tencent ARC",
        description="Blind face restoration with a StyleGAN2 prior. Repairs "
                    "eyes, teeth and skin detail in close-up 80s/90s footage.",
        scale=1, arch="gfpgan1",
        homepage="https://github.com/TencentARC/GFPGAN",
        recommended_vram_mb=3072, short_name="GFPGAN",
        quality="highest", speed="slow"),
    # ----------------------------------------------------------------- RIFE
    ModelSpec(
        key="rife47",
        title="RIFE 4.7",
        category="interpolation",
        filename="rife47.pth",
        url="https://github.com/Fannovel16/ComfyUI-Frame-Interpolation/releases/download/models/rife47.pth",
        size_bytes=21344827,
        sha256="6a8a825ab2750558bdd20dcced386fd82b7222c7ba58c11d3b611d9c44f1be63",
        version="4.7",
        license="MIT (RIFE) / MIT (ComfyUI-Frame-Interpolation packaging)",
        author="Zhewei Huang (hzwer), packaged by Fannovel16",
        description="Real-time intermediate flow estimation. Doubles frame "
                    "rate (e.g. 25 -> 50 fps) for smooth slow-motion and modern "
                    "playback on 60 Hz displays.",
        scale=0, arch="rife",
        homepage="https://github.com/hzwer/Practical-RIFE",
        recommended_vram_mb=2048, short_name="RIFE 4.7",
        quality="high", speed="fast"),
    ModelSpec(
        key="rife49",
        title="RIFE 4.9",
        category="interpolation",
        filename="rife49.pth",
        url="https://github.com/Fannovel16/ComfyUI-Frame-Interpolation/releases/download/models/rife49.pth",
        size_bytes=21345274,
        sha256="e55fd00f3cc184e3c65961f4bb827a9da022e78eed36b055242c0ac30000d533",
        version="4.9",
        license="MIT (RIFE) / MIT (ComfyUI-Frame-Interpolation packaging)",
        author="Zhewei Huang (hzwer), packaged by Fannovel16",
        description="Newer RIFE checkpoint with better handling of occlusion "
                    "and motion blur than 4.7.",
        scale=0, arch="rife",
        homepage="https://github.com/hzwer/Practical-RIFE",
        recommended_vram_mb=2048, short_name="RIFE 4.9",
        quality="highest", speed="fast"),
    # ------------------------------------------------------------ BasicVSR++
    ModelSpec(
        key="basicvsr_pp",
        title="BasicVSR++ (REDS4)",
        category="temporal",
        filename="basicvsr_plusplus_c64n7_8x1_600k_reds4_20210217-db622b2f.pth",
        url="https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/"
            "basicvsr_plusplus_c64n7_8x1_600k_reds4_20210217-db622b2f.pth",
        size_bytes=29349005,
        sha256="db622b2fd4caae0a4c63ab5e54f1cfef7a62a0f3b8ad101aba2eae068d928549",
        version="1.0 (2021-02-17)",
        license="Apache-2.0",
        author="OpenMMLab / Shanghai AI Lab",
        description="Recurrent video super-resolution with optical-flow-guided "
                    "deformable alignment. Restores 4x while keeping motion "
                    "consistent between frames.",
        scale=4, arch="basicvsrpp",
        homepage="https://github.com/open-mmlab/mmagic",
        requires=["spynet"],
        recommended_vram_mb=6144, short_name="BasicVSR++",
        quality="highest", speed="very slow"),
    ModelSpec(
        key="spynet",
        title="SPyNet optical flow",
        category="auxiliary",
        filename="spynet_20210409-c6c1bd09.pth",
        url="https://download.openmmlab.com/mmediting/restorers/basicvsr/"
            "spynet_20210409-c6c1bd09.pth",
        size_bytes=5770988,
        sha256="c6c1bd09b52d05ba17f3e701f549d6faf5e314aabce8ae462c1c171a8d6c4914",
        version="2021-04-09",
        license="Apache-2.0",
        author="OpenMMLab",
        description="Spatial-pyramid optical flow network required by "
                    "BasicVSR++. Installed automatically with it.",
        scale=0, arch="spynet",
        homepage="https://github.com/open-mmlab/mmagic",
        recommended_vram_mb=512, short_name="SPyNet",
        quality="-", speed="fast"),
]

CATALOGUE: Dict[str, ModelSpec] = {m.key: m for m in _CATALOGUE}


def all_models() -> List[ModelSpec]:
    return list(_CATALOGUE)


def get_model(key: str) -> Optional[ModelSpec]:
    return CATALOGUE.get(key)


def by_category(category: str) -> List[ModelSpec]:
    return [m for m in _CATALOGUE if m.category == category]


def upscale_models() -> List[ModelSpec]:
    return by_category("upscale")


def default_upscaler() -> ModelSpec:
    return CATALOGUE["realesrgan_x2plus"]


def resolve_with_dependencies(keys: List[str]) -> List[ModelSpec]:
    """Expand a key list to include required auxiliary models (e.g. SPyNet)."""
    out: List[ModelSpec] = []
    seen = set()

    def add(key: str) -> None:
        if key in seen or key not in CATALOGUE:
            return
        seen.add(key)
        spec = CATALOGUE[key]
        out.append(spec)
        for dep in spec.requires:
            add(dep)

    for k in keys:
        add(k)
    return out


LICENSE_SUMMARY = [
    ("Real-ESRGAN", "BSD-3-Clause", "https://github.com/xinntao/Real-ESRGAN"),
    ("GFPGAN", "Apache-2.0", "https://github.com/TencentARC/GFPGAN"),
    ("RIFE", "MIT", "https://github.com/hzwer/Practical-RIFE"),
    ("BasicVSR++ / SPyNet (MMagic)", "Apache-2.0",
     "https://github.com/open-mmlab/mmagic"),
    ("PyTorch", "BSD-3-Clause", "https://pytorch.org"),
    ("PySide6 / Qt for Python", "LGPL-3.0", "https://www.qt.io/qt-for-python"),
    ("FFmpeg", "LGPL/GPL (build dependent)", "https://ffmpeg.org"),
    ("OpenCV", "Apache-2.0", "https://opencv.org"),
]
