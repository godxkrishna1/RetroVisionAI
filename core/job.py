"""Job specification and live progress data structures."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

from utils.config import INTENSITIES, TARGET_RESOLUTIONS
from utils.ffmpeg import EncodeSettings


class JobStage:
    QUEUED = "Queued"
    ANALYZING = "Analyzing video"
    PREPARING = "Preparing workspace"
    EXTRACTING_AUDIO = "Extracting audio"
    EXTRACTING = "Extracting frames"
    LOADING_MODELS = "Loading AI models"
    RESTORING = "Restoring frames"
    INTERPOLATING = "Interpolating motion"
    TEMPORAL = "Temporal restoration"
    ENCODING = "Encoding video"
    MUXING = "Muxing audio"
    VERIFYING = "Verifying output"
    DONE = "Completed"
    FAILED = "Failed"
    CANCELLED = "Cancelled"


@dataclass
class EnhancementSettings:
    """Everything the enhancer page collects, validated."""

    target_resolution: str = "1080p"
    ai_model: str = "realesrgan_x2plus"
    use_super_resolution: bool = True

    use_face_restoration: bool = False
    face_strength: float = 0.8

    use_denoise: bool = True
    denoise_strength: str = "medium"
    use_deblur: bool = False
    deblur_strength: str = "medium"
    use_sharpen: bool = False
    sharpen_strength: str = "medium"
    use_color_restore: bool = True
    color_strength: str = "medium"
    use_brightness_fix: bool = False
    brightness_strength: str = "medium"
    use_contrast_fix: bool = False
    contrast_strength: str = "medium"
    use_grain_reduction: bool = False
    grain_strength: str = "medium"
    use_scratch_reduction: bool = False
    scratch_strength: str = "medium"
    use_interpolation: bool = False
    interpolation_factor: int = 2
    interpolation_model: str = "rife47"
    use_temporal_restore: bool = False
    temporal_window: int = 5

    def validate(self) -> List[str]:
        warnings: List[str] = []
        if self.target_resolution not in TARGET_RESOLUTIONS:
            self.target_resolution = "1080p"
        if self.interpolation_factor not in (2, 3, 4):
            self.interpolation_factor = 2
        for name in ("denoise_strength", "deblur_strength", "sharpen_strength",
                     "color_strength", "brightness_strength", "contrast_strength",
                     "grain_strength", "scratch_strength"):
            if getattr(self, name) not in INTENSITIES:
                setattr(self, name, "medium")
        self.face_strength = max(0.0, min(1.0, float(self.face_strength)))
        if self.use_temporal_restore and self.use_interpolation:
            warnings.append(
                "BasicVSR++ temporal restoration and RIFE interpolation run in "
                "separate passes and make processing much slower.")
        return warnings

    def ai_steps(self) -> List[str]:
        steps = []
        if self.use_super_resolution:
            steps.append("superres")
        if self.use_face_restoration:
            steps.append("face")
        if self.use_temporal_restore:
            steps.append("temporal")
        if self.use_interpolation:
            steps.append("interpolation")
        return steps

    def filters_enabled(self) -> bool:
        return any(getattr(self, k) for k in (
            "use_denoise", "use_deblur", "use_sharpen", "use_color_restore",
            "use_brightness_fix", "use_contrast_fix", "use_grain_reduction",
            "use_scratch_reduction"))

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "EnhancementSettings":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in (data or {}).items() if k in known})


@dataclass
class JobSpec:
    input_path: str
    enhancement: EnhancementSettings
    encode: EncodeSettings
    output_path: str
    job_id: str = field(default_factory=lambda: f"job_{uuid.uuid4().hex[:10]}")
    priority: int = 0
    label: str = ""

    def __post_init__(self) -> None:
        if not self.label:
            self.label = Path(self.input_path).stem

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["enhancement"] = self.enhancement.to_dict()
        d["encode"] = asdict(self.encode)
        return d


@dataclass
class JobProgress:
    """Real counters only - never synthesised."""

    stage: str = JobStage.QUEUED
    stage_index: int = 0
    stage_count: int = 6
    frames_done: int = 0
    frames_total: int = 0
    fps: float = 0.0
    elapsed: float = 0.0
    eta: Optional[float] = None
    message: str = ""
    percent: float = 0.0
    output_frames: int = 0
    detail: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class JobResult:
    job_id: str
    status: str                     # completed | failed | cancelled
    output_path: str = ""
    error: str = ""
    input_info: Dict = field(default_factory=dict)
    output_info: Dict = field(default_factory=dict)
    duration_s: float = 0.0
    frames_processed: int = 0
    started_at: float = field(default_factory=time.time)
    finished_at: float = field(default_factory=time.time)
    log_tail: str = ""
