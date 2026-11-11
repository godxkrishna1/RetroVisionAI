"""Pure-PyTorch model architectures ported for RetroVision AI.

Nothing here imports ``basicsr``, ``mmcv`` or ``realesrgan`` - the original
networks are re-implemented from their published sources so the app keeps a
small, dependency-free inference tree.

Public names
------------
RRDBNet, SRVGGNetCompact                  Real-ESRGAN generators
StyleGAN2GeneratorCSFT, GFPGANv1Clean     GFPGAN face restoration
IFNet                                     RIFE intermediate flow estimator
BasicVSRPlusPlusNet, SPyNet, flow_warp    BasicVSR++ temporal restoration
"""
from __future__ import annotations

from .basicvsr_pp import BasicVSRPlusPlusNet, SPyNet, flow_warp, load_basicvsr_pp
from .gfpganv1 import GFPGANv1Clean, StyleGAN2GeneratorCSFT
from .rife import IFNet, load_ifnet
from .rrdbnet import RRDBNet, SRVGGNetCompact
from .stylegan2_clean import StyleGAN2GeneratorClean

__all__ = [
    "RRDBNet",
    "SRVGGNetCompact",
    "StyleGAN2GeneratorClean",
    "StyleGAN2GeneratorCSFT",
    "GFPGANv1Clean",
    "IFNet",
    "load_ifnet",
    "BasicVSRPlusPlusNet",
    "SPyNet",
    "flow_warp",
    "load_basicvsr_pp",
]
