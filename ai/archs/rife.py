"""RIFE 4.x frame-interpolation network (IFNet).

Port of the official ``model/IFNet.py`` from hzwer/Practical-RIFE (MIT),
matching the parameter layout of ``rife47.pth`` / ``rife49.pth`` distributed
by the ComfyUI-Frame-Interpolation release page.

Paper: Huang et al., "Real-Time Intermediate Flow Estimation for Video Frame
Interpolation", ECCV 2022.
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResConv(nn.Module):
    def __init__(self, c: int, dilation: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, 1, dilation, dilation=dilation, groups=1)
        self.beta = nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x) * self.beta + x)


def warp(ten_input: torch.Tensor, ten_flow: torch.Tensor) -> torch.Tensor:
    """Backward warp ``ten_input`` by ``ten_flow`` (pixel units)."""
    device = ten_input.device
    h, w = ten_input.shape[2], ten_input.shape[3]
    grid_y, grid_x = torch.meshgrid(
        torch.arange(0, h, device=device, dtype=ten_input.dtype),
        torch.arange(0, w, device=device, dtype=ten_input.dtype),
        indexing="ij")
    grid = torch.stack((grid_x, grid_y), 2).unsqueeze(0)   # 1,h,w,2

    flow = ten_flow.permute(0, 2, 3, 1)                    # b,h,w,2
    grid_flow = grid + flow
    # normalise to [-1, 1]
    grid_flow_x = 2.0 * grid_flow[..., 0] / max(w - 1.0, 1.0) - 1.0
    grid_flow_y = 2.0 * grid_flow[..., 1] / max(h - 1.0, 1.0) - 1.0
    grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=-1)
    return F.grid_sample(ten_input, grid_flow, mode="bilinear",
                         padding_mode="border", align_corners=True)


def conv(in_planes: int, out_planes: int, kernel_size: int = 3, stride: int = 1,
         padding: int = 1, dilation: int = 1, arch_ver: str = "4.7") -> nn.Sequential:
    """Conv + LeakyReLU used by RIFE >= 4.2 checkpoints."""
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.LeakyReLU(0.2, True))


class IFBlock(nn.Module):
    def __init__(self, in_planes: int, c: int = 64, arch_ver: str = "4.7") -> None:
        super().__init__()
        self.arch_ver = arch_ver
        self.conv0 = nn.Sequential(
            conv(in_planes, c // 2, 3, 2, 1, arch_ver=arch_ver),
            conv(c // 2, c, 3, 2, 1, arch_ver=arch_ver),
        )
        self.convblock = nn.Sequential(
            ResConv(c), ResConv(c), ResConv(c), ResConv(c),
            ResConv(c), ResConv(c), ResConv(c), ResConv(c),
        )
        self.lastconv = nn.Sequential(
            nn.ConvTranspose2d(c, 4 * 6, 4, 2, 1), nn.PixelShuffle(2))

    def forward(self, x: torch.Tensor, flow: torch.Tensor | None = None,
                scale: float = 1):
        if scale != 1:
            x = F.interpolate(x, scale_factor=1.0 / scale, mode="bilinear",
                              align_corners=False)
        if flow is not None:
            if scale != 1:
                flow = F.interpolate(flow, scale_factor=1.0 / scale,
                                     mode="bilinear", align_corners=False) / scale
            x = torch.cat((x, flow), 1)
        feat = self.conv0(x)
        feat = self.convblock(feat)
        tmp = self.lastconv(feat)
        if scale != 1:
            tmp = F.interpolate(tmp, scale_factor=scale, mode="bilinear",
                                align_corners=False)
            flow = tmp[:, :4] * scale
        else:
            flow = tmp[:, :4]
        mask = tmp[:, 4:5]
        return flow, mask


class IFNet(nn.Module):
    """RIFE 4.7 / 4.9 network (encode + 4 IFBlocks)."""

    def __init__(self, arch_ver: str = "4.7") -> None:
        super().__init__()
        self.arch_ver = arch_ver
        in0 = 15 if arch_ver in ("4.7", "4.9") else 23
        mid = 20 if arch_ver in ("4.7", "4.9") else 28
        self.block0 = IFBlock(in0, c=192, arch_ver=arch_ver)
        self.block1 = IFBlock(mid, c=128, arch_ver=arch_ver)
        self.block2 = IFBlock(mid, c=96, arch_ver=arch_ver)
        self.block3 = IFBlock(mid, c=64, arch_ver=arch_ver)
        self.encode = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1), nn.ConvTranspose2d(16, 4, 4, 2, 1))

    def forward(self, img0: torch.Tensor, img1: torch.Tensor,
                timestep: float = 0.5, scale_list: Tuple[float, ...] = (16, 8, 4, 2, 1),
                training: bool = False, fastmode: bool = True,
                ensemble: bool = False):
        img0 = torch.clamp(img0, 0, 1)
        img1 = torch.clamp(img1, 0, 1)

        n, c, h, w = img0.shape
        ph = ((h - 1) // 64 + 1) * 64
        pw = ((w - 1) // 64 + 1) * 64
        padding = (0, pw - w, 0, ph - h)
        img0 = F.pad(img0, padding)
        img1 = F.pad(img1, padding)

        timestep_t = torch.tensor([timestep], dtype=img0.dtype,
                                  device=img0.device).view(1, 1, 1, 1).expand(
            n, 1, img0.shape[2], img0.shape[3]).contiguous()

        f0 = self.encode(img0[:, :3])
        f1 = self.encode(img1[:, :3])

        warped_img0 = img0
        warped_img1 = img1
        flow = None
        mask = None
        merged: List[Tuple[torch.Tensor, torch.Tensor]] = []
        blocks = [self.block0, self.block1, self.block2, self.block3]
        scales = list(scale_list)

        for i in range(4):
            if flow is None:
                flow, mask = blocks[i](
                    torch.cat((img0[:, :3], img1[:, :3], f0, f1, timestep_t), 1),
                    None, scale=scales[i])
                if ensemble:
                    f_, m_ = blocks[i](
                        torch.cat((img1[:, :3], img0[:, :3], f1, f0, 1 - timestep_t), 1),
                        None, scale=scales[i])
                    flow = (flow + torch.cat((f_[:, 2:4], f_[:, :2]), 1)) / 2
                    mask = (mask + (-m_)) / 2
            else:
                fd, m0 = blocks[i](
                    torch.cat((warped_img0[:, :3], warped_img1[:, :3],
                               warp(f0, flow[:, :2]), warp(f1, flow[:, 2:4]),
                               timestep_t, mask), 1),
                    flow, scale=scales[i])
                flow = flow + fd
                if ensemble:
                    f_, m_ = blocks[i](
                        torch.cat((warped_img1[:, :3], warped_img0[:, :3],
                                   warp(f1, flow[:, 2:4]), warp(f0, flow[:, :2]),
                                   1 - timestep_t, -mask), 1),
                        torch.cat((flow[:, 2:4], flow[:, :2]), 1), scale=scales[i])
                    fd = (fd + torch.cat((f_[:, 2:4], f_[:, :2]), 1)) / 2
                    mask = (m0 + (-m_)) / 2
                else:
                    mask = m0

            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])
            merged.append((warped_img0, warped_img1))

        mask = torch.sigmoid(mask)
        merged_final = warped_img0 * mask + warped_img1 * (1 - mask)

        if not fastmode:
            # refine stage of RIFE 4.0-4.6 is not present in 4.7 checkpoints;
            # fastmode=True is the reference inference path.
            pass
        return torch.clamp(merged_final, 0, 1)


def load_ifnet(weights_path, device="cpu", half: bool = False,
               arch_ver: str = "4.7") -> IFNet:
    """Instantiate IFNet and load a RIFE checkpoint (dict or wrapped)."""
    net = IFNet(arch_ver=arch_ver)
    state = torch.load(weights_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and "module." in next(iter(state), ""):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    state = {k: v for k, v in state.items() if not k.startswith("step_counter")}
    missing, unexpected = net.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"RIFE checkpoint mismatch (missing={len(missing)}, "
            f"unexpected={len(unexpected)}). Is this a RIFE 4.7/4.9 model?")
    net.eval()
    if half:
        net = net.half()
    return net.to(device)
