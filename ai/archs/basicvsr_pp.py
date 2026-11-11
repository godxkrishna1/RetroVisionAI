"""BasicVSR++ video super-resolution architecture.

Port of ``mmagic/models/editors/basicvsr_plusplus_net`` (OpenMMLab MMagic,
Apache-2.0) with three compatibility shims so it runs without mmcv/mmengine:

* ``ModulatedDeformConv2d`` / ``modulated_deform_conv2d`` are re-implemented
  in pure PyTorch (see :class:`ModulatedDeformConv2dPure`);
* ``PixelShufflePack`` is re-implemented locally;
* checkpoint loading tolerates the ``generator.`` prefix produced by
  MMagic training.

Loads ``basicvsr_plusplus_c64n7_8x1_600k_reds4_20210217-db622b2f.pth``.
SPyNet optical-flow weights are loaded from the separate ``spynet_20210409``
checkpoint bundled by the model manager.

Speed note: the pure-PyTorch deformable conv is slower than mmcv's CUDA
kernel but works everywhere (CUDA and CPU).  It processes frames in windows
to keep VRAM bounded.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Compatibility layers (replace mmcv / mmengine)
# ---------------------------------------------------------------------------


def flow_warp(x: torch.Tensor, flow: torch.Tensor, interpolation: str = "bilinear",
              padding_mode: str = "zeros", align_corners: bool = True) -> torch.Tensor:
    """Warp ``x`` with ``flow`` of shape (n, h, w, 2) (pixel offsets)."""
    if x.size()[-2:] != flow.size()[1:3]:
        raise ValueError(f"Spatial sizes differ: {x.size()[-2:]} vs {flow.size()[1:3]}")
    _, _, h, w = x.size()
    device = flow.device
    grid_y, grid_x = torch.meshgrid(
        torch.arange(0, h, device=device, dtype=x.dtype),
        torch.arange(0, w, device=device, dtype=x.dtype),
        indexing="ij")
    grid = torch.stack((grid_x, grid_y), 2)
    grid_flow = grid + flow
    grid_flow_x = 2.0 * grid_flow[:, :, :, 0] / max(w - 1, 1) - 1.0
    grid_flow_y = 2.0 * grid_flow[:, :, :, 1] / max(h - 1, 1) - 1.0
    grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=3).type(x.type())
    return F.grid_sample(x, grid_flow, mode=interpolation,
                         padding_mode=padding_mode, align_corners=align_corners)


def modulated_deform_conv2d(input: torch.Tensor, offset: torch.Tensor,
                            mask: torch.Tensor, weight: torch.Tensor,
                            bias: Optional[torch.Tensor], stride, padding,
                            dilation, groups: int, deform_groups: int
                            ) -> torch.Tensor:
    """Pure PyTorch modulated deformable convolution (DCN v2).

    Layouts follow mmcv: ``offset`` has ``2 * deform_groups * kh * kw``
    channels ordered as ``[dg][kh][kw][(y, x)]``; ``mask`` has
    ``deform_groups * kh * kw`` channels.  Sampling is vectorised over the
    deform groups by folding them into the batch dimension, so one
    ``grid_sample`` call covers all groups for each kernel position.
    """
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)

    out_c, in_c_per_group, kh, kw = weight.shape
    in_c = in_c_per_group * groups                       # total input channels
    c_per_dg = in_c // deform_groups

    n, _, in_h, in_w = input.shape
    out_h = (in_h + 2 * padding[0] - dilation[0] * (kh - 1) - 1) // stride[0] + 1
    out_w = (in_w + 2 * padding[1] - dilation[1] * (kw - 1) - 1) // stride[1] + 1

    if padding[0] or padding[1]:
        input = F.pad(input, (padding[1], padding[1], padding[0], padding[0]))
    in_h_p, in_w_p = input.shape[2], input.shape[3]

    # (n, dg*2*kh*kw, oh, ow) -> (n, oh, ow, dg, kh, kw, 2)
    off = offset.reshape(n, deform_groups, 2, kh, kw, out_h, out_w)
    off = off.permute(0, 5, 6, 1, 3, 4, 2).contiguous()
    # mask: (n, dg*kh*kw, oh, ow) -> (n, oh, ow, dg, kh, kw)
    msk = mask.reshape(n, deform_groups, kh, kw, out_h, out_w)
    msk = msk.permute(0, 4, 5, 1, 2, 3).contiguous()

    # Fold deform groups into the batch dimension: (n*dg, c/dg, H, W)
    inp_g = input.reshape(n * deform_groups, c_per_dg, in_h_p, in_w_p)

    base_y = (torch.arange(out_h, device=input.device, dtype=input.dtype)
              .view(1, out_h, 1) * stride[0])
    base_x = (torch.arange(out_w, device=input.device, dtype=input.dtype)
              .view(1, 1, out_w) * stride[1])

    cols: List[torch.Tensor] = []
    for kh_i in range(kh):
        for kw_i in range(kw):
            # last dim order is (y, x) as in mmcv
            dy = off[..., kh_i, kw_i, 0]                  # n,oh,ow,dg
            dx = off[..., kh_i, kw_i, 1]                  # n,oh,ow,dg
            m = msk[..., kh_i, kw_i].unsqueeze(1)         # n,1,oh,ow,dg

            gy = base_y.unsqueeze(-1) + kh_i * dilation[0] + dy      # n,oh,ow,dg
            gx = base_x.unsqueeze(-1) + kw_i * dilation[1] + dx
            gy = 2.0 * gy / max(in_h_p - 1, 1) - 1.0
            gx = 2.0 * gx / max(in_w_p - 1, 1) - 1.0
            # (n, oh, ow, dg, 2) -> (n*dg, oh, ow, 2)
            grid = torch.stack((gx, gy), dim=-1).permute(0, 3, 1, 2, 4).reshape(
                n * deform_groups, out_h, out_w, 2)

            sampled = F.grid_sample(inp_g, grid, mode="bilinear",
                                    padding_mode="zeros", align_corners=True)
            sampled = sampled.view(n, deform_groups, c_per_dg, out_h, out_w)
            sampled = sampled * m.permute(0, 4, 1, 2, 3)   # n,dg,1,oh,ow
            cols.append(sampled.reshape(n, in_c, out_h, out_w))

    # (n, kh*kw, c, oh, ow) -> (n, kh*kw, c, oh*ow)
    col = torch.stack(cols, dim=1).reshape(n, kh * kw, in_c, out_h * out_w)

    out_c_per_group = out_c // groups
    c_per_group = in_c * kh * kw // groups
    out = torch.zeros(n, out_c, out_h, out_w, dtype=input.dtype,
                      device=input.device)
    col = col.reshape(n, kh * kw * in_c, out_h * out_w)
    for gi in range(groups):
        w = weight[gi * out_c_per_group:(gi + 1) * out_c_per_group].reshape(
            out_c_per_group, c_per_group)
        c = col[:, gi * c_per_group:(gi + 1) * c_per_group]
        o = torch.einsum("oc,ncp->nop", w, c)
        out[:, gi * out_c_per_group:(gi + 1) * out_c_per_group] = o.reshape(
            n, out_c_per_group, out_h, out_w)
    if bias is not None:
        out = out + bias.view(1, -1, 1, 1)
    return out


def _pair(x):
    if isinstance(x, (tuple, list)):
        return tuple(x)
    return (x, x)


class ModulatedDeformConv2dPure(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size,
                 stride=1, padding=0, dilation=1, groups: int = 1,
                 deform_groups: int = 1, bias: bool = True) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deform_groups = deform_groups
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_channels // self.groups * self.kernel_size[0] * \
                     self.kernel_size[1]
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor, offset: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        return modulated_deform_conv2d(
            x, offset, mask, self.weight, self.bias, self.stride,
            self.padding, self.dilation, self.groups, self.deform_groups)


class PixelShufflePack(nn.Module):
    """Upsample by PixelShuffle, preceded by a conv (MMagic version)."""

    def __init__(self, in_channels: int, out_channels: int, scale_factor: int,
                 upsample_kernel: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        self.upsample_kernel = upsample_kernel
        self.upsample_conv = nn.Conv2d(
            self.in_channels, self.out_channels * scale_factor * scale_factor,
            self.upsample_kernel, padding=(self.upsample_kernel - 1) // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample_conv(x)
        return F.pixel_shuffle(x, self.scale_factor)


def make_layer(basic_block, num_basic_block: int, **kwarg) -> nn.Sequential:
    return nn.Sequential(*[basic_block(**kwarg) for _ in range(num_basic_block)])


class ResidualBlockNoBN(nn.Module):
    def __init__(self, mid_channels: int = 64) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.conv1(x))
        return self.conv2(out) + x


class ResidualBlocksWithInputConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 64,
                 num_blocks: int = 30) -> None:
        super().__init__()
        main: List[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            make_layer(ResidualBlockNoBN, num_blocks, mid_channels=out_channels),
        ]
        self.main = nn.Sequential(*main)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.main(feat)


# ---------------------------------------------------------------------------
# SPyNet
# ---------------------------------------------------------------------------
class SPyNetBasicModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.basic_module = nn.Sequential(
            nn.Conv2d(8, 32, 7, 1, 3), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 7, 1, 3), nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 7, 1, 3), nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 7, 1, 3), nn.ReLU(inplace=True),
            nn.Conv2d(16, 2, 7, 1, 3))

    def forward(self, tensor_input: torch.Tensor) -> torch.Tensor:
        return self.basic_module(tensor_input)


class SPyNet(nn.Module):
    def __init__(self, pretrained: Optional[str] = None) -> None:
        super().__init__()
        self.basic_module = nn.ModuleList(
            [SPyNetBasicModule() for _ in range(6)])
        if pretrained:
            state = torch.load(pretrained, map_location="cpu", weights_only=False)
            if "state_dict" in state:
                state = state["state_dict"]
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
            self.load_state_dict(state, strict=False)
        self.register_buffer("mean", torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def compute_flow(self, ref: torch.Tensor, supp: torch.Tensor) -> torch.Tensor:
        n, _, h, w = ref.size()
        ref_pyr = [(ref - self.mean) / self.std]
        supp_pyr = [(supp - self.mean) / self.std]
        for _ in range(5):
            ref_pyr.append(F.avg_pool2d(ref_pyr[-1], kernel_size=2, stride=2,
                                        count_include_pad=False))
            supp_pyr.append(F.avg_pool2d(supp_pyr[-1], kernel_size=2, stride=2,
                                         count_include_pad=False))
        ref_pyr = ref_pyr[::-1]
        supp_pyr = supp_pyr[::-1]

        flow = ref_pyr[0].new_zeros(n, 2, h // 32, w // 32)
        for level in range(len(ref_pyr)):
            if level == 0:
                flow_up = flow
            else:
                flow_up = F.interpolate(flow, scale_factor=2, mode="bilinear",
                                        align_corners=True) * 2.0
            flow = flow_up + self.basic_module[level](torch.cat([
                ref_pyr[level],
                flow_warp(supp_pyr[level], flow_up.permute(0, 2, 3, 1),
                          padding_mode="border"),
                flow_up], 1))
        return flow

    def forward(self, ref: torch.Tensor, supp: torch.Tensor) -> torch.Tensor:
        h, w = ref.shape[2:4]
        w_up = w if (w % 32) == 0 else 32 * (w // 32 + 1)
        h_up = h if (h % 32) == 0 else 32 * (h // 32 + 1)
        ref = F.interpolate(ref, size=(h_up, w_up), mode="bilinear", align_corners=False)
        supp = F.interpolate(supp, size=(h_up, w_up), mode="bilinear", align_corners=False)
        flow = F.interpolate(self.compute_flow(ref, supp), size=(h, w),
                             mode="bilinear", align_corners=False)
        flow[:, 0, :, :] *= float(w) / float(w_up)
        flow[:, 1, :, :] *= float(h) / float(h_up)
        return flow


# ---------------------------------------------------------------------------
# Second-order deformable alignment
# ---------------------------------------------------------------------------
class SecondOrderDeformableAlignment(ModulatedDeformConv2dPure):
    def __init__(self, *args, max_residue_magnitude: int = 10, **kwargs) -> None:
        self.max_residue_magnitude = max_residue_magnitude
        super().__init__(*args, **kwargs)
        self.conv_offset = nn.Sequential(
            nn.Conv2d(3 * self.out_channels + 4, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 27 * self.deform_groups, 3, 1, 1))
        self.init_offset()

    def init_offset(self) -> None:
        nn.init.constant_(self.conv_offset[-1].weight, 0)
        nn.init.constant_(self.conv_offset[-1].bias, 0)

    def forward(self, x: torch.Tensor, extra_feat: torch.Tensor,
                flow_1: torch.Tensor, flow_2: torch.Tensor) -> torch.Tensor:
        extra_feat = torch.cat([extra_feat, flow_1, flow_2], dim=1)
        out = self.conv_offset(extra_feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim=1))
        offset_1, offset_2 = torch.chunk(offset, 2, dim=1)
        offset_1 = offset_1 + flow_1.flip(1).repeat(1, offset_1.size(1) // 2, 1, 1)
        offset_2 = offset_2 + flow_2.flip(1).repeat(1, offset_2.size(1) // 2, 1, 1)
        offset = torch.cat([offset_1, offset_2], dim=1)
        mask = torch.sigmoid(mask)
        return modulated_deform_conv2d(
            x, offset, mask, self.weight, self.bias, self.stride, self.padding,
            self.dilation, self.groups, self.deform_groups)


# ---------------------------------------------------------------------------
# BasicVSR++
# ---------------------------------------------------------------------------
class BasicVSRPlusPlusNet(nn.Module):
    def __init__(self, mid_channels: int = 64, num_blocks: int = 7,
                 max_residue_magnitude: int = 10, is_low_res_input: bool = True,
                 spynet_pretrained: Optional[str] = None,
                 cpu_cache_length: int = 100) -> None:
        super().__init__()
        self.mid_channels = mid_channels
        self.is_low_res_input = is_low_res_input
        self.cpu_cache_length = cpu_cache_length
        self.cpu_cache = False

        self.spynet = SPyNet(pretrained=spynet_pretrained)

        if is_low_res_input:
            self.feat_extract = ResidualBlocksWithInputConv(3, mid_channels, 5)
        else:
            self.feat_extract = nn.Sequential(
                nn.Conv2d(3, mid_channels, 3, 2, 1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv2d(mid_channels, mid_channels, 3, 2, 1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                ResidualBlocksWithInputConv(mid_channels, mid_channels, 5))

        self.deform_align = nn.ModuleDict()
        self.backbone = nn.ModuleDict()
        modules = ["backward_1", "forward_1", "backward_2", "forward_2"]
        for i, module in enumerate(modules):
            self.deform_align[module] = SecondOrderDeformableAlignment(
                2 * mid_channels, mid_channels, 3, padding=1, deform_groups=16,
                max_residue_magnitude=max_residue_magnitude)
            self.backbone[module] = ResidualBlocksWithInputConv(
                (2 + i) * mid_channels, mid_channels, num_blocks)

        self.reconstruction = ResidualBlocksWithInputConv(
            5 * mid_channels, mid_channels, 5)
        self.upsample1 = PixelShufflePack(mid_channels, mid_channels, 2, upsample_kernel=3)
        self.upsample2 = PixelShufflePack(mid_channels, 64, 2, upsample_kernel=3)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.img_upsample = nn.Upsample(scale_factor=4, mode="bilinear",
                                        align_corners=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    # -- helpers ---------------------------------------------------------
    def check_if_mirror_extended(self, lqs: torch.Tensor) -> None:
        self.is_mirror_extended = False
        if lqs.size(1) % 2 == 0:
            lqs_1, lqs_2 = torch.chunk(lqs, 2, dim=1)
            if torch.norm(lqs_1 - lqs_2.flip(1)) == 0:
                self.is_mirror_extended = True

    def compute_flow(self, lqs: torch.Tensor):
        n, t, c, h, w = lqs.size()
        lqs_1 = lqs[:, :-1].reshape(-1, c, h, w)
        lqs_2 = lqs[:, 1:].reshape(-1, c, h, w)
        flows_backward = self.spynet(lqs_1, lqs_2).view(n, t - 1, 2, h, w)
        if self.is_mirror_extended:
            flows_forward = None
        else:
            flows_forward = self.spynet(lqs_2, lqs_1).view(n, t - 1, 2, h, w)
        if self.cpu_cache:
            flows_backward = flows_backward.cpu()
            flows_forward = flows_forward.cpu()
        return flows_forward, flows_backward

    def propagate(self, feats: Dict[str, list], flows: torch.Tensor,
                  module_name: str) -> Dict[str, list]:
        n, t, _, h, w = flows.size()
        frame_idx = list(range(0, t + 1))
        flow_idx = list(range(-1, t))
        mapping_idx = list(range(0, len(feats["spatial"])))
        mapping_idx += mapping_idx[::-1]
        if "backward" in module_name:
            frame_idx = frame_idx[::-1]
            flow_idx = frame_idx

        feat_prop = flows.new_zeros(n, self.mid_channels, h, w)
        for i, idx in enumerate(frame_idx):
            feat_current = feats["spatial"][mapping_idx[idx] % len(mapping_idx)]
            if i > 0:
                flow_n1 = flows[:, flow_idx[i] % t, :, :, :]
                cond_n1 = flow_warp(feat_prop, flow_n1.permute(0, 2, 3, 1))
                feat_n2 = torch.zeros_like(feat_prop)
                flow_n2 = torch.zeros_like(flow_n1)
                cond_n2 = torch.zeros_like(cond_n1)
                if i > 1:
                    feat_n2 = feats[module_name][-2]
                    flow_n2 = flows[:, flow_idx[i - 1] % t, :, :, :]
                    flow_n2 = flow_n1 + flow_warp(flow_n2,
                                                  flow_n1.permute(0, 2, 3, 1))
                    cond_n2 = flow_warp(feat_n2, flow_n2.permute(0, 2, 3, 1))
                cond = torch.cat([cond_n1, feat_current, cond_n2], dim=1)
                feat_prop = torch.cat([feat_prop, feat_n2], dim=1)
                feat_prop = self.deform_align[module_name](feat_prop, cond,
                                                           flow_n1, flow_n2)

            feat = [feat_current] + [
                feats[k][idx] for k in feats if k not in ("spatial", module_name)
            ] + [feat_prop]
            feat = torch.cat(feat, dim=1)
            feat_prop = feat_prop + self.backbone[module_name](feat)
            feats[module_name].append(feat_prop)

            if self.cpu_cache:
                feats[module_name][-1] = feats[module_name][-1].cpu()
                torch.cuda.empty_cache()

        if "backward" in module_name:
            feats[module_name] = feats[module_name][::-1]
        return feats

    def upsample(self, lqs: torch.Tensor, feats: Dict[str, list]) -> torch.Tensor:
        outputs = []
        num_outputs = len(feats["spatial"])
        mapping_idx = list(range(0, num_outputs))
        mapping_idx += mapping_idx[::-1]
        for i in range(0, lqs.size(1)):
            hr = [feats[k].pop(0) for k in feats if k != "spatial"]
            hr.insert(0, feats["spatial"][mapping_idx[i] % len(mapping_idx)])
            hr = torch.cat(hr, dim=1)
            if self.cpu_cache:
                hr = hr.cuda()
            hr = self.reconstruction(hr)
            hr = self.lrelu(self.upsample1(hr))
            hr = self.lrelu(self.upsample2(hr))
            hr = self.lrelu(self.conv_hr(hr))
            hr = self.conv_last(hr)
            if self.is_low_res_input:
                hr += self.img_upsample(lqs[:, i, :, :, :])
            else:
                hr += lqs[:, i, :, :, :]
            if self.cpu_cache:
                hr = hr.cpu()
                torch.cuda.empty_cache()
            outputs.append(hr)
        return torch.stack(outputs, dim=1)

    def forward(self, lqs: torch.Tensor) -> torch.Tensor:
        n, t, c, h, w = lqs.size()
        if t > self.cpu_cache_length and lqs.is_cuda:
            self.cpu_cache = True
        else:
            self.cpu_cache = False

        if self.is_low_res_input:
            lqs_downsample = lqs.clone()
        else:
            lqs_downsample = F.interpolate(
                lqs.view(-1, c, h, w), scale_factor=0.25, mode="bicubic").view(
                n, t, c, h // 4, w // 4)

        self.check_if_mirror_extended(lqs)

        feats: Dict[str, list] = {}
        if self.cpu_cache:
            feats["spatial"] = []
            for i in range(t):
                feats["spatial"].append(self.feat_extract(lqs[:, i]).cpu())
                torch.cuda.empty_cache()
        else:
            feats_ = self.feat_extract(lqs.view(-1, c, h, w))
            h_f, w_f = feats_.shape[2:]
            feats_ = feats_.view(n, t, -1, h_f, w_f)
            feats["spatial"] = [feats_[:, i] for i in range(t)]

        assert lqs_downsample.size(3) >= 64 and lqs_downsample.size(4) >= 64, (
            "BasicVSR++ needs at least 64x64 low-res inputs, got "
            f"{lqs_downsample.size(3)}x{lqs_downsample.size(4)}")

        flows_forward, flows_backward = self.compute_flow(lqs_downsample)

        for _ in [1, 2]:
            for direction in ["backward", "forward"]:
                module = f"{direction}_{_}"
                feats[module] = []
                if direction == "backward":
                    flows = flows_backward
                elif flows_forward is not None:
                    flows = flows_forward
                else:
                    flows = flows_backward.flip(1)
                feats = self.propagate(feats, flows, module)
                if self.cpu_cache:
                    del flows
                    torch.cuda.empty_cache()
        return self.upsample(lqs, feats)


def load_basicvsr_pp(weights_path, spynet_path: Optional[str] = None,
                     device="cpu") -> BasicVSRPlusPlusNet:
    net = BasicVSRPlusPlusNet(spynet_pretrained=spynet_path)
    state = torch.load(weights_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    clean = {}
    for k, v in state.items():
        if k == "step_counter":
            continue
        if k.startswith("generator."):
            k = k[len("generator."):]
        # SPyNet weights live inside the checkpoint when pretrained is given
        if k.startswith("spynet.") and spynet_path:
            continue
        clean[k] = v
    missing, unexpected = net.load_state_dict(clean, strict=False)
    real_missing = [m for m in missing
                    if not m.startswith("spynet.")]
    if real_missing:
        raise RuntimeError(
            f"BasicVSR++ checkpoint mismatch: {len(real_missing)} missing keys "
            f"(e.g. {real_missing[:3]})")
    net.eval()
    return net.to(device)
