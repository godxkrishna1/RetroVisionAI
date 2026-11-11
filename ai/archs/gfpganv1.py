"""GFPGAN v1 (clean) generator architecture.

Port of ``gfpgan/archs/gfpganv1_clean_arch.py`` (TencentARC/GFPGAN, Apache-2.0)
with the BasicSR registry removed.  Loads ``GFPGANv1.4.pth`` (``params_ema``,
285 tensors) with strict=True.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .stylegan2_clean import StyleGAN2GeneratorClean


class StyleGAN2GeneratorCSFT(StyleGAN2GeneratorClean):
    """StyleGAN2 generator with SFT (spatial feature transform) modulation."""

    def __init__(self, out_size: int, num_style_feat: int = 512, num_mlp: int = 8,
                 channel_multiplier: int = 2, narrow: float = 1,
                 sft_half: bool = False) -> None:
        super().__init__(out_size, num_style_feat=num_style_feat, num_mlp=num_mlp,
                         channel_multiplier=channel_multiplier, narrow=narrow)
        self.sft_half = sft_half

    def forward(self, styles, conditions, input_is_latent: bool = False,
                noise=None, randomize_noise: bool = True, truncation: float = 1,
                truncation_latent=None, inject_index=None,
                return_latents: bool = False):
        if not input_is_latent:
            styles = [self.style_mlp(s) for s in styles]
        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [getattr(self.noises, f"noise{i}")
                         for i in range(self.num_layers)]
        if truncation < 1:
            styles = [truncation_latent + truncation * (s - truncation_latent)
                      for s in styles]
        if len(styles) == 1:
            inject_index = self.num_latent
            if styles[0].ndim < 3:
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            else:
                latent = styles[0]
        elif len(styles) == 2:
            import random as _random
            if inject_index is None:
                inject_index = _random.randint(1, self.num_latent - 1)
            latent1 = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(1, self.num_latent - inject_index, 1)
            latent = torch.cat([latent1, latent2], 1)
        else:  # pragma: no cover
            raise ValueError("styles must contain 1 or 2 tensors")

        out = self.constant_input(latent.shape[0])
        out = self.style_conv1(out, latent[:, 0], noise=noise[0])
        skip = self.to_rgb1(out, latent[:, 1])

        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(
                self.style_convs[::2], self.style_convs[1::2],
                noise[1::2], noise[2::2], self.to_rgbs):
            out = conv1(out, latent[:, i], noise=noise1)
            if i < len(conditions):
                if self.sft_half:
                    out_same, out_sft = torch.split(out, int(out.size(1) // 2), dim=1)
                    out_sft = out_sft * conditions[i - 1] + conditions[i]
                    out = torch.cat([out_same, out_sft], dim=1)
                else:
                    out = out * conditions[i - 1] + conditions[i]
            out = conv2(out, latent[:, i + 1], noise=noise2)
            skip = to_rgb(out, latent[:, i + 2], skip)
            i += 2
        image = skip
        return (image, latent) if return_latents else (image, None)


class ResBlock(nn.Module):
    """Residual block with bilinear up/down-sampling."""

    def __init__(self, in_channels: int, out_channels: int, mode: str = "down") -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.skip = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        if mode == "down":
            self.scale_factor = 0.5
        elif mode == "up":
            self.scale_factor = 2
        else:  # pragma: no cover
            raise ValueError("mode must be 'down' or 'up'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.leaky_relu_(self.conv1(x), negative_slope=0.2)
        out = F.interpolate(out, scale_factor=self.scale_factor, mode="bilinear",
                            align_corners=False)
        out = F.leaky_relu_(self.conv2(out), negative_slope=0.2)
        x = F.interpolate(x, scale_factor=self.scale_factor, mode="bilinear",
                          align_corners=False)
        return out + self.skip(x)


class GFPGANv1Clean(nn.Module):
    """UNet encoder + StyleGAN2 decoder with SFT for blind face restoration."""

    def __init__(self, out_size: int, num_style_feat: int = 512,
                 channel_multiplier: int = 1, decoder_load_path: Optional[str] = None,
                 fix_decoder: bool = True, num_mlp: int = 8,
                 input_is_latent: bool = False, different_w: bool = False,
                 narrow: float = 1, sft_half: bool = False) -> None:
        super().__init__()
        self.input_is_latent = input_is_latent
        self.different_w = different_w
        self.num_style_feat = num_style_feat

        unet_narrow = narrow * 0.5
        channels = {
            "4": int(512 * unet_narrow), "8": int(512 * unet_narrow),
            "16": int(512 * unet_narrow), "32": int(512 * unet_narrow),
            "64": int(256 * channel_multiplier * unet_narrow),
            "128": int(128 * channel_multiplier * unet_narrow),
            "256": int(64 * channel_multiplier * unet_narrow),
            "512": int(32 * channel_multiplier * unet_narrow),
            "1024": int(16 * channel_multiplier * unet_narrow),
        }
        self.log_size = int(math.log(out_size, 2))
        first_out_size = 2 ** int(math.log(out_size, 2))

        self.conv_body_first = nn.Conv2d(3, channels[f"{first_out_size}"], 1)
        in_channels = channels[f"{first_out_size}"]
        self.conv_body_down = nn.ModuleList()
        for i in range(self.log_size, 2, -1):
            out_channels = channels[f"{2 ** (i - 1)}"]
            self.conv_body_down.append(ResBlock(in_channels, out_channels, mode="down"))
            in_channels = out_channels
        self.final_conv = nn.Conv2d(in_channels, channels["4"], 3, 1, 1)

        in_channels = channels["4"]
        self.conv_body_up = nn.ModuleList()
        for i in range(3, self.log_size + 1):
            out_channels = channels[f"{2 ** i}"]
            self.conv_body_up.append(ResBlock(in_channels, out_channels, mode="up"))
            in_channels = out_channels

        self.toRGB = nn.ModuleList()
        for i in range(3, self.log_size + 1):
            self.toRGB.append(nn.Conv2d(channels[f"{2 ** i}"], 3, 1))

        if different_w:
            linear_out_channel = (int(math.log(out_size, 2)) * 2 - 2) * num_style_feat
        else:
            linear_out_channel = num_style_feat
        self.final_linear = nn.Linear(channels["4"] * 4 * 4, linear_out_channel)

        self.stylegan_decoder = StyleGAN2GeneratorCSFT(
            out_size=out_size, num_style_feat=num_style_feat, num_mlp=num_mlp,
            channel_multiplier=channel_multiplier, narrow=narrow,
            sft_half=sft_half)

        if decoder_load_path:
            self.stylegan_decoder.load_state_dict(
                torch.load(decoder_load_path,
                           map_location=lambda storage, loc: storage)["params_ema"])
        if fix_decoder:
            for _, param in self.stylegan_decoder.named_parameters():
                param.requires_grad = False

        self.condition_scale = nn.ModuleList()
        self.condition_shift = nn.ModuleList()
        for i in range(3, self.log_size + 1):
            out_channels = channels[f"{2 ** i}"]
            sft_out_channels = out_channels if sft_half else out_channels * 2
            self.condition_scale.append(nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, 1, 1),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(out_channels, sft_out_channels, 3, 1, 1)))
            self.condition_shift.append(nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, 1, 1),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(out_channels, sft_out_channels, 3, 1, 1)))

    def forward(self, x: torch.Tensor, return_latents: bool = False,
                return_rgb: bool = True, randomize_noise: bool = True, **kwargs):
        conditions: List[torch.Tensor] = []
        unet_skips: List[torch.Tensor] = []
        out_rgbs: List[torch.Tensor] = []

        feat = F.leaky_relu_(self.conv_body_first(x), negative_slope=0.2)
        for i in range(self.log_size - 2):
            feat = self.conv_body_down[i](feat)
            unet_skips.insert(0, feat)
        feat = F.leaky_relu_(self.final_conv(feat), negative_slope=0.2)

        style_code = self.final_linear(feat.view(feat.size(0), -1))
        if self.different_w:
            style_code = style_code.view(style_code.size(0), -1, self.num_style_feat)

        for i in range(self.log_size - 2):
            feat = feat + unet_skips[i]
            feat = self.conv_body_up[i](feat)
            conditions.append(self.condition_scale[i](feat).clone())
            conditions.append(self.condition_shift[i](feat).clone())
            if return_rgb:
                out_rgbs.append(self.toRGB[i](feat))

        image, _ = self.stylegan_decoder(
            [style_code], conditions, return_latents=return_latents,
            input_is_latent=self.input_is_latent, randomize_noise=randomize_noise)
        return image, out_rgbs
