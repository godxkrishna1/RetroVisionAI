"""StyleGAN2 generator (clean version, no CUDA extensions).

Port of ``gfpgan/archs/stylegan2_clean_arch.py`` from TencentARC/GFPGAN
(Apache-2.0).  Used as the decoder of GFPGAN - required to load
``GFPGANv1.4.pth`` with strict=True.
"""
from __future__ import annotations

import math
import random
from typing import List, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


def default_init_weights(module_list, scale: float = 1, bias_fill: float = 0,
                         **kwargs) -> None:
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


class NormStyleCode(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)


class ModulatedConv2d(nn.Module):
    """Modulated convolution used in StyleGAN2 (no bias)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 num_style_feat: int, demodulate: bool = True,
                 sample_mode: Optional[str] = None, eps: float = 1e-8) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.demodulate = demodulate
        self.sample_mode = sample_mode
        self.eps = eps
        self.modulation = nn.Linear(num_style_feat, in_channels, bias=True)
        default_init_weights(self.modulation, scale=1, bias_fill=1, a=0,
                             mode="fan_in", nonlinearity="linear")
        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, kernel_size, kernel_size) /
            math.sqrt(in_channels * kernel_size ** 2))
        self.padding = kernel_size // 2

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        style = self.modulation(style).view(b, 1, c, 1, 1)
        weight = self.weight * style
        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + self.eps)
            weight = weight * demod.view(b, self.out_channels, 1, 1, 1)
        weight = weight.view(b * self.out_channels, c, self.kernel_size,
                             self.kernel_size)
        if self.sample_mode == "upsample":
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        elif self.sample_mode == "downsample":
            x = F.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=False)
        b, c, h, w = x.shape
        x = x.view(1, b * c, h, w)
        out = F.conv2d(x, weight, padding=self.padding, groups=b)
        return out.view(b, self.out_channels, *out.shape[2:4])


class StyleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 num_style_feat: int, demodulate: bool = True,
                 sample_mode: Optional[str] = None) -> None:
        super().__init__()
        self.modulated_conv = ModulatedConv2d(
            in_channels, out_channels, kernel_size, num_style_feat,
            demodulate=demodulate, sample_mode=sample_mode)
        self.weight = nn.Parameter(torch.zeros(1))     # noise injection
        self.bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
        self.activate = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor, style: torch.Tensor,
                noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        out = self.modulated_conv(x, style) * 2 ** 0.5
        if noise is None:
            b, _, h, w = out.shape
            noise = out.new_empty(b, 1, h, w).normal_()
        out = out + self.weight * noise
        out = out + self.bias
        return self.activate(out)


class ToRGB(nn.Module):
    def __init__(self, in_channels: int, num_style_feat: int,
                 upsample: bool = True) -> None:
        super().__init__()
        self.upsample = upsample
        self.modulated_conv = ModulatedConv2d(
            in_channels, 3, kernel_size=1, num_style_feat=num_style_feat,
            demodulate=False, sample_mode=None)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, x: torch.Tensor, style: torch.Tensor,
                skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        out = self.modulated_conv(x, style)
        out = out + self.bias
        if skip is not None:
            if self.upsample:
                skip = F.interpolate(skip, scale_factor=2, mode="bilinear",
                                     align_corners=False)
            out = out + skip
        return out


class ConstantInput(nn.Module):
    def __init__(self, num_channel: int, size: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1, num_channel, size, size))

    def forward(self, batch: int) -> torch.Tensor:
        return self.weight.repeat(batch, 1, 1, 1)


class StyleGAN2GeneratorClean(nn.Module):
    """Clean StyleGAN2 generator."""

    def __init__(self, out_size: int, num_style_feat: int = 512, num_mlp: int = 8,
                 channel_multiplier: int = 2, narrow: float = 1.0) -> None:
        super().__init__()
        self.num_style_feat = num_style_feat
        style_mlp_layers: List[nn.Module] = [NormStyleCode()]
        for _ in range(num_mlp):
            style_mlp_layers.extend([
                nn.Linear(num_style_feat, num_style_feat, bias=True),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)])
        self.style_mlp = nn.Sequential(*style_mlp_layers)
        default_init_weights(self.style_mlp, scale=1, bias_fill=0, a=0.2,
                             mode="fan_in", nonlinearity="leaky_relu")

        channels = {
            "4": int(512 * narrow), "8": int(512 * narrow),
            "16": int(512 * narrow), "32": int(512 * narrow),
            "64": int(256 * channel_multiplier * narrow),
            "128": int(128 * channel_multiplier * narrow),
            "256": int(64 * channel_multiplier * narrow),
            "512": int(32 * channel_multiplier * narrow),
            "1024": int(16 * channel_multiplier * narrow),
        }
        self.channels = channels
        self.constant_input = ConstantInput(channels["4"], size=4)
        self.style_conv1 = StyleConv(channels["4"], channels["4"], 3,
                                     num_style_feat, demodulate=True,
                                     sample_mode=None)
        self.to_rgb1 = ToRGB(channels["4"], num_style_feat, upsample=False)

        self.log_size = int(math.log(out_size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1
        self.num_latent = self.log_size * 2 - 2

        self.style_convs = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()

        in_channels = channels["4"]
        for layer_idx in range(self.num_layers):
            resolution = 2 ** ((layer_idx + 5) // 2)
            self.noises.register_buffer(f"noise{layer_idx}",
                                        torch.randn(1, 1, resolution, resolution))
        for i in range(3, self.log_size + 1):
            out_channels = channels[f"{2 ** i}"]
            self.style_convs.append(StyleConv(in_channels, out_channels, 3,
                                              num_style_feat, demodulate=True,
                                              sample_mode="upsample"))
            self.style_convs.append(StyleConv(out_channels, out_channels, 3,
                                              num_style_feat, demodulate=True,
                                              sample_mode=None))
            self.to_rgbs.append(ToRGB(out_channels, num_style_feat, upsample=True))
            in_channels = out_channels

    def forward(self, styles, input_is_latent: bool = False, noise=None,
                randomize_noise: bool = True, truncation: float = 1,
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
            if inject_index is None:
                inject_index = random.randint(1, self.num_latent - 1)
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
            out = conv2(out, latent[:, i + 1], noise=noise2)
            skip = to_rgb(out, latent[:, i + 2], skip)
            i += 2
        image = skip
        return (image, latent) if return_latents else (image, None)
