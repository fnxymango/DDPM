"""Small U-Net for 28x28 single-channel images (28 -> 14 -> 7)."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, tdim, dropout=0.1):
        super().__init__()
        self.n1 = nn.GroupNorm(8, cin)
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.t = nn.Linear(tdim, cout)
        self.n2 = nn.GroupNorm(8, cout)
        self.drop = nn.Dropout(dropout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, temb):
        h = self.c1(F.silu(self.n1(x)))
        h = h + self.t(F.silu(temb))[:, :, None, None]
        h = self.c2(self.drop(F.silu(self.n2(h))))
        return h + self.skip(x)


class Attn(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.norm = nn.GroupNorm(8, c)
        self.qkv = nn.Conv2d(c, 3 * c, 1)
        self.proj = nn.Conv2d(c, c, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, c, h * w).transpose(-1, -2).unbind(1)
        out = F.scaled_dot_product_attention(q, k, v)
        return x + self.proj(out.transpose(-1, -2).reshape(b, c, h, w))


class Upsample(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class UNet(nn.Module):
    def __init__(self, in_ch=1, base=64, mults=(1, 2, 4), tdim=256):
        super().__init__()
        self.base = base
        self.temb = nn.Sequential(nn.Linear(base, tdim), nn.SiLU(), nn.Linear(tdim, tdim))
        self.inc = nn.Conv2d(in_ch, base, 3, padding=1)
        chs = [base * m for m in mults]
        self.down, skip_chs, prev = nn.ModuleList(), [], base
        for i, c in enumerate(chs):
            ds = nn.Conv2d(c, c, 3, stride=2, padding=1) if i < len(chs) - 1 else nn.Identity()
            self.down.append(nn.ModuleList([ResBlock(prev, c, tdim), ResBlock(c, c, tdim), ds]))
            skip_chs.append(c)
            prev = c
        self.mid1, self.attn, self.mid2 = ResBlock(prev, prev, tdim), Attn(prev), ResBlock(prev, prev, tdim)
        self.up = nn.ModuleList()
        for i in reversed(range(len(chs))):
            c = chs[i]
            us = Upsample(c) if i > 0 else nn.Identity()
            self.up.append(nn.ModuleList([ResBlock(prev + skip_chs[i], c, tdim), ResBlock(c, c, tdim), us]))
            prev = c
        self.out = nn.Sequential(nn.GroupNorm(8, prev), nn.SiLU(), nn.Conv2d(prev, in_ch, 3, padding=1))

    def forward(self, x, t):
        temb = self.temb(timestep_embedding(t, self.base))
        h, skips = self.inc(x), []
        for r1, r2, ds in self.down:
            h = r2(r1(h, temb), temb)
            skips.append(h)
            h = ds(h)
        h = self.mid2(self.attn(self.mid1(h, temb)), temb)
        for r1, r2, us in self.up:
            h = torch.cat([h, skips.pop()], dim=1)
            h = us(r2(r1(h, temb), temb))
        return self.out(h)
