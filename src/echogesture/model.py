"""EchoGesture classifier: depthwise-separable temporal CNN + light self-attention.

~2-3 M parameters for 2000 classes -> ~3 MB after INT8 quantisation. Every op
(Conv1d, BatchNorm, Linear, Softmax, LayerNorm) maps cleanly to ONNX / TFLite
and quantises well.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Inverted-residual temporal block (MobileNetV2 style, 1D)."""

    def __init__(self, d: int, k: int = 5, dil: int = 1, expand: int = 2, drop: float = 0.1):
        super().__init__()
        h = d * expand
        self.pw1 = nn.Conv1d(d, h, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(h)
        self.dw = nn.Conv1d(h, h, k, padding=dil * (k - 1) // 2, dilation=dil, groups=h, bias=False)
        self.bn2 = nn.BatchNorm1d(h)
        self.pw2 = nn.Conv1d(h, d, 1, bias=False)
        self.bn3 = nn.BatchNorm1d(d)
        self.drop = nn.Dropout(drop)

    def forward(self, x):                      # x: (B, D, T)
        y = F.silu(self.bn1(self.pw1(x)))
        y = F.silu(self.bn2(self.dw(y)))
        y = self.bn3(self.pw2(y))
        return x + self.drop(y)


class AttnBlock(nn.Module):
    """Pre-LN transformer block written with plain ops (export-friendly)."""

    def __init__(self, d: int, heads: int = 4, ff: int = 2, drop: float = 0.1):
        super().__init__()
        self.h, self.dh = heads, d // heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, d * ff), nn.SiLU(), nn.Linear(d * ff, d))
        self.drop = nn.Dropout(drop)

    def forward(self, x):                      # x: (B, D, T)
        x = x.transpose(1, 2)                  # (B, T, D)
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).reshape(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        att = torch.softmax((q @ k.transpose(-1, -2)) / math.sqrt(self.dh), dim=-1)
        y = (att @ v).transpose(1, 2).reshape(B, T, D)
        x = x + self.drop(self.proj(y))
        x = x + self.drop(self.ff(self.ln2(x)))
        return x.transpose(1, 2)


class EchoGestureNet(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, d: int = 192, seq_len: int = 48,
                 drop: float = 0.1, head_drop: float = 0.3):
        super().__init__()
        self.cfg = dict(in_dim=in_dim, n_classes=n_classes, d=d, seq_len=seq_len,
                        drop=drop, head_drop=head_drop)
        self.stem = nn.Linear(in_dim, d, bias=False)
        self.stem_bn = nn.BatchNorm1d(d)
        self.pos = nn.Parameter(torch.zeros(1, d, seq_len))
        self.blocks = nn.Sequential(
            ConvBlock(d, 5, 1, drop=drop), ConvBlock(d, 5, 2, drop=drop), ConvBlock(d, 5, 4, drop=drop),
            AttnBlock(d, drop=drop),
            ConvBlock(d, 5, 1, drop=drop), ConvBlock(d, 5, 2, drop=drop), ConvBlock(d, 5, 4, drop=drop),
            AttnBlock(d, drop=drop),
        )
        self.head_ln = nn.LayerNorm(2 * d)
        self.head_drop = nn.Dropout(head_drop)
        self.head = nn.Linear(2 * d, n_classes)

    def forward(self, x):                      # x: (B, T, F)
        x = self.stem(x).transpose(1, 2)       # (B, D, T)
        x = F.silu(self.stem_bn(x)) + self.pos
        x = self.blocks(x)
        x = torch.cat([x.mean(dim=2), x.amax(dim=2)], dim=1)
        return self.head(self.head_drop(self.head_ln(x)))


def build_model(cfg: dict) -> EchoGestureNet:
    return EchoGestureNet(**cfg)
