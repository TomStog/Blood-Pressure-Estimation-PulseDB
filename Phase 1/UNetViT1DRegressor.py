import math
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


# -----------------------
# Building blocks (1-D)
# -----------------------

class ConvBlock1d(nn.Module):
    """Single conv with BN + ReLU (simplified from double conv)."""
    def __init__(self, in_ch, out_ch, k=3, dropout=0.0):
        super().__init__()
        p = k // 2
        layers = [
            nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=1, padding=p, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):  # [B, C, L]
        return self.net(x)


class DownsampleBlock1d(nn.Module):
    """Downsample: single conv -> stride-2 max pool."""
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.conv = ConvBlock1d(in_ch, out_ch, dropout=dropout)
        self.down = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):                # [B, Cin, L]
        y = self.conv(x)                 # [B, Cout, L]
        y = self.down(y)                 # [B, Cout, L/2]
        return y


class UpsampleBlock1d(nn.Module):
    """Upsample: interpolate (x2) -> concat skip -> single conv."""
    def __init__(self, in_ch, skip_ch, out_ch, dropout=0.0):
        super().__init__()
        self.fuse = ConvBlock1d(in_ch + skip_ch, out_ch, dropout=dropout)

    def forward(self, x, skip):          # x: [B, Cin, L], skip: [B, Cskip, 2L]
        x = F.interpolate(x, scale_factor=2, mode='linear', align_corners=False)
        # Handle size mismatch
        diff = skip.shape[-1] - x.shape[-1]
        if diff > 0:
            x = F.pad(x, (0, diff))
        elif diff < 0:
            x = x[..., :skip.shape[-1]]
        x = torch.cat([x, skip], dim=1)  # [B, Cin+Cskip, 2L]
        return self.fuse(x)              # [B, Cout, 2L]


# -----------------------
# Simplified ViT bottleneck
# -----------------------

class SinePositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for 1-D sequences."""
    def __init__(self, d_model: int, max_len: int = 5_000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) *
                             (-math.log(10_000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):  # x: [B, T, C]
        T = x.size(1)
        return x + self.pe[:T, :].unsqueeze(0)


class ViTBottleneck1d(nn.Module):
    """Simplified ViT: fewer layers, smaller d_model."""
    def __init__(self, in_ch: int, d_model: int = 128, nhead: int = 4, depth: int = 2, 
                 mlp_ratio: float = 2.0, dropout: float = 0.1):
        super().__init__()
        self.to_tokens = nn.Linear(in_ch, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=int(d_model * mlp_ratio),
            dropout=dropout, batch_first=True, activation="relu", norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.pos = SinePositionalEncoding(d_model)
        self.to_features = nn.Linear(d_model, in_ch)

    def forward(self, x):  # x: [B, C, L]
        x = x.transpose(1, 2)             # [B, L, C]
        tokens = self.to_tokens(x)        # [B, L, d_model]
        tokens = self.pos(tokens)
        tokens = self.encoder(tokens)
        feats = self.to_features(tokens)  # [B, L, C]
        return feats.transpose(1, 2)      # [B, C, L]


# -----------------------
# Simplified model
# -----------------------

class UNetViT1DRegressor(nn.Module):
    """
    Simplified UNet-ViT: 3 encoder levels + ViT bottleneck + 3 decoder levels.
    Reduced channels, single convs, and simpler ViT.
    Input: 6xL 1-D signals. Output: scalar regression value.
    """
    def __init__(self, in_ch=2, base_ch=16, vit_d_model=128, vit_heads=4, vit_depth=2, dropout=0.1):
        super().__init__()
        # Encoder (3 levels instead of 4)
        self.enc0 = ConvBlock1d(in_ch, base_ch)           # L
        self.down1 = DownsampleBlock1d(base_ch, base_ch * 2, dropout = 0.0)   # L/2
        self.down2 = DownsampleBlock1d(base_ch * 2, base_ch * 4, dropout = 0.0)  # L/4
        self.down3 = DownsampleBlock1d(base_ch * 4, base_ch * 8, dropout = 0.0)  # L/8

        # ViT bottleneck
        self.vit = ViTBottleneck1d(in_ch=base_ch * 8, d_model=vit_d_model,
                                   nhead=vit_heads, depth=vit_depth, dropout=dropout)

        # Decoder (3 levels)
        self.up2 = UpsampleBlock1d(in_ch=base_ch * 8,  skip_ch=base_ch * 4, out_ch=base_ch * 4, dropout = 0.0)  # L/4
        self.up1 = UpsampleBlock1d(in_ch=base_ch * 4,  skip_ch=base_ch * 2, out_ch=base_ch * 2, dropout = 0.0)  # L/2
        self.up0 = UpsampleBlock1d(in_ch=base_ch * 2,  skip_ch=base_ch,     out_ch=base_ch, dropout = 0.0)      # L

        # Simpler head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(base_ch, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):  # x: [B, 6, L]
        # Encoder
        e0 = self.enc0(x)           # [B, 1*base, L]
        e1 = self.down1(e0)         # [B, 2*base, L/2]
        e2 = self.down2(e1)         # [B, 4*base, L/4]
        e3 = self.down3(e2)         # [B, 8*base, L/8]

        # ViT bottleneck
        b  = self.vit(e3)           # [B, 8*base, L/8]

        # Decoder with skip connections
        d2 = self.up2(b,  e2)       # [B, 4*base, L/4]
        d1 = self.up1(d2, e1)       # [B, 2*base, L/2]
        d0 = self.up0(d1, e0)       # [B, 1*base, L]

        # Regression head
        y  = self.head(d0)          # [B, 1]
        return y