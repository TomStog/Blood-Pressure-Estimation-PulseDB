import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# Small helpers (same as yours)
# -------------------------
def _best_gn_groups(C: int, max_groups: int = 8) -> int:
    for g in range(min(max_groups, C), 0, -1):
        if C % g == 0:
            return g
    return 1


class ConvGNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, d=1, act=True, dropout=0.0):
        super().__init__()
        pad = (k // 2) * d
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=s, dilation=d, padding=pad, bias=False)
        self.gn = nn.GroupNorm(_best_gn_groups(out_ch), out_ch)
        self.act = nn.SiLU() if act else nn.Identity()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.drop(self.act(self.gn(self.conv(x))))


class DepthwiseSeparableConv1D(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, d=1, dropout=0.0):
        super().__init__()
        pad = (k // 2) * d
        self.dw = nn.Conv1d(in_ch, in_ch, kernel_size=k, stride=s, dilation=d, padding=pad,
                            groups=in_ch, bias=False)
        self.pw = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)
        self.gn = nn.GroupNorm(_best_gn_groups(out_ch), out_ch)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.gn(x)
        x = self.act(x)
        x = self.drop(x)
        return x


# -------------------------
# Core blocks (same as yours)
# -------------------------
class ResDilateBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.proj = ConvGNAct(in_ch, out_ch, k=1, dropout=0.0)
        self.b1 = DepthwiseSeparableConv1D(out_ch, out_ch, k=3, d=1, dropout=dropout)
        self.b2 = DepthwiseSeparableConv1D(out_ch, out_ch, k=3, d=2, dropout=dropout)
        self.b4 = DepthwiseSeparableConv1D(out_ch, out_ch, k=3, d=4, dropout=dropout)
        self.fuse = ConvGNAct(out_ch * 3, out_ch, k=1, dropout=dropout)
        self.skip = nn.Identity() if in_ch == out_ch else ConvGNAct(in_ch, out_ch, k=1, act=False)

    def forward(self, x):
        res = self.skip(x)
        x = self.proj(x)
        y = torch.cat([self.b1(x), self.b2(x), self.b4(x)], dim=1)
        y = self.fuse(y)
        return y + res


class SkipAdapter1D(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.pw = ConvGNAct(in_ch, out_ch, k=1, dropout=0.0)
        self.dw = DepthwiseSeparableConv1D(out_ch, out_ch, k=7, d=1, dropout=dropout)

    def forward(self, x):
        return self.dw(self.pw(x))


class AttentionGate1D(nn.Module):
    def __init__(self, skip_ch, gate_ch, inter_ch=None):
        super().__init__()
        inter_ch = inter_ch or max(8, skip_ch // 2)
        self.Wx = nn.Conv1d(skip_ch, inter_ch, kernel_size=1, bias=False)
        self.Wg = nn.Conv1d(gate_ch, inter_ch, kernel_size=1, bias=False)
        self.psi = nn.Conv1d(inter_ch, 1, kernel_size=1, bias=True)
        self.act = nn.SiLU()

    def forward(self, x_skip, x_gate):
        Ls = x_skip.shape[-1]
        Lg = x_gate.shape[-1]
        if Lg != Ls:
            x_gate = F.interpolate(x_gate, size=Ls, mode="linear", align_corners=False)

        h = self.act(self.Wx(x_skip) + self.Wg(x_gate))
        a = torch.sigmoid(self.psi(h))
        return x_skip * a


class ScaleFusion1D(nn.Module):
    def __init__(self, ch, hidden=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(ch * 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2)
        )

    def forward(self, a, b):
        ga = a.mean(dim=-1)
        gb = b.mean(dim=-1)
        logits = self.mlp(torch.cat([ga, gb], dim=1))
        w = torch.softmax(logits, dim=1)
        w1 = w[:, 0].view(-1, 1, 1)
        w2 = w[:, 1].view(-1, 1, 1)
        return w1 * a + w2 * b


class ASPP1D(nn.Module):
    def __init__(self, ch, dropout=0.0, dilations=(1, 2, 4, 8)):
        super().__init__()
        self.branches = nn.ModuleList([
            DepthwiseSeparableConv1D(ch, ch, k=3, d=d, dropout=dropout) for d in dilations
        ])
        self.glob = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            ConvGNAct(ch, ch, k=1, dropout=dropout),
        )
        self.fuse = ConvGNAct(ch * (len(dilations) + 1), ch, k=1, dropout=dropout)

    def forward(self, x):
        L = x.shape[-1]
        ys = [b(x) for b in self.branches]
        g = self.glob(x)
        g = F.interpolate(g, size=L, mode="linear", align_corners=False)
        y = torch.cat(ys + [g], dim=1)
        return self.fuse(y)


class AttnPool1D(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.score = nn.Conv1d(ch, 1, kernel_size=1)

    def forward(self, x):
        a = torch.softmax(self.score(x), dim=-1)
        return (x * a).sum(dim=-1)


# ============================================================
# NEW: Cross-signal modulation at bottleneck (FiLM / gating)
# ============================================================
class CrossFiLM1D(nn.Module):
    """
    Cross-signal FiLM:
      h2 <- h2 * (1 + gamma12) + beta12    where (gamma12,beta12) come from h1
      h1 <- h1 * (1 + gamma21) + beta21    where (gamma21,beta21) come from h2

    gamma/beta are (N,C,1) from GAP + small MLP.
    """
    def __init__(self, ch, hidden=128, dropout=0.0):
        super().__init__()
        hidden = max(32, hidden)

        def make_mlp():
            return nn.Sequential(
                nn.Linear(ch, hidden),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, ch * 2)  # gamma + beta
            )

        self.mlp_12 = make_mlp()  # params from h1 -> modulate h2
        self.mlp_21 = make_mlp()  # params from h2 -> modulate h1

        # optional stabilization
        self.ln = nn.LayerNorm(ch)

    def forward(self, h1, h2):
        # h1,h2: (N,C,L)
        g1 = h1.mean(dim=-1)  # (N,C)
        g2 = h2.mean(dim=-1)  # (N,C)

        p12 = self.mlp_12(self.ln(g1))  # (N,2C)
        p21 = self.mlp_21(self.ln(g2))  # (N,2C)

        gamma12, beta12 = p12.chunk(2, dim=1)
        gamma21, beta21 = p21.chunk(2, dim=1)

        gamma12 = gamma12.unsqueeze(-1)
        beta12  = beta12.unsqueeze(-1)
        gamma21 = gamma21.unsqueeze(-1)
        beta21  = beta21.unsqueeze(-1)

        h2m = h2 * (1.0 + torch.tanh(gamma12)) + beta12
        h1m = h1 * (1.0 + torch.tanh(gamma21)) + beta21
        return h1m, h2m


# -------------------------
# Full model with:
#   - CrossFiLM at bottleneck BEFORE fusion
#   - "mean + attn" pooled head
# -------------------------
class DualSignalDilatedAttnUNet1D(nn.Module):
    def __init__(
        self,
        in_ch_per_signal=1,
        base_ch=32,
        levels=4,
        dropout=0.10,
        head_hidden=128,
    ):
        super().__init__()
        assert levels >= 3

        chs = [base_ch * (2 ** i) for i in range(levels)]

        # per-signal stems
        self.stem1 = ConvGNAct(in_ch_per_signal, chs[0], k=7, dropout=0.0)
        self.stem2 = ConvGNAct(in_ch_per_signal, chs[0], k=7, dropout=0.0)

        # encoders (per signal)
        self.enc1 = nn.ModuleList()
        self.enc2 = nn.ModuleList()
        self.down1 = nn.ModuleList()
        self.down2 = nn.ModuleList()

        for i in range(levels):
            blk1 = ResDilateBlock1D(chs[i], chs[i], dropout=dropout)
            blk2 = ResDilateBlock1D(chs[i], chs[i], dropout=dropout)
            self.enc1.append(blk1)
            self.enc2.append(blk2)
            if i < levels - 1:
                self.down1.append(ConvGNAct(chs[i], chs[i+1], k=3, s=2, dropout=0.0))
                self.down2.append(ConvGNAct(chs[i], chs[i+1], k=3, s=2, dropout=0.0))

        # fusion at each scale
        self.fuse = nn.ModuleList([ScaleFusion1D(ch) for ch in chs])

        # skip adapters on fused skips (except bottleneck)
        self.skip_adapt = nn.ModuleList([SkipAdapter1D(chs[i], chs[i], dropout=dropout) for i in range(levels - 1)])

        # NEW: cross-signal FiLM at bottleneck
        self.cross_film = CrossFiLM1D(chs[-1], hidden=max(64, chs[-1] // 2), dropout=dropout)

        # bottleneck ASPP on fused bottleneck features
        self.aspp = ASPP1D(chs[-1], dropout=dropout)

        # decoder
        self.up = nn.ModuleList()
        self.gate = nn.ModuleList()
        self.dec = nn.ModuleList()

        for i in range(levels - 2, -1, -1):
            up_in = chs[i+1]
            up_out = chs[i]
            self.up.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
                ConvGNAct(up_in, up_out, k=3, dropout=0.0)
            ))
            self.gate.append(AttentionGate1D(skip_ch=chs[i], gate_ch=chs[i]))
            self.dec.append(ResDilateBlock1D(chs[i] * 2, chs[i], dropout=dropout))

        # NEW: head = concat(mean_pool, attn_pool)
        self.attn_pool = AttnPool1D(chs[0])
        self.mean_pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(chs[0] * 2, head_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1)
        )

    def forward(self, x):
        if isinstance(x, (tuple, list)):
            x1, x2 = x
        else:
            assert x.dim() == 3 and x.size(1) == 2, "Expected input (N,2,L)"
            x1 = x[:, 0:1, :]
            x2 = x[:, 1:2, :]

        # stems
        h1 = self.stem1(x1)
        h2 = self.stem2(x2)

        fused_skips = []
        bottleneck = None

        for i in range(len(self.enc1)):
            h1 = self.enc1[i](h1)
            h2 = self.enc2[i](h2)

            if i == len(self.enc1) - 1:
                # NEW: Cross-signal modulation at bottleneck BEFORE fusion
                h1m, h2m = self.cross_film(h1, h2)
                hf = self.fuse[i](h1m, h2m)
                bottleneck = hf
            else:
                hf = self.fuse[i](h1, h2)
                fused_skips.append(self.skip_adapt[i](hf))
                h1 = self.down1[i](h1)
                h2 = self.down2[i](h2)

        # bottleneck
        y = self.aspp(bottleneck)

        # decode
        for idx, i in enumerate(range(len(fused_skips) - 1, -1, -1)):
            y = self.up[idx](y)
            skip = fused_skips[i]
            skip_g = self.gate[idx](skip, y)
            y = self.dec[idx](torch.cat([y, skip_g], dim=1))

        # NEW: mean + attn pooling head
        v_attn = self.attn_pool(y)                       # (N,C)
        v_mean = self.mean_pool(y).squeeze(-1)           # (N,C)
        v = torch.cat([v_mean, v_attn], dim=1)           # (N,2C)
        out = self.head(v)
        return out
