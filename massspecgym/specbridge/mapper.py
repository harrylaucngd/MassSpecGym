from __future__ import annotations

import torch
import torch.nn as nn


def unit_normalize(x, eps: float = 1e-12):
    return x / torch.linalg.norm(x, dim=-1, keepdim=True).clamp(min=eps)


class ResidualBlock(nn.Module):
    def __init__(self, d: int, hidden: int, drop: float = 0.0):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, hidden)
        self.fc2 = nn.Linear(hidden, d)
        self.drop = nn.Dropout(drop)
        self.act = nn.GELU()

        nn.init.xavier_uniform_(self.fc1.weight, gain=0.7)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        h = self.ln(x)
        h = self.fc2(self.drop(self.act(self.fc1(h))))
        return x + h


class ProcrustesResidualMapper(nn.Module):
    """
    SpecBridge mapper: linear map + residual MLP blocks (optionally Gaussian head).
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        n_blocks: int = 8,
        hidden: int = 0,
        drop: float = 0.0,
        gaussian: bool = True,
    ):
        super().__init__()
        self.gaussian = gaussian
        self.W = nn.Linear(d_in, d_out, bias=True)
        nn.init.orthogonal_(self.W.weight)
        nn.init.zeros_(self.W.bias)

        if hidden <= 0:
            hidden = max(256, min(d_out, 1024))
        self.blocks = nn.ModuleList([ResidualBlock(d_out, hidden, drop) for _ in range(n_blocks)])
        self.lv = nn.Linear(d_out, d_out) if gaussian else None

    def forward(self, x):
        mu = self.W(x)
        for blk in self.blocks:
            mu = blk(mu)
        if self.gaussian:
            lv = self.lv(mu)
            return mu, lv
        return mu, None
