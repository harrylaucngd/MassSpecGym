from __future__ import annotations

import torch
import torch.nn as nn


def unit_normalize(x, eps: float = 1e-12):
    return x / torch.linalg.norm(x, dim=-1, keepdim=True).clamp(min=eps)


class DreamsAdapter(nn.Module):
    """
    SpecBridge's spectrum branch: DreaMS backbone + pooling + projection + LayerNorm.
    """

    def __init__(
        self,
        dreams_encoder,
        d_out: int,
        hidden: int = 0,
        freeze_backbone: bool = True,
        pool: str = "mean",
        peak_dropout: float = 0.0,
    ):
        super().__init__()
        self.dreams = dreams_encoder
        self.pool = pool
        self.peak_dropout = peak_dropout
        d_in = int(getattr(dreams_encoder, "embed_dim", 1024))

        if hidden == 0:
            proj0 = nn.Linear(d_in, d_out)
        else:
            proj0 = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, d_out))
        self.proj = nn.Sequential(proj0, nn.LayerNorm(d_out))

        if freeze_backbone:
            for p in self.dreams.parameters():
                p.requires_grad = False

        if pool == "attn":
            self.attn = nn.Linear(d_in, 1)

    def _pool(self, seq, peaks):
        import torch

        mask = (peaks[..., 0] > 0).float()
        if self.pool == "max":
            v = seq.masked_fill(mask.unsqueeze(-1) == 0, float("-inf"))
            pooled = torch.nan_to_num(v.max(dim=1).values, nan=0.0, neginf=0.0)
        elif self.pool == "attn":
            w = self.attn(seq).squeeze(-1)
            w = w + (mask + 1e-6).log()
            w = torch.softmax(w, dim=1)
            pooled = (seq * w.unsqueeze(-1)).sum(dim=1)
        else:
            denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            pooled = (seq * mask.unsqueeze(-1)).sum(dim=1) / denom
        return pooled

    def forward(self, peaks, meta=None):
        with torch.inference_mode():
            out = self.dreams(peaks, meta)
            if out.dim() == 3:
                z0 = self._pool(out, peaks)
            else:
                z0 = out
            z = self.proj(z0)
        return unit_normalize(z)
