"""ノイズ除去ネットワーク（第09章）。

入力: ノイズ付きデータ x_t、時刻 t、条件（任意）
出力: 加えられたノイズ eps の予測

時刻 t は正弦波（sinusoidal）埋め込みでベクトル化して入力する（第09章）。
条件付き生成では、初期位置などを条件として入力に連結する（第10章）。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """時刻 t（整数）を sin/cos で埋め込む（第09章）。"""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class Denoiser(nn.Module):
    """全結合のノイズ除去ネット（第09章）。"""

    def __init__(self, data_dim: int, cond_dim: int = 0, hidden: int = 256, time_dim: int = 64):
        super().__init__()
        self.time_dim = time_dim
        self.cond_dim = cond_dim
        in_dim = data_dim + time_dim + cond_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, data_dim),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor,
                condition: torch.Tensor | None = None) -> torch.Tensor:
        batch = x_t.shape[0]
        x_flat = x_t.reshape(batch, -1)
        t_emb = sinusoidal_embedding(t, self.time_dim)
        pieces = [x_flat, t_emb]
        if self.cond_dim > 0:
            assert condition is not None, "条件付きモデルには条件が必要です"
            pieces.append(condition.reshape(batch, -1))
        inp = torch.cat(pieces, dim=-1)
        return self.net(inp).reshape_as(x_t)
