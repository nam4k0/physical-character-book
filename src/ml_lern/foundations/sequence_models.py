"""系列モデル（第05章）。

時間の並びを扱うネットワークの最小実装。
- FrameStackMLP : 直近フレームをまとめて入力する MLP
- RNNPredictor   : LSTM
- ConvPredictor  : 1次元 CNN
- TransformerPredictor : Transformer Encoder

いずれも入力 (B, T, F) を想定する（第04章の形）。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FrameStackMLP(nn.Module):
    """直近 stack フレームを1入力にまとめる MLP（第05章）。"""

    def __init__(self, feat_dim: int, stack: int = 4, hidden: int = 128, out_dim: int = 1):
        super().__init__()
        self.stack = stack
        self.net = nn.Sequential(
            nn.Linear(feat_dim * stack, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F) -> 最後の stack フレームを使う
        recent = x[:, -self.stack:, :].reshape(x.shape[0], -1)
        return self.net(recent)


class RNNPredictor(nn.Module):
    """LSTM による系列予測（第05章）。"""

    def __init__(self, feat_dim: int, hidden: int = 128, num_layers: int = 1, out_dim: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(feat_dim, hidden, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, (h, c) = self.lstm(x)          # x: (B, T, F) -> out: (B, T, hidden)
        return self.head(out[:, -1, :])     # 最後の時刻の出力を使う


class ConvPredictor(nn.Module):
    """1次元 CNN による系列予測（第05章）。"""

    def __init__(self, feat_dim: int, channels: int = 64, kernel: int = 5, out_dim: int = 1):
        super().__init__()
        self.conv = nn.Conv1d(feat_dim, channels, kernel_size=kernel, padding=kernel // 2)
        self.act = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(channels, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv1d は (B, C, T) を要求するので時間を最後に移す
        h = x.transpose(1, 2)               # (B, F, T)
        h = self.act(self.conv(h))          # (B, channels, T)
        h = self.pool(h).squeeze(-1)        # (B, channels)
        return self.head(h)


class TransformerPredictor(nn.Module):
    """Transformer Encoder による系列予測（第05章）。"""

    def __init__(self, feat_dim: int, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 2, out_dim: int = 1):
        super().__init__()
        self.inp = nn.Linear(feat_dim, d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.inp(x)                     # (B, T, d_model)
        h = self.encoder(h)
        h = h.mean(dim=1)                   # 時間方向に平均プール
        return self.head(h)


def demo() -> None:
    """各モデルの形の確認（第05章）。"""
    B, T, F = 8, 20, 6
    x = torch.randn(B, T, F)
    for name, model in [
        ("FrameStackMLP", FrameStackMLP(F, stack=4)),
        ("RNN", RNNPredictor(F)),
        ("Conv1d", ConvPredictor(F)),
        ("Transformer", TransformerPredictor(F)),
    ]:
        y = model(x)
        print(f"{name:16s} -> {tuple(y.shape)}")


if __name__ == "__main__":
    demo()
