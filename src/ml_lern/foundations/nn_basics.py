"""ニューラルネットと学習の例（第03章）。

- MLP            : 全結合ネットワーク（論文の方策・識別器の原型）
- 線形回帰        : 最小の学習
- sin の近似      : 非線形関数をデータから作る
- 簡単な分類      : クロスエントロピー
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    """全結合＋ReLU の多層パーセプトロン。

    論文のネットワーク構成 {1024, 512} の縮小版として使う。
    """

    def __init__(self, in_dim: int, hidden: tuple[int, ...] = (64, 64), out_dim: int = 1):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_linear_regression(steps: int = 300, lr: float = 1e-2) -> tuple[float, float]:
    """y = 3x + 2 を Adam で学習する（第03章）。"""
    torch.manual_seed(0)
    x = torch.randn(200, 1)
    y = 3.0 * x + 2.0 + 0.1 * torch.randn(200, 1)

    model = nn.Linear(1, 1)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        loss = nn.functional.mse_loss(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    w, b = model.weight.item(), model.bias.item()
    print(f"[線形回帰] w={w:.3f} (真値3.0), b={b:.3f} (真値2.0)")
    return w, b


def train_mlp_sine(steps: int = 2000, lr: float = 1e-2) -> MLP:
    """y = sin(x) を MLP で近似する（第03章）。"""
    x = torch.linspace(-3, 3, 200).unsqueeze(1)
    y = torch.sin(x)

    model = MLP(1, hidden=(64, 64), out_dim=1)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for step in range(steps):
        loss = nn.functional.mse_loss(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 500 == 0:
            print(f"[sin近似] step {step:4d} loss={loss.item():.5f}")
    return model


def train_classifier(steps: int = 500, lr: float = 1e-2) -> MLP:
    """2つのガウス分布の分類（クロスエントロピー、第03章）。

    識別器（第07章）の最小例。
    """
    torch.manual_seed(0)
    n = 500
    # クラス0: 平均(-2,-2), クラス1: 平均(2,2)
    x0 = torch.randn(n, 2) * 0.7 + torch.tensor([-2.0, -2.0])
    x1 = torch.randn(n, 2) * 0.7 + torch.tensor([2.0, 2.0])
    x = torch.cat([x0, x1], dim=0)
    y = torch.cat([torch.zeros(n), torch.ones(n)]).long()

    model = MLP(2, hidden=(32, 32), out_dim=2)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        loss = nn.functional.cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        acc = (model(x).argmax(dim=-1) == y).float().mean().item()
    print(f"[分類] 精度={acc:.3f}")
    return model


def main() -> None:
    train_linear_regression()
    train_mlp_sine()
    train_classifier()


if __name__ == "__main__":
    main()
