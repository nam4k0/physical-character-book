#!/usr/bin/env python3
"""レシピ04: 成長曲線を MLP で学習し、未見レベルを予測（第03章）。

    uv run python recipes/r04_mlp_growth_curve.py
"""

import torch

from ml_lern.foundations.nn_basics import MLP


def main() -> None:
    level = torch.linspace(1, 50, 50).unsqueeze(1)
    exp = 100 * level**1.5
    exp_n = exp / exp.max()                 # 0..1 に正規化

    net = MLP(1, hidden=(64, 64), out_dim=1)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    for step in range(3000):
        pred = net(level / 50)              # 入力も正規化
        loss = ((pred - exp_n) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        pred37 = net(torch.tensor([[37 / 50]])).item() * exp.max().item()
    print("レベル37の予測経験値:", round(pred37))
    print("真値:", round(100 * 37**1.5))


if __name__ == "__main__":
    main()
