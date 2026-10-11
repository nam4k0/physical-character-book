#!/usr/bin/env python3
"""レシピ03: ゲームの数式を自動微分して最適化（第02章）。

    uv run python recipes/r03_autograd_optimize.py

ジャンプ到達高さ h=v^2/(2g) を勾配で最大化し、目標高さにする v を求める。
"""

import torch


def main() -> None:
    g = 9.8

    # 1) 最大化: v を増やすと h が増える
    v = torch.tensor(1.0, requires_grad=True)
    for step in range(30):
        h = v**2 / (2 * g)
        h.backward()
        with torch.no_grad():
            v += 0.5 * v.grad
            v.grad.zero_()
        if step % 10 == 0:
            print(f"step {step}: v={v.item():.2f} h={h.item():.2f}")

    # 2) 目標合わせ: h を 3.0 にする v を求める（二乗誤差を最小化）
    v2 = torch.tensor(1.0, requires_grad=True)
    for _ in range(200):
        loss = ((v2**2 / (2 * g)) - 3.0) ** 2
        loss.backward()
        with torch.no_grad():
            v2 -= 0.05 * v2.grad
            v2.grad.zero_()
    print("h=3.0 にする v:", round(v2.item(), 3))   # 約 7.67


if __name__ == "__main__":
    main()
