#!/usr/bin/env python3
"""レシピ39: 評価関数と外乱テスト（第38章）。

    uv run python recipes/r39_evaluate_disturbance.py
"""

import torch


def success_rate(disturbance: float = 0.0, trials: int = 2000) -> float:
    dist = torch.rand(trials) * 5.0
    noise = disturbance * torch.randn(trials)
    acc = torch.exp(-2.0 * (dist + noise.abs()))
    return (acc > 0.5).float().mean().item()


def main() -> None:
    for d in [0.0, 0.5, 1.0, 2.0]:
        print(f"外乱 {d}: 成功率={success_rate(d):.3f}")


if __name__ == "__main__":
    main()
