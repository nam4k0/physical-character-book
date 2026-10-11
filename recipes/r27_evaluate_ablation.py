#!/usr/bin/env python3
"""レシピ27: 成功率評価とアブレーション（第26章）。

    uv run python recipes/r27_evaluate_ablation.py
"""

import torch


def success_rate(has_object_info: bool = True, trials: int = 1000) -> float:
    dist = torch.rand(trials) * 5.0
    if has_object_info:
        acc = torch.exp(-2.0 * dist)          # 情報あり: 近づける
    else:
        acc = torch.full_like(dist, 0.5)      # 情報なし: 当てずっぽう
    return (acc > 0.5).float().mean().item()


def main() -> None:
    print("物体情報あり:", success_rate(True))
    print("物体情報なし:", success_rate(False))


if __name__ == "__main__":
    main()
