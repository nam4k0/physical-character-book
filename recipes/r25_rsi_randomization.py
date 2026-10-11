#!/usr/bin/env python3
"""レシピ25: RSI と物体のランダム化（第24章）。

    uv run python recipes/r25_rsi_randomization.py
"""

import math

import torch


def sample_object_placement(n, lo=1.0, hi=10.0):
    dist = torch.empty(n).uniform_(lo, hi)
    theta = torch.empty(n).uniform_(0.0, 2 * math.pi)
    x, y = dist * torch.cos(theta), dist * torch.sin(theta)
    return torch.stack([x, y, torch.full_like(x, 0.5)], dim=-1)


def main() -> None:
    print("配置例:\n", sample_object_placement(3).round(2))

    scale = torch.empty(5).uniform_(0.5, 1.5)
    print("スケール:", scale.round(2).tolist())

    ref = torch.randn(2000, 30)
    starts = ref[torch.randint(0, len(ref), (4,))]
    print("RSI 初期状態:", tuple(starts.shape))


if __name__ == "__main__":
    main()
