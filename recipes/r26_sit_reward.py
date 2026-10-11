#!/usr/bin/env python3
"""レシピ26: sit 報酬（式7-9）とモード切替（第25章）。

    uv run python recipes/r26_sit_reward.py
"""

import torch

from ml_lern.core.rewards import exp_reward, sit_task_reward


def main() -> None:
    root = torch.zeros(6, 2)
    obj = torch.tensor([[0.0, 0.0], [0.3, 0.0], [0.49, 0.0], [0.51, 0.0], [1.0, 0.0], [5.0, 0.0]])
    r = sit_task_reward(root, obj, threshold=0.5)
    for o, ri in zip(obj[:, 0].tolist(), r.tolist()):
        print(f"距離 {o:4.2f}m -> 報酬 {ri:.3f}")

    print("k=3 :", exp_reward(torch.tensor([0.2, 0.5]), k=3.0).round(3).tolist())
    print("k=20:", exp_reward(torch.tensor([0.2, 0.5]), k=20.0).round(3).tolist())


if __name__ == "__main__":
    main()
