#!/usr/bin/env python3
"""レシピ18: far/near 報酬と RSI を実装する（第17章）。

    uv run python recipes/r18_reward_far_near.py
"""

import torch

from ml_lern.core.rewards import exp_reward, reference_state_init


def main() -> None:
    dist = torch.tensor([0.0, 0.3, 0.49, 0.51, 1.0, 3.0, 8.0])
    r_near = exp_reward(dist, k=10.0)
    r_far = exp_reward(dist, k=0.5)
    near_mask = (dist <= 0.5).float()
    reward = 0.7 * r_near + 0.3 * (near_mask * 1.0 + (1 - near_mask) * r_far)
    for d, r in zip(dist.tolist(), reward.tolist()):
        print(f"距離 {d:4.2f}m -> 報酬 {r:.3f}")

    ref = torch.randn(500, 30)
    start = reference_state_init(ref)
    print("RSI の初期状態:", tuple(start.shape))


if __name__ == "__main__":
    main()
