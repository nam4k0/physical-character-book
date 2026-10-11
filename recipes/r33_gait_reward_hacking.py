#!/usr/bin/env python3
"""レシピ33: 歩行と報酬ハッキングを観察する（第32章）。

    uv run python recipes/r33_gait_reward_hacking.py
"""

import gymnasium as gym
import torch

from ml_lern.core.rewards import task_reward_locomotion


def main() -> None:
    env = gym.make("Walker2d-v5")
    obs, _ = env.reset(seed=0)

    for step in range(200):
        obs, _, term, trunc, _ = env.step(env.action_space.sample())
        if step % 50 == 0:
            o = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            total = task_reward_locomotion(o, target_speed=1.5).item()
            speed_r = torch.exp(-2 * (1.5 - o[0, 0]) ** 2).item()
            print(f"step {step}: 合計={total:.3f} 速度項={speed_r:.3f}")
        if term or trunc:
            break
    env.close()


if __name__ == "__main__":
    main()
