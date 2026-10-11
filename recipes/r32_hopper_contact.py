#!/usr/bin/env python3
"""レシピ32: 接地と速度報酬（Hopper）（第31章）。

    uv run python recipes/r32_hopper_contact.py
"""

import gymnasium as gym
import torch

from ml_lern.core.rewards import task_reward_locomotion


def main() -> None:
    env = gym.make("Hopper-v5")
    obs, _ = env.reset(seed=0)

    for step in range(100):
        obs, _, term, trunc, _ = env.step(env.action_space.sample())
        if step % 20 == 0:
            o = torch.as_tensor(obs, dtype=torch.float32)
            r = task_reward_locomotion(o.unsqueeze(0), target_speed=1.5).item()
            print(f"step {step}: 速度ベース報酬={r:.3f}, 高さ={obs[0]:.2f}")
        if term or trunc:
            break
    env.close()


if __name__ == "__main__":
    main()
