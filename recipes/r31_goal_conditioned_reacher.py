#!/usr/bin/env python3
"""レシピ31: 目標条件付きの Reacher を作る（第30章）。

    uv run python recipes/r31_goal_conditioned_reacher.py
"""

import gymnasium as gym
import torch


def split(obs):
    return obs[:-2], obs[-2:]          # (状態, 目標への相対座標)


def main() -> None:
    env = gym.make("Reacher-v5")
    obs, _ = env.reset(seed=0)

    state, goal = split(torch.as_tensor(obs, dtype=torch.float32))
    print("状態:", state.shape, "目標:", goal.shape)

    policy_input = torch.cat([state, goal])
    print("方策の入力次元:", policy_input.shape[-1])
    env.close()


if __name__ == "__main__":
    main()
