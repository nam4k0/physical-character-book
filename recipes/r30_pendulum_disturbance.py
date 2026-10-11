#!/usr/bin/env python3
"""レシピ30: 外乱つきの倒立振子で頑健さを試す（第29章）。

    uv run python recipes/r30_pendulum_disturbance.py
"""

import gymnasium as gym
import numpy as np


def main() -> None:
    env = gym.make("InvertedPendulum-v5")
    env.reset(seed=0)

    survived = 0
    for _ in range(500):
        action = env.action_space.sample() * 0.1
        if np.random.rand() < 0.1:                       # 一定確率で外乱
            action = action + np.random.uniform(-0.5, 0.5, size=action.shape)
        _, _, term, trunc, _ = env.step(action)
        if term or trunc:
            break
        survived += 1
    env.close()
    print(f"外乱ありで {survived} ステップ持ちこたえた")


if __name__ == "__main__":
    main()
