#!/usr/bin/env python3
"""レシピ13: PDゲイン／timestep／摩擦を変えて観察（第12章）。

    uv run python recipes/r13_pd_tuning.py
"""

import gymnasium as gym


def rollout(env_id: str = "InvertedPendulum-v5", steps: int = 200, action_scale: float = 0.1) -> float:
    env = gym.make(env_id)
    env.reset(seed=0)
    total = 0.0
    for _ in range(steps):
        action = env.action_space.sample() * action_scale
        _, r, term, trunc, _ = env.step(action)
        total += r
        if term or trunc:
            break
    env.close()
    return total


def main() -> None:
    for s in [0.05, 0.1, 0.5, 1.0]:
        print(f"action_scale={s}: 合計報酬={rollout(action_scale=s):.1f}")
    # 大きすぎると振動・転倒、小さすぎると応答しない


if __name__ == "__main__":
    main()
