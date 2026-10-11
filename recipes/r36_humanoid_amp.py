#!/usr/bin/env python3
"""レシピ36: 人型で AMP を統合する（第35章）。

    uv run python recipes/r36_humanoid_amp.py

既定では環境の次元確認のみ。学習は scripts/train_05_humanoid_amp.py で回す。
"""

from ml_lern.charcontrol.train import make_env
from ml_lern.core.config import default_config


def main() -> None:
    cfg = default_config(num_envs=8, episode_length_s=10.0)
    env = make_env("humanoid_amp", cfg, device="cpu")

    print("状態次元:", env.obs_dim, "行動次元:", env.action_dim, "AMP遷移:", env.amp_obs_dim)
    obs = env.reset()
    print("初期観測:", tuple(obs.shape), "目標:", tuple(env.goal.shape))
    env.close()

    print("学習: uv run python scripts/train_05_humanoid_amp.py")


if __name__ == "__main__":
    main()
