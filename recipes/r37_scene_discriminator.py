#!/usr/bin/env python3
"""レシピ37: 物体を足してシーン条件付き識別器にする（第36章）。

    uv run python recipes/r37_scene_discriminator.py
"""

from ml_lern.charcontrol.train import make_env
from ml_lern.core.config import default_config


def main() -> None:
    cfg = default_config(num_envs=4, episode_length_s=15.0)
    env = make_env("scene", cfg, device="cpu")
    obs = env.reset()
    goal = env.goal

    amp = env.get_amp_obs(obs, goal)
    print("状態:", tuple(obs.shape), "物体込みAMP遷移:", tuple(amp.shape))
    print("目標(物体の中心＋bbox):", tuple(goal.shape))
    env.close()


if __name__ == "__main__":
    main()
