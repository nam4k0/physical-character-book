#!/usr/bin/env python3
"""レシピ34: 自作タスクで PPO 学習ループを組む（第33章）。

    uv run python recipes/r34_ppo_loop.py
"""

import torch

from ml_lern.core.config import default_config
from ml_lern.core.models import Policy, Value
from ml_lern.core.ppo import compute_gae


def main() -> None:
    cfg = default_config()

    # 自作タスクの次元に合わせて、方策と価値関数を作る
    policy = Policy(obs_dim=10, goal_dim=1, action_dim=2)
    value = Value(obs_dim=10, goal_dim=1)

    obs, goal = torch.randn(32, 10), torch.randn(32, 1)
    actions, logp = policy.act(obs, goal)
    values = value(obs, goal)
    rewards = torch.randn(32)
    dones = torch.zeros(32)
    adv, ret = compute_gae(rewards, values, dones, cfg.gamma, cfg.gae_lambda, torch.zeros(1))

    print("アドバンテージ:", tuple(adv.shape), "収益:", tuple(ret.shape))
    print("行動:", tuple(actions.shape), "対数確率:", tuple(logp.shape))


if __name__ == "__main__":
    main()
