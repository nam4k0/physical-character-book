#!/usr/bin/env python3
"""第14・15・18章: 強化学習と模倣学習の基本。

    uv run python scripts/p02_rl_basics.py
"""

import torch

from ml_lern.core import imitation, rewards, rl_basics


def main() -> None:
    # 第14・15章: 収益と GAE
    rl_basics.demo()

    # 第17章: 歩行報酬
    obs = torch.randn(4, 10)
    print("歩行報酬:", rewards.task_reward_locomotion(obs).round(3).tolist())

    # 第18章: モーショントラッキング報酬
    pose, ref = torch.zeros(4, 6), torch.randn(4, 6) * 0.1
    print("追従報酬:", imitation.tracking_reward(pose, ref).round(3).tolist())


if __name__ == "__main__":
    main()
