#!/usr/bin/env python3
"""第19・20章: AMP の識別器とスタイル報酬。

    uv run python scripts/p04_amp.py

識別器をランダムデータで動かし、スタイル報酬と正解率の挙動を確認する。
（MuJoCo と組み合わせた本格的な学習は train_05_humanoid_amp.py を使う。）
"""

import torch

from ml_lern.core.amp import AMPUpdater, style_reward
from ml_lern.core.config import default_config
from ml_lern.core.models import Discriminator


def main() -> None:
    torch.manual_seed(0)
    cfg = default_config()
    amp_dim = 32

    updater = AMPUpdater(amp_dim, cfg)
    dataset_amp_obs = torch.randn(1024, amp_dim)            # 本物（データ）
    policy_amp_obs = torch.randn(1024, amp_dim) + 0.5        # 偽物（方策）

    for step in range(200):
        if step % 50 == 0:
            stats = updater.update(policy_amp_obs, dataset_amp_obs)
            print(f"step {step}: {stats}")
        else:
            updater.update(policy_amp_obs, dataset_amp_obs)

    # スタイル報酬の例
    d = torch.sigmoid(updater.discriminator(dataset_amp_obs[:4]))
    print("識別器の出力（シグモイド）:", d.round(3).tolist())
    print("スタイル報酬:", style_reward(updater.discriminator, dataset_amp_obs[:4]).round(3).tolist())


if __name__ == "__main__":
    main()
