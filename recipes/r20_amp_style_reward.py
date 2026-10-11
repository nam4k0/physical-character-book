#!/usr/bin/env python3
"""レシピ20: AMP の識別器とスタイル報酬を動かす（第19章）。

    uv run python recipes/r20_amp_style_reward.py
"""

import torch

from ml_lern.core.amp import AMPUpdater
from ml_lern.core.config import default_config


def main() -> None:
    up = AMPUpdater(16, default_config())
    real = torch.randn(512, 16) - 1.0                 # 本物（データの遷移）
    fake = torch.randn(512, 16) + 1.0                 # 偽物（方策の遷移）

    for step in range(400):
        stats = up.update(fake, real)
        if step % 100 == 0:
            print(f"step {step}: acc_real={stats['disc_acc_real']:.2f}")

    print("本物のスタイル報酬:", up.predict_style_reward(real[:3]).round(3).tolist())
    print("偽物のスタイル報酬:", up.predict_style_reward(fake[:3]).round(3).tolist())


if __name__ == "__main__":
    main()
