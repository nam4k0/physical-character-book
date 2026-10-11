#!/usr/bin/env python3
"""レシピ21: 識別器の正解率を監視してバランスを取る（第20章）。

    uv run python recipes/r21_discriminator_monitor.py
"""

import torch

from ml_lern.core.amp import AMPUpdater
from ml_lern.core.config import default_config


def main() -> None:
    for lr, gp in [(1e-3, 10.0), (1e-4, 10.0), (1e-3, 0.0)]:
        cfg = default_config(disc_learning_rate=lr, gradient_penalty_weight=gp)
        up = AMPUpdater(16, cfg)
        real = torch.randn(256, 16) - 1.0
        fake = torch.randn(256, 16) + 1.0
        for _ in range(300):
            up.update(fake, real)
        with torch.no_grad():
            conf = torch.sigmoid(up.discriminator(real[:128])).mean().item()
        print(f"lr={lr}, gp={gp}: 本物の確信度={conf:.2f} (0.5が健全)")


if __name__ == "__main__":
    main()
