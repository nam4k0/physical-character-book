#!/usr/bin/env python3
"""レシピ09: 前方過程（ノイズ付加）を観察する（第08章）。

    uv run python recipes/r09_forward_diffusion.py
"""

import torch

from ml_lern.generative.diffusion.schedule import NoiseSchedule


def main() -> None:
    sched = NoiseSchedule(num_timesteps=100)

    # きれいなデータ: 直線上に並んだ点
    x0 = torch.zeros(64, 2)
    x0[:, 0] = torch.linspace(-1, 1, 64)

    for t in [0, 10, 30, 60, 99]:
        tt = torch.full((64,), t)
        noise = torch.randn_like(x0)
        x_t = sched.q_sample(x0, tt, noise)
        print(f"t={t:3d}: 標準偏差={x_t.std():.2f}  平均={x_t.mean():.2f}")
    # t が大きいほど構造が失われ、ノイズに近づく


if __name__ == "__main__":
    main()
