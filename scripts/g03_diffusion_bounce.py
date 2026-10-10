#!/usr/bin/env python3
"""第09章: 2D バウンスボールの拡散モデル。

    uv run python scripts/g03_diffusion_bounce.py

- 無条件生成: 跳ねる軌道をランダムに生成
- 条件付き生成: 初期位置を指定して軌道を生成（第10章 Diffusion Policy の入口）
"""

import torch

from ml_lern.generative.diffusion.sample import sample_conditional, sample_unconditional
from ml_lern.generative.diffusion.train import train_diffusion


def main() -> None:
    steps = 64
    model, schedule, (steps, cond_dim) = train_diffusion(
        num_samples=8000, steps=steps, num_timesteps=100, num_epochs=100,
        condition_on_start=True,
    )

    traj = sample_unconditional(model, schedule, steps, num_samples=4, seed=0)
    print("無条件生成の軌道形状:", tuple(traj.shape))     # (4, 64, 2)

    starts = torch.tensor([[0.2, 0.8], [0.8, 0.8], [0.5, 0.5]], dtype=torch.float32)
    traj_c = sample_conditional(model, schedule, steps, starts, seed=0)
    print("条件付き生成の軌道形状:", tuple(traj_c.shape))  # (3, 64, 2)

    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        for i in range(traj.shape[0]):
            axes[0].plot(traj[i, :, 0], traj[i, :, 1], alpha=0.7)
        axes[0].set_title("unconditional")
        for i in range(traj_c.shape[0]):
            axes[1].plot(traj_c[i, :, 0], traj_c[i, :, 1], alpha=0.7)
            axes[1].scatter(starts[i, 0], starts[i, 1], marker="x", color="k")
        axes[1].set_title("conditional on start")
        plt.savefig("diffusion_bounce.png", dpi=120)
        print("可視化を diffusion_bounce.png に保存しました")
    except Exception as e:
        print("可視化はスキップ:", e)


if __name__ == "__main__":
    main()
