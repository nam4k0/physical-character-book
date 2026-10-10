#!/usr/bin/env python3
"""第10章: モーション生成への拡張（拡散モデル）。

    uv run python scripts/g04_diffusion_motion.py

初期姿勢を条件に、そこから続くモーションを生成する（inpainting 風）。
第09章の 2D 実装を、データの形 (B, T, F) に拡張したもの。
"""

import torch

from ml_lern.generative.diffusion.motion import (
    make_motion_dataset,
    sample_motion,
    train_motion_diffusion,
)


def main() -> None:
    frames, num_joints = 32, 4
    feat = num_joints * 3

    model, schedule, (frames, feat) = train_motion_diffusion(
        num_samples=2000, frames=frames, num_joints=num_joints,
        num_timesteps=100, num_epochs=80,
    )

    # 生成する初期姿勢をいくつか用意（データから取り出す）
    data = make_motion_dataset(num_samples=3, frames=frames, num_joints=num_joints, seed=1)
    first_pose = data["first_pose"]

    motion = sample_motion(model, schedule, frames, feat, first_pose, seed=0)
    print("生成モーションの形状:", tuple(motion.shape))     # (3, 32, 12)
    print("第1フレームが指定通りか:",
          torch.allclose(motion[:, 0, :], first_pose, atol=1e-4))


if __name__ == "__main__":
    main()
