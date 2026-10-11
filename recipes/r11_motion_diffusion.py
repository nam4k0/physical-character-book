#!/usr/bin/env python3
"""レシピ11: 関節軌道の拡散でモーションを生成する（第10章）。

    uv run python recipes/r11_motion_diffusion.py
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

    model, sched, (frames, feat) = train_motion_diffusion(
        num_samples=2000, frames=frames, num_joints=num_joints,
        num_timesteps=100, num_epochs=80,
    )

    data = make_motion_dataset(num_samples=3, frames=frames, num_joints=num_joints, seed=1)
    motion = sample_motion(model, sched, frames, feat, data["first_pose"], seed=0)
    print("生成モーション:", tuple(motion.shape))     # (3, 32, 12)
    print("第1フレームが指定通り:",
          torch.allclose(motion[:, 0, :], data["first_pose"], atol=1e-4))


if __name__ == "__main__":
    main()
