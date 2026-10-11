#!/usr/bin/env python3
"""レシピ14: フレームスタッキングで履歴を状態にする（第13章）。

    uv run python recipes/r14_frame_stack_state.py
"""

import torch

from ml_lern.core.mujoco_basics import frame_stack


def main() -> None:
    hist = []
    obs = torch.zeros(8)
    for t in range(5):
        obs = obs + 0.1 * torch.randn(8)             # 観測が少しずつ変化
        stacked = frame_stack(hist, obs, stack=4)    # 直近4フレームを連結
        print(f"t={t}: 状態の次元 = {stacked.shape[-1]}")


if __name__ == "__main__":
    main()
