#!/usr/bin/env python3
"""レシピ19: 追従報酬で見本を真似る（第18章）。

    uv run python recipes/r19_tracking_reward.py
"""

import torch

from ml_lern.core.imitation import ReferenceMotionBuffer, tracking_reward


def main() -> None:
    pose = torch.zeros(4, 6)
    ref = torch.zeros(4, 6)
    print("一致:", tracking_reward(pose, ref).round(3).tolist())

    ref2 = ref + 0.3 * torch.randn(4, 6)
    print("ずれ:", tracking_reward(pose, ref2).round(3).tolist())

    buf = ReferenceMotionBuffer()
    for _ in range(10):
        buf.add_frame(torch.randn(30))
    buf.finalize()
    print("参照の遷移:", tuple(buf.ref_amp_obs.shape))   # (9, 60)


if __name__ == "__main__":
    main()
