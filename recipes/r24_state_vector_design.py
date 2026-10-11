#!/usr/bin/env python3
"""レシピ24: 状態ベクトルを設計し local 変換する（第23章）。

    uv run python recipes/r24_state_vector_design.py
"""

import numpy as np

N_JOINTS = 12


def main() -> None:
    state = {
        "root_height": 1,
        "root_rot_6d": 6,
        "root_vel": 6,
        "joint_rot_6d": 6 * N_JOINTS,
        "joint_vel": N_JOINTS,
        "key_joints_pos": 4 * 3,
        "object_pos": 3,
        "object_rot_6d": 6,
    }
    for k, v in state.items():
        print(f"{k:16s}: {v:4d}")
    print("合計:", sum(state.values()))

    def world_to_local(p_world, root_pos, root_yaw):
        c, s = np.cos(-root_yaw), np.sin(-root_yaw)
        d = p_world - root_pos
        return np.array([c * d[0] - s * d[1], s * d[0] + c * d[1], d[2]])

    print("local 座標:", world_to_local(np.array([3, 4, 1.0]), np.array([1, 1, 0.0]), 0.5).round(2))


if __name__ == "__main__":
    main()
