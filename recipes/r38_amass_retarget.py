#!/usr/bin/env python3
"""レシピ38: AMASS の読み込みとリターゲット（第37章）。

    uv run python recipes/r38_amass_retarget.py

データ実体（AMASS / SMPL）はライセンス登録のうえ取得が必要。ここでは構造と手順のみ。
"""

import numpy as np

AMASS_KEYS = ["trans", "poses", "betas", "mocap_framerate"]


def retarget(amass_pose_aa: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """SMPL の関節角を MuJoCo humanoid の関節角へフィットする（概念）。"""
    # 1) 関節の対応付け（SMPL <-> robot）
    # 2) 体型 beta をロボットに合わせて最適化
    # 3) 各フレームで最適化して関節角を得る
    return np.zeros((20,))


def main() -> None:
    print("AMASS の典型キー:", AMASS_KEYS)
    print("  trans: (フレーム, 3)  ルート位置")
    print("  poses: (フレーム, 72) ルート3 + 関節23×3（軸角）")
    print("  betas: (10,)          体型パラメータ")

    j = 23
    pose_aa = np.zeros((100, j * 3))
    q = retarget(pose_aa, beta=np.zeros(10))
    print("リターゲット後の関節角:", q.shape)


if __name__ == "__main__":
    main()
