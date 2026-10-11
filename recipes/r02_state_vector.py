#!/usr/bin/env python3
"""レシピ02: キャラの状態をベクトル化して正規化（第01章）。

    uv run python recipes/r02_state_vector.py

1フレームの状態をベクトルにし、時系列・スライス・正規化を体験する。
"""

import numpy as np


def main() -> None:
    # 状態 = 位置(3) + 回転クォータニオン(4) + 関節角(20)
    d = 3 + 4 + 20
    state = np.zeros(d)
    state[:3] = [10.0, 0.0, 2.0]
    state[3:7] = [1, 0, 0, 0]
    state[7:] = np.random.uniform(-1, 1, size=20)

    # 1000フレーム分を (フレーム, 特徴) にまとめる
    motion = np.random.randn(1000, d)
    print("motion.shape =", motion.shape)          # (1000, 27)
    print("位置の時系列:", motion[:, 0:3].shape)    # (1000, 3)

    # ブロードキャストで特徴ごとに正規化（平均0・分散1）
    mu = motion.mean(axis=0)
    sigma = motion.std(axis=0)
    normalized = (motion - mu) / (sigma + 1e-8)
    print("正規化後の平均(先頭3):", np.round(normalized.mean(0), 5)[:3])


if __name__ == "__main__":
    main()
