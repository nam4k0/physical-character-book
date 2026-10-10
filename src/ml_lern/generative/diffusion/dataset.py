"""2D バウンスボールのデータ生成（第09章）。

箱の中を跳ね回るボールの軌道（x, y の時系列）を生成する。
物理は単純（重力＋壁で反射）なので、教師データを手軽に作れる。

条件付き生成用に、初期位置 x0 も保存する（第09・10章）。
"""

from __future__ import annotations

import numpy as np
import torch


def simulate_bounce(
    steps: int = 64,
    dt: float = 0.05,
    box: float = 1.0,
    gravity: float = 2.0,
    x0: np.ndarray | None = None,
    v0: np.ndarray | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """1本の跳ねる軌道を生成する。返り値は (steps, 2)。"""
    rng = np.random.default_rng(seed)
    if x0 is None:
        x0 = rng.uniform(0.1, 0.9, size=2).astype(np.float32)
    if v0 is None:
        v0 = rng.uniform(-1.0, 1.0, size=2).astype(np.float32)

    pos = x0.astype(np.float32).copy()
    vel = v0.astype(np.float32).copy()
    traj = []
    for _ in range(steps):
        vel[1] -= gravity * dt          # 重力（y方向）
        pos = pos + vel * dt
        for d in range(2):              # 壁で反射
            if pos[d] < 0.0:
                pos[d] = -pos[d]
                vel[d] = -vel[d]
            elif pos[d] > box:
                pos[d] = 2 * box - pos[d]
                vel[d] = -vel[d]
        traj.append(pos.copy())
    return np.stack(traj)


def make_dataset(
    num_samples: int = 20000,
    steps: int = 64,
    condition_on_start: bool = True,
    seed: int = 0,
) -> dict:
    """軌道データセットを作る（第09章）。

    condition_on_start=True のとき、初期位置 x0 を条件として保存する。
    """
    rng = np.random.default_rng(seed)
    trajs = np.zeros((num_samples, steps, 2), dtype=np.float32)
    starts = np.zeros((num_samples, 2), dtype=np.float32)
    for i in range(num_samples):
        x0 = rng.uniform(0.1, 0.9, size=2).astype(np.float32)
        trajs[i] = simulate_bounce(steps=steps, x0=x0, seed=int(rng.integers(1 << 30)))
        starts[i] = x0

    return {
        "traj": torch.from_numpy(trajs),        # (N, steps, 2)
        "start": torch.from_numpy(starts),      # (N, 2)
    }
