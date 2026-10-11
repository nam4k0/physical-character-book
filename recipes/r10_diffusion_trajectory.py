#!/usr/bin/env python3
"""レシピ10: 自分の2D軌道で拡散モデルを作る（第09章）。

    uv run python recipes/r10_diffusion_trajectory.py
"""

import numpy as np
import torch

from ml_lern.generative.diffusion.model import Denoiser
from ml_lern.generative.diffusion.sample import sample_conditional
from ml_lern.generative.diffusion.schedule import NoiseSchedule


def make_traj(steps: int = 32, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, steps)
    y0 = rng.uniform(0.2, 0.8)
    v = rng.uniform(1.0, 2.0)
    y = y0 + v * t - 2.0 * t**2
    return np.stack([t, y], axis=-1).astype(np.float32)


def main() -> None:
    steps = 32
    data = torch.tensor(np.stack([make_traj(seed=i) for i in range(2000)]))  # (N,32,2)
    starts = data[:, 0, :].clone()

    sched = NoiseSchedule(100)
    model = Denoiser(data_dim=steps * 2, cond_dim=2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    flat = data.reshape(len(data), -1)

    loss = None
    for _ in range(150):
        idx = torch.randint(0, len(data), (128,))
        x0, cond = flat[idx], starts[idx]
        t = torch.randint(0, 100, (128,))
        noise = torch.randn_like(x0)
        x_t = sched.q_sample(x0, t, noise)
        loss = ((model(x_t, t, cond) - noise) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    print("最終 loss:", round(loss.item(), 4))

    out = sample_conditional(model, sched, steps, starts[:3], seed=0)
    print("生成軌道:", tuple(out.shape))     # (3, 32, 2)


if __name__ == "__main__":
    main()
