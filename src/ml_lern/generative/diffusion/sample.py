"""拡散モデルによる生成（第09・10章）。

- 無条件生成: 乱数から跳ねる軌道を生成
- 条件付き生成: 初期位置 x0 を与え、そこから始まる軌道を生成（第10章 Diffusion Policy の入口）
"""

from __future__ import annotations

import torch

from .model import Denoiser
from .schedule import NoiseSchedule


@torch.no_grad()
def sample(
    model: Denoiser,
    schedule: NoiseSchedule,
    steps: int,
    num_samples: int = 8,
    condition: torch.Tensor | None = None,
    device: str = "cpu",
    seed: int | None = None,
) -> torch.Tensor:
    """逆過程 T→0 を回して軌道を生成する。返り値は (num_samples, steps, 2)。"""
    if seed is not None:
        torch.manual_seed(seed)

    x = torch.randn(num_samples, steps * 2, device=device)   # 完全なノイズから開始
    for t in reversed(range(schedule.num_timesteps)):
        t_batch = torch.full((num_samples,), t, dtype=torch.long, device=device)
        x = schedule.p_sample_step(model, x, t_batch, condition)
    return x.reshape(num_samples, steps, 2)


def sample_unconditional(model, schedule, steps, num_samples=8, device="cpu", seed=0):
    """無条件生成（第09章）。"""
    return sample(model, schedule, steps, num_samples, condition=None, device=device, seed=seed)


def sample_conditional(model, schedule, steps, start_positions, device="cpu", seed=0):
    """条件付き生成（初期位置を指定、第10章）。"""
    return sample(model, schedule, steps, num_samples=start_positions.shape[0],
                  condition=start_positions.to(device), device=device, seed=seed)
