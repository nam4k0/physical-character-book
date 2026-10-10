"""モーション生成への拡張（第10章）。

2D バウンスボール（第09章）を、モーションへ拡張する。
データの形が (ステップ, 2) から (フレーム, 関節×特徴) に変わるだけで、
拡散モデルの仕組みは同じ。

ここでは手続き的に生成した「関節の軌道」をデータとし、
初期姿勢を条件にモーションを生成する（inpainting 風、第10章）。
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from .model import Denoiser
from .schedule import NoiseSchedule


def make_motion_dataset(
    num_samples: int = 4000,
    frames: int = 32,
    num_joints: int = 4,
    seed: int = 0,
) -> dict:
    """手続き生成のモーション（関節の軌道）を作る（第10章）。

    各関節が少しずつ違う位相・振幅で振動する、という単純な「動き」を生成する。
    実データ（MoCap）の代わりに、構造を確認するためのダミー。

    戻り値:
      motion (N, frames, num_joints*3), first_pose (N, num_joints*3)
    """
    rng = np.random.default_rng(seed)
    feat = num_joints * 3
    motion = np.zeros((num_samples, frames, feat), dtype=np.float32)
    for i in range(num_samples):
        amp = rng.uniform(0.2, 1.0, size=feat)
        phase = rng.uniform(0, 2 * math.pi, size=feat)
        freq = rng.uniform(0.1, 0.5, size=feat)
        tt = np.arange(frames, dtype=np.float32)
        motion[i] = amp * np.sin(2 * math.pi * freq * tt[:, None] + phase[None, :])
    first_pose = motion[:, 0, :].copy()
    return {
        "motion": torch.from_numpy(motion),           # (N, frames, feat)
        "first_pose": torch.from_numpy(first_pose),   # (N, feat)
    }


def train_motion_diffusion(
    num_samples: int = 4000,
    frames: int = 32,
    num_joints: int = 4,
    num_timesteps: int = 100,
    batch_size: int = 128,
    num_epochs: int = 150,
    lr: float = 1e-3,
    device: str = "cpu",
    seed: int = 0,
):
    """モーションの拡散モデルを学習する（第10章）。

    条件 = 初期姿勢（第1フレーム）。inpainting により、
    初期姿勢を固定して続きを生成できるようにする。
    """
    torch.manual_seed(seed)
    data = make_motion_dataset(num_samples, frames, num_joints, seed)
    feat = num_joints * 3
    motion = data["motion"].reshape(num_samples, frames * feat).to(device)
    first_pose = data["first_pose"].to(device)

    schedule = NoiseSchedule(num_timesteps, device=device)
    model = Denoiser(data_dim=frames * feat, cond_dim=feat).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    n = motion.shape[0]
    for epoch in range(num_epochs):
        perm = torch.randperm(n, device=device)
        total, count = 0.0, 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            x0 = motion[idx]
            cond = first_pose[idx]
            t = torch.randint(0, num_timesteps, (x0.shape[0],), device=device)
            noise = torch.randn_like(x0)
            x_t = schedule.q_sample(x0, t, noise)
            loss = mse(model(x_t, t, cond), noise)
            optim.zero_grad(); loss.backward(); optim.step()
            total += loss.item(); count += 1
        if (epoch + 1) % 30 == 0:
            print(f"[motion-diffusion] epoch {epoch + 1}/{num_epochs} loss={total / count:.4f}")

    return model, schedule, (frames, feat)


@torch.no_grad()
def sample_motion(model, schedule, frames: int, feat: int,
                  first_pose: torch.Tensor, device: str = "cpu", seed: int = 0) -> torch.Tensor:
    """初期姿勢を条件にモーションを生成する（第10章、inpainting 風）。"""
    torch.manual_seed(seed)
    n = first_pose.shape[0]
    cond = first_pose.to(device)
    x = torch.randn(n, frames * feat, device=device)
    for t in reversed(range(schedule.num_timesteps)):
        t_batch = torch.full((n,), t, dtype=torch.long, device=device)
        x = schedule.p_sample_step(model, x, t_batch, cond)
        # inpainting: 第1フレームは与えられた姿勢で上書きする
        x = x.reshape(n, frames, feat)
        x[:, 0, :] = cond
        x = x.reshape(n, frames * feat)
    return x.reshape(n, frames, feat)
