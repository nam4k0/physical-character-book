"""拡散モデルの学習（付録B）。

DDPM の学習は非常に単純:

1. x_0（本物の軌道）をサンプル
2. 時刻 t をランダムに選ぶ
3. ノイズ eps をサンプルし、x_t = sqrt(a_bar) x_0 + sqrt(1-a_bar) eps を作る
4. モデルに eps を予測させ、二乗誤差を最小化する

これだけ。敵対的学習のように2つのネットワークを競わせる必要がない
（AMP との対比点。付録B）。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .dataset import make_dataset
from .model import Denoiser
from .schedule import NoiseSchedule


def train_diffusion(
    num_samples: int = 20000,
    steps: int = 64,
    num_timesteps: int = 200,
    batch_size: int = 128,
    num_epochs: int = 200,
    lr: float = 1e-3,
    condition_on_start: bool = True,
    device: str = "cpu",
    seed: int = 0,
):
    torch.manual_seed(seed)

    data = make_dataset(num_samples, steps, condition_on_start=condition_on_start, seed=seed)
    # 軌道を平坦化して 1 ベクトルにする（steps*2 次元）
    traj = data["traj"].reshape(num_samples, steps * 2).to(device)
    start = data["start"].to(device)
    cond_dim = 2 if condition_on_start else 0

    schedule = NoiseSchedule(num_timesteps, device=device)
    model = Denoiser(data_dim=steps * 2, cond_dim=cond_dim).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    n = traj.shape[0]
    for epoch in range(num_epochs):
        perm = torch.randperm(n, device=device)
        total = 0.0
        count = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            x0 = traj[idx]
            cond = start[idx] if condition_on_start else None

            t = torch.randint(0, num_timesteps, (x0.shape[0],), device=device)
            noise = torch.randn_like(x0)
            x_t = schedule.q_sample(x0, t, noise)

            eps_pred = model(x_t, t, cond)
            loss = mse(eps_pred, noise)

            optim.zero_grad()
            loss.backward()
            optim.step()
            total += loss.item()
            count += 1

        if (epoch + 1) % 20 == 0:
            print(f"[diffusion] epoch {epoch + 1}/{num_epochs} loss={total / count:.4f}")

    return model, schedule, (steps, cond_dim)


if __name__ == "__main__":
    train_diffusion()
