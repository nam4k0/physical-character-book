"""GAN と勾配ペナルティ（第07章）。

生成器（偽物を作る）と識別器（本物/偽物を見分ける）を競わせる。
識別器は後の AMP（第19章）にそのまま使われる。

データは 2D の8つのガウス分布で動作確認する。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Generator(nn.Module):
    """生成器。乱数 z → データを生成する（第07章）。"""

    def __init__(self, z_dim: int = 16, out_dim: int = 2, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Discriminator(nn.Module):
    """識別器。入力が本物であるロジットを出力する（第07章）。"""

    def __init__(self, in_dim: int = 2, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)      # ロジット


def gradient_penalty(discriminator: Discriminator, x: torch.Tensor) -> torch.Tensor:
    """勾配ペナルティ（第07章・論文本体の式(5)）。

    識別器の出力が入力の変化で激しく変わらないように平滑化する。
    """
    x = x.clone().detach().requires_grad_(True)
    logits = discriminator(x)
    grads = torch.autograd.grad(
        outputs=logits.sum(), inputs=x,
        create_graph=True, retain_graph=True, only_inputs=True,
    )[0]
    return grads.pow(2).sum(dim=-1).mean()


def _make_2d_data(n: int = 2000, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    centers = torch.tensor([[1., 0.], [-1., 0.], [0., 1.], [0., -1.],
                            [0.7, 0.7], [-0.7, 0.7], [0.7, -0.7], [-0.7, -0.7]])
    idx = torch.randint(0, len(centers), (n,), generator=g)
    return centers[idx] + 0.1 * torch.randn(n, 2, generator=g)


def train_gan(steps: int = 3000, batch: int = 256, lr: float = 2e-4,
              gp_weight: float = 10.0, z_dim: int = 16) -> tuple[Generator, Discriminator]:
    """GAN を学習する（第07章）。勾配ペナルティ付き。"""
    torch.manual_seed(0)
    data = _make_2d_data()

    G = Generator(z_dim=z_dim)
    D = Discriminator(in_dim=2)
    opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.9))
    opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.9))
    bce = nn.BCEWithLogitsLoss()

    for step in range(steps):
        # --- 識別器の更新 ---
        idx = torch.randint(0, len(data), (batch,))
        real = data[idx]
        z = torch.randn(batch, z_dim)
        fake = G(z).detach()
        loss_real = bce(D(real), torch.ones(batch))
        loss_fake = bce(D(fake), torch.zeros(batch))
        gp = gradient_penalty(D, real)
        loss_D = loss_real + loss_fake + gp_weight * gp
        opt_D.zero_grad(); loss_D.backward(); opt_D.step()

        # --- 生成器の更新 ---
        z = torch.randn(batch, z_dim)
        fake = G(z)
        loss_G = bce(D(fake), torch.ones(batch))    # 本物と言わせたい
        opt_G.zero_grad(); loss_G.backward(); opt_G.step()

        if step % 500 == 0:
            acc_r = (torch.sigmoid(D(real)) > 0.5).float().mean().item()
            print(f"[GAN] step {step:4d} loss_D={loss_D.item():.3f} "
                  f"loss_G={loss_G.item():.3f} acc_real={acc_r:.2f}")

    return G, D


if __name__ == "__main__":
    train_gan()
