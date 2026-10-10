"""オートエンコーダと VAE（第06章）。

- Autoencoder : 圧縮→復元（潜在 z に分布の制約なし）
- VAE         : 潜在を正規分布に整える（再パラメータ化トリック）

データは「2D の分布（例: 8つのガウス）」で動作確認する。
モーションでも形を変えるだけで同じ実装が使える。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    """オートエンコーダ（第06章）。"""

    def __init__(self, in_dim: int, latent: int = 16, hidden: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, latent),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, hidden), nn.ReLU(),
            nn.Linear(hidden, in_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


class VAE(nn.Module):
    """変分オートエンコーダ（第06章）。

    エンコーダは潜在の平均 mu と log 分散 logvar を出力し、
    再パラメータ化トリックで z をサンプルする。
    """

    def __init__(self, in_dim: int, latent: int = 16, hidden: int = 128):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, latent)
        self.logvar = nn.Linear(hidden, latent)
        self.decoder = nn.Sequential(
            nn.Linear(latent, hidden), nn.ReLU(),
            nn.Linear(hidden, in_dim),
        )

    def encode(self, x: torch.Tensor):
        h = self.enc(x)
        return self.mu(h), self.logvar(h)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """z = mu + sigma * eps（第06章）。"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar, z


def vae_loss(x_hat, x, mu, logvar, beta: float = 1.0):
    """VAE の損失 = 復元誤差 + beta * KL（第06章）。"""
    recon = nn.functional.mse_loss(x_hat, x, reduction="mean")
    # KL(N(mu, sigma^2) || N(0,1)) の閉形式
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta * kl, recon, kl


def _make_2d_data(n: int = 4000, seed: int = 0) -> torch.Tensor:
    """8つのガウス分布からなる2Dデータ（生成モデルの練習用）。"""
    g = torch.Generator().manual_seed(seed)
    centers = torch.tensor([[1., 0.], [-1., 0.], [0., 1.], [0., -1.],
                            [0.7, 0.7], [-0.7, 0.7], [0.7, -0.7], [-0.7, -0.7]])
    idx = torch.randint(0, len(centers), (n,), generator=g)
    noise = 0.1 * torch.randn(n, 2, generator=g)
    return centers[idx] + noise


def train_ae(kind: str = "vae", steps: int = 2000, lr: float = 1e-3) -> nn.Module:
    """AE または VAE を学習する（第06章）。`kind` は "ae" or "vae"。"""
    torch.manual_seed(0)
    data = _make_2d_data()
    if kind == "vae":
        model = VAE(in_dim=2, latent=8)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        for step in range(steps):
            x_hat, mu, logvar, z = model(data)
            loss, recon, kl = vae_loss(x_hat, data, mu, logvar)
            opt.zero_grad(); loss.backward(); opt.step()
            if step % 500 == 0:
                print(f"[VAE] step {step:4d} loss={loss.item():.4f} recon={recon.item():.4f} kl={kl.item():.4f}")
    else:
        model = Autoencoder(in_dim=2, latent=8)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        for step in range(steps):
            x_hat, z = model(data)
            loss = nn.functional.mse_loss(x_hat, data)
            opt.zero_grad(); loss.backward(); opt.step()
            if step % 500 == 0:
                print(f"[AE] step {step:4d} loss={loss.item():.4f}")
    return model


if __name__ == "__main__":
    train_ae("vae")
