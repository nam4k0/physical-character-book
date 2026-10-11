#!/usr/bin/env python3
"""レシピ07: VAE でモーションを補間する（第06章）。

    uv run python recipes/r07_vae_motion_interp.py
"""

import torch

from ml_lern.generative.autoencoder import VAE, vae_loss


def main() -> None:
    d = 66
    motion = torch.randn(500, d)
    vae = VAE(in_dim=d, latent=8)
    opt = torch.optim.Adam(vae.parameters(), lr=1e-3)
    for _ in range(2000):
        x_hat, mu, logvar, z = vae(motion)
        loss, recon, kl = vae_loss(x_hat, motion, mu, logvar)
        opt.zero_grad()
        loss.backward()
        opt.step()

    # 2つの動きを潜在空間で混ぜる
    mu1 = vae.encode(motion[0])[0]
    mu2 = vae.encode(motion[1])[0]
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        z = (1 - alpha) * mu1 + alpha * mu2
        mid = vae.decoder(z)
        print(f"alpha={alpha}: 中間モーション {tuple(mid.shape)}")


if __name__ == "__main__":
    main()
