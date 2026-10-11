#!/usr/bin/env python3
"""レシピ08: GAN で本物らしいデータを作る（第07章）。

    uv run python recipes/r08_gan_2d.py
"""

import torch

from ml_lern.generative.gan import Discriminator, Generator, gradient_penalty


def main() -> None:
    centers = torch.tensor([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]])
    real_data = centers[torch.randint(0, 4, (2000,))] + 0.1 * torch.randn(2000, 2)

    g = Generator(z_dim=16, out_dim=2)
    d = Discriminator(in_dim=2)
    opt_g = torch.optim.Adam(g.parameters(), lr=2e-4, betas=(0.5, 0.9))
    opt_d = torch.optim.Adam(d.parameters(), lr=2e-4, betas=(0.5, 0.9))
    bce = torch.nn.BCEWithLogitsLoss()

    for step in range(3000):
        real = real_data[torch.randint(0, len(real_data), (256,))]
        fake = g(torch.randn(256, 16)).detach()
        loss_d = bce(d(real), torch.ones(256)) + bce(d(fake), torch.zeros(256))
        loss_d = loss_d + 10.0 * gradient_penalty(d, real)
        opt_d.zero_grad(); loss_d.backward(); opt_d.step()

        fake = g(torch.randn(256, 16))
        loss_g = bce(d(fake), torch.ones(256))
        opt_g.zero_grad(); loss_g.backward(); opt_g.step()

        if step % 500 == 0:
            acc = (torch.sigmoid(d(real)) > 0.5).float().mean().item()
            print(f"step {step}: lossD={loss_d.item():.3f} acc_real={acc:.2f}")


if __name__ == "__main__":
    main()
