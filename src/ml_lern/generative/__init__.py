"""生成モデル（第2部）。

- autoencoder.py : AE / VAE（第06章）
- gan.py         : GAN と勾配ペナルティ（第07章）
- diffusion/     : 拡散モデル（第08〜10章）
"""

from .autoencoder import Autoencoder, VAE, train_ae
from .gan import Discriminator, Generator, gradient_penalty, train_gan

__all__ = [
    "Autoencoder",
    "VAE",
    "train_ae",
    "Generator",
    "Discriminator",
    "gradient_penalty",
    "train_gan",
]
