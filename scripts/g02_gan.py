#!/usr/bin/env python3
"""第07章: GAN（敵対的生成）と勾配ペナルティのデモ。

    uv run python scripts/g02_gan.py

識別器の考え方は、後の AMP（第19章）にそのまま使われる。
"""

from ml_lern.generative.gan import train_gan

if __name__ == "__main__":
    G, D = train_gan()
    print("GAN の学習が完了しました")
