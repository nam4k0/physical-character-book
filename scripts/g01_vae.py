#!/usr/bin/env python3
"""第06章: オートエンコーダと VAE のデモ。

    uv run python scripts/g01_vae.py
"""

import argparse

from ml_lern.generative.autoencoder import train_ae

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["ae", "vae"], default="vae")
    args = parser.parse_args()
    train_ae(args.kind)
