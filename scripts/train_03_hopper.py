#!/usr/bin/env python3
"""寄り道3: Hopper で接地と蹴り出しを学ぶ（第20章）。

    python scripts/train_03_hopper.py
"""

from ml_lern.mini_amp.train import train

if __name__ == "__main__":
    train("hopper", use_amp=False, overrides={"num_envs": 16})
