#!/usr/bin/env python3
"""本命1: 標準 humanoid で AMP を統合する（第25章）。

    python scripts/train_05_humanoid_amp.py

スタイル報酬（AMP）を加えて、自然な歩行を学ばせる。
"""

from ml_lern.charcontrol.train import train

if __name__ == "__main__":
    train("humanoid_amp", use_amp=True, overrides={"num_envs": 16, "episode_length_s": 10.0})
