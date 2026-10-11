#!/usr/bin/env python3
"""寄り道1: 倒立振子で体幹バランスを学ぶ（第18章）。

    python scripts/train_01_pendulum.py

AMP は使わず、まず素の PPO だけを回して RL のループに慣れる。
人型の「体幹を保つ」能力の最小モデル。
"""

from ml_lern.charcontrol.train import train

if __name__ == "__main__":
    train("pendulum", use_amp=False, overrides={"num_envs": 16, "episode_length_s": 10.0})
