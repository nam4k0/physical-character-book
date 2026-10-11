#!/usr/bin/env python3
"""第16・17章: PPO で倒立振子を学習する。

    uv run python scripts/p03_ppo.py

MuJoCo 環境（倒立振子）で、PPO の学習が回ることを確認する。
AMP は使わない（スタイル報酬なし）。
"""

from ml_lern.charcontrol.train import train

if __name__ == "__main__":
    train("pendulum", use_amp=False, overrides={"num_envs": 16, "episode_length_s": 10.0})
