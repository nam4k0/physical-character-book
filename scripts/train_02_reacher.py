#!/usr/bin/env python3
"""寄り道2: Reacher で腕制御を学ぶ（第19章）。

    python scripts/train_02_reacher.py

目標（物体）を状態に含める「シーン条件付けの入り口」を体験する。
"""

from ml_lern.charcontrol.train import train

if __name__ == "__main__":
    train("reacher", use_amp=False, overrides={"num_envs": 16})
