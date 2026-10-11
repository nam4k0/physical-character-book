#!/usr/bin/env python3
"""本命2: 物体を足してシーン相互作用（第26〜27章）。

    python scripts/train_06_scene.py

シーン条件付き識別器と物体ランダム化を有効にする（論文の中核）。
"""

from ml_lern.charcontrol.train import train

if __name__ == "__main__":
    train("scene", use_amp=True, overrides={"num_envs": 16, "episode_length_s": 15.0})
