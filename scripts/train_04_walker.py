#!/usr/bin/env python3
"""寄り道4: Walker2d / Ant で歩行を学ぶ（第21章）。

    python scripts/train_04_walker.py
    python scripts/train_04_walker.py --ant
"""

import argparse

from ml_lern.mini_amp.train import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ant", action="store_true", help="四足（Ant）を使う")
    args = parser.parse_args()
    stage = "ant" if args.ant else "walker"
    train(stage, use_amp=False, overrides={"num_envs": 16})
