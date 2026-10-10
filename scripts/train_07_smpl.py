#!/usr/bin/env python3
"""本命3: SMPL 人型へ合流（第28章）。

    python scripts/train_07_smpl.py

前提:
  - SMPL モデルと AMASS データをライセンス登録のうえ取得済み
  - SMPLSim で SMPL → MuJoCo XML を生成済み（assets/smpl/humanoid.xml）
  - AMASS のモーションを参照モーションに変換済み

このスクリプトは本リポジトリ（CPU・CUDAなし）では実行しない。
読者の環境で動かす想定。
"""

from ml_lern.mini_amp.train import train

if __name__ == "__main__":
    train("smpl", use_amp=True, overrides={"num_envs": 16, "episode_length_s": 15.0})
