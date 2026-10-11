#!/usr/bin/env python3
"""レシピ17: 自作の簡単なタスクで PPO を回す（第16章）。

    uv run python recipes/r17_ppo_toy.py

既定では設定の確認のみ。実際に学習するには下の train 呼び出しを有効にする。
"""

from ml_lern.charcontrol.train import train
from ml_lern.core.config import default_config


def main() -> None:
    # clip_epsilon（更新幅）を変えて、学習の安定さを観察する
    for eps in [0.05, 0.2, 0.5]:
        cfg = default_config(clip_epsilon=eps, num_envs=16, log_interval=5)
        print(f"clip_epsilon={cfg.clip_epsilon}, num_envs={cfg.num_envs}")

    # 実学習（時間がかかるので既定ではコメントアウト）
    # train("pendulum", use_amp=False, overrides={"clip_epsilon": 0.2, "num_envs": 16})
    print("train() の stage を自作環境へ向ければ、自分のタスクを PPO で学習できる")


if __name__ == "__main__":
    main()
