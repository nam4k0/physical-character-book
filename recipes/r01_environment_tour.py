#!/usr/bin/env python3
"""レシピ01: 環境をのぞいて、作りたいものをマッピング（第00章）。

    uv run python recipes/r01_environment_tour.py

「物理側」と「AI側」の入口を一度に触り、自分のゲームのタスクを章に割り当てる。
"""

from ml_lern.core.mujoco_basics import inspect_spaces


def main() -> None:
    # 1) 物理側: 各環境の「状態・行動の次元」を確認する
    for env_id in ["InvertedPendulum-v5", "Walker2d-v5", "Humanoid-v5"]:
        try:
            inspect_spaces(env_id)
        except Exception as e:
            print(f"{env_id} はスキップ:", e)

    # 2) AI側: 自動微分の感触
    from ml_lern.foundations.tensors_demo import autograd_demo
    autograd_demo()

    # 3) マッピング: 作りたい動作 -> どこで学ぶか
    tasks = {
        "敵が自然に歩く":       ("物理側", "第32/35章", "RL/AMPで制御"),
        "アニメの補間を自動化": ("AI側",   "第06/10章", "生成モデル"),
        "椅子に座る":           ("物理側", "第36章",    "シーン条件付き識別器"),
    }
    print("\n=== タスク -> 章 のマッピング ===")
    for name, (side, ch, how) in tasks.items():
        print(f"{name}: {side} / {ch} / {how}")


if __name__ == "__main__":
    main()
