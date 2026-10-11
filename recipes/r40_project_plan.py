#!/usr/bin/env python3
"""レシピ40: プロジェクト計画を作る（第39章）。

    uv run python recipes/r40_project_plan.py
"""

PLAN = {
    "作りたいもの": "（例: プレイヤーの周りで自然に動く群衆）",
    "AI側（生成）": "（例: 拡散で歩行バリエーションを生成）",
    "物理側（制御）": "（例: AMPで物理キャラに追従させる）",
    "データ": "（例: 自作MoCap / 公開アセット）",
    "計算資源": "（CPUのみ / GPUあり）",
    "最初の一歩": "（例: 第10章の拡散で1本作る）",
}

COMBOS = [
    "生成 -> 追従: 拡散で作った動きをモーショントラッキングで再生",
    "方策として生成: Diffusion Policy（第10章）",
    "事前分布として: 生成モデルの分布を AMP に使う",
]


def main() -> None:
    for k, v in PLAN.items():
        print(f"{k:14s}: {v}")
    print("\n組み合わせの型:")
    for c in COMBOS:
        print(" -", c)


if __name__ == "__main__":
    main()
