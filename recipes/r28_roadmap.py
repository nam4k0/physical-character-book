#!/usr/bin/env python3
"""レシピ28: 自分のロードマップを作る（第27章）。

    uv run python recipes/r28_roadmap.py
"""

STAGES = [
    ("ミニ実装(CPU)", "倒立振子〜Walker", "第29〜32章", "数分〜数時間"),
    ("GPU 導入", "humanoid の AMP", "第35章", "数時間"),
    ("シーン条件付け", "物体との相互作用", "第36章", "数時間〜"),
    ("IsaacGym", "論文の本実装", "第39章", "数日"),
]


def main() -> None:
    has_gpu = False   # この環境は False。自分の環境に合わせて書き換える
    for name, goal, ch, time in STAGES:
        if "GPU" in name and not has_gpu:
            print(f"[保留] {name}: GPU が必要（{ch}）")
        else:
            print(f"[やる] {name}: {goal} / {ch} / 目安 {time}")


if __name__ == "__main__":
    main()
