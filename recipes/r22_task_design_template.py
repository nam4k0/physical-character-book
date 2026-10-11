#!/usr/bin/env python3
"""レシピ22: タスク設計テンプレートを作る（第21章）。

    uv run python recipes/r22_task_design_template.py
"""

TEMPLATE = {
    "タスク名": "",
    "物体": "",
    "キャラの動き": "",
    "成功条件": "",
    "難しさの源": "",
}


def main() -> None:
    examples = [
        {"タスク名": "操縦席に着く", "物体": "コックピット",
         "キャラの動き": "歩いて近づき、座る", "成功条件": "腰が座面20cm以内",
         "難しさの源": "座面の高さ・狭さ"},
        {"タスク名": "旗を運ぶ", "物体": "旗（軽い箱）",
         "キャラの動き": "持ち上げて目的地へ", "成功条件": "旗が目標20cm以内",
         "難しさの源": "箱の大きさ・重さ"},
    ]
    for i, ex in enumerate(examples, 1):
        print(f"--- タスク{i} ---")
        for k in TEMPLATE:
            print(f"  {k}: {ex.get(k, '')}")


if __name__ == "__main__":
    main()
