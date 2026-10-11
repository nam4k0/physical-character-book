#!/usr/bin/env python3
"""レシピ35: スタイル重みを変えて比べる（第34章）。

    uv run python recipes/r35_style_weight_compare.py
"""

from ml_lern.core.config import default_config


def main() -> None:
    for w_style in [0.0, 0.2, 0.5, 0.8, 1.0]:
        cfg = default_config(style_reward_weight=w_style)
        print(f"w^S={cfg.style_reward_weight}: タスク重み={cfg.task_reward_weight}, "
              f"勾配ペナルティ={cfg.gradient_penalty_weight}")


if __name__ == "__main__":
    main()
