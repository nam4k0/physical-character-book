#!/usr/bin/env python3
"""レシピ15: グリッド世界を自作して報酬を設計する（第14章）。

    uv run python recipes/r15_gridworld.py
"""

import random


class GridWorld:
    def __init__(self, size: int = 5):
        self.size, self.goal = size, (size - 1, size - 1)
        self.holes = {(1, 1), (3, 2)}
        self.pos = (0, 0)

    def reset(self):
        self.pos = (0, 0)
        return self.pos

    def step(self, a):                          # 0上 1右 2下 3左
        x, y = self.pos
        dx, dy = [(-1, 0), (0, 1), (1, 0), (0, -1)][a]
        self.pos = (min(max(x + dx, 0), self.size - 1), min(max(y + dy, 0), self.size - 1))
        if self.pos in self.holes:
            return self.pos, -1.0, True         # 穴: 大罰
        if self.pos == self.goal:
            return self.pos, +1.0, True         # ゴール
        return self.pos, -0.01, False           # 毎ステップ小罰

    def actions(self) -> int:
        return 4


def main() -> None:
    env = GridWorld()
    for _ in range(3):
        env.reset()
        total = 0.0
        for _ in range(30):
            _, r, done = env.step(random.randrange(env.actions()))
            total += r
            if done:
                break
        print("1エピソードの収益:", round(total, 2))


if __name__ == "__main__":
    main()
