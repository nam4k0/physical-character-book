#!/usr/bin/env python3
"""レシピ16: 多腕バンディットを方策勾配で解く（第15章）。

    uv run python recipes/r16_bandit_policy_gradient.py
"""

import torch


def main() -> None:
    true_rewards = torch.tensor([0.1, 0.9, 0.3, 0.2, 0.5])   # 各腕の当たりやすさ
    probs = torch.ones(5) / 5                                # 選択確率（方策）
    lr = 0.1

    for step in range(2000):
        arm = torch.multinomial(probs, 1).item()
        r = true_rewards[arm].item()
        grad = torch.zeros(5)
        grad[arm] = r - true_rewards.mean()                  # 平均より良い腕を強める
        probs = torch.softmax(torch.log(probs) + lr * grad, dim=0)
        if step % 500 == 0:
            print(step, probs.round(2).tolist())
    print("最終:", probs.round(2).tolist())                  # 腕1が最大になる


if __name__ == "__main__":
    main()
