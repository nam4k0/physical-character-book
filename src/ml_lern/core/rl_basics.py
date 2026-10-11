"""強化学習の基本（第14〜15章）。

- ReplayBuffer          : 遷移のバッファ（第14章）
- discounted_returns    : 収益の計算（第14章）
- compute_gae           : GAE（第15章）
- value_loss            : 価値関数の損失（第15章）
- policy_gradient_loss  : 素朴な方策勾配の損失（第15章・比較用）
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


@dataclass
class ReplayBuffer:
    """(s, a, r, s', done) を貯めるバッファ（第14章）。"""

    states: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    next_states: list = field(default_factory=list)
    dones: list = field(default_factory=list)

    def add(self, s, a, r, s_next, done):
        self.states.append(s)
        self.actions.append(a)
        self.rewards.append(r)
        self.next_states.append(s_next)
        self.dones.append(done)

    def __len__(self) -> int:
        return len(self.rewards)


def discounted_returns(rewards: list[float], gamma: float = 0.99) -> torch.Tensor:
    """割引収益 G_t を計算する（第14章）。"""
    returns = torch.zeros(len(rewards))
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        returns[t] = running
    return returns


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    last_value: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GAE でアドバンテージと収益を計算する（第15章）。

    入出力は 1次元（時間方向）を想定。
    """
    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    last_gae = 0.0
    for t in reversed(range(T)):
        next_value = last_value if t == T - 1 else values[t + 1]
        not_done = 1.0 - dones[t].float()
        delta = rewards[t] + gamma * next_value * not_done - values[t]
        last_gae = delta + gamma * gae_lambda * not_done * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


def value_loss(values: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
    """価値関数の回帰損失（第15章）。"""
    return F.mse_loss(values, returns)


def policy_gradient_loss(log_probs: torch.Tensor, advantages: torch.Tensor) -> torch.Tensor:
    """素朴な方策勾配の損失（第15章、比較用）。

    advantages を正規化して安定化する。
    """
    adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return -(log_probs * adv.detach()).mean()


def demo() -> None:
    """GAE と収益の動作確認（第14〜15章）。"""
    torch.manual_seed(0)
    T = 10
    rewards = torch.rand(T)
    values = torch.rand(T)
    dones = torch.zeros(T)

    returns = discounted_returns(rewards.tolist(), gamma=0.99)
    adv, ret = compute_gae(rewards, values, dones)
    print("割引収益の例:", returns[:3].round(3).tolist())
    print("GAE アドバンテージの例:", adv[:3].round(3).tolist())
    print("収益（価値関数の教師）の例:", ret[:3].round(3).tolist())


if __name__ == "__main__":
    demo()
