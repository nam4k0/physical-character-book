"""ニューラルネットワークの定義。

論文 §6 / 第23章 / 第24章 の構成に対応する。

- 方策 π       : 状態 s と目標 g → 行動分布 N(μ, Σ) の平均 μ を出力（Σ は固定）
- 価値関数 V    : 状態 → 状態価値（スカラー）
- 識別器 D      : 状態遷移 (s, s') → 本物らしさ（スカラー、勾配ペナルティ付き）

いずれも {1024, 512} の全結合 MLP ＋ ReLU。
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _build_mlp(in_dim: int, hidden_dims: tuple[int, ...], out_dim: int) -> nn.Sequential:
    """全結合＋ReLU の MLP を組む。"""
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.ReLU())
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class Policy(nn.Module):
    """ガウス方策 π(a | s, g) = N(μ(s, g), Σ)。

    第12章で述べたとおり、行動は PD 制御への「目標関節角」。
    共分散 Σ は学習せず固定（第24章）。これにより探索量が一定になり安定する。
    """

    def __init__(self, obs_dim: int, goal_dim: int, action_dim: int,
                 hidden_dims: tuple[int, ...] = (1024, 512), action_std: float = 0.25):
        super().__init__()
        self.net = _build_mlp(obs_dim + goal_dim, hidden_dims, action_dim)
        # 固定の標準偏差（log 空間で保持し、exp して使う）
        self.register_buffer("log_std", torch.full((action_dim,), float(torch.log(torch.tensor(action_std)))))
        self.action_dim = action_dim

    def forward(self, obs: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """平均 μ を返す。"""
        x = torch.cat([obs, goal], dim=-1)
        return self.net(x)

    def distribution(self, obs: torch.Tensor, goal: torch.Tensor) -> torch.distributions.Normal:
        """行動分布 N(μ, Σ) を返す。"""
        mu = self.forward(obs, goal)
        std = torch.exp(self.log_std).expand_as(mu)
        return torch.distributions.Normal(mu, std)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, goal: torch.Tensor):
        """行動をサンプルし、対数確率も返す（ロールアウト用）。"""
        dist = self.distribution(obs, goal)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob

    def evaluate_actions(self, obs: torch.Tensor, goal: torch.Tensor, actions: torch.Tensor):
        """与えられた行動の対数確率とエントロピーを返す（PPO 更新用）。"""
        dist = self.distribution(obs, goal)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy


class Value(nn.Module):
    """状態価値関数 V(s)。"""

    def __init__(self, obs_dim: int, goal_dim: int, hidden_dims: tuple[int, ...] = (1024, 512)):
        super().__init__()
        self.net = _build_mlp(obs_dim + goal_dim, hidden_dims, 1)

    def forward(self, obs: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, goal], dim=-1)
        return self.net(x).squeeze(-1)


class Discriminator(nn.Module):
    """シーン条件付き識別器 D(s_t, s_{t+1})（第22章）。

    AMP（第19章）の識別器と同じ構造だが、入力 s に
    物体特徴が含まれる点が本論文のポイント（第36章）。
    """

    def __init__(self, amp_obs_dim: int, hidden_dims: tuple[int, ...] = (1024, 512)):
        super().__init__()
        self.net = _build_mlp(amp_obs_dim, hidden_dims, 1)

    def forward(self, amp_obs: torch.Tensor) -> torch.Tensor:
        """ロジット（シグモイド前の生スコア）を返す。"""
        return self.net(amp_obs).squeeze(-1)

    @staticmethod
    def gradient_penalty(logits: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        """勾配ペナルティ（式(5)）。

        識別器の出力が入力のわずかな変化で激しく変わらないようにする。
        (第7章の安定化テクニック)
        """
        grads = torch.autograd.grad(
            outputs=logits.sum(), inputs=inputs,
            create_graph=True, retain_graph=True, only_inputs=True,
        )[0]
        return grads.pow(2).sum(dim=-1).mean()
