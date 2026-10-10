"""PPO の更新処理（第02章）。

方策（クリップ付き目的関数）と価値関数を更新する。
AMP のスタイル報酬は、この更新の前に「報酬」へ混ぜておく（rollout.py 参照）。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .models import Policy, Value


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    gae_lambda: float,
    last_value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GAE（Generalized Advantage Estimation）でアドバンテージと収益を計算する。

    引数は (num_steps, num_envs) の形。
    第02章の擬似コードと同じく、時間方向に逆順で走査する。
    """
    num_steps, num_envs = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(num_envs, device=rewards.device)

    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            next_value = last_value
        else:
            next_value = values[t + 1]
        # 終了した環境は次ステップの価値を 0 とみなす
        not_done = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * not_done - values[t]
        last_gae = delta + gamma * gae_lambda * not_done * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


class PPOUpdater:
    """方策と価値関数の更新（第02章）。"""

    def __init__(self, obs_dim: int, goal_dim: int, action_dim: int, cfg, device: str = "cpu"):
        self.cfg = cfg
        self.device = device
        self.policy = Policy(obs_dim, goal_dim, action_dim, cfg.hidden_dims, cfg.action_std).to(device)
        self.value = Value(obs_dim, goal_dim, cfg.hidden_dims).to(device)
        self.optimizer = torch.optim.Adam(
            list(self.policy.parameters()) + list(self.value.parameters()),
            lr=cfg.learning_rate,
        )

    def update(self, batch: dict, advantages: torch.Tensor, returns: torch.Tensor) -> dict:
        cfg = self.cfg
        obs = batch["obs"]
        goal = batch["goal"]
        actions = batch["actions"]
        old_log_probs = batch["log_probs"]
        old_values = batch["values"]

        # アドバンテージを正規化（学習を安定させる定石）
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = returns.to(self.device)

        n = obs.shape[0]
        minibatch_size = max(1, n // cfg.num_minibatches)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        updates = 0

        # 同じデータを K 周する（第02章）。クリップがあるから安全。
        for _ in range(cfg.num_epochs):
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, minibatch_size):
                mb = perm[start:start + minibatch_size]
                mb_obs = obs[mb]
                mb_goal = goal[mb]
                mb_actions = actions[mb]
                mb_old_log_probs = old_log_probs[mb]
                mb_advantages = advantages[mb]
                mb_returns = returns[mb]

                # --- 方策の損失（クリップ付き目的関数）---
                log_probs, entropy = self.policy.evaluate_actions(mb_obs, mb_goal, mb_actions)
                ratio = torch.exp(log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - cfg.clip_epsilon, 1 + cfg.clip_epsilon) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # --- 価値関数の損失 ---
                values = self.value(mb_obs, mb_goal)
                value_loss = ((values - mb_returns) ** 2).mean()

                # --- エントロピー（探索の促進）---
                entropy_loss = -entropy.mean()

                loss = (
                    policy_loss
                    + cfg.value_loss_coef * value_loss
                    + cfg.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.policy.parameters()) + list(self.value.parameters()),
                    cfg.max_grad_norm,
                )
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                updates += 1

        # 参考指標: 方策がどれだけ変わったか（クリップ前の比の平均）
        with torch.no_grad():
            new_log_probs, _ = self.policy.evaluate_actions(obs, goal, actions)
            approx_kl = (old_log_probs - new_log_probs).mean().item()

        return {
            "policy_loss": total_policy_loss / max(1, updates),
            "value_loss": total_value_loss / max(1, updates),
            "entropy": total_entropy / max(1, updates),
            "approx_kl": approx_kl,
        }
