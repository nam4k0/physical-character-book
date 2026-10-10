"""ロールアウトバッファ。

第02章（PPO）で説明したとおり、RL は「ステップごとに更新」ではなく
「一定量を貯めてからまとめて更新」する。その貯蔵庫がこれ。

AMP 用に状態遷移も保持する点が本論文の実装上のポイント（第24章）。
"""

from __future__ import annotations

import torch


class RolloutBuffer:
    """PPO と AMP に必要なデータを蓄積する。

    保持するもの:
      - obs, goal, actions   : 方策の入出力
      - log_probs, values    : PPO の更新に必要
      - rewards, dones       : 報酬と終了フラグ
      - amp_obs              : 識別器用の状態遷移 (s_t, s_{t+1}) を平坦化したもの
    """

    def __init__(self, num_steps: int, num_envs: int, obs_dim: int, goal_dim: int,
                 action_dim: int, amp_obs_dim: int, device: str = "cpu"):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = device

        self.obs = torch.zeros(num_steps, num_envs, obs_dim, device=device)
        self.goal = torch.zeros(num_steps, num_envs, goal_dim, device=device)
        self.actions = torch.zeros(num_steps, num_envs, action_dim, device=device)
        self.log_probs = torch.zeros(num_steps, num_envs, device=device)
        self.values = torch.zeros(num_steps, num_envs, device=device)
        self.rewards = torch.zeros(num_steps, num_envs, device=device)
        self.dones = torch.zeros(num_steps, num_envs, device=device)
        self.amp_obs = torch.zeros(num_steps, num_envs, amp_obs_dim, device=device)
        self.next_obs = torch.zeros(num_steps, num_envs, obs_dim, device=device)

    def add(self, t: int, obs, goal, actions, log_probs, values, rewards, dones, amp_obs, next_obs):
        self.obs[t] = obs
        self.goal[t] = goal
        self.actions[t] = actions
        self.log_probs[t] = log_probs
        self.values[t] = values
        self.rewards[t] = rewards
        self.dones[t] = dones
        self.amp_obs[t] = amp_obs
        self.next_obs[t] = next_obs

    def flatten(self):
        """(steps, envs, ...) -> (steps*envs, ...) に平坦化する。"""
        flat = lambda x: x.reshape(-1, *x.shape[2:])
        return {
            "obs": flat(self.obs),
            "goal": flat(self.goal),
            "actions": flat(self.actions),
            "log_probs": flat(self.log_probs),
            "values": flat(self.values),
            "rewards": flat(self.rewards),
            "dones": flat(self.dones),
            "amp_obs": flat(self.amp_obs),
            "next_obs": flat(self.next_obs),
        }
