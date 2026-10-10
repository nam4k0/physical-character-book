"""寄り道2: Reacher — 腕制御（第19章）。

アームの先端を目標地点に近づけるタスク。人型でいう
「腕を伸ばして箱に手をやる」能力の最小モデル。carry の前提。

Reacher は「目標位置」を環境が返すので、それを目標 g_t として使う。
状態に目標を含めることで、AMP の識別器にも目標文脈が入る。
"""

from __future__ import annotations

import numpy as np
import torch

from .base import RefMotionEnv


class ReacherEnv(RefMotionEnv):
    """Reacher をラップする。

    gymnasium の Reacher は obs に「関節角＋目標への相対座標」を含む。
    本論文流に、目標（目標地点）を g_t として切り出して明示的に扱う。
    """

    def __init__(self, cfg, device: str = "cpu"):
        super().__init__("Reacher-v5", cfg, device=device)
        # Reacher の obs は「関節角(2) + 指先座標(2) + 目標への相対(2)」等
        self.goal_dim = 2                     # 目標への相対座標
        self.obs_dim = self._base_obs_dim
        self.object_dim = 2                   # 目標（=物体）を物体特徴としても扱う
        self._last_obs = torch.zeros(self.num_envs, self.obs_dim, device=device)

    def _split(self, obs: torch.Tensor):
        """obs から「状態」と「目標（物体）」を切り出す。"""
        # 実装依存のため末尾 2 次元を目標への相対座標とみなす
        state = obs[..., :-2]
        goal = obs[..., -2:]
        return state, goal

    def get_amp_obs(self, obs, goal):
        # シーン条件付けの簡易版: キャラクタ状態に物体（目標）特徴を連結する。
        state, obj = self._split(obs)
        return torch.cat([state, obj], dim=-1)

    def task_reward(self, obs, goal, action, next_obs):
        """指先が目標に近いほど高い（第12章の r^near と同型）。"""
        _, obj_next = self._split(next_obs)
        dist2 = obj_next.pow(2).sum(dim=-1)
        return torch.exp(-10.0 * dist2)
