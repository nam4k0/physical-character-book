"""寄り道1: 倒立振子（InvertedPendulum）— 体幹バランス（第18章）。

台車の上で棒を立て続けるタスク。人型でいう「体幹を起こして
バランスを取る」能力の最小モデル。立つ・座るときの姿勢維持に通じる。
"""

from __future__ import annotations

import torch

from .base import RefMotionEnv


class InvertedPendulumEnv(RefMotionEnv):
    """倒立振子をラップする。

    - 行動: 台車を押す力（1次元）。本実装では PD 制御は使わず直接力。
    - 報酬: 棒が立っている（角度が小さい）ほど高い。第12章の exp(-k d^2) の形。
    """

    def __init__(self, cfg, device: str = "cpu"):
        super().__init__("InvertedPendulum-v5", cfg, device=device)
        self.obs_dim = self._base_obs_dim
        self.goal_dim = 0
        self._last_obs = torch.zeros(self.num_envs, self.obs_dim, device=device)

    def get_amp_obs(self, obs, goal):
        # 物体なし。キャラクタ（台車＋棒）の状態のみを識別器へ。
        return obs

    def task_reward(self, obs, goal, action, next_obs):
        """棒の角度（obs の該当成分）が小さいほど高い報酬。

        MuJoCo の InvertedPendulum は obs に棒の角度情報を含む。
        傾きが大きいほど exp(-k x^2) が小さくなる（第12章の誘導と同型）。
        """
        pole_angle = obs[..., 1]
        upright = torch.exp(-5.0 * pole_angle.pow(2))
        alive = torch.ones_like(upright)
        return 1.0 * upright + 0.1 * alive
