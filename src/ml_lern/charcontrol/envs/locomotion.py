"""寄り道3・4: Hopper / Walker2d / Ant（第20〜21章）。

片足ホッピング（Hopper）、二足歩行（Walker2d）、四足歩行（Ant）。
人型でいう「接地・蹴り出し・歩行」につながる。

いずれも「前進速度」をタスク報酬にする（論文の r^far の速度項と同型）。
ここでは PD 制御された関節に「目標角」を与える。
"""

from __future__ import annotations

import torch

from .base import RefMotionEnv


class ForwardLocomotionEnv(RefMotionEnv):
    """前進を促す locomotion 環境の共通実装。"""

    def __init__(self, env_id: str, cfg, device: str = "cpu"):
        super().__init__(env_id, cfg, device=device)
        self.obs_dim = self._base_obs_dim
        self.goal_dim = 1                      # 目標速度
        self.target_speed = 1.5                # v* = 1.5 m/s（論文 §付録A）
        self._last_obs = torch.zeros(self.num_envs, self.obs_dim, device=device)

    def _sample_goal(self, obs):
        return torch.full((self.num_envs, self.goal_dim), self.target_speed, device=self.device)

    def get_amp_obs(self, obs, goal):
        return obs

    def _forward_speed(self, obs: torch.Tensor) -> torch.Tensor:
        """obs から前進速度成分を取り出す（環境により index が異なる）。

        MuJoCo locomotion 系では obs[0] が root の x 速度であることが多い。
        """
        return obs[..., 0]

    def task_reward(self, obs, goal, action, next_obs):
        """目標速度への追従（第25章の速度項と同型）。

        + 生存ボーナス。速すぎ・遅すぎは減点。
        """
        speed = self._forward_speed(next_obs)
        speed_reward = torch.exp(-2.0 * (self.target_speed - speed).pow(2))
        alive = torch.ones_like(speed_reward)
        return 1.0 * speed_reward + 0.1 * alive


class HopperEnv(ForwardLocomotionEnv):
    """片足ホッピング。接地と蹴り出しを学ぶ（第31章）。"""

    def __init__(self, cfg, device: str = "cpu"):
        super().__init__("Hopper-v5", cfg, device=device)


class Walker2dEnv(ForwardLocomotionEnv):
    """二足歩行。人型歩行に最も近い寄り道（第32章）。"""

    def __init__(self, cfg, device: str = "cpu"):
        super().__init__("Walker2d-v5", cfg, device=device)


class AntEnv(ForwardLocomotionEnv):
    """四足歩行。多脚の協調を学ぶ（第32章）。"""

    def __init__(self, cfg, device: str = "cpu"):
        super().__init__("Ant-v5", cfg, device=device)
