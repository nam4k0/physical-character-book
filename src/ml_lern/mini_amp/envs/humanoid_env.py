"""本命: 標準 humanoid で歩行を学習（第25章）。

MuJoCo の humanoid.xml を使った人型。物体はまだ置かず、
まずは「AMP で自然に歩く」ところまでを作る。

論文の図2のパイプラインのうち、物体を除いた部分に相当する。
次の章（scene_env.py）で物体を足し、シーン条件付き識別器にする。
"""

from __future__ import annotations

import torch

from .base import RefMotionEnv


class HumanoidLocomotionEnv(RefMotionEnv):
    """MuJoCo Humanoid の歩行環境。

    - 状態: humanoid の関節角・速度など（第09章）
    - 行動: 各関節の目標角（PD制御、第06章）
    - 報酬: 前進速度（第12章の速度項）
    - AMP: 歩行の参照モーション（本実装では手続き的に生成 or サンプル運動）
    """

    def __init__(self, cfg, device: str = "cpu"):
        super().__init__("Humanoid-v5", cfg, device=device)
        self.obs_dim = self._base_obs_dim
        self.goal_dim = 1                      # 目標速度
        self.target_speed = cfg_attr(cfg, "target_speed", 1.5)
        self._last_obs = torch.zeros(self.num_envs, self.obs_dim, device=device)
        self._build_reference_motion()

    def _build_reference_motion(self):
        """参照モーションを用意する（第10章）。

        論文は MoCap（SAMP 等）を使うが、このミニ実装では
        「手続き的に生成した歩行」または「短いクリップ」を参照とする。
        ここでは枠だけ用意し、実データは scripts/prepare_ref.py で差し替える。
        """
        # 例: N フレーム分の参照状態をランダムに生成（ダミー）。
        # 実運用では np.load('reference_walk.npy') などで実データを読む。
        n = 2000
        self.ref_obs = torch.randn(n, self.obs_dim, device=self.device) * 0.1
        self.ref_amp_obs = self.ref_obs.clone()

    def _sample_goal(self, obs):
        return torch.full((self.num_envs, self.goal_dim), self.target_speed, device=self.device)

    def get_amp_obs(self, obs, goal):
        # 物体がないので、キャラクタ状態のみを識別器へ（AMP 本来の形）。
        return obs

    def _forward_speed(self, obs: torch.Tensor) -> torch.Tensor:
        # Humanoid-v5 の obs[0] は root の x 方向線速度。
        return obs[..., 0]

    def task_reward(self, obs, goal, action, next_obs):
        """前進速度への追従 ＋ 生存（第12章）。

        論文のタスク報酬（座る位置など）の代わりに、
        ここでは歩行を学ばせるための速度追従報酬を使う。
        """
        speed = self._forward_speed(next_obs)
        speed_reward = torch.exp(-2.0 * (self.target_speed - speed).pow(2))
        # 直立ボーナス（体幹が立っているほど高い）
        upright = torch.exp(-5.0 * next_obs[..., 1].pow(2)) if next_obs.shape[-1] > 1 else torch.ones_like(speed)
        alive = torch.ones_like(speed_reward)
        return 1.0 * speed_reward + 0.5 * upright + 0.1 * alive


def cfg_attr(cfg, name, default):
    return getattr(cfg, name, default)
