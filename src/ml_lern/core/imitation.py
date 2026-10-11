"""模倣学習（第18章）。

- behavior_clone_loss : 行動クローンの損失（第18章）
- tracking_reward     : モーショントラッキングの報酬（第18章）
- ReferenceMotionBuffer : 見本モーションの保持
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


def behavior_clone_loss(log_probs: torch.Tensor) -> torch.Tensor:
    """行動クローンの損失（負の対数尤度、第18章）。

    log_probs は見本の行動に対する方策の対数確率。
    """
    return -log_probs.mean()


def tracking_reward(
    pose: torch.Tensor,
    ref_pose: torch.Tensor,
    vel: torch.Tensor | None = None,
    ref_vel: torch.Tensor | None = None,
    w_pose: float = 0.65,
    w_vel: float = 0.1,
) -> torch.Tensor:
    """見本の姿勢・速度への追従報酬（第18章）。

    exp(-k * ||pose - ref_pose||^2) の形。
    """
    pose_err = (pose - ref_pose).pow(2).sum(dim=-1)
    r = w_pose * torch.exp(-2.0 * pose_err)
    if vel is not None and ref_vel is not None:
        vel_err = (vel - ref_vel).pow(2).sum(dim=-1)
        r = r + w_vel * torch.exp(-0.1 * vel_err)
    return r


@dataclass
class ReferenceMotionBuffer:
    """見本モーション（RSI と AMP のデータ）を保持する（第18章）。

    - ref_obs     : 参照する状態の系列 (N, obs_dim)
    - ref_amp_obs : 参照する状態遷移 (N, amp_obs_dim)（AMP の「本物」）
    """

    ref_obs: torch.Tensor | None = None
    ref_amp_obs: torch.Tensor | None = None
    _frames: list = field(default_factory=list)

    def add_frame(self, obs: torch.Tensor) -> None:
        self._frames.append(obs.detach().clone())

    def finalize(self) -> None:
        if self._frames:
            self.ref_obs = torch.stack(self._frames)
            # 遷移（隣接フレームの連結）を作る
            if len(self._frames) > 1:
                self.ref_amp_obs = torch.cat(
                    [self.ref_obs[:-1], self.ref_obs[1:]], dim=-1
                )


if __name__ == "__main__":
    pose = torch.zeros(4, 6)
    ref = torch.randn(4, 6) * 0.1
    print("追従報酬:", tracking_reward(pose, ref).round(3).tolist())
