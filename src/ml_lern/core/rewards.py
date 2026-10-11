"""報酬設計（第17章・第25章）。

- exp_reward              : exp(-k d^2) のヘルパー（第17・25章）
- task_reward_locomotion  : 速度＋直立＋生存（第17章）
- sit_task_reward         : 論文の sit を簡略化（第25章の式(7)〜(9)）
- reference_state_init    : RSI の例（第17・24章）
"""

from __future__ import annotations

import torch


def exp_reward(distance: torch.Tensor, k: float = 1.0) -> torch.Tensor:
    """exp(-k d^2) の形の滑らかな報酬（第17・25章）。"""
    return torch.exp(-k * distance.pow(2))


def task_reward_locomotion(
    obs: torch.Tensor,
    target_speed: float = 1.5,
    w_speed: float = 1.0,
    w_upright: float = 0.5,
    w_alive: float = 0.1,
) -> torch.Tensor:
    """歩行の報酬: 速度追従＋直立＋生存（第17章）。

    obs[..., 0] を前進速度、obs[..., 1] を体幹の傾きと仮定する。
    """
    speed = obs[..., 0]
    speed_r = torch.exp(-2.0 * (target_speed - speed).pow(2))
    upright = torch.exp(-5.0 * obs[..., 1].pow(2))
    alive = torch.ones_like(speed_r)
    return w_speed * speed_r + w_upright * upright + w_alive * alive


def sit_task_reward(
    root_xy: torch.Tensor,
    obj_xy: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    """sit のタスク報酬を簡略化（第25章の式(7)〜(9)）。

    - 0.5m より遠い: 0.7*r_near + 0.3*r_far
    - 0.5m 以内  : 0.7*r_near + 0.3
    """
    dist = (obj_xy - root_xy).norm(dim=-1)
    r_near = exp_reward(dist, k=10.0)                  # 式(9)相当
    r_far = exp_reward(dist, k=0.5)                    # 式(8)の距離項を簡略化
    far_mask = (dist > threshold).float()
    r_far_term = far_mask * r_far + (1.0 - far_mask) * 1.0
    return 0.7 * r_near + 0.3 * r_far_term             # 式(7)


def reference_state_init(ref_obs: torch.Tensor) -> torch.Tensor:
    """参照モーションからランダムな初期状態をサンプルする（第17・24章）。

    RSI: 学習の各エピソードを、見本のランダムなフレームから始める。
    """
    idx = torch.randint(0, ref_obs.shape[0], (1,)).item()
    return ref_obs[idx]


if __name__ == "__main__":
    obs = torch.randn(4, 10)
    print("歩行報酬:", task_reward_locomotion(obs).round(3).tolist())
    root = torch.zeros(4, 2)
    obj = torch.tensor([[0.2, 0.0], [1.0, 0.0], [5.0, 0.0], [10.0, 0.0]])
    print("sit報酬:", sit_task_reward(root, obj).round(3).tolist())
