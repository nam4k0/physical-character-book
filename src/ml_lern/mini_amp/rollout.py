"""ロールアウト収集（第02章・第05章の擬似コードに対応）。

方策で環境を一定ステップ進め、PPO と AMP に必要なデータを集める。
タスク報酬とスタイル報酬をここで合成する（式(1)）。
"""

from __future__ import annotations

import torch

from .amp import AMPUpdater
from .buffer import RolloutBuffer
from .models import Policy, Value


@torch.no_grad()
def collect_rollout(
    envs,
    policy: Policy,
    value: Value,
    amp_updater: AMPUpdater,
    cfg,
    obs: torch.Tensor,
    goal: torch.Tensor,
    device: str = "cpu",
) -> tuple[RolloutBuffer, torch.Tensor, torch.Tensor, dict]:
    """num_steps 分のロールアウトを集める。

    戻り値: (バッファ, 最終 obs, 最終 goal, 統計)
    """
    num_envs = cfg.num_envs
    # 方策の制御周波数と物理周波数から、1更新あたりの制御ステップ数を決める
    num_steps = int(cfg.episode_length_s * cfg.control_freq)

    buffer = RolloutBuffer(
        num_steps, num_envs,
        obs_dim=envs.obs_dim, goal_dim=envs.goal_dim,
        action_dim=envs.action_dim, amp_obs_dim=envs.amp_obs_dim,
        device=device,
    )

    ep_returns = torch.zeros(num_envs, device=device)
    completed_returns: list[float] = []
    completed_lengths: list[int] = []
    ep_len = torch.zeros(num_envs, device=device)

    for t in range(num_steps):
        amp_obs = envs.get_amp_obs(obs, goal)
        action, log_prob = policy.act(obs, goal)
        v = value(obs, goal)

        # 環境を1制御ステップ分進める（内部で sim_freq/control_freq 回だけ物理を回す）
        next_obs, goal_next, task_reward, done, info = envs.step(action)

        # --- スタイル報酬（式(6)、第08章）---
        next_amp_obs = envs.get_amp_obs(next_obs, goal_next)
        if amp_updater is not None:
            r_style = amp_updater.predict_style_reward(next_amp_obs)
        else:
            r_style = torch.zeros(num_envs, device=device)   # AMP なしなら 0

        # --- 報酬の合成（式(1)）---
        reward = (
            cfg.task_reward_weight * task_reward
            + cfg.style_reward_weight * cfg.amp_reward_scale * r_style
        )

        buffer.add(t, obs, goal, action, log_prob, v, reward, done.float(),
                   amp_obs, next_obs)

        ep_returns += task_reward
        ep_len += 1
        for i in range(num_envs):
            if done[i]:
                completed_returns.append(ep_returns[i].item())
                completed_lengths.append(int(ep_len[i].item()))
                ep_returns[i] = 0.0
                ep_len[i] = 0

        obs = next_obs
        goal = goal_next

    # 最後の価値（GAE の終端処理用）
    with torch.no_grad():
        last_value = value(obs, goal)

    stats = {
        "mean_episode_return": (sum(completed_returns) / len(completed_returns)) if completed_returns else float("nan"),
        "mean_episode_length": (sum(completed_lengths) / len(completed_lengths)) if completed_lengths else float("nan"),
        "num_episodes": len(completed_returns),
    }
    return buffer, obs, goal, stats
