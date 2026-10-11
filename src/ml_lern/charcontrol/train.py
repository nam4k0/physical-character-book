"""学習のメインループ（第27〜36章の統合）。

PPO（方策・価値関数）と AMP（識別器）を交互に更新する。
論文の学習ループ（第21章の図）を、CPU で動く規模に縮小したもの。

使い方:
    python -m ml_lern.charcontrol.train --stage humanoid_amp
    python -m ml_lern.charcontrol.train --stage scene
"""

from __future__ import annotations

import argparse
import os

import torch

from ..core.amp import AMPUpdater
from ..core.config import default_config
from ..core.ppo import PPOUpdater, compute_gae
from ..core.rollout import collect_rollout


def make_env(stage: str, cfg, device: str):
    """段階（stage）に応じた環境を返す（第27章のロードマップ参照）。"""
    if stage == "pendulum":
        from .envs.pendulum import InvertedPendulumEnv
        return InvertedPendulumEnv(cfg, device=device)
    if stage == "reacher":
        from .envs.reacher import ReacherEnv
        return ReacherEnv(cfg, device=device)
    if stage == "hopper":
        from .envs.locomotion import HopperEnv
        return HopperEnv(cfg, device=device)
    if stage == "walker":
        from .envs.locomotion import Walker2dEnv
        return Walker2dEnv(cfg, device=device)
    if stage == "ant":
        from .envs.locomotion import AntEnv
        return AntEnv(cfg, device=device)
    if stage == "humanoid":
        from .envs.humanoid_env import HumanoidLocomotionEnv
        return HumanoidLocomotionEnv(cfg, device=device)
    if stage == "humanoid_amp":
        from .envs.humanoid_env import HumanoidLocomotionEnv
        return HumanoidLocomotionEnv(cfg, device=device)
    if stage == "scene":
        from .envs.scene_env import HumanoidSceneEnv
        return HumanoidSceneEnv(cfg, device=device)
    if stage == "smpl":
        from .envs.smpl_humanoid import SMPLHumanoidEnv
        return SMPLHumanoidEnv(cfg, device=device)
    raise ValueError(f"未知の stage: {stage}")


def train(stage: str, use_amp: bool = True, overrides: dict | None = None):
    cfg = default_config(**(overrides or {}))
    device = cfg.device
    torch.manual_seed(cfg.seed)

    env = make_env(stage, cfg, device)
    obs = env.reset()
    goal = env.goal

    ppo = PPOUpdater(env.obs_dim, env.goal_dim, env.action_dim, cfg, device=device)
    amp = AMPUpdater(env.amp_obs_dim, cfg, device=device) if use_amp else None

    # 参照モーションの AMP データ（識別器の「本物」）
    dataset_amp_obs = env.ref_amp_obs if env.ref_amp_obs is not None else None

    iteration = 0
    while True:
        iteration += 1

        # --- 1) ロールアウト収集（第33章）---
        buffer, obs, goal, ep_stats = collect_rollout(
            env, ppo.policy, ppo.value, amp, cfg, obs, goal, device=device
        )
        batch = buffer.flatten()

        # --- 2) GAE でアドバンテージを計算（第16章）---
        last_value = ppo.value(obs, goal).detach()
        advantages, returns = compute_gae(
            buffer.rewards, buffer.values, buffer.dones,
            cfg.gamma, cfg.gae_lambda, last_value,
        )
        advantages = advantages.reshape(-1)

        # --- 3) PPO で方策・価値関数を更新（第16章）---
        ppo_stats = ppo.update(batch, advantages, returns.reshape(-1))

        # --- 4) 識別器を更新（第19・22章）---
        disc_stats = {}
        if amp is not None and dataset_amp_obs is not None:
            disc_stats = amp.update(batch["amp_obs"], dataset_amp_obs)

        if iteration % cfg.log_interval == 0:
            line = (
                f"[iter {iteration}] "
                f"ep_return={ep_stats['mean_episode_return']:.1f} "
                f"ep_len={ep_stats['mean_episode_length']:.0f} "
                f"pi_loss={ppo_stats['policy_loss']:.3f} "
                f"v_loss={ppo_stats['value_loss']:.3f} "
                f"kl={ppo_stats['approx_kl']:.4f}"
            )
            if disc_stats:
                line += f" disc_loss={disc_stats['disc_loss']:.3f} acc_r={disc_stats['disc_acc_real']:.2f}"
            print(line)

        if cfg.save_interval and iteration % cfg.save_interval == 0:
            os.makedirs(cfg.output_dir, exist_ok=True)
            ckpt = {
                "policy": ppo.policy.state_dict(),
                "value": ppo.value.state_dict(),
                "iteration": iteration,
                "cfg": cfg.__dict__,
            }
            if amp is not None:
                ckpt["discriminator"] = amp.discriminator.state_dict()
            torch.save(ckpt, os.path.join(cfg.output_dir, f"{stage}_{iteration}.pt"))


def main():
    parser = argparse.ArgumentParser(description="物理キャラクタの学習スクリプト")
    parser.add_argument("--stage", default="humanoid_amp",
                        choices=["pendulum", "reacher", "hopper", "walker", "ant",
                                 "humanoid", "humanoid_amp", "scene", "smpl"])
    parser.add_argument("--no-amp", action="store_true", help="AMP を使わず PPO のみで学習")
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--iters", type=int, default=0, help="0 で無限（Ctrl-C で停止）")
    args = parser.parse_args()

    overrides = {}
    if args.num_envs is not None:
        overrides["num_envs"] = args.num_envs
    train(args.stage, use_amp=not args.no_amp, overrides=overrides)


if __name__ == "__main__":
    main()
