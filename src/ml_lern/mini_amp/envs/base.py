"""環境ラッパの基底クラス。

MuJoCo（gymnasium.Env）を包み、論文の学習に必要なものを追加する:

- 目標 g_t の生成（第09章）
- タスク報酬 r^G の計算（第12章）
- AMP 用の状態遷移 amp_obs の抽出（第05章, 第08章）
- Reference State Initialization（RSI）と物体ランダム化（第11章）

「物体」を持つかどうかで、humanoid_env（物体なし）と
scene_env（物体あり、シーン条件付き識別器）に分かれる。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class EnvSpec:
    obs_dim: int
    goal_dim: int
    action_dim: int
    amp_obs_dim: int


class RefMotionEnv:
    """gymnasium の MuJoCo 環境をラップする基底クラス。

    実運用では gymnasium を import して使うが、ここでは依存を明示するために
    コンストラクタで生成する。CPU でも動くよう、環境はベクトル化せず
    Python ループで複数環境を持つ簡易実装にしている。
    """

    def __init__(self, env_id: str, cfg, num_envs: int | None = None, device: str = "cpu"):
        self.cfg = cfg
        self.env_id = env_id
        self.device = device
        self.num_envs = num_envs or cfg.num_envs

        # 遅延 import（MuJoCo が無い環境でもモジュールを読めるように）
        import gymnasium as gym  # noqa: F401

        self._gym = gym
        self.envs = [gym.make(env_id) for _ in range(self.num_envs)]
        for e in self.envs:
            e.reset(seed=cfg.seed)

        # 観測・行動の次元（gymnasium の空間から取得）
        obs_space = self.envs[0].observation_space
        act_space = self.envs[0].action_space
        self._base_obs_dim = int(np.prod(obs_space.shape))
        self.action_dim = int(np.prod(act_space.shape))

        # 参照モーション（RSI と AMP データ）を用意する。サブクラスで上書き。
        self.ref_obs: torch.Tensor | None = None   # (N, obs_dim)
        self.ref_amp_obs: torch.Tensor | None = None  # (N, amp_obs_dim)

        self.obs_dim = self._base_obs_dim
        self.goal_dim = 0
        self.goal = torch.zeros(self.num_envs, self.goal_dim, device=device)
        self._sim_substeps = max(1, int(cfg.sim_freq / cfg.control_freq))

    # --- サブクラスが実装する ---
    @property
    def amp_obs_dim(self) -> int:
        """識別器に入れる遷移ベクトルの次元。"""
        return self.obs_dim * 2 + getattr(self, "object_dim", 0) * 2

    def get_amp_obs(self, obs: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """識別器用の入力を作る。

        基本は「今の状態」＋「物体特徴」。実際の遷移ペア (s_t, s_{t+1}) は
        バッファ側で連結する（本実装では簡略化のため状態ベクトルをそのまま使う）。
        """
        raise NotImplementedError

    def task_reward(self, obs, goal, action, next_obs):
        """タスク報酬 r^G を計算する（第12章）。"""
        raise NotImplementedError

    # --- 共通処理 ---
    def reset(self):
        obs_list = []
        for e in self.envs:
            o, _ = e.reset()
            obs_list.append(o)
        obs = torch.as_tensor(np.stack(obs_list), dtype=torch.float32, device=self.device)
        self._maybe_randomize(obs)
        self.goal = self._sample_goal(obs)
        return obs

    def step(self, action: torch.Tensor):
        """1制御ステップ進める。内部で物理を複数回（sim_freq/control_freq）回す。"""
        action_np = action.detach().cpu().numpy()
        next_list, reward_list, done_list = [], [], []
        for i, e in enumerate(self.envs):
            total_task_reward = 0.0
            done = False
            o = None
            for _ in range(self._sim_substeps):
                o, _, terminated, truncated, _ = e.step(action_np[i])
                done = terminated or truncated
                if done:
                    break
            next_list.append(o)
            reward_list.append(0.0)  # タスク報酬は下で一括計算
            done_list.append(done)

        next_obs = torch.as_tensor(np.stack(next_list), dtype=torch.float32, device=self.device)
        task_r = self.task_reward(self._last_obs, self.goal, action, next_obs)
        done_t = torch.as_tensor(done_list, dtype=torch.bool, device=self.device)

        self._last_obs = next_obs
        # 終了した環境はリセットし、目標を再サンプル（RSI とランダム化）
        if done_t.any():
            self._reset_done(done_t)

        return next_obs, self.goal, task_r, done_t, {}

    # --- 内部ユーティリティ ---
    def _reset_done(self, done_t: torch.Tensor):
        idx = done_t.nonzero(as_tuple=False).squeeze(-1).tolist()
        for i in idx:
            o, _ = self.envs[i].reset()
            self._place_env(i, torch.as_tensor(o, dtype=torch.float32, device=self.device))

    def _place_env(self, env_index: int, base_obs: torch.Tensor):
        """RSI: 参照モーションのランダムなフレーム状態へ初期化する（第11章）。

        本実装では簡略化のため、参照モーションがあればその状態を、
        なければ通常のリセットを使う。
        """
        if self.ref_obs is None:
            return
        j = int(torch.randint(self.ref_obs.shape[0], (1,)).item())
        ref = self.ref_obs[j]
        qpos = ref[: self.envs[env_index].model.nq].cpu().numpy()
        qvel = ref[self.envs[env_index].model.nq:
                   self.envs[env_index].model.nq + self.envs[env_index].model.nv].cpu().numpy()
        self.envs[env_index].set_state(qpos, qvel)

    def _maybe_randomize(self, obs: torch.Tensor):
        """物体配置のランダム化（第11章）。物体を持つ環境で上書きする。"""
        return

    def _sample_goal(self, obs: torch.Tensor) -> torch.Tensor:
        """目標 g_t をサンプルする（第09章）。"""
        return torch.zeros(self.num_envs, self.goal_dim, device=self.device)

    def close(self):
        for e in self.envs:
            e.close()
