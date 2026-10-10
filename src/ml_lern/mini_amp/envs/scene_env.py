"""本命: 物体を足してシーン相互作用（第26〜27章）。

人型（humanoid）に「物体」（椅子・箱に相当する剛体）を追加し、
論文の中核である「シーン条件付き識別器」を実現する。

論文との対応:
  - 状態に物体特徴（位置・向き）を含める（第09章）
  - 識別器がキャラクタ遷移＋物体状態を見る（第08章）
  - 物体の配置を毎エピソードランダム化する（第11章）

論文の sit / carry を簡略化し、
「物体へ歩いて近づき、目標位置に体（または保持物）を合わせる」
というタスクとして再現する。
"""

from __future__ import annotations

import numpy as np
import torch

from .base import RefMotionEnv

# 物体を追加した humanoid シーンの MJCF。
# 実運用では scripts/ で humanoid.xml に box/chair を足した XML を生成する。
SCENE_XML = """
<mujoco model="humanoid_scene">
  <!-- 元の humanoid.xml をインクルードし、物体（テーブル相当の箱）を足す -->
  <include file="humanoid.xml"/>
  <worldbody>
    <body name="object" pos="2 0 0.5">
      <geom name="object_geom" type="box" size="0.3 0.3 0.5" rgba="0.6 0.4 0.2 1"/>
      <freejoint name="object_joint"/>
    </body>
  </worldbody>
</mujoco>
"""


class HumanoidSceneEnv(RefMotionEnv):
    """人型 ＋ 物体のシーン相互作用環境。

    goal g_t は物体のバウンディングボックス（＝物体の位置・大きさ）。
    状態 s には物体の位置・向きを含める（第09章）。
    """

    def __init__(self, cfg, device: str = "cpu", scene_xml: str = SCENE_XML):
        super().__init__("Humanoid-v5", cfg, device=device)
        self.scene_xml = scene_xml
        self.obs_dim = self._base_obs_dim
        # 物体: 位置(3) + 向き(6D)（第09章）
        self.object_dim = 9
        # 目標: 物体の中心位置(3) + バウンディングボックス半サイズ(3)（第09, 12章）
        self.goal_dim = 6
        self._last_obs = torch.zeros(self.num_envs, self.obs_dim, device=device)
        self._object_state = torch.zeros(self.num_envs, self.object_dim, device=device)
        self._object_center = torch.zeros(self.num_envs, 3, device=device)
        self._object_halfsize = torch.zeros(self.num_envs, 3, device=device)
        self._build_reference_motion()

    def _build_reference_motion(self):
        """参照モーション（第10章）。

        論文は SAMP データ（キャラ＋物体の軌跡）を使う。
        ここではキャラ状態＋物体状態のダミー参照を用意する。
        scripts/prepare_ref.py で実データに差し替える。
        """
        n = 2000
        self.ref_obs = torch.randn(n, self.obs_dim, device=self.device) * 0.1
        self.ref_object_state = torch.randn(n, self.object_dim, device=self.device) * 0.1
        # 識別器用の参照遷移（キャラ＋物体を連結）
        self.ref_amp_obs = torch.cat([self.ref_obs, self.ref_object_state], dim=-1)

    def get_amp_obs(self, obs, goal):
        """シーン条件付き識別器の入力（第08章）。

        「キャラクタ状態」と「物体状態」を連結する。
        これにより識別器は、物体の文脈を踏まえて自然さを判定できる。
        """
        return torch.cat([obs, self._object_state], dim=-1)

    # --- 物体配置のランダム化（第11章）---
    def _maybe_randomize(self, obs):
        """各環境の物体位置・向きをランダム化する。

        - 物体位置: キャラクタから 1〜10m の環状（第11章）
        - 物体向き: [0, 2π] 一様
        """
        lo, hi = self.cfg.init_obj_dist_range
        dist = torch.empty(self.num_envs, device=self.device).uniform_(lo, hi)
        theta = torch.empty(self.num_envs, device=self.device).uniform_(0.0, 2.0 * np.pi)
        x = dist * torch.cos(theta)
        y = dist * torch.sin(theta)
        self._object_center = torch.stack([x, y, torch.full_like(x, 0.5)], dim=-1)
        # 6D 回転表現を適当に初期化（第09章）
        self._object_state[:, :3] = self._object_center
        self._object_state[:, 3:] = torch.randn(self.num_envs, 6, device=self.device) * 0.1
        # バウンディングボックス（半サイズ）。このミニ実装では固定＋ノイズ。
        self._object_halfsize = 0.3 + 0.1 * torch.rand(self.num_envs, 3, device=self.device)

    def _sample_goal(self, obs):
        """目標 = 物体の中心 ＋ バウンディングボックス（第09章, 第12章）。"""
        return torch.cat([self._object_center, self._object_halfsize], dim=-1)

    def _update_object_state(self):
        """物体は行動中に動く。次の状態を物体状態に反映する（第08章の反応）。"""
        # 実運用では物体 joint の位置・速度を環境から読み、フォワード更新する。
        # ここでは物体状態をそのまま維持（物体は受動的）とする簡略化。
        self._object_state[:, :3] = self._object_center

    def task_reward(self, obs, goal, action, next_obs):
        """タスク報酬 r^G（第12章の sit/carry を簡略化）。

        - r^far : 物体へ近づく（距離＋速度）
        - r^near: 腰（root）を物体中心の目標位置に合わせる
        0.5m を境にモードを切り替える（式(7)と同型）。
        """
        root_xy = next_obs[..., :2]
        obj_xy = self._object_center[:, :2]
        dist = (obj_xy - root_xy).norm(dim=-1)

        r_near = torch.exp(-10.0 * dist.pow(2))       # 近距離: 位置合わせ
        r_far = torch.exp(-0.5 * dist.pow(2))          # 遠距離: 誘導（距離項のみ簡略化）

        far_mask = (dist > 0.5).float()
        r_far_term = far_mask * r_far + (1.0 - far_mask) * 1.0
        return 0.7 * r_near + 0.3 * r_far_term

    def step(self, action):
        out = super().step(action)
        self._update_object_state()
        return out
