"""MuJoCo の基本（第11〜13章）。

- load_and_step      : 生の MuJoCo でモデルを読み込み、ステップ
- make_env           : gymnasium の環境を作る
- inspect_spaces     : 観測・行動の次元を確認
- random_rollout     : ランダム行動のロールアウト
- pd_control_demo    : PD制御（目標角）のデモ
- SimpleTaskEnv      : 既存環境を内包した自作環境の例
- frame_stack        : 履歴のフレームスタッキング
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

# 生の MuJoCo で使う最小モデル（倒立振子）
PENDULUM_XML = """
<mujoco model="pendulum">
  <option timestep="0.01"/>
  <worldbody>
    <body name="cart" pos="0 0 0">
      <joint name="slider" type="slide" axis="1 0 0"/>
      <geom type="capsule" size="0.1" fromto="-0.1 0 0 0.1 0 0"/>
      <body name="pole" pos="0 0 0">
        <joint name="hinge" type="hinge" axis="0 1 0"/>
        <geom type="capsule" size="0.05" fromto="0 0 0 0 0 0.6"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="slider" ctrlrange="-1 1"/>
  </actuator>
</mujoco>
"""


def load_and_step(steps: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """生の MuJoCo でモデルを読み込み、ステップを回す（第11章）。"""
    import mujoco

    model = mujoco.MjModel.from_xml_string(PENDULUM_XML)
    data = mujoco.MjData(model)
    for _ in range(steps):
        data.ctrl[:] = 0.1
        mujoco.mj_step(model, data)
    print("qpos:", data.qpos, "qvel:", data.qvel)
    return data.qpos.copy(), data.qvel.copy()


def make_env(env_id: str = "InvertedPendulum-v5", render_mode: str | None = None):
    """gymnasium の MuJoCo 環境を作る（第11章）。"""
    return gym.make(env_id, render_mode=render_mode)


def inspect_spaces(env_id: str = "Humanoid-v5") -> tuple[tuple, tuple]:
    """観測・行動の次元を確認する（第11・13章）。"""
    env = make_env(env_id)
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape
    print(f"{env_id}: obs={obs_shape}, action={act_shape}")
    env.close()
    return obs_shape, act_shape


def random_rollout(env_id: str = "Walker2d-v5", max_steps: int = 200) -> float:
    """ランダムな行動でロールアウトし、何ステップもつか確認（第12章）。"""
    env = make_env(env_id)
    obs, _ = env.reset(seed=0)
    total_reward = 0.0
    for t in range(max_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if terminated or truncated:
            print(f"{env_id}: {t} ステップで終了, 合計報酬={total_reward:.2f}")
            break
    env.close()
    return total_reward


def pd_control_demo(env_id: str = "InvertedPendulum-v5", steps: int = 100) -> None:
    """PD制御（目標角）のデモ（第12章）。

    行動として「目標角」を渡すと、MuJoCo（position アクチュエータ）が
    内部で PD 制御してトルクに変換する。
    """
    env = make_env(env_id)
    env.reset(seed=0)
    for _ in range(steps):
        target = env.action_space.sample() * 0.1     # 小さめの目標
        env.step(target)
    env.close()
    print("PD制御のデモを実行しました（目標角を渡す）")


class SimpleTaskEnv(gym.Env):
    """既存の MuJoCo 環境を内包し、報酬だけ変える自作環境（第13章）。"""

    def __init__(self, base_env_id: str = "InvertedPendulum-v5", max_steps: int = 500):
        super().__init__()
        self.sim = gym.make(base_env_id)
        self.observation_space = self.sim.observation_space
        self.action_space = self.sim.action_space
        self.max_steps = max_steps
        self.steps = 0

    def reset(self, seed=None, options=None):
        obs, info = self.sim.reset(seed=seed)
        self.steps = 0
        return obs, info

    def step(self, action):
        obs, _, terminated, truncated, info = self.sim.step(action)
        reward = float(np.exp(-5.0 * obs[1] ** 2))   # 例: 棒が立っているほど高い
        self.steps += 1
        terminated = terminated or self.steps > self.max_steps
        return obs, reward, terminated, truncated, info

    def close(self):
        self.sim.close()


def frame_stack(history: list[torch.Tensor], obs: torch.Tensor, stack: int = 4) -> torch.Tensor:
    """直近 stack フレームを連結して状態にする（第13章）。"""
    history.append(obs)
    if len(history) > stack:
        history.pop(0)
    return torch.cat(history, dim=-1)


def main() -> None:
    inspect_spaces("InvertedPendulum-v5")
    inspect_spaces("Humanoid-v5")
    random_rollout("Walker2d-v5")
    try:
        load_and_step(100)
    except Exception as e:
        print("生の MuJoCo のデモはスキップ:", e)
    pd_control_demo()


if __name__ == "__main__":
    main()
