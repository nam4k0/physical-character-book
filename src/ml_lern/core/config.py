"""ハイパーパラメータの集約。

第24章（データセットと訓練）と第16章（PPO）で説明した値を、
論文の数値と MuJoCo で現実的な値の両方を意識してまとめている。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Config:
    # --- 環境 ---
    env_id: str = "Humanoid-v5"      # 使う MuJoCo 環境（第13章）
    num_envs: int = 32               # 並列環境数（論文は数千。CPU では控えめに）
    episode_length_s: float = 10.0   # 1エピソードの長さ（秒）（第24章）
    control_freq: int = 30           # 方策が行動を出す周波数（Hz）
    sim_freq: int = 60               # 物理シミュレーション周波数（Hz）
    init_obj_dist_range: tuple[float, float] = (1.0, 10.0)  # 物体距離 1〜10m（第24章）

    # --- 方策・価値関数・識別器（第23章, 第24章）---
    hidden_dims: tuple[int, ...] = (1024, 512)  # 論文の MLP 構成
    action_std: float = 0.25         # 固定の行動標準偏差 Σ の対角成分
    activation: str = "relu"

    # --- PPO（第16章）---
    learning_rate: float = 3.0e-4
    gamma: float = 0.99              # 割引率
    gae_lambda: float = 0.95         # GAE の λ
    clip_epsilon: float = 0.2        # PPO のクリップ幅
    num_epochs: int = 5              # 同じデータを何周するか（K）
    num_minibatches: int = 4
    value_loss_coef: float = 0.5     # c1
    entropy_coef: float = 0.001      # c2
    max_grad_norm: float = 1.0

    # --- AMP（第19章, 第22章）---
    disc_learning_rate: float = 1.0e-4
    task_reward_weight: float = 0.7  # w^G（式(1)）
    style_reward_weight: float = 0.3 # w^S（式(1)）
    gradient_penalty_weight: float = 10.0  # w^gp（式(5)）
    disc_batch_size: int = 256
    amp_reward_scale: float = 2.0    # スタイル報酬のスケール調整用

    # --- 一般 ---
    seed: int = 0
    device: str = "cpu"              # このデバイスは CUDA なし
    log_interval: int = 10
    save_interval: int = 100
    output_dir: str = "runs/charcontrol"


def default_config(**overrides) -> Config:
    """既定値を生成し、指定された項目だけ上書きする。"""
    cfg = Config()
    for key, value in overrides.items():
        if not hasattr(cfg, key):
            raise KeyError(f"未知の設定項目: {key}")
        setattr(cfg, key, value)
    return cfg
