"""core: 物理シミュレーション・強化学習・模倣学習の部品（第3〜5部に対応）。

- mujoco_basics.py : MuJoCo の基本（第11〜13章）
- rl_basics.py     : RL の基本・GAE（第14〜15章）
- rewards.py       : 報酬設計（第17章）
- imitation.py     : 模倣学習（第18章）
- models.py        : 方策・価値関数・識別器（第16・19・20章）
- ppo.py           : PPO（第16章）
- amp.py           : AMP の識別器・スタイル報酬（第19・20章）
- rollout.py       : ロールアウト収集
- buffer.py        : ロールアウトバッファ
- config.py        : ハイパーパラメータ
"""

from .config import Config, default_config

__all__ = ["Config", "default_config"]
