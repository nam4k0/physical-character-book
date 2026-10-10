"""mini_amp: 論文「Synthesizing Physical Character-Scene Interactions」の
MuJoCo + PyTorch(CPU) による縮小実装。

本パッケージは第III部（実装）で使う。
論文本来の IsaacGym 実装とは規模が異なるが、
- 物理キャラクタ（関節リンク ＋ PD制御）
- PPO（方策・価値関数）
- AMP（敵対的識別器 ＋ スタイル報酬）
- シーン条件付き識別器と物体ランダム化
という核となる部品は同じ構成で実装している。
"""

from .config import Config, default_config

__all__ = ["Config", "default_config"]
