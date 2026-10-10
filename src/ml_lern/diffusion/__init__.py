"""拡散モデル（付録B）。

論文では使わない関連技術として、2D バウンスボールの軌道生成を扱う。
「動作の分布を学ぶ」という点で AMP と対比できる（付録B参照）。

- dataset.py : バウンスボールの軌道データ生成
- schedule.py: ノイズスケジュール（DDPM）
- model.py   : ノイズ除去ネットワーク
- train.py   : 拡散モデルの学習
- sample.py  : 生成（無条件／条件付き）
"""

from .schedule import NoiseSchedule
from .model import Denoiser

__all__ = ["NoiseSchedule", "Denoiser"]
