"""第1部（PyTorchと深層学習の基礎）のサンプルコード。

- tensors_demo.py  : テンソル操作と自動微分（第01・02章）
- nn_basics.py     : ニューラルネットと学習（第03章）
- timeseries.py    : 時系列データの整形（第04章）
- sequence_models.py: 系列モデル（第05章）

本パッケージのコードは「読者が自分の環境で動かす」前提の教材です。
"""

from .nn_basics import MLP, train_linear_regression, train_mlp_sine
from .timeseries import make_windows, normalize, pad_batch

__all__ = [
    "MLP",
    "train_linear_regression",
    "train_mlp_sine",
    "make_windows",
    "normalize",
    "pad_batch",
]
