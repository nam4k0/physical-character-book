#!/usr/bin/env python3
"""レシピ06: 次フレームのポーズを予測する（第05章）。

    uv run python recipes/r06_pose_prediction.py
"""

import torch

from ml_lern.foundations.sequence_models import (
    ConvPredictor,
    FrameStackMLP,
    RNNPredictor,
    TransformerPredictor,
)


def main() -> None:
    feat = 12
    x = torch.randn(16, 20, feat)              # (バッチ, 時間, 特徴)
    y = torch.randn(16, feat)

    model = RNNPredictor(feat, hidden=64, out_dim=feat)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(500):
        loss = ((model(x) - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    print("LSTM 予測:", model(x).shape)

    for name, m in [("stack", FrameStackMLP(feat, out_dim=feat)),
                    ("conv", ConvPredictor(feat, out_dim=feat)),
                    ("tf", TransformerPredictor(feat, out_dim=feat))]:
        print(name, m(x).shape)


if __name__ == "__main__":
    main()
