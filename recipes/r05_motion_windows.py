#!/usr/bin/env python3
"""レシピ05: アニメを窓に切ってミニバッチ化（第04章）。

    uv run python recipes/r05_motion_windows.py
"""

import torch

from ml_lern.foundations.timeseries import make_loader, make_windows, normalize, pad_batch


def main() -> None:
    motion = torch.randn(300, 22 * 3)                  # (フレーム, 関節×xyz)

    w = make_windows(motion, window=64, stride=16)
    print("窓の形:", w.shape)                            # (N, 64, 66)

    norm, mu, sigma = normalize(w)
    print("正規化後の平均(全体):", round(norm.mean().item(), 5))

    loader = make_loader(w, batch_size=8)
    print("バッチ:", next(iter(loader)).shape)           # (8, 64, 66)

    a, b = torch.randn(100, 6), torch.randn(150, 6)
    padded, mask = pad_batch([a, b])
    print("padded:", padded.shape, "mask:", mask.shape)  # (2,150,6), (2,150)


if __name__ == "__main__":
    main()
