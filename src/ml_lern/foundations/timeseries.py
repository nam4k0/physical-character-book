"""時系列データの整形（第04章）。

モーションを (バッチ, 時間, 特徴) で扱うためのユーティリティ。
- make_windows : 長い系列を固定長の窓に切り出す
- normalize    : 特徴ごとの正規化
- pad_batch    : 可変長の系列をパディングしてマスクを返す
- MotionDataset/DataLoader : ミニバッチ化
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset


def make_windows(motion: torch.Tensor, window: int = 64, stride: int = 16) -> torch.Tensor:
    """(T, F) のモーションを (num_windows, window, F) に切り出す（第04章）。"""
    assert motion.dim() == 2, "motion は (T, F) の形であること"
    windows = []
    for start in range(0, len(motion) - window + 1, stride):
        windows.append(motion[start:start + window])
    return torch.stack(windows)


def normalize(data: torch.Tensor, dim=(0, 1)) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """特徴ごとの正規化（第04章）。

    統計量は訓練データから計算する。戻り値は (正規化データ, 平均, 標準偏差)。
    """
    mean = data.mean(dim=dim, keepdim=True)
    std = data.std(dim=dim, keepdim=True)
    normalized = (data - mean) / (std + 1e-8)
    return normalized, mean, std


def pad_batch(motions: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """可変長のモーションをパディングする（第04章）。

    戻り値: (padded (B, Tmax, F), mask (B, Tmax))。mask が 1 のところが本物。
    """
    max_len = max(len(m) for m in motions)
    feat = motions[0].shape[-1]
    padded = torch.zeros(len(motions), max_len, feat)
    mask = torch.zeros(len(motions), max_len)
    for i, m in enumerate(motions):
        n = len(m)
        padded[i, :n] = m
        mask[i, :n] = 1.0
    return padded, mask


class MotionDataset(Dataset):
    """(N, T, F) の窓を扱うデータセット（第04章）。"""

    def __init__(self, windows: torch.Tensor):
        self.windows = windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, i: int) -> torch.Tensor:
        return self.windows[i]


def make_loader(windows: torch.Tensor, batch_size: int = 32, shuffle: bool = True) -> DataLoader:
    return DataLoader(MotionDataset(windows), batch_size=batch_size, shuffle=shuffle)


def demo() -> None:
    """動作確認（第04章）。"""
    torch.manual_seed(0)
    # 例: 120フレーム、関節22×特徴6 = 132次元
    motion = torch.randn(300, 132)

    print("=== 窓の切り出し ===")
    windows = make_windows(motion, window=64, stride=16)
    print("windows.shape =", windows.shape)

    print("=== 正規化 ===")
    norm, mean, std = normalize(windows)
    print("正規化後の平均(全体):", norm.mean().item().__round__(5))

    print("=== パディング ===")
    a = torch.randn(100, 6)
    b = torch.randn(150, 6)
    c = torch.randn(120, 6)
    padded, mask = pad_batch([a, b, c])
    print("padded.shape =", padded.shape, "mask.shape =", mask.shape)

    print("=== DataLoader ===")
    loader = make_loader(windows, batch_size=32)
    for batch in loader:
        print("バッチ形:", batch.shape)   # (32, 64, 132)
        break


if __name__ == "__main__":
    demo()
