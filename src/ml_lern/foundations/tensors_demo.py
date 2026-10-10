"""テンソル操作と自動微分のデモ（第01・02章）。

NumPy の配列操作と、PyTorch の自動微分を、短いコードで確認する。
"""

from __future__ import annotations

import numpy as np
import torch


def array_basics() -> None:
    """配列の形（shape）と基本操作（第01章）。"""
    print("=== 配列の基本（第01章）===")

    pos = np.array([1.0, 2.0, 3.0])
    print("pos.shape =", pos.shape)      # (3,)

    mat = np.array([[1, 2, 3], [4, 5, 6]])
    print("mat.shape =", mat.shape)      # (2, 3)

    # モーションを (フレーム, 関節×特徴) で表す
    motion = np.zeros((120, 22 * 6))
    print("motion.shape =", motion.shape)  # (120, 132)

    # スライス
    print("先頭10フレーム:", motion[:10].shape)       # (10, 132)
    print("特徴0-5:", motion[:, 0:6].shape)           # (120, 6)


def broadcast_demo() -> None:
    """ブロードキャスト（第01章）。"""
    print("=== ブロードキャスト（第01章）===")
    data = np.random.rand(4, 3)
    bias = np.array([1.0, 2.0, 3.0])         # (3,)
    result = data + bias                     # (4,3) + (3,) → (4,3)
    print("(4,3) + (3,) →", result.shape)

    mean = data.mean(axis=0)                 # 列ごとの平均
    std = data.std(axis=0)
    normalized = (data - mean) / (std + 1e-8)
    print("正規化後の平均:", normalized.mean(axis=0).round(4))


def matrix_product_demo() -> None:
    """要素ごとの積と行列積（第01章）。"""
    print("=== 行列積（第01章）===")
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6], [7, 8]])
    print("要素ごと * =\n", a * b)
    print("行列積 @ =\n", a @ b)


def motion_shape_demo() -> None:
    """モーションの (フレーム, 関節, 3) から (フレーム, 関節*3) への変換（第01章）。"""
    print("=== モーションの形（第01章）===")
    x = np.zeros((120, 22, 3))               # (フレーム, 関節, xyz)
    flat = x.reshape(120, 66)
    print("reshape:", x.shape, "->", flat.shape)
    t = x.transpose(1, 0, 2)
    print("transpose:", t.shape)             # (22, 120, 3)


def autograd_demo() -> None:
    """自動微分のデモ（第02章）。"""
    print("=== 自動微分（第02章）===")
    x = torch.tensor(2.0, requires_grad=True)
    y = x ** 2 + 3 * x + 1
    y.backward()
    print("y = x^2 + 3x + 1, x=2 → dy/dx =", x.grad.item())  # 2x+3 = 7

    # もう1つ: 合成関数
    a = torch.tensor(1.0, requires_grad=True)
    b = torch.sin(a) * torch.exp(a)          # d/da [sin(a) e^a] = e^a (sin a + cos a)
    b.backward()
    expected = float(np.exp(1.0) * (np.sin(1.0) + np.cos(1.0)))
    print("d/da [sin(a)e^a] :", a.grad.item(), "期待値:", round(expected, 5))


def autograd_training_demo() -> None:
    """勾配降下の1ステップを手で書く（第02章）。"""
    print("=== 手書きの学習（第02章）===")
    torch.manual_seed(0)
    x = torch.randn(100, 1)
    y_true = 3.0 * x + 2.0 + 0.1 * torch.randn(100, 1)

    w = torch.zeros(1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    lr = 0.1
    for _ in range(200):
        loss = ((w * x + b - y_true) ** 2).mean()
        loss.backward()
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad
            w.grad.zero_()
            b.grad.zero_()
    print(f"学習結果: w={w.item():.3f} (真値3.0), b={b.item():.3f} (真値2.0)")


def main() -> None:
    array_basics()
    broadcast_demo()
    matrix_product_demo()
    motion_shape_demo()
    autograd_demo()
    autograd_training_demo()


if __name__ == "__main__":
    main()
