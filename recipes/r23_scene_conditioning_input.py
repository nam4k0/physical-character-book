#!/usr/bin/env python3
"""レシピ23: シーン条件付けの入力を組み立てる（第22章）。

    uv run python recipes/r23_scene_conditioning_input.py
"""

import torch


def main() -> None:
    char = torch.randn(4, 30)          # キャラクタ状態
    obj = torch.randn(4, 9)            # 物体: 位置3 + 向き6D

    amp_input = char
    scene_input = torch.cat([char, obj], dim=-1)
    print("AMP 入力次元:", amp_input.shape[-1])        # 30
    print("シーン入力次元:", scene_input.shape[-1])    # 39

    # 「物体が無い」ケースを、位置を far にして表現する
    obj_far = obj.clone()
    obj_far[:, :3] += 100.0
    scene_far = torch.cat([char, obj_far], dim=-1)
    print("物体が遠いとき:", scene_far.shape)


if __name__ == "__main__":
    main()
