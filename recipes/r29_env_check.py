#!/usr/bin/env python3
"""レシピ29: 環境構築をチェックする（第28章）。

    uv run python recipes/r29_env_check.py
"""


def main() -> None:
    torch_ok = False
    try:
        import torch
        torch_ok = True
        print("torch:", torch.__version__, "| CUDA:", torch.cuda.is_available())
    except Exception as e:
        print("torch 未導入:", e)

    try:
        import mujoco
        print("mujoco:", mujoco.__version__)
    except Exception as e:
        print("mujoco 未導入:", e)

    try:
        import gymnasium as gym
        env = gym.make("InvertedPendulum-v5")
        print("env obs/act:", env.observation_space.shape, env.action_space.shape)
        env.close()
    except Exception as e:
        print("gymnasium 未導入:", e)

    device = "cuda" if (torch_ok and torch.cuda.is_available()) else "cpu"
    print("使うデバイス:", device)


if __name__ == "__main__":
    main()
