"""環境（物理シミュレーション）のラッパ群（第III部）。

MuJoCo（gymnasium）の環境をラップし、論文で必要な
- タスク報酬 r^G（第12章）
- 参照モーション（RSI と AMP のデータ、第10〜11章）
- シーン（物体）のランダム化（第11章）
を追加する。
"""

from .base import RefMotionEnv
from .humanoid_env import HumanoidLocomotionEnv
from .scene_env import HumanoidSceneEnv

__all__ = ["RefMotionEnv", "HumanoidLocomotionEnv", "HumanoidSceneEnv"]
