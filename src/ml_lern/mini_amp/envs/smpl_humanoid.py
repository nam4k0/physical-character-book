"""SMPL 人型へ合流（第28章）。

標準 humanoid.xml の代わりに、SMPL/SMPLH/SMPLX から生成した
MuJoCo 用 humanoid XML を使う。生成には SMPLSim（ZhengyiLuo）を利用する。

> SMPLSim: https://github.com/ZhengyiLuo/SMPLSim
> PHC:     https://github.com/ZhengyiLuo/PHC
> smplx:   https://github.com/vchoutas/smplx

注意（第28章・付録D）:
- SMPL のモデルファイルと AMASS はライセンス登録が必要。
- SMPLSim の一部は CUDA 前提。本リポジトリでは実行しない（コードのみ）。
"""

from __future__ import annotations

from .scene_env import HumanoidSceneEnv


class SMPLHumanoidEnv(HumanoidSceneEnv):
    """SMPL 由来の人体モデルを使う環境。

    使い方（読者の環境で）:
      1. SMPL モデルファイルと AMASS データをライセンス登録のうえ取得
      2. SMPLSim で SMPL → MuJoCo XML を生成
      3. 生成した XML のパスを scene_xml に渡す
      4. AMASS のモーションを参照モーション（RSI・AMP データ）に変換
    """

    def __init__(self, cfg, smpl_xml_path: str = "assets/smpl/humanoid.xml",
                 amass_ref_path: str | None = None, device: str = "cpu"):
        self.smpl_xml_path = smpl_xml_path
        self.amass_ref_path = amass_ref_path
        # 生成済みの SMPL humanoid XML をシーンに使う
        with open(smpl_xml_path, "r", encoding="utf-8") as f:
            scene_xml = f.read()
        super().__init__(cfg, device=device, scene_xml=scene_xml)
        self._load_amass_reference()

    def _load_amass_reference(self):
        """AMASS の SMPL モーションを、シミュレータの関節状態へ変換して読み込む。

        実際には PHC の retargeting（fit_smpl_motion.py 相当）で
        SMPL → MuJoCo の関節角へ変換する必要がある（第28章）。
        ここでは読み込み口だけを示す。
        """
        if self.amass_ref_path is None:
            return
        import numpy as np  # noqa: F401
        # data = np.load(self.amass_ref_path)   # 実際の読み込み
        # self.ref_obs = torch.as_tensor(data["obs"], ...)
        # self.ref_object_state = torch.as_tensor(data["object"], ...)
        # self.ref_amp_obs = torch.cat([self.ref_obs, self.ref_object_state], dim=-1)
        return
