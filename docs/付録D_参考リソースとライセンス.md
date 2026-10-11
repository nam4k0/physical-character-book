# 付録D 参考リソースとライセンス

本教科書で参照した論文・データ・ソフトウェアをまとめます。**ライセンスと利用条件は必ず各自で確認**してください（特に SMPL / AMASS は登録制）。

## 1. 中心となる論文

| 論文 | リンク |
| --- | --- |
| Hassan et al., *Synthesizing Physical Character-Scene Interactions*, 2023（本論文） | <https://arxiv.org/abs/2302.00883> |
| 補助動画 | <https://youtu.be/q3hyQdaElQQ> |

## 2. 第1〜2部で参照した論文

| 論文 | リンク |
| --- | --- |
| Goodfellow et al., *Generative Adversarial Networks*, 2014 | <https://arxiv.org/abs/1406.2661> |
| Mescheder et al., *Which Training Methods for GANs do actually Converge?*, 2018 | <http://proceedings.mlr.press/v80/mescheder18a.html> |
| Ho et al., *Denoising Diffusion Probabilistic Models (DDPM)*, 2020 | <https://arxiv.org/abs/2006.11239> |
| Song et al., *Score-Based Generative Modeling*, 2020 | <https://arxiv.org/abs/2011.13456> |
| Chi et al., *Diffusion Policy*, 2023 | <https://arxiv.org/abs/2303.04137> |
| Kingma & Welling, *Auto-Encoding Variational Bayes (VAE)*, 2014 | <https://arxiv.org/abs/1312.6114> |

## 3. 第3〜5部で参照した論文

| 論文 | リンク |
| --- | --- |
| Schulman et al., *Proximal Policy Optimization (PPO)*, 2017 | <https://arxiv.org/abs/1707.06347> |
| Schulman et al., *High-Dimensional Continuous Control Using GAE*, 2015 | <https://arxiv.org/abs/1506.02438> |
| Peng et al., *DeepMimic*, 2018 | <https://arxiv.org/abs/1804.02717> |
| Ho & Ermon, *Generative Adversarial Imitation Learning (GAIL)*, 2016 | <https://arxiv.org/abs/1606.03476> |
| Peng et al., *AMP: Adversarial Motion Priors*, 2021 | <https://arxiv.org/abs/2104.02180> |
| Makoviychuk et al., *Isaac Gym*, 2021 | <https://arxiv.org/abs/2108.10470> |

## 4. 第6部で参照した論文

| 論文 | リンク |
| --- | --- |
| Hassan et al., *Stochastic Scene-Aware Motion Prediction (SAMP)*, 2021 | <https://arxiv.org/abs/2108.08284> |
| Starke et al., *Neural State Machine for Character-Scene Interactions (NSM)*, 2019 | <https://doi.org/10.1145/3355089.3356505> |
| Chao et al., *Learning to Sit*, 2019 | <https://arxiv.org/abs/1812.02713> |
| Chang et al., *ShapeNet*, 2015 | <https://arxiv.org/abs/1512.03012> |

## 5. 第7部・SMPL 系・ツール

| リソース | リンク | 用途 |
| --- | --- | --- |
| MuJoCo | <https://mujoco.org/> | 物理シミュレータ（CPU可） |
| Gymnasium | <https://gymnasium.farama.org/> | 環境インターフェース |
| PyTorch | <https://pytorch.org/> | 深層学習 |
| IsaacGymEnvs | <https://github.com/isaac-sim/IsaacGymEnvs> | AMP を含む RL 環境集（GPU必須） |
| AMP プロジェクトページ | <https://xbpeng.github.io/projects/AMP/index.html> | AMP 公式 |
| SMPLSim | <https://github.com/ZhengyiLuo/SMPLSim> | SMPL → MuJoCo/IsaacGym humanoid 生成 |
| PHC | <https://github.com/ZhengyiLuo/PHC> | SMPL 人型のモーション模倣 |
| PULSE | <https://github.com/ZhengyiLuo/PULSE> | 普遍的動作表現 |
| smplx | <https://github.com/vchoutas/smplx> | SMPL-X の PyTorch 実装 |
| Mink | <https://github.com/kevinzakka/mink> | MuJoCo ベースの逆運動学 |

## 6. データセットとモデル（ライセンス注意）

### 6.1 SMPL / SMPL-X

- **SMPL**: <https://smpl.is.tue.mpg.de/> … 登録制・非商用。
- **SMPL-X**: <https://smpl-x.is.tue.mpg.de/> … 登録制。
- 商用利用は別途ライセンス（Meshcapade など）。

### 6.2 AMASS

- **AMASS**: <https://amass.is.tue.mpg.de/> … 登録制・研究用。

### 6.3 SAMP / ShapeNet

- SAMP: Hassan et al. 2021 のデータ。配布元の条件を確認。
- **ShapeNet**: <https://shapenet.org/> … 登録制。

> **重要**: 本リポジトリには、SMPL / AMASS / SAMP / ShapeNet のデータやモデルファイルを**含めていません**。各自でライセンスに同意のうえ取得してください。商用利用の可否は各配布元の条件を確認してください。

## 7. 実行環境に関する注意（この環境）

- aarch64（ARM64）、CPU 8コア、メモリ約 5GiB。
- **NVIDIA GPU なし（CUDA 不可）** → IsaacGym は動作しません。SMPLSim の一部も CUDA 前提です。
- 第1〜2部（PyTorch・拡散）と第3〜7部（MuJoCo）は CPU で動作します。
- 本文のコードは読者の環境で動かす前提で、本リポジトリでは実行・検証していません。

## 8. 引用について

本教科書を引用・参照する場合は、原論文を明示してください。

```bibtex
@article{hassan2023synthesizing,
  title   = {Synthesizing Physical Character-Scene Interactions},
  author  = {Hassan, Mohamed and Guo, Yunrong and Wang, Tingwu and
             Black, Michael and Fidler, Sanja and Peng, Xue Bin},
  journal = {arXiv preprint arXiv:2302.00883},
  year    = {2023}
}
```

---

[目次に戻る](README.md)
