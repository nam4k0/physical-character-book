# 37. SMPL 人型へ合流

## この章でわかること

- SMPL とは何か、なぜ「そのままでは物理制御できない」のか
- SMPL を MuJoCo の人型に変換する流れ（SMPLSim）
- AMASS を参照モーションとして使う方法
- ライセンスと注意点

## 前提

- 第10章（データセット）、第23章（状態）、第36章（シーン条件付け）

## つながり

- MuJoCo の標準 humanoid で一通りの実装をしてきました。この章で、より本格的な人体モデル（SMPL）へ合流します。

---

## 1. SMPL とは

**SMPL（Skinned Multi-Person Linear model）** は、人体の**メッシュ（表面形状）**を表す統計モデルです。

- パラメータ: **体型 $\beta$**、**姿勢 $\theta$**（関節の回転）、**ルート位置**。
- 出力: 6890頂点のメッシュ。

> Matthew Loper, et al. **SMPL: A Skinned Multi-Person Linear Model.** ACM TOG 34(6), 2015. <https://smpl.is.tue.mpg.de/>

## 2. なぜ SMPL をそのまま物理制御できないのか

ここが重要です。SMPL は**運動学（キネマティクス）モデル**であり、**物理（ダイナミクス）を持ちません**。

- SMPL は「関節角を与えるとメッシュがどう変形するか」を計算します。
- 「関節にトルクをかけるとどう動くか」は計算しません。それは物理シミュレータの仕事です。

第11〜12章で学んだとおり、物理キャラ制御には**剛体リンク＋関節＋トルク**が必要です。SMPL のメッシュをそのままシミュレータに投入しても動きません。

```mermaid
flowchart LR
    S["SMPL<br/>運動学モデル（メッシュ）"] -->|"変換が必要"| P["物理人型<br/>剛体リンク＋関節"]
    P --> SIM["物理シミュレータ（MuJoCo）"]
```

## 3. 変換の流れ — SMPL → MuJoCo humanoid

この変換を担うのが **SMPLSim**（ZhengyiLuo 氏）です。

> SMPLSim: <https://github.com/ZhengyiLuo/SMPLSim>
> 関連: PHC（ICCV 2023）<https://github.com/ZhengyiLuo/PHC> ／ PULSE（ICLR 2024）

```mermaid
flowchart TD
    A["SMPL モデルファイル<br/>（ライセンス登録が必要）"] --> B["SMPLSim"]
    B --> C["MuJoCo humanoid XML<br/>（SMPL 互換の関節構成）"]
    C --> D["charcontrol の SMPLHumanoidEnv"]
    E["AMASS<br/>（SMPL 形式の MoCap）"] --> F["リターゲット<br/>（SMPL → 関節角）"]
    F --> G["参照モーション（RSI・AMP データ）"]
    G --> D
```

### 生成される humanoid の特徴

- SMPL の関節構成（骨盤ルート、脊椎、四肢）に対応した剛体リンク。
- 体型パラメータ $\beta$ を変えれば、体格の異なるキャラクタも生成できる。
- カプセルベース（軽量）とメッシュベース（見た目重視）を選べる。

## 4. AMASS を参照モーションにする

**AMASS** は、SMPL 形式に統一された大規模 MoCap データベースです。

> AMASS: <https://amass.is.tue.mpg.de/>

- 数千のモーション（歩行、走行、ダンス、座る、など）。
- すべて SMPL 形式なので、SMPL 人型と相性が良い。

### リターゲットが必要

AMASS は「SMPL の関節角」を持ちますが、MuJoCo humanoid は「別の関節構成」です。そのため、**リターゲット**（関節の対応付け・最適化）が必要です。

```mermaid
flowchart LR
    A["AMASS<br/>SMPL 関節角"] --> B["関節の対応付け"] --> C["最適化でフィット"] --> D["MuJoCo 形式の参照モーション"]
```

参考:

- PHC の `fit_smpl_motion.py` … SMPL モーションをロボットへフィット。
- TrackerLab … SMPL/AMASS/FBX をリターゲット。

## 5. 実装へのつなぎ

`SMPLHumanoidEnv` は、生成済みの SMPL humanoid XML をシーンに使うだけです（第36章の `HumanoidSceneEnv` を継承）。

```python
class SMPLHumanoidEnv(HumanoidSceneEnv):
    def __init__(self, cfg, smpl_xml_path="assets/smpl/humanoid.xml", amass_ref_path=None, device="cpu"):
        with open(smpl_xml_path, "r", encoding="utf-8") as f:
            scene_xml = f.read()
        super().__init__(cfg, device=device, scene_xml=scene_xml)
        self._load_amass_reference()
```

使い方（読者の環境で）:

1. SMPL モデルと AMASS をライセンス登録のうえ取得。
2. SMPLSim で `humanoid.xml` を生成。
3. AMASS のモーションをリターゲットし、参照モーションに変換。
4. `SMPLHumanoidEnv` で学習。

## 6. 事実確認: この系譜は論文とどう違うか

| 項目 | 本論文（2023） | SMPL 系（PHC / PULSE など） |
| --- | --- | --- |
| キャラクタ | 論文独自の剛体人型（IsaacGym） | SMPL から変換した物理人型 |
| データ | SAMP（sit/lie）＋自作 carry | AMASS（SMPL 形式） |
| 目的 | シーン相互作用（物体との接触） | 単一キャラのモーション模倣（物体なしが多い） |
| 識別器 | シーン条件付き（物体を見る） | 通常の AMP 系（キャラのみ） |

つまり **SMPL 系は「本論文の系列そのもの」ではなく、関連する別系列**です。共通点は「物理人型を AMP 系で制御する」こと。本教科書は、本論文を主軸にしつつ、**SMPL を使えばより本格的な人体モデルで実験できる**という発展として紹介しています。

## 7. ライセンスと注意（重要）

- **SMPL**: <https://smpl.is.tue.mpg.de/> で登録してダウンロード。**非商用**。
- **SMPL-X**: <https://smpl-x.is.tue.mpg.de/>
- **AMASS**: <https://amass.is.tue.mpg.de/> で登録。研究用途。
- 商用利用は別途ライセンスが必要な場合があります。

> **注意**: 本リポジトリには SMPL モデル・AMASS データを含めません。各自でライセンスに同意のうえ取得してください。SMPLSim の一部は CUDA 前提のため、この CPU 環境では実行しません。詳細は付録D。

## 8. 実装コード

- `src/ml_lern/charcontrol/envs/smpl_humanoid.py` … `SMPLHumanoidEnv`
- `scripts/train_07_smpl.py`

## つまずきやすいポイント

| 症状 | 対処 |
| --- | --- |
| XML が生成できない | SMPLSim のセットアップ、Python バージョンを確認 |
| モーションがずれる | リターゲットの対応付け・座標系を確認 |
| 関節が足りない/多い | extend_config などで関節を追加 |
| 動かない/埋まる | 足元の高さ合わせ、SMPL の座標系変換 |

## まとめ

- SMPL はメッシュ（運動学）で、そのままでは物理制御できない。
- SMPLSim で SMPL → MuJoCo humanoid XML に変換する。
- AMASS を参照モーションに使うにはリターゲットが必要。
- SMPL 系は論文と関連する別系列。ライセンス登録が必要。

## 演習

1. SMPL をそのまま物理シミュレータに入れても動かない理由を、「運動学」と「ダイナミクス」の違いから説明しましょう。
2. SMPL 人型を生成する利点（体型の多様性など）を2つ挙げましょう。
3. AMASS を参照モーションに使うとき、リターゲットが必要な理由を関節構成の観点で述べましょう。
4. PHC（モーション模倣）と本論文（シーン相互作用）の共通点と相違点を整理しましょう。

---

次章: [38. 評価とトラブル対応](38_評価とトラブル対応.md) ／ 前章: [36. 物体とシーン条件付き識別器](36_物体とシーン条件付き識別器.md)
