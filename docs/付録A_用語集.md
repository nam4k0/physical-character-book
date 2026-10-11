# 付録A 用語集

機械学習・グラフィックス・本論文の用語を、テーマ別にまとめます。

## 1. 深層学習の基本

| 用語 | 英語 | 意味 |
| --- | --- | --- |
| テンソル | tensor | 多次元配列。PyTorch のデータの単位 |
| 自動微分 | autograd | 勾配を自動計算する仕組み |
| パラメータ | parameter | 学習で調整される重み・バイアス |
| 全結合層 | fully-connected layer | 入力と出力を全結合する層 |
| MLP | Multi-Layer Perceptron | 全結合層を重ねたネットワーク |
| 活性化関数 | activation | 非線形性を与える関数（ReLU 等） |
| ReLU | Rectified Linear Unit | $\max(0,x)$ |
| 損失関数 | loss function | 間違いの大きさを測る関数 |
| MSE | Mean Squared Error | 二乗誤差（回帰） |
| クロスエントロピー | cross entropy | 分類の損失 |
| 勾配降下法 | gradient descent | 勾配の逆向きにパラメータを動かす |
| 学習率 | learning rate | 1回の更新幅 |
| ミニバッチ | mini-batch | 一部のデータで更新すること |
| エポック | epoch | 全データを1周すること |
| 過学習 | overfitting | 訓練データを丸暗記すること |
| ドロップアウト | dropout | 一部のユニットを無効化する正則化 |

## 2. 生成モデル

| 用語 | 英語 | 意味 |
| --- | --- | --- |
| 生成モデル | generative model | データの分布 $p(x)$ を学ぶ |
| 識別モデル | discriminative model | 入力から答え $p(y\mid x)$ を学ぶ |
| オートエンコーダ | autoencoder (AE) | 圧縮→復元 |
| VAE | Variational AE | 潜在変数を正規分布に整える |
| 潜在変数 | latent variable | データを圧縮した表現 |
| KLダイバージェンス | KL divergence | 2つの分布の近さ |
| 再パラメータ化 | reparameterization | サンプリングを微分可能にするトリック |
| GAN | Generative Adversarial Network | 生成器と識別器を競わせる |
| 生成器 | generator | 偽物を作る側 |
| 識別器 | discriminator | 本物/偽物を見分ける側 |
| モード崩壊 | mode collapse | 生成の多様性が失われる現象 |
| 拡散モデル | diffusion model | ノイズ除去で生成する |
| 前方過程 | forward process | ノイズを加える過程 |
| 逆過程 | reverse process | ノイズを除去する過程 |
| DDPM | Denoising Diffusion Probabilistic Models | 拡散モデルの代表 |
| ノイズスケジュール | noise schedule | ノイズの加え方（$\beta, \alpha, \bar\alpha$） |
| 条件付け | conditioning | お題（初期姿勢等）を与えること |
| classifier-free guidance | — | 条件への忠実さを強める手法 |
| Diffusion Policy | — | 生成モデルによる方策 |

## 3. 時系列・系列モデル

| 用語 | 英語 | 意味 |
| --- | --- | --- |
| 時系列データ | time-series data | 時間順のデータ |
| フレーム | frame | 時系列の1コマ |
| fps | frames per second | 1秒あたりのフレーム数 |
| 窓 | window | 切り出した部分系列 |
| パディング | padding | 長さを揃えるための詰め物 |
| マスク | mask | 有効な部分を示すフラグ |
| RNN | Recurrent NN | 隠れ状態で過去を記憶 |
| LSTM | Long Short-Term Memory | ゲート付き RNN |
| CNN | Convolutional NN | 畳み込みネットワーク |
| Transformer | — | アテンションで系列を扱う |
| アテンション | attention | どこに注目するかの重み |
| フレームスタッキング | frame stacking | 直近フレームを連結する |

## 4. 物理・グラフィックス

| 用語 | 英語 | 意味 |
| --- | --- | --- |
| 剛体 | rigid body | 変形しない物体 |
| リンク | link | 剛体のつながり |
| 関節 | joint | 剛体をつなぐ可動部 |
| 自由度 | DoF | 関節の運動の次元 |
| ルート | root | キャラクタの基準剛体（骨盤等） |
| トルク | torque $\tau$ | 回転を生む力 |
| PD制御 | PD control | 目標へバネ＋ダンパで近づける |
| 比例ゲイン | $k_p$ | バネの強さ |
| 微分ゲイン | $k_d$ | ダンパの強さ |
| アクチュエータ | actuator | 力を加える装置 |
| timestep | timestep | 物理の刻み幅 |
| decimation | decimation | 制御と物理の周波数を分ける間引き |
| 接触 | contact | 物体が触れて力が生じる状態 |
| 摩擦 | friction | 接触面で動きを妨げる力 |
| 運動学 | kinematics | 位置・角度の関係（力は扱わない） |
| ダイナミクス | dynamics | 力と運動の関係 |
| MJCF | MuJoCo XML Format | MuJoCo のモデル記述 |
| バウンディングボックス | bounding box | 物体を囲む直方体 |
| リターゲット | retargeting | 動作を別の骨格へ移す |

## 5. 回転の表現

| 用語 | 英語 | 意味 |
| --- | --- | --- |
| オイラー角 | Euler angle | 3つの角度で表す |
| クォータニオン | quaternion | 4要素で表す |
| 回転行列 | rotation matrix | 3×3 で表す |
| 6D表現 | 6D rotation | 回転行列の2列を使う連続表現 |
| 指数写像 | exponential map | 回転軸×角度で表す3次元表現 |
| ジンバルロック | gimbal lock | オイラー角の特異点 |

## 6. 強化学習

| 用語 | 英語 | 意味 |
| --- | --- | --- |
| 強化学習 | Reinforcement Learning (RL) | 報酬を最大化する方策を学ぶ |
| エージェント | agent | 行動を選ぶ主体 |
| 環境 | environment | 応答を返す相手（シミュレータ） |
| 状態 | state $s_t$ | 時刻 $t$ の状況 |
| 行動 | action $a_t$ | 操作（本論文では目標関節角） |
| 報酬 | reward $r_t$ | 行動の良さ（スカラー） |
| 方策 | policy $\pi$ | 状態→行動の関数 |
| 価値関数 | value function $V$ | 状態の良さ |
| 行動価値 | $Q$ | 状態と行動の組の良さ |
| アドバンテージ | advantage $A$ | 平均よりどれだけ良いか |
| 収益 | return $G_t$ | 割引を適用した報酬の合計 |
| 割引率 | discount factor $\gamma$ | 未来をどれだけ軽く見るか |
| エピソード | episode | 1回の挑戦 |
| 軌跡 | trajectory $\tau$ | 1エピソードの系列 |
| MDP | Markov Decision Process | RL の問題の定式化 |
| 探索 / 活用 | exploration / exploitation | 新しい行動 / 既知の良い行動 |
| 方策勾配 | policy gradient | 良い行動の確率を上げる |
| PPO | Proximal Policy Optimization | 確率比をクリップする手法 |
| GAE | Generalized Advantage Estimation | アドバンテージの安定推定 |
| TD誤差 | temporal-difference error | 「予想外れ」の量 |
| エントロピー | entropy | 行動のばらつき |
| 報酬ハッキング | reward hacking | 意図しない抜け道を学習すること |
| RSI | Reference State Initialization | 見本のランダムフレームから開始 |

## 7. 模倣学習・敵対的学習

| 用語 | 英語 | 意味 |
| --- | --- | --- |
| 模倣学習 | imitation learning | 見本の動きを真似る |
| 行動クローン | behavior cloning | 状態→行動を教師あり学習 |
| モーショントラッキング | motion tracking | RL で見本の姿勢を追従 |
| DeepMimic | — | 代表的モーショントラッキング |
| GAIL | Generative Adversarial Imitation Learning | 敵対的模倣学習 |
| AMP | Adversarial Motion Priors | 物理キャラのスタイル模倣 |
| スタイル報酬 | style reward $r^S$ | 動きの自然さの報酬 |
| タスク報酬 | task reward $r^G$ | 目的達成の報酬 |
| 勾配ペナルティ | gradient penalty | 識別器を平滑化する安定化 |

## 8. 本論文の用語

| 用語 | 意味 |
| --- | --- |
| シーン条件付き識別器 | 物体の文脈も見て自然さを判定する識別器 |
| シーン相互作用 | キャラクタと物体の相互作用 |
| sit / lie down / carry | 座る / 横になる / 運ぶ の3タスク |
| ドメインランダム化 | 環境・物体をランダム化して一般化 |
| 参照モーション | 見本となる動きのデータ |

## 9. データ・モデル・ツール

| 用語 | 意味 |
| --- | --- |
| MoCap | モーションキャプチャ |
| SAMP | 座る・横になる MoCap データセット |
| ShapeNet | 3D モデルのデータベース |
| AMASS | SMPL 形式の大規模 MoCap |
| SMPL | 人体の統計メッシュモデル（運動学） |
| SMPL-X | 手・顔も含む SMPL の拡張 |
| SMPLSim | SMPL → MuJoCo/IsaacGym humanoid 生成 |
| MuJoCo | ロボティクス向け物理シミュレータ（CPU可） |
| gymnasium | 環境インターフェース |
| IsaacGym | NVIDIA の GPU 物理シミュレータ |
| Genesis | 新しい多機能物理エンジン（GPU中心） |
| PyTorch | 深層学習ライブラリ |
| PHC | Perpetual Humanoid Control |
| PULSE | 普遍的動作表現 |
| ASE | Adversarial Skill Embeddings |
| CALM | Compositionally Augmented Latent Motion |
| MaskedMimic | 部分目標から動作を統合 |

## 10. 略語一覧

| 略語 | 正式名称 |
| --- | --- |
| AE / VAE | Autoencoder / Variational Autoencoder |
| GAN | Generative Adversarial Network |
| GAIL | Generative Adversarial Imitation Learning |
| AMP | Adversarial Motion Priors |
| DDPM | Denoising Diffusion Probabilistic Models |
| RL | Reinforcement Learning |
| MDP | Markov Decision Process |
| PPO | Proximal Policy Optimization |
| GAE | Generalized Advantage Estimation |
| RSI | Reference State Initialization |
| DoF | Degrees of Freedom |
| MJCF | MuJoCo XML Format |
| MoCap | Motion Capture |

---

[目次に戻る](README.md)
