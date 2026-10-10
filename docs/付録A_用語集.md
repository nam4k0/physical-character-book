# 付録A 用語集

機械学習・グラフィックス・本論文の用語をまとめます。五十音順ではなく、テーマ別に整理しています。

## 1. 強化学習の基本

| 用語 | 英語 | 意味 |
| --- | --- | --- |
| 強化学習 | Reinforcement Learning (RL) | 報酬を最大化するよう試行錯誤で方策を学ぶ枠組み |
| エージェント | agent | 学習して行動を選ぶ主体（キャラクタ） |
| 環境 | environment | エージェントに応答を返す相手（物理シミュレータ） |
| 状態 | state $s_t$ | 時刻 $t$ の状況を表す数値のまとまり |
| 行動 | action $a_t$ | エージェントが取る操作（本論文では関節の目標角） |
| 報酬 | reward $r_t$ | 行動の良さを表すスカラー |
| 方策 | policy $\pi$ | 状態から行動を決める関数 |
| 価値関数 | value function $V$ | 状態の良さ（期待収益） |
| 行動価値関数 | action-value $Q$ | 状態と行動の組の良さ |
| アドバンテージ | advantage $A$ | 平均よりどれだけ良い行動か（$Q - V$） |
| 収益 | return $G_t$ | 割引を適用した報酬の合計 |
| 割引率 | discount factor $\gamma$ | 未来の報酬をどれだけ軽く見るか |
| エピソード | episode | 1回の挑戦（初期状態から終了まで） |
| 軌跡 | trajectory $\tau$ | 1エピソードの系列 |
| マルコフ決定過程 | MDP | RL の問題の定式化 |
| 探索 | exploration | 新しい行動を試すこと |
| 活用 | exploitation | 既知の良い行動を選ぶこと |

## 2. 学習アルゴリズム

| 用語 | 英語 | 意味 |
| --- | --- | --- |
| 方策勾配 | policy gradient | 良い行動の確率を上げる更新 |
| PPO | Proximal Policy Optimization | 確率比をクリップする安定な方策勾配法 |
| GAE | Generalized Advantage Estimation | アドバンテージの安定な推定 |
| クリップ | clip | 更新幅を制限すること |
| エントロピー | entropy | 行動のばらつき（探索の度合い） |
| TD誤差 | temporal-difference error | 「予想外れ」の量 |
| 価値関数損失 | value loss | 価値関数の回帰誤差 |

## 3. 模倣学習・敵対的学習

| 用語 | 英語 | 意味 |
| --- | --- | --- |
| 模倣学習 | imitation learning | 見本の動きを真似る学習 |
| 行動クローン | behavior cloning | 状態→行動を教師あり学習 |
| モーショントラッキング | motion tracking | RL で見本の姿勢を追従 |
| GAIL | Generative Adversarial Imitation Learning | 敵対的模倣学習 |
| GAN | Generative Adversarial Network | 生成器と識別器を競わせる |
| 識別器 | discriminator $D$ | 本物/偽物を判定するネットワーク |
| 生成器 | generator | 偽物を作るネットワーク |
| 勾配ペナルティ | gradient penalty | 識別器を平滑化する安定化 |
| AMP | Adversarial Motion Priors | 物理キャラのスタイル模倣 |
| スタイル報酬 | style reward $r^S$ | 動きの自然さの報酬 |
| タスク報酬 | task reward $r^G$ | 目的達成の報酬 |
| クロスエントロピー | cross entropy | 二値分類の損失 |
| ロジット | logit | シグモイド前の生スコア |
| モード崩壊 | mode collapse | 生成器が多様性を失う現象 |

## 4. 物理・グラフィックス

| 用語 | 英語 | 意味 |
| --- | --- | --- |
| 剛体 | rigid body | 変形しない物体 |
| 関節 | joint | 剛体をつなぐ可動部 |
| 自由度 | Degrees of Freedom (DoF) | 関節が持つ運動の次元 |
| トルク | torque $\tau$ | 回転を生む力 |
| PD制御 | PD control | 目標へバネ＋ダンパで近づける制御 |
| 比例ゲイン | proportional gain $k_p$ | バネの強さ |
| 微分ゲイン | derivative gain $k_d$ | ダンパの強さ |
| ルート | root | キャラクタの基準となる剛体（多くは骨盤） |
| シミュレータ | simulator | 物理を計算するソフト（IsaacGym, MuJoCo） |
| 接触 | contact | 物体が触れて力が生じる状態 |
| 摩擦 | friction | 接触面で動きを妨げる力 |
| 運動学 | kinematics | 位置・角度の関係（力は扱わない） |
| ダイナミクス | dynamics | 力と運動の関係 |
| MJCF | MuJoCo XML Format | MuJoCo のモデル記述 |
| timestep | timestep | 物理の刻み幅 |
| decimation | decimation | 制御と物理の周波数を分ける間引き |
| バウンディングボックス | bounding box | 物体を囲む直方体 |
| リターゲット | retargeting | 動作を別の骨格へ移す |

## 5. 回転の表現

| 用語 | 英語 | 意味 |
| --- | --- | --- |
| オイラー角 | Euler angle | 3つの角度で回転を表す |
| クォータニオン | quaternion | 4要素で回転を表す |
| 回転行列 | rotation matrix | 3×3 で回転を表す |
| 6D表現 | 6D rotation | 回転行列の2列を使う連続表現 |
| 指数写像 | exponential map | 回転軸×角度で表す3次元表現 |
| ジンバルロック | gimbal lock | オイラー角の特異点 |

## 6. 本論文の用語

| 用語 | 意味 |
| --- | --- |
| シーン条件付き識別器 | 物体の文脈も見て自然さを判定する識別器 |
| シーン相互作用 | キャラクタと物体の相互作用 |
| sit / lie down / carry | 座る / 横になる / 運ぶ の3タスク |
| RSI | Reference State Initialization。参照モーションのランダムフレームから開始 |
| ドメインランダム化 | 環境・物体をランダム化して一般化する |
| 参照モーション | 見本となる動きのデータ |
| バウンディングボックス（目標） | 物体の範囲。carry では不可欠（第13章） |

## 7. データ・モデル

| 用語 | 意味 |
| --- | --- |
| MoCap | モーションキャプチャ（実写の動き計測） |
| SAMP | 座る・横になる MoCap データセット（Hassan et al. 2021） |
| ShapeNet | 3D モデルのデータベース |
| AMASS | SMPL 形式に統一された大規模 MoCap データベース |
| SMPL | 人体の統計メッシュモデル（運動学） |
| SMPL-X | 手・顔も含む SMPL の拡張 |
| SMPLSim | SMPL から MuJoCo/IsaacGym の humanoid を生成するライブラリ |
| IsaacGym | NVIDIA の GPU 物理シミュレータ |
| MuJoCo | ロボティクス向け物理シミュレータ（CPU 可） |

## 8. 略語一覧

| 略語 | 正式名称 |
| --- | --- |
| RL | Reinforcement Learning |
| MDP | Markov Decision Process |
| PPO | Proximal Policy Optimization |
| GAE | Generalized Advantage Estimation |
| GAIL | Generative Adversarial Imitation Learning |
| GAN | Generative Adversarial Network |
| AMP | Adversarial Motion Priors |
| RSI | Reference State Initialization |
| DoF | Degrees of Freedom |
| MJCF | MuJoCo XML Format |
| PHC | Perpetual Humanoid Control |
| ASE | Adversarial Skill Embeddings |
| CALM | Compositionally Augmented Latent Motion |
| DDPM | Denoising Diffusion Probabilistic Models |
