# 付録F 再現レシピ（物理側） — 第3〜7部

この付録は、各章の **「夢が広がる」** を実際に手で再現するための、**コメント多めのレシピ集**です。

- 各レシピに対応する**実行できるコード**は `recipes/` にあります（例: `recipes/r12_mjcf_scene.py`）。
- 実行は `uv run python recipes/rNN_....py`（`uv sync --extra train --extra viz` 済み）。
- AI側（第0〜2部）のレシピは [付録E](付録E_再現レシピ_AI側.md) にあります。
- コードは読者の環境で動かす前提で、本リポジトリでは実行・検証していません。

> アンカーは HTML の `id`（例: `recipe-12`）で明示しているので、docsite でも GitHub でもジャンプできます。

---

<a id="recipe-12"></a>
## レシピ12. 自作シーンを MuJoCo で書いて転がす（第11章）

**もとになった「夢」**: ゲームの物理挙動を MuJoCo で再現し、パラメータ調整に使う。
**使うもの**: MuJoCo
**実行**: `recipes/r12_mjcf_scene.py`

```python
import mujoco

XML = """
<mujoco model="my_scene">
  <option timestep="0.002"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="box" pos="0 0 1">
      <geom type="box" size="0.2 0.2 0.2" rgba="1 0.5 0 1"/>
      <freejoint/>                              <!-- 自由に落ちる -->
    </body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)
for step in range(2000):
    mujoco.mj_step(model, data)                   # シミュレーションを進める
    if step % 500 == 0:
        print(f"step {step}: 箱の高さ = {data.qpos[2]:.3f}")
# 箱が落ちて跳ね、やがて静止する（高さが収束する）
```

**実行すると**: 箱の高さが時間とともに下がり、床で止まります。
**発展**: `size`/`density`/`friction` を変える。`freejoint` を `hinge` にすると回転板になる。

---

<a id="recipe-13"></a>
## レシピ13. PDゲイン／timestep／摩擦を変えて観察（第12章）

**もとになった「夢」**: キャラクタコントローラを PD＋目標角で作り直す。
**使うもの**: gymnasium(MuJoCo)
**実行**: `recipes/r13_pd_tuning.py`

```python
import gymnasium as gym
import numpy as np

def rollout(env_id="InvertedPendulum-v5", steps=200, action_scale=0.1):
    env = gym.make(env_id)
    env.reset(seed=0)
    total = 0.0
    for _ in range(steps):
        action = env.action_space.sample() * action_scale   # 小さめの目標角
        _, r, term, trunc, _ = env.step(action)
        total += r
        if term or trunc:
            break
    env.close()
    return total

# action_scale を変えて「制御の強さ」による挙動差を見る
for s in [0.05, 0.1, 0.5, 1.0]:
    print(f"action_scale={s}: 合計報酬={rollout(action_scale=s):.1f}")
# 大きすぎると振動・転倒、小さすぎると応答しない
```

**実行すると**: 制御の強さで結果が変わる様子が見えます。
**発展**: XML の `kp`/`kv`、`timestep`、`friction` を直接変えて比較する（第12章の式で説明する）。

---

<a id="recipe-14"></a>
## レシピ14. フレームスタッキングで履歴を状態にする（第13章）

**もとになった「夢」**: 速度などが直接観測できないとき、履歴で補う。
**使うもの**: `core.mujoco_basics.frame_stack`
**実行**: `recipes/r14_frame_stack_state.py`

```python
import torch
from ml_lern.core.mujoco_basics import frame_stack

hist = []
obs = torch.zeros(8)                 # 例: 8次元の観測
stacked = None
for t in range(5):
    obs = obs + 0.1 * torch.randn(8)  # 観測が少しずつ変化
    stacked = frame_stack(hist, obs, stack=4)   # 直近4フレームを連結
    print(f"t={t}: 状態の次元 = {stacked.shape[-1]}")   # 4 フレーム目以降は 32
# スタックなし（1フレーム）では速度が分からないが、スタックすると推定できる
```

**実行すると**: スタック数に応じて状態次元が増えます。
**発展**: スタック=1 と 4 で、学習しやすさがどう変わるか（ログで）比べる。

---

<a id="recipe-15"></a>
## レシピ15. グリッド世界を自作して報酬を設計する（第14章）

**もとになった「夢」**: ゲーム AI（追跡・逃走・パトロール）を報酬設計から作る。
**使うもの**: 標準ライブラリのみ（環境は自作）
**実行**: `recipes/r15_gridworld.py`

```python
import random

class GridWorld:
    def __init__(self, size=5):
        self.size, self.goal = size, (size-1, size-1)
        self.holes = {(1,1), (3,2)}
        self.pos = (0, 0)
    def reset(self):
        self.pos = (0, 0); return self.pos
    def step(self, a):                       # 0上 1右 2下 3左
        x, y = self.pos
        dx, dy = [(-1,0),(0,1),(1,0),(0,-1)][a]
        self.pos = (min(max(x+dx,0), self.size-1), min(max(y+dy,0), self.size-1))
        if self.pos in self.holes:  return self.pos, -1.0, True   # 穴: 大罰
        if self.pos == self.goal:   return self.pos, +1.0, True   # ゴール: 報酬
        return self.pos, -0.01, False            # 毎ステップ小罰（急ぐ）
    def actions(self):
        return 4

# ランダム方策で試す（学習前のベースライン）
env = GridWorld()
for _ in range(3):
    env.reset(); total = 0
    for _ in range(30):
        s, r, done = env.step(random.randrange(env.actions())); total += r
        if done: break
    print("1エピソードの収益:", round(total, 2))
```

**実行すると**: ランダムでもゴールに届くが、遠回りして「毎ステップ小罰」を食らいます。
**発展**: 「毎ステップ -0.01」を外すと、ゴールせずうろつく解になる（報酬ハッキング）。

---

<a id="recipe-16"></a>
## レシピ16. 多腕バンディットを方策勾配で解く（第15章）

**もとになった「夢」**: 「どの戦略が勝ちやすいか」を試行から学ぶ。
**使うもの**: PyTorch（方策勾配）
**実行**: `recipes/r16_bandit_policy_gradient.py`

```python
import torch

true_rewards = torch.tensor([0.1, 0.9, 0.3, 0.2, 0.5])   # 各腕の当たりやすさ
probs = torch.ones(5) / 5                                # 選択確率（方策）
lr = 0.1

for step in range(2000):
    arm = torch.multinomial(probs, 1).item()             # 方策に従い腕を選ぶ
    r = true_rewards[arm].item()                         # 実際の報酬
    # 方策勾配: 報酬が平均より高かった腕の確率を上げる
    grad = torch.zeros(5); grad[arm] = r - true_rewards.mean()
    probs = torch.softmax(torch.log(probs) + lr * grad, dim=0)
    if step % 500 == 0:
        print(step, probs.round(2).tolist())
print("最終:", probs.round(2).tolist())                   # 腕1が最大になる
```

**実行すると**: 一番当たりやすい「腕1」の確率が上がります。
**発展**: `lr` を大きく／小さくする。確率の下限（探索）を入れて局所解を抜ける。

---

<a id="recipe-17"></a>
## レシピ17. 自作の簡単なタスクで PPO を回す（第16章）

**もとになった「夢」**: 自分のゲームの制御タスクを PPO で解く。
**使うもの**: `charcontrol.train`（既存環境で素早く）
**実行**: `recipes/r17_ppo_toy.py`

```python
from ml_lern.charcontrol.train import train
from ml_lern.core.config import default_config

# 既存の倒立振子を「自作タスクの代わり」に使い、PPO の設定を変えて比べる。
# clip_epsilon（更新幅）を変えて、学習の安定さを観察する。
for eps in [0.05, 0.2, 0.5]:
    cfg = default_config(clip_epsilon=eps, num_envs=16, log_interval=5)
    print(f"--- clip_epsilon={eps} ---")
    # train("pendulum", use_amp=False, overrides={"clip_epsilon": eps})  # 実学習（時間がかかる）
print("train() の stage を自作環境へ向ければ、自分のタスクを PPO で学習できる")
```

> 実行時間を抑えるため、このスクリプトは設定を確認するだけにしています。実際に回すには `train("pendulum", overrides={"clip_epsilon": ...})` を呼びます。

**実行すると**: 比較の手順が確認できます。
**発展**: 第13章の `SimpleTaskEnv` を自作報酬に差し替え、`train` の stage を向ける。

---

<a id="recipe-18"></a>
## レシピ18. far/near 報酬と RSI を実装する（第17章）

**もとになった「夢」**: 「自然で強い行動」を報酬で書く。
**使うもの**: `core.rewards`
**実行**: `recipes/r18_reward_far_near.py`

```python
import torch
from ml_lern.core.rewards import exp_reward, reference_state_init

# far/near の2段構え（論文の式(7)(8)(9) の簡略形）
dist = torch.tensor([0.0, 0.3, 0.49, 0.51, 1.0, 3.0, 8.0])
r_near = exp_reward(dist, k=10.0)          # 近距離: 精密に合わせる
r_far = exp_reward(dist, k=0.5)            # 遠距離: なだらかに誘導
near_mask = (dist <= 0.5).float()
reward = 0.7 * r_near + 0.3 * (near_mask * 1.0 + (1 - near_mask) * r_far)
for d, r in zip(dist.tolist(), reward.tolist()):
    print(f"距離 {d:4.2f}m -> 報酬 {r:.3f}")

# RSI: 参照モーションのランダムなフレームからエピソードを始める
ref = torch.randn(500, 30)                 # 参照モーション（500フレーム）
start = reference_state_init(ref)
print("RSI の初期状態:", tuple(start.shape))
```

**実行すると**: 0.5m を境に報酬の構成が切り替わり、RSI が初期状態を返します。
**発展**: 速度項（物体方向の速度）を足し、動作に「歩き」を促す（式(8)）。

---

<a id="recipe-19"></a>
## レシピ19. 追従報酬で見本を真似る（第18章）

**もとになった「夢」**: アニメクリップから物理キャラの動きを作る。
**使うもの**: `core.imitation`
**実行**: `recipes/r19_tracking_reward.py`

```python
import torch
from ml_lern.core.imitation import tracking_reward, ReferenceMotionBuffer

# 見本のポーズに近いほど高い報酬
pose = torch.zeros(4, 6)
ref  = torch.zeros(4, 6)                    # 見本どおりなら満点
print("一致:", tracking_reward(pose, ref).round(3).tolist())
ref2 = ref + 0.3 * torch.randn(4, 6)        # 見本にノイズ
print("ずれ:", tracking_reward(pose, ref2).round(3).tolist())   # 報酬が下がる

# 見本モーションを貯めて、状態遷移（AMPの「本物」）を作る
buf = ReferenceMotionBuffer()
for t in range(10):
    buf.add_frame(torch.randn(30))
buf.finalize()
print("参照の遷移:", tuple(buf.ref_amp_obs.shape))   # (9, 60)
```

**実行すると**: 見本とのずれで報酬が下がり、参照モーションの遷移が作られます。
**発展**: 見本の体格を変えて（リターゲット）、再現度がどう落ちるか見る。

---

<a id="recipe-20"></a>
## レシピ20. AMP の識別器とスタイル報酬を動かす（第19章）

**もとになった「夢」**: MoCap で自然な敵の動きを作る。
**使うもの**: `core.amp`
**実行**: `recipes/r20_amp_style_reward.py`

```python
import torch
from ml_lern.core.amp import AMPUpdater, style_reward
from ml_lern.core.config import default_config

up = AMPUpdater(16, default_config())                     # amp_obs_dim=16
real = torch.randn(512, 16) - 1.0                         # 本物（データの遷移）
fake = torch.randn(512, 16) + 1.0                         # 偽物（方策の遷移）

for step in range(400):
    stats = up.update(fake, real)                         # 識別器を更新
    if step % 100 == 0:
        print(f"step {step}: acc_real={stats['disc_acc_real']:.2f}")

print("本物のスタイル報酬:", up.predict_style_reward(real[:3]).round(3).tolist())
print("偽物のスタイル報酬:", up.predict_style_reward(fake[:3]).round(3).tolist())
# 本物 > 偽物 になれば、識別器が「自然さ」を捉えている
```

**実行すると**: 本物のスタイル報酬が偽物より高くなります。
**発展**: `real` と `fake` を近づけて acc≈0.5 にし、「区別できない＝自然」を体験する。

---

<a id="recipe-21"></a>
## レシピ21. 識別器の正解率を監視してバランスを取る（第20章）

**もとになった「夢」**: 識別器を「自然さの採点器」として品質チェックに使う。
**使うもの**: `core.amp`
**実行**: `recipes/r21_discriminator_monitor.py`

```python
import torch
from ml_lern.core.amp import AMPUpdater
from ml_lern.core.config import default_config

# 識別器の学習率・勾配ペナルティの重みを変えて、正解率の推移を比べる
for lr, gp in [(1e-3, 10.0), (1e-4, 10.0), (1e-3, 0.0)]:
    cfg = default_config(disc_learning_rate=lr, gradient_penalty_weight=gp)
    up = AMPUpdater(16, cfg)
    real = torch.randn(256, 16) - 1.0
    fake = torch.randn(256, 16) + 1.0
    for _ in range(300):
        up.update(fake, real)
    d = torch.sigmoid(up.discriminator(real[:128])).mean().item()
    print(f"lr={lr}, gp={gp}: 本物の確信度={d:.2f} (0.5が健全)")
# lr が大きいほど識別器が過剰に強くなり、確信度が 1 に張り付く
```

**実行すると**: 学習率を上げると識別器が強くなりすぎる様子が見えます。
**発展**: 本物と偽物の分布の重なりを変えて、識別の難しさと正解率の関係を見る。

---

<a id="recipe-22"></a>
## レシピ22. タスク設計テンプレートを作る（第21章）

**もとになった「夢」**: 自分のゲームに「物理キャラが物体と相互作用する」場面を設計する。
**使うもの**: 標準ライブラリ
**実行**: `recipes/r22_task_design_template.py`

```python
# 論文の3タスクを自分のゲームに置き換えるためのテンプレート
TEMPLATE = {
    "タスク名": "",
    "物体":      "",   # 何を相手にするか（椅子/箱/ドア/旗…）
    "キャラの動き": "", # 近づく/座る/持ち上げる/押す…
    "成功条件":  "",   # 距離・時間・スコアなどの閾値
    "難しさの源": "",  # 物体の形・動き・接触
}

examples = [
    {"タスク名": "操縦席に着く", "物体": "コックピット",
     "キャラの動き": "歩いて近づき、座る", "成功条件": "腰が座面20cm以内",
     "難しさの源": "座面の高さ・狭さ"},
    {"タスク名": "旗を運ぶ", "物体": "旗（軽い箱）",
     "キャラの動き": "持ち上げて目的地へ", "成功条件": "旗が目標20cm以内",
     "難しさの源": "箱の大きさ・重さ"},
]
for i, ex in enumerate(examples, 1):
    print(f"--- タスク{i} ---")
    for k in TEMPLATE:
        print(f"  {k}: {ex.get(k, '')}")
```

**実行すると**: タスク設計のテンプレートと記入例が出ます。
**発展**: 自分のゲームのタスクを3つ書き、難易度順に並べて、対応章（第7部）を割り当てる。

---

<a id="recipe-23"></a>
## レシピ23. シーン条件付けの入力を組み立てる（第22章）

**もとになった「夢」**: 「この攻撃は壁の近くで自然か」のような文脈依存の判定を作る。
**使うもの**: PyTorch
**実行**: `recipes/r23_scene_conditioning_input.py`

```python
import torch

# 識別器の入力を「キャラの遷移」と「キャラ＋物体」で比べる
char = torch.randn(4, 30)          # キャラクタ状態
obj  = torch.randn(4, 9)           # 物体: 位置3 + 向き6D

amp_input = char                                   # AMP 本来
scene_input = torch.cat([char, obj], dim=-1)       # シーン条件付き
print("AMP 入力次元:", amp_input.shape[-1])          # 30
print("シーン入力次元:", scene_input.shape[-1])      # 39

# 「物体が無い」ケースを、位置を far にして表現する（遠い＝無いに近い）
obj_far = obj.clone(); obj_far[:, :3] += 100.0
scene_far = torch.cat([char, obj_far], dim=-1)
print("物体が遠いとき:", scene_far.shape)
```

**実行すると**: 入力次元が変わり、物体の状態を差し替えられることが分かります。
**発展**: 物体を「座れる面がある/ない」で2通り作り、識別器の判定が変わるかを実験する。

---

<a id="recipe-24"></a>
## レシピ24. 状態ベクトルを設計し local 変換する（第23章）

**もとになった「夢」**: 自分のゲームのキャラの「学習用の観測」を設計する。
**使うもの**: NumPy
**実行**: `recipes/r24_state_vector_design.py`

```python
import numpy as np

# 114次元相当の状態を、自分のキャラの関節数で組み立てる
N_JOINTS = 12
state = {
    "root_height": 1,              # world
    "root_rot_6d": 6,              # world
    "root_vel": 6,                 # local
    "joint_rot_6d": 6 * N_JOINTS,  # local
    "joint_vel": N_JOINTS,         # local
    "key_joints_pos": 4 * 3,       # 手足4点 (local)
    "object_pos": 3,               # local
    "object_rot_6d": 6,            # local
}
total = sum(state.values())
for k, v in state.items():
    print(f"{k:16s}: {v:4d}")
print("合計:", total)

# local 変換: 世界座標の点を、キャラ基準（回転・平行移動を戻す）に直す
def world_to_local(p_world, root_pos, root_yaw):
    # ヨー回転だけ戻す簡易版（実際はクォータニオン/行列で）
    c, s = np.cos(-root_yaw), np.sin(-root_yaw)
    d = p_world - root_pos
    return np.array([c*d[0] - s*d[1], s*d[0] + c*d[1], d[2]])

print("local 座標:", world_to_local(np.array([3,4,1.0]), np.array([1,1,0.0]), 0.5).round(2))
```

**実行すると**: 状態の内訳と合計次元、local 変換の結果が出ます。
**発展**: `N_JOINTS` を自分のキャラに合わせ、物体を増やして次元を拡張する。

---

<a id="recipe-25"></a>
## レシピ25. RSI と物体のランダム化（第24章）

**もとになった「夢」**: データが少ないときの「水増し」を設計する。
**使うもの**: PyTorch
**実行**: `recipes/r25_rsi_randomization.py`

```python
import torch
import math

# 物体を毎エピソード 1〜10m・ランダム向きに配置する（第24章）
def sample_object_placement(n, lo=1.0, hi=10.0):
    dist = torch.empty(n).uniform_(lo, hi)
    theta = torch.empty(n).uniform_(0.0, 2 * math.pi)
    x, y = dist * torch.cos(theta), dist * torch.sin(theta)
    return torch.stack([x, y, torch.full_like(x, 0.5)], dim=-1)   # 中心高さ0.5

print("配置例:\n", sample_object_placement(3).round(2))

# サイズのランダム化（carry は 0.5〜1.5倍）
scale = torch.empty(5).uniform_(0.5, 1.5)
print("スケール:", scale.round(2).tolist())

# RSI: 参照モーションのランダムなフレームを初期化に使う
ref = torch.randn(2000, 30)
starts = ref[torch.randint(0, len(ref), (4,))]
print("RSI 初期状態:", tuple(starts.shape))
```

**実行すると**: 物体の配置・スケール・RSI 初期状態が得られます。
**発展**: 配置範囲を狭める／広げて、学習の一般化がどう変わるか（第26章の評価で）測る。

---

<a id="recipe-26"></a>
## レシピ26. sit 報酬（式7-9）とモード切替（第25章）

**もとになった「夢」**: 「物体に近づいて操作」の報酬を設計する。
**使うもの**: `core.rewards`
**実行**: `recipes/r26_sit_reward.py`

```python
import torch
from ml_lern.core.rewards import sit_task_reward, exp_reward

# キャラの腰(root) と物体中心の距離に対する報酬
root = torch.zeros(6, 2)
obj = torch.tensor([[0.0,0.0],[0.3,0.0],[0.49,0.0],[0.51,0.0],[1.0,0.0],[5.0,0.0]])
r = sit_task_reward(root, obj, threshold=0.5)
for o, ri in zip(obj[:,0].tolist(), r.tolist()):
    print(f"距離 {o:4.2f}m -> 報酬 {ri:.3f}")   # 0.5m で構成が切り替わる

# k を変えると「厳しさ」が変わる
print("k=3 :", exp_reward(torch.tensor([0.2, 0.5]), k=3.0).round(3).tolist())
print("k=20:", exp_reward(torch.tensor([0.2, 0.5]), k=20.0).round(3).tolist())
```

**実行すると**: 0.5m 前後で報酬の傾向が変わり、$k$ で厳しさが変わるのが分かります。
**発展**: 速度項（物体方向の速度）を足して、静止の抜け道を塞ぐ。

---

<a id="recipe-27"></a>
## レシピ27. 成功率評価とアブレーション（第26章）

**もとになった「夢」**: アブレーション（情報を抜くと崩れる）で AI の弱点を見つける。
**使うもの**: PyTorch
**実行**: `recipes/r27_evaluate_ablation.py`

```python
import torch

# 簡単な「座り」モデルの評価（概念的なシミュレーション）
def success_rate(has_object_info=True, trials=1000):
    dist = torch.rand(trials) * 5.0                 # 目標までの距離
    if has_object_info:
        acc = torch.exp(-2.0 * dist)                 # 情報あり: 近づける
    else:
        acc = torch.full_like(dist, 0.5)             # 情報なし: 当てずっぽう
    return (acc > 0.5).float().mean().item()

print("物体情報あり:", success_rate(True))
print("物体情報なし:", success_rate(False))          # 成功率が落ちる

# アブレーションは「情報を1つ抜いて、指標がどう落ちるか」で本質を測る
```

**実行すると**: 情報を抜くと成功率が下がる様子が数値で見えます。
**発展**: 自分のタスクで「物体サイズ」「速度」「履歴」を1つずつ抜いて、指標の変化を記録する。

---

<a id="recipe-28"></a>
## レシピ28. 自分のロードマップを作る（第27章）

**もとになった「夢」**: 「ミニ実装→GPU→論文再現→自作」の道筋を自分の計画に落とす。
**使うもの**: 標準ライブラリ
**実行**: `recipes/r28_roadmap.py`

```python
# 自分の計算資源に応じて、現実的な到達点を計画する
STAGES = [
    ("ミニ実装(CPU)", "倒立振子〜Walker", "第29〜32章", "数分〜数時間"),
    ("GPU 導入",     "humanoid の AMP",   "第35章",     "数時間"),
    ("シーン条件付け", "物体との相互作用", "第36章",     "数時間〜"),
    ("IsaacGym",     "論文の本実装",       "第39章",     "数日"),
]
has_gpu = False   # この環境は False。自分の環境に合わせて書き換える
for name, goal, ch, time in STAGES:
    if "GPU" in name and not has_gpu:
        print(f"[保留] {name}: GPU が必要（{ch}）")
    else:
        print(f"[やる] {name}: {goal} / {ch} / 目安 {time}")
```

**実行すると**: GPU の有無で、実行する段階と保留する段階が分かれます。
**発展**: 各行に実際の締め切り・成果物を書き込み、プロジェクト計画にする。

---

<a id="recipe-29"></a>
## レシピ29. 環境構築をチェックする（第28章）

**もとになった「夢」**: 自分のマシンで学習環境を一式整える。
**使うもの**: PyTorch, MuJoCo, gymnasium
**実行**: `recipes/r29_env_check.py`

```python
# 依存が揃っているかを確認する（無ければ uv sync --extra train --extra viz）
try:
    import torch
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

# どのデバイスで学習するか
device = "cuda" if (torch.cuda.is_available() if 'torch' in dir() else False) else "cpu"
print("使うデバイス:", device)
```

**実行すると**: ライブラリのバージョンと、使うデバイス（cpu/cuda）が分かります。
**発展**: `num_envs` を変えて学習速度を測り、自分のマシンの限界を知る。

---

<a id="recipe-30"></a>
## レシピ30. 外乱つきの倒立振子で頑健さを試す（第29章）

**もとになった「夢」**: 外乱に強い制御（風、揺れ、攻撃）を設計する。
**使うもの**: gymnasium(MuJoCo)
**実行**: `recipes/r30_pendulum_disturbance.py`

```python
import gymnasium as gym
import numpy as np

env = gym.make("InvertedPendulum-v5")
env.reset(seed=0)

survived = 0
for step in range(500):
    action = env.action_space.sample() * 0.1
    # 一定確率で外乱（ランダムな力）を加える
    if np.random.rand() < 0.1:
        action = action + np.random.uniform(-0.5, 0.5, size=action.shape)
    _, _, term, trunc, _ = env.step(action)
    if term or trunc:
        break
    survived += 1
env.close()
print(f"外乱ありで {survived} ステップ持ちこたえた")
# 学習後は、外乱があっても長く持ちこたえるようになる
```

**実行すると**: ランダム方策は外乱で早く倒れます。学習後は改善します。
**発展**: 外乱の強さ・頻度を変えて、制御の頑健さを測る。

---

<a id="recipe-31"></a>
## レシピ31. 目標条件付きの Reacher を作る（第30章）

**もとになった「夢」**: 目標（敵・宝・スイッチ）を状態に入れ、任意の位置に対応する AI を作る。
**使うもの**: gymnasium, PyTorch
**実行**: `recipes/r31_goal_conditioned_reacher.py`

```python
import gymnasium as gym
import torch

env = gym.make("Reacher-v5")
obs, _ = env.reset(seed=0)

# Reacher の観測の末尾は「目標への相対座標」を含む
def split(obs):
    return obs[:-2], obs[-2:]          # (状態, 目標)

state, goal = split(torch.as_tensor(obs, dtype=torch.float32))
print("状態:", state.shape, "目標:", goal.shape)

# 方策の入力は「状態＋目標」。これで任意の目標に対応できる。
policy_input = torch.cat([state, goal])
print("方策の入力次元:", policy_input.shape[-1])

# 目標を変えると（reset のたびに変わる）、入力が変わり行動が変わる
env.close()
```

**実行すると**: 状態と目標が分離でき、方策の入力が「状態＋目標」になることが分かります。
**発展**: 目標を毎ステップランダムに差し替え、「任意の目標への追従」を学習させる。

---

<a id="recipe-32"></a>
## レシピ32. 接地と速度報酬（Hopper）（第31章）

**もとになった「夢」**: ジャンプ＆ランディングを物理的に自然に作る。
**使うもの**: gymnasium, `core.rewards`
**実行**: `recipes/r32_hopper_contact.py`

```python
import gymnasium as gym
import torch
from ml_lern.core.rewards import task_reward_locomotion

env = gym.make("Hopper-v5")
obs, _ = env.reset(seed=0)

# 速度報酬で「前進」を促す（論文の r^far の速度項と同型）
for step in range(100):
    obs, _, term, trunc, _ = env.step(env.action_space.sample())
    if step % 20 == 0:
        o = torch.as_tensor(obs, dtype=torch.float32)
        r = task_reward_locomotion(o.unsqueeze(0), target_speed=1.5).item()
        print(f"step {step}: 速度ベース報酬={r:.3f}, 高さ={obs[0]:.2f}")
    if term or trunc:
        break
env.close()
# 摩擦や timestep を変えると、接地の安定性が変わる
```

**実行すると**: ランダム行動でも、速度と高さに応じた報酬が出ます。
**発展**: 摩擦を 0 にして滑りを観察し、接地の重要性を体感する。

---

<a id="recipe-33"></a>
## レシピ33. 歩行と報酬ハッキングを観察する（第32章）

**もとになった「夢」**: 二足・四足キャラの移動 AI を作る。
**使うもの**: gymnasium, `core.rewards`
**実行**: `recipes/r33_gait_reward_hacking.py`

```python
import gymnasium as gym
import torch
from ml_lern.core.rewards import task_reward_locomotion

env = gym.make("Walker2d-v5")
obs, _ = env.reset(seed=0)

# 速度報酬だけを最大化すると「跳ねる」などの不自然な動きが出やすい。
# ここでは報酬の内訳（速度/直立/生存）を観察する。
for step in range(200):
    obs, _, term, trunc, _ = env.step(env.action_space.sample())
    if step % 50 == 0:
        o = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        total = task_reward_locomotion(o).item()
        speed_r = torch.exp(-2*(1.5-o[0,0])**2).item()
        print(f"step {step}: 合計={total:.3f} 速度項={speed_r:.3f}")
    if term or trunc:
        break
env.close()
```

**実行すると**: 速度・直立・生存の各項がどう合わさるかが見えます。
**発展**: 直立報酬の重みを 0 にして姿勢が崩れるか、順に確かめる。

---

<a id="recipe-34"></a>
## レシピ34. 自作タスクで PPO 学習ループを組む（第33章）

**もとになった「夢」**: 自分のゲームの制御タスクを PPO で解く。
**使うもの**: `core.ppo`, `core.models`
**実行**: `recipes/r34_ppo_loop.py`

```python
import torch
from ml_lern.core.ppo import PPOUpdater, compute_gae
from ml_lern.core.config import default_config
from ml_lern.core.models import Policy, Value

cfg = default_config()
# 自作タスクの次元に合わせて、方策と価値関数を作る
policy = Policy(obs_dim=10, goal_dim=1, action_dim=2, **{})
value = Value(obs_dim=10, goal_dim=1)

# PPO の更新に必要な材料（バッファ）を作る
obs, goal = torch.randn(32, 10), torch.randn(32, 1)
actions, logp = policy.act(obs, goal)
values = value(obs, goal)
rewards = torch.randn(32)
dones = torch.zeros(32)
adv, ret = compute_gae(rewards, values, dones, cfg.gamma, cfg.gae_lambda, torch.zeros(1))
print("アドバンテージ:", tuple(adv.shape), "収益:", tuple(ret.shape))
print("行動:", tuple(actions.shape), "対数確率:", tuple(logp.shape))
# 実際の学習は train.py のループに、自作環境を差し込んで回す
```

**実行すると**: PPO に必要なデータの形が確認できます。
**発展**: 自分で集めたロールアウトで `PPOUpdater.update` を呼び、損失を確認する。

---

<a id="recipe-35"></a>
## レシピ35. スタイル重みを変えて比べる（第34章）

**もとになった「夢」**: スタイル（重い/軽いなど）を切り替えられるキャラを作る。
**使うもの**: `core.config`
**実行**: `recipes/r35_style_weight_compare.py`

```python
from ml_lern.core.config import default_config

# スタイル報酬の重み w^S を変えて、「自然さ」と「目的達成」のバランスを考える
for w_style in [0.0, 0.2, 0.5, 0.8, 1.0]:
    cfg = default_config(style_reward_weight=w_style)
    print(f"w^S={cfg.style_reward_weight}: タスク重み={cfg.task_reward_weight}, "
          f"勾配ペナルティ={cfg.gradient_penalty_weight}")
# w^S=0 は不自然（跳ねる）、大きすぎると目的を疎かにする
```

**実行すると**: 重みの組み合わせが一覧できます。
**発展**: 実際に `train` を回し、`w^S` ごとに動きと成功率を比べる（第38章の評価）。

---

<a id="recipe-36"></a>
## レシピ36. 人型で AMP を統合する（第35章）

**もとになった「夢」**: ゲームの人型キャラを物理的に自然に歩かせる。
**使うもの**: `charcontrol.envs.humanoid_env`, `charcontrol.train`
**実行**: `recipes/r36_humanoid_amp.py`

```python
from ml_lern.charcontrol.train import make_env
from ml_lern.core.config import default_config

cfg = default_config(num_envs=8, episode_length_s=10.0)
env = make_env("humanoid_amp", cfg, device="cpu")

print("状態次元:", env.obs_dim, "行動次元:", env.action_dim, "AMP遷移:", env.amp_obs_dim)
obs = env.reset()
print("初期観測:", tuple(obs.shape), "目標:", tuple(env.goal.shape))
env.close()

# 実際の学習は scripts/train_05_humanoid_amp.py で回す（時間がかかる）
print("学習: uv run python scripts/train_05_humanoid_amp.py")
```

**実行すると**: 人型環境の状態・行動・AMP遷移の次元が分かります。
**発展**: 参照モーションを「歩行」「走行」に差し替えて、歩容が変わるか見る。

---

<a id="recipe-37"></a>
## レシピ37. 物体を足してシーン条件付き識別器にする（第36章）

**もとになった「夢」**: プレイヤーが置いた物体と相互作用する AI を作る。
**使うもの**: `charcontrol.envs.scene_env`
**実行**: `recipes/r37_scene_discriminator.py`

```python
from ml_lern.charcontrol.train import make_env
from ml_lern.core.config import default_config

cfg = default_config(num_envs=4, episode_length_s=15.0)
env = make_env("scene", cfg, device="cpu")
obs = env.reset()
goal = env.goal

# シーン条件付き識別器の入力（キャラ遷移＋物体状態）を確認
amp = env.get_amp_obs(obs, goal)
print("状態:", tuple(obs.shape), "物体込みAMP遷移:", tuple(amp.shape))
print("目標(物体の中心＋bbox):", tuple(goal.shape))

# 物体を押したときの反応を見るには、実際に step を回す
env.close()
```

**実行すると**: 物体込みの識別器入力の次元が確認できます。
**発展**: `get_amp_obs` から物体を除いた版に切り替えて、挙動の違いを比べる。

---

<a id="recipe-38"></a>
## レシピ38. AMASS の読み込みとリターゲット（第37章）

**もとになった「夢」**: 市販の MoCap をリターゲットして自分のキャラに流用する。
**使うもの**: NumPy（データ実体はライセンス登録が必要）
**実行**: `recipes/r38_amass_retarget.py`

```python
import numpy as np

# AMASS の SMPL 形式データの構造（実際はライセンス登録のうえ取得）
AMASS_KEYS = ["trans", "poses", "betas", "mocap_framerate"]
print("AMASS の典型キー:", AMASS_KEYS)
print("  trans: (フレーム, 3)  ルート位置")
print("  poses: (フレーム, 72) ルート3 + 関節23×3（軸角）")
print("  betas: (10,)          体型パラメータ")

# リターゲットの流れ（概念）
def retarget(amass_pose_aa, beta):
    """SMPL の関節角を、MuJoCo humanoid の関節角へフィットする。
    実際には順運動学と最適化（PHC の fit_smpl_motion.py 相当）を行う。"""
    # 1) 関節の対応付け（SMPL <-> robot）
    # 2) 体型 beta をロボットに合わせて最適化
    # 3) 各フレームで最適化して関節角を得る
    return np.zeros((20,))          # ここではダミー

J = 23
pose_aa = np.zeros((100, J * 3))
q = retarget(pose_aa, beta=np.zeros(10))
print("リターゲット後の関節角:", q.shape)
```

**実行すると**: AMASS のデータ構造とリターゲットの手順が確認できます。
**発展**: 実際の AMASS をダウンロードし、PHC の手順で自分の humanoid に流し込む。

---

<a id="recipe-39"></a>
## レシピ39. 評価関数と外乱テスト（第38章）

**もとになった「夢」**: 自分のゲーム AI の品質を、数値で継続的に測る。
**使うもの**: PyTorch
**実行**: `recipes/r39_evaluate_disturbance.py`

```python
import torch

def success_rate(disturbance=0.0, trials=2000):
    """距離と外乱から成功率を推定する（概念的な評価）。"""
    dist = torch.rand(trials) * 5.0
    noise = disturbance * torch.randn(trials)
    acc = torch.exp(-2.0 * (dist + noise.abs()))
    return (acc > 0.5).float().mean().item()

for d in [0.0, 0.5, 1.0, 2.0]:
    print(f"外乱 {d}: 成功率={success_rate(d):.3f}")
# 外乱が大きいほど成功率が下がる（論文の表2のような頑健性テスト）
```

**実行すると**: 外乱の大きさと成功率の関係が見えます。
**発展**: 実際の環境で、投射物・物体移動などの外乱を加えて成功率を測る。

---

<a id="recipe-40"></a>
## レシピ40. プロジェクト計画を作る（第39章）

**もとになった「夢」**: 拡散（AI側）と AMP（物理側）をどう組み合わせるかを設計する。
**使うもの**: 標準ライブラリ
**実行**: `recipes/r40_project_plan.py`

```python
# 2つの柱を自分のプロジェクトにどう組み合わせるかの計画テンプレート
PLAN = {
    "作りたいもの": "（例: プレイヤーの周りで自然に動く群衆）",
    "AI側（生成）":  "（例: 拡散で歩行バリエーションを生成）",
    "物理側（制御）": "（例: AMPで物理キャラに追従させる）",
    "データ":        "（例: 自作MoCap / 公開アセット）",
    "計算資源":      "（CPUのみ / GPUあり）",
    "最初の一歩":    "（例: 第10章の拡散で1本作る）",
}
for k, v in PLAN.items():
    print(f"{k:14s}: {v}")

# 組み合わせパターン
COMBOS = [
    "生成 -> 追従: 拡散で作った動きをモーショントラッキングで再生",
    "方策として生成: Diffusion Policy（第10章）",
    "事前分布として: 生成モデルの分布を AMP に使う",
]
print("\n組み合わせの型:")
for c in COMBOS:
    print(" -", c)
```

**実行すると**: プロジェクト計画の枠と、AI側×物理側の組み合わせ方が出ます。
**発展**: これに沿って、第7部の実装を自分のタスクへ移植する。

---

[目次に戻る](README.md) ／ AI側: [付録E](付録E_再現レシピ_AI側.md)
