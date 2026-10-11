# 付録E 再現レシピ（AI側） — 第0〜2部

この付録は、各章の **「夢が広がる」** を実際に手で再現するための、**コメント多めのレシピ集**です。

- 各レシピに対応する**実行できるコード**は `recipes/` にあります（例: `recipes/r03_autograd_optimize.py`）。
- 実行は `uv run python recipes/rNN_....py`（`uv sync --extra train --extra viz` 済み）。
- 物理側（第3〜7部）のレシピは [付録F](付録F_再現レシピ_物理側.md) にあります。
- コードは読者の環境で動かす前提で、本リポジトリでは実行・検証していません。

> リンク用のアンカーは HTML の `id`（例: `recipe-12`）で明示しているので、docsite でも GitHub でもジャンプできます。

---

<a id="recipe-01"></a>
## レシピ01. 環境をのぞいて、作りたいものをマッピング（第00章）

**もとになった「夢」**: 自分のゲームの「学習で作りたい動作」を、AI側か物理側かに仕分ける。
**使うもの**: `core.mujoco_basics`, `foundations.tensors_demo`
**実行**: `recipes/r01_environment_tour.py`

```python
# 1) 物理側の入口: MuJoCo の各環境の「状態・行動の次元」を確認する。
from ml_lern.core.mujoco_basics import inspect_spaces
inspect_spaces("InvertedPendulum-v5")   # 一番やさしい: 状態4, 行動1
inspect_spaces("Walker2d-v5")           # 二足歩行: 行動6
inspect_spaces("Humanoid-v5")           # 人型: 行動17前後

# 2) AI側の入口: テンソルと自動微分の感触。
from ml_lern.foundations.tensors_demo import autograd_demo
autograd_demo()   # dy/dx が自動で求まる

# 3) マッピング（自作）: 作りたい動作 -> どこで学ぶか。
TASKS = {
    "敵が自然に歩く":       ("物理側", "第32/35章", "RL/AMPで制御"),
    "アニメの補間を自動化": ("AI側",   "第06/10章", "生成モデル"),
    "椅子に座る":           ("物理側", "第36章",    "シーン条件付き識別器"),
}
for name, (side, ch, how) in TASKS.items():
    print(f"{name}: {side} / {ch} / {how}")
```

**実行すると**: 各環境の次元と、テンソルの勾配、タスク→章の対応表が出ます。
**発展**: 自分のゲームの動作を10個リストアップし、上のような表にする。

---

<a id="recipe-02"></a>
## レシピ02. キャラの状態をベクトル化して正規化（第01章）

**もとになった「夢」**: ゲームのキャラ状態を、そのまま学習の入力にできる形へ変換する。
**使うもの**: NumPy
**実行**: `recipes/r02_state_vector.py`

```python
import numpy as np

# 1フレームの状態 = 位置(3) + 回転クォータニオン(4) + 関節角(20)
D = 3 + 4 + 20
state = np.zeros(D)
state[:3] = [10.0, 0.0, 2.0]        # ワールド座標の位置
state[3:7] = [1, 0, 0, 0]            # 回転（クォータニオン）
state[7:] = np.random.uniform(-1, 1, size=20)   # 関節角（ラジアン）

# 1000フレーム分をまとめる -> (フレーム, 特徴)
motion = np.random.randn(1000, D)
print("motion.shape =", motion.shape)          # (1000, 27)

# スライス: 位置の時系列だけ取り出す
pos_series = motion[:, 0:3]
print("位置の時系列:", pos_series.shape)        # (1000, 3)

# ブロードキャストで特徴ごとに正規化（平均0・分散1）
mu = motion.mean(axis=0)                        # (27,)
sigma = motion.std(axis=0)                      # (27,)
normalized = (motion - mu) / (sigma + 1e-8)     # (1000,27) に一括適用
print("正規化後の平均:", np.round(normalized.mean(0), 5)[:3])
```

**実行すると**: 状態の形、スライス、正規化後の平均（ほぼ0）が確認できます。
**発展**: 関節速度を足して次元を倍にし、`(キャラ数, 時間, 特徴)` の3次元に拡張する。

---

<a id="recipe-03"></a>
## レシピ03. ゲームの数式を自動微分して最適化（第02章）

**もとになった「夢」**: バランス式や砲弾の軌道を、勾配で自動調整する。
**使うもの**: PyTorch（自動微分）
**実行**: `recipes/r03_autograd_optimize.py`

```python
import torch

# ジャンプ到達高さ h(v) = v^2 / (2g)。v を勾配で最大化する。
g = 9.8
v = torch.tensor(1.0, requires_grad=True)   # 初期速度（学習する変数）
lr = 0.5

for step in range(30):
    h = v**2 / (2*g)                        # 到達高さ
    h.backward()                            # dh/dv を自動計算 -> v.grad
    with torch.no_grad():
        v += lr * v.grad                    # 勾配の向き（h が増える向き）へ動かす
        v.grad.zero_()                      # 勾配をリセット（必須）
    if step % 10 == 0:
        print(f"step {step}: v={v.item():.2f} h={h.item():.2f}")

# 制約つき最適化の例: 「h を 3.0 にしたい」→ 二乗誤差を最小化する
v2 = torch.tensor(1.0, requires_grad=True)
for _ in range(200):
    loss = ((v2**2 / (2*g)) - 3.0) ** 2     # 目標 3.0 との差
    loss.backward()
    with torch.no_grad():
        v2 -= 0.05 * v2.grad
        v2.grad.zero_()
print("h=3.0 にする v:", round(v2.item(), 3))   # sqrt(2*9.8*3)≒7.67
```

**実行すると**: `v` が増えるほど `h` が増え、目標高さにする `v` が求まります。
**発展**: ダメージ式を「狙ったTTK（撃破時間）にする」ようにパラメータ最適化する。

---

<a id="recipe-04"></a>
## レシピ04. 成長曲線を MLP で学習し、未見レベルを予測（第03章）

**もとになった「夢」**: 手書きのカーブをデータからフィットし、表にない値も予測する。
**使うもの**: PyTorch（`foundations.nn_basics.MLP`）
**実行**: `recipes/r04_mlp_growth_curve.py`

```python
import torch
from ml_lern.foundations.nn_basics import MLP

# レベル(1..50) -> 必要経験値（仮の成長曲線）
level = torch.linspace(1, 50, 50).unsqueeze(1)     # (50,1)
exp = 100 * level**1.5                              # 真のカーブ
exp_n = exp / exp.max()                             # 0..1 に正規化（学習を安定化）

net = MLP(1, hidden=(64, 64), out_dim=1)
opt = torch.optim.Adam(net.parameters(), lr=1e-2)
for step in range(3000):
    pred = net(level / 50)                          # 入力も正規化
    loss = ((pred - exp_n) ** 2).mean()
    opt.zero_grad(); loss.backward(); opt.step()

# 表にないレベル（37）を予測
with torch.no_grad():
    pred37 = net(torch.tensor([[37/50]])).item() * exp.max().item()
print("レベル37の予測経験値:", round(pred37))
print("真値:", round(100 * 37**1.5))
```

**実行すると**: 未見レベルの予測値が真値に近くなります。
**発展**: 隠れ層を `(4,4)` にして精度が落ちること、入力正規化を外すと学習が乱れることを確認する。

---

<a id="recipe-05"></a>
## レシピ05. アニメを窓に切ってミニバッチ化（第04章）

**もとになった「夢」**: ゲームのアニメを統一フォーマットにして、学習・検索できるようにする。
**使うもの**: `foundations.timeseries`
**実行**: `recipes/r05_motion_windows.py`

```python
import torch
from ml_lern.foundations.timeseries import make_windows, normalize, make_loader, pad_batch

# 1つのモーション (フレーム, 関節×特徴)
motion = torch.randn(300, 22 * 3)                  # 300フレーム, 22関節×xyz

# 固定長の窓に切り出す（拡散やNNの入力に）
w = make_windows(motion, window=64, stride=16)     # (N, 64, 66)
print("窓の形:", w.shape)

# 特徴ごとに正規化（統計量は訓練データから）
norm, mu, sigma = normalize(w)

# ミニバッチ化
loader = make_loader(w, batch_size=8)
print("バッチ:", next(iter(loader)).shape)          # (8, 64, 66)

# 長さの違うモーションを揃える（パディング＋マスク）
a, b = torch.randn(100, 6), torch.randn(150, 6)
padded, mask = pad_batch([a, b])
print("padded:", padded.shape, "mask:", mask.shape) # (2,150,6), (2,150)
```

**実行すると**: 窓の数、バッチの形、パディングとマスクの形が確認できます。
**発展**: `stride` を変えてサンプル数と学習時間のトレードオフを体感する。

---

<a id="recipe-06"></a>
## レシピ06. 次フレームのポーズを予測する（第05章）

**もとになった「夢」**: 通信遅延でカクつく他プレイヤーの姿勢を、予測で補間する。
**使うもの**: `foundations.sequence_models`
**実行**: `recipes/r06_pose_prediction.py`

```python
import torch
from ml_lern.foundations.sequence_models import RNNPredictor

feat = 12                                     # 関節4×xyz など
model = RNNPredictor(feat, hidden=64, out_dim=feat)
opt = torch.optim.Adam(model.parameters(), lr=1e-2)

# ダミーの時系列: 直近20フレーム -> 次の1フレーム
for step in range(500):
    x = torch.randn(16, 20, feat)             # (バッチ, 時間, 特徴)
    y = torch.randn(16, feat)
    loss = ((model(x) - y) ** 2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
print("予測の形:", model(x).shape)              # (16, 12)

# 比較: フレームスタッキング / 1D CNN / Transformer に差し替えて精度を比べる
from ml_lern.foundations.sequence_models import FrameStackMLP, ConvPredictor, TransformerPredictor
for name, m in [("stack", FrameStackMLP(feat, out_dim=feat)),
                ("conv", ConvPredictor(feat, out_dim=feat)),
                ("tf", TransformerPredictor(feat, out_dim=feat))]:
    print(name, m(x).shape)
```

**実行すると**: 各モデルの出力形状が同じであることが確認できます。
**発展**: 予測をロールアウト（予測を入力に戻す）して、何フレーム先まで破綻しないか測る。

---

<a id="recipe-07"></a>
## レシピ07. VAE でモーションを補間する（第06章）

**もとになった「夢」**: 「歩く」と「走る」の中間の動きを自動生成する。
**使うもの**: `generative.autoencoder.VAE`
**実行**: `recipes/r07_vae_motion_interp.py`

```python
import torch
from ml_lern.generative.autoencoder import VAE, vae_loss

D = 66                                        # モーション1本を1ベクトルに
motion = torch.randn(500, D)                  # 500本（ダミー）
vae = VAE(in_dim=D, latent=8)
opt = torch.optim.Adam(vae.parameters(), lr=1e-3)
for _ in range(2000):
    x_hat, mu, logvar, z = vae(motion)
    loss, recon, kl = vae_loss(x_hat, motion, mu, logvar)
    opt.zero_grad(); loss.backward(); opt.step()

# 2つの動きを潜在空間で混ぜる -> 中間モーション
mu1 = vae.encode(motion[0])[0]
mu2 = vae.encode(motion[1])[0]
for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
    z = (1 - alpha) * mu1 + alpha * mu2       # 線形補間
    mid = vae.decoder(z)
    print(f"alpha={alpha}: 中間モーション {tuple(mid.shape)}")
```

**実行すると**: 補間の各段階のモーションが得られます。
**発展**: `latent=2` にして潜在空間を2Dプロットし、動きの「地形」を見る。

---

<a id="recipe-08"></a>
## レシピ08. GAN で本物らしいデータを作る（第07章）

**もとになった「夢」**: 敵のモーションを識別器に学ばせ、自然さの報酬にする（→ AMP）。
**使うもの**: `generative.gan`
**実行**: `recipes/r08_gan_2d.py`

```python
import torch
from ml_lern.generative.gan import Generator, Discriminator, gradient_penalty

# 本物データ（ここでは8クラスタ）。自分のゲームのデータに置き換えてよい。
centers = torch.tensor([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]])
real_data = centers[torch.randint(0,4,(2000,))] + 0.1*torch.randn(2000,2)

G = Generator(z_dim=16, out_dim=2)
D = Discriminator(in_dim=2)
optG = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5,0.9))
optD = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5,0.9))
bce = torch.nn.BCEWithLogitsLoss()

for step in range(3000):
    idx = torch.randint(0, len(real_data), (256,))
    real = real_data[idx]
    fake = G(torch.randn(256, 16)).detach()
    # 識別器: 本物を1、偽物を0に。勾配ペナルティで安定化。
    lossD = bce(D(real), torch.ones(256)) + bce(D(fake), torch.zeros(256))
    lossD = lossD + 10.0 * gradient_penalty(D, real)
    optD.zero_grad(); lossD.backward(); optD.step()
    # 生成器: 識別器に「本物」と言わせる。
    fake = G(torch.randn(256, 16))
    lossG = bce(D(fake), torch.ones(256))
    optG.zero_grad(); lossG.backward(); optG.step()
    if step % 500 == 0:
        acc = (torch.sigmoid(D(real)) > 0.5).float().mean().item()
        print(f"step {step}: lossD={lossD.item():.3f} acc_real={acc:.2f}")
```

**実行すると**: `acc_real` が 0.5 前後なら健全、1.0 に張り付くなら識別器が強すぎます。
**発展**: 勾配ペナルティを外して不安定さを見る。識別器を「動きの自然さ採点器」にする。

---

<a id="recipe-09"></a>
## レシピ09. 前方過程（ノイズ付加）を観察する（第08章）

**もとになった「夢」**: ゲームのレベル形状を「ノイズから生成」する発想へ拡張する。
**使うもの**: `generative.diffusion.schedule`
**実行**: `recipes/r09_forward_diffusion.py`

```python
import torch
from ml_lern.generative.diffusion.schedule import NoiseSchedule

sched = NoiseSchedule(num_timesteps=100, beta_start=1e-4, beta_end=0.02)

# きれいなデータ: 直線上に並んだ点（例えるなら「整ったレベル形状」）
x0 = torch.zeros(64, 2)
x0[:, 0] = torch.linspace(-1, 1, 64)

for t in [0, 10, 30, 60, 99]:
    tt = torch.full((64,), t)
    noise = torch.randn_like(x0)
    x_t = sched.q_sample(x0, tt, noise)          # ノイズを加えた状態
    print(f"t={t:3d}: 標準偏差={x_t.std():.2f}  平均={x_t.mean():.2f}")
# t が大きいほど「構造」が失われ、ノイズに近づく
```

**実行すると**: `t` が増えるほどばらつきが増え、$t=99$ でほぼノイズになります。
**発展**: `beta_end` を大きくして、少ないステップでノイズに潰れるようにする。

---

<a id="recipe-10"></a>
## レシピ10. 自分の2D軌道で拡散モデルを作る（第09章）

**もとになった「夢」**: プレイヤーの操作軌道を学習し、それらしい動きをする AI を作る。
**使うもの**: `generative.diffusion`
**実行**: `recipes/r10_diffusion_trajectory.py`

```python
import numpy as np, torch
from ml_lern.generative.diffusion.schedule import NoiseSchedule
from ml_lern.generative.diffusion.model import Denoiser

# 1) 自分のデータ: ここでは「画面を横切る放物線」を生成する
def make_traj(steps=32, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, steps)
    y0 = rng.uniform(0.2, 0.8)
    v = rng.uniform(1.0, 2.0)
    y = y0 + v * t - 2.0 * t**2                     # 放物線
    return np.stack([t, y], axis=-1).astype(np.float32)

data = torch.tensor(np.stack([make_traj(seed=i) for i in range(2000)]))  # (N,32,2)
starts = data[:, 0, :].clone()

# 2) 学習: ノイズを当てるだけ
steps, sched = 32, NoiseSchedule(100)
model = Denoiser(data_dim=steps*2, cond_dim=2)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
flat = data.reshape(len(data), -1)
for epoch in range(150):
    idx = torch.randint(0, len(data), (128,))
    x0, cond = flat[idx], starts[idx]
    t = torch.randint(0, 100, (128,))
    noise = torch.randn_like(x0)
    x_t = sched.q_sample(x0, t, noise)
    loss = ((model(x_t, t, cond) - noise) ** 2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
print("最終 loss:", round(loss.item(), 4))

# 3) 生成（初期位置を条件に）: 詳しいサンプリングは sample.py を参照
from ml_lern.generative.diffusion.sample import sample_conditional
out = sample_conditional(model, sched, steps, starts[:3], seed=0)
print("生成軌道:", tuple(out.shape))              # (3, 32, 2)
```

**実行すると**: 条件（初期位置）に合った放物線軌道が生成されます。
**発展**: `make_traj` を自分のデータ（マウス入力、弾道）に置き換える。

---

<a id="recipe-11"></a>
## レシピ11. 関節軌道の拡散でモーションを生成する（第10章）

**もとになった「夢」**: 「この2ポーズの間を埋めて」で中間モーションを自動生成する。
**使うもの**: `generative.diffusion.motion`
**実行**: `recipes/r11_motion_diffusion.py`

```python
import torch
from ml_lern.generative.diffusion.motion import (
    train_motion_diffusion, sample_motion, make_motion_dataset)

frames, num_joints = 32, 4
feat = num_joints * 3

# 学習（条件 = 初期姿勢）
model, sched, (frames, feat) = train_motion_diffusion(
    num_samples=2000, frames=frames, num_joints=num_joints,
    num_timesteps=100, num_epochs=80)

# 生成: 初期姿勢を固定したまま続きを作る（inpainting）
data = make_motion_dataset(num_samples=3, frames=frames, num_joints=num_joints, seed=1)
motion = sample_motion(model, sched, frames, feat, data["first_pose"], seed=0)
print("生成モーション:", tuple(motion.shape))                 # (3, 32, 12)
print("第1フレームが指定通り:", torch.allclose(motion[:,0], data["first_pose"], atol=1e-4))
```

**実行すると**: 指定した初期姿勢から続くモーションが生成されます。
**発展**: 条件に「ゴール姿勢」も足し、始点と終点の間を生成させてみる（アニメの補間）。

---

[目次に戻る](README.md) ／ 物理側: [付録F](付録F_再現レシピ_物理側.md)
