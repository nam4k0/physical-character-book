# 付録B 数学の補足

この本で使う数学を、必要最小限だけまとめます。「式が出てきたときに戻る辞書」として使ってください。

## 1. 確率

| 記号 | 意味 |
| --- | --- |
| $p(x)$ | $x$ が生じる確率（または確率密度） |
| $\mathbb{E}[x]$ | 期待値（平均） |
| $\mathrm{Var}[x]$ | 分散（ばらつき） |
| $\sigma$ | 標準偏差（分散の平方根） |
| $\mathcal{N}(\mu, \sigma^2)$ | 平均 $\mu$・分散 $\sigma^2$ の正規分布 |

### 期待値と分散

$$
\mathbb{E}[x] = \sum_x x\,p(x), \qquad
\mathrm{Var}[x] = \mathbb{E}\big[(x - \mathbb{E}[x])^2\big]
$$

- 期待値は「平均的にどのくらいの値か」。
- 分散は「どれだけばらつくか」。

### 正規分布（ガウス分布）

$$
\mathcal{N}(x \mid \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
$$

- 釣鐘型の分布。本で扱う方策（第14章）や拡散モデル（第8章）の基礎です。

正規分布の足し算（拡散モデルで使用）:

$$
x \sim \mathcal{N}(\mu_1, \sigma_1^2), \quad y \sim \mathcal{N}(\mu_2, \sigma_2^2)
\ \Rightarrow\ x + y \sim \mathcal{N}(\mu_1 + \mu_2,\ \sigma_1^2 + \sigma_2^2)
$$

- 正規分布同士を足すと、また正規分布。拡散の前方過程（第8章）で使います。

### KLダイバージェンス

2つの分布の「近さ」を測る量です（第6章のVAE）。

$$
D_{\text{KL}}(q \,\|\, p) = \mathbb{E}_{x\sim q}\!\left[\log \frac{q(x)}{p(x)}\right]
$$

- $q = p$ なら 0。離れるほど大きい（非負）。
- 正規分布同士のKLは閉じた式で書けます（VAE の実装で使用）。

## 2. 微分と勾配

| 記号 | 意味 |
| --- | --- |
| $\frac{df}{dx}$ | $f$ の $x$ での微分（傾き） |
| $\nabla_\theta f$ | パラメータ $\theta$ に関する勾配（各成分の偏微分） |

### 微分の直感

「$x$ を少し増やしたとき、$f$ がどれだけ増えるか」＝傾き。これが勾配降下法（第3章）の基礎です。

$$
\theta \leftarrow \theta - \eta \nabla_\theta f
$$

### よく使う微分

$$
\frac{d}{dx} x^2 = 2x, \quad
\frac{d}{dx} e^x = e^x, \quad
\frac{d}{dx} \log x = \frac{1}{x}, \quad
\frac{d}{dx} \sin x = \cos x
$$

### 連鎖律（チェーンルール）

合成関数の微分は、掛け算で分解できます。これが**誤差逆伝播**の原理です。

$$
\frac{d}{dx} f(g(x)) = f'(g(x)) \cdot g'(x)
$$

- 例: $\frac{d}{dx} e^{2x} = e^{2x} \cdot 2 = 2e^{2x}$

`loss.backward()`（第2章）は、この連鎖律を自動で適用しています。

### 指数関数の性質

$$
e^{a+b} = e^a e^b, \qquad e^0 = 1, \qquad e^{-\infty} \to 0
$$

- 報酬の $\exp(-k d^2)$（第17章）は、$d=0$ で 1、離れると 0 に近づく、なめらかな関数です。

## 3. 線形代数

| 記号 | 意味 |
| --- | --- |
| $\mathbf{x} \in \mathbb{R}^n$ | $n$ 次元ベクトル |
| $\mathbf{A} \in \mathbb{R}^{m\times n}$ | $m$ 行 $n$ 列の行列 |
| $\mathbf{x}^\top \mathbf{y}$ | 内積（ドット積） |
| $\mathbf{A}\mathbf{x}$ | 行列とベクトルの積 |

### 内積

$$
\mathbf{x} \cdot \mathbf{y} = \sum_i x_i y_i = \|\mathbf{x}\|\,\|\mathbf{y}\|\cos\theta
$$

- 「2つのベクトルが同じ向きにどれだけ揃っているか」。
- 論文本体の速度項（第25章の式(8)）で、「物体方向の速度成分」を計算するのに使います。

### 行列積

$$
(\mathbf{A}\mathbf{B})_{ij} = \sum_k A_{ik} B_{kj}
$$

- ニューラルネットの全結合層（第3章）は $\mathbf{y} = \mathbf{W}\mathbf{x} + \mathbf{b}$。

### ノルム

$$
\|\mathbf{x}\| = \sqrt{\sum_i x_i^2}
$$

- ベクトルの長さ＝距離。報酬の $\|x - x^*\|^2$（第25章）は「距離の二乗」です。

## 4. 記号まとめ

| 記号 | 意味 |
| --- | --- |
| $\mathbb{E}[\cdot]$ | 期待値 |
| $\nabla$ | 勾配 |
| $\sim$ | 「〜からサンプルする」 |
| $\mid$ | 条件付き（$p(x\mid y)$ は $y$ が与えられたときの $x$） |
| $\mathcal{N}$ | 正規分布 |
| $\prod$ | 総積 |
| $\sum$ | 総和 |
| $\arg\min / \arg\max$ | 最小/最大にする引数 |
| $\propto$ | 比例 |
| $\in$ | 要素である |
| $\mathbb{R}^n$ | $n$ 次元実数ベクトル |

## 5. どこで使うか（逆引き）

| 数学 | 使う章 |
| --- | --- |
| 期待値・分散 | 方策・価値（第14章）、正規化（第4・36章） |
| 正規分布 | 方策（第14章）、拡散（第8章）、VAE（第6章） |
| KLダイバージェンス | VAE（第6章） |
| 微分・連鎖律 | 学習全般（第3章） |
| 指数関数 | 報酬設計（第17・25章） |
| 内積 | 速度項（第25章の式8） |
| 行列積 | 全結合層（第3章） |
| ノルム | 距離報酬（第25章） |

---

[目次に戻る](README.md)
