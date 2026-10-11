# ml-lern — アニメーションAI学習パス

このリポジトリは、ゲームエンジニア向けに **アニメーションAIを PyTorch から物理キャラクタ制御まで**学ぶ教科書です。最終的な到達点は次の論文です。

> Mohamed Hassan, Yunrong Guo, Tingwu Wang, Michael Black, Sanja Fidler, Xue Bin Peng.
> **Synthesizing Physical Character-Scene Interactions.**
> arXiv:2302.00883 (2023). <https://arxiv.org/abs/2302.00883>

## どこから読むか

- **教科書本体**: [`docs/README.md`](docs/README.md)（目次・学習パス・全体像）
- 第0部: はじめに
- 第1部: Python・PyTorch・ニューラルネット・時系列データ
- 第2部: 生成モデル・GAN・拡散モデル（モーション生成）
- 第3部: 物理シミュレーション（MuJoCo）
- 第4部: 強化学習（PPO）
- 第5部: 模倣学習・AMP
- 第6部: 論文を読む
- 第7部: 実装
- 付録: 用語集・数学の補足・演習・参考リソース

## 実装コード

```
src/ml_lern/
  foundations/   第1部: テンソル・NN・時系列・系列モデル
  generative/    第2部: AE/VAE・GAN・拡散モデル
  core/          第3〜5部: MuJoCo・RL・PPO・AMP・報酬・模倣
  charcontrol/   第6〜7部: 環境・学習
  docsite/       docs/ をブラウザ表示するツール
scripts/         各段階のエントリポイント
recipes/         再現レシピ（付録E・F）の実行コード
```

学習用の依存を導入する場合:

```bash
uv sync --extra train --extra viz
```

## ドキュメントをブラウザで見る

```bash
uv run ml-lern-docs            # http://localhost:8000
uv run ml-lern-docs --build site/
```

本文の描画はブラウザ側で行います（サーバーは Markdown・HTML の殻・ライブラリを返すだけ）。
ページ間リンクや数式（KaTeX）・図（Mermaid）もブラウザ側で処理し、必要なページだけ
ライブラリを読み込みます。

### GitHub Pages で公開

`main` への push を契機に GitHub Actions が `docs/` をビルドして公開します。

- 公開先: <https://nam4k0.github.io/physical-character-book/>
- ワークフロー: [`.github/workflows/pages.yml`](.github/workflows/pages.yml)
- 成果物は `_site/` に生成（`.nojekyll` を含む）

初回のみ、リポジトリの Settings → Pages で Source を **GitHub Actions** に設定します
（`gh api -X POST repos/nam4k0/physical-character-book/pages -f build_type=workflow` でも可）。
サイト内のリンクは相対 URL なので、`/<repo>/` のようなサブパス配信でもそのまま動作します。

## 注意

- 本リポジトリの実行環境（CPU・メモリ約5GiB・CUDAなし）では、論文本来の IsaacGym は動きません。物理・RL のコードは MuJoCo + PyTorch（CPU）の小規模版です。
- 本文のコードは読者が自分の環境で動かす前提で、本リポジトリでは実行・検証していません。

## 元資料

- `article.md` / `article.html` … arXiv の HTML 版を変換したもの（本教科書の元ネタ）
