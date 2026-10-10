# ml-lern — 物理キャラクタのシーン相互作用を学ぶ教科書

このリポジトリは、ゲームエンジニア向けに次の論文を解説・実装する教科書です。

> Mohamed Hassan, Yunrong Guo, Tingwu Wang, Michael Black, Sanja Fidler, Xue Bin Peng.
> **Synthesizing Physical Character-Scene Interactions.**
> arXiv:2302.00883 (2023). <https://arxiv.org/abs/2302.00883>

## どこから読むか

- **教科書本体**: [`docs/README.md`](docs/README.md)（目次・読み方・全体像）
- 第I部: 機械学習の基礎（RL・PPO・模倣学習・敵対学習・AMP・PD制御）
- 第II部: 論文の解説（数式・実験・比較）
- 第III部: 実装（人型本線＋寄り道、MuJoCo + PyTorch）
- 付録: 用語集・拡散モデル・演習・参考リソース

## 実装コード

```
src/ml_lern/mini_amp/   AMP による物理キャラ制御のミニ実装
src/ml_lern/diffusion/  拡散モデル（付録B）
scripts/                学習・生成のエントリポイント
```

第III部の依存を導入する場合:

```bash
uv sync --extra train --extra viz
```

> **注意**: 本リポジトリの実行環境（CPU・メモリ約5GiB・CUDAなし）では、論文本来の IsaacGym は動きません。第III部のコードは MuJoCo + PyTorch（CPU）で動く小規模版で、本リポジトリでは実行・検証していません（読者の環境で動かす前提）。

## 元資料

- `article.md` / `article.html` … arXiv の HTML 版を変換したもの（本教科書の元ネタ）
