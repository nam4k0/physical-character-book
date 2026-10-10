# デバイス利用ガイド（AI向け）

このファイルは、AIエージェントがこのデバイスで作業するときの前提と手順をまとめたものです。作業前に必ず読んでください。

## 1. デバイス概要（2026-10-09 時点で確認）

| 項目 | 内容 |
| --- | --- |
| ホスト名 | `debian` |
| OS | Debian GNU/Linux 13 (trixie) 13.7 |
| カーネル | `6.12.93-android16-...`（Android 16 系カーネル） |
| アーキテクチャ | aarch64（ARM64） |
| CPU | 8 コア |
| メモリ | 約 5 GiB（スワップ 1.2 GiB） |
| ディスク | ルート `/` 約 103 GB（空き約 71 GB） |
| ユーザー | `droid`（`sudo` 利用可） |
| ログインシェル | `/bin/bash` |
| ロケール | `LANG=C.UTF-8` |
| ネットワーク | `enp0s13` に IPv4 割り当て済み、外部 HTTPS 接続可 |
| systemd | 稼働中（`running`） |

## 2. 基本方針

- **パッケージ管理・開発環境は Nix Flakes を優先する。**
  - ツールを試す場合は `nix run` / `nix shell` / `nix develop` を使い、システム全体への追加は避ける。
  - プロジェクトには `flake.nix` を用意し、依存関係を再現可能にする。
  - `git` はインストール済み（`jj` の Git バックエンドやリモート操作用）。ただし通常の VCS 操作は `jj` を使う。
  - **apt でのインストールは必ず事前にユーザーの許可を得ること。** 許可なく `apt install` を実行しない。
- **バージョン管理は Jujutsu（`jj`）を使う。** Git コマンドを直接使う前に、`jj` で代替できないかを検討する。
- **このデバイスの想定用途は「ちょっとした開発」。** 大規模なビルドや重い処理はリソース（メモリ約 5 GiB）に注意する。

## 3. インストール済みツール

| ツール | 場所 / バージョン | 備考 |
| --- | --- | --- |
| Nix | `/nix/var/nix/profiles/default/bin/nix`（2.35.2） | マルチユーザー（daemon）構成。flakes 有効 |
| jj | `~/.cargo/bin/jj`（0.46.0） | 主要 VCS |
| git | `/usr/bin/git`（2.47.3） | jj の Git バックエンド・リモート用。VCS 操作は jj 優先 |
| cargo / rustc | `~/.cargo/bin` | Rust 開発用 |
| gcc / make | `/usr/bin` | C/C++ ビルド用 |
| python3 | `/usr/bin/python3` | |
| curl / wget / ssh | `/usr/bin` | |
| opencode | `~/.opencode/bin` | |


## 4. Nix の使い方

### 4.1 flakes について

`/etc/nix/nix.conf` に以下を設定済みで、`nix-command` と `flakes` は有効です。フラグを付ける必要はありません。

```
experimental-features = nix-command flakes
```

（変更前の設定は `/etc/nix/nix.conf.bak` にバックアップ済み。設定を変更した場合は `sudo systemctl restart nix-daemon` で反映する。）


### 4.2 よく使うコマンド

```bash
# 一時的にツールを使う
nix run nixpkgs#<pkg> -- <args>
nix shell nixpkgs#<pkg1> nixpkgs#<pkg2>

# プロジェクトの開発環境に入る（flake.nix の devShell）
nix develop
nix develop -c <command>

# flake の検査・ビルド
nix flake check
nix build
```

### 4.3 flake を新規作成するとき

- プロジェクト直下に `flake.nix` を作成し、`devShells.default` を定義する。
- `flake.lock` はコミットして再現性を保つ。
- 対象は `aarch64-linux` であることを前提にする（`x86_64-linux` 向けの記述は不要）。

## 5. バージョン管理（Jujutsu）

- 新規リポジトリ: `jj git init`（または既存 Git リポジトリがあれば `jj git init --colocate`）。
- 変更の確認: `jj st`, `jj diff`, `jj log`
- 変更の記録: `jj describe -m "メッセージ"` で説明を付け、`jj new` で次の変更を開始する。
- ブックマーク（ブランチ相当）: `jj bookmark create|set|list`
- リモート操作: `jj git fetch` / `jj git push`
- **Git の `add` / `commit` / `checkout` などは使わない。** 作業コピーは jj が自動で追跡する。
- コミットは意味のある単位で分ける。作業の途中でも `jj describe` で内容を明示する。
- **コミットメッセージ・説明文は日本語で書く。**

## 6. 作業ディレクトリ

- ホームディレクトリ `/home/droid` 配下に作業ファイルを散らかさない。プロジェクトごとにディレクトリを分ける。
- 一時ファイルは `/tmp/opencode` を使う（事前に作成済みで、外部アクセスも承認されている）。

## 7. 安全上の注意

- 不明なファイルや既存の設定は、削除・上書きする前に内容を確認する。
- ネットワーク経由で取得したスクリプトを、内容を確認せずに実行しない。

## 8. 決定事項と残りの確認事項

### 決定済み
- Nix の `experimental-features`（`nix-command flakes`）は有効化済み。
- `git` はインストール済み。
- パッケージ導入は Nix 優先。**apt はユーザーの許可が必要。**
- 用途は「ちょっとした開発」。
- コミットメッセージ・説明文は**日本語**。

