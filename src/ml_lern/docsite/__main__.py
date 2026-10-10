"""ml-lern-docs: docs/ の Markdown をブラウザで描画する Web ページとして配信する。

サーバーは「HTML の殻」「生の Markdown」「ライブラリ」だけを返し、
本文の描画（Markdown・数式・図）はブラウザ側で行う。おかげで応答が速い。

使い方::

    uv run ml-lern-docs                 # http://localhost:8000 で配信
    uv run ml-lern-docs --port 8080
    uv run ml-lern-docs --build site/    # 静的ファイルとして書き出す
    uv run ml-lern-docs --reload         # Markdown の編集を即反映（再走査のみ）
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .build import Site, build_site, write_site
from .libs import prepare_libs
from .server import serve


def _default_docs_dir() -> Path:
    cwd_docs = Path.cwd() / "docs"
    if cwd_docs.is_dir():
        return cwd_docs
    repo = Path(__file__).resolve().parents[3]
    return repo / "docs"


def _default_cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME")
    root = Path(base) if base else Path.home() / ".cache"
    return root / "ml-lern" / "docsite"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ml-lern-docs",
        description="docs/ の Markdown をブラウザ描画で Web ページとして配信します。",
    )
    p.add_argument("--docs", type=Path, default=None,
                   help="Markdown を置いたディレクトリ（既定: ./docs）")
    p.add_argument("--host", default="127.0.0.1", help="待ち受けアドレス（既定: 127.0.0.1）")
    p.add_argument("--port", type=int, default=8000, help="待ち受けポート（既定: 8000）")
    p.add_argument("--title", default="ml-lern docs", help="サイドバーに出すサイト名")
    p.add_argument("--build", type=Path, default=None, metavar="DIR",
                   help="配信せず、静的サイトを DIR に書き出す")
    p.add_argument("--reload", action="store_true",
                   help="リクエストごとに再走査する（執筆中のプレビュー向け）")
    p.add_argument("--open", action="store_true", help="起動後にブラウザを開く")
    p.add_argument("--no-download", action="store_true",
                   help="ライブラリを取得・キャッシュせず CDN を直接使う")
    p.add_argument("--assets-dir", type=Path, default=None,
                   help="ライブラリのキャッシュ先（既定: ~/.cache/ml-lern/docsite）")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    docs_dir = (args.docs or _default_docs_dir()).resolve()
    if not docs_dir.is_dir():
        print(f"エラー: docs ディレクトリが見つかりません: {docs_dir}", file=sys.stderr)
        return 2

    cache_dir = args.assets_dir or _default_cache_dir()
    libs, lib_assets = prepare_libs(cache_dir, download=not args.no_download)

    def make_site() -> Site:
        site = build_site(docs_dir, libs, site_title=args.title)
        site.assets.update(lib_assets)
        return site

    try:
        site = make_site()
    except Exception as exc:
        print(f"エラー: サイト生成に失敗しました: {exc}", file=sys.stderr)
        return 1

    print(f"{docs_dir} の {len(site.page_urls)} ページを用意しました")

    if args.build is not None:
        out = args.build.resolve()
        write_site(site, out)
        print(f"静的サイトを書き出しました: {out}")
        print(f"  例) cd {out} && python -m http.server")
        return 0

    if args.open:
        _open_later(args.host, args.port)

    rebuild = make_site if args.reload else None
    serve(site, host=args.host, port=args.port, rebuild=rebuild)
    return 0


def _open_later(host: str, port: int) -> None:
    import threading
    import webbrowser

    display_host = "localhost" if host in ("0.0.0.0", "127.0.0.1") else host
    url = f"http://{display_host}:{port}/"
    threading.Timer(0.5, lambda: webbrowser.open(url)).start()


if __name__ == "__main__":
    raise SystemExit(main())
