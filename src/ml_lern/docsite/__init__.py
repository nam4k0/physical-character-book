"""docs/ の Markdown をブラウザ描画で配信するツール。

サーバーは HTML の殻・生の Markdown・ライブラリだけを返し、
本文の描画はブラウザ側（static/app.js）で行う。応答が速く、再読み込みも軽い。

公開 API:
    build_site(docs_dir, libs)  ... Markdown を走査して Site を作る
    write_site(site, out_dir)    ... Site を静的ファイルとして書き出す
    prepare_libs(cache_dir)      ... ブラウザ側ライブラリを用意する
    serve(site, host, port)      ... Site を HTTP で配信する
"""

from __future__ import annotations

from .build import Site, build_site, write_site
from .libs import prepare_libs
from .server import serve

__all__ = ["Site", "build_site", "write_site", "prepare_libs", "serve"]
