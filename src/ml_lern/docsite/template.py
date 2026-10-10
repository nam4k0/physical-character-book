"""HTML の「殻」（アプリシェル）と CSS/JS の場所を決めるところ。

方針: 本文（Markdown → HTML・数式・図）の描画はブラウザ側（static/app.js）に任せる。
サーバーはページごとに同じ殻を返し、生の Markdown とライブラリだけを配る。
"""

from __future__ import annotations

import json
from pathlib import Path

STATIC_DIR = Path(__file__).parent / "static"

SHELL = """<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{site_title}</title>
<link rel="stylesheet" href="{app_css}">
<link rel="preconnect" href="https://cdn.jsdelivr.net" crossorigin>
<link rel="dns-prefetch" href="https://cdn.jsdelivr.net">
<script>window.__DOCS__ = {docs_json};</script>
</head>
<body>
<div class="layout">
  <aside class="sidebar">
    <h1>{site_title}</h1>
    <p class="sub">docs/ の Markdown をブラウザで描画</p>
    <nav id="nav"></nav>
  </aside>
  <div class="content">
    <main id="content"><p class="loading">読み込み中…</p></main>
  </div>
</div>
<noscript>
  <div style="max-width:40rem;margin:2rem auto;padding:0 1rem;font-family:sans-serif">
    <h1>JavaScript が必要です</h1>
    <p>このページはブラウザ側で Markdown を描画します。JavaScript を有効にするか、
    リポジトリ内の <code>docs/</code> の Markdown を直接お読みください。</p>
  </div>
</noscript>
<script src="{app_js}"></script>
</body>
</html>
"""


def render_shell(docs: dict, *, app_js: str = "/app.js", app_css: str = "/app.css") -> str:
    """docs（ページ一覧・サイト名・ライブラリ URL）を埋め込んだ殻を作る。"""
    payload = json.dumps(docs, ensure_ascii=False).replace("</", "<\\/")
    return SHELL.format(
        site_title=docs.get("siteTitle", "docs"),
        docs_json=payload,
        app_js=app_js,
        app_css=app_css,
    )


def read_static(name: str) -> bytes:
    """static/ のファイルを読む（app.js / app.css を配信するため）。"""
    return (STATIC_DIR / name).read_bytes()
