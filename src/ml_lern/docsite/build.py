"""docs/ 配下の Markdown を「ブラウザで描画する Web サイト」にまとめる。

サーバー側では Markdown を HTML に変換しない（＝速い）。
やることは次の 3 つだけ:

- docs/ を走査してページ一覧（サイドバー用）を作る
- 生の Markdown をそのまま配れる形で持つ
- 全ページ共通の HTML の殻（アプリシェル）を 1 枚作る

本文の描画はブラウザ側の static/app.js が行う。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .template import read_static, render_shell


@dataclass
class Site:
    """生成済みサイト。"""

    shell: str = ""
    markdown: dict[str, str] = field(default_factory=dict)   # "/x.md" → 本文
    assets: dict[str, tuple[str, bytes]] = field(default_factory=dict)
    nav: list[dict] = field(default_factory=list)
    page_urls: list[str] = field(default_factory=list)


def _label_for(path: Path) -> str:
    if path.stem == "README":
        return "トップ（はじめに）"
    return path.stem.replace("_", " ")


def _first_heading(text: str, fallback: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


def _page_url(rel: Path) -> str:
    if rel.name == "README.md":
        return "/index.html"
    return "/" + rel.with_suffix(".html").as_posix()


def _md_url(rel: Path) -> str:
    return "/" + rel.as_posix()


def _sorted_sources(docs_dir: Path) -> list[Path]:
    files = [p for p in docs_dir.rglob("*.md")]
    # README を先頭、残りはファイル名順（ゼロ埋め番号がそのまま並ぶ）
    files.sort(key=lambda p: (p.name != "README.md", p.relative_to(docs_dir).as_posix()))
    return files


def build_site(
    docs_dir: Path,
    libs: dict[str, str],
    *,
    site_title: str = "ml-lern docs",
) -> Site:
    """docs_dir 以下の Markdown から Site を組み立てる（軽量）。"""
    docs_dir = Path(docs_dir)
    if not docs_dir.is_dir():
        raise FileNotFoundError(f"docs ディレクトリが見つかりません: {docs_dir}")

    site = Site()
    for src in _sorted_sources(docs_dir):
        rel = src.relative_to(docs_dir)
        text = src.read_text(encoding="utf-8")
        page_url = _page_url(rel)
        md_url = _md_url(rel)
        title = _first_heading(text, rel.stem)
        site.markdown[md_url] = text
        site.page_urls.append(page_url)
        site.nav.append(
            {
                "url": page_url,
                "label": _label_for(rel),
                "title": title,
            }
        )

    docs = {
        "siteTitle": site_title,
        "pages": site.nav,
        "libs": libs,
    }
    site.shell = render_shell(docs)

    site.assets["/app.js"] = ("text/javascript; charset=utf-8", read_static("app.js"))
    site.assets["/app.css"] = ("text/css; charset=utf-8", read_static("app.css"))
    return site


def write_site(site: Site, out_dir: Path) -> None:
    """Site を静的なファイル群として書き出す。"""
    import urllib.parse

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for page_url in site.page_urls:
        rel = urllib.parse.unquote(page_url.lstrip("/"))
        dest = out_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(site.shell, encoding="utf-8")

    for md_url, text in site.markdown.items():
        dest = out_dir / urllib.parse.unquote(md_url.lstrip("/"))
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text, encoding="utf-8")

    for url, (_ctype, data) in site.assets.items():
        dest = out_dir / urllib.parse.unquote(url.lstrip("/"))
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
