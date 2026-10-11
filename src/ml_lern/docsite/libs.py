"""ブラウザ側ライブラリ（markdown-it / KaTeX / Mermaid）の取得とキャッシュ。

初回起動時にダウンロードして ``~/.cache/ml-lern/docsite/lib`` に保存する。
次回以降はローカルから配信するので、オフラインでも動くうえ読み込みも速い。
ダウンロードできなかったものは CDN の URL にフォールバックする。
"""

from __future__ import annotations

import re
import urllib.request
from pathlib import Path

JS = "text/javascript; charset=utf-8"
CSS = "text/css; charset=utf-8"
FONT = "font/woff2"

# 各ライブラリ: 表示に必要な最小サイズも持たせて、壊れたキャッシュを弾く
_LIBS = [
    {
        "name": "markdownit",
        "version": "14.1.0",
        "remote": "https://cdn.jsdelivr.net/npm/markdown-it@14.1.0/dist/markdown-it.min.js",
        "ctype": JS,
        "min_size": 50_000,
    },
    {
        "name": "texmath",
        "version": "1.0.0",
        "remote": "https://cdn.jsdelivr.net/npm/markdown-it-texmath@1.0.0/texmath.js",
        "ctype": JS,
        "min_size": 5_000,
    },
    {
        "name": "katex",
        "version": "0.16.11",
        "remote": "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js",
        "ctype": JS,
        "min_size": 100_000,
    },
    {
        "name": "katexCss",
        "version": "0.16.11",
        "remote": "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css",
        "ctype": CSS,
        "min_size": 5_000,
        "katex_fonts": True,
    },
    {
        "name": "mermaid",
        "version": "11",
        "remote": "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js",
        "ctype": JS,
        "min_size": 500_000,
    },
]

_KATEX_FONT_BASE = "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/fonts/"
_FONT_URL_RE = re.compile(r"url\((fonts/[A-Za-z0-9_.\-]+\.woff2)\)")


def _download(url: str, timeout: float = 60.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "ml-lern-docsite/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (既知の URL)
        return resp.read()


def _cached(cache_file: Path, min_size: int) -> bytes | None:
    try:
        data = cache_file.read_bytes()
    except OSError:
        return None
    return data if len(data) >= min_size else None


def _fetch(cache_file: Path, remote: str, min_size: int, download: bool) -> bytes | None:
    data = _cached(cache_file, min_size)
    if data is not None:
        return data
    if not download:
        return None
    try:
        data = _download(remote)
    except Exception:  # ネットワーク不通など → CDN フォールバック
        return None
    if len(data) < min_size:
        return None
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_bytes(data)
    return data


def _prepare_katex_fonts(
    cache_dir: Path, css_file: Path, download: bool
) -> tuple[dict[str, tuple[str, bytes]], dict[str, str]]:
    """KaTeX のフォントを取得し、URL の書き換え表を返す。"""
    fonts_dir = cache_dir / "katex-fonts"
    assets: dict[str, tuple[str, bytes]] = {}
    url_map: dict[str, str] = {}
    if not css_file.exists():
        return assets, url_map
    for name in sorted(set(_FONT_URL_RE.findall(css_file.read_text(encoding="utf-8")))):
        font_file = fonts_dir / Path(name).name
        data = _fetch(font_file, _KATEX_FONT_BASE + Path(name).name, 1_000, download)
        if data is None:
            continue
        url = f"/lib/katex/fonts/{Path(name).name}"
        assets[url] = (FONT, data)
        # CSS からの相対（CSS は lib/ 直下にあるので katex/fonts/...）
        url_map[name] = f"katex/fonts/{Path(name).name}"
    return assets, url_map


def prepare_libs(
    cache_dir: Path, *, download: bool = True
) -> tuple[dict[str, str], dict[str, tuple[str, bytes]]]:
    """ライブラリを用意し、(名前→URL の対応, 配信すべきアセット) を返す。"""
    cache_dir = Path(cache_dir)
    lib_dir = cache_dir / "lib"
    urls: dict[str, str] = {}
    assets: dict[str, tuple[str, bytes]] = {}

    for spec in _LIBS:
        filename = f"{spec['name']}-{spec['version']}"
        ext = ".css" if spec["ctype"] is CSS else ".js"
        cache_file = lib_dir / (filename + ext)
        data = _fetch(cache_file, spec["remote"], spec["min_size"], download)

        if data is None:
            urls[spec["name"]] = spec["remote"]  # CDN 直参照
            continue

        if spec.get("katex_fonts"):
            font_assets, font_map = _prepare_katex_fonts(cache_dir, cache_file, download)
            assets.update(font_assets)
            text = data.decode("utf-8")
            for original, local in font_map.items():
                text = text.replace(f"url({original})", f"url({local})")
            data = text.encode("utf-8")

        url = f"/lib/{filename}{ext}"
        assets[url] = (spec["ctype"], data)
        urls[spec["name"]] = f"lib/{filename}{ext}"  # ブラウザから見た相対 URL

    return urls, assets
