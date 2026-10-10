"""生成済みサイトをメモリから配信する簡易 HTTP サーバー。

返すものは 3 種類だけ:

- HTML の殻（全ページ共通）
- 生の Markdown（``/x.md``）
- 静的アセット（``/app.js`` ``/app.css`` と ``/lib/*``）

重いライブラリには長めの Cache-Control を付け、2 ページ目以降を速くする。
"""

from __future__ import annotations

import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable

from .build import Site

NOT_FOUND = """<!DOCTYPE html>
<html lang="ja"><head><meta charset="utf-8"><title>404 Not Found</title></head>
<body style="font-family:sans-serif;max-width:40rem;margin:4rem auto">
<h1>404 Not Found</h1>
<p>見つかりません: <code>{path}</code></p>
<p><a href="/">トップへ戻る</a></p>
</body></html>
"""


class _Handler(BaseHTTPRequestHandler):
    server_version = "ml-lern-docsite/0.2"

    def _resolve(self) -> str:
        path = urllib.parse.urlsplit(self.path).path
        return urllib.parse.unquote(path) or "/"

    def _cache_control(self, path: str) -> str:
        if path.startswith("/lib/"):
            return "public, max-age=86400, immutable"
        if path in ("/app.js", "/app.css"):
            return "public, max-age=600"
        return "no-cache"

    def _send(
        self, status: int, body: bytes, content_type: str, *, head_only: bool, path: str = ""
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if path:
            self.send_header("Cache-Control", self._cache_control(path))
        self.end_headers()
        if not head_only:
            self.wfile.write(body)

    # --- ルーティング -----------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802
        self._serve(head_only=False)

    def do_HEAD(self) -> None:  # noqa: N802
        self._serve(head_only=True)

    def _serve(self, *, head_only: bool) -> None:
        site: Site = self.server.get_site()  # type: ignore[attr-defined]
        path = self._resolve()

        if path == "/favicon.ico":
            self._send(204, b"", "image/x-icon", head_only=True, path=path)
            return

        if path in site.assets:
            ctype, data = site.assets[path]
            self._send(200, data, ctype, head_only=head_only, path=path)
            return

        if path.endswith(".md"):
            text = site.markdown.get(path)
            if text is not None:
                self._send(200, text.encode("utf-8"), "text/markdown; charset=utf-8",
                           head_only=head_only, path=path)
                return

        # ページ本体（HTML）はすべて共通の殻を返す
        if path == "/" or path.endswith(".html"):
            self._send(200, site.shell.encode("utf-8"), "text/html; charset=utf-8",
                       head_only=head_only, path=path)
            return

        body = NOT_FOUND.format(path=path).encode("utf-8")
        self._send(404, body, "text/html; charset=utf-8", head_only=head_only, path=path)

    def log_message(self, fmt: str, *args) -> None:  # noqa: A003
        print(f"  {self.address_string()} - {fmt % args}", flush=True)


class DocsHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        addr: tuple[str, int],
        site: Site,
        rebuild: Callable[[], Site] | None = None,
    ) -> None:
        super().__init__(addr, _Handler)
        self._site = site
        self._rebuild = rebuild
        self._lock = threading.Lock()

    def get_site(self) -> Site:
        if self._rebuild is None:
            return self._site
        with self._lock:
            self._site = self._rebuild()
        return self._site


def serve(
    site: Site,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    rebuild: Callable[[], Site] | None = None,
) -> None:
    httpd = DocsHTTPServer((host, port), site, rebuild=rebuild)
    display_host = "localhost" if host in ("0.0.0.0", "127.0.0.1") else host
    port_actual = httpd.server_address[1]
    print(f"docs を {port_actual} 番ポートで配信中")
    print(f"  → http://{display_host}:{port_actual}/")
    print("  本文はブラウザ側で描画します（サーバーは Markdown を返すだけ）")
    print("  終了するには Ctrl-C")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n停止します")
    finally:
        httpd.server_close()
