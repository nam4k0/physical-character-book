/* ml-lern docs: ブラウザ側で Markdown を描画する小さなアプリ。
 *
 * サーバーは「HTML の殻」と「生の .md」と「ライブラリ」だけを返す。
 * 本文の変換（Markdown → HTML / 数式 / 図）はすべてこのスクリプトが行う。
 * Node からも renderMarkdown を呼べるように UMD 風のラッパーにしている。
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) {
    module.exports = factory();
  } else {
    root.DocsApp = factory();
  }
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  // --- 純粋な文字列処理（Node でもテスト可能） --------------------------

  function slugify(text) {
    var s = (text || "")
      .normalize("NFKC")
      .trim()
      .toLowerCase()
      .replace(/[^\p{L}\p{N}\s\-_/]+/gu, "")
      .replace(/[\s\-_/]+/g, "-")
      .replace(/^-+|-+$/g, "");
    return s || "section";
  }

  function makeRenderer(libs, useMath) {
    var md = libs.markdownit({
      html: true,
      linkify: false,
      breaks: false,
      typographer: false,
    });
    if (useMath) {
      md.use(libs.texmath, {
        engine: libs.katex,
        delimiters: "dollars",
        katexOptions: { throwOnError: false, strict: false },
      });
    }
    return md;
  }

  /** Markdown 本文を HTML 文字列にする。 */
  function renderMarkdown(text, libs) {
    var useMath = text.indexOf("$") !== -1;
    var md = makeRenderer(libs, useMath);
    var html = md.render(text);
    return {
      html: html,
      hasMath: useMath,
      hasMermaid: /(^|\n)\s*```mermaid\b/.test(text),
    };
  }

  /** `....md` へのリンクを `....html` に張り替える（README.md → index.html）。 */
  function rewriteHref(href) {
    var m = href.match(/^(.*?)\.md(#.*)?$/i);
    if (!m) return href;
    var basePath = m[1];
    var frag = m[2] || "";
    if (/(^|\/)README$/i.test(basePath)) {
      basePath = basePath.replace(/README$/i, "index");
    }
    return basePath + ".html" + frag;
  }

  // --- ブラウザ側の処理 -------------------------------------------------

  function currentUrl() {
    var path = decodeURIComponent(location.pathname);
    if (path === "" || path === "/") return "/index.html";
    return path;
  }

  function mdUrlFor(pageUrl, pages) {
    var page = null;
    for (var i = 0; i < pages.length; i++) {
      if (pages[i].url === pageUrl) { page = pages[i]; break; }
    }
    if (!page && pageUrl.endsWith(".html")) {
      return pageUrl.replace(/\.html$/, ".md");
    }
    if (!page) return "/README.md";
    return page.url === "/index.html" ? "/README.md" : page.url.replace(/\.html$/, ".md");
  }

  var _scripts = {};
  function loadScript(url) {
    if (_scripts[url]) return _scripts[url];
    _scripts[url] = new Promise(function (resolve, reject) {
      var s = document.createElement("script");
      s.src = url;
      s.async = true;
      s.onload = function () { resolve(url); };
      s.onerror = function () { reject(new Error("スクリプトを読み込めません: " + url)); };
      document.head.appendChild(s);
    });
    return _scripts[url];
  }

  var _styles = {};
  function loadStyle(url) {
    if (_styles[url]) return _styles[url];
    _styles[url] = new Promise(function (resolve, reject) {
      var l = document.createElement("link");
      l.rel = "stylesheet";
      l.href = url;
      l.onload = function () { resolve(url); };
      l.onerror = function () { reject(new Error("スタイルを読み込めません: " + url)); };
      document.head.appendChild(l);
    });
    return _styles[url];
  }

  function renderNav(data) {
    var nav = document.getElementById("nav");
    if (!nav) return;
    var html = "<ul>";
    data.pages.forEach(function (p) {
      html += '<li><a href="' + p.url + '" title="' + escapeAttr(p.title) + '">' +
        escapeHtml(p.label) + "</a></li>";
    });
    html += "</ul>";
    nav.innerHTML = html;
  }

  function markActive(pageUrl) {
    var links = document.querySelectorAll("#nav a");
    for (var i = 0; i < links.length; i++) {
      var href = links[i].getAttribute("href");
      if (href === pageUrl) links[i].classList.add("current");
    }
  }

  function headingIds(root) {
    var used = {};
    var heads = root.querySelectorAll("h1, h2, h3, h4");
    heads.forEach(function (h) {
      var id = slugify(h.textContent);
      if (used[id]) { used[id] += 1; id = id + "-" + used[id]; }
      else { used[id] = 1; }
      h.id = id;
    });
  }

  function buildToc(root) {
    var heads = root.querySelectorAll("h2, h3");
    if (heads.length === 0) return "";
    var items = [];
    heads.forEach(function (h) {
      var depth = h.tagName === "H3" ? "toc3" : "toc2";
      items.push('<li class="' + depth + '"><a href="#' + h.id + '">' +
        escapeHtml(h.textContent) + "</a></li>");
    });
    return "<ul>" + items.join("") + "</ul>";
  }

  function insertToc(tocHtml) {
    if (!tocHtml) return;
    var box = document.createElement("div");
    box.className = "page-toc";
    box.innerHTML = "<strong>目次</strong>" + tocHtml;
    var main = document.getElementById("content");
    main.insertBefore(box, main.firstChild);
  }

  function rewriteLinks(root) {
    root.querySelectorAll("a[href]").forEach(function (a) {
      var href = a.getAttribute("href");
      if (!href) return;
      var out = rewriteHref(href);
      if (out !== href) a.setAttribute("href", out);
    });
  }

  function setTitle(root, data) {
    var h1 = root.querySelector("h1");
    document.title = (h1 ? h1.textContent + " — " : "") + (data.siteTitle || "docs");
  }

  function renderMermaid(root, url) {
    var blocks = root.querySelectorAll("pre > code.language-mermaid, code.language-mermaid");
    if (blocks.length === 0) return Promise.resolve();
    blocks.forEach(function (code) {
      var div = document.createElement("div");
      div.className = "mermaid";
      div.textContent = code.textContent;
      (code.closest("pre") || code).replaceWith(div);
    });
    return loadScript(url).then(function () {
      var dark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
      window.mermaid.initialize({ startOnLoad: false, theme: dark ? "dark" : "neutral" });
      return window.mermaid.run({ querySelector: ".mermaid" });
    });
  }

  function showError(mdUrl, err) {
    var main = document.getElementById("content");
    main.innerHTML =
      '<div class="docs-error"><h1>読み込みに失敗しました</h1>' +
      "<p>" + escapeHtml(String(err && err.message ? err.message : err)) + "</p>" +
      '<p>元のファイル: <a href="' + escapeAttr(mdUrl) + '">' + escapeHtml(mdUrl) + "</a></p></div>";
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }
  function escapeAttr(s) { return escapeHtml(s); }

  async function boot() {
    var data = window.__DOCS__;
    if (!data) return;
    renderNav(data);
    var pageUrl = currentUrl();
    markActive(pageUrl === "/" ? "/index.html" : pageUrl);
    var mdUrl = mdUrlFor(pageUrl, data.pages);

    var text;
    try {
      var res = await fetch(mdUrl, { cache: "no-cache" });
      if (!res.ok) throw new Error("HTTP " + res.status);
      text = await res.text();
    } catch (e) {
      showError(mdUrl, e);
      return;
    }

    try {
      await loadScript(data.libs.markdownit);
      var libs = { markdownit: window.markdownit };
      if (text.indexOf("$") !== -1) {
        loadStyle(data.libs.katexCss).catch(function () { /* CSS 失敗は致命的ではない */ });
        await loadScript(data.libs.katex);
        await loadScript(data.libs.texmath);
        libs.katex = window.katex;
        libs.texmath = window.texmath;
      }
      var out = renderMarkdown(text, libs);
      var main = document.getElementById("content");
      main.innerHTML = out.html;
      headingIds(main);
      insertToc(buildToc(main));
      rewriteLinks(main);
      setTitle(main, data);
      if (out.hasMermaid) await renderMermaid(main, data.libs.mermaid);
      if (location.hash) {
        var target = document.getElementById(decodeURIComponent(location.hash.slice(1)));
        if (target) target.scrollIntoView();
      }
    } catch (e) {
      showError(mdUrl, e);
    }
  }

  if (typeof document !== "undefined") {
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", boot);
    } else {
      boot();
    }
  }

  return {
    renderMarkdown: renderMarkdown,
    rewriteHref: rewriteHref,
    slugify: slugify,
    boot: boot,
  };
});
