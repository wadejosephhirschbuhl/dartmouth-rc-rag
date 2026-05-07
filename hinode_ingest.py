"""
Hinode mod-llm aware site ingester.

Discovery order:
  1. /llms.txt   (Hinode mod-llm) - fastest & cleanest
  2. /sitemap.xml (any Hugo site)
  3. Same-host BFS crawl

Per-page content order:
  1. <page>/index.md (mod-llm) - clean markdown
  2. HTML -> markdown via html2text (fallback)

Reference: https://gethinode.com/tutorials/generating-llm-content/
"""
from __future__ import annotations
import re
import time
import hashlib
from typing import Callable, Dict, List, Optional
from urllib.parse import urljoin, urlparse

import requests
import html2text
from bs4 import BeautifulSoup

USER_AGENT = "dartmouth-rc-rag/1.0 (+https://github.com/)"
TIMEOUT = 20


def _http_get(url: str) -> Optional[requests.Response]:
    try:
        r = requests.get(url, timeout=TIMEOUT, headers={"User-Agent": USER_AGENT})
        if r.status_code == 200:
            return r
    except requests.RequestException:
        return None
    return None


def discover_via_llms_txt(base_url: str) -> Optional[List[Dict]]:
    r = _http_get(f"{base_url.rstrip('/')}/llms.txt")
    if not r or "text/plain" not in r.headers.get("content-type", "").lower():
        return None
    pages, section = [], "General"
    link_re = re.compile(r"^\s*-\s*\[([^\]]+)\]\(([^)]+)\)\s*:?\s*(.*)$")
    for line in r.text.splitlines():
        if line.startswith("## "):
            section = line[3:].strip()
            continue
        m = link_re.match(line)
        if m:
            title, href, desc = m.groups()
            pages.append({
                "title": title.strip(),
                "url": urljoin(base_url + "/", href.strip()),
                "description": desc.strip(),
                "section": section,
            })
    return pages or None


def discover_via_sitemap(base_url: str) -> Optional[List[Dict]]:
    r = _http_get(f"{base_url.rstrip('/')}/sitemap.xml")
    if not r:
        return None
    urls = re.findall(r"<loc>\s*([^<\s]+)\s*</loc>", r.text)
    if not urls:
        return None
    host = urlparse(base_url).netloc
    return [
        {"title": u, "url": u, "description": "", "section": "Sitemap"}
        for u in urls if urlparse(u).netloc == host
    ]


def discover_via_crawl(base_url: str, max_pages: int) -> List[Dict]:
    host = urlparse(base_url).netloc
    seen, queue, found = set(), [base_url], []
    while queue and len(found) < max_pages:
        url = queue.pop(0).split("#")[0]
        if url in seen:
            continue
        seen.add(url)
        r = _http_get(url)
        if not r or "text/html" not in r.headers.get("content-type", "").lower():
            continue
        soup = BeautifulSoup(r.text, "html.parser")
        title = (soup.title.string.strip() if soup.title and soup.title.string else url)
        found.append({"title": title, "url": url, "description": "", "section": "Crawl"})
        for a in soup.find_all("a", href=True):
            link = urljoin(url, a["href"]).split("#")[0]
            if urlparse(link).netloc == host and link not in seen:
                queue.append(link)
        time.sleep(0.15)
    return found


def _try_index_md(page_url: str) -> Optional[str]:
    if page_url.endswith(".md"):
        r = _http_get(page_url)
        return r.text if r else None
    parsed = urlparse(page_url)
    path = parsed.path if parsed.path.endswith("/") else parsed.path + "/"
    md_url = f"{parsed.scheme}://{parsed.netloc}{path}index.md"
    r = _http_get(md_url)
    return r.text if r else None


def _html_to_markdown(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["nav", "footer", "script", "style", "header", "aside", "form"]):
        tag.decompose()
    main = soup.find("main") or soup.find("article") or soup.body or soup
    h = html2text.HTML2Text()
    h.ignore_images = True
    h.body_width = 0
    return h.handle(str(main)).strip()


def fetch_page_markdown(page_url: str) -> Optional[str]:
    md = _try_index_md(page_url)
    if md and md.strip():
        return md
    html_url = page_url[:-len("index.md")] if page_url.endswith("index.md") else page_url
    r = _http_get(html_url)
    if not r:
        return None
    return _html_to_markdown(r.text)


def ingest_hinode_site(
    col,
    site_url: str,
    max_pages: int,
    chunk_size: int,
    overlap: int,
    chunk_text_fn: Callable[[str, int, int], List[str]],
    on_progress: Optional[Callable[[int, int, str], None]] = None,
) -> Dict:
    site_url = site_url.rstrip("/")
    on_progress = on_progress or (lambda d, t, m: None)

    on_progress(0, 0, f"Trying {site_url}/llms.txt ...")
    pages = discover_via_llms_txt(site_url)
    mode = "llms.txt"
    if not pages:
        on_progress(0, 0, f"No llms.txt - trying {site_url}/sitemap.xml ...")
        pages = discover_via_sitemap(site_url)
        mode = "sitemap.xml"
    if not pages:
        on_progress(0, 0, "No sitemap - falling back to crawl.")
        pages = discover_via_crawl(site_url, max_pages)
        mode = "crawl"

    pages = pages[:max_pages]
    total = len(pages)
    on_progress(0, total, f"Discovered {total} pages via {mode}.")

    ids, docs, metas = [], [], []
    fetched, skipped = 0, 0
    host = urlparse(site_url).netloc

    for i, p in enumerate(pages, start=1):
        url = p["url"]
        on_progress(i, total, f"[{i}/{total}] {url}")
        md = fetch_page_markdown(url)
        if not md:
            skipped += 1
            continue
        fetched += 1

        path = urlparse(url).path or "/"
        source_label = f"{host}{path}"
        url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]

        chunks = chunk_text_fn(md, chunk_size, overlap)
        for ci, ch in enumerate(chunks, start=1):
            ids.append(f"site_{url_hash}_c{ci:04d}")
            docs.append(ch)
            metas.append({
                "source": source_label,
                "page": 1,
                "chunk": ci,
                "file_hash": url_hash,
                "url": url,
                "title": p.get("title", ""),
                "section": p.get("section", ""),
            })

        if len(ids) >= 500:
            col.upsert(ids=ids, documents=docs, metadatas=metas)
            ids, docs, metas = [], [], []

    if ids:
        col.upsert(ids=ids, documents=docs, metadatas=metas)

    return {
        "mode": mode,
        "site_url": site_url,
        "discovered": total,
        "fetched": fetched,
        "skipped": skipped,
    }
