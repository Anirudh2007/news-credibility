"""Read an article's headline, summary and publish date from its URL (2026-11-05).

Before this, a URL was only used to rate the outlet: a pasted article link with no text gave
"Unverified: only a source was given", which is not what a user expects. Now, when the user gives a
link and no text, the API fetches the page and checks the headline (+ the summary the page publishes
for link previews) like pasted text; the page's publish date feeds the temporal check.

What is read: the page's own metadata (og:title / <title>, og:description / description,
article:published_time / JSON-LD datePublished), not the body text. Metadata is what the site chose
as the item's claim, is short enough for claim extraction, and is present on almost every news site.

Safety (the server fetches a URL the user typed): http(s) only; every hop's host must resolve to a
public address (no localhost, private, link-local or reserved ranges: blocks SSRF into the server's
network), at most 3 redirects, each re-checked; HTML only; at most 2 MB read; 6 s timeout.
Social-media posts and link shorteners are not fetched (login walls; the user should paste the text).
Off with NEWSCRED_FETCH_URL=0.
"""

from __future__ import annotations

import ipaddress
import json
import os
import re
import socket
from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse

TIMEOUT_S = 6.0
MAX_BYTES = 2_000_000
MAX_REDIRECTS = 3
USER_AGENT = "Mozilla/5.0 (compatible; NewsCredibilityChecker/0.7; research prototype)"


def enabled() -> bool:
    return os.environ.get("NEWSCRED_FETCH_URL", "1") != "0"


@dataclass
class Article:
    url: str
    title: str | None = None
    description: str | None = None
    published: str | None = None  # ISO date (YYYY-MM-DD)
    site_name: str | None = None
    error: str | None = None

    def text(self) -> str | None:
        """Headline, plus the summary when it adds something (not a repeat of the headline)."""
        if not self.title:
            return None
        t = self.title.strip()
        d = (self.description or "").strip()
        if d and d.lower()[:60] not in t.lower() and t.lower()[:60] not in d.lower():
            return f"{t}. {d}" if not t.endswith((".", "!", "?")) else f"{t} {d}"
        return t


def is_article_url(url: str | None) -> bool:
    """A full http(s) URL with a path (a bare domain or a homepage is an outlet, not an item)."""
    if not url:
        return False
    u = urlparse(url.strip())
    return u.scheme in ("http", "https") and bool(u.netloc) and len(u.path.strip("/")) > 0


def _public_host(host: str) -> bool:
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return False
    for info in infos:
        ip = ipaddress.ip_address(info[4][0].split("%")[0])
        if (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast
                or ip.is_unspecified):
            return False
    return True


class _MetaParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.meta: dict[str, str] = {}
        self.title_parts: list[str] = []
        self.ld_json: list[str] = []
        self._in_title = self._in_ld = False

    def handle_starttag(self, tag, attrs):
        a = {k.lower(): (v or "") for k, v in attrs}
        if tag == "meta":
            key = (a.get("property") or a.get("name") or a.get("itemprop") or "").lower()
            if key and a.get("content") and key not in self.meta:
                self.meta[key] = a["content"]
        elif tag == "title":
            self._in_title = True
        elif tag == "script" and a.get("type", "").lower() == "application/ld+json":
            self._in_ld = True
            self.ld_json.append("")

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
        elif tag == "script":
            self._in_ld = False

    def handle_data(self, data):
        if self._in_title:
            self.title_parts.append(data)
        elif self._in_ld:
            self.ld_json[-1] += data


_DATE_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
_SITE_SUFFIX_RE = re.compile(r"\s+[|\-–—:]\s+[^|\-–—:]{2,40}$")


def _iso_date(value: str | None) -> str | None:
    m = _DATE_RE.search(value or "")
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else None


def _ld_date(blocks: list[str]) -> str | None:
    for b in blocks:
        try:
            data = json.loads(b)
        except ValueError:
            m = re.search(r'"datePublished"\s*:\s*"([^"]+)"', b)
            if m:
                return _iso_date(m.group(1))
            continue
        stack = [data]
        while stack:
            x = stack.pop()
            if isinstance(x, dict):
                if isinstance(x.get("datePublished"), str):
                    return _iso_date(x["datePublished"])
                stack.extend(x.values())
            elif isinstance(x, list):
                stack.extend(x)
    return None


def _clean_title(title: str | None, site: str | None) -> str | None:
    if not title:
        return None
    t = re.sub(r"\s+", " ", unescape(title)).strip()
    # "Headline - The Hindu" / "Headline | NDTV": drop a trailing site name (only a short last segment)
    m = _SITE_SUFFIX_RE.search(t)
    if m and len(t) - len(m.group(0)) >= 25:
        t = t[: m.start()].strip()
    if site and t.lower().endswith(site.lower()):
        t = t[: -len(site)].rstrip(" |-–—:")
    return t or None


def parse_html(html: str, url: str) -> Article:
    p = _MetaParser()
    try:
        p.feed(html)
    except Exception:  # malformed HTML: keep what was parsed
        pass
    m = p.meta
    site = m.get("og:site_name")
    title = m.get("og:title") or m.get("twitter:title") or "".join(p.title_parts)
    desc = m.get("og:description") or m.get("description") or m.get("twitter:description")
    published = (_iso_date(m.get("article:published_time")) or _iso_date(m.get("og:published_time"))
                 or _iso_date(m.get("datepublished")) or _ld_date(p.ld_json) or _iso_date(m.get("publish-date"))
                 or _iso_date(m.get("pubdate")) or _iso_date(m.get("date")))
    return Article(url=url, title=_clean_title(title, site),
                   description=re.sub(r"\s+", " ", unescape(desc)).strip() if desc else None,
                   published=published, site_name=site)


def fetch_article(url: str, timeout: float = TIMEOUT_S) -> Article:
    """Never raises: on any failure returns Article(error=...) and the caller falls back to the URL alone."""
    import requests

    from src.source.credibility import is_platform, parse_domain

    if not is_article_url(url):
        return Article(url=url, error="not an article URL (http(s) link with a path)")
    if is_platform(parse_domain(url)[0]):
        return Article(url=url, error="social-media posts and short links are not fetched: paste the post's text")
    current = url.strip()
    try:
        for _ in range(MAX_REDIRECTS + 1):
            u = urlparse(current)
            if u.scheme not in ("http", "https") or not u.hostname or not _public_host(u.hostname):
                return Article(url=url, error="address not allowed (not a public web address)")
            with requests.get(current, timeout=timeout, stream=True, allow_redirects=False,
                              headers={"User-Agent": USER_AGENT, "Accept": "text/html"}) as r:
                if r.is_redirect or r.status_code in (301, 302, 303, 307, 308):
                    current = urljoin(current, r.headers.get("Location", ""))
                    continue
                if r.status_code != 200:
                    return Article(url=url, error=f"page returned HTTP {r.status_code}")
                if "html" not in r.headers.get("Content-Type", "").lower():
                    return Article(url=url, error="not an HTML page")
                body = b""
                for chunk in r.iter_content(65536):
                    body += chunk
                    if len(body) >= MAX_BYTES:
                        break
                html = body.decode(r.encoding or "utf-8", errors="replace")
                art = parse_html(html, current)
                if not art.title:
                    art.error = "no headline found on the page"
                return art
        return Article(url=url, error="too many redirects")
    except requests.RequestException as e:
        return Article(url=url, error=f"could not fetch the page ({type(e).__name__})")
