"""Wikipedia corroboration: what did Wikipedia say about this, *on the day of the claim*?

Why: fact-checks (every other evidence source) exist almost only for false claims, so a true claim had
nothing that could support it. Encyclopaedic text is the classic evidence source for claim verification
(FEVER, Thorne et al. 2018, verifies claims against Wikipedia).

How
- MediaWiki search (English Wikipedia, articles only) with the claim's keywords: up to MAX_PAGES pages.
- For each page, the **revision that was current at the end of the claim date** (rvstart = claim date,
  rvdir=older), so nothing written after the claim is used. A page created later is skipped.
- Wikitext -> plain sentences; two-sentence windows that share the most keywords with the claim become
  passages "Wikipedia, <page> (as of <date>): <text>". The re-ranker and verifier then treat them like
  any other passage; claim matching does not apply (an article is not a verdict on one claim).
- Never used: the "Current events" portal (the true half of the India test set was written there).

Rules: off with NEWSCRED_WIKI_SEARCH=0 and for all dataset builds and benchmarks except the India
evaluation; 10 s timeout; circuit breaker; SQLite cache (data/cache/wiki.sqlite); User-Agent as
Wikimedia's API policy asks. Text is CC BY-SA (Wikipedia contributors).
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import time
from datetime import date, datetime
from pathlib import Path
from typing import Callable, Sequence

ROOT = Path(__file__).resolve().parents[2]
API_URL = "https://en.wikipedia.org/w/api.php"
CACHE_PATH = Path(os.environ.get("NEWSCRED_WIKI_CACHE", ROOT / "data" / "cache" / "wiki.sqlite"))
TIMEOUT_S = 10.0
# Wikimedia's 2026 limits: 10 requests/min for clients that identify only by IP, 200/min when the
# User-Agent carries contact details (an email or URL). NEWSCRED_CONTACT (in .env) adds them; without
# it the client paces itself to stay under 10/min.
def _contact() -> str:
    from src.retrieval.factcheck_api import load_env_file

    load_env_file(("NEWSCRED_CONTACT",))
    return os.environ.get("NEWSCRED_CONTACT", "").strip()


CONTACT = _contact()
HEADERS = {"User-Agent": f"NewsCredibilityCollegeProject/1.0 ({CONTACT + '; ' if CONTACT else ''}academic) python-requests"}
MIN_INTERVAL_S = float(os.environ.get("NEWSCRED_WIKI_INTERVAL", "0.35" if CONTACT else "6.5"))
BREAKER_FAILURES = 3
BREAKER_COOLDOWN_S = 120
MAX_PAGES = 2  # each page costs one request (its revision as of the claim date)
PASSAGES_PER_CLAIM = 6
ID_PREFIX = "wiki:"
WELL_KNOWN_VIEWS = 3000  # page views in the last 30 days (~100 a day) for a name to count as well known

_PASSAGE = re.compile(r"^Wikipedia, (?P<page>.+?) \(as of (?P<date>[0-9-]+)\): (?P<text>.+)$", re.S)
_STOP = frozenset("""a an the and or but of to in on at for by with from is are was were be been being it its this that
these those as has have had will would can could should may might do does did not no says said after before over
into about than then there their they he she his her who which what when where while during new also more most""".split())


def parse_wiki_passage(text: str) -> dict | None:
    m = _PASSAGE.match(text or "")
    return m.groupdict() if m else None


def _day(value) -> date | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value
    try:
        return datetime.fromisoformat(str(value)[:10]).date()
    except ValueError:
        return None


def keywords(text: str) -> list[str]:
    return [w for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9-]*", text or "") if w.lower() not in _STOP and len(w) > 2]


def _strip_templates(t: str) -> str:
    """Remove {{...}} including nested ones (infoboxes, citation templates)."""
    out, depth, i = [], 0, 0
    while i < len(t):
        if t.startswith("{{", i):
            depth += 1
            i += 2
        elif t.startswith("}}", i) and depth:
            depth -= 1
            i += 2
        else:
            if depth == 0:
                out.append(t[i])
            i += 1
    return "".join(out)


def wikitext_to_sentences(wikitext: str) -> list[str]:
    t = re.sub(r"<!--.*?-->", " ", wikitext or "", flags=re.S)
    t = re.sub(r"<ref[^>]*/>|<ref.*?</ref>", " ", t, flags=re.S)
    t = _strip_templates(t)
    t = re.sub(r"\{\|.*?\|\}", " ", t, flags=re.S)  # tables
    t = re.sub(r"\[\[(?:File|Image|Category):[^\]]*\]\]", " ", t)
    t = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", t)
    t = re.sub(r"\[https?://\S+\s*([^\]]*)\]", r"\1", t)
    t = re.sub(r"'{2,}|<[^>]+>", "", t)
    lines = [ln.strip() for ln in t.splitlines()]
    prose = " ".join(ln for ln in lines if ln and not ln.startswith(("=", "|", "!", "*", "#", "{", "}", ":")))
    sents = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", " ".join(prose.split()))
    return [s for s in sents if 6 <= len(s.split()) <= 80]


class WikiSearchClient:
    def __init__(self, cache_path: Path | None = CACHE_PATH, fetch: Callable[[dict], dict] | None = None,
                 timeout: float = TIMEOUT_S, retries: int = 0):
        self._fetch = fetch  # tests: params -> JSON body
        self.timeout = timeout
        self.retries = retries
        self.cache_path = cache_path
        self._lock = threading.Lock()
        self._last_call = 0.0
        self._failures, self._open_until = 0, 0.0
        self.calls = self.cache_hits = self.failed_queries = 0
        self.last_error: str | None = None

    @property
    def status(self) -> str:
        if time.time() < self._open_until:
            return f"paused after errors ({self.last_error})"
        return "ready (Wikipedia, revision as of the claim date)"

    def _db(self):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(self.cache_path)
        con.execute("CREATE TABLE IF NOT EXISTS wiki (query TEXT PRIMARY KEY, body TEXT)")
        return con

    def _cache_get(self, key: str):
        if self.cache_path is None:
            return None
        try:
            with self._lock, self._db() as con:
                row = con.execute("SELECT body FROM wiki WHERE query = ?", (key,)).fetchone()
        except (OSError, sqlite3.Error):
            self.cache_path = None
            return None
        return json.loads(row[0]) if row else None

    def _cache_put(self, key: str, body) -> None:
        if self.cache_path is None:
            return
        try:
            with self._lock, self._db() as con:
                con.execute("INSERT OR REPLACE INTO wiki VALUES (?, ?)", (key, json.dumps(body)))
        except (OSError, sqlite3.Error):
            self.cache_path = None

    def _get(self, params: dict) -> dict:
        if self._fetch is not None:
            return self._fetch(params)
        import requests

        wait = MIN_INTERVAL_S - (time.time() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.time()
        sent = {k: v for k, v in params.items() if not k.startswith("_")}  # "_month": part of the cache key only
        r = requests.get(API_URL, params={**sent, "format": "json", "formatversion": 2}, headers=HEADERS,
                         timeout=self.timeout)
        if r.status_code == 429:
            self._retry_after = float(r.headers.get("Retry-After") or 30)
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}")
        return r.json()

    def query(self, params: dict) -> dict:
        """Cached API call (old revisions never change, so no expiry). {} when unavailable."""
        key = json.dumps(params, sort_keys=True)
        cached = self._cache_get(key)
        if cached is not None:
            self.cache_hits += 1
            return cached
        if time.time() < self._open_until:
            return {}
        try:
            self.calls += 1
            for attempt in range(self.retries + 1):
                try:
                    body = self._get(params)
                    break
                except Exception:
                    if attempt == self.retries:
                        raise
                    time.sleep(max(getattr(self, "_retry_after", 0.0), 10 * (attempt + 1)))
                    self._retry_after = 0.0
        except Exception as e:
            self._failures += 1
            self.failed_queries += 1
            self.last_error = type(e).__name__ if not isinstance(e, RuntimeError) else str(e)
            if self._failures >= BREAKER_FAILURES:
                self._open_until = time.time() + BREAKER_COOLDOWN_S
            return {}
        self._failures = 0
        self._cache_put(key, body)
        return body

    def pages(self, claim: str, n: int = MAX_PAGES) -> list[str]:
        kw = keywords(claim)[:8]
        if len(kw) < 2:
            return []
        body = self.query({"action": "query", "list": "search", "srsearch": " ".join(kw), "srnamespace": 0,
                           "srlimit": n, "srprop": ""})
        titles = [h["title"] for h in (body.get("query", {}).get("search") or [])]
        return [t for t in titles if not t.startswith(("Portal:", "List of"))]

    def is_well_known(self, name: str) -> bool | None:
        """Would the news report an event about `name`? Test: an English Wikipedia article (after redirects,
        not a disambiguation page) read at least WELL_KNOWN_VIEWS times in the last 30 days. An article alone
        is not enough: "Raju Sharma" has one (43 views in 30 days, 2026-11-05) and any private Raju Sharma
        would pass; Shah Rukh Khan 199k, Virat Kohli 123k, Government of India 31k. None: Wikipedia could not
        be asked. Used by the expected-coverage rule (src/retrieval/coverage.py)."""
        name = " ".join((name or "").split())[:120]
        if len(name) < 3:
            return False
        body = self.query({"action": "query", "titles": name, "redirects": 1, "prop": "pageprops|pageviews",
                           "ppprop": "disambiguation", "pvipdays": 30, "_month": date.today().strftime("%Y-%m")})
        if not body:
            return None
        pages = body.get("query", {}).get("pages") or []
        if not pages:
            return False
        page = pages[0]
        if page.get("missing") or page.get("invalid") or "disambiguation" in (page.get("pageprops") or {}):
            return False
        return sum(v or 0 for v in (page.get("pageviews") or {}).values()) >= WELL_KNOWN_VIEWS

    def revision_as_of(self, title: str, day: date) -> tuple[str, str] | None:
        """(wikitext, timestamp) of the revision current at the end of `day`, or None (page did not exist)."""
        body = self.query({"action": "query", "prop": "revisions", "titles": title, "rvprop": "content|timestamp",
                           "rvslots": "main", "rvlimit": 1, "rvdir": "older", "rvstart": f"{day.isoformat()}T23:59:59Z"})
        pages = body.get("query", {}).get("pages") or []
        if not pages or "revisions" not in pages[0]:
            return None
        rev = pages[0]["revisions"][0]
        return rev["slots"]["main"].get("content", ""), rev.get("timestamp", "")[:10]

    def search(self, claim: str, before=None) -> list:
        """Passages from Wikipedia as it stood at the end of the claim date (default today)."""
        from src.retrieval.search import Passage

        day = _day(before) or date.today()
        kw = {w.lower() for w in keywords(claim)}
        cands = []
        for title in self.pages(claim):
            rev = self.revision_as_of(title, day)
            if rev is None or rev[0].lstrip().lower().startswith("#redirect"):
                continue
            sents = wikitext_to_sentences(rev[0])
            for i in range(len(sents)):
                window = " ".join(sents[i:i + 2])
                overlap = len(kw & {w.lower() for w in keywords(window)})
                if overlap:
                    cands.append((overlap, title, rev[1], window))
        cands.sort(key=lambda c: -c[0])
        out, seen = [], set()
        for _, title, ts, window in cands:
            if window in seen:
                continue
            seen.add(window)
            out.append(Passage(id=f"{ID_PREFIX}{title}#{len(out)}", text=f"Wikipedia, {title} (as of {ts}): {window[:600]}",
                               source_url=f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                               domain="en.wikipedia.org", date=ts))
            if len(out) >= PASSAGES_PER_CLAIM:
                break
        return out

    def merge(self, claim: str, hits: Sequence, before=None, reranker=None, k: int = 5) -> list:
        from src.retrieval.search import Hit

        wiki = self.search(claim, before)
        if not wiki:
            return list(hits)
        if reranker is not None:
            scores = list(map(float, reranker(claim, [p.text for p in wiki])))
        else:
            best = max((h.score for h in hits), default=0.0)
            scores = [best] * len(wiki)
        merged = list(hits) + [Hit(p, s, None, None) for p, s in zip(wiki, scores)]
        merged.sort(key=lambda h: -h.score)
        return merged[:k]


def default_client() -> WikiSearchClient | None:
    """On in the app when NEWSCRED_CONTACT is set (200 requests/min: about 1 s per claim) or when
    NEWSCRED_WIKI_SEARCH=1; without contact details each claim would wait ~20 s (10 requests/min)."""
    flag = os.environ.get("NEWSCRED_WIKI_SEARCH")
    if flag == "0" or (flag != "1" and not CONTACT):
        return None
    return WikiSearchClient()
