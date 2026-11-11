"""News corroboration: does reporting by trusted outlets back the claim? (Tavily or GDELT)

Why this exists: every other evidence source is a fact-check, and fact-checkers almost only publish
"False" (Indian fact-checkers: 1 "True" per 117 "False", count of 2026-11-03). So a true claim could
never be *supported*: it found nothing (no score) or a debunk of a similar rumour (wrongly Low).
News reports are the evidence that can say "yes, this happened". Since 2026-11-05 the same search also
decides when silence is evidence: a big claim that no trusted outlet reported (src/retrieval/coverage.py).

Backends
- Tavily (api.tavily.com, default when TAVILY_API_KEY is set): a web search API built for AI applications;
  free plan 1,000 searches a month, no card. The search is restricted to trusted outlets
  (`include_domains`, up to 300) and to a date range (`start_date` / `end_date`).
- GDELT DOC 2.0 (api.gdeltproject.org, NEWSCRED_NEWS_SEARCH=gdelt): free, no key, open for research,
  but it answered HTTP 429 to this project for days (2026-11-03 to 11-05), so it is only a fallback.
- Not used: Google News RSS. It works without a key, but its feed terms allow only "rendering Google News
  results within a personal feed reader for personal, non-commercial use".

How
- Query: the claim (Tavily) or its keywords (GDELT). Window: for a dated claim the 7 days up to the claim
  date; for an undated claim everything up to today (Tavily), so a real past event still finds its reports.
  Nothing published after the claim date is used (no hindsight), as for the evidence index.
- Only outlets rated >= TRUSTED_MIN in data/reference/sources.csv are kept (the 1,082 rated domains:
  Wikipedia RSP "generally reliable", IFCN members, official bodies, established Indian outlets).
- Each article becomes a passage "<outlet> reported (<date>): <title>. <snippet>". The re-ranker and the
  verifier treat it like any other passage (confirm-only, see src/verify/model.py), so the meta-learner
  is not retrained (same principle as the live fact-check lookup: adapt through evidence).
- `lookup()` also says whether the search really ran (`ok`): a timeout or a used-up quota is not silence.

Rules (same as factcheck_api): timeout, circuit breaker after two failures, answers cached in SQLite
for 30 days (data/cache/news.sqlite). Dataset builds for the meta-learner run with it off.

Citation (GDELT): Leetaru, K. & Schrodt, P. (2013). GDELT: Global Data on Events, Location and Tone.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, Sequence
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
TAVILY_URL = "https://api.tavily.com/search"
CACHE_PATH = Path(os.environ.get("NEWSCRED_NEWS_CACHE", ROOT / "data" / "cache" / "news.sqlite"))
CACHE_TTL_S = 30 * 24 * 3600
TIMEOUT_S = float(os.environ.get("NEWSCRED_NEWS_TIMEOUT", "20"))
MIN_INTERVAL_S = {"gdelt": 5.2, "tavily": 0.2}  # GDELT's rate rule: one request per 5 s
BREAKER_FAILURES = 2
BREAKER_COOLDOWN_S = 300
QUOTA_COOLDOWN_S = 6 * 3600  # Tavily 432/433: the monthly quota is used up; do not keep asking
WINDOW_DAYS = 7
MAX_RECORDS = 50  # GDELT
TAVILY_RESULTS = 10
MAX_DOMAINS = 300  # Tavily's include_domains limit
TRUSTED_MIN = 0.75
SNIPPET_CHARS = 300
ID_PREFIX = "news:"
USER_AGENT = "NewsCredibilityCollegeProject/1.0 (academic)"

# Outlets searched first when the trusted list is longer than MAX_DOMAINS: Indian national outlets and
# official sites, then international wires and broadcasters; the rest by rating.
INDIAN_OUTLETS = frozenset("""thehindu.com indianexpress.com ndtv.com hindustantimes.com timesofindia.indiatimes.com
indiatoday.in livemint.com business-standard.com deccanherald.com scroll.in thewire.in theprint.in
economictimes.indiatimes.com news18.com firstpost.com tribuneindia.com telegraphindia.com newindianexpress.com
moneycontrol.com thehindubusinessline.com financialexpress.com outlookindia.com thequint.com newslaundry.com
pib.gov.in rbi.org.in mea.gov.in""".split())
WIRES = frozenset("""reuters.com apnews.com bbc.com bbc.co.uk aljazeera.com dw.com theguardian.com nytimes.com
washingtonpost.com cnn.com afp.com france24.com npr.org bloomberg.com ft.com economist.com wsj.com
abcnews.go.com cbsnews.com nbcnews.com""".split())
NOT_NEWS = frozenset({"wikipedia.org", "en.wikipedia.org"})  # Wikipedia is its own channel
# A headline that is a fact-check or debunk, not a report of the event (narrow on purpose: "fake currency racket
# busted" is a real event, so "fake" alone does not count).
DEBUNK = re.compile(r"\b(?:fact[\s-]?check\w*|hoax(?:es)?|fake news|false claims?|misleading claims?|"
                    r"(?:is|are|was|were) (?:fake|false))\b", re.I)

_PASSAGE = re.compile(r"^(?P<outlet>\S+) reported \((?P<date>[0-9-]+)\): (?P<title>.+)$", re.S)


def clean_title(title: str) -> str:
    """GDELT spaces out punctuation ('Chandrayaan - 3 Lands : ISRO'); undo it."""
    t = re.sub(r"\s+([:;,.!?%)\]'’])", r"\1", title or "")
    t = re.sub(r"([(\[‘])\s+", r"\1", t)
    t = re.sub(r"(\w)\s+-\s+(\d)", r"\1-\2", t)
    t = re.sub(r"(\w)(['’])\s+(s|t|re|ll|ve|d|m)\b", r"\1\2\3", t)
    return " ".join(t.split()).strip()


def parse_news_passage(text: str) -> dict | None:
    m = _PASSAGE.match(text or "")
    return m.groupdict() if m else None


def _day(value: str | date | None) -> date | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value
    try:
        return datetime.fromisoformat(str(value)[:10]).date()
    except ValueError:
        return None


def _published(value: str | None) -> str | None:
    """Tavily's published_date (RFC 2822 'Thu, 24 Aug 2023 07:00:00 GMT' or ISO) -> 'YYYY-MM-DD'."""
    if not value:
        return None
    from email.utils import parsedate_to_datetime

    try:
        return parsedate_to_datetime(value).date().isoformat()
    except (TypeError, ValueError, IndexError):
        d = _day(value)
        return d.isoformat() if d else None


def _domain(url: str | None) -> str:
    return (urlparse(url or "").netloc or "").lower().split(":")[0].removeprefix("www.")


def trusted_domains(reliability: dict[str, float], trusted_min: float = TRUSTED_MIN, limit: int = MAX_DOMAINS) -> list[str]:
    """The outlets a Tavily search is restricted to: rated >= trusted_min, Indian outlets and official sites
    first, then wires, then the rest by rating."""
    def rank(d: str) -> tuple:
        indian = d in INDIAN_OUTLETS or d.endswith(".in")
        return (0 if indian else 1 if d in WIRES else 2, -reliability[d], d)

    doms = [d for d, r in reliability.items() if r >= trusted_min and d not in NOT_NEWS and "." in d]
    return sorted(doms, key=rank)[:limit]


@dataclass
class NewsLookup:
    """One search for one claim. ok=False: the search did not run (no answer, quota, breaker), so finding
    nothing must not be read as silence."""
    passages: list = field(default_factory=list)
    ok: bool = False
    n_results: int = 0  # articles returned before the trusted-outlet filter
    start: str | None = None
    end: str | None = None
    backend: str = ""
    # Trusted reports left out of the evidence only because they carry no date (strict_dates: an undated report
    # cannot be shown to be from before the claim). They still show that outlets covered the topic, so they count
    # against silence (`relevant_reports`); a URL naming a later year is left out of both.
    undated: list = field(default_factory=list)
    # Ids of reports that are themselves fact-checks / debunks ("Fact check: viral claim that X died is false").
    # They exist because the claim is false, so they are not the coverage a true claim would get: they never count
    # against silence (they remain ordinary evidence).
    debunk_ids: set = field(default_factory=set)
    queries: int = 1  # 2 when the claim was searched twice (`lookup_twice`)


class NewsSearchClient:
    """lookup(claim, before, start) -> NewsLookup; search(...) -> trusted passages; merge(...) adds them to a
    claim's hits."""

    def __init__(self, reliability: dict[str, float] | None = None, cache_path: Path | None = CACHE_PATH,
                 fetch: Callable[[dict], list[dict]] | None = None, timeout: float = TIMEOUT_S,
                 trusted_min: float = TRUSTED_MIN, retries: int = 0, backend: str = "gdelt",
                 api_key: str | None = None):
        from src.verify.model import load_source_reliability

        self.reliability = load_source_reliability() if reliability is None else reliability
        self.backend = backend
        self.api_key = api_key
        self._fetch = fetch  # tests: params -> list of article dicts in the backend's own format
        self.timeout = timeout
        # Evaluation runs pass retries > 0: on HTTP 429 wait and retry instead of skipping the item
        # (a skipped item would quietly count as "no news found"). The app keeps 0: answer fast.
        self.retries = retries
        self.trusted_min = trusted_min
        self.domains = trusted_domains(self.reliability, trusted_min) if backend == "tavily" else []
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
        if self.backend == "tavily":
            return f"ready (Tavily news search, {len(self.domains)} trusted outlets)"
        return "ready (GDELT news search)"

    # cache -------------------------------------------------------------------------------------
    def _db(self):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(self.cache_path)
        con.execute("CREATE TABLE IF NOT EXISTS news (query TEXT PRIMARY KEY, fetched REAL, body TEXT)")
        return con

    def _cache_get(self, key: str) -> list | None:
        if self.cache_path is None:
            return None
        try:
            with self._lock, self._db() as con:
                row = con.execute("SELECT fetched, body FROM news WHERE query = ?", (key,)).fetchone()
        except (OSError, sqlite3.Error):
            self.cache_path = None
            return None
        if row and time.time() - row[0] < CACHE_TTL_S:
            return json.loads(row[1])
        return None

    def _cache_put(self, key: str, body: list) -> None:
        if self.cache_path is None:
            return
        try:
            with self._lock, self._db() as con:
                con.execute("INSERT OR REPLACE INTO news VALUES (?, ?, ?)", (key, time.time(), json.dumps(body)))
        except (OSError, sqlite3.Error):
            self.cache_path = None

    # network -----------------------------------------------------------------------------------
    def _wait_turn(self) -> None:
        wait = MIN_INTERVAL_S.get(self.backend, 1.0) - (time.time() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.time()

    def _get(self, params: dict) -> list[dict]:
        if self._fetch is not None:
            return self._fetch(params)
        import requests

        self._wait_turn()
        if self.backend == "tavily":
            r = requests.post(TAVILY_URL, json=params, timeout=self.timeout,
                              headers={"Authorization": f"Bearer {self.api_key}", "User-Agent": USER_AGENT})
            if r.status_code in (432, 433):
                self._open_until = time.time() + QUOTA_COOLDOWN_S
                raise RuntimeError(f"HTTP {r.status_code} (Tavily quota used up)")
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}")
            return r.json().get("results", []) or []
        r = requests.get(API_URL, params=params, timeout=self.timeout, headers={"User-Agent": USER_AGENT})
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}")
        txt = r.text.strip()
        if not txt.startswith("{"):  # GDELT answers plain text for a query it rejects (too short, etc.)
            return []
        return json.loads(txt).get("articles", []) or []

    def _params(self, query: str, start: date | None, end: date) -> dict:
        if self.backend == "tavily":
            p = {"query": query[:400], "search_depth": "basic", "max_results": TAVILY_RESULTS,
                 "include_domains": self.domains, "include_published_date": True, "include_answer": False,
                 "include_raw_content": False, "include_images": False, "end_date": end.isoformat()}
            # Always the general (web) index with the date range. Its "news" index looked better (every result
            # dated) but missed the reports themselves: for Ratan Tata's death week it returned hurricane and
            # Pfizer stories; the general index with the same dates returned 10 of 10 relevant reports, and 9 of 9
            # for the electoral bonds verdict (checked 2026-11-06).
            p["topic"] = "general"
            if start is not None:
                p["start_date"] = start.isoformat()
            return p
        start = start or end - timedelta(days=WINDOW_DAYS)
        return {"query": f"{query} sourcelang:english", "mode": "artlist", "format": "json",
                "maxrecords": MAX_RECORDS, "sort": "hybridrel",
                "startdatetime": start.strftime("%Y%m%d000000"), "enddatetime": end.strftime("%Y%m%d235959")}

    def query(self, keywords: str, start: date | None, end: date) -> list[dict] | None:
        """Raw articles for one query, or None when the search did not run (error, breaker, quota)."""
        params = self._params(keywords, start, end)
        key = json.dumps({"backend": self.backend, **{k: v for k, v in params.items() if k != "include_domains"},
                          "n_domains": len(params.get("include_domains", []))}, sort_keys=True)
        cached = self._cache_get(key)
        if cached is not None:
            self.cache_hits += 1
            return cached
        if time.time() < self._open_until:
            return None
        try:
            self.calls += 1
            for attempt in range(self.retries + 1):
                try:
                    arts = self._get(params)
                    break
                except Exception as e:
                    if attempt == self.retries or "quota" in str(e):
                        raise
                    time.sleep(60 * (attempt + 1))
        except Exception as e:  # timeout, DNS, rate limit, quota
            self._failures += 1
            self.failed_queries += 1
            self.last_error = type(e).__name__ if not isinstance(e, RuntimeError) else str(e)
            if self._failures >= BREAKER_FAILURES:
                self._open_until = max(self._open_until, time.time() + BREAKER_COOLDOWN_S)
            return None
        self._failures = 0
        self._cache_put(key, arts)
        return arts

    # passages ----------------------------------------------------------------------------------
    def _normalise(self, a: dict) -> dict:
        if self.backend == "tavily":
            return {"title": clean_title(a.get("title", "")), "domain": _domain(a.get("url")), "url": a.get("url"),
                    "date": _published(a.get("published_date")), "content": " ".join((a.get("content") or "").split())}
        seen_date = (a.get("seendate") or "")[:8]
        iso = f"{seen_date[:4]}-{seen_date[4:6]}-{seen_date[6:8]}" if len(seen_date) == 8 else None
        return {"title": clean_title(a.get("title", "")), "domain": (a.get("domain") or "").lower().removeprefix("www."),
                "url": a.get("url"), "date": iso, "content": ""}

    def lookup(self, claim: str, before: str | date | None = None, start: str | date | None = None,
               strict_dates: bool = False) -> NewsLookup:
        """Trusted-outlet articles about the claim, published between `start` (None: GDELT 7 days before the
        end, Tavily no limit) and `before` (default today). strict_dates: the claim or item has a date, so an
        undated article (which cannot be shown to be earlier) is dropped: no hindsight."""
        from src.retrieval.factcheck_api import keyword_query
        from src.verify.model import reliability_of

        end = min(_day(before) or date.today(), date.today())
        start_d = _day(start)
        raw: list[dict] | None = None
        if self.backend == "tavily":
            q = " ".join((claim or "").split())
            raw = self.query(q, start_d, end) if len(q.split()) >= 2 else []
        else:
            for n in (6, 4):
                kw = keyword_query(claim or "", n=n)
                if len(kw.split()) < 2:
                    raw = raw if raw is not None else []
                    break
                raw = self.query(kw, start_d, end)
                if raw:
                    break
        out, undated, seen = [], [], set()
        for a in map(self._normalise, raw or []):
            rel = reliability_of(a["domain"], self.reliability, default=0.0)
            if (not a["title"] or rel < self.trusted_min or a["domain"] in NOT_NEWS
                    or (a["date"] and a["date"] > end.isoformat())
                    or (a["date"] and start_d and a["date"] < start_d.isoformat()) or a["title"].lower() in seen):
                continue
            if strict_dates and not a["date"]:
                if not url_after(a["url"], end):
                    undated.append({**a, "reliability": rel})
                continue
            seen.add(a["title"].lower())
            out.append({**a, "reliability": rel})
        passages = [self._passage(a) for a in out]
        late = [self._passage(a) for a in undated]
        debunks = {p.id for a, p in zip(out + undated, passages + late) if DEBUNK.search(a["title"])}
        return NewsLookup(passages, raw is not None, len(raw or []), start_d.isoformat() if start_d else None,
                          end.isoformat(), self.backend, late, debunks)

    def lookup_twice(self, claim: str, before: str | date | None = None, start: str | date | None = None,
                     strict_dates: bool = False, reranker=None) -> NewsLookup:
        """Listen twice before calling it silence. If the claim's own words find no relevant trusted report, search
        once more with its keywords ("Indian Navy rescues crew of hijacked ship MV Lila Norfolk" ->
        "Indian Navy rescues crew hijacked ship"): a second phrasing finds reports the first missed. Only for
        claims the news would cover, so it costs at most one extra search for such a claim, and only when the
        first found nothing. Relevance is always judged against the claim itself."""
        from src.retrieval.factcheck_api import keyword_query

        first = self.lookup(claim, before, start, strict_dates)
        if self.backend != "tavily" or not first.ok or relevant_reports(claim, first, reranker):
            return first
        kw = keyword_query(claim or "", n=6)
        if len(kw.split()) < 2 or kw.lower() == " ".join((claim or "").split()).lower():
            return first
        second = self.lookup(kw, before, start, strict_dates)
        if not second.ok:  # the second search did not run: the first answer stands
            return first
        seen = {p.id for p in first.passages + first.undated}
        return NewsLookup(first.passages + [p for p in second.passages if p.id not in seen], True,
                          first.n_results + second.n_results, first.start, first.end, first.backend,
                          first.undated + [p for p in second.undated if p.id not in seen],
                          first.debunk_ids | second.debunk_ids, 2)

    @staticmethod
    def _passage(a: dict):
        from src.retrieval.search import Passage

        text = f"{a['domain']} reported ({a['date'] or 'undated'}): {a['title']}"
        snippet = a.get("content", "")[:SNIPPET_CHARS]
        if snippet and snippet.lower() not in a["title"].lower():
            text += f". {snippet}" if not a["title"].endswith((".", "!", "?")) else f" {snippet}"
        return Passage(id=f"{ID_PREFIX}{a['url'] or a['title'][:80]}", text=text, source_url=a["url"],
                       domain=a["domain"], date=a["date"])

    def articles(self, claim: str, before: str | date | None = None, start: str | date | None = None) -> list[dict]:
        """Trusted-outlet articles (title, domain, url, date) about the claim (see `lookup`)."""
        lk = self.lookup(claim, before, start)
        return [{"title": parse_news_passage(p.text)["title"] if parse_news_passage(p.text) else p.text,
                 "domain": p.domain, "url": p.source_url, "date": p.date} for p in lk.passages]

    def search(self, claim: str, before: str | date | None = None, start: str | date | None = None) -> list:
        return self.lookup(claim, before, start).passages

    def merge(self, claim: str, hits: Sequence, before: str | date | None = None, reranker=None, k: int = 5,
              start: str | date | None = None, lookup: NewsLookup | None = None) -> list:
        """Index (and fact-check) hits + trusted news passages, re-ranked together, top `k`."""
        from src.retrieval.search import Hit

        news = (lookup or self.lookup(claim, before, start)).passages
        if not news:
            return list(hits)
        known = {h.passage.source_url for h in hits if h.passage.source_url}
        news = [p for p in news if not p.source_url or p.source_url not in known]
        if not news:
            return list(hits)
        if reranker is not None:
            scores = list(map(float, reranker(claim, [p.text for p in news])))
        else:
            best = max((h.score for h in hits), default=0.0)
            scores = [best] * len(news)
        merged = list(hits) + [Hit(p, s, None, None) for p, s in zip(news, scores)]
        merged.sort(key=lambda h: -h.score)
        return merged[:k]


def url_after(url: str | None, end: date) -> bool:
    """The URL names a year after `end` (".../2025/03/..."): a later report, whatever its missing date."""
    return any(int(y) > end.year for y in re.findall(r"(?<!\d)(20[0-3]\d)(?!\d)", url or ""))


def relevant_reports(claim: str, lookup: NewsLookup | None, reranker=None) -> int:
    """How many trusted reports the search returned are about the claim (re-ranker relevance >= RELEVANT_REPORT).
    Counted over everything the search returned, undated reports included: silence is a property of the news
    search. Two mistakes this avoids (both in the 6 Nov measurement, src/fusion/india_tune.py `absence-recheck`):
    counting only the claim's top-5 evidence, where Wikipedia, fact-check and index passages can push a relevant
    report out; and ignoring undated reports, which made 42% of real big events look unreported instead of 6%.
    Debunks ("Fact check: ...") are not counted: they exist because the claim is false. Without a re-ranker every
    other returned report counts."""
    from src.retrieval.coverage import RELEVANT_REPORT
    from src.retrieval.factcheck_api import relevance

    passages = [p for p in lookup.passages + lookup.undated if p.id not in lookup.debunk_ids] if lookup is not None else []
    if not passages:
        return 0
    if reranker is None:
        return len(passages)
    return sum(relevance(float(s)) >= RELEVANT_REPORT for s in reranker(claim, [p.text for p in passages]))


def default_client() -> NewsSearchClient | None:
    """NEWSCRED_NEWS_SEARCH: "0" off; "gdelt" (or "1", the old switch) GDELT; "tavily" Tavily; unset: Tavily
    when TAVILY_API_KEY is set (environment or .env), else off. GDELT blocked this project's IP for days
    after a short burst (2026-11-03, HTTP 429 despite one request per 5 s), so it is never the default."""
    from src.retrieval.factcheck_api import load_env_file

    flag = os.environ.get("NEWSCRED_NEWS_SEARCH", "").strip().lower()
    if flag == "0":
        return None
    if flag in ("1", "gdelt"):
        return NewsSearchClient(backend="gdelt")
    load_env_file(("TAVILY_API_KEY",))
    key = os.environ.get("TAVILY_API_KEY", "").strip()
    if key:
        return NewsSearchClient(backend="tavily", api_key=key)
    return None
