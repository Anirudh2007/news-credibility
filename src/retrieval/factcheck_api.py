"""Live evidence: Google Fact Check Tools API (claims:search).

Published fact-checks (ClaimReview markup from BOOM, Alt News, Factly, PIB, AFP, PolitiFact ...)
become extra evidence passages for the verifier, in the same "Q: ... A: ..." format as the
Indian fact-check passages in the offline index (`passages_from_factchecks`):

    Q: Is it true that <claim as reviewed>? A: No. BOOM rated this claim False. <article title>

They are re-ranked together with the index passages by the same cross-encoder, so a live
fact-check only enters the top 5 if it is about this claim. The meta-learner is not retrained:
its features (verdict score, relevance) mean the same thing whichever index the passage came
from (the "adapt through evidence" principle used for the Indian index).

Rules
- Off when no key is set (GOOGLE_FACTCHECK_API_KEY in the environment or in .env) or when
  NEWSCRED_LIVE_FACTCHECK=0. Dataset builds and benchmark evaluations run with it off: a live
  lookup on a dataset claim would find the fact-check the dataset was made from (leakage).
- `before=`: reviews dated after the claim are dropped, as for the offline index.
- Network: 4 s timeout, two failures in a row switch it off for 5 minutes (circuit breaker),
  answers cached in SQLite for 7 days (data/cache/factcheck.sqlite), so a demo repeats offline.
- The key is never logged, stored or returned.

Free quota: Google's default for this API is generous for a demo (thousands of queries a day);
each claim costs one query, cached.
"""

from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

ROOT = Path(__file__).resolve().parents[2]
API_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
# Docker: NEWSCRED_FACTCHECK_CACHE=/db/factcheck.sqlite (the writable database volume)
CACHE_PATH = Path(os.environ.get("NEWSCRED_FACTCHECK_CACHE", ROOT / "data" / "cache" / "factcheck.sqlite"))
CACHE_TTL_S = 7 * 24 * 3600
TIMEOUT_S = 4.0
BREAKER_FAILURES = 2
BREAKER_COOLDOWN_S = 300
PAGE_SIZE = 10  # per claim: the re-ranker keeps the relevant ones
ID_PREFIX = "gfc:"


def load_env_file(names: Sequence[str] = ("GOOGLE_FACTCHECK_API_KEY",)) -> None:
    """Set the named variables from news-credibility/.env or the project folder's .env, if they are
    not already in the environment. Minimal KEY=VALUE parser (python-dotenv is not a dependency)."""
    for path in (ROOT / ".env", ROOT.parent / ".env"):
        if all(os.environ.get(n) for n in names) or not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            key, sep, value = line.strip().partition("=")
            key = key.strip().removeprefix("export ").strip()
            if sep and key in names and not os.environ.get(key):
                value = value.strip().strip('"').strip("'")
                if value:
                    os.environ[key] = value


# --- Verdict wording -> credible value (the Phase 2 scale: 1 true, 0.5 mixed, 0 false) ----------

_FALSE = re.compile(r"\b(false|fake|misleading|incorrect|inaccurate|wrong|untrue|not true|baseless|hoax|scam|"
                    r"fabricated|distorted|manipulated|doctored|edited|altered|morphed|misattributed|"
                    r"false context|out of context|mostly false|pants on fire|four pinocchios|debunked|myth|"
                    r"satire shared as real|cheapfake|deepfake|ai[- ]generated)\b", re.I)
_PARTLY = re.compile(r"\b(partly|partially|half[- ]true|mixed|mixture|missing context|needs context|"
                     r"exaggerat\w*|cherry[- ]?pick\w*|unsubstantiated)\b", re.I)
_TRUE = re.compile(r"\b(true|correct|accurate|mostly true|real|genuine|verified)\b", re.I)
VERDICT_WORD = {0.0: "No", 0.5: "Partly", 1.0: "Yes"}


def rating_to_credible(textual_rating: str | None) -> float | None:
    """'False' / 'Misleading' -> 0.0, 'Half True' / 'Missing context' -> 0.5, 'True' -> 1.0.
    Anything else ('Satire', 'Unproven', 'Explainer', a sentence in Hindi) -> None (not used).
    Partly is tested before false ('Partly false' is mixed), false before true ('Not true')."""
    if not textual_rating:
        return None
    r = " ".join(str(textual_rating).split())
    if _PARTLY.search(r):
        return 0.5
    if _FALSE.search(r):
        return 0.0
    if _TRUE.search(r):
        return 1.0
    return None


_FC_PASSAGE = re.compile(r"^Q: Is it true that (?P<claim>.+?)\? A: (?P<answer>Yes|No|Partly)\. "
                         r"(?P<who>.+?) rated this claim (?P<rating>.+?)\.(?: (?P<title>.*))?$", re.S)


def parse_factcheck_passage(text: str) -> dict | None:
    """Fields of a fact-check passage (live or from the offline Indian index), else None."""
    m = _FC_PASSAGE.match(text or "")
    return m.groupdict() if m else None


@dataclass
class FactCheck:
    claim: str
    claimant: str | None
    claim_date: str | None
    publisher: str
    site: str | None
    url: str | None
    title: str | None
    review_date: str | None
    rating: str
    credible: float

    def passage(self):
        from src.retrieval.search import Passage

        text = (f"Q: Is it true that {self.claim[:300].rstrip(' ?.')}? A: {VERDICT_WORD[self.credible]}. "
                f"{self.publisher} rated this claim {self.rating.rstrip('.')}.")
        if self.title:
            text += f" {self.title.strip()[:200]}"
        return Passage(id=f"{ID_PREFIX}{self.url or self.claim[:80]}", text=text, source_url=self.url,
                       domain=self.site, date=self.review_date)


def parse_response(body: dict) -> list[FactCheck]:
    """One FactCheck per (claim, review) in a claims:search response with a usable English verdict."""
    out = []
    for c in body.get("claims", []) or []:
        for r in c.get("claimReview", []) or []:
            credible = rating_to_credible(r.get("textualRating"))
            lang = (r.get("languageCode") or "en").lower()
            if credible is None or not lang.startswith("en") or not c.get("text"):
                continue
            pub = r.get("publisher") or {}
            site = (pub.get("site") or "").lower().removeprefix("www.") or None
            out.append(FactCheck(
                claim=" ".join(c["text"].split()), claimant=c.get("claimant"),
                claim_date=(c.get("claimDate") or "")[:10] or None,
                publisher=pub.get("name") or site or "A fact-checker", site=site, url=r.get("url"),
                title=r.get("title"), review_date=(r.get("reviewDate") or "")[:10] or None,
                rating=" ".join(str(r.get("textualRating")).split()), credible=credible,
            ))
    return out


_STOP = frozenset("""a an the and or but of to in on at for by with from is are was were be been being it its this that
these those as has have had will would can could should may might do does did not no says said claims claimed
claiming viral video photo image post shows showing shared new just also than then there their they he she his her
""".split())


def keyword_query(text: str, n: int = 6) -> str:
    """'The new Rs 2000 note contains a GPS nano chip ...' -> 'Rs 2000 note contains GPS nano'. The API matches
    keywords, and a long sentence often returns nothing; this is the fallback query."""
    words = [w for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9-]*", text) if w.lower() not in _STOP and len(w) > 1]
    return " ".join(words[:n])


# --- Client ------------------------------------------------------------------------------------

class FactCheckClient:
    """search(claim, before) -> passages; merge(claim, hits, ...) -> top-k hits incl. live ones."""

    def __init__(self, api_key: str | None = None, cache_path: Path | None = CACHE_PATH,
                 fetch: Callable[[dict], dict] | None = None, timeout: float = TIMEOUT_S):
        self._key = api_key
        self._fetch = fetch  # tests inject a fake: params -> JSON body
        self.timeout = timeout
        self.cache_path = cache_path
        self._lock = threading.Lock()
        self._failures, self._open_until = 0, 0.0
        self.calls = self.cache_hits = 0
        self.last_error: str | None = None

    @property
    def status(self) -> str:
        if time.time() < self._open_until:
            return f"paused after errors ({self.last_error})"
        return "ready (live Google Fact Check lookups)"

    # cache ---------------------------------------------------------------------
    def _db(self):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(self.cache_path)
        con.execute("CREATE TABLE IF NOT EXISTS factcheck (query TEXT PRIMARY KEY, fetched REAL, body TEXT)")
        return con

    def _cache_get(self, query: str) -> dict | None:
        if self.cache_path is None:
            return None
        try:
            with self._lock, self._db() as con:
                row = con.execute("SELECT fetched, body FROM factcheck WHERE query = ?", (query,)).fetchone()
        except (OSError, sqlite3.Error):  # read-only folder, locked file: work without the cache
            self.cache_path = None
            return None
        if row and time.time() - row[0] < CACHE_TTL_S:
            return json.loads(row[1])
        return None

    def _cache_put(self, query: str, body: dict) -> None:
        if self.cache_path is None:
            return
        try:
            with self._lock, self._db() as con:
                con.execute("INSERT OR REPLACE INTO factcheck VALUES (?, ?, ?)", (query, time.time(), json.dumps(body)))
        except (OSError, sqlite3.Error):
            self.cache_path = None

    # network -------------------------------------------------------------------
    def _get(self, params: dict) -> dict:
        if self._fetch is not None:
            return self._fetch(params)
        import requests

        r = requests.get(API_URL, params={**params, "key": self._key}, timeout=self.timeout)
        if r.status_code != 200:  # the message can echo the request URL: never pass it on (it holds the key)
            raise RuntimeError(f"HTTP {r.status_code}")
        return r.json()

    def query(self, params: dict) -> dict:
        """Raw claims:search call with cache and circuit breaker. {} when unavailable."""
        cache_key = json.dumps(params, sort_keys=True)
        cached = self._cache_get(cache_key)
        if cached is not None:
            self.cache_hits += 1
            return cached
        if time.time() < self._open_until:
            return {}
        try:
            self.calls += 1
            body = self._get(params)
        except Exception as e:  # timeout, DNS, quota, bad key
            self._failures += 1
            self.last_error = type(e).__name__ if not isinstance(e, RuntimeError) else str(e)
            if self._failures >= BREAKER_FAILURES:
                self._open_until = time.time() + BREAKER_COOLDOWN_S
            return {}
        self._failures = 0
        self._cache_put(cache_key, body)
        return body

    def search(self, claim: str, before: str | None = None, language: str = "en") -> list:
        """Passages for published reviews of claims like `claim`, dated on/before `before`."""
        q = " ".join((claim or "").split())[:500]
        if not q:
            return []
        body = self.query({"query": q, "languageCode": language, "pageSize": PAGE_SIZE})
        if not body.get("claims") and (kq := keyword_query(q)) and kq != q:  # one retry with keywords only
            body = self.query({"query": kq, "languageCode": language, "pageSize": PAGE_SIZE})
        out, seen = [], set()
        for fc in parse_response(body):
            if before and (not fc.review_date or fc.review_date > str(before)[:10]):
                # No hindsight: a review written after the claim cannot decide it. An UNDATED review cannot be
                # shown to be earlier, so it is dropped too (2026-11-03: 51 undated reviews of the same claim
                # reached the evidence of the 2023-2025 test fakes). In the app `before` is usually None.
                continue
            p = fc.passage()
            if p.id not in seen:
                seen.add(p.id)
                out.append(p)
        return out

    def merge(self, claim: str, hits: Sequence, before: str | None = None, reranker=None, k: int = 5) -> list:
        """Index hits + live fact-checks, re-ranked together, top `k`. Without a re-ranker the live
        passages get the best index score (they matched Google's own claim search)."""
        from src.retrieval.search import Hit

        live = self.search(claim, before)
        if not live:
            return list(hits)
        known = {h.passage.source_url for h in hits if h.passage.source_url}
        live = [p for p in live if not p.source_url or p.source_url not in known]
        if reranker is not None:
            scores = list(map(float, reranker(claim, [p.text for p in live])))
        else:
            best = max((h.score for h in hits), default=0.0)
            scores = [best] * len(live)
        merged = list(hits) + [Hit(p, s, None, None) for p, s in zip(live, scores)]
        merged.sort(key=lambda h: -h.score)
        return merged[:k]


def default_client() -> FactCheckClient | None:
    """A client if a key is configured and live lookups are not switched off, else None."""
    if os.environ.get("NEWSCRED_LIVE_FACTCHECK", "1") == "0":
        return None
    load_env_file()
    key = os.environ.get("GOOGLE_FACTCHECK_API_KEY")
    return FactCheckClient(api_key=key) if key else None


def relevance(score: float) -> float:
    """Re-ranker logit -> 0-1, the same sigmoid the verifier uses for `evidence_relevance`."""
    return 1.0 / (1.0 + math.exp(-score))
