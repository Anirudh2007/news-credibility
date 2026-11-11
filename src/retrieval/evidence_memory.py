"""Evidence memory: the system gets better with use, without trusting its users (Phase 8, 2026-11-05).

Every passage a *trusted live source* returned for a checked claim (a published fact-check from Google
Fact Check Tools, a report from a trusted outlet, Wikipedia as it stood on a date) is stored with its date
and source, and searched again for every later claim. So:
- a later claim worded differently ("SRK passed away" after "Shah Rukh Khan died") finds the same evidence
  (search by meaning, MiniLM sentence vectors, then the same cross-encoder re-ranking as all evidence);
- the system keeps working when a live service is slow, down or out of quota (demo resilience);
- each live search is paid for once (Tavily: 1,000 free searches a month).

What is NOT stored: anything a user typed, user feedback, or the system's own verdicts. Only text written
by the trusted sources themselves goes in, with its original id ("gfc:", "news:", "wiki:"), so every rule
that applies to that source still applies (fact-checks need a claim match; news and Wikipedia only
confirm). Users cannot teach it a falsehood (no data poisoning through the feedback form).

No hindsight: a passage is offered only to claims dated on or after it (same rule as the evidence index).
Off for every evaluation and dataset build (results must not depend on what was checked before);
NEWSCRED_EVIDENCE_MEMORY=0 turns it off in the app.
"""

from __future__ import annotations

import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MEMORY_PATH = Path(os.environ.get("NEWSCRED_MEMORY_PATH", ROOT / "data" / "cache" / "evidence_memory.sqlite"))
LIVE_PREFIXES = ("gfc:", "news:", "wiki:")  # passages from trusted live sources: the only ones remembered
CANDIDATES = 10  # nearest stored passages offered to the re-ranker per claim
MIN_COSINE = 0.35  # below this a stored passage is about something else (MiniLM cosine)
MAX_ITEMS = 50_000


class EvidenceMemory:
    def __init__(self, path: Path | None = MEMORY_PATH, encoder: Callable[[Sequence[str]], np.ndarray] | None = None):
        self.path = path
        self._encoder = encoder
        self._lock = threading.Lock()
        self._rows: list[tuple] | None = None  # (id, text, url, domain, date) in the order of `_vecs`
        self._vecs: np.ndarray | None = None
        self.added = 0

    # storage -----------------------------------------------------------------------------------
    def _db(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(self.path)
        con.execute("CREATE TABLE IF NOT EXISTS memory (id TEXT PRIMARY KEY, text TEXT, source_url TEXT, domain TEXT, "
                    "date TEXT, added REAL, vec BLOB)")
        return con

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        if self._encoder is None:
            from src.retrieval.search import sentence_transformer_encoder

            self._encoder = sentence_transformer_encoder()
        v = np.asarray(self._encoder(list(texts)), dtype=np.float32)
        return v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-12, None)

    def _load(self) -> None:
        if self._rows is not None:
            return
        rows, vecs = [], []
        if self.path is not None:
            try:
                with self._db() as con:
                    for pid, text, url, dom, d, blob in con.execute(
                            "SELECT id, text, source_url, domain, date, vec FROM memory ORDER BY added"):
                        rows.append((pid, text, url, dom, d))
                        vecs.append(np.frombuffer(blob, dtype=np.float32))
            except (OSError, sqlite3.Error):
                self.path = None
        self._rows = rows
        self._vecs = np.vstack(vecs) if vecs else np.zeros((0, 0), dtype=np.float32)

    @property
    def size(self) -> int:
        with self._lock:
            self._load()
            return len(self._rows)

    @property
    def status(self) -> str:
        return f"on ({self.size} passages learned from earlier checks)"

    def remember(self, passages: Iterable) -> int:
        """Store passages from trusted live sources that are not stored yet. Returns how many were added."""
        with self._lock:
            self._load()
            known = {r[0] for r in self._rows}
            new = []
            for p in passages:
                pid = str(getattr(p, "id", ""))
                if pid.startswith(LIVE_PREFIXES) and pid not in known and getattr(p, "text", ""):
                    known.add(pid)
                    new.append(p)
            if not new or len(self._rows) >= MAX_ITEMS:
                return 0
            vecs = self._encode([p.text for p in new])
            rows = [(str(p.id), p.text, p.source_url, p.domain, p.date) for p in new]
            if self.path is not None:
                try:
                    with self._db() as con:
                        con.executemany("INSERT OR IGNORE INTO memory VALUES (?, ?, ?, ?, ?, ?, ?)",
                                        [(*r, time.time(), v.tobytes()) for r, v in zip(rows, vecs)])
                except (OSError, sqlite3.Error):
                    self.path = None
            self._rows += rows
            self._vecs = vecs if self._vecs.size == 0 else np.vstack([self._vecs, vecs])
            self.added += len(rows)
            return len(rows)

    # search ------------------------------------------------------------------------------------
    def search(self, claim: str, before: str | None = None, k: int = CANDIDATES) -> list:
        """Stored passages most similar to the claim (cosine >= MIN_COSINE), dated on or before `before`."""
        from src.retrieval.search import Passage

        with self._lock:
            self._load()
            if not self._rows or not (claim or "").strip():
                return []
            rows, vecs = self._rows, self._vecs
        sims = vecs @ self._encode([claim])[0]
        out = []
        for i in np.argsort(-sims):
            if sims[i] < MIN_COSINE or len(out) >= k:
                break
            pid, text, url, dom, d = rows[i]
            if before and (not d or d > str(before)[:10]):  # no hindsight; undated cannot be shown earlier
                continue
            out.append(Passage(id=pid, text=text, source_url=url, domain=dom, date=d))
        return out

    def merge(self, claim: str, hits: Sequence, before: str | None = None, reranker=None, k: int = 5) -> list:
        """Hits + remembered passages (not already among the hits), re-ranked together, top `k`."""
        from src.retrieval.search import Hit

        have = {str(h.passage.id) for h in hits}
        mem = [p for p in self.search(claim, before) if str(p.id) not in have]
        if not mem:
            return list(hits)
        if reranker is not None:
            scores = list(map(float, reranker(claim, [p.text for p in mem])))
        else:
            best = max((h.score for h in hits), default=0.0)
            scores = [best] * len(mem)
        merged = list(hits) + [Hit(p, s, None, None) for p, s in zip(mem, scores)]
        merged.sort(key=lambda h: -h.score)
        return merged[:k]


def default_memory(encoder=None) -> EvidenceMemory | None:
    if os.environ.get("NEWSCRED_EVIDENCE_MEMORY", "1") == "0":
        return None
    return EvidenceMemory(encoder=encoder)
