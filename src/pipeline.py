"""Runs the full analysis for one news item (inference workflow).

    item -> 3A claims -> 3B evidence -> 3C verdicts
         -> 3D text-image -> 3E media reuse (rule-based, see below) -> 3F source -> 3G temporal
         -> feature vector (NaN = check not applicable) -> meta-learner -> score 0-100 + band
         -> Phase 5: points per check (SHAP) + per-claim justification
    (The LLM-written summary is added by the API via `src.explain.generator.Explainer`.)

Every channel is optional. A channel whose model is not trained yet, or whose
input is absent (no image, no text), is skipped and listed in `skipped_checks`
with the reason; its features stay NaN. The item is then scored on what is
available instead of being penalised.

Decisions baked in:
- 3E uses its rule-based fallback, not `models/media_reuse_clf.joblib`: the
  trained classifier scored below chance (cross-validated macro-F1 0.24,
  reports/media_reuse_metrics.json).
- Before the meta-learner is trained, the score is the equal-weight average of
  the available channel scores (evaluation baseline (b)), and `meta_model` says so.

Components load lazily on first use and are cached, so the API loads each model
once. Tests inject fakes through `components=`.
"""

from __future__ import annotations

import hashlib
import math
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from src.fusion import features as F
from src.fusion.meta import MetaModel, average_channel_scores, band, to_score

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
# Evidence index in use. Default: AVeriTeC train passages. NEWSCRED_RETRIEVAL_DIR=models/retrieval_index_india
# adds published Indian and other fact-checks (python -m src.retrieval.search --india).
RETRIEVAL_DIR = ROOT / os.environ.get("NEWSCRED_RETRIEVAL_DIR", "models/retrieval_index")  # absolute paths stay as given
MODEL_VERSION = "0.10.1"  # 0.10.1: separate evidence budgets (confirm-only passages no longer crowd out refutations), silence rule off again (its rule fails); 0.10.0: scored by the evidence checks only (no wording-based check), meta-learner refitted; 0.9.2: a claim the news would cover is searched twice before its silence is believed (the second search adds evidence), silence counted over every returned report (undated too, debunks not); 0.9.1: template headline says what decided the score (no scoring change); 0.9: silence is evidence for big claims (expected news coverage), Wikipedia must match the claim's time, news search (Tavily), evidence memory; 0.8.1: a date check with none of the item's own dates to compare is not evidence; 0.8: source = outlet rating (platforms, unrated: not scored), article URLs read; 0.7.1: a URL alone is not scored, location threshold 0.49 (calibrated); 0.7: claim matching (a verdict on another claim does not count); 0.6: abstain without evidence
# Trained files whose change changes the scores. Their size + modification time go into
# `model_fingerprint()`, so the API's result cache never serves a report made by an older model.
ARTIFACTS = (
    "meta_lgbm.txt", "meta_calibrator.joblib", "source_clf.joblib", "text_image_clf.joblib",
    "retrieval_index/passages.jsonl", "verifier-deberta/config.json", "verifier-deberta/model.safetensors",
    "checkworthy-deberta/config.json", "checkworthy-deberta/model.safetensors", "media_index/records.parquet",
)
MAX_CLAIMS = 5  # bounds latency: each claim costs one retrieval + up to 10 verifier passes
EVIDENCE_PER_CLAIM = 5  # passages that can refute (offline index, fact-checks) verified per claim
CONFIRM_PER_CLAIM = 5  # plus passages that may only confirm (Wikipedia, news): their own budget, see evidence_budget
_POOL = 1000  # merges keep every candidate; evidence_budget makes the cut
MAX_NEWS_CLAIMS = 2  # news searches per item (the most check-worthy claims): Tavily's free plan is 1,000 a month
ABSTAIN_BAND = "Unverified"  # band of a report with no score (no check found evidence)
_FAILED = object()  # cache marker: component tried and unavailable


def _abstain_reason(claims: list[dict], item: "NewsItem", text: str) -> str:
    """Why no score was given, in one sentence the user can act on."""
    from src.source.credibility import parse_domain

    if not text and item.image is None and parse_domain(item.source_url)[0] is not None:
        return ("Not scored: only a source was given. Its rating says how reliable the outlet is in general, "
                "not whether this item is true; add the text or image to check it.")
    if not text:
        why = "there is no text to check"
    elif not claims:
        why = "no checkable statement of fact was found (questions, opinions and jokes are not checked)"
    elif all(c.get("context") == "missing" for c in claims):
        why = "the text does not say who or what it is about"
    else:
        why = "no evidence about its claims was found"
    if item.image is None and parse_domain(item.source_url)[0] is None:
        why += ", and no image or source URL was given"
    return f"Not scored: {why}. A number here would only be a guess."


def _absence(claims: list[dict], row: dict) -> dict | None:
    """The absence rule's decision for one report, or None. Applies when some claim would certainly be
    reported if true, the news search for it ran and answered, no trusted report confirmed it, and no claim
    evidence counted at all (a refuting fact-check or a confirming passage already decides the claim)."""
    from src.retrieval.coverage import ABSENCE_LR, ABSENCE_LR_SOURCE, ABSENCE_MODE, is_silent

    if ABSENCE_LR is None or ABSENCE_LR >= 1:  # off unless NEWSCRED_ABSENCE_LR is set: src/retrieval/coverage.py
        return None
    claim_evidence = row.get("claim_evidence_score")
    if claim_evidence is not None and not math.isnan(claim_evidence):
        return None
    silent = [c for c in claims if is_silent(c.get("coverage"))]
    if not silent:
        return None
    c = silent[0]
    cv = c["coverage"]
    s = cv["search"] or {}
    where = f"{s['outlets']} trusted news sites" if s.get("outlets") else "trusted news sites"
    when = f"from {s['from']} to {s['to']}" if s.get("from") else f"up to {s.get('to')}"
    found = "no report about it" if ABSENCE_MODE == "no_relevant_report" else "no report confirming it"
    reason = (f"No trusted outlet reported this. A claim like this ({cv['why']}) would be widely reported if it "
              f"were true, but a search of {where} ({when}) found {found}. Silence counts as "
              "evidence here because reports would be expected; it is not proof (an event from the last few hours "
              "may not be indexed yet).")
    return {"applied": True, "claim": c["text"], "why": cv["why"], "event": cv.get("event"),
            "subject": cv.get("subject"), "search": s, "mode": ABSENCE_MODE, "likelihood_ratio": ABSENCE_LR,
            "likelihood_ratio_source": ABSENCE_LR_SOURCE, "reason": reason}


def _unheard_note(claims: list[dict]) -> str | None:
    """For an unscored big claim, say what the news search found: nothing relevant (shown, not scored: on real
    Indian claims silence was more common for true claims), or that the search did not run."""
    from src.retrieval.coverage import is_silent

    for c in claims:
        cv = c.get("coverage")
        if not cv or not cv["expected"]:
            continue
        if is_silent(cv, "no_relevant_report"):
            s = cv.get("search") or {}
            where = f"{s['outlets']} trusted news sites" if s.get("outlets") else "trusted news sites"
            return (f"No trusted outlet reported this: a search of {where} found no report about it, although a "
                    f"claim like this would normally be news ({cv['why']}). Treat it with suspicion: in our tests "
                    "fakes went unreported more often than real events. It is not scored on that alone, because the "
                    "search also found nothing for about 1 in 10 real big events.")
        state = cv.get("news_state")
        if not cv["searched"] and state in ("off", "no answer"):
            why = "the news search is off (no TAVILY_API_KEY)" if state == "off" else "the news search did not answer"
            return (f"This is the kind of claim trusted outlets would report if it were true ({cv['why']}), but "
                    f"{why}, so their silence could not be checked.")
    return None


@dataclass
class NewsItem:
    text: str | None = None
    image: Any = None  # PIL.Image.Image
    source_url: str | None = None  # article URL or bare domain (3F)
    claimed_date: str | None = None  # ISO; else taken from the text by 3G
    claimed_place: str | None = None  # 3E location check
    claimed_event: str | None = None  # 3E different-event flag
    publish_date: str | None = None
    has_video: bool = False
    tweet_ids: str | None = None  # FakeNewsNet social ids (optional 3F feature)
    media_id: str | None = None  # dataset image id; excluded from reuse matching when building training rows

    def input_hash(self) -> str:
        h = hashlib.sha256()
        for part in (self.text, self.source_url, self.claimed_date, self.claimed_place, self.publish_date):
            h.update(("" if part is None else str(part)).encode("utf-8") + b"\x1f")
        if self.image is not None:
            h.update(np.asarray(self.image.convert("RGB").resize((64, 64))).tobytes())
        return h.hexdigest()[:16]


def model_fingerprint(models_dir: Path = MODELS) -> str:
    """MODEL_VERSION + a short hash of the trained artifacts' sizes and modification
    times (files are not read), e.g. "0.5.0+3f9a1c2e". Retraining any model changes it."""
    h = hashlib.sha256()
    for name in ARTIFACTS:
        f = models_dir / name
        st = f.stat() if f.exists() else None
        h.update(f"{name}:{st.st_size if st else -1}:{st.st_mtime_ns if st else -1}".encode())
    if models_dir == MODELS and RETRIEVAL_DIR != MODELS / "retrieval_index":  # another evidence index: other results
        f = RETRIEVAL_DIR / "passages.jsonl"
        st = f.stat() if f.exists() else None
        h.update(f"index:{RETRIEVAL_DIR.name}:{st.st_size if st else -1}:{st.st_mtime_ns if st else -1}".encode())
    return f"{MODEL_VERSION}+{h.hexdigest()[:8]}"


def published_fact_checks(hits, min_relevance: float = 0.5, not_matching: set[str] | frozenset = frozenset()) -> list[dict]:
    """Published fact-checks among a claim's evidence that are about this claim (re-ranker relevance
    >= 0.5, and the claim they checked matches this one): the "previously fact-checked by ..." badge.
    Live (Google) or from the offline index. `not_matching`: passage ids the verifier left out because
    they judged a different claim."""
    from src.retrieval.factcheck_api import parse_factcheck_passage, relevance

    out = []
    for h in hits:
        info = parse_factcheck_passage(h.passage.text)
        if info and relevance(h.score) >= min_relevance and h.passage.id not in not_matching:
            out.append({"publisher": info["who"], "rating": info["rating"], "claim_reviewed": info["claim"],
                        "url": h.passage.source_url, "date": h.passage.date, "relevance": relevance(h.score)})
    return out


def _clean(x):
    """JSON-safe: NaN/inf -> None, numpy scalars -> Python, recursively."""
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not math.isfinite(float(x)) else float(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


class Pipeline:
    COMPONENTS = ("scorer", "searcher", "reranker", "verifier", "factcheck", "news", "wiki", "memory", "clip",
                  "text_image", "reuse", "source", "meta")

    def __init__(self, components: dict[str, Any] | None = None, autoload: bool = True):
        """components: pre-built objects by name (tests). autoload=False: anything not
        supplied is treated as unavailable instead of loaded from models/."""
        self._cache: dict[str, Any] = dict(components or {})
        self.autoload = autoload
        self.status: dict[str, str] = {k: "injected" for k in self._cache}
        # Reentrant: a loader may load another component (text_image and reuse load clip).
        # A plain Lock deadlocked here on the first item with an image.
        self._lock = threading.RLock()
        # Fixed for this process: models are loaded once and stay in memory.
        self.model_version = model_fingerprint()
        self._claims_memo: dict[tuple[str, str | None], list] = {}  # filled by _prefetch, cleared after each batch

    # --- component loading ----------------------------------------------------
    def get(self, name: str):
        with self._lock:
            if name not in self._cache:
                if not self.autoload:
                    self._cache[name], self.status[name] = _FAILED, "not loaded (autoload off)"
                else:
                    try:
                        obj, status = getattr(self, f"_load_{name}")()
                    except Exception as e:  # missing model file, blocked DLL, no GPU ...
                        obj, status = None, f"unavailable: {type(e).__name__}: {str(e)[:150]}"
                    self._cache[name] = _FAILED if obj is None else obj
                    self.status[name] = status
            obj = self._cache[name]
        return None if obj is _FAILED else obj

    def load_all(self) -> dict[str, str]:
        for name in self.COMPONENTS:
            self.get(name)
        return dict(self.status)

    def _load_scorer(self):
        from src.claims.extract import DebertaCheckWorthiness, default_scorer

        s = default_scorer()
        return s, "deberta" if isinstance(s, DebertaCheckWorthiness) else "heuristic fallback (DeBERTa not trained)"

    def _load_searcher(self):
        from src.retrieval.search import HybridSearcher

        if not (RETRIEVAL_DIR / "passages.jsonl").exists():
            return None, "unavailable: evidence index not built (python -m src.retrieval.search)"
        return HybridSearcher.load(RETRIEVAL_DIR), "ready"

    def _load_reranker(self):
        from src.retrieval.search import cross_encoder_reranker

        return cross_encoder_reranker(), "ready"

    def _load_verifier(self):
        from src.verify.model import DEFAULT_MODEL_DIR, ClaimVerifier

        if not (DEFAULT_MODEL_DIR / "config.json").exists():
            return None, "unavailable: verifier not trained (python -m src.verify.model)"
        return ClaimVerifier(), "ready"

    def _load_factcheck(self):
        from src.retrieval.factcheck_api import default_client

        client = default_client()
        if client is None:
            return None, "off (no GOOGLE_FACTCHECK_API_KEY, or NEWSCRED_LIVE_FACTCHECK=0)"
        return client, client.status

    def _load_news(self):
        from src.retrieval.news_search import default_client

        client = default_client()
        if client is None:
            return None, "off (no TAVILY_API_KEY in .env, or NEWSCRED_NEWS_SEARCH=0)"
        return client, client.status

    def _load_wiki(self):
        from src.retrieval.wiki_search import default_client

        client = default_client()
        if client is None:
            return None, "off (NEWSCRED_WIKI_SEARCH=0)"
        return client, client.status

    def _load_memory(self):
        from src.retrieval.evidence_memory import default_memory

        searcher = self.get("searcher")
        mem = default_memory(encoder=getattr(searcher, "encoder", None))
        if mem is None:
            return None, "off (NEWSCRED_EVIDENCE_MEMORY=0)"
        return mem, mem.status

    def _load_clip(self):
        from src.media.text_image import ClipEncoder

        return ClipEncoder(), "ready"

    def _load_text_image(self):
        from src.media.text_image import MODEL_PATH, TextImageChecker

        if not MODEL_PATH.exists():
            return None, "unavailable: text-image classifier not trained (python -m src.media.text_image)"
        clip = self.get("clip")
        if clip is None:
            return None, "unavailable: CLIP encoder failed to load"
        return TextImageChecker(encoder=clip), "ready"

    def _load_reuse(self):
        from src.media.reuse import INDEX_DIR, LocationReference, MediaIndex, MediaReuseChecker, load_index

        index = load_index() if (INDEX_DIR / "records.parquet").exists() else MediaIndex([])
        clip = self.get("clip")
        enc = clip.encode_images if clip is not None else None
        locations = LocationReference(enc) if enc is not None else None
        n = len(index.records)
        return MediaReuseChecker(index, encoder=enc, locations=locations, model=None), f"rule-based fallback ({n} indexed images)"

    def _load_source(self):
        from src.source.credibility import MODEL_PATH, SOURCE_MODE, SourceChecker

        if SOURCE_MODE != "classifier":
            return SourceChecker(), "ready (ratings of 1,082 outlets; platforms and unrated domains not scored)"
        return SourceChecker(), "ready" if MODEL_PATH.exists() else "rule-based fallback (model not trained)"

    def _load_meta(self):
        from src.fusion.meta import META_MODEL

        if not META_MODEL.exists():
            return None, "unavailable: meta-learner not trained; using equal-weight average"
        return MetaModel.load(), "ready"

    # --- analysis ---------------------------------------------------------------
    def analyze(self, item: NewsItem, channels: Iterable[str] | None = None) -> dict:
        """Full report as a JSON-safe dict. `channels` limits which checks run
        (used when building the meta-learner table); default all."""
        from src.temporal.consistency import check as temporal_check

        run = set(channels or F.CHANNELS)
        text = (item.text or "").strip()
        parts: list[dict] = []
        skipped: list[str] = []
        reasons: list[str] = []
        claims_out: list[dict] = []
        media: dict[str, Any] = {}
        channel_out: dict[str, dict] = {}
        evidence_dates: list[str] = []
        reuse_res = None
        places: list[str] = []  # places named in the claims: the location check uses them when none is given

        # 3A-3C: claims, evidence, verdicts
        coverage: dict[str, dict] = {}  # claim -> expected news coverage and what the news search found
        if "claims" in run:
            if not text:
                skipped.append("claim_verification: no text")
                parts.append(F.claim_features(None, None))
            else:
                from src.claims.extract import needs_context

                claims = self._claims(text, item.publish_date)
                # A claim that names nothing ("They said it was true.") cannot be verified: it is
                # reported but not searched, so evidence about other claims cannot decide it.
                vague = {c.text for c in claims if needs_context(c.text)}
                checkable = [c for c in claims if c.text not in vague]
                verifier, searcher = (self.get("verifier"), self.get("searcher")) if checkable else (None, None)
                places = [p for c in claims for p in getattr(c, "locations", [])]
                verdicts = None
                fact_checks: dict[str, list[dict]] = {}
                if checkable and verifier is not None and searcher is not None:
                    from src.retrieval import coverage as cov
                    from src.retrieval.factcheck_api import relevance
                    from src.retrieval.news_search import relevant_reports
                    from src.verify.model import evidence_budget

                    before = item.claimed_date or item.publish_date
                    live = self.get("factcheck")
                    news = self.get("news")
                    wiki = self.get("wiki")
                    memory = self.get("memory")
                    reranker = self.get("reranker")
                    verdicts = []
                    for n_claim, c in enumerate(checkable):
                        when = cov.claim_time(c.text, c.dates, item.claimed_date, item.publish_date)
                        hits = searcher.search(c.text, k=EVIDENCE_PER_CLAIM, before=before, reranker=reranker)
                        if live is not None:  # published fact-checks, re-ranked with the index passages
                            hits = live.merge(c.text, hits, before=before, reranker=reranker, k=EVIDENCE_PER_CLAIM)
                        # Would trusted outlets certainly have reported this if it were true? Decided before the
                        # news search: such a claim is searched twice before its silence is believed.
                        expect = cov.expected_coverage(c.text, c.entities,
                                                       notable=wiki.is_well_known if wiki is not None else None)
                        # Evidence that can *support* a claim (fact-checks almost only refute):
                        lookup = None
                        if news is not None and n_claim < MAX_NEWS_CLAIMS:  # reports by trusted outlets
                            start, end = cov.news_window(when, before)
                            if expect.expected and hasattr(news, "lookup_twice"):
                                lookup = news.lookup_twice(c.text, before=end, start=start,
                                                           strict_dates=before is not None, reranker=reranker)
                            else:
                                lookup = news.lookup(c.text, before=end, start=start, strict_dates=before is not None)
                            hits = news.merge(c.text, hits, before=end, reranker=reranker, k=_POOL,
                                              lookup=lookup)
                        if wiki is not None:  # Wikipedia as it stood on the claim date
                            hits = wiki.merge(c.text, hits, before=before, reranker=reranker, k=_POOL)
                        if memory is not None:  # passages trusted sources returned for earlier checks
                            hits = memory.merge(c.text, hits, before=before, reranker=reranker, k=_POOL)
                        # Separate budgets: a passage that may only confirm never takes the place of one that can
                        # refute (before 0.10.1 Wikipedia articles about a myth crowded out its refutations).
                        hits = evidence_budget(hits, EVIDENCE_PER_CLAIM, CONFIRM_PER_CLAIM)
                        # A passage about another time cannot confirm this claim (it may still refute).
                        no_support = cov.time_mismatch_ids(hits, when)
                        verdicts.append(verifier.verify(c.text, hits, no_support=no_support) if no_support
                                        else verifier.verify(c.text, hits))
                        other_claim = {e.passage_id for e in verdicts[-1].evidence if not getattr(e, "counted", True)}
                        fact_checks[c.text] = published_fact_checks(hits, not_matching=other_claim)
                        evidence_dates += [h.passage.date for h in hits if h.passage.date]
                        if memory is not None:  # learn: keep what trusted sources said about this topic
                            if memory.remember(h.passage for h in hits if reranker is None or relevance(h.score) >= 0.5):
                                self.status["memory"] = memory.status  # /health shows the current size
                        v = verdicts[-1]
                        coverage[c.text] = {
                            **expect.as_dict(),
                            "searched": bool(lookup is not None and lookup.ok),
                            "news_state": ("off" if news is None else "skipped (claim limit)" if lookup is None
                                           else "ran" if lookup.ok else "no answer"),
                            "search": None if lookup is None else {
                                "backend": lookup.backend, "ok": lookup.ok, "from": lookup.start, "to": lookup.end,
                                "results": lookup.n_results, "trusted_reports": len(lookup.passages),
                                "queries": getattr(lookup, "queries", 1),
                                "outlets": len(getattr(news, "domains", []) or []) or None},
                            "confirmed": v.label == "Supported" and not math.isnan(v.score),
                            # Over everything the news search returned (undated reports too), not the top-5
                            # evidence: silence is a property of the search (news_search.relevant_reports).
                            "relevant_reports": relevant_reports(c.text, lookup, reranker),
                            "time": None if not when.known else {"day": when.day.isoformat() if when.day else None,
                                                                   "current": when.current},
                            "not_confirming_other_time": len(no_support),
                        }
                elif checkable:
                    skipped.append("claim_verification: verifier or evidence index not available (claims extracted only)")
                elif claims:
                    skipped.append("claim_verification: the claims found do not say who or what they are about")
                else:
                    skipped.append("claim_verification: no check-worthy claim found")
                if vague:
                    reasons.append("Some sentences do not say who or what they are about, so they were not checked; "
                                   "add names, places or dates.")
                parts.append(F.claim_features(checkable, verdicts))
                verdict_of = dict(zip([c.text for c in checkable], verdicts or []))
                for c in claims:
                    v = verdict_of.get(c.text)
                    # Same rule as the score: a verdict on evidence about other claims does not count,
                    # so the card shows "Not Enough Evidence" instead of the verifier's raw label.
                    off_topic = (v is not None and not math.isnan(v.score) and not math.isnan(v.relevance)
                                 and v.relevance < F.RELEVANCE_MIN)
                    claims_out.append({
                        "text": c.text,
                        "checkworthiness": c.checkworthiness,
                        "verdict": ("Not Enough Evidence" if off_topic else v.label) if v else None,
                        "credibility": None if off_topic else (v.score if v else None),
                        "confidence": None if off_topic else (v.confidence if v else None),
                        "evidence_relevance": v.relevance if v else None,
                        "evidence_off_topic": off_topic,
                        "evidence": [
                            {"source": e.domain, "url": e.source_url, "snippet": e.text[:300], "weight": e.weight,
                             "date": e.date, "counted": getattr(e, "counted", True),
                             "claim_match": _clean(getattr(e, "claim_match", None))}
                            for e in (v.evidence[:2] if v else [])
                        ],
                        "fact_checks": fact_checks.get(c.text, []),
                        "entities": [list(e) for e in c.entities],
                        "dates": c.dates,
                        "context": "missing" if c.text in vague else None,
                        "coverage": coverage.get(c.text),
                    })
                channel_out["claims"] = {k: v for k, v in parts[-1].items()}

        # 3D: text-image consistency
        if "text_image" in run:
            checker = self.get("text_image") if (item.image is not None and text) else None
            if item.image is None:
                skipped.append("text_image_match: no image")
            elif not text:
                skipped.append("text_image_match: no caption text")
            elif checker is None:
                skipped.append(f"text_image_match: {self.status.get('text_image', 'unavailable')}")
            else:
                res = checker.score(text, item.image)
                parts.append(F.text_image_features(res))
                media["text_image_match"] = res.score
                media["text_image_subtype"] = res.subtype
                channel_out["text_image"] = {"score": res.score, "confidence": res.confidence, "subtype": res.subtype}
                if res.subtype:
                    reasons.append(f"Caption and image look inconsistent (likely '{res.subtype.replace('_', ' ')}').")

        # 3E: media reuse (rule-based)
        if "reuse" in run:
            if item.image is None:
                skipped.append("media_reuse: no image")
            else:
                reuse = self.get("reuse")
                if reuse is None:
                    skipped.append(f"media_reuse: {self.status.get('reuse', 'unavailable')}")
                else:
                    claimed = item.claimed_date or item.publish_date
                    # Location runs in the background: the place given, else the first place named in the
                    # text that has reference photos. It is reported only when the image does not match.
                    place = item.claimed_place or (reuse.locations.find(places) if reuse.locations is not None else None)
                    reuse_res = reuse.score(
                        item.image, claimed, item.claimed_event, place,
                        exclude_ids={item.media_id} if item.media_id else None,
                    )
                    parts.append(F.reuse_features(reuse_res))
                    sig = reuse_res.signals
                    media.update(
                        reuse_found=bool(sig and sig.n_matches > 0),
                        earliest_match_date=sig.earliest_match if sig else None,
                        location_match=sig.location_similarity if sig else None,
                        location_checked=place if sig and not math.isnan(sig.location_similarity) else None,
                        location_mismatch=bool(sig and sig.location_mismatch),
                    )
                    channel_out["reuse"] = {"score": reuse_res.score, "confidence": reuse_res.confidence}
                    reasons += sig.reasons if sig else []

        # 3F: source. Needs a URL/domain: on text alone the source model only sees the headline's
        # wording, which carried no signal in training (MediaEval tweets: ROC-AUC 0.52) yet was read
        # with the trust learned from items that had a domain (a true ISRO claim lost 20 points).
        if "source" in run:
            from src.source.credibility import parse_domain

            if parse_domain(item.source_url)[0] is None:
                skipped.append("source: no source URL or domain given")
            else:
                src = self.get("source")
                if src is None:
                    skipped.append(f"source: {self.status.get('source', 'unavailable')}")
                else:
                    res = src.score(item.source_url, text or None, tweet_ids=item.tweet_ids)
                    parts.append(F.source_features(res))
                    if math.isnan(res.score):  # platform or unrated domain: no source verdict, not counted
                        skipped.append("source: " + (res.reasons[0] if res.reasons else "not rated"))
                    else:
                        channel_out["source"] = {"score": res.score, "confidence": res.confidence}
                        reasons += res.reasons

        # 3G: temporal consistency (uses 3E's earliest copy and 3B's evidence dates when present)
        if "temporal" in run:
            exif_date = None
            if item.image is not None:
                from src.media.reuse import read_exif

                exif_date = read_exif(item.image)["capture_date"]
            earliest = reuse_res.signals.earliest_match if (reuse_res and reuse_res.signals) else None
            res = temporal_check(
                claimed_date=item.claimed_date, publish_date=item.publish_date, exif_date=exif_date,
                earliest_copy_date=earliest, evidence_dates=evidence_dates, claim_text=text or None,
            )
            if math.isnan(res.score):
                skipped.append("temporal: no claimed date found")
            else:
                parts.append(F.temporal_features(res))
                channel_out["temporal"] = {"score": res.score, "confidence": res.confidence, "flags": res.flags}
                reasons += res.reasons

        parts.append(F.modality_features(bool(text), item.image is not None, item.has_video))
        row = F.assemble(*parts)
        vec = np.array([F.to_vector(row)])

        meta = self.get("meta")
        if meta is not None:
            p, meta_name = float(meta.predict(vec)[0]), "lightgbm+isotonic"
        else:
            p, meta_name = float(average_channel_scores(vec)[0]), "equal-weight average (meta-learner not trained)"
        # Abstain when no check found evidence about this item: the model's output would only be its
        # prior for the input type (every gibberish text scored 60 before this rule).
        # A URL alone: the source rating is the only check, and it rates the outlet, not this item. The
        # training data has almost no items without text or image, so the model's output there is an
        # extrapolation (thehindu.com, rated 0.90, scored 3/100 before this rule).
        abstained = not F.has_evidence(row) or (not text and item.image is None)
        # Silence as evidence (0.9): a claim that trusted outlets would certainly report if true, a news search
        # that really ran, no trusted report confirming it, and no other claim evidence that counted. Bayes:
        # the model's odds (even odds if nothing else found evidence) x the likelihood ratio of that silence.
        absence = _absence(claims_out, row) if text else None
        p_model = None
        if absence is not None:
            from src.retrieval.coverage import absence_probability

            p_model = None if abstained else p
            p = absence_probability(p_model, absence["likelihood_ratio"])
            absence["model_score"] = None if p_model is None else to_score(p_model)
            abstained = False
            reasons.insert(0, absence["reason"])
        elif abstained:
            p = math.nan
            reasons.insert(0, _abstain_reason(claims_out, item, text))
            if (note := _unheard_note(claims_out)) is not None:
                reasons.insert(1, note)
        score = to_score(p)

        # Phase 5: points per check (SHAP for LightGBM, exact split for the average) + claim justifications
        from src.explain.generator import claim_justification, contributions, evidence_box

        if absence is not None and p_model is None:  # nothing else spoke: even odds, then the silence
            baseline = 50.0
            contribs = [{"check": "missing_coverage", "points": round(100 * p - 50, 1), "applicable": True}]
        else:
            baseline, contribs = contributions(meta, vec, row) if score is not None else (math.nan, [])
            if absence is not None:
                contribs.append({"check": "missing_coverage", "points": round(100 * (p - p_model), 1),
                                 "applicable": True})
        for c in claims_out:
            c["justification"] = claim_justification(c)
        return _clean({
            "analysis_id": item.input_hash(),
            "score": score,
            "probability": p,
            "band": ABSTAIN_BAND if abstained else band(score),
            "abstained": abstained,
            "claims": claims_out,
            "evidence_box": evidence_box(claims_out),
            "media": media,
            "channels": channel_out,
            "features": row,
            "baseline_score": baseline,  # score of an average item; baseline + sum(points) = score
            "contributions": contribs,
            "skipped_checks": skipped,
            "reasons": reasons,
            "coverage": absence,  # the absence rule, when it decided or moved the score
            "meta_model": meta_name,
            "model_version": self.model_version,
        })


    # --- batch analysis ------------------------------------------------------------
    def _claims(self, text: str, publish_date: str | None) -> list:
        """Top check-worthy claims; reuses the batch prefetch's extraction (spaCy + NER) when present."""
        key = (text, publish_date)
        if key not in self._claims_memo:
            from src.claims.extract import extract_claims
            from src.temporal.consistency import to_date

            claims = extract_claims(text, to_date(publish_date), scorer=self.get("scorer"))[:MAX_CLAIMS]
            if not self._batch_active:
                return claims
            self._claims_memo[key] = claims
        return self._claims_memo[key]

    _batch_active = False

    def clear_memos(self) -> None:
        self._claims_memo.clear()
        for obj in self._cache.values():
            if obj is not _FAILED and hasattr(obj, "clear_memo"):
                obj.clear_memo()

    def _prefetch(self, items: list[NewsItem], run: set[str]) -> None:
        """Run every model call the batch will need, in large batches, filling the
        component caches. Order follows the data dependencies:
        sentences -> claims -> query vectors -> re-ranker pairs -> verifier pairs; CLIP."""
        from src.claims.extract import needs_context, split_sentences
        from src.preprocess.clean_text import clean_text

        texts = [(it.text or "").strip() for it in items]
        if "claims" in run and any(texts):
            scorer = self.get("scorer")
            if hasattr(scorer, "prefetch"):
                scorer.prefetch([s.text for t in texts if t for s in split_sentences(clean_text(t))])
            jobs = []  # (claim text, before) exactly as analyze() will query them
            for it, t in zip(items, texts):
                if t:
                    claims = self._claims(t, it.publish_date)
                    jobs += [(c.text, it.claimed_date or it.publish_date) for c in claims if not needs_context(c.text)]
            verifier = self.get("verifier") if jobs else None
            searcher = self.get("searcher") if jobs else None
            if verifier is not None and searcher is not None:
                reranker = self.get("reranker")
                searcher.prefetch_queries([q for q, _ in jobs])
                if hasattr(reranker, "prefetch"):
                    from src.retrieval.search import CANDIDATES

                    pairs = []
                    for q, before in jobs:
                        order = searcher.candidates(q, CANDIDATES, before)[0][:CANDIDATES]
                        pairs += [(q, searcher.passages[i].text) for i in order]
                    reranker.prefetch(pairs)
                if hasattr(verifier, "prefetch"):
                    vpairs = []
                    for q, before in jobs:
                        hits = searcher.search(q, k=EVIDENCE_PER_CLAIM, before=before, reranker=reranker)
                        vpairs += [(q, h.passage.text) for h in hits]
                    verifier.prefetch(vpairs)

        with_image = [it for it in items if it.image is not None]
        if with_image and ({"text_image", "reuse"} & run):
            needs_text = "text_image" in run and self.get("text_image") is not None
            needs_img = needs_text or ("reuse" in run and self.get("reuse") is not None)
            clip = self.get("clip") if needs_img else None
            if clip is not None and hasattr(clip, "prefetch_images"):
                clip.prefetch_images([it.image for it in with_image])
                if needs_text:
                    clip.prefetch_texts([t for it, t in zip(items, texts) if it.image is not None and t])

    def analyze_batch(self, items: list[NewsItem], channels: Iterable[str] | None = None) -> list[dict]:
        """Same reports as [analyze(i) for i in items], but the heavy model calls run in
        shared batches first (prefetch), then each item is analysed by the unchanged
        per-item code, which reads the cached outputs. Features therefore come from the
        same code path as the API (no train/serve skew). Caches are cleared after the batch."""
        run = set(channels or F.CHANNELS)
        self._batch_active = True
        try:
            self._prefetch(items, run)
            return [self.analyze(it, channels=run) for it in items]
        finally:
            self._batch_active = False
            self.clear_memos()


_default: Pipeline | None = None


def analyze(item: NewsItem) -> dict:
    """Module-level entry point: `pipeline.analyze(item)`."""
    global _default
    if _default is None:
        _default = Pipeline()
    return _default.analyze(item)
