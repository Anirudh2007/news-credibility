"""Tune the India fixes on the development set, then score the test set once (2026-11-03).

Problem being fixed: on the 2023-2025 India test set, 24% of true claims were rated Low (0-30) and only
8% Credible. Causes found: (1) fact-checks and AVeriTeC answers about a
*different* claim on the same topic were read as refutations; (2) no evidence source could support a
true claim (fact-checkers publish almost only "False").

Method (no test-set tuning):
    record   run the full pipeline once on a set (Indian index, live fact-check lookup, optional news
             search); for every claim keep its hits, per-passage verifier probabilities, the claim-match
             similarity of each passage, and the item's other features.
    replay   for each setting (claim-match threshold) rebuild the verdicts with the
             production function `verdict_from`, the claim features with `claim_features`, and the score
             with the meta-learner, exactly as the pipeline would.
    choose   on the dev set: among settings with <= 5% of true claims rated Low and <= 5% of fakes rated
             Credible, the one that rates the most fakes Low.
    test     the chosen setting, once, on the test set.

    python -m src.fusion.india_tune record --set dev  [--sources wiki news]
    python -m src.fusion.india_tune augment --set dev --sources wiki   # same result as record --sources wiki,
                                                                      # reusing dev.pkl (low memory)
    python -m src.fusion.india_tune choose            [--sources ...]
    python -m src.fusion.india_tune record --set test [--sources ...]
    python -m src.fusion.india_tune report            [--sources ...]   -> reports/india_fix_eval[_sources].json

Silence as evidence (2026-11-06): news search on top of the Wikipedia recordings, then the absence rule's
likelihood ratio measured on dev and applied once to test; the time rule alone needs no network.
    python -m src.fusion.india_tune augment --set dev --sources news --on wiki --only-expected   # -> dev_news_wiki.pkl
    python -m src.fusion.india_tune augment --set test --sources news --on wiki --only-expected
    python -m src.fusion.india_tune absence --sources news wiki   -> reports/absence_eval_news_wiki.json
    python -m src.fusion.india_tune time-rule                     -> reports/time_rule_eval.json
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.fusion import features as F

ROOT = Path(__file__).resolve().parents[2]
INDIA = ROOT / "data" / "processed" / "india"
CACHE = ROOT / "data" / "processed" / "india" / "tune"
REPORT = ROOT / "reports" / "india_fix_eval.json"
SETS = {"dev": INDIA / "live_devset.parquet", "test": INDIA / "live_testset.parquet",
        "matched": INDIA / "matched_testset.parquet",  # matched: both halves written by fact-checkers (test only)
        "hoax": INDIA / "hoax_set.parquet"}  # invented-event hoaxes, fakes only, split dev / test (2026-11-07)
OWN_CLAIM = 0.95  # a verdict passage this similar to the item is (almost) the item's own fact-check: leakage check
OFF = -1.0  # claim matching off: no cosine is below -1 (the pipeline as it was before 2026-11-03)
THRESHOLDS = [OFF, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
MAX_TRUE_LOW = 0.05  # target: at most 5% of true claims rated Low (the user's goal was < 10%)
MAX_FAKE_CREDIBLE = 0.05  # and at most 5% of fakes rated Credible


def load_set(name: str) -> pd.DataFrame:
    """date = claim date, else the day before the review (no hindsight), as in evaluate.live_items."""
    df = pd.read_parquet(SETS[name])
    review = pd.to_datetime(df["review_date"], errors="coerce") - pd.Timedelta(days=1)
    df["date"] = df["claim_date"].fillna(review.dt.strftime("%Y-%m-%d"))
    return df[df["credible"].isin([0.0, 1.0])].reset_index(drop=True)


# --- record ----------------------------------------------------------------------------------

def record(name: str, sources: list[str], batch_size: int = 16) -> Path:
    """sources: supporting-evidence lookups to switch on ("news", "wiki"); the live fact-check lookup is
    always on (it is in the app)."""
    from src.fusion.train import release_memory
    from src.pipeline import NewsItem, Pipeline
    from src.retrieval.factcheck_api import default_client
    from src.retrieval.search import HybridSearcher
    from src.verify.model import reliability_of

    df = load_set(name)
    components = {"searcher": HybridSearcher.load(ROOT / "models" / "retrieval_index_india"),
                  "factcheck": default_client(), "memory": None}
    # Patient clients (retries): wait out rate limits, so no item silently counts as "nothing found".
    if "news" in sources:
        from src.retrieval.news_search import NewsSearchClient

        components["news"] = NewsSearchClient(retries=5)
    else:
        components["news"] = None
    if "wiki" in sources:
        from src.retrieval.wiki_search import WikiSearchClient

        components["wiki"] = WikiSearchClient(retries=6)
    else:
        components["wiki"] = None
    pipe = Pipeline(components=components)
    verifier = pipe.get("verifier")
    per_item: list[list[dict]] = []  # one list of verify calls per analyze() call, in order
    original_verify, original_analyze = verifier.verify, pipe.analyze
    from src.verify.model import ClaimMatcher, claim_behind

    cosine = ClaimMatcher(model=None)  # recordings keep the plain cosine, whatever matcher is deployed

    def recording_verify(claim, hits, **kw):
        v = original_verify(claim, hits, **kw)
        probs = verifier.predict(claim, [h.passage.text for h in hits]) if hits else np.zeros((0, 4))
        per_item[-1].append({"claim": claim, "hits": list(hits), "probs": probs,
                             "matches": cosine.similarity(claim, [claim_behind(h.passage) for h in hits]),
                             "rel": [reliability_of(h.passage.domain, verifier.reliability) for h in hits]})
        return v

    def recording_analyze(item, channels=None):
        per_item.append([])
        return original_analyze(item, channels=channels)

    verifier.verify = recording_verify
    pipe.analyze = recording_analyze  # analyze_batch calls self.analyze: the instance attribute wins
    items = [NewsItem(text=t, publish_date=d if isinstance(d, str) else None) for t, d in zip(df["text"], df["date"])]
    out = []
    for i in range(0, len(items), batch_size):
        start = len(per_item)
        reports = pipe.analyze_batch(items[i:i + batch_size])
        for rep, calls in zip(reports, per_item[start:]):
            out.append({"features": rep["features"], "abstained": rep["abstained"], "score": rep["score"],
                        "claims": [{"text": c["text"], "checkworthiness": c["checkworthiness"],
                                    "context": c.get("context")} for c in rep["claims"]],
                        "calls": calls})
        print(f"  {min(i + batch_size, len(items))}/{len(items)}", flush=True)
    release_memory()
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{name}{suffix_of(sources)}.pkl"
    queries = {s: {"calls": components[s].calls, "cache_hits": components[s].cache_hits,
                   "failed": components[s].failed_queries} for s in sources}
    with open(path, "wb") as f:
        pickle.dump({"labels": df["credible"].tolist(), "origin": df.get("origin", pd.Series([None] * len(df))).tolist(),
                     "items": out, "sources": sorted(sources), "queries": queries}, f)
    print("queries:", queries)
    return path


def suffix_of(sources) -> str:
    return "".join(f"_{s}" for s in sorted(sources or []))


def augment(name: str, source: str = "wiki", k: int = 5, on: list[str] | None = None, only_expected: bool = False,
            twice: bool = False) -> Path:
    """Add a supporting-evidence source to an existing recording without re-running the pipeline (low memory:
    only the re-ranker and verifier load). For each recorded claim this does exactly what the pipeline does
    after the live lookup: `client.merge(claim, hits, before, reranker, k)`, then the verifier on the merged
    hits. Recorded hits keep their recorded probabilities and claim-match values; new passages get theirs here.
    on: sources already in the base recording (augment --set dev --sources news --on wiki: dev_wiki.pkl ->
    dev_news_wiki.pkl). As in the pipeline since 0.10.1, the merged passages go through `evidence_budget`: the
    recorded top-k that can refute stays whole, and the best k confirm-only passages are added on top (the result
    does not depend on the order the sources are added in).
    News: as in the pipeline, only the first MAX_NEWS_CLAIMS claims are searched, in the claim's window
    (src/retrieval/coverage.py `news_window`), and each call records whether the search ran (`news_ok`).
    only_expected: search news only for items with a claim the news would cover (enough to measure the absence
    rule within Tavily's free 1,000 searches a month; other items keep their recording, news_ok None).
    twice: a claim the news would cover is searched a second time with its keywords when the first search heard
    no relevant report (`NewsSearchClient.lookup_twice`, as the pipeline since 2026-11-07)."""
    from src.pipeline import _POOL as POOL
    from src.pipeline import CONFIRM_PER_CLAIM, MAX_NEWS_CLAIMS
    from src.preprocess.clean_text import extract_dates
    from src.retrieval.coverage import claim_time, news_window
    from src.retrieval.search import cross_encoder_reranker
    from src.verify.model import ClaimVerifier, evidence_budget, reliability_of

    if source == "wiki":
        from src.retrieval.wiki_search import WikiSearchClient

        client = WikiSearchClient(retries=6)
    else:
        from src.retrieval.news_search import NewsSearchClient, default_client

        client = default_client() or NewsSearchClient(backend="gdelt")
        client.retries = 3 if client.backend == "tavily" else 5
        print("news backend:", client.status, flush=True)
    rec = _load(f"{name}{suffix_of(on)}")
    dates = load_set(name)["date"].tolist()
    expected = None
    expected_claim = None
    if source == "news" and (only_expected or twice):
        from src.claims.extract import tag_entities
        from src.retrieval.coverage import expected_coverage
        from src.retrieval.wiki_search import WikiSearchClient

        wk = WikiSearchClient(retries=6)

        def expected_claim(text: str) -> bool:
            return expected_coverage(text, tag_entities(text), wk.is_well_known).expected

        if only_expected:
            expected = [any(expected_claim(c["claim"]) for c in it["calls"]) for it in rec["items"]]
            print(f"items with a claim the news would cover: {sum(expected)}/{len(expected)}", flush=True)
    reranker = cross_encoder_reranker()
    verifier = ClaimVerifier(matcher=None, match_min=-1.0)  # matching of new passages: they are not verdicts
    items = []
    for n, (it, d) in enumerate(zip(rec["items"], dates), 1):
        before = d if isinstance(d, str) else None
        calls = []
        for j, c in enumerate(it["calls"]):
            old = {id(h): (p, m) for h, p, m in zip(c["hits"], c["probs"], c["matches"])}
            extra = {}
            if source == "news":
                if j < MAX_NEWS_CLAIMS and (expected is None or expected[n - 1]):
                    start, end = news_window(claim_time(c["claim"], extract_dates(c["claim"]), None, before), before)
                    if twice and expected_claim(c["claim"]):  # as the pipeline: search a covered claim twice
                        lk = client.lookup_twice(c["claim"], before=end, start=start, strict_dates=before is not None,
                                                 reranker=reranker)
                    else:
                        lk = client.lookup(c["claim"], before=end, start=start, strict_dates=before is not None)
                    hits = evidence_budget(client.merge(c["claim"], c["hits"], before=end, reranker=reranker,
                                                        k=POOL, lookup=lk), k, CONFIRM_PER_CLAIM)
                    extra = {"news_ok": lk.ok, "news_found": len(lk.passages), "news_queries": lk.queries}
                else:
                    hits, extra = list(c["hits"]), {"news_ok": None, "news_found": 0}
            else:
                hits = evidence_budget(client.merge(c["claim"], c["hits"], before=before, reranker=reranker, k=POOL),
                                       k, CONFIRM_PER_CLAIM)
            new = [h for h in hits if id(h) not in old]
            new_probs = verifier.predict(c["claim"], [h.passage.text for h in new]) if new else np.zeros((0, 4))
            fresh = {id(h): (p, np.nan) for h, p in zip(new, new_probs)}
            pm = [old.get(id(h)) or fresh[id(h)] for h in hits]
            calls.append({**c, "hits": hits, "probs": np.array([p for p, _ in pm]).reshape(-1, 4),
                          "matches": np.array([m for _, m in pm], dtype=float),
                          "rel": [reliability_of(h.passage.domain, verifier.reliability) for h in hits], **extra})
        items.append({**it, "calls": calls})
        if n % 50 == 0:
            print(f"  {n}/{len(dates)}  {source} calls {client.calls} cache hits {client.cache_hits}", flush=True)
    sources = sorted(set(rec.get("sources", [])) | {source})
    path = CACHE / f"{name}{suffix_of(sources)}.pkl"
    with open(path, "wb") as f:
        pickle.dump({**rec, "items": items, "sources": sources,
                     "queries": {source: {"calls": client.calls, "cache_hits": client.cache_hits,
                                          "failed": client.failed_queries}}}, f)
    if source == "news":
        searched = [c for it in items for c in it["calls"] if c.get("news_ok") is not None]
        print(f"{source}: {client.calls} searches spent, {client.cache_hits} from the cache, {client.failed_queries} "
              f"failed | claims searched {len(searched)}, answered {sum(c['news_ok'] for c in searched)}, searched "
              f"twice {sum(c.get('news_queries') == 2 for c in searched)}", flush=True)
    return path


# --- replay ----------------------------------------------------------------------------------

def attach_dates(rec: dict, name: str) -> dict:
    """The record with each item's claim date (the set's `date` column), needed by the time rule."""
    dates = load_set(name)["date"].tolist()
    assert len(dates) == len(rec["items"]), "recording and set differ in length"
    return {**rec, "items": [{**it, "date": d if isinstance(d, str) else None} for it, d in zip(rec["items"], dates)]}


def no_support_ids(claim_text: str, hits, item_date: str | None) -> set[str]:
    """The pipeline's time rule (src/retrieval/coverage.py) for a recorded claim: the claim's dates are in its
    text (3A rewrote "yesterday" to an ISO date using the item date)."""
    from src.preprocess.clean_text import extract_dates
    from src.retrieval.coverage import claim_time, time_mismatch_ids

    return time_mismatch_ids(hits, claim_time(claim_text, extract_dates(claim_text), None, item_date))


def replay_rows(rec: dict, match_min: float, support_only: bool = True,
                time_rule: bool = False, support_prefixes: tuple | None = None,
                entail_mode: str | None = None, entail_min: float | None = None) -> tuple[list[dict], list[list]]:
    """Feature rows and per-claim verdicts for every item under one setting, as the pipeline builds them.
    entail_mode / entail_min: the entailment gate between claims (src/verify/model.py); default: as deployed.
    A mode other than "off" needs the recording's entailment values (`with_entail`)."""
    from src.verify import model as vm
    from src.verify.model import SUPPORT_ONLY_PREFIXES, verdict_from

    entail_mode = vm.ENTAIL_MODE if entail_mode is None else entail_mode
    entail_min = vm.ENTAIL_MIN if entail_min is None else entail_min
    if entail_mode != "off" and any("entail" not in c for it in rec["items"] for c in it["calls"]):
        raise ValueError("this recording has no entailment values: python -m src.fusion.india_tune root-cause-a --stage entail")
    rows, all_verdicts = [], []
    for it in rec["items"]:
        row = {k: (math.nan if v is None else v) for k, v in it["features"].items()}
        checkable = [SimpleNamespace(checkworthiness=c["checkworthiness"]) for c in it["claims"]
                     if c.get("context") != "missing"]
        verdicts = []
        if it["calls"] and len(it["calls"]) == len(checkable):
            prefixes = support_prefixes if support_prefixes is not None else (SUPPORT_ONLY_PREFIXES if support_only else ())
            verdicts = [verdict_from(c["hits"], c["probs"], c["matches"], c["rel"], match_min, prefixes,
                                     no_support=no_support_ids(c["claim"], c["hits"], it.get("date")) if time_rule
                                     else frozenset(), entail=c.get("entail"), entail_mode=entail_mode,
                                     entail_min=entail_min)
                        for c in it["calls"]]
            row.update(F.claim_features(checkable, verdicts))
        rows.append(row)
        all_verdicts.append(verdicts)
    return rows, all_verdicts


def replay(rec: dict, match_min: float, model, support_only: bool = True,
           time_rule: bool = False, support_prefixes: tuple | None = None,
           entail_mode: str | None = None, entail_min: float | None = None) -> np.ndarray:
    """Scores 0-100 (NaN = no score) for every item under one setting. support_only=False: Wikipedia and
    news passages may also refute (the ablation of the corroboration-only rule). time_rule: a passage about
    another time cannot confirm (needs `attach_dates`)."""
    from src.fusion.meta import to_score

    rows, _ = replay_rows(rec, match_min, support_only, time_rule, support_prefixes, entail_mode, entail_min)
    X = np.array([F.to_vector(r) for r in rows])
    p = model.predict(X)
    return np.array([math.nan if not F.has_evidence(r) else (to_score(float(pi)) or math.nan)
                     for r, pi in zip(rows, p)], dtype=float)


def own_claim_share(rec: dict) -> dict:
    """Share of items (per label) whose evidence held a verdict on (almost) the same text: the claim's own
    fact-check reached the evidence. Should be ~0; reported next to every result."""
    y = np.array(rec["labels"], dtype=float)
    hit = np.array([any(np.nanmax(np.r_[c["matches"], -1]) >= OWN_CLAIM for c in it["calls"]) for it in rec["items"]])
    return {"true": float(hit[y == 1].mean()), "fake": float(hit[y == 0].mean())}


def drop_same_claim(rec: dict) -> dict:
    """The record with every verdict passage on (almost) the same text removed: how the system does when the
    claim was never fact-checked before (the strict view of 'previously fact-checked' evidence)."""
    def keep(m):
        return np.isnan(m) or m < OWN_CLAIM

    items = []
    for it in rec["items"]:
        calls = []
        for c in it["calls"]:
            k = [keep(m) for m in c["matches"]]
            calls.append({**c, "hits": [h for h, ok in zip(c["hits"], k) if ok],
                          "probs": np.array([p for p, ok in zip(c["probs"], k) if ok]).reshape(-1, 4),
                          "matches": np.array([m for m, ok in zip(c["matches"], k) if ok], dtype=float),
                          "rel": [r for r, ok in zip(c["rel"], k) if ok],
                          **({"entail": np.array([e for e, ok in zip(c["entail"], k) if ok], dtype=float).reshape(-1, 2)}
                             if "entail" in c else {})})
        items.append({**it, "calls": calls})
    return {**rec, "items": items}


def bands(y: np.ndarray, s: np.ndarray) -> dict:
    out = {}
    for lab, name in ((1.0, "true"), (0.0, "fake")):
        m = y == lab
        sm = s[m]
        out[name] = {"n": int(m.sum()), "no_score": float(np.isnan(sm).mean()), "low": float(np.mean(sm <= 30)),
                     "questionable": float(np.mean((sm > 30) & (sm <= 60))), "credible": float(np.mean(sm > 60))}
    ok = ~np.isnan(s)
    if len(set(y[ok])) > 1:
        from sklearn.metrics import roc_auc_score

        out["roc_auc_scored"] = float(roc_auc_score(y[ok], s[ok]))
        out["scored_share"] = float(ok.mean())
    return out


def models():
    """The deployed meta-learner, by name (the reports keep one entry per meta-learner)."""
    from src.fusion.meta import MetaModel

    return {"deployed": MetaModel.load()}


# --- learned claim matcher (second version, 2026-11-03) --------------------------------------
# Proxy label on the dev set: a verdict passage read as refuting (P(Refuted) >= 0.5) is about a DIFFERENT
# claim when it was retrieved for a true item (a true claim cannot really be refuted), and mostly the same
# rumour when retrieved for a fake item. A logistic regression on simple features (embedding cosine,
# shared words, length ratio, conflicting numbers, missing names) learns the difference.

def matcher_rows(rec: dict) -> dict:
    from src.verify.model import R as REFUTED
    from src.verify.model import claim_behind, match_features

    keys, feats, train, y, groups = [], [], [], [], []
    for i, (lab, it) in enumerate(zip(rec["labels"], rec["items"])):
        for j, c in enumerate(it["calls"]):
            for k, (h, p, m) in enumerate(zip(c["hits"], c["probs"], c["matches"])):
                other = claim_behind(h.passage)
                if np.isnan(m) or not other:
                    continue
                keys.append((i, j, k))
                feats.append(match_features(c["claim"], other, m))
                train.append(p[REFUTED] >= 0.5)
                y.append(int(lab == 0.0))
                groups.append(i)
    return {"keys": keys, "X": np.array(feats), "train": np.array(train), "y": np.array(y), "groups": np.array(groups)}


def fit_matcher(rec: dict, n_splits: int = 5) -> tuple[dict, dict]:
    """(model, out-of-fold probabilities by passage key). Folds are grouped by item, so no passage is scored by
    a model that saw any passage of its own item; the dev replay uses these OOF values, the test set the model
    fitted on all dev rows."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    from src.verify.model import MATCH_FEATURES

    d = matcher_rows(rec)
    X, y, tr, g = d["X"], d["y"], d["train"], d["groups"]
    oof = np.full(len(X), np.nan)
    for fit_idx, held_idx in GroupKFold(n_splits).split(X, y, g):
        f = fit_idx[tr[fit_idx]]
        lr = LogisticRegression(max_iter=2000).fit(X[f], y[f])
        oof[held_idx] = lr.predict_proba(X[held_idx])[:, 1]
    lr = LogisticRegression(max_iter=2000).fit(X[tr], y[tr])
    model = {"features": MATCH_FEATURES, "coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0]),
             "trained_on": "India dev set (live_devset), refuting verdict passages; label = retrieved for a fake item",
             "n_train": int(tr.sum()), "cv_auc": float(roc_auc_score(y[tr], oof[tr])),
             "cosine_auc": float(roc_auc_score(y[tr], X[tr, 0]))}
    return model, dict(zip(d["keys"], oof))


def rematch(rec: dict, probs_by_key: dict | None = None, model: dict | None = None) -> dict:
    """A copy of `rec` whose claim-match values are the learned matcher's probabilities (OOF values when given,
    else computed with `model`). Passages that are not verdicts keep NaN."""
    from src.verify.model import claim_behind, match_features, match_probability

    items = []
    for i, it in enumerate(rec["items"]):
        calls = []
        for j, c in enumerate(it["calls"]):
            m = np.array(c["matches"], dtype=float)
            for k, h in enumerate(c["hits"]):
                if np.isnan(m[k]):
                    continue
                if probs_by_key is not None:
                    m[k] = probs_by_key[(i, j, k)]
                else:
                    other = claim_behind(h.passage)
                    m[k] = match_probability(model, match_features(c["claim"], other, m[k])) if other else np.nan
            calls.append({**c, "matches": m})
        items.append({**it, "calls": calls})
    return {**rec, "items": items}


def grid(variants: dict[str, dict]) -> list[dict]:
    """variants: matcher name -> record whose `matches` hold that matcher's values."""
    out = []
    ms = models()
    for matcher, rec in variants.items():
        y = np.array(rec["labels"], dtype=float)
        for mname, m in ms.items():
            for t in THRESHOLDS:
                out.append({"matcher": matcher, "meta": mname, "match_min": t, **bands(y, replay(rec, t, m))})
    return out


def choose(results: list[dict]) -> dict:
    """Among settings with true claims rated Low <= MAX_TRUE_LOW and fakes rated Credible <= MAX_FAKE_CREDIBLE,
    the one that rates the most fakes Low (then: most true claims Credible). A higher threshold than needed
    only throws away fake detection, so the rule stops at the first one that meets the error targets."""
    ok = [r for r in results if r["true"]["low"] <= MAX_TRUE_LOW and r["fake"]["credible"] <= MAX_FAKE_CREDIBLE]
    if not ok:  # no setting meets the targets: the fewest false alarms
        return min(results, key=lambda r: (r["true"]["low"], -r["fake"]["low"]))
    return max(ok, key=lambda r: (round(r["fake"]["low"], 3), round(r["true"]["credible"], 3), -r["match_min"]))


# --- absence rule: is silence evidence? (2026-11-06) -------------------------------------------------------

def absence_items(rec: dict, match_min: float, model, wiki=None,
                  mode: str = "no_relevant_report") -> list[dict]:
    """Per item, exactly as the pipeline decides: the model score (time rule on), and whether the absence rule is
    eligible (some claim expected to be covered and its news search ran) and fires (eligible, no trusted report
    confirmed it, and no claim evidence counted at all)."""
    from src.claims.extract import tag_entities
    from src.fusion.meta import to_score
    from src.retrieval.coverage import RELEVANT_REPORT, expected_coverage, is_silent
    from src.retrieval.factcheck_api import relevance

    rows, verdicts = replay_rows(rec, match_min, support_only=True, time_rule=True)
    p = model.predict(np.array([F.to_vector(r) for r in rows]))
    out = []
    for it, row, vs, pi in zip(rec["items"], rows, verdicts, p):
        claims = []
        for c, v in zip(it["calls"], vs):
            exp = expected_coverage(c["claim"], tag_entities(c["claim"]),
                                    notable=wiki.is_well_known if wiki is not None else None)
            claims.append({"expected": exp.expected, "event": exp.event, "searched": c.get("news_ok") is True,
                           "confirmed": v.label == "Supported" and not math.isnan(v.score),
                           "relevant_reports": sum(1 for h in c["hits"] if str(h.passage.id).startswith("news:")
                                                   and relevance(h.score) >= RELEVANT_REPORT)})
        no_claim_evidence = math.isnan(row.get("claim_evidence_score", math.nan))
        eligible = any(c["expected"] and c["searched"] for c in claims)
        fires = no_claim_evidence and any(is_silent(c, mode) for c in claims)
        score = to_score(float(pi)) if F.has_evidence(row) else None
        out.append({"score": score, "eligible": eligible, "fires": fires,
                    "expected": any(c["expected"] for c in claims), "events": [c["event"] for c in claims if c["expected"]]})
    return out


def absence_counts(y: np.ndarray, items: list[dict]) -> dict:
    out = {}
    for lab, name in ((1.0, "true"), (0.0, "fake")):
        m = [it for it, yy in zip(items, y) if yy == lab]
        out[name] = {"n": len(m), "expected_coverage": sum(it["expected"] for it in m),
                     "eligible": sum(it["eligible"] for it in m), "fires": sum(it["fires"] for it in m)}
    return out


def measured_lr(counts: dict) -> float:
    """LR of "the rule fires" among eligible items: P(fires | true) / P(fires | fake), add-one smoothed."""
    from src.retrieval.coverage import likelihood_ratio

    t, f = counts["true"], counts["fake"]
    return likelihood_ratio(t["eligible"] - t["fires"], t["eligible"], f["eligible"] - f["fires"], f["eligible"])


def bootstrap_lr(y: np.ndarray, items: list[dict], n: int = 2000, seed: int = 42) -> list[float]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    lrs = []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        lrs.append(measured_lr(absence_counts(y[b], [items[i] for i in b])))
    return [float(np.percentile(lrs, 2.5)), float(np.percentile(lrs, 97.5))]


def apply_absence(items: list[dict], lr: float) -> np.ndarray:
    from src.fusion.meta import to_score
    from src.retrieval.coverage import absence_probability

    return np.array([to_score(absence_probability(None if it["score"] is None else it["score"] / 100, lr))
                     if it["fires"] else (math.nan if it["score"] is None else it["score"]) for it in items], dtype=float)


def absence_report(sources: list[str]) -> dict:
    """LR measured on the dev set, then the test (and matched) sets scored once with it. Needs recordings with
    news (augment --sources news --on wiki)."""
    from src.retrieval.coverage import ABSENCE_LR
    from src.retrieval.wiki_search import WikiSearchClient

    suffix = suffix_of(sources)
    chosen = json.loads((CACHE / f"chosen{suffix_of([s for s in sources if s != 'news'])}.json").read_text())
    ms = models()
    wiki = WikiSearchClient(retries=6)
    report = {"rule": "LR = P(rule fires | true) / P(rule fires | fake) among eligible dev items (a claim expected "
                      "to be covered whose news search ran), add-one smoothed; test scored once with that LR",
              "provisional_lr_in_code": ABSENCE_LR, "sources": sources}
    dev = attach_dates(_load(f"dev{suffix}"), "dev")
    ydev = np.array(dev["labels"], dtype=float)
    for m in ("deployed",):
        b = chosen["best"][m]
        for mode in ("no_relevant_report", "no_confirming_report"):
            dev_items = absence_items(dev, b["match_min"], ms[m], wiki, mode)
            counts = absence_counts(ydev, dev_items)
            lr = measured_lr(counts)
            entry = {"dev_counts": counts, "lr": lr, "lr_95ci_bootstrap": bootstrap_lr(ydev, dev_items),
                     "dev_before": bands(ydev, np.array([math.nan if it["score"] is None else it["score"]
                                                         for it in dev_items])),
                     "dev_after": bands(ydev, apply_absence(dev_items, lr))}
            report[f"{m}/{mode}"] = entry
            d = entry["dev_after"]
            print(f"{m} {mode}: LR {lr:.3f} CI {np.round(entry['lr_95ci_bootstrap'], 3)} | dev after: true low "
                  f"{d['true']['low']:.3f} none {d['true']['no_score']:.3f}, fake low {d['fake']['low']:.3f} none "
                  f"{d['fake']['no_score']:.3f} | {counts}", flush=True)
    # Choice on dev only (deployed meta-learner): silence must be evidence AGAINST the claim
    # (LR < 1), true claims rated Low <= MAX_TRUE_LOW and fakes rated Credible <= MAX_FAKE_CREDIBLE; then the
    # mode that rates the most fakes Low. None qualifies -> the rule stays off.
    ok = [mode for mode in ("no_relevant_report", "no_confirming_report")
          if report[f"deployed/{mode}"]["lr"] < 1
          and report[f"deployed/{mode}"]["dev_after"]["true"]["low"] <= MAX_TRUE_LOW
          and report[f"deployed/{mode}"]["dev_after"]["fake"]["credible"] <= MAX_FAKE_CREDIBLE]
    choice = max(ok, key=lambda mo: report[f"deployed/{mo}"]["dev_after"]["fake"]["low"]) if ok else None
    report["chosen_mode"] = choice
    report["chosen_lr"] = report[f"deployed/{choice}"]["lr"] if choice else None
    # What is deployed either way: news search as confirm-only evidence (+ the time rule), with no absence rule
    # when none qualifies. Scored once on the test sets.
    for name in ("dev", "test", "matched"):
        if not (CACHE / f"{name}{suffix}.pkl").exists():
            continue
        rec = attach_dates(_load(f"{name}{suffix}"), name)
        base = attach_dates(_load(f"{name}{suffix_of([s for s in sources if s != 'news'])}"), name)
        y = np.array(rec["labels"], dtype=float)
        report[f"news_confirm_only/{name}"] = {}
        for m in ("deployed",):
            b = chosen["best"][m]
            report[f"news_confirm_only/{name}"][m] = {
                "without_news": bands(y, replay(base, b["match_min"], ms[m], time_rule=True)),
                "with_news": bands(y, replay(rec, b["match_min"], ms[m], time_rule=True)),
                "with_news_may_refute": bands(y, replay(rec, b["match_min"], ms[m], time_rule=True,
                                                        support_prefixes=("wiki:",)))}
    for m in ("deployed",):
        if choice is None:
            break
        b, lr = chosen["best"][m], report[f"{m}/{choice}"]["lr"]
        for name in ("test", "matched"):
            if not (CACHE / f"{name}{suffix}.pkl").exists():
                continue
            rec = attach_dates(_load(f"{name}{suffix}"), name)
            y = np.array(rec["labels"], dtype=float)
            items = absence_items(rec, b["match_min"], ms[m], wiki, choice)
            report[f"{m}/{choice}"][name] = {
                "counts": absence_counts(y, items),
                "before": bands(y, np.array([math.nan if it["score"] is None else it["score"] for it in items])),
                "after": bands(y, apply_absence(items, lr))}
    print("chosen on dev:", choice, report["chosen_lr"])
    path = ROOT / "reports" / f"absence_eval{suffix}.json"
    path.write_text(json.dumps(report, indent=2))
    print("saved", path)
    return report


# --- absence rule, re-measured (2026-11-07) ---------------------------------------------------------------
# Two flaws in the 6 Nov measurement (absence_report above):
# 1. Silence was read from the claim's top-5 evidence AFTER news passages were re-ranked together with Wikipedia,
#    fact-check and index passages. A relevant news report ranked 6th was invisible, so the claim looked
#    "silent" although the news search had found it. Silence is a property of the news search: it is now read
#    from every trusted passage the search returned.
# 2. The LR's denominator was every eligible item, but the rule "fired" only when nothing else had decided the
#    item. Our fakes were collected from fact-checks, so most are already refuted by their own fact-check: they
#    counted as "not silent" whatever the news said, which made silence look like evidence of truth. Silence is
#    one piece of evidence; its LR is P(silent | true) / P(silent | fake) among eligible items, whatever the other
#    checks said. (Naive-Bayes assumption: given the label, whether a fact-check exists does not change what the
#    news search finds. The LR among undecided items only is reported as a check of that assumption.)
#
# Decision rule, written before these numbers were computed (dev set only; test scored once afterwards):
#   switch the rule on with the measured LR if (a) the upper end of the 95% bootstrap CI of the LR is below 1
#   (silence must be shown to be evidence against the claim, not just lean that way), and (b) applied to the dev
#   items where nothing else decided, true claims rated Low <= MAX_TRUE_LOW and fakes rated Credible <=
#   MAX_FAKE_CREDIBLE. Otherwise it stays off. Per-event numbers are exploratory (too few fakes per event).

def _cache_only_news_client():
    """The Tavily client reading only the 30-day answer cache of the 6 Nov run: no key, breaker open, so a
    cache miss returns None (search counted as "did not run") and no search is ever spent."""
    from src.retrieval.news_search import NewsSearchClient

    client = NewsSearchClient(backend="tavily", api_key=None)
    client._open_until = float("inf")
    return client


def news_relevance_all(rec: dict, name: str, reranker, client=None,
                       twice_if=None) -> tuple[list[list[dict | None]], dict]:
    """Per item and recorded claim, over every trusted report the news search returned (before the top-k cut):
    relevant (re-ranker relevance >= RELEVANT_REPORT among dated reports: the evaluation's strict date rule),
    relevant_all (dated and undated, `news_search.relevant_reports`: what the app now uses), trusted, undated,
    raw (articles before the trusted-outlet and date filters), max_relevance. None: no news search for that
    claim, or not in the cache.
    client: default the cache-only client (no searches). twice_if(claim) -> bool: search that claim a second time
    with its keywords when the first search heard nothing (`lookup_twice`; spends a search unless cached)."""
    from src.pipeline import MAX_NEWS_CLAIMS
    from src.preprocess.clean_text import extract_dates
    from src.retrieval.coverage import RELEVANT_REPORT, claim_time, news_window
    from src.retrieval.factcheck_api import relevance
    from src.retrieval.news_search import relevant_reports

    client = client or _cache_only_news_client()
    calls_before = client.calls
    dates = load_set(name)["date"].tolist()
    out, misses = [], 0
    for it, d in zip(rec["items"], dates):
        before = d if isinstance(d, str) else None
        counts = []
        for j, c in enumerate(it["calls"]):
            if j >= MAX_NEWS_CLAIMS or c.get("news_ok") is not True:
                counts.append(None)
                continue
            start, end = news_window(claim_time(c["claim"], extract_dates(c["claim"]), None, before), before)
            if twice_if is not None and twice_if(c["claim"]):
                lk = client.lookup_twice(c["claim"], before=end, start=start, strict_dates=before is not None,
                                         reranker=reranker)
            else:
                lk = client.lookup(c["claim"], before=end, start=start, strict_dates=before is not None)
            if not lk.ok:
                misses += 1
                counts.append(None)
                continue
            rels = [relevance(float(x)) for x in reranker(c["claim"], [p.text for p in lk.passages])] if lk.passages else []
            counts.append({"relevant": sum(r >= RELEVANT_REPORT for r in rels),
                           "relevant_all": relevant_reports(c["claim"], lk, reranker), "trusted": len(lk.passages),
                           "undated": len(lk.undated), "raw": lk.n_results, "max_relevance": max(rels, default=None),
                           "queries": lk.queries, "debunks": len(lk.debunk_ids)})
        out.append(counts)
    return out, {"cache_hits": client.cache_hits, "cache_misses": misses, "searches_spent": client.calls - calls_before}


def silence_items(rec: dict, rel_all: list[list[dict | None]], match_min: float, model,
                  wiki=None) -> list[dict]:
    """Per item: model score, eligible (a claim the news would cover, searched), silent (eligible and no relevant
    trusted report for any such claim, counted over everything the search returned), decided (claim evidence
    counted), and the event kinds of its expected claims."""
    from src.claims.extract import tag_entities
    from src.fusion.meta import to_score
    from src.retrieval.coverage import expected_coverage

    rows, _ = replay_rows(rec, match_min, support_only=True, time_rule=True)
    p = model.predict(np.array([F.to_vector(r) for r in rows]))
    out = []
    for it, counts, row, pi in zip(rec["items"], rel_all, rows, p):
        eligible_claims = []
        for c, news in zip(it["calls"], counts):
            ents = tag_entities(c["claim"])
            exp = expected_coverage(c["claim"], ents, notable=wiki.is_well_known if wiki is not None else None)
            if exp.expected and news is not None:
                # Diagnostic: is the "well-known subject" a place (spaCy GPE / LOC)? A famous state is not a
                # famous person.
                place = exp.kind == "notable name" and any(
                    lab in ("GPE", "LOC") and (n in exp.subject or exp.subject in n) for n, lab in ents)
                eligible_claims.append({"event": exp.event, "kind": exp.kind, "place_subject": place, **news})
        decided = not math.isnan(row.get("claim_evidence_score", math.nan))
        silent = bool(eligible_claims) and all(e["relevant"] == 0 for e in eligible_claims)
        silent_live = bool(eligible_claims) and all(e["relevant_all"] == 0 for e in eligible_claims)
        out.append({"score": to_score(float(pi)) if F.has_evidence(row) else None,
                    "eligible": bool(eligible_claims), "silent": silent, "silent_live": silent_live,
                    "decided": decided,
                    "events": [e["event"] for e in eligible_claims],
                    "kinds": [e["kind"] + (" (a place)" if e["place_subject"] else "") for e in eligible_claims],
                    "why_silent": None if not silent else
                    "no articles at all" if all(e["raw"] == 0 for e in eligible_claims) else
                    "articles, none from a trusted outlet in the window" if all(e["trusted"] == 0 for e in eligible_claims)
                    else "trusted reports, none relevant",
                    "max_relevance": max((e["max_relevance"] for e in eligible_claims
                                          if e["max_relevance"] is not None), default=None)})
    return out


def silence_counts(y: np.ndarray, items: list[dict], undecided_only: bool = False, key: str = "silent") -> dict:
    """key "silent": dated reports only (the evaluation's strict date rule); "silent_live": undated reports count
    too, as in the app for a claim typed without a date."""
    out = {}
    for lab, name in ((1.0, "true"), (0.0, "fake")):
        m = [it for it, yy in zip(items, y) if yy == lab and it["eligible"] and not (undecided_only and it["decided"])]
        out[name] = {"eligible": len(m), "silent": sum(it[key] for it in m)}
    return out


def silence_lr(counts: dict) -> float:
    """P(silent | true) / P(silent | fake), add-one smoothed (`coverage.likelihood_ratio` takes "found" counts)."""
    from src.retrieval.coverage import likelihood_ratio

    t, f = counts["true"], counts["fake"]
    return likelihood_ratio(t["eligible"] - t["silent"], t["eligible"], f["eligible"] - f["silent"], f["eligible"])


def silence_lr_ci(y: np.ndarray, items: list[dict], undecided_only: bool = False, n: int = 2000,
                  seed: int = 42, key: str = "silent") -> list[float]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    lrs = [silence_lr(silence_counts(y[b], [items[i] for i in b], undecided_only, key))
           for b in (rng.choice(idx, len(idx), replace=True) for _ in range(n))]
    return [float(np.percentile(lrs, 2.5)), float(np.percentile(lrs, 97.5))]


def apply_silence(items: list[dict], lr: float, key: str = "silent") -> np.ndarray:
    """Scores after the rule: an undecided silent item gets the Bayes update from the model's probability (even
    odds when no other check spoke); everything else keeps its score."""
    from src.fusion.meta import to_score
    from src.retrieval.coverage import absence_probability

    return np.array([to_score(absence_probability(None if it["score"] is None else it["score"] / 100, lr))
                     if it[key] and not it["decided"] else (math.nan if it["score"] is None else it["score"])
                     for it in items], dtype=float)


def silence_recheck_report(sources: list[str]) -> dict:
    """The absence rule re-measured with both flaws fixed; decision by the rule written above."""
    from src.retrieval.search import cross_encoder_reranker
    from src.retrieval.wiki_search import WikiSearchClient

    suffix = suffix_of(sources)
    chosen = json.loads((CACHE / f"chosen{suffix_of([s for s in sources if s != 'news'])}.json").read_text())
    ms, wiki, reranker = models(), WikiSearchClient(retries=6), cross_encoder_reranker()
    report = {"rule": "LR = P(silent | true) / P(silent | fake) among eligible dev items (a claim the news would "
                      "cover, searched); silent = no relevant trusted report among ALL passages the search returned; "
                      "on if the 95% CI upper end < 1 and, applied to undecided dev items, true Low <= "
                      f"{MAX_TRUE_LOW} and fakes Credible <= {MAX_FAKE_CREDIBLE}; test scored once",
              "sources": sources}
    recs = {}
    for name in ("dev", "test", "matched"):
        if (CACHE / f"{name}{suffix}.pkl").exists():
            rec = attach_dates(_load(f"{name}{suffix}"), name)
            rel, q = news_relevance_all(rec, name, reranker)
            recs[name] = (rec, rel)
            report[f"{name}_news_cache"] = q
            print(name, "news cache:", q, flush=True)
    b = chosen["best"]["deployed"]  # the deployed meta-learner and claim-match threshold
    dev, rel = recs["dev"]
    y = np.array(dev["labels"], dtype=float)
    items = silence_items(dev, rel, b["match_min"], ms["deployed"], wiki)
    counts, counts_u = silence_counts(y, items), silence_counts(y, items, undecided_only=True)
    lr, lr_u = silence_lr(counts), silence_lr(counts_u)
    ci, ci_u = silence_lr_ci(y, items), silence_lr_ci(y, items, undecided_only=True)
    before = bands(y, np.array([math.nan if it["score"] is None else it["score"] for it in items]))
    after = bands(y, apply_silence(items, lr))
    events = sorted({e for it in items for e in it["events"]})
    per_event = {e: silence_counts(y, [{**it, "eligible": it["eligible"] and e in it["events"]} for it in items])
                 for e in events}
    kinds = sorted({k for it in items for k in it["kinds"]})
    per_kind = {k: silence_counts(y, [{**it, "eligible": it["eligible"] and k in it["kinds"]} for it in items])
                for k in kinds}
    why_silent = {lab: {w: sum(1 for it, yy in zip(items, y) if yy == v and it["why_silent"] == w)
                        for w in ("no articles at all", "articles, none from a trusted outlet in the window",
                                  "trusted reports, none relevant")} for v, lab in ((1.0, "true"), (0.0, "fake"))}
    near = {lab: int(sum(1 for it, yy in zip(items, y) if yy == v and it["silent"] and it["max_relevance"] is not None
                         and it["max_relevance"] >= 0.2)) for v, lab in ((1.0, "true"), (0.0, "fake"))}
    counts_l = silence_counts(y, items, key="silent_live")
    lr_l, ci_l = silence_lr(counts_l), silence_lr_ci(y, items, key="silent_live")
    per_event_l = {e: silence_counts(y, [{**it, "eligible": it["eligible"] and e in it["events"]} for it in items],
                                     key="silent_live") for e in events}
    print(f"dev, undated reports counted (as the app for an undated claim): LR {lr_l:.3f} CI {np.round(ci_l, 3)} "
          f"{counts_l}", flush=True)
    for e, c in per_event_l.items():
        print(f"  {e}: {c}", flush=True)
    # Rule for this instrument, written after seeing only its dev LR above and before its effect on scores or any
    # test number (it was added after the strict instrument failed, so the test set must confirm it):
    #   dev:  95% CI upper end < 1, and applied: true Low <= MAX_TRUE_LOW, fakes Credible <= MAX_FAKE_CREDIBLE;
    #   test: its own LR's 95% CI upper end < 1, and with the dev LR: true Low < 10% (the project's goal; the test
    #         set was already at 6.3% before) and fakes Credible <= MAX_FAKE_CREDIBLE.
    after_l = bands(y, apply_silence(items, lr_l, key="silent_live"))
    ok_live_dev = (ci_l[1] < 1 and after_l["true"]["low"] <= MAX_TRUE_LOW
                   and after_l["fake"]["credible"] <= MAX_FAKE_CREDIBLE)
    print(f"dev after (undated counted, LR {lr_l:.3f}): true low {after_l['true']['low']:.3f} cred "
          f"{after_l['true']['credible']:.3f} none {after_l['true']['no_score']:.3f} | fake low {after_l['fake']['low']:.3f} "
          f"cred {after_l['fake']['credible']:.3f} none {after_l['fake']['no_score']:.3f} | AUC "
          f"{after_l.get('roc_auc_scored', float('nan')):.3f} -> dev passes: {ok_live_dev}", flush=True)
    ok = ci[1] < 1 and after["true"]["low"] <= MAX_TRUE_LOW and after["fake"]["credible"] <= MAX_FAKE_CREDIBLE
    report["dev"] = {"counts": counts, "lr": lr, "lr_95ci": ci, "counts_undecided_only": counts_u,
                     "lr_undecided_only": lr_u, "lr_undecided_only_95ci": ci_u, "per_event": per_event,
                     "per_subject_kind": per_kind, "why_silent": why_silent,
                     "silent_with_a_report_of_relevance_0.2_to_0.5": near,
                     "undated_reports_counted": {"counts": counts_l, "lr": lr_l, "lr_95ci": ci_l,
                                                 "per_event": per_event_l, "after": after_l,
                                                 "passes_dev": bool(ok_live_dev)},
                     "before": before, "after": after}
    report["switch_on"] = bool(ok)
    report["chosen_lr"] = lr if ok else None
    print(f"dev: LR {lr:.3f} CI {np.round(ci, 3)} {counts} | undecided only: LR {lr_u:.3f} CI {np.round(ci_u, 3)} "
          f"{counts_u}", flush=True)
    for e, c in {**per_event, **per_kind}.items():
        print(f"  {e}: {c}", flush=True)
    print("why silent:", why_silent, "| silent but a report of relevance 0.2-0.5:", near, flush=True)
    print(f"dev after: true low {after['true']['low']:.3f} cred {after['true']['credible']:.3f} none "
          f"{after['true']['no_score']:.3f} | fake low {after['fake']['low']:.3f} cred {after['fake']['credible']:.3f} "
          f"none {after['fake']['no_score']:.3f} -> switch on: {ok}", flush=True)
    # The test sets are scored once with the dev LR, whatever the decision (reported, not used to choose).
    for name in ("test", "matched"):
        if name not in recs:
            continue
        rec, rel = recs[name]
        yt = np.array(rec["labels"], dtype=float)
        it_t = silence_items(rec, rel, b["match_min"], ms["deployed"], wiki)
        ct = silence_counts(yt, it_t)
        ct_l = silence_counts(yt, it_t, key="silent_live")
        report[name] = {"counts": ct, "lr_on_this_set": silence_lr(ct), "lr_95ci_on_this_set": silence_lr_ci(yt, it_t),
                        "before": bands(yt, np.array([math.nan if i["score"] is None else i["score"] for i in it_t])),
                        "after_with_dev_lr": bands(yt, apply_silence(it_t, lr)),
                        "undated_reports_counted": {
                            "counts": ct_l, "lr_on_this_set": silence_lr(ct_l),
                            "lr_95ci_on_this_set": silence_lr_ci(yt, it_t, key="silent_live"),
                            "after_with_dev_lr": bands(yt, apply_silence(it_t, lr_l, key="silent_live"))}}
        print(f"{name}: LR on this set {report[name]['lr_on_this_set']:.3f} {ct}", flush=True)
        u = report[name]["undated_reports_counted"]
        a = u["after_with_dev_lr"]
        print(f"{name}, undated counted: LR on this set {u['lr_on_this_set']:.3f} CI {np.round(u['lr_95ci_on_this_set'], 3)} "
              f"{ct_l} | with dev LR: true low {a['true']['low']:.3f} cred {a['true']['credible']:.3f} none "
              f"{a['true']['no_score']:.3f} | fake low {a['fake']['low']:.3f} cred {a['fake']['credible']:.3f} none "
              f"{a['fake']['no_score']:.3f} | AUC {a.get('roc_auc_scored', float('nan')):.3f}", flush=True)
    if "test" in report:
        u = report["test"]["undated_reports_counted"]
        confirmed = (u["lr_95ci_on_this_set"][1] < 1 and u["after_with_dev_lr"]["true"]["low"] < 0.10
                     and u["after_with_dev_lr"]["fake"]["credible"] <= MAX_FAKE_CREDIBLE)
        report["undated_reports_counted_switch_on"] = bool(ok_live_dev and confirmed)
        report["undated_reports_counted_lr"] = lr_l if ok_live_dev and confirmed else None
        print("undated counted: dev passes", ok_live_dev, "| test confirms", confirmed, flush=True)
    path = ROOT / "reports" / f"absence_recheck{suffix}.json"
    path.write_text(json.dumps(report, indent=2))
    print("saved", path)
    return report


# --- absence rule with invented-event hoaxes (2026-11-07) ---------------------------------------------------
# Written before the hoax set was searched (python -m src.ingest.live_testset --hoax built it from fact-checks;
# no news result for it existed when this rule was fixed).
#
# Instrument (the pipeline's, src/retrieval/news_search.py): eligible = a claim the news would cover
#   (expected_coverage) whose news search answered; heard = at least one trusted report with re-ranker relevance
#   >= 0.5 among everything the search returned (undated too, unless the URL names a later year; fact-check /
#   debunk headlines never count); a claim that heard nothing is searched a second time with its keywords
#   (`lookup_twice`); silent = eligible and not heard.
# Data: dev = India dev set + the hoax set's "dev" half; test = India test set + the hoax "test" half, scored
#   once, after the decision.
# LR = P(silent | true) / P(silent | fake) with the dev fakes pooled (India fakes + hoaxes), add-one smoothed,
#   95% bootstrap CI. Applied where nothing else decided: posterior = model probability (even odds if nothing
#   spoke) x LR.
# Switch on with the dev LR if
#   dev:  CI upper end < 1, India dev true claims rated Low <= 5%, India dev fakes rated Credible <= 5%;
#   test: pooled test LR's CI upper end < 1, India test true claims rated Low < 10%, fakes rated Credible <= 5%.
# Otherwise the rule stays off. Hoaxes are also reported "before their fact-check existed" (their own fact-check
# removed, `drop_same_claim`): the situation the rule is for, since a new hoax has no fact-check yet.
HOAX_DECISION = ("on if dev pooled LR CI upper < 1 and India dev true Low <= 5% and fakes Credible <= 5%, and test "
                 "pooled LR CI upper < 1 and India test true Low < 10% and fakes Credible <= 5%")


def _hoax_split(name: str) -> np.ndarray:
    return load_set(name)["split"].to_numpy()


def silence_hoax_report(sources: list[str]) -> dict:
    """The absence rule measured with the invented-event hoaxes; decision by HOAX_DECISION (written above before
    any hoax result). Spends Tavily searches only for second queries and hoax claims not yet in the cache."""
    from src.claims.extract import tag_entities
    from src.retrieval.coverage import expected_coverage
    from src.retrieval.news_search import default_client
    from src.retrieval.search import cross_encoder_reranker
    from src.retrieval.wiki_search import WikiSearchClient

    suffix = suffix_of(sources)
    chosen = json.loads((CACHE / f"chosen{suffix_of([s for s in sources if s != 'news'])}.json").read_text())
    ms, wiki, reranker = models(), WikiSearchClient(retries=6), cross_encoder_reranker()
    live = default_client()
    if live is None or live.backend != "tavily":
        raise SystemExit("TAVILY_API_KEY is needed for the second searches")
    live.retries = 3

    def covered(text: str) -> bool:
        return expected_coverage(text, tag_entities(text), wiki.is_well_known).expected

    b = chosen["best"]["deployed"]  # the deployed meta-learner and claim-match threshold
    report = {"decision_rule": HOAX_DECISION, "sources": sources, "meta": "deployed", "match_min": b["match_min"]}
    sets = {}
    for name in ("dev", "test", "hoax"):
        rec = attach_dates(_load(f"{name}{suffix}"), name)
        rel, q = news_relevance_all(rec, name, reranker, client=live, twice_if=covered)
        report[f"{name}_news"] = q
        print(name, "news:", q, flush=True)
        items = silence_items(rec, rel, b["match_min"], ms["deployed"], wiki)
        sets[name] = (rec, np.array(rec["labels"], dtype=float), items, rel)
    # Hoaxes before their fact-check existed: the same recording without their own fact-check.
    h_items_new = silence_items(drop_same_claim(sets["hoax"][0]), sets["hoax"][3], b["match_min"], ms["deployed"],
                                wiki)
    split = _hoax_split("hoax")

    def pool(part: str) -> tuple[np.ndarray, list[dict]]:
        _, y, it, _ = sets[part]
        _, yh, ih, _ = sets["hoax"]
        m = split == part
        return np.concatenate([y, yh[m]]), it + [x for x, keep in zip(ih, m) if keep]

    decision = {}
    for part in ("dev", "test"):
        y, items = pool(part)
        counts = silence_counts(y, items, key="silent_live")
        lr = silence_lr(counts)
        ci = silence_lr_ci(y, items, key="silent_live")
        _, yi, ii, _ = sets[part]
        india_counts = silence_counts(yi, ii, key="silent_live")
        m = split == part
        hoax_counts = silence_counts(sets["hoax"][1][m], [x for x, keep in zip(sets["hoax"][2], m) if keep],
                                     key="silent_live")
        decision[part] = {"pooled_counts": counts, "lr": lr, "lr_95ci": ci, "india_counts": india_counts,
                          "hoax_counts": hoax_counts,
                          "india_lr": silence_lr(india_counts)}
        print(f"{part}: pooled LR {lr:.3f} CI {np.round(ci, 3)} {counts} | India {india_counts} | hoaxes {hoax_counts}",
              flush=True)
    lr_dev = decision["dev"]["lr"]
    for part in ("dev", "test"):
        _, yi, ii, _ = sets[part]
        before = bands(yi, np.array([math.nan if it["score"] is None else it["score"] for it in ii]))
        after = bands(yi, apply_silence(ii, lr_dev, key="silent_live"))
        m = split == part
        hy = sets["hoax"][1][m]
        h_rec = [x for x, keep in zip(sets["hoax"][2], m) if keep]
        h_new = [x for x, keep in zip(h_items_new, m) if keep]
        hoax_bands = {
            "as_recorded_before": bands(hy, np.array([math.nan if it["score"] is None else it["score"] for it in h_rec])),
            "as_recorded_after": bands(hy, apply_silence(h_rec, lr_dev, key="silent_live")),
            "own_factcheck_removed_before": bands(hy, np.array([math.nan if it["score"] is None else it["score"]
                                                                for it in h_new])),
            "own_factcheck_removed_after": bands(hy, apply_silence(h_new, lr_dev, key="silent_live"))}
        decision[part].update({"india_before": before, "india_after_with_dev_lr": after, "hoax": hoax_bands})
        a, hb = after, hoax_bands["own_factcheck_removed_after"]
        print(f"{part} India with dev LR {lr_dev:.3f}: true low {before['true']['low']:.3f} -> {a['true']['low']:.3f}, "
              f"cred {a['true']['credible']:.3f} | fake low {before['fake']['low']:.3f} -> {a['fake']['low']:.3f}, cred "
              f"{a['fake']['credible']:.3f} | AUC {before.get('roc_auc_scored', float('nan')):.3f} -> "
              f"{a.get('roc_auc_scored', float('nan')):.3f}", flush=True)
        hb0 = hoax_bands["own_factcheck_removed_before"]
        print(f"  hoaxes without their fact-check: low {hb0['fake']['low']:.3f} -> {hb['fake']['low']:.3f}, no score "
              f"{hb0['fake']['no_score']:.3f} -> {hb['fake']['no_score']:.3f}, cred {hb['fake']['credible']:.3f}",
              flush=True)
    d, t = decision["dev"], decision["test"]
    pass_dev = (d["lr_95ci"][1] < 1 and d["india_after_with_dev_lr"]["true"]["low"] <= MAX_TRUE_LOW
                and d["india_after_with_dev_lr"]["fake"]["credible"] <= MAX_FAKE_CREDIBLE
                and d["hoax"]["own_factcheck_removed_after"]["fake"]["credible"] <= MAX_FAKE_CREDIBLE)
    pass_test = (t["lr_95ci"][1] < 1 and t["india_after_with_dev_lr"]["true"]["low"] < 0.10
                 and t["india_after_with_dev_lr"]["fake"]["credible"] <= MAX_FAKE_CREDIBLE
                 and t["hoax"]["own_factcheck_removed_after"]["fake"]["credible"] <= MAX_FAKE_CREDIBLE)
    report.update({"dev": d, "test": t, "passes_dev": bool(pass_dev), "passes_test": bool(pass_test),
                   "switch_on": bool(pass_dev and pass_test), "chosen_lr": lr_dev if pass_dev and pass_test else None})
    print(f"passes dev: {pass_dev} | passes test: {pass_test} -> switch on: {pass_dev and pass_test}", flush=True)
    path = ROOT / "reports" / "absence_hoax_eval.json"
    path.write_text(json.dumps(report, indent=2))
    print("saved", path)
    return report


def time_rule_report() -> dict:
    """Effect of the time rule alone on the deployed setting (Wikipedia on), every India set: offline replay."""
    chosen = json.loads((CACHE / "chosen_wiki.json").read_text())
    ms = models()
    out = {"rule": "a Wikipedia passage can confirm a dated claim only if it names the claim's year; other dated "
                   "passages (not news) written > 31 days before a day-dated claim cannot confirm it"}
    for name in ("dev", "test", "matched"):
        rec = attach_dates(_load(f"{name}_wiki"), name)
        y = np.array(rec["labels"], dtype=float)
        for m in ("deployed",):
            b = chosen["best"][m]
            r = rematch(rec, model=chosen["matcher"]) if b.get("matcher") == "learned" else rec
            s0, s1 = replay(r, b["match_min"], ms[m]), replay(r, b["match_min"], ms[m], time_rule=True)
            changed = int(np.sum(~((s0 == s1) | (np.isnan(s0) & np.isnan(s1)))))
            out[f"{name}/{m}"] = {"items_changed": changed, "n": len(y), "before": bands(y, s0), "after": bands(y, s1)}
    path = ROOT / "reports" / "time_rule_eval.json"
    path.write_text(json.dumps(out, indent=2))
    print("saved", path)
    return out


# --- root cause A: a verdict on a related claim is not a verdict on ours (2026-11-12) ----------------------
# Found 2026-11-08 (reports/known_issues.md): claim matching accepts a fact-check whose claim is only *similar*
# to ours (cosine >= MATCH_MIN), so a "False" on a related rumour refutes a true claim ("The Eiffel Tower is
# located in Paris, France." vs AFP's "Video shows the Eiffel Tower collapsing": 13 Low).
# Hypothesis under test: a verdict on the judged claim B carries over to our claim A only where logic allows it:
# "B is false" refutes A only if A implies B, "B is true" supports A only if B implies A. An NLI model gives both
# directions (src/verify/model.py `ClaimEntailment`, `entailment_allows`).
#
# Candidates, simplest first (RCA_FAMILIES):
#   cosine           a higher similarity threshold only (no new model)
#   learned          the learned matcher of 2026-11-03 (`fit_matcher`; out-of-fold values on dev)
#   nli_both         similarity >= m and each claim entails the other with P >= t (the same claim)
#   nli_directional  similarity >= m and the direction the logic needs has P >= t (the hypothesis)
# Grids are fixed here: m in RCA_MATCH_GRID, t in RCA_ENTAIL_GRID, cosine / learned thresholds from THRESHOLDS.
#
# Decision rule, written before the first dev measurement of any candidate. Everything is measured by replay of
# the cached recordings in the deployed setting (news and Wikipedia confirm-only, time rule on):
ROOT_CAUSE_A_DECISION = (
    "ship a candidate only if, on the India dev set in the deployed setting, ALL hold: true claims rated Low <= "
    "10/227 (4.4%, the value before the change); fakes rated Low >= 65.7% (68.7% before); fakes rated Credible <= "
    "5% (0% before); ROC-AUC on scored items >= 0.934 (0.944 before); at least 3 of the 5 known cases KI-10, "
    "KI-11, FB-04, KI-06, KI-07 no longer have a Refuted verdict and are not rated Low; none of the fixed bugs "
    "FB-01..FB-09 comes back (offline: FB-03 and FB-09 stay Low, FB-05 stays not Credible; all nine again in the "
    "known-issues run against the API after the change, where FB-04 must not be worse than before). Families are "
    "tried in order of simplicity (cosine, learned, nli_both, nli_directional) and the first family with a "
    "passing setting wins; inside it the setting that rates the most fakes Low, then the fewest true claims Low, "
    "then the highest ROC-AUC. No family passes: nothing ships and the served behaviour stays as it was. The "
    "baseline replay must first reproduce the numbers before the change, or the measurement is not trusted. The "
    "test, matched and hoax sets are scored once, after the decision, and reported whatever they show.")
RCA_SOURCES = ["news", "wiki"]
RCA_MAX_TRUE_LOW = 10 / 227
RCA_MIN_FAKE_LOW = 0.657
RCA_MAX_FAKE_CREDIBLE = 0.05
RCA_MIN_AUC = 0.934
RCA_CASES = ("KI-10", "KI-11", "FB-04", "KI-06", "KI-07")
RCA_MIN_CASES_FIXED = 3
RCA_GUARDS = {"FB-03": "low", "FB-05": "not credible", "FB-09": "low"}  # fixed bugs that can be replayed offline
RCA_FAMILIES = ("cosine", "learned", "nli_both", "nli_directional")
RCA_MATCH_GRID = (0.3, 0.4, 0.5, 0.55)
RCA_ENTAIL_GRID = (0.3, 0.5, 0.7, 0.9)
RCA_BEFORE_MATCH_MIN = 0.55  # the served threshold (plain cosine) before this change
# Added after the first dev grid, before any test-set number: on the 0.05 grid no cosine threshold passed, and the
# boundary lay between 0.60 (fakes rated Low 66.0%, 1 known case fixed) and 0.65 (64.0%, 3 fixed). A single
# threshold is the simplest candidate, so the steps in between are measured too, under the same rule.
RCA_COSINE_REFINED = (0.61, 0.62, 0.63, 0.64)
RCA_BASELINE = {"true_low": 10 / 227, "fake_low": 0.687, "fake_credible": 0.0, "roc_auc": 0.944}  # reports/absence_eval_news_wiki.json
# Tightening, added after the dev grids and before any test-set number or any rewording was measured: cosine 0.61
# passed with the three cases it fixes sitting 0.01 or less below the threshold (their related fact-checks have
# cosine 0.60-0.61 to the claim). A fix that holds only for the exact sentence in the bug list is not a fix: a
# user types the same fact in other words. So a known case counts as fixed only if it is fixed as written AND
# for at least RCA_MIN_REWORDINGS_FIXED of its three rewordings below (same meaning, written before measuring),
# replayed on the case's recorded evidence with the verifier and the matchers re-run on the new wording.
RCA_MIN_REWORDINGS_FIXED = 2
# Second tightening, added after looking at WHY the learned matcher's three cases changed and before any test-set
# number. All five cases are true claims, so every verdict passage that is read as refuting one of them and still
# counts is the bug itself. With the learned matcher at 0.80 "Barack Obama was born in Hawaii" became Supported
# while the Kenya birth-certificate fact-check still counted against it (match probability 0.90): two other
# answers about the Kenya hoax were misread as support and outvoted it. On the dev set the same matcher left 17
# of 227 true claims with a counted refuting verdict passage (20 before; entailment: 0 to 3) and rated 3 true
# claims Low that were not Low before. "No longer Refuted" must mean the related fact-check no longer counts:
# a known case is fixed only if, as written and in at least two rewordings, it has no Refuted verdict, is not
# rated Low, and no verdict passage read as refuting (P(Refuted) >= RCA_REFUTING) counts for it.
RCA_REFUTING = 0.5
# Tested for information on the test sets when nothing ships: the hypothesis at its plain setting (similarity
# threshold unchanged, entailment in the needed direction more likely than not).
RCA_HYPOTHESIS = {"family": "nli_directional", "match_min": 0.55, "entail_mode": "directional", "entail_min": 0.5}
RCA_REWORDINGS = {
    "KI-10": ["The Eiffel Tower is in Paris.",
              "The Eiffel Tower stands in Paris, the capital of France.",
              "Paris is home to the Eiffel Tower."],
    "KI-11": ["Ratan Tata passed away in October 2024.",
              "Industrialist Ratan Tata died in Mumbai in October 2024.",
              "Ratan Tata, former chairman of Tata Sons, died on 9 October 2024."],
    "FB-04": ["Obama was born in Hawaii.",
              "Barack Obama was born in Honolulu, Hawaii.",
              "Former US President Barack Obama was born in the state of Hawaii."],
    "KI-06": ["Nigeria's foreign reserves rose from $33.42 billion at the end of April 2020 to around $36 billion in May 2020.",
              "Nigeria's external reserves increased from $33.42 billion on 29 April 2020 to roughly $36 billion in May 2020.",
              "Between 29 April 2020 and May 2020, Nigeria's external reserves grew from $33.42 billion to about $36 billion."],
    "KI-07": ["In his phone call with Ukraine's President Zelensky on 25 July 2019, Donald Trump said the United States "
              "had been very good to Ukraine.",
              "Donald Trump told Ukrainian President Zelensky in their 25 July 2019 call that the United States has "
              "been good to Ukraine.",
              "During the 25 July 2019 telephone call with President Zelensky of Ukraine, Trump said the US had been "
              "good to Ukraine."],
}
RCA_REPORT = ROOT / "reports" / "root_cause_a_eval.json"
ENTAIL_TABLE = CACHE / "entail_pairs.pkl"


def record_cases(save: bool = True) -> Path:
    """Run the pipeline as the app does (live lookups on, no dates) on the known cases that have a text, and keep
    each claim's hits, verifier probabilities and claim-match values like `record`. Spends live searches: the
    counts are printed and saved. save=False: only print what the pipeline returns now (stage "served": checks
    that the served code gives what the replay predicted; the lookups come from the caches)."""
    from src.fusion.known_issues import CASES
    from src.pipeline import NewsItem, Pipeline
    from src.verify.model import ClaimMatcher, claim_behind, reliability_of

    cosine = ClaimMatcher(model=None)  # recordings keep the plain cosine, whatever matcher is deployed
    ids = list(RCA_CASES) + list(RCA_GUARDS)
    texts = {c.id: c.inputs["text"] for c in CASES if c.id in ids}
    pipe = Pipeline()
    verifier = pipe.get("verifier")
    calls: list[dict] = []
    original_verify = verifier.verify

    def recording_verify(claim, hits, **kw):
        v = original_verify(claim, hits, **kw)
        probs = verifier.predict(claim, [h.passage.text for h in hits]) if hits else np.zeros((0, 4))
        calls.append({"claim": claim, "hits": list(hits), "probs": probs,
                      "matches": cosine.similarity(claim, [claim_behind(h.passage) for h in hits]),
                      "rel": [reliability_of(h.passage.domain, verifier.reliability) for h in hits]})
        return v

    verifier.verify = recording_verify
    out = []
    for cid in ids:
        calls.clear()
        rep = pipe.analyze(NewsItem(text=texts[cid]))
        out.append({"features": rep["features"], "abstained": rep["abstained"], "score": rep["score"],
                    "claims": [{"text": c["text"], "checkworthiness": c["checkworthiness"],
                                "context": c.get("context")} for c in rep["claims"]],
                    "calls": list(calls), "date": None})
        print(f"{cid}: {rep['score']} {rep['band']} {[c.get('verdict') for c in rep['claims']]}", flush=True)
    queries = {}
    for name in ("factcheck", "news"):
        obj = pipe.get(name)
        if obj is not None:
            queries[name] = {"live_searches": obj.calls, "cache_hits": obj.cache_hits}
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / "cases.pkl"
    if save:
        with open(path, "wb") as f:
            pickle.dump({"ids": ids, "labels": [None] * len(ids), "items": out, "queries": queries,
                         "model_version": pipe.model_version, "components": dict(pipe.status)}, f)
    print("live searches:", queries, "| verifier:", pipe.status.get("verifier"), "| saved" if save else "| not saved", path)
    return path


def _verdict_pairs(rec: dict) -> list[tuple[str, str]]:
    """(our claim, the claim a verdict passage judged) for every verdict passage of a recording."""
    from src.verify.model import claim_behind

    out = []
    for it in rec["items"]:
        for c in it["calls"]:
            for h, m in zip(c["hits"], c["matches"]):
                other = claim_behind(h.passage)
                if other and not np.isnan(m):
                    out.append((c["claim"], other))
    return out


def entail_table(recs: list[dict], compute: bool = True) -> dict:
    """(our claim, judged claim) -> (P(ours entails judged), P(judged entails ours)), kept in ENTAIL_TABLE so the
    NLI model runs once per pair."""
    table = {}
    if ENTAIL_TABLE.exists():
        with open(ENTAIL_TABLE, "rb") as f:
            table = pickle.load(f)
    miss = [p for p in dict.fromkeys(p for r in recs for p in _verdict_pairs(r)) if p not in table]
    if miss:
        if not compute:
            raise SystemExit("entailment values missing: python -m src.fusion.india_tune root-cause-a --stage entail")
        from src.verify.model import ClaimEntailment

        ent = ClaimEntailment()
        fwd, bwd = ent.entails(miss), ent.entails([(b, a) for a, b in miss])
        table.update({p: (float(f), float(b)) for p, f, b in zip(miss, fwd, bwd)})
        with open(ENTAIL_TABLE, "wb") as f:
            pickle.dump(table, f)
    return table


def with_entail(rec: dict, table: dict) -> dict:
    """The recording with an `entail` array (n, 2) per claim, aligned with its hits (NaN: not a verdict)."""
    from src.verify.model import claim_behind

    items = []
    for it in rec["items"]:
        calls = []
        for c in it["calls"]:
            e = np.full((len(c["hits"]), 2), np.nan)
            for k, (h, m) in enumerate(zip(c["hits"], c["matches"])):
                other = claim_behind(h.passage)
                if other and not np.isnan(m):
                    e[k] = table[(c["claim"], other)]
            calls.append({**c, "entail": e})
        items.append({**it, "calls": calls})
    return {**rec, "items": items}


def reword_cases() -> Path:
    """The known cases in other words (RCA_REWORDINGS), on the evidence recorded for the original sentence: the
    verifier reads every recorded passage against the new wording and the cosine matcher compares the new wording
    with each judged claim. The passages and their re-ranker scores are kept (no search is repeated)."""
    from src.verify.model import ClaimMatcher, ClaimVerifier, claim_behind

    cases = _load("cases")
    verifier, matcher = ClaimVerifier(matcher=None, match_min=-1.0), ClaimMatcher(model=None)
    ids, items = [], []
    for cid, texts in RCA_REWORDINGS.items():
        it = cases["items"][cases["ids"].index(cid)]
        assert len(it["calls"]) == 1 and len(it["claims"]) == 1, f"{cid}: one claim expected"
        c = it["calls"][0]
        for k, text in enumerate(texts, 1):
            call = {**c, "claim": text, "probs": verifier.predict(text, [h.passage.text for h in c["hits"]]),
                    "matches": matcher.similarity(text, [claim_behind(h.passage) for h in c["hits"]])}
            ids.append(f"{cid}#{k}")
            items.append({**it, "claims": [{**it["claims"][0], "text": text}], "calls": [call]})
    path = CACHE / "cases_reworded.pkl"
    with open(path, "wb") as f:
        pickle.dump({"ids": ids, "labels": [None] * len(ids), "items": items}, f)
    return path


def _rca_rec(name: str, compute: bool = False) -> dict:
    if name == "cases_reworded" and not (CACHE / "cases_reworded.pkl").exists():
        reword_cases()
    rec = _load(name) if name.startswith("cases") else attach_dates(_load(f"{name}{suffix_of(RCA_SOURCES)}"), name)
    return with_entail(rec, entail_table([rec], compute=compute))


def entail_stage() -> None:
    """Entailment values for every verdict passage of the dev set and the known cases (the sets used to decide).
    The test sets get theirs in the test stage, after the decision."""
    from src.verify.model import ENTAIL_MODEL

    for name in ("dev", "cases", "cases_reworded"):
        rec = _rca_rec(name, compute=True)
        pairs = _verdict_pairs(rec)
        print(f"{name}: {len(pairs)} verdict passages, {len(set(pairs))} distinct claim pairs ({ENTAIL_MODEL})", flush=True)


def _rca_replay(rec: dict, setting: dict, model, oof: dict | None = None, matcher: dict | None = None):
    """(scores 0-100 with NaN = no score, per-claim verdicts) of a recording under one candidate setting, in the
    deployed setting (confirm-only sources, time rule)."""
    from src.fusion.meta import to_score

    r = rec
    if setting["family"] == "learned":
        r = rematch(rec, probs_by_key=oof) if oof is not None else rematch(rec, model=matcher)
    rows, verdicts = replay_rows(r, setting["match_min"], support_only=True, time_rule=True,
                                 entail_mode=setting["entail_mode"], entail_min=setting["entail_min"])
    p = model.predict(np.array([F.to_vector(x) for x in rows]))
    scores = np.array([math.nan if not F.has_evidence(x) else (to_score(float(pi)) or math.nan)
                       for x, pi in zip(rows, p)], dtype=float)
    return scores, verdicts


def case_outcomes(cases: dict, scores: np.ndarray, verdicts: list[list]) -> dict:
    out = {}
    for cid, s, vs in zip(cases["ids"], scores, verdicts):
        labels = [v.label for v in vs]
        scored = not math.isnan(s)
        refuting = sum(1 for v in vs for e in v.evidence
                       if e.counted and not math.isnan(e.claim_match) and e.probs["Refuted"] >= RCA_REFUTING)
        out[cid] = {"score": float(s) if scored else None, "verdicts": labels, "refuted": "Refuted" in labels,
                    "low": bool(scored and s <= 30), "credible": bool(scored and s > 60),
                    "refuting_verdict_passages_counted": refuting}
    return out


def rca_check(b: dict, outcomes: dict, reworded: dict) -> dict:
    """The decision rule applied to one candidate's dev bands and case outcomes (as written and reworded)."""
    def ok(o):
        return not o["refuted"] and not o["low"] and o["refuting_verdict_passages_counted"] == 0

    as_written = [c for c in RCA_CASES if ok(outcomes[c])]
    held = {c: sum(ok(o) for k, o in reworded.items() if k.split("#")[0] == c) for c in RCA_CASES}
    fixed = [c for c in as_written if held[c] >= RCA_MIN_REWORDINGS_FIXED]
    back = [c for c, need in RCA_GUARDS.items()
            if (need == "low" and not outcomes[c]["low"]) or (need == "not credible" and outcomes[c]["credible"])]
    failed = []
    if b["true"]["low"] > RCA_MAX_TRUE_LOW + 1e-9:
        failed.append("true claims rated Low")
    if b["fake"]["low"] < RCA_MIN_FAKE_LOW:
        failed.append("fakes rated Low")
    if b["fake"]["credible"] > RCA_MAX_FAKE_CREDIBLE:
        failed.append("fakes rated Credible")
    if b.get("roc_auc_scored", 0.0) < RCA_MIN_AUC:
        failed.append("ROC-AUC")
    if len(fixed) < RCA_MIN_CASES_FIXED:
        failed.append("known cases fixed")
    if back:
        failed.append("fixed bug back: " + ", ".join(back))
    return {"cases_fixed": fixed, "cases_fixed_as_written": as_written, "rewordings_fixed": held,
            "fixed_bugs_back": back, "failed": failed, "passes": not failed}


def rca_settings() -> list[dict]:
    out = [{"family": "cosine", "match_min": t, "entail_mode": "off", "entail_min": 0.0}
           for t in sorted(set(THRESHOLDS) | set(RCA_COSINE_REFINED)) if t > RCA_BEFORE_MATCH_MIN]
    out += [{"family": "learned", "match_min": t, "entail_mode": "off", "entail_min": 0.0}
            for t in THRESHOLDS if t > 0]
    for fam, mode in (("nli_both", "both"), ("nli_directional", "directional")):
        out += [{"family": fam, "match_min": m, "entail_mode": mode, "entail_min": t}
                for m in RCA_MATCH_GRID for t in RCA_ENTAIL_GRID]
    return out


def rca_choose(results: list[dict]) -> dict | None:
    for fam in RCA_FAMILIES:
        ok = [r for r in results if r["family"] == fam and r["passes"]]
        if ok:
            return max(ok, key=lambda r: (round(r["fake"]["low"], 4), -round(r["true"]["low"], 4),
                                          r.get("roc_auc_scored", 0.0)))
    return None


def _json_safe(x):
    if isinstance(x, dict):
        return {k: _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return None if not math.isfinite(float(x)) else float(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def root_cause_a_dev() -> dict:
    """Every candidate on the dev set and the known cases; the decision by ROOT_CAUSE_A_DECISION."""
    from src.verify import model as vm

    model = models()["deployed"]
    dev, cases, reworded = _rca_rec("dev"), _rca_rec("cases"), _rca_rec("cases_reworded")
    y = np.array(dev["labels"], dtype=float)
    base = {"family": "before", "match_min": RCA_BEFORE_MATCH_MIN, "entail_mode": "off", "entail_min": 0.0}
    s0, _ = _rca_replay(dev, base, model)
    b0 = bands(y, s0)
    c0 = case_outcomes(cases, *_rca_replay(cases, base, model))
    r0 = case_outcomes(reworded, *_rca_replay(reworded, base, model))
    print("  rewordings before:", {k: (o["score"], o["verdicts"]) for k, o in r0.items()})
    print(f"before: true low {b0['true']['low']:.4f} cred {b0['true']['credible']:.3f} none {b0['true']['no_score']:.3f} | "
          f"fake low {b0['fake']['low']:.4f} cred {b0['fake']['credible']:.3f} | AUC {b0['roc_auc_scored']:.4f}", flush=True)
    for cid, o in c0.items():
        print(f"  {cid}: {o['score']} {o['verdicts']}  (recorded by the pipeline: {cases['items'][cases['ids'].index(cid)]['score']})")
    reproduced = (abs(b0["true"]["low"] - RCA_BASELINE["true_low"]) < 1e-6 and abs(b0["fake"]["low"] - RCA_BASELINE["fake_low"]) < 6e-4
                  and abs(b0["fake"]["credible"] - RCA_BASELINE["fake_credible"]) < 1e-6
                  and abs(b0["roc_auc_scored"] - RCA_BASELINE["roc_auc"]) < 6e-4)
    if not reproduced:
        raise SystemExit("the baseline replay does not reproduce the numbers before the change: measurement not trusted")
    matcher, oof = fit_matcher(dev)
    results = []
    for st in rca_settings():
        s, _ = _rca_replay(dev, st, model, oof=oof if st["family"] == "learned" else None)
        b = bands(y, s)
        outcomes = case_outcomes(cases, *_rca_replay(cases, st, model, matcher=matcher))
        outcomes_r = case_outcomes(reworded, *_rca_replay(reworded, st, model, matcher=matcher))
        res = {**st, **b, "cases": outcomes, "cases_reworded": outcomes_r, **rca_check(b, outcomes, outcomes_r)}
        results.append(res)
        print(f"{st['family']:16s} m={st['match_min']:.2f} t={st['entail_min']:.1f}  TRUE low {b['true']['low']:.4f} cred "
              f"{b['true']['credible']:.3f} none {b['true']['no_score']:.3f} | FAKE low {b['fake']['low']:.4f} cred "
              f"{b['fake']['credible']:.3f} | AUC {b.get('roc_auc_scored', float('nan')):.4f} | cases fixed "
              f"{len(res['cases_fixed'])}/5 {res['cases_fixed']} (as written {len(res['cases_fixed_as_written'])}/5; rewordings "
              f"fixed {list(res['rewordings_fixed'].values())}) | {'PASS' if res['passes'] else 'fail: ' + '; '.join(res['failed'])}",
              flush=True)
    chosen = rca_choose(results)
    report = {"decision_rule": ROOT_CAUSE_A_DECISION, "nli_model": vm.ENTAIL_MODEL, "nli_model_licence": "MIT",
              "setting": "India dev set, news and Wikipedia confirm-only, time rule, deployed meta-learner",
              "verdict_passages_dev": len(_verdict_pairs(dev)), "cases_recording": {
                  "model_version": cases.get("model_version"), "live_searches": cases.get("queries")},
              "dev_before": {**base, **b0, "cases": c0, "cases_reworded": r0},
              "rewordings": RCA_REWORDINGS, "min_rewordings_fixed": RCA_MIN_REWORDINGS_FIXED,
              "tightenings": [
                  "after the dev grids: a known case counts as fixed only if it is also fixed for at least "
                  f"{RCA_MIN_REWORDINGS_FIXED} of 3 rewordings (cosine 0.61 fixed three cases by a margin of 0.01)",
                  "after inspecting the learned matcher's cases: fixed also requires that no verdict passage read "
                  "as refuting still counts for the (true) claim; with the learned matcher at 0.80 the Kenya "
                  "birth-certificate fact-check still counted against 'Obama was born in Hawaii'",
                  "added to the cosine grid after the first run: 0.61-0.64 (the boundary lay between 0.60 and 0.65)"],
              "passes_rule_as_first_written": [
                  {k: r[k] for k in ("family", "match_min", "entail_mode", "entail_min")} for r in results
                  if not [f for f in r["failed"] if f != "known cases fixed"]
                  and sum(not o["refuted"] and not o["low"] for c, o in r["cases"].items() if c in RCA_CASES)
                  >= RCA_MIN_CASES_FIXED],
              "learned_matcher": {k: v for k, v in matcher.items() if k != "trained_on"},
              "dev_candidates": results, "chosen": chosen, "ship": chosen is not None}
    if chosen is not None and chosen["family"] == "learned":
        report["chosen_matcher"] = matcher
    RCA_REPORT.write_text(json.dumps(_json_safe(report), indent=2))
    print("decision:", "SHIP " + json.dumps({k: chosen[k] for k in ("family", "match_min", "entail_mode", "entail_min")})
          if chosen else "NO SHIP (no candidate passes)", "| saved", RCA_REPORT)
    return report


def root_cause_a_test() -> dict:
    """After the decision: the test, matched and hoax sets, once each, before and after. Without a passing
    candidate nothing ships: "before" is then what stays served, and "after" is the hypothesis at its plain
    setting (RCA_HYPOTHESIS), for information only."""
    report = json.loads(RCA_REPORT.read_text())
    if "test_sets" in report:
        raise SystemExit("the test sets were already scored once for this decision (reports/root_cause_a_eval.json)")
    st = report["chosen"] or RCA_HYPOTHESIS
    before = report["dev_before"]
    model = models()["deployed"]
    matcher = report.get("chosen_matcher") or (fit_matcher(_rca_rec("dev"))[0] if st["family"] == "learned" else None)
    out = {"scored_with": {k: st[k] for k in ("family", "match_min", "entail_mode", "entail_min")},
           "is_the_shipped_setting": bool(report["ship"])}
    for name in ("test", "matched", "hoax"):
        rec = _rca_rec(name, compute=True)
        y = np.array(rec["labels"], dtype=float)
        entry = {}
        for when, setting in (("before", before), ("after", st)):
            s, _ = _rca_replay(rec, setting, model, matcher=matcher)
            strict, _ = _rca_replay(drop_same_claim(rec), setting, model, matcher=matcher)
            entry[when] = {**bands(y, s), "strict_no_earlier_factcheck_of_same_claim": bands(y, strict)}
            if name == "hoax":
                split = _hoax_split("hoax")
                entry[when]["by_split"] = {part: bands(y[split == part], s[split == part]) for part in ("dev", "test")}
        entry["own_claim_in_evidence"] = own_claim_share(rec)
        out[name] = entry
        b, a = entry["before"], entry["after"]
        if name != "hoax":
            print(f"{name}: true low {b['true']['low']:.4f} -> {a['true']['low']:.4f}, cred {b['true']['credible']:.4f} -> "
                  f"{a['true']['credible']:.4f}, none {b['true']['no_score']:.4f} -> {a['true']['no_score']:.4f} | fake low "
                  f"{b['fake']['low']:.4f} -> {a['fake']['low']:.4f}, cred {b['fake']['credible']:.4f} -> "
                  f"{a['fake']['credible']:.4f} | AUC {b.get('roc_auc_scored', float('nan')):.4f} -> "
                  f"{a.get('roc_auc_scored', float('nan')):.4f} | strict fake low "
                  f"{b['strict_no_earlier_factcheck_of_same_claim']['fake']['low']:.4f} -> "
                  f"{a['strict_no_earlier_factcheck_of_same_claim']['fake']['low']:.4f}", flush=True)
        else:
            print(f"hoax: low {b['fake']['low']:.4f} -> {a['fake']['low']:.4f}, cred {b['fake']['credible']:.4f} -> "
                  f"{a['fake']['credible']:.4f}, none {b['fake']['no_score']:.4f} -> {a['fake']['no_score']:.4f}", flush=True)
    report["test_sets"] = out
    RCA_REPORT.write_text(json.dumps(_json_safe(report), indent=2))
    print("saved", RCA_REPORT)
    return report


def _load(name: str) -> dict:
    with open(CACHE / f"{name}.pkl", "rb") as f:
        return pickle.load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("record", "choose", "report", "augment", "absence", "absence-recheck", "absence-hoax", "time-rule",
                 "root-cause-a"):
        p = sub.add_parser(name)
        if name == "root-cause-a":
            p.add_argument("--stage", choices=["cases", "entail", "dev", "served", "test"], required=True,
                           help="cases: record the known cases (live lookups); entail: entailment values for the "
                                "recordings; dev: measure every candidate and decide; served: the known cases "
                                "through the served code (from the lookup caches); test: score the test sets once")
        p.add_argument("--sources", nargs="*", default=[], choices=["news", "wiki"],
                       help="supporting-evidence lookups (live fact-check lookup is always on)")
        if name in ("record", "augment"):
            p.add_argument("--set", choices=list(SETS), required=True)
        if name == "augment":
            p.add_argument("--on", nargs="*", default=[], choices=["news", "wiki"],
                           help="sources already in the base recording")
            p.add_argument("--only-expected", action="store_true",
                           help="news: only items with a claim the news would cover (saves searches)")
            p.add_argument("--twice", action="store_true",
                           help="news: search a covered claim again with its keywords if the first search heard nothing")
    args = ap.parse_args()
    suffix = suffix_of(args.sources)
    if args.cmd == "record":
        print("saved", record(args.set, args.sources))
    elif args.cmd == "augment":  # e.g. augment --set dev --sources wiki: dev.pkl -> dev_wiki.pkl
        on = list(args.on)
        for s in args.sources:
            print("saved", augment(args.set, s, on=on, only_expected=args.only_expected, twice=args.twice))
            on = sorted(set(on) | {s})
    elif args.cmd == "absence":  # absence --sources news wiki -> reports/absence_eval_news_wiki.json
        absence_report(args.sources)
    elif args.cmd == "absence-recheck":  # -> reports/absence_recheck_news_wiki.json (no searches spent)
        silence_recheck_report(args.sources)
    elif args.cmd == "absence-hoax":  # -> reports/absence_hoax_eval.json (spends Tavily searches: second queries)
        silence_hoax_report(args.sources)
    elif args.cmd == "time-rule":
        time_rule_report()
    elif args.cmd == "root-cause-a":
        {"cases": record_cases, "entail": entail_stage, "dev": root_cause_a_dev,
         "served": lambda: record_cases(save=False), "test": root_cause_a_test}[args.stage]()
    elif args.cmd == "choose":
        dev = _load(f"dev{suffix}")
        matcher, oof = fit_matcher(dev)
        print(f"learned matcher: CV AUC {matcher['cv_auc']:.3f} vs cosine {matcher['cosine_auc']:.3f} "
              f"(n={matcher['n_train']}); coef {dict(zip(matcher['features'], np.round(matcher['coef'], 2)))}")
        res = grid({"cosine": dev, "learned": rematch(dev, probs_by_key=oof)})
        for x in res:
            print(f"{x['matcher']:7s} {x['meta']:14s} t={x['match_min']:.2f}  TRUE low {x['true']['low']:.2f} "
                  f"cred {x['true']['credible']:.2f} none {x['true']['no_score']:.2f} | FAKE low {x['fake']['low']:.2f} "
                  f"cred {x['fake']['credible']:.2f} none {x['fake']['no_score']:.2f} | AUC {x.get('roc_auc_scored', float('nan')):.3f}")
        # One setting (matcher + threshold) for the deployed meta-learner, chosen on dev only.
        best = {}
        for m in ("deployed",):
            simple = choose([r for r in res if r["meta"] == m and r["matcher"] == "cosine"])
            learned = choose([r for r in res if r["meta"] == m and r["matcher"] == "learned"])
            # Simpler wins ties: the learned matcher only if it rates >= 1 point more fakes Low under the targets.
            ok = learned["true"]["low"] <= MAX_TRUE_LOW and learned["fake"]["credible"] <= MAX_FAKE_CREDIBLE
            best[m] = learned if ok and learned["fake"]["low"] >= simple["fake"]["low"] + 0.01 else simple
        for m, b in best.items():
            print(f"chosen for {m}: {b['matcher']} matcher, match_min {b['match_min']}")
        CACHE.mkdir(parents=True, exist_ok=True)
        (CACHE / f"chosen{suffix}.json").write_text(json.dumps({"deployed_meta": "deployed", "best": best,
                                                                "matcher": matcher, "dev_grid": res}, indent=1))
    elif args.cmd == "report":
        chosen = json.loads((CACHE / f"chosen{suffix}.json").read_text())
        ms = models()

        def variant(r: dict, b: dict) -> dict:
            return rematch(r, model=chosen["matcher"]) if b.get("matcher") == "learned" else r

        rec = _load(f"test{suffix}")
        y = np.array(rec["labels"], dtype=float)
        report = {"rule": f"threshold chosen on the dev set: true rated Low <= {MAX_TRUE_LOW}, fakes rated Credible <= "
                          f"{MAX_FAKE_CREDIBLE}, then most fakes rated Low; test set scored once",
                  "deployed_meta": chosen["deployed_meta"],
                  "dev": {m: b for m, b in chosen["best"].items()},
                  "test_baseline_before_fix": bands(y, replay(rec, OFF, ms["deployed"])),
                  "test": {m: {"matcher": b.get("matcher", "cosine"), "match_min": b["match_min"],
                               **bands(y, replay(variant(rec, b), b["match_min"], ms[m]))}
                           for m, b in chosen["best"].items()},
                  "matcher": {k: v for k, v in chosen.get("matcher", {}).items() if k != "trained_on"},
                  "own_claim_in_evidence": own_claim_share(rec),
                  "strict_no_earlier_factcheck_of_same_claim": {
                      m: bands(y, replay(variant(drop_same_claim(rec), b), b["match_min"], ms[m]))
                      for m, b in chosen["best"].items()},
                  "ablation_sources_may_refute": None if not rec.get("sources") else {
                      m: bands(y, replay(variant(rec, b), b["match_min"], ms[m], support_only=False))
                      for m, b in chosen["best"].items()},
                  "supporting_sources": rec.get("sources", []), "queries": rec.get("queries")}
        if (CACHE / f"matched{suffix}.pkl").exists():  # second test set: fact-checker-written true claims
            mrec = _load(f"matched{suffix}")
            my = np.array(mrec["labels"], dtype=float)
            report["matched_set"] = {
                "about": "true AND false claims written by Indian fact-checkers, paired by publisher and year",
                "own_claim_in_evidence": own_claim_share(mrec),
                "strict_no_earlier_factcheck_of_same_claim": {
                    m: bands(my, replay(variant(drop_same_claim(mrec), b), b["match_min"], ms[m]))
                    for m, b in chosen["best"].items()},
                "baseline_before_fix": bands(my, replay(mrec, OFF, ms["deployed"])),
                **{m: {"matcher": b.get("matcher", "cosine"), "match_min": b["match_min"],
                       **bands(my, replay(variant(mrec, b), b["match_min"], ms[m]))}
                   for m, b in chosen["best"].items()}}
        path = REPORT.with_name(f"india_fix_eval{suffix}.json")
        path.write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
