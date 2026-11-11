"""Live fact-check evidence (Google Fact Check Tools API, faked), the "previously fact-checked"
badge, the "What the evidence says" box and the background location check."""

import math

import numpy as np

from src.claims.extract import HeuristicCheckWorthiness
from src.explain.generator import evidence_box
from src.media.reuse import LocationReference
from src.pipeline import NewsItem, Pipeline, published_fact_checks
from src.retrieval import factcheck_api as fc
from src.retrieval.search import Hit, Passage
from src.verify.model import ClaimVerifier

BODY = {"claims": [
    {"text": "New Rs 2000 notes have a GPS nano chip", "claimant": "WhatsApp forward", "claimDate": "2016-11-09T00:00:00Z",
     "claimReview": [
         {"publisher": {"name": "BOOM", "site": "boomlive.in"}, "url": "https://www.boomlive.in/gps-chip",
          "title": "No, the Rs 2000 note does not have a GPS chip", "reviewDate": "2016-11-12T00:00:00Z",
          "textualRating": "False", "languageCode": "en"},
         {"publisher": {"name": "Vishvas News", "site": "vishvasnews.com"}, "url": "https://www.vishvasnews.com/x",
          "reviewDate": "2016-11-20T00:00:00Z", "textualRating": "भ्रामक", "languageCode": "hi"},  # Hindi: skipped
     ]},
    {"text": "ISRO launched Chandrayaan-3 on 14 July 2023", "claimReview": [
        {"publisher": {"name": "Factly", "site": "factly.in"}, "url": "https://factly.in/c3",
         "reviewDate": "2023-07-20T00:00:00Z", "textualRating": "True", "languageCode": "en"}]},
]}


def test_rating_to_credible():
    assert fc.rating_to_credible("False") == 0.0
    assert fc.rating_to_credible("Misleading") == 0.0
    assert fc.rating_to_credible("Not true") == 0.0  # false is tested before true
    assert fc.rating_to_credible("Partly False") == 0.5  # partly is tested before false
    assert fc.rating_to_credible("Missing Context") == 0.5
    assert fc.rating_to_credible("Mostly True") == 1.0 and fc.rating_to_credible("Correct") == 1.0
    assert fc.rating_to_credible("Satire") is None and fc.rating_to_credible("") is None


def test_parse_response_keeps_english_verdicts():
    out = fc.parse_response(BODY)
    assert [f.publisher for f in out] == ["BOOM", "Factly"]
    p = out[0].passage()
    assert p.text.startswith("Q: Is it true that New Rs 2000 notes have a GPS nano chip? A: No. BOOM rated this claim False.")
    assert p.domain == "boomlive.in" and p.date == "2016-11-12" and p.id.startswith(fc.ID_PREFIX)
    info = fc.parse_factcheck_passage(p.text)
    assert info["who"] == "BOOM" and info["rating"] == "False" and info["answer"] == "No"


def client(body=BODY, **kw):
    calls = []

    def fetch(params):
        calls.append(params)
        if isinstance(body, Exception):
            raise body
        return body

    c = fc.FactCheckClient(api_key="test", cache_path=None, fetch=fetch, **kw)
    return c, calls


def test_search_drops_reviews_written_after_the_claim():
    c, _ = client()
    assert len(c.search("gps chip in 2000 note")) == 2
    assert [p.domain for p in c.search("gps chip in 2000 note", before="2016-12-31")] == ["boomlive.in"]
    assert c.search("gps chip in 2000 note", before="2016-11-10") == []  # both reviews came later


def test_undated_reviews_do_not_count_for_a_dated_claim():
    body = {"claims": [{"text": "A viral claim", "claimReview": [
        {"publisher": {"name": "X", "site": "x.in"}, "url": "https://x.in/a", "textualRating": "False", "languageCode": "en"}]}]}
    c, _ = client(body)
    assert len(c.search("a viral claim")) == 1  # no date given (the app): kept
    assert c.search("a viral claim", before="2024-01-01") == []  # dated claim: cannot show the review came first


def test_cache_and_circuit_breaker(tmp_path):
    calls = []
    c = fc.FactCheckClient(api_key="t", cache_path=tmp_path / "c.sqlite", fetch=lambda p: calls.append(p) or BODY)
    c.search("a claim")
    c.search("a claim")
    assert len(calls) == 1 and c.cache_hits == 1  # the second answer came from SQLite

    bad, calls = client(TimeoutError("slow"))
    for _ in range(4):
        assert bad.search("another claim") == []
    assert len(calls) == fc.BREAKER_FAILURES and "paused" in bad.status  # stopped calling after 2 failures


def test_merge_reranks_live_with_index_hits():
    c, _ = client()
    index_hit = Hit(Passage("i1", "Unrelated passage about rainfall.", "https://x.org/a", "x.org"), -3.0, 1, 1)
    rerank = lambda q, texts: [4.0 if "GPS" in t else -5.0 for t in texts]  # noqa: E731
    merged = c.merge("Rs 2000 note has a GPS chip", [index_hit], reranker=rerank, k=2)
    assert merged[0].passage.domain == "boomlive.in" and merged[1] is index_hit
    badges = published_fact_checks(merged)
    assert [b["publisher"] for b in badges] == ["BOOM"] and badges[0]["rating"] == "False"  # rainfall not a fact-check


def test_keyword_fallback_when_the_sentence_finds_nothing():
    calls = []

    def fetch(params):
        calls.append(params["query"])
        return BODY if len(params["query"].split()) <= 6 else {}

    c = fc.FactCheckClient(api_key="t", cache_path=None, fetch=fetch)
    assert c.search("The new Rs 2000 note contains a GPS nano chip that lets the government track it")
    assert calls == ["The new Rs 2000 note contains a GPS nano chip that lets the government track it",
                     "Rs 2000 note contains GPS nano"]


def test_default_client_off_switch(monkeypatch):
    monkeypatch.setenv("NEWSCRED_LIVE_FACTCHECK", "0")
    assert fc.default_client() is None


class EmptySearcher:
    def search(self, query, k=5, before=None, reranker=None):
        return []


def test_pipeline_uses_live_fact_check_and_fills_badge_and_box():
    live, _ = client()

    def predict(claim, passages):  # NLI stand-in: a "No." answer refutes
        return np.array([[0.02, 0.9, 0.05, 0.03] if " A: No." in p else [0.1, 0.1, 0.7, 0.1] for p in passages])

    pipe = Pipeline(components=dict(scorer=HeuristicCheckWorthiness(), searcher=EmptySearcher(),
                                    verifier=ClaimVerifier(predict_fn=predict, reliability={}), reranker=None,
                                    factcheck=live, meta=None), autoload=False)
    rep = pipe.analyze(NewsItem(text="The new Rs 2000 note contains a GPS nano chip, says a viral message.",
                                publish_date="2017-01-05"))
    claim = rep["claims"][0]
    assert claim["verdict"] == "Refuted" and claim["fact_checks"][0]["publisher"] == "BOOM"
    assert rep["evidence_box"] and rep["evidence_box"][0]["source"] == "boomlive.in"
    assert rep["evidence_box"][0]["quote"].startswith("Q: Is it true that")
    assert rep["score"] is not None and rep["score"] <= 30


def test_evidence_box_only_for_confident_on_topic_verdicts():
    ev = [{"source": "bbc.co.uk", "url": "https://bbc.co.uk/a", "snippet": "The minister denied it.", "date": "2023-01-02"}]
    base = {"text": "c", "verdict": "Refuted", "confidence": 0.8, "evidence_relevance": 0.9, "evidence": ev}
    assert len(evidence_box([base])) == 1 and evidence_box([base])[0]["quote"] == "The minister denied it."
    assert evidence_box([{**base, "verdict": "Not Enough Evidence"}]) == []
    assert evidence_box([{**base, "confidence": 0.4}]) == []
    assert evidence_box([{**base, "evidence_relevance": 0.2}]) == []
    assert evidence_box([{**base, "evidence_off_topic": True}]) == []
    assert evidence_box([{**base, "evidence": []}]) == []


def test_location_reference_finds_places_named_in_text():
    loc = LocationReference(lambda imgs: np.zeros((len(imgs), 4)), refs={"india_gate": np.ones((1, 4)),
                                                                          "jantar_mantar": np.ones((1, 4))})
    assert loc.find(["Delhi", "Jantar Mantar"]) == "Jantar Mantar"
    assert loc.find(["India Gate, New Delhi"]) == "india gate"
    assert loc.find(["Mumbai"]) is None and loc.find([]) is None
    assert math.isnan(loc.similarity(np.ones(4), "Mumbai"))  # no reference photos: check silently not run
