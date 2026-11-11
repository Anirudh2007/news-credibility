"""India fixes (2026-11-03): claim matching (a verdict on a different claim does not count) and news
corroboration (GDELT, faked)."""

import math

import numpy as np

from src.claims.extract import HeuristicCheckWorthiness
from src.pipeline import NewsItem, Pipeline
from src.retrieval import news_search as ns
from src.retrieval.search import Hit, Passage
from src.verify.model import ClaimMatcher, ClaimVerifier, claim_behind, verdict_from

REFUTE = [0.02, 0.9, 0.05, 0.03]
SUPPORT = [0.9, 0.02, 0.05, 0.03]


def fc_passage(claim, answer="No"):
    return Passage("fc1", f"Q: Is it true that {claim}? A: {answer}. factly.in rated this claim False.", None, "factly.in")


def toy_matcher():
    """Bag-of-words embedding: similarity = shared words, so the tests need no model download."""
    vocab = {}

    def enc(texts):
        out = np.zeros((len(texts), 64))
        for i, t in enumerate(texts):
            for w in t.lower().split():
                out[i, vocab.setdefault(w, len(vocab) % 64)] += 1
        return out

    return ClaimMatcher(encoder=enc, model=None)  # plain cosine, whatever models/claim_matcher.json holds


def test_claim_behind_reads_factcheck_and_skips_news():
    assert claim_behind(fc_passage("images show an IED blast in Kanker")) == "images show an IED blast in Kanker"
    assert claim_behind(Passage("news:x", "thehindu.com reported (2023-04-26): Ten killed in Dantewada blast")) is None


def test_verdict_on_a_different_claim_reads_as_nee():
    hits = [Hit(fc_passage("images show an IED blast in Kanker district"), 3.0, 1, 1)]
    off = verdict_from(hits, np.array([REFUTE]), [0.2], [0.5], match_min=0.6)
    assert math.isnan(off.score) and off.label == "Not Enough Evidence" and math.isnan(off.relevance)
    assert off.evidence[0].counted is False
    on = verdict_from(hits, np.array([REFUTE]), [0.9], [0.5], match_min=0.6)
    assert on.label == "Refuted" and on.score < 0.1 and on.evidence[0].counted


def test_wikipedia_and_news_can_confirm_but_not_refute():
    wiki_hit = Hit(Passage("wiki:Page#0", "Wikipedia, Page (as of 2023-04-27): Ten killed in blast.", None,
                           "en.wikipedia.org"), 2.0, None, None)
    refuting = verdict_from([wiki_hit], np.array([REFUTE]), [np.nan], [0.5])
    assert math.isnan(refuting.score) and refuting.label == "Not Enough Evidence"  # its "Refuted" is not trusted
    supporting = verdict_from([wiki_hit], np.array([SUPPORT]), [np.nan], [0.5])
    assert supporting.label == "Supported" and supporting.score == 1.0
    ablation = verdict_from([wiki_hit], np.array([REFUTE]), [np.nan], [0.5], support_only=())
    assert ablation.label == "Refuted"


def test_matching_off_equals_old_aggregation():
    hits = [Hit(fc_passage("x"), 1.0, 1, 1), Hit(Passage("p", "Q: a A: b"), 0.0, 2, 2)]
    probs = np.array([REFUTE, SUPPORT])
    a = verdict_from(hits, probs, [np.nan, np.nan], [0.5, 0.5])
    b = verdict_from(hits, probs, [0.1, 0.1], [0.5, 0.5], match_min=-1.0)
    assert a.score == b.score and a.relevance == b.relevance


def test_verifier_demotes_unmatched_factcheck_but_keeps_news():
    v = ClaimVerifier(predict_fn=lambda c, ps: np.array([REFUTE if " A: No." in p else SUPPORT for p in ps]),
                      reliability={}, matcher=toy_matcher(), match_min=0.6)
    claim = "Maoists kill eight police officers in Bijapur blast"
    hits = [Hit(fc_passage("viral video shows dance at a wedding in Goa"), 2.0, 1, 1),
            Hit(Passage("news:1", "thehindu.com reported (2023-04-26): Maoists kill eight police officers in Bijapur blast",
                        None, "thehindu.com"), 1.0, None, None)]
    verdict = v.verify(claim, hits)
    assert verdict.label == "Supported" and verdict.score > 0.9  # only the news report counted
    assert [e.counted for e in verdict.evidence] == [True, False]


# --- Wikipedia as of the claim date ------------------------------------------------------------------

from src.retrieval import wiki_search as ws  # noqa: E402

WIKITEXT = """{{Infobox event|name=Dantewada bombing}}
The '''2023 Dantewada bombing''' was an attack in which [[Communist Party of India (Maoist)|Maoists]] killed ten
police officers and a driver near Aranpur in [[Dantewada district]], Chhattisgarh, on 26 April 2023.<ref>{{cite web|x}}</ref>
== Aftermath ==
The Chief Minister condemned the attack and said the security forces would continue operations.
"""


def wiki_fetch(calls):
    def fetch(params):
        calls.append(params)
        if params.get("list") == "search":
            return {"query": {"search": [{"title": "2023 Dantewada bombing"}, {"title": "Portal:Current events"}]}}
        if params["rvstart"] < "2023-04-26":
            return {"query": {"pages": [{"title": params["titles"], "missing": True}]}}  # page did not exist yet
        return {"query": {"pages": [{"title": params["titles"], "revisions": [
            {"timestamp": "2023-04-27T10:00:00Z", "slots": {"main": {"content": WIKITEXT}}}]}]}}
    return fetch


def test_wikitext_to_sentences_strips_markup():
    s = ws.wikitext_to_sentences(WIKITEXT)
    assert s[0].startswith("The 2023 Dantewada bombing was an attack in which Maoists killed ten police officers")
    assert not any("{{" in x or "[[" in x or "<ref" in x for x in s)


def test_wiki_uses_revision_as_of_claim_date_and_skips_portal():
    calls = []
    c = ws.WikiSearchClient(cache_path=None, fetch=wiki_fetch(calls))
    ps = c.search("Maoists kill ten police officers in Dantewada, Chhattisgarh", before="2023-04-27")
    assert ps and ws.parse_wiki_passage(ps[0].text)["page"] == "2023 Dantewada bombing"
    assert {p["titles"] for p in calls if "titles" in p} == {"2023 Dantewada bombing"}  # portal never fetched
    assert [p["rvstart"] for p in calls if "rvstart" in p] == ["2023-04-27T23:59:59Z"]
    assert c.search("Maoists kill ten police officers in Dantewada, Chhattisgarh", before="2023-04-20") == []


# --- news search ---------------------------------------------------------------------------------

ARTICLES = [
    {"title": "Chandrayaan - 3 lands on Moon ' s south pole : ISRO", "domain": "thehindu.com",
     "url": "https://thehindu.com/a", "seendate": "20230823T130000Z"},
    {"title": "Chandrayaan 3 landing LIVE", "domain": "unknown-blog.xyz", "url": "https://x.xyz/b", "seendate": "20230823T120000Z"},
    {"title": "Chandrayaan-3 one week after landing", "domain": "reuters.com", "url": "https://reuters.com/c",
     "seendate": "20230830T090000Z"},
]


def news_client(articles=ARTICLES):
    calls = []
    c = ns.NewsSearchClient(reliability={"thehindu.com": 0.75, "reuters.com": 0.9}, cache_path=None,
                            fetch=lambda p: calls.append(p) or articles)
    return c, calls


def test_clean_title():
    assert ns.clean_title("Chandrayaan - 3 lands on Moon ' s south pole : ISRO") == "Chandrayaan-3 lands on Moon's south pole: ISRO"


def test_news_keeps_trusted_outlets_up_to_the_claim_date():
    c, calls = news_client()
    arts = c.articles("Chandrayaan-3 lands near the Moon's south pole", before="2023-08-23")
    assert [a["domain"] for a in arts] == ["thehindu.com"]  # blog untrusted, Reuters piece came later
    assert calls[0]["enddatetime"] == "20230823235959" and calls[0]["startdatetime"] == "20230816000000"
    p = c.search("Chandrayaan-3 lands near the Moon's south pole", before="2023-08-23")[0]
    assert ns.parse_news_passage(p.text)["outlet"] == "thehindu.com" and p.id.startswith(ns.ID_PREFIX)


def test_news_off_by_default(monkeypatch):
    monkeypatch.setattr("src.retrieval.factcheck_api.load_env_file", lambda names=(): None)  # ignore a real .env key
    monkeypatch.delenv("NEWSCRED_NEWS_SEARCH", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    assert ns.default_client() is None
    monkeypatch.setenv("NEWSCRED_NEWS_SEARCH", "1")  # the old switch still means GDELT
    assert ns.default_client() is not None and ns.default_client().backend == "gdelt"


class EmptySearcher:
    def search(self, query, k=5, before=None, reranker=None):
        return []


class FactCheckSearcher:
    def search(self, query, k=5, before=None, reranker=None):
        return [Hit(fc_passage("viral video shows a dance at a wedding in Goa"), 3.0, 1, 1)]


def test_no_fact_check_badge_or_score_from_a_different_claim():
    v = ClaimVerifier(predict_fn=lambda c, ps: np.array([REFUTE for _ in ps]), reliability={},
                      matcher=toy_matcher(), match_min=0.6)
    pipe = Pipeline(components=dict(scorer=HeuristicCheckWorthiness(), searcher=FactCheckSearcher(), verifier=v,
                                    reranker=None, factcheck=None, news=None, meta=None), autoload=False)
    rep = pipe.analyze(NewsItem(text="Maoists killed eight police officers in a blast in Bijapur, Chhattisgarh in April 2023."))
    claim = rep["claims"][0]
    assert claim["fact_checks"] == [] and claim["credibility"] is None
    assert claim["evidence"][0]["counted"] is False and rep["score"] is None  # abstains instead of rating it Low


def test_pipeline_scores_a_true_claim_from_news_reports():
    news, _ = news_client()
    predict = lambda c, ps: np.array([SUPPORT if "reported" in p else REFUTE for p in ps])  # noqa: E731
    pipe = Pipeline(components=dict(scorer=HeuristicCheckWorthiness(), searcher=EmptySearcher(),
                                    verifier=ClaimVerifier(predict_fn=predict, reliability={}), reranker=None,
                                    factcheck=None, news=news, meta=None), autoload=False)
    rep = pipe.analyze(NewsItem(text="ISRO's Chandrayaan-3 landed near the south pole of the Moon on 23 August 2023.",
                                publish_date="2023-08-23"))
    assert rep["claims"][0]["verdict"] == "Supported" and rep["score"] is not None and rep["score"] > 60
    assert rep["claims"][0]["evidence"][0]["source"] == "thehindu.com"


def test_claim_matching_works_without_processed_data(tmp_path, monkeypatch):
    """Docker / server: no data/processed/, so the claim behind an index answer comes from the exported map."""
    import json as _json

    from src.verify import model as vm

    m = tmp_path / "averitec_claims.json"
    m.write_text(_json.dumps({"123": "Cricketer Yasir Shah died in the Karachi plane crash"}))
    monkeypatch.setattr(vm, "AVERITEC_PATH", tmp_path / "missing.parquet")
    monkeypatch.setattr(vm, "AVERITEC_CLAIMS_JSON", m)
    monkeypatch.setattr(vm, "_averitec_claims", None)
    assert claim_behind(Passage("123_q0_a1", "Q: Did Yasir Shah die? A: No.")) == "Cricketer Yasir Shah died in the Karachi plane crash"
