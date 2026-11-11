"""Silence as evidence (2026-11-05): claim time, the time rule for support, expected coverage, the Bayes update,
the Tavily news search (faked) and the evidence memory. No network, no model downloads."""

import math
from datetime import date

import numpy as np
import pytest

from src.claims.extract import HeuristicCheckWorthiness
from src.pipeline import NewsItem, Pipeline
from src.retrieval import coverage as cov
from src.retrieval import news_search as ns
from src.retrieval.evidence_memory import EvidenceMemory
from src.retrieval.search import Hit, Passage
from src.verify.model import ClaimVerifier, verdict_from

SUPPORT = [0.9, 0.02, 0.05, 0.03]
REFUTE = [0.02, 0.9, 0.05, 0.03]
NEUTRAL = [0.05, 0.05, 0.85, 0.05]
TODAY = date(2026, 11, 5)
# For the pipeline tests on a claim about "tomorrow": its news window is counted back from the day the tests run,
# so the faked report must be dated relative to that day (a fixed 2026-11-05 fell out of the window on 12 Nov).
RECENT = date.today().isoformat()


def hit(pid, text, d=None, score=2.0, domain="x.org"):
    return Hit(Passage(pid, text, f"https://{domain}/{pid}", domain, d), score, 1, 1)


# --- claim time --------------------------------------------------------------------------------------

def test_claim_time_priorities():
    assert cov.claim_time("Protest in Delhi", claimed_date="2024-03-05").day == date(2024, 3, 5)
    assert cov.claim_time("x", dates=["2024-03-09", "2024-03-01"]).day == date(2024, 3, 9)
    assert cov.claim_time("Chandrayaan-3 landed in August 2023").month == (2023, 8)
    assert cov.claim_time("GDP grew 7% in 2022").year == 2022 and cov.claim_time("GDP grew 7% in 2022").day is None
    assert not cov.claim_time("Chandrayaan-3 landed on the Moon").known


def test_claim_time_current_words():
    t = cov.claim_time("RBI says Rs 500 notes banned from tomorrow", today=TODAY)
    assert t.current and t.day == date(2026, 11, 6)
    t = cov.claim_time("Shah Rukh Khan died this morning", publish_date="2024-01-10", today=TODAY)
    assert t.current and t.day == date(2024, 1, 10)


def test_news_window():
    t = cov.ClaimTime(day=date(2023, 8, 23), month=(2023, 8), year=2023)
    assert cov.news_window(t, before="2023-08-23", today=TODAY) == (date(2023, 8, 16), date(2023, 8, 23))
    assert cov.news_window(t, before=None, today=TODAY) == (date(2023, 8, 16), date(2023, 8, 24))
    assert cov.news_window(cov.ClaimTime(), before=None, today=TODAY) == (None, TODAY)  # undated: all reports so far
    assert cov.news_window(cov.ClaimTime(), before="2023-08-23", today=TODAY) == (date(2023, 8, 16), date(2023, 8, 23))
    assert cov.news_window(cov.ClaimTime(month=(2023, 2), year=2023), today=TODAY) == (date(2023, 2, 1), date(2023, 2, 28))
    assert cov.news_window(cov.ClaimTime(year=2022), before="2022-06-30", today=TODAY) == (date(2022, 1, 1), date(2022, 6, 30))


# --- time rule for support ---------------------------------------------------------------------------

def test_wikipedia_must_name_the_claims_time():
    t = cov.claim_time("RBI announces Rs 500 notes banned from tomorrow", today=TODAY)
    old = hit("wiki:Demonetisation#0", "Wikipedia, Demonetisation (as of 2026-11-04): In November 2016 the government "
                                       "withdrew Rs 500 and Rs 1000 notes.", "2026-11-04")
    new = hit("wiki:RBI#0", "Wikipedia, RBI (as of 2026-11-05): On 5 November 2026 the RBI announced ...", "2026-11-05")
    news = hit("news:a", "thehindu.com reported (2016-11-08): Rs 500 notes banned", "2016-11-08")
    assert cov.time_mismatch_ids([old, new, news], t) == {"wiki:Demonetisation#0"}


def test_old_dated_passages_cannot_confirm_a_dated_claim():
    t = cov.ClaimTime(day=date(2024, 3, 9), month=(2024, 3), year=2024)
    hits = [hit("q1", "Q: ... A: Yes.", "2021-05-01"), hit("q2", "Q: ... A: Yes.", "2024-03-01"), hit("q3", "Q: A: Yes.")]
    assert cov.time_mismatch_ids(hits, t) == {"q1"}
    assert cov.time_mismatch_ids(hits, cov.ClaimTime()) == set()  # undated claim: no time rule


def test_verdict_no_support_moves_support_to_nee_but_keeps_refutation():
    wiki = hit("wiki:P#0", "Wikipedia, P (as of 2026-11-04): ...", "2026-11-04")
    v = verdict_from([wiki], np.array([SUPPORT]), [math.nan], [0.9])
    assert v.label == "Supported" and v.score == 1.0  # one confirming Wikipedia passage = full credibility
    v = verdict_from([wiki], np.array([SUPPORT]), [math.nan], [0.9], no_support={"wiki:P#0"})
    assert math.isnan(v.score) and v.evidence[0].counted is False
    old = hit("q1", "Q: ... A: No.", "2017-01-01")
    v = verdict_from([old], np.array([REFUTE]), [math.nan], [0.9], no_support={"q1"})
    assert v.label == "Refuted"  # an old debunk of a recurring rumour still refutes


# --- expected coverage -------------------------------------------------------------------------------

def test_expected_coverage_cases():
    well_known = {"Shah Rukh Khan"}.__contains__
    e = cov.expected_coverage("Shah Rukh Khan died in a car accident in Mumbai this morning.", [("Khan", "PERSON")], well_known)
    assert e.expected and e.subject == "Shah Rukh Khan" and e.event == "a death"
    assert cov.expected_coverage("RBI announces that Rs 500 notes will be banned from tomorrow.").expected
    assert cov.expected_coverage("The Government will give Rs 5,000 to every citizen from 1 December.").expected
    assert cov.expected_coverage("Drinking cow urine cures cancer.").expected  # extraordinary
    assert not cov.expected_coverage("A man in a village in Bihar rescued a stray dog from a well.").expected
    assert not cov.expected_coverage("A woman died after taking the vaccine in Pune.").expected  # local story
    assert not cov.expected_coverage("Ramesh Kumar died in an accident.", [("Ramesh Kumar", "PERSON")], well_known).expected
    assert not cov.expected_coverage("The man who won the lottery gave it to us.").expected  # "who", "us": not WHO, US
    assert cov.expected_coverage("The WHO declared a new pandemic.").expected
    assert not cov.expected_coverage("THE MAN WHO DIED WAS MY NEIGHBOUR AND US").expected  # capitals are not acronyms


def test_candidate_names_widen_spacy_entities():
    assert cov.candidate_names("Shah Rukh Khan died in Mumbai.", [("Khan", "PERSON")])[0] == "Shah Rukh Khan"


def test_bayes_update():
    assert cov.absence_probability(None, 0.25) == pytest.approx(0.2)  # even odds x 0.25 -> 1:4
    assert cov.absence_probability(0.8, 0.25) == pytest.approx(0.5)  # odds 4 x 0.25 = 1
    assert cov.absence_probability(0.5, 1.0) == pytest.approx(0.5)  # LR 1: silence says nothing
    lr = cov.likelihood_ratio(found_true=80, n_true=100, found_fake=0, n_fake=100)
    assert lr == pytest.approx((1 - 81 / 102) / (1 - 1 / 102))


# --- Tavily news search (faked) ----------------------------------------------------------------------

TAVILY = [
    {"title": "Chandrayaan-3 lands on Moon's south pole", "url": "https://www.thehindu.com/a",
     "published_date": "Wed, 23 Aug 2023 13:00:00 GMT", "content": "ISRO's lander touched down at 6.04 pm."},
    {"title": "Chandrayaan 3 landing LIVE", "url": "https://unknown-blog.xyz/b", "published_date": "2023-08-23"},
    {"title": "One week after landing", "url": "https://www.reuters.com/c", "published_date": "2023-08-30"},
]
RELIABILITY = {"thehindu.com": 0.75, "reuters.com": 0.9, "unknown-blog.xyz": 0.2, "en.wikipedia.org": 0.9}


def tavily_client(results=TAVILY, fail=False):
    calls = []

    def fetch(params):
        calls.append(params)
        if fail:
            raise RuntimeError("HTTP 500")
        return results

    return ns.NewsSearchClient(reliability=RELIABILITY, cache_path=None, fetch=fetch, backend="tavily", api_key="k"), calls


def test_tavily_lookup_restricts_to_trusted_outlets_and_window():
    c, calls = tavily_client()
    lk = c.lookup("Chandrayaan-3 lands near the Moon's south pole", before="2023-08-23", start="2023-08-16")
    assert lk.ok and lk.n_results == 3 and [p.domain for p in lk.passages] == ["thehindu.com"]
    p = calls[0]
    assert p["start_date"] == "2023-08-16" and p["end_date"] == "2023-08-23" and p["topic"] == "general"
    assert set(p["include_domains"]) == {"thehindu.com", "reuters.com"}  # rated >= 0.75, Wikipedia excluded
    assert lk.passages[0].text.startswith("thehindu.com reported (2023-08-23): Chandrayaan-3 lands")
    assert "touched down" in lk.passages[0].text


def test_undated_articles_dropped_for_a_dated_claim():
    c, _ = tavily_client(results=[{"title": "Chandrayaan-3 lands", "url": "https://www.thehindu.com/u"}])
    assert len(c.lookup("Chandrayaan-3 lands on the Moon", before="2023-08-23").passages) == 1
    assert c.lookup("Chandrayaan-3 lands on the Moon", before="2023-08-23", strict_dates=True).passages == []


def test_undated_reports_still_count_against_silence():
    """Not evidence (could be from after the claim), but proof that outlets covered the topic: the 6 Nov
    measurement ignored them and 42% of real big events looked unreported instead of 6%."""
    c, _ = tavily_client(results=[{"title": "Chandrayaan-3 lands", "url": "https://www.thehindu.com/u"},
                                  {"title": "Chandrayaan-3 a year on", "url": "https://www.thehindu.com/2024/08/x"}])
    lk = c.lookup("Chandrayaan-3 lands on the Moon", before="2023-08-23", strict_dates=True)
    assert lk.passages == [] and [p.source_url for p in lk.undated] == ["https://www.thehindu.com/u"]  # 2024 URL: later
    assert ns.relevant_reports("Chandrayaan-3 lands on the Moon", lk) == 1
    relevant_only = lambda claim, texts: [5.0 if "lands" in t else -5.0 for t in texts]  # noqa: E731
    assert ns.relevant_reports("Chandrayaan-3 lands on the Moon", lk, relevant_only) == 1
    assert ns.relevant_reports("Chandrayaan-3 lands on the Moon", None) == 0


def test_silence_reads_every_returned_report_not_the_top5():
    """A relevant report ranked below five index passages is still a report: not silence."""
    news, _ = tavily_client(results=[{"title": "RBI announces Rs 500 notes withdrawn from tomorrow",
                                      "url": "https://www.reuters.com/z", "published_date": "2026-11-05"}])
    lk = news.lookup(BIG)
    top5 = [hit(f"q{i}", "unrelated passage", score=9.0) for i in range(5)]
    merged = news.merge(BIG, top5, k=5, lookup=lk, reranker=lambda c, t: [1.0] * len(t))
    assert not any(str(h.passage.id).startswith("news:") for h in merged)  # cut from the evidence
    assert ns.relevant_reports(BIG, lk, lambda c, t: [1.0] * len(t)) == 1  # relevance 0.73 >= 0.5: still heard


def test_debunk_headlines_do_not_count_against_silence():
    """A fact-check exists because the claim is false: it is not the coverage a true claim would get."""
    c, _ = tavily_client(results=[{"title": "Fact Check: Viral claim that RBI banned Rs 500 notes is false",
                                   "url": "https://www.reuters.com/fact-check/x", "published_date": "2026-11-05"},
                                  {"title": "Police bust fake currency racket", "url": "https://www.thehindu.com/y",
                                   "published_date": "2026-11-05"}])
    lk = c.lookup("RBI bans Rs 500 notes from tomorrow")
    assert [p.source_url for p in lk.passages if p.id in lk.debunk_ids] == ["https://www.reuters.com/fact-check/x"]
    assert ns.relevant_reports("RBI bans Rs 500 notes from tomorrow", lk) == 1  # the real event still counts


def keyword_only_client():
    """Answers only the shorter keyword query: the claim's own words find nothing."""
    calls = []

    def fetch(params):
        calls.append(params["query"])
        if len(params["query"].split()) <= 6:
            return [{"title": "Indian Navy rescues crew of hijacked ship", "url": "https://www.thehindu.com/n",
                     "published_date": params["end_date"]}]  # inside the claim's window
        return []

    return ns.NewsSearchClient(reliability=RELIABILITY, cache_path=None, fetch=fetch, backend="tavily",
                               api_key="k"), calls


def test_lookup_twice_searches_again_only_when_nothing_was_heard():
    claim = "The Indian Navy rescued all the crew of the hijacked ship MV Lila Norfolk in the Arabian Sea on Friday."
    c, calls = keyword_only_client()
    lk = c.lookup_twice(claim, before="2024-01-06")
    assert len(calls) == 2 and lk.queries == 2 and lk.ok
    assert ns.relevant_reports(claim, lk) == 1  # heard on the second search: not silence
    c, calls = tavily_client()  # the first search already hears a report: no second search
    assert c.lookup_twice("Chandrayaan-3 lands near the Moon's south pole", before="2023-08-23").queries == 1
    assert len(calls) == 1


def test_pipeline_searches_a_covered_claim_twice():
    claim = "RBI announced that all Rs 500 notes in every bank account will be withdrawn from circulation tomorrow."
    news, calls = keyword_only_client()
    rep = pipeline(news=news).analyze(NewsItem(text=claim))
    cv = rep["claims"][0]["coverage"]
    assert cv["expected"] and cv["search"]["queries"] == 2 and cv["relevant_reports"] == 1 and len(calls) == 2


def test_failed_search_is_not_silence():
    c, _ = tavily_client(fail=True)
    lk = c.lookup("RBI bans Rs 500 notes from tomorrow")
    assert not lk.ok and lk.passages == []
    c, _ = tavily_client(results=[])
    assert c.lookup("RBI bans Rs 500 notes from tomorrow").ok  # answered, nothing found: that is silence


def test_trusted_domains_put_indian_outlets_first():
    doms = ns.trusted_domains({"zz.com": 0.95, "reuters.com": 0.9, "thehindu.com": 0.8, "pib.gov.in": 0.9, "bad.com": 0.3})
    assert doms[:2] == ["pib.gov.in", "thehindu.com"] and doms[2] == "reuters.com" and "bad.com" not in doms


def test_default_news_client(monkeypatch):
    monkeypatch.setattr("src.retrieval.factcheck_api.load_env_file", lambda names=(): None)
    monkeypatch.delenv("NEWSCRED_NEWS_SEARCH", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    assert ns.default_client() is None
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    assert ns.default_client().backend == "tavily"
    monkeypatch.setenv("NEWSCRED_NEWS_SEARCH", "0")
    assert ns.default_client() is None
    monkeypatch.setenv("NEWSCRED_NEWS_SEARCH", "gdelt")
    assert ns.default_client().backend == "gdelt"


# --- pipeline ----------------------------------------------------------------------------------------

class ListSearcher:
    def __init__(self, hits=()):
        self.hits = list(hits)

    def search(self, query, k=5, before=None, reranker=None):
        return list(self.hits)


class FakeWiki:
    def __init__(self, passages=(), known=()):
        self.passages, self.known = list(passages), set(known)

    def is_well_known(self, name):
        return name in self.known

    def merge(self, claim, hits, before=None, reranker=None, k=5):
        return (list(hits) + [Hit(p, 2.0, None, None) for p in self.passages])[:k]


def pipeline(news=None, searcher=None, wiki=None, predict=None, memory=None):
    predict = predict or (lambda c, ps: np.array([SUPPORT if "reported" in p else NEUTRAL for p in ps]))
    return Pipeline(components=dict(scorer=HeuristicCheckWorthiness(), searcher=searcher or ListSearcher(),
                                    verifier=ClaimVerifier(predict_fn=predict, reliability=RELIABILITY), reranker=None,
                                    factcheck=None, news=news, wiki=wiki, memory=memory, meta=None),
                    autoload=False)


BIG = "RBI announced that Rs 500 notes will be banned from tomorrow."


def test_big_claim_with_no_trusted_report_scores_low(monkeypatch):
    monkeypatch.setattr(cov, "ABSENCE_LR", 0.25)  # the arithmetic below; the measured value lives in coverage.py
    news, _ = tavily_client(results=[])
    rep = pipeline(news=news).analyze(NewsItem(text=BIG))
    assert not rep["abstained"] and rep["score"] == 20 and rep["band"] == "Low"
    assert rep["coverage"]["applied"] and rep["coverage"]["model_score"] is None
    assert rep["reasons"][0].startswith("No trusted outlet reported this.")
    assert rep["baseline_score"] == 50 and rep["contributions"] == [
        {"check": "missing_coverage", "points": -30.0, "applicable": True}]
    assert rep["claims"][0]["coverage"]["expected"] and rep["claims"][0]["coverage"]["searched"]


def test_rule_off_by_default():
    """Decided by the rule written in advance (reports/absence_hoax_eval.json): it fails the 5% target, so off;
    the measured LR is kept for NEWSCRED_ABSENCE_LR=0.27."""
    assert cov.ABSENCE_LR is None and cov.MEASURED_ABSENCE_LR == 0.27


def test_rule_switched_off_still_reports_silence(monkeypatch):
    """NEWSCRED_ABSENCE_LR=off: silence is shown, not scored."""
    monkeypatch.setattr(cov, "ABSENCE_LR", None)
    news, _ = tavily_client(results=[])
    rep = pipeline(news=news).analyze(NewsItem(text=BIG))
    assert rep["abstained"] and rep["coverage"] is None
    assert rep["reasons"][1].startswith("No trusted outlet reported this")


def test_silence_does_not_count_when_the_search_did_not_run(monkeypatch):
    monkeypatch.setattr(cov, "ABSENCE_LR", 0.25)
    news, _ = tavily_client(fail=True)
    rep = pipeline(news=news).analyze(NewsItem(text=BIG))
    assert rep["abstained"] and rep["score"] is None and "did not answer" in rep["reasons"][1]
    rep = pipeline(news=None).analyze(NewsItem(text=BIG))
    assert rep["abstained"] and "news search is off" in rep["reasons"][1]


def test_local_claim_with_no_report_stays_unverified(monkeypatch):
    monkeypatch.setattr(cov, "ABSENCE_LR", 0.25)
    news, _ = tavily_client(results=[])
    rep = pipeline(news=news).analyze(NewsItem(text="A man in a village in Bihar rescued 3 stray dogs from a well on Sunday."))
    assert rep["abstained"] and rep["coverage"] is None


def test_big_claim_confirmed_by_a_trusted_report_is_not_penalised():
    news, _ = tavily_client(results=[{"title": "RBI announces Rs 500 notes withdrawn from tomorrow",
                                      "url": "https://www.reuters.com/x", "published_date": RECENT}])
    rep = pipeline(news=news).analyze(NewsItem(text=BIG))
    assert rep["claims"][0]["verdict"] == "Supported" and rep["coverage"] is None and rep["score"] > 60


def test_old_wikipedia_text_no_longer_confirms_a_new_event():
    """Live before the fix: 'RBI announces that Rs 500 notes will be banned from tomorrow' scored 67 (Mostly
    credible), "Supported" by Wikipedia's text about the 2016 demonetisation."""
    wiki = FakeWiki([Passage("wiki:Indian black money#0", "Wikipedia, Indian black money (as of 2026-09-26): In "
                             "November 2016 the government demonetised Rs 500 and Rs 1000 notes.", None,
                             "en.wikipedia.org", "2026-09-26")])
    predict = lambda c, ps: np.array([SUPPORT for _ in ps])  # noqa: E731 - the verifier reads it as support
    rep = pipeline(wiki=wiki, predict=predict).analyze(NewsItem(text=BIG))
    assert rep["claims"][0]["credibility"] is None and rep["score"] is None  # no longer "Supported"
    assert rep["claims"][0]["coverage"]["not_confirming_other_time"] == 1


def test_refuted_claim_is_left_to_the_evidence(monkeypatch):
    monkeypatch.setattr(cov, "ABSENCE_LR", 0.25)
    fc = Passage("fc1", "Q: Is it true that RBI banned Rs 500 notes from tomorrow? A: No. factly.in rated this claim "
                        "False.", None, "factly.in", None)
    news, _ = tavily_client(results=[])
    rep = pipeline(news=news, searcher=ListSearcher([Hit(fc, 3.0, 1, 1)]),
                   predict=lambda c, ps: np.array([REFUTE for _ in ps])).analyze(NewsItem(text=BIG))
    assert rep["coverage"] is None and rep["claims"][0]["verdict"] == "Refuted"


# --- evidence memory ---------------------------------------------------------------------------------

def bow_encoder():
    vocab = {}

    def enc(texts):
        out = np.zeros((len(texts), 128), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in t.lower().replace(".", " ").replace(":", " ").split():
                out[i, vocab.setdefault(w, len(vocab) % 128)] += 1
        return out

    return enc


def test_memory_keeps_only_trusted_live_passages_and_respects_dates(tmp_path):
    enc = bow_encoder()
    mem = EvidenceMemory(tmp_path / "m.sqlite", encoder=enc)
    added = mem.remember([
        Passage("news:a", "thehindu.com reported (2023-08-23): Chandrayaan-3 lands on the Moon", "u", "thehindu.com", "2023-08-23"),
        Passage("q1", "Q: Did Chandrayaan-3 land on the Moon? A: Yes.", None, "x.org", "2023-08-24"),  # index: not live
        Passage("news:a", "duplicate", "u", "thehindu.com", "2023-08-23"),
    ])
    assert added == 1 and mem.size == 1
    assert [p.id for p in mem.search("Chandrayaan-3 lands on the Moon")] == ["news:a"]
    assert mem.search("Chandrayaan-3 lands on the Moon", before="2023-08-01") == []  # no hindsight
    again = EvidenceMemory(tmp_path / "m.sqlite", encoder=enc)  # persisted
    assert again.size == 1
    merged = again.merge("Chandrayaan-3 lands on the Moon", [hit("q9", "other passage", score=1.0)])
    assert {h.passage.id for h in merged} == {"q9", "news:a"}


def test_pipeline_learns_from_a_live_report_and_reuses_it(tmp_path):
    mem = EvidenceMemory(tmp_path / "m.sqlite", encoder=bow_encoder())
    news, _ = tavily_client(results=[{"title": "RBI announces Rs 500 notes withdrawn from tomorrow",
                                      "url": "https://www.reuters.com/x", "published_date": RECENT}])
    pipeline(news=news, memory=mem).analyze(NewsItem(text=BIG))
    assert mem.size == 1
    news_down, _ = tavily_client(fail=True)  # later: the news search is down
    rep = pipeline(news=news_down, memory=mem).analyze(NewsItem(text="RBI announced Rs 500 notes will be banned from tomorrow."))
    assert rep["claims"][0]["verdict"] == "Supported"  # found in the memory


def test_mode_relevant_but_unconfirming_report(monkeypatch):
    """A trusted report about the topic that does not confirm the claim: real silence only in the riskier mode."""
    monkeypatch.setattr(cov, "ABSENCE_LR", 0.25)
    news, _ = tavily_client(results=[{"title": "RBI governor speaks on currency management", "url": "https://www.reuters.com/y",
                                      "published_date": RECENT}])
    neutral = lambda c, ps: np.array([NEUTRAL for _ in ps])  # noqa: E731 - read as "says nothing"
    rep = pipeline(news=news, predict=neutral).analyze(NewsItem(text=BIG))
    assert rep["claims"][0]["coverage"]["relevant_reports"] == 1 and rep["abstained"]  # default: not silence
    monkeypatch.setattr(cov, "ABSENCE_MODE", "no_confirming_report")
    monkeypatch.setattr(cov, "ABSENCE_LR", 0.25)
    rep = pipeline(news=news, predict=neutral).analyze(NewsItem(text=BIG))
    assert not rep["abstained"] and rep["band"] == "Low"
