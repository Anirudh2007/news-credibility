"""Separate evidence budgets (0.10.1): passages that may only confirm (Wikipedia, news) never take the place of
passages that can refute (offline index, fact-checks). Before, "The earth is flat." came back Unverified: the
Wikipedia article about the myth filled the shared top 5, did not count, and pushed the refutation out."""

import numpy as np

from src.claims.extract import HeuristicCheckWorthiness
from src.pipeline import CONFIRM_PER_CLAIM, EVIDENCE_PER_CLAIM, NewsItem, Pipeline
from src.retrieval.search import Hit, Passage
from src.verify.model import ClaimVerifier, evidence_budget

SUPPORT = [0.9, 0.02, 0.05, 0.03]
REFUTE = [0.02, 0.9, 0.05, 0.03]
NEUTRAL = [0.05, 0.05, 0.85, 0.05]


def hit(pid, score, text="text", domain="x.org"):
    return Hit(Passage(pid, text, f"https://{domain}/{pid}", domain, None), score, None, None)


def test_budget_keeps_both_kinds():
    refute = [hit(f"q{i}", 1.0 - i / 10) for i in range(7)]
    confirm = [hit(f"wiki:A#{i}", 5.0 - i / 10) for i in range(7)] + [hit("news:r", 4.0)]
    kept = evidence_budget(refute + confirm, k=5, k_confirm=5)
    ids = [str(h.passage.id) for h in kept]
    assert [i for i in ids if not i.startswith(("wiki:", "news:"))] == ["q0", "q1", "q2", "q3", "q4"]
    assert len([i for i in ids if i.startswith(("wiki:", "news:"))]) == 5
    assert [h.score for h in kept] == sorted((h.score for h in kept), reverse=True)


def test_budget_contains_the_old_shared_top_k():
    hits = [hit("q0", 3.0), hit("wiki:A#0", 2.5), hit("q1", 2.0), hit("news:b", 1.0), hit("q2", 0.5), hit("q3", 0.1)]
    old = sorted(hits, key=lambda h: -h.score)[:5]
    new = evidence_budget(hits, k=5)
    assert set(map(id, old)) <= set(map(id, new))


class ListSearcher:
    def __init__(self, hits):
        self.hits = list(hits)

    def search(self, query, k=5, before=None, reranker=None):
        return list(self.hits)[:k]


class MythWiki:
    """Wikipedia's article about a myth: very relevant, but it describes the myth, so it does not confirm it."""

    def __init__(self, n=5):
        self.passages = [Passage(f"wiki:Flat_Earth#{i}", f"Wikipedia, Flat Earth (as of 2026-10-17): myth text {i}",
                                 f"https://en.wikipedia.org/wiki/Flat_Earth#{i}", "en.wikipedia.org", "2026-10-17")
                         for i in range(n)]

    def is_well_known(self, name):
        return False

    def merge(self, claim, hits, before=None, reranker=None, k=5):
        merged = list(hits) + [Hit(p, 9.0, None, None) for p in self.passages]  # outranks every index passage
        merged.sort(key=lambda h: -h.score)
        return merged[:k]


def test_a_myths_wikipedia_article_does_not_crowd_out_its_refutation():
    index = [hit("q-flat", 3.0, "Q: Is the earth flat? A: No. The earth is an oblate spheroid.")]

    def predict(claim, passages):
        return np.array([REFUTE if "A: No." in p else NEUTRAL for p in passages])

    pipe = Pipeline(components=dict(scorer=HeuristicCheckWorthiness(), searcher=ListSearcher(index),
                                    verifier=ClaimVerifier(predict_fn=predict, reliability={}, match_min=-1.0),
                                    reranker=None, factcheck=None, news=None, wiki=MythWiki(), memory=None, meta=None),
                    autoload=False)
    rep = pipe.analyze(NewsItem(text="The earth is flat, scientists in Kenya announced on Monday."))
    claim = rep["claims"][0]
    assert claim["verdict"] == "Refuted"
    assert not rep["abstained"] and rep["score"] is not None and rep["score"] <= 30
    assert EVIDENCE_PER_CLAIM == CONFIRM_PER_CLAIM == 5
