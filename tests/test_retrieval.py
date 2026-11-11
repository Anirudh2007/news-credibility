import json
import zlib

import numpy as np
import pandas as pd
import pytest

from src.retrieval.search import HybridSearcher, Passage, passages_from_averitec, rrf_fuse, tokenize

PASSAGES = [
    Passage("p0", "The unemployment rate fell to 6.1 percent in March 2021.", date="2021-04-02"),
    Passage("p1", "Section 144 was imposed in Delhi to stop gatherings near Jantar Mantar.", date="2020-01-10"),
    Passage("p2", "The jobless rate dropped sharply last spring, officials said.", date="2021-05-01"),
    Passage("p3", "India Gate is a war memorial in New Delhi.", date=None),
    Passage("p4", "A fact-check published later rated the unemployment claim false.", date="2023-01-01"),
]
_SYNONYMS = {"jobless": "unemployment", "dropped": "fell"}


def fake_encoder(texts):
    """Deterministic bag-of-words hashing encoder with a tiny synonym map, so the
    dense side can match paraphrases without downloading a model."""
    out = np.zeros((len(texts), 256), dtype=np.float32)
    for i, t in enumerate(texts):
        for tok in tokenize(t):
            out[i, zlib.crc32(_SYNONYMS.get(tok, tok).encode()) % 256] += 1.0
    return out


@pytest.fixture(scope="module")
def searcher():
    return HybridSearcher(PASSAGES, encoder=fake_encoder)


def test_rrf_rewards_agreement():
    fused = rrf_fuse([[1, 2, 3], [2, 1, 4]])
    assert fused[1] == fused[2] > fused[3] > 0 and fused[4] < fused[3] + 1e-9


def test_keyword_query_finds_exact_term(searcher):
    hits = searcher.search("Section 144 Jantar Mantar", k=3)
    assert hits[0].passage.id == "p1" and hits[0].bm25_rank == 1


def test_hybrid_returns_k_sorted_hits(searcher):
    hits = searcher.search("unemployment fell in March", k=3)
    assert len(hits) == 3 and hits[0].passage.id == "p0"
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_before_filter_drops_later_evidence(searcher):
    ids = [h.passage.id for h in searcher.search("unemployment claim", k=5, before="2022-01-01")]
    assert "p4" not in ids  # dated after the claim
    assert "p0" in ids  # dated before: kept


def test_reranker_reorders(searcher):
    hits = searcher.search("unemployment", k=2, reranker=lambda q, texts: [len(t) for t in texts])
    assert hits[0].score >= hits[1].score


def test_save_and_load_roundtrip(searcher, tmp_path):
    searcher.save(tmp_path)
    again = HybridSearcher.load(tmp_path, encoder=fake_encoder)
    q = "Section 144 Delhi"
    assert [h.passage.id for h in again.search(q)] == [h.passage.id for h in searcher.search(q)]


def test_passages_from_averitec():
    questions = [
        {"question": "What was the rate?", "answers": [
            {"answer": "6.1%", "answer_type": "Extractive", "source_url": "https://www.bls.gov/x"},
            {"answer": "No answer could be found.", "answer_type": "Unanswerable"},
        ]},
        {"question": "Did it fall?", "answers": [
            {"answer": "Yes", "answer_type": "Boolean", "boolean_explanation": "It fell 0.2 points."}]},
        {"question": "Did RBI make this announcement?", "answers": [{"answer": "No", "answer_type": "Boolean"}]},
    ]
    df = pd.DataFrame({"item_id": ["averitec_train_0"], "questions_json": [json.dumps(questions)]})
    ps = passages_from_averitec(df)
    # verifier training format; the context-free "No" is not indexed
    assert [p.text for p in ps] == ["Q: What was the rate? A: 6.1%", "Q: Did it fall? A: Yes. It fell 0.2 points."]
    assert ps[0].domain == "bls.gov"


def test_index_passages_use_the_verifier_training_format():
    from src.verify.model import _qa_texts

    questions = [{"question": "What was the rate?", "answers": [{"answer": "6.1%", "answer_type": "Extractive"}]}]
    df = pd.DataFrame({"item_id": ["x"], "questions_json": [json.dumps(questions)]})
    assert [p.text for p in passages_from_averitec(df)] == _qa_texts(json.dumps(questions))


def test_empty_passages_rejected():
    with pytest.raises(ValueError):
        HybridSearcher([], encoder=fake_encoder)
