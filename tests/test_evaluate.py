"""Phase 8 evaluation code: component metrics (3A, 3B, 3C) and system-level experiments,
on tiny synthetic inputs with fake models (no downloads, no data/ or models/ files)."""

import json
import math

import numpy as np
import pandas as pd
import pytest

from src.claims.extract import HeuristicCheckWorthiness, evaluate_checkworthiness
from src.fusion import evaluate as E
from src.fusion import train as T
from src.retrieval.search import HybridSearcher, Passage, evaluate_retrieval, rank_metrics
from src.verify.model import LABELS, ClaimVerifier, classification_metrics, evaluate_verifier
from test_meta import synthetic_table
from test_retrieval import fake_encoder

# --- 3B retrieval ---------------------------------------------------------------------


def test_rank_metrics_hit_recall_mrr():
    rankings = [["a", "x", "b"], ["x", "y", "c"], ["x", "y", "z"]]
    gold = [{"a", "b"}, {"c"}, {"q"}]
    m = rank_metrics(rankings, gold, k=2)
    assert m["queries"] == 3
    assert m["hit@2"] == pytest.approx(1 / 3)  # only the first query has gold in its top 2
    assert m["recall@2"] == pytest.approx((1 / 2 + 0 + 0) / 3)
    assert m["mrr"] == pytest.approx((1 / 1 + 1 / 3 + 0) / 3)  # rank beyond k still counts for MRR


def _qa(*answers):
    return json.dumps([{"question": "What happened?", "answers": [{"answer": a, "answer_type": "Abstractive"}]}
                       for a in answers])


AVERITEC_LIKE = pd.DataFrame({
    "item_id": ["t1", "t2", "v1", "v2", "s1"],
    "split": ["train", "train", "val", "val", "test"],
    "text": ["Unemployment fell to 6 percent in March.", "India Gate is in Delhi.",
             "The jobless rate dropped last spring.", "Section 144 imposed near Jantar Mantar.",
             "A new bridge opened in Mumbai."],
    "raw_label": ["Supported", "Supported", "Refuted", "Supported", "Not Enough Evidence"],
    "questions_json": [_qa("The unemployment rate fell to 6.1 percent."), _qa("India Gate is a war memorial in New Delhi."),
                       _qa("The jobless rate dropped sharply in spring.", "Officials said jobless claims fell."),
                       _qa("Section 144 was imposed in Delhi near Jantar Mantar."), _qa("No bridge opened.")],
})


def test_evaluate_retrieval_finds_own_gold_passages():
    rep = evaluate_retrieval(AVERITEC_LIKE, encoder=fake_encoder, k=1)
    assert rep["index_passages"] == 5  # train + val passages, not test
    assert set(rep["stages"]) == {"dense", "bm25", "hybrid_rrf"}  # no re-ranker given
    hybrid = rep["stages"]["hybrid_rrf"]
    assert hybrid["queries"] == 2 and hybrid["hit@1"] == 1.0 and hybrid["mrr"] == 1.0


# --- 3C verification -------------------------------------------------------------------


def test_classification_metrics_confusion_orientation():
    m = classification_metrics([0, 0, 1, 2], [0, 1, 1, 2])
    cm = np.array(m["confusion_matrix"]["rows_true_cols_predicted"])
    assert cm[0, 1] == 1 and cm[0, 0] == 1  # row = true Supported, one predicted Refuted
    assert m["accuracy"] == pytest.approx(0.75) and m["per_class"][LABELS[3]]["support"] == 0


def _keyword_verifier(claim, passages):
    """Fake 4-way model: 'fell'/'dropped' -> Supported, 'no ' -> Refuted, else NEE."""
    out = []
    for p in passages:
        row = np.full(4, 0.05)
        row[0 if ("fell" in p or "dropped" in p) else 1 if "No " in p else 2] = 0.85
        out.append(row)
    return np.array(out)


def test_evaluate_verifier_gold_and_retrieved_settings():
    df = AVERITEC_LIKE.assign(split=["train", "train", "test", "test", "test"])
    verifier = ClaimVerifier(predict_fn=_keyword_verifier, reliability={})
    searcher = HybridSearcher([Passage("p1", "Q: x A: The unemployment rate fell."), Passage("p2", "Q: y A: No bridge opened.")],
                              encoder=fake_encoder)
    rep = evaluate_verifier(df, verifier, searcher=searcher, k=2)
    assert rep["n_claims"] == 3 and rep["label_counts"]["Refuted"] == 1
    assert {"gold_concat", "gold_passages", "retrieved", "retrieved_gated"} <= set(rep)
    assert 0 <= rep["retrieved_off_topic_share"] <= 1
    # gold evidence for "v1" says "dropped" -> Supported, but its label is Refuted: a counted error
    assert rep["gold_passages"]["confusion_matrix"]["rows_true_cols_predicted"][1][0] == 1


# --- 3A check-worthiness ---------------------------------------------------------------


def test_evaluate_checkworthiness_reports_per_source():
    table = pd.DataFrame({
        "text": ["Unemployment fell to 6.1 percent in March 2021.", "I think this is a sad day.",
                 "The minister said 40 people were arrested in Delhi.", "What do you think?", "Thank you all."],
        "label": [1, 0, 1, 0, 0], "source": ["averitec", "checkthat", "checkthat", "checkthat", "checkthat"],
        "split": ["test"] * 5,
    })
    rep = evaluate_checkworthiness(table, {"heuristic": HeuristicCheckWorthiness()})
    h = rep["heuristic"]
    assert rep["test_rows"] == 5 and set(h) == {"all", "averitec", "checkthat"}
    assert h["all"]["f1"] == 1.0 and h["checkthat"]["roc_auc"] == 1.0
    assert "macro_f1" not in h["averitec"]  # positives only: macro-F1 undefined


# --- system level ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_grid():
    grid = T.PARAM_GRID
    T.PARAM_GRID = [{"num_leaves": 7, "learning_rate": 0.1}]
    yield
    T.PARAM_GRID = grid


def test_retrain_ablation_reports_every_channel(small_grid):
    table = synthetic_table(n_per=120)
    rep = E.retrain_ablation(table, channels=["text_image", "reuse"])
    assert rep["full"]["overall"]["roc_auc"] > 0.9
    assert set(rep["without"]) == {"text_image", "reuse"}
    for r in rep["without"].values():
        assert math.isfinite(r["drop_roc_auc"]) and math.isfinite(r["drop_mean_dataset_auc"])


def _items_for(table):
    words = {1.0: "official report confirms figures", 0.0: "shocking secret they hide"}
    return pd.DataFrame({
        "item_id": table["item_id"], "dataset": table["dataset"], "credible": table["credible"],
        "split": table["split"], "text": [words[y] + f" item {i}" for i, y in enumerate(table["credible"])],
        "source_domain": None, "date": None, "raw_label": None,
    })


def test_text_baseline_scores_all_methods_on_the_same_test_rows(small_grid, monkeypatch):
    table = synthetic_table(n_per=80)
    model, _ = T.fit_meta(table)
    monkeypatch.setattr(E.MetaModel, "load", staticmethod(lambda *a, **k: model))
    items = _items_for(table)
    items = pd.concat([items, items[items["split"] == "val"].assign(split="train",
                                                                    item_id=lambda d: d["item_id"] + "_tr")])
    fake_bert = lambda tr, va, te: (np.full(len(va), 0.5), te["credible"].to_numpy(float))  # noqa: E731
    rep = E.text_baseline(table, items, encoder=fake_encoder, c_grid=(1.0,), bert_fn=fake_bert)
    assert set(rep["methods"]) == {"text_tfidf", "text_minilm", "text_bert", "equal_weight", "meta"}
    assert rep["methods"]["text_bert"]["overall"]["roc_auc"] == 1.0  # scored on the right rows
    assert rep["test_rows"] == int((table["split"] == "test").sum())
    assert rep["methods"]["text_tfidf"]["overall"]["roc_auc"] > 0.9  # the words give the label away


def test_error_analysis_lists_only_wrong_predictions(small_grid):
    table = synthetic_table(n_per=80, seed=3)
    model, _ = T.fit_meta(table)
    flip = table.index[(table["split"] == "test")][::10]  # label noise: guaranteed confident mistakes
    table.loc[flip, "credible"] = 1.0 - table.loc[flip, "credible"]
    df = E.error_analysis(table, _items_for(table), model, per_dataset=3)
    assert (df.groupby("dataset").size() <= 3).all()
    wrong = ((df["score"] >= 50) & (df["label"] == "not credible")) | ((df["score"] < 50) & (df["label"] == "credible"))
    assert wrong.all() and df["wrong_way_check"].notna().all() and (df["cause"] == "").all()


def test_make_figures_from_reports(tmp_path):
    cm = {"labels": LABELS, "rows_true_cols_predicted": np.eye(4, dtype=int).tolist()}
    (tmp_path / "verifier_metrics.json").write_text(json.dumps(
        {"gold_concat": {"confusion_matrix": cm, "macro_f1": 1.0, "accuracy": 1.0}}))
    per = {"overall": {"roc_auc": 0.8}, "auc_by_dataset": {"averitec": 0.7, "fakeddit": 0.9}}
    (tmp_path / "text_baseline_metrics.json").write_text(json.dumps({"methods": {"meta": per, "text_tfidf": per}}))
    (tmp_path / "retrieval_metrics.json").write_text(json.dumps(
        {"split": "val", "stages": {"dense": {"queries": 2, "hit@5": 0.5, "mrr": 0.4}, "bm25": {"queries": 2, "hit@5": 0.6, "mrr": 0.5}}}))
    (tmp_path / "meta_ablation").mkdir()
    (tmp_path / "meta_ablation" / "retrain_without_channel.json").write_text(json.dumps(
        {"without": {"claims": {"drop_roc_auc": 0.0, "drop_mean_dataset_auc": 0.04},
                     "source": {"drop_roc_auc": -0.002, "drop_mean_dataset_auc": 0.05}}}))
    written = E.make_figures(tmp_path)
    assert len(written) == 4 and all(p.exists() and p.stat().st_size > 0 for p in written)
