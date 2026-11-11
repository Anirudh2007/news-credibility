import json
import math

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.claims.extract import HeuristicCheckWorthiness
from src.fusion import features as F
from src.fusion import train as T
from src.fusion.meta import MetaModel, average_channel_scores, band, to_score
from src.pipeline import NewsItem, Pipeline
from src.retrieval.search import Hit, Passage
from src.source.credibility import SourceResult
from src.verify.model import Verdict

# --- feature vector -----------------------------------------------------------------


class _C:  # minimal Claim / Verdict stand-ins
    def __init__(self, cw):
        self.checkworthiness = cw


def _v(label, score, conf=0.9):
    return Verdict(label, {}, score, conf, [])


def test_feature_columns_unique_and_grouped():
    assert len(F.FEATURE_COLUMNS) == len(set(F.FEATURE_COLUMNS))
    assert set(F.CHANNEL_SCORE.values()) <= set(F.FEATURE_COLUMNS)


def test_claim_features_nan_vs_zero():
    assert all(math.isnan(v) for v in F.claim_features(None, None).values())  # no text: not applicable
    none_found = F.claim_features([], None)
    assert none_found["n_claims"] == 0 and math.isnan(none_found["claim_score_mean"])  # text, nothing checkable
    extracted_only = F.claim_features([_C(0.7)], None)
    assert extracted_only["checkworthiness_max"] == 0.7 and math.isnan(extracted_only["claim_support_ratio"])
    f = F.claim_features([_C(0.9), _C(0.6), _C(0.8)], [_v("Supported", 1.0), _v("Refuted", 0.0), _v("Not Enough Evidence", math.nan, 0.1)])
    assert f["n_claims"] == 3 and f["claim_support_ratio"] == 0.5 and f["claim_refute_ratio"] == 0.5
    assert f["claim_score_mean"] == 0.5 and f["claim_score_min"] == 0.0
    assert f["verifier_confidence_mean"] == pytest.approx((0.9 + 0.9 + 0.1) / 3)
    assert math.isnan(f["evidence_relevance"])  # verdicts without re-ranker relevance


def test_claim_features_average_evidence_relevance():
    a, b = _v("Supported", 1.0), _v("Refuted", 0.0)
    a.relevance, b.relevance = 0.9, 0.3
    f = F.claim_features([_C(0.9), _C(0.8)], [a, b])
    assert f["evidence_relevance"] == pytest.approx(0.6)
    assert f["claim_evidence_score"] == pytest.approx(0.5)  # mean score 0.5 -> neutral


def test_assemble_fills_missing_with_nan():
    row = F.assemble({"source_score": 0.8}, F.modality_features(True, False, False))
    assert row["source_score"] == 0.8 and row["has_image"] == 0.0 and math.isnan(row["text_image_score"])
    assert len(F.to_vector(row)) == len(F.FEATURE_COLUMNS)


# --- bands, scores, baseline ------------------------------------------------------------

@pytest.mark.parametrize("score,name", [(0, "Low"), (30, "Low"), (31, "Questionable"), (60, "Questionable"),
                                        (61, "Mostly credible"), (80, "Mostly credible"), (81, "High"), (100, "High"),
                                        (None, "Unknown")])
def test_band_boundaries(score, name):
    assert band(score) == name


def test_to_score():
    assert to_score(0.804) == 80 and to_score(1.2) == 100 and to_score(math.nan) is None


def test_average_channel_scores_ignores_nan():
    a = F.to_vector(F.assemble({"source_score": 0.8, "temporal_score": 0.4}))
    b = F.to_vector(F.assemble({}))
    out = average_channel_scores([a, b])
    assert out[0] == pytest.approx(0.6) and math.isnan(out[1])


# --- weights and metrics ----------------------------------------------------------------

def test_balance_weights_equal_per_dataset_and_label():
    d = ["fakeddit"] * 90 + ["averitec"] * 10
    y = [1.0] * 60 + [0.0] * 30 + [1.0] * 2 + [0.0] * 6 + [0.5] * 2
    w = pd.Series(T.balance_weights(d, y))
    per_d = w.groupby(pd.Series(d)).sum()
    assert per_d["fakeddit"] == pytest.approx(per_d["averitec"]) and w.mean() == pytest.approx(1.0)
    av = pd.DataFrame({"w": w[90:].to_numpy(), "y": y[90:]}).groupby("y")["w"].sum()
    assert av[0.0] == pytest.approx(av[0.5]) == pytest.approx(av[1.0])


def test_ece_and_binary_metrics():
    p = np.array([0.1] * 10 + [0.9] * 10)
    y = np.array([0] * 9 + [1] + [1] * 9 + [0])
    assert T.ece(y, p) == pytest.approx(0.0)
    m = T.binary_metrics(np.r_[y, 0.5], np.r_[p, np.nan])
    assert m["coverage"] == pytest.approx(20 / 21) and m["roc_auc"] > 0.8 and "brier" in m


# --- fit + evaluate on a synthetic meta table ---------------------------------------------

def synthetic_table(n_per=300, seed=0):
    """Four datasets, each with its own channels present (like the real table).
    Credibility signal sits in each dataset's channel scores."""
    rng = np.random.default_rng(seed)
    rows = []
    home = {"averitec": ["claims", "temporal"], "fakeddit": ["text_image", "source"],
            "fakenewsnet": ["source"], "mediaeval": ["reuse", "text_image"]}
    for d, chans in home.items():
        for split in ("val", "test"):
            for i in range(n_per):
                y = float(rng.integers(0, 2))
                row = {c: np.nan for c in F.FEATURE_COLUMNS}
                for ch in chans:
                    for col in F.CHANNEL_FEATURES[ch]:
                        row[col] = float(np.clip(0.25 + 0.5 * y + rng.normal(0, 0.15), 0, 1))
                row["has_text"] = 1.0
                row["has_image"] = float("text_image" in chans)
                row["has_video"] = 0.0
                rows.append({"item_id": f"{d}_{split}_{i}", "dataset": d, "split": split,
                             "topic_group": f"{d}:t{i % 25}", "credible": y, "channels": "x", **row})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def fitted():
    table = synthetic_table()
    grid = T.PARAM_GRID
    T.PARAM_GRID = [{"num_leaves": 7, "learning_rate": 0.1}]  # keep the test fast
    try:
        model, report = T.fit_meta(table)
    finally:
        T.PARAM_GRID = grid
    return table, model, report


def test_fit_meta_learns_and_outputs_probabilities(fitted):
    table, model, report = fitted
    test = table[table["split"] == "test"]
    p = model.predict(test[F.FEATURE_COLUMNS].to_numpy(float))
    assert p.min() >= 0 and p.max() <= 1
    assert T.binary_metrics(test["credible"], p)["roc_auc"] > 0.9
    assert report["train_rows"] == 1200 and report["best_params"]["num_leaves"] == 7


def test_meta_model_save_load_roundtrip(fitted, tmp_path):
    table, model, _ = fitted
    paths = (tmp_path / "m.txt", tmp_path / "c.joblib", tmp_path / "i.json")
    model.save(*paths)
    again = MetaModel.load(*paths)
    X = table[F.FEATURE_COLUMNS].to_numpy(float)[:20]
    assert np.allclose(again.predict(X), model.predict(X))
    assert json.loads(paths[2].read_text())["feature_names"] == F.FEATURE_COLUMNS


def test_evaluate_report(fitted, tmp_path, monkeypatch):
    table, model, _ = fitted
    monkeypatch.setattr(T, "REPORTS", tmp_path)
    rep = T.evaluate(model, table)
    assert rep["meta"]["roc_auc"] > 0.9 and set(rep["per_dataset"]) == set(T.DATASETS)
    assert rep["fairness"]["n_text_only"] > 0 and rep["fairness"]["n_multimodal"] > 0 and "ks_pvalue" in rep["fairness"]
    assert rep["missing_modality"]["n"] > 0 and set(rep["ablation"]) == set(F.CHANNELS)
    assert (tmp_path / "figures" / "meta_reliability.png").exists()


def test_fit_meta_rejects_tiny_table():
    with pytest.raises(ValueError):
        T.fit_meta(synthetic_table(n_per=5))


# --- pipeline with injected components -------------------------------------------------------

class FakeSearcher:
    def search(self, query, k=5, before=None, reranker=None):
        return [Hit(Passage("p1", "Official data confirm the figure.", "https://x.gov/a", "x.gov", "2020-01-01"), 2.0, 1, 1)]


class FakeVerifier:
    def verify(self, claim, hits, **kw):
        return Verdict("Supported", {"Supported": 0.9}, 0.9, 0.95, [])


class FakeSource:
    def score(self, url, headline, **social):
        return SourceResult(0.7, 0.4, {"reliability": 0.9, "clickbait_hits": 0.0, "capital_ratio": 0.1}, ["ok source"])


def make_pipeline(**extra):
    comps = dict(scorer=HeuristicCheckWorthiness(), searcher=FakeSearcher(), verifier=FakeVerifier(),
                 reranker=None, source=FakeSource(), meta=None)
    comps.update(extra)
    return Pipeline(components=comps, autoload=False)


def test_pipeline_text_only_skips_media_and_is_json_safe():
    rep = make_pipeline().analyze(NewsItem(
        text="The unemployment rate fell to 6.1% in March 2021, the ministry announced.",
        source_url="https://www.x.gov/news", publish_date="2021-04-02"))
    assert rep["claims"] and rep["claims"][0]["verdict"] == "Supported"
    assert any(s.startswith("text_image_match: no image") for s in rep["skipped_checks"])
    assert any(s.startswith("media_reuse: no image") for s in rep["skipped_checks"])
    assert rep["features"]["text_image_score"] is None and rep["features"]["has_image"] == 0.0
    assert rep["meta_model"].startswith("equal-weight") and 0 <= rep["score"] <= 100
    json.dumps(rep, allow_nan=False)  # no NaN leaks into the API


def test_pipeline_image_with_unavailable_channels_reports_why():
    img = Image.new("RGB", (64, 64), (200, 30, 30))
    rep = make_pipeline().analyze(NewsItem(text="Flood photo from Mumbai on 3 July 2024", image=img))
    assert any(s.startswith("text_image_match: not loaded") for s in rep["skipped_checks"])
    assert any(s.startswith("media_reuse: not loaded") for s in rep["skipped_checks"])
    assert rep["features"]["has_image"] == 1.0


def test_pipeline_uses_meta_model_when_present(fitted):
    _, model, _ = fitted
    rep = make_pipeline(meta=model).analyze(NewsItem(text="Parliament passed the data protection bill on 9 August 2023.",
                                                     publish_date="2023-08-10"))  # a real date comparison = evidence
    assert rep["meta_model"] == "lightgbm+isotonic" and rep["band"] in {"Low", "Questionable", "Mostly credible", "High"}


def test_pipeline_nothing_to_score():
    rep = Pipeline(components={"meta": None}, autoload=False).analyze(NewsItem())
    assert rep["score"] is None and rep["band"] == "Unverified" and rep["abstained"]


def test_pipeline_abstains_when_no_check_found_evidence(fitted):
    """Before: any text whose checks found nothing got the model's prior (60), gibberish included."""
    class OffTopicVerifier:
        def verify(self, claim, hits, **kw):
            v = Verdict("Refuted", {"Refuted": 0.9}, 0.1, 0.9, [])
            v.relevance = 0.0001
            return v

    _, model, _ = fitted
    rep = make_pipeline(verifier=OffTopicVerifier(), meta=model).analyze(NewsItem(
        text="The unemployment rate fell to 6.1% in March 2021, the ministry announced."))
    assert rep["score"] is None and rep["band"] == "Unverified" and rep["abstained"] and rep["contributions"] == []
    assert rep["reasons"][0].startswith("Not scored: no evidence about its claims was found, and no image or source URL")
    with_url = make_pipeline(verifier=OffTopicVerifier(), meta=model).analyze(NewsItem(
        text="The unemployment rate fell to 6.1% in March 2021.", source_url="https://www.x.gov/news"))
    assert with_url["score"] is not None and not with_url["abstained"]  # the source check found something


def test_pipeline_abstains_on_url_alone(fitted):
    """A URL alone rates the outlet, not the item: thehindu.com (0.90) scored 3/100 before this rule."""
    _, model, _ = fitted
    rep = make_pipeline(meta=model).analyze(NewsItem(source_url="https://www.x.gov/news"))
    assert rep["score"] is None and rep["band"] == "Unverified" and rep["abstained"]
    assert rep["reasons"][0].startswith("Not scored: only a source was given")


def test_date_check_is_evidence_only_when_compared_with_the_items_own_dates():
    assert not F.has_evidence(F.assemble({"temporal_score": 1.0}))  # claimed date, nothing to compare
    assert not F.has_evidence(F.assemble({"temporal_score": 0.8}))  # compared only with evidence-passage dates
    assert F.has_evidence(F.assemble({"temporal_score": 0.7, "gap_publish_days": 14.0}))
    assert F.has_evidence(F.assemble({"temporal_score": 0.4, "gap_earliest_copy_days": 400.0}))


def test_pipeline_abstains_when_only_the_date_check_ran(fitted):
    """Before: "On 5 March 2024 a crowd ... India Gate" + claimed date got a score, because the date check
    (compared with nothing about the item) counted as evidence."""
    class OffTopicVerifier:
        def verify(self, claim, hits, **kw):
            v = Verdict("Refuted", {"Refuted": 0.9}, 0.1, 0.9, [])
            v.relevance = 0.0001
            return v

    _, model, _ = fitted
    text = "On 5 March 2024 a crowd of protesters gathered at India Gate in New Delhi."
    rep = make_pipeline(verifier=OffTopicVerifier(), meta=model).analyze(NewsItem(text=text, claimed_date="2024-03-05"))
    assert not math.isnan(rep["features"]["temporal_score"] or math.nan)  # the check still runs and feeds the model
    assert rep["score"] is None and rep["band"] == "Unverified" and rep["abstained"]
    with_publish = make_pipeline(verifier=OffTopicVerifier(), meta=model).analyze(
        NewsItem(text=text, claimed_date="2024-03-05", publish_date="2024-02-20"))
    assert with_publish["score"] is not None and not with_publish["abstained"]  # a real comparison still counts


def test_pipeline_does_not_verify_claims_that_name_nothing():
    from src.claims.extract import load_nlp

    nlp = load_nlp()
    if nlp is None or "ner" not in nlp.pipe_names:
        pytest.skip("spaCy model en_core_web_sm not installed")

    class Everything:  # every sentence counts as check-worthy
        def score(self, sentences):
            return [0.9] * len(sentences)

    seen = []

    class RecordingVerifier(FakeVerifier):
        def verify(self, claim, hits, **kw):
            seen.append(claim)
            return super().verify(claim, hits)

    rep = make_pipeline(scorer=Everything(), verifier=RecordingVerifier()).analyze(
        NewsItem(text="They said it was true. Vaccines cause autism."))
    texts = {c["text"]: c for c in rep["claims"]}
    assert seen == ["Vaccines cause autism."]  # the vague sentence was never searched or verified
    assert texts["They said it was true."]["context"] == "missing" and texts["They said it was true."]["verdict"] is None
    assert "who or what" in texts["They said it was true."]["justification"]


def test_item_from_row_turns_parquet_nan_into_none(tmp_path):
    """Regression: parquet NaN for missing strings crashed NewsItem.input_hash()."""
    row = pd.Series({"item_id": "averitec_train_1", "dataset": "averitec", "text": "Claim text.",
                     "source_domain": float("nan"), "date": float("nan"), "topic_group": "averitec:topic:3",
                     "has_video": False, "tweet_ids": float("nan"), "media_id": float("nan")})
    item = T.item_from_row(row, {})
    assert item.source_url is None and item.publish_date is None and item.tweet_ids is None and item.media_id is None
    assert item.claimed_event is None and item.image is None and len(item.input_hash()) == 16
    me = T.item_from_row(row.copy().replace({"averitec": "mediaeval"}).set_axis(row.index), {})
    assert me.claimed_event == "averitec:topic:3"


def test_nested_component_loading_does_not_deadlock():
    """Regression: _load_text_image calls get("clip") while get() holds the lock.
    With a non-reentrant Lock the first item with an image hung forever."""
    import threading

    class P(Pipeline):
        def _load_clip(self):
            return object(), "ready"

        def _load_text_image(self):
            return ("checker", self.get("clip")), "ready"

    p = P(components={})
    done = []
    t = threading.Thread(target=lambda: done.append(p.get("text_image")), daemon=True)
    t.start()
    t.join(timeout=5)
    assert done and done[0][0] == "checker", "nested get() deadlocked"


# --- batched analysis (prefetch + replay) ----------------------------------------------------

def _batch_components():
    """Real HybridSearcher / CrossEncoderReranker / ClaimVerifier with fake models that count calls."""
    import zlib

    from src.retrieval.search import CrossEncoderReranker, HybridSearcher, tokenize
    from src.verify.model import ClaimVerifier

    calls = {"encoder": 0, "rerank": 0, "verify": 0}

    def encoder(texts):
        calls["encoder"] += 1
        out = np.zeros((len(texts), 64), np.float32)
        for i, t in enumerate(texts):
            for tok in tokenize(t):
                out[i, zlib.crc32(tok.encode()) % 64] += 1
        return out

    passages = [Passage(f"p{i}", t, f"https://s{i}.org/x", f"s{i}.org", None) for i, t in enumerate([
        "Unemployment fell to 6.1 percent in March 2021.", "The bridge collapsed in 2018 in Genoa.",
        "Parliament passed the data protection bill in August 2023.", "The repo rate was kept at 6.5 percent.",
        "ISRO launched Chandrayaan-3 in July 2023.", "Floods hit Chennai in December 2015."])]
    searcher = HybridSearcher(passages, encoder=encoder)
    calls["encoder"] = 0

    class FakeCE:
        def predict(self, pairs, batch_size=64):
            calls["rerank"] += 1
            return [len(set(tokenize(q)) & set(tokenize(t))) for q, t in pairs]

    class CountingVerifier(ClaimVerifier):
        def _infer_pairs(self, pairs):
            calls["verify"] += 1
            return np.array([[0.7, 0.1, 0.1, 0.1] if "6" in p else [0.1, 0.7, 0.1, 0.1] for _, p in pairs])

    comps = dict(scorer=HeuristicCheckWorthiness(), searcher=searcher, reranker=CrossEncoderReranker(model=FakeCE()),
                 verifier=CountingVerifier(reliability={}), source=FakeSource(), meta=None)
    return comps, calls


BATCH_ITEMS = [
    "The unemployment rate fell to 6.1% in March 2021, the ministry announced. Parliament passed the data protection bill in August 2023.",
    "ISRO launched Chandrayaan-3 in July 2023 from Sriharikota.",
    "What a shame.",
    "The repo rate was kept at 6.5% by the central bank on 8 February 2024.",
]


def test_analyze_batch_matches_per_item_and_batches_model_calls():
    items = [NewsItem(text=t, publish_date="2024-03-01") for t in BATCH_ITEMS]
    comps, calls = _batch_components()
    single = [Pipeline(components=comps, autoload=False).analyze(it) for it in items]
    per_item_calls = dict(calls)
    comps2, calls2 = _batch_components()
    batched = Pipeline(components=comps2, autoload=False).analyze_batch(items)
    assert batched == single  # identical reports: same per-item code, cached model outputs
    assert calls2 == {"encoder": 1, "rerank": 1, "verify": 1}  # one batched call per model
    assert per_item_calls["encoder"] > 1 and per_item_calls["verify"] > 1
    # caches emptied after the batch
    assert comps2["searcher"]._qmemo == {} and comps2["reranker"]._memo == {} and comps2["verifier"]._memo == {}


def test_clip_encoder_cache():
    from PIL import Image

    from src.media.text_image import ClipEncoder

    enc = ClipEncoder.__new__(ClipEncoder)  # no model download: fake the two inference calls
    enc._tmemo, enc._imemo, n = {}, {}, {"t": 0, "i": 0}

    def fake_texts(texts, batch_size=256):
        n["t"] += 1
        return np.array([[len(t), 1.0] for t in texts], np.float32)

    def fake_images(images, batch_size=64):
        n["i"] += 1
        return np.array([[im.size[0], 2.0] for im in images], np.float32)

    enc._texts, enc._images = fake_texts, fake_images
    a, b = Image.new("RGB", (40, 40)), Image.new("RGB", (50, 50))
    enc.prefetch_texts(["x", "yy", "x"])
    enc.prefetch_images([a, b, a])
    assert n == {"t": 1, "i": 1}
    assert enc.encode_texts(["yy", "x"]).tolist() == [[2, 1], [1, 1]] and enc.encode_images([b])[0, 0] == 50
    assert n == {"t": 1, "i": 1}  # served from cache
    enc.clear_memo()
    enc.encode_images([a])
    assert n["i"] == 2


def test_build_table_uses_batches_and_checkpoints(tmp_path, monkeypatch):
    import src.media.reuse as R

    monkeypatch.setattr(T, "PROCESSED", tmp_path)
    monkeypatch.setattr(T, "META_DIR", tmp_path / "meta")
    monkeypatch.setattr(T, "TABLE_PATH", tmp_path / "meta" / "t.parquet")
    monkeypatch.setattr(R, "image_files", lambda *a, **k: {})
    rows = [{"item_id": f"averitec_{s}_{i}", "split": s, "meta_eligible": True, "credible": float(i % 2),
             "text": f"t {i}", "topic_group": "g"} for s in ("val", "test") for i in range(70)]
    pd.DataFrame(rows).to_parquet(tmp_path / "averitec.parquet", index=False)

    class P:
        status, sizes = {}, []

        def analyze_batch(self, items, channels=None):
            self.sizes.append(len(items))
            return [{"features": {c: 0.5 for c in F.FEATURE_COLUMNS}} for _ in items]

    p = P()
    done = T.build_table(pipeline=p, batch_size=32, checkpoint_every=64)
    assert len(done) == 140 and p.sizes == [32, 32, 32, 32, 12]  # full sets, no downsampling by default
    p.sizes.clear()
    T.build_table(pipeline=p, batch_size=32)
    assert p.sizes == []  # everything cached


# --- fit fixes: image dropout, monotone constraints, per-dataset selection, regrouping -----

def test_image_dropout_appends_blanked_copies():
    table = synthetic_table(n_per=20)
    tr = table[table["split"] == "val"]
    aug = T.image_dropout(tr, {"image": T.IMAGE_FEATURES})
    n_img = int(tr[T.IMAGE_FEATURES].notna().any(axis=1).sum())
    assert len(aug) == len(tr) + n_img and aug["augmented"].sum() == n_img
    copies = aug[aug["augmented"]]
    assert copies[T.IMAGE_FEATURES].isna().all().all() and (copies["has_image"] == 0).all()
    # labels and groups are kept, so a copy sits in the same CV fold as its original
    orig = tr.set_index("item_id").loc[copies["item_id"]]
    assert (orig["credible"].to_numpy() == copies["credible"].to_numpy()).all()
    assert (orig["topic_group"].to_numpy() == copies["topic_group"].to_numpy()).all()


def test_missing_image_moves_scores_toward_the_middle_not_down(fitted):
    """With dropout, deleting the image costs certainty in both directions; it is not
    evidence against the item (credible scores fall, not-credible scores rise)."""
    table, model, _ = fitted
    mm = T.missing_modality_test(model, table[table["split"] == "test"])
    assert mm["n"] > 0 and mm["not_credible"]["n"] > 0
    assert mm["mean_change_points"] < 0 < mm["not_credible"]["mean_change_points"]


def test_monotone_constraints_hold(fitted):
    """Raising a constrained channel score never lowers the prediction."""
    table, model, _ = fitted
    X = table[F.FEATURE_COLUMNS].to_numpy(float)[:200]
    for col, sign in T.MONOTONE.items():
        j = F.FEATURE_COLUMNS.index(col)
        lo, hi = X.copy(), X.copy()
        lo[:, j], hi[:, j] = 0.1, 0.9
        diff = sign * (model.predict_raw(hi) - model.predict_raw(lo))
        assert (diff >= -1e-9).all(), col
    assert model.info["monotone"] == T.MONOTONE


def test_mean_dataset_auc_weights_datasets_equally():
    d = ["a"] * 4 + ["b"] * 100
    y = [0, 0, 1, 1] + [0, 1] * 50
    p = [0.1, 0.2, 0.8, 0.9] + [0.5] * 100  # a: perfect, b: chance
    mean, per = T.mean_dataset_auc(d, y, p)
    assert per == {"a": 1.0, "b": 0.5} and mean == pytest.approx(0.75)


def test_tuning_selects_by_mean_dataset_auc_and_reports_it(fitted):
    _, _, report = fitted
    assert {"cv_mean_dataset_auc", "cv_auc_by_dataset", "cv_weighted_log_loss"} <= set(report["grid"][0])
    assert report["augmented_rows"] > 0 and 0.5 < report["oof_mean_dataset_auc"] <= 1


def test_regroup_one_class_test_redeals_val_and_test_groups():
    table = synthetic_table(n_per=40)
    me = table["dataset"] == "mediaeval"
    # make every mediaeval test row credible, as in the real Phase 2 split
    table.loc[me & (table["split"] == "test"), "credible"] = 1.0
    fixed, changes = T.regroup_one_class_test(table)
    assert set(changes) == {"mediaeval"}
    test = fixed[(fixed["dataset"] == "mediaeval") & (fixed["split"] == "test")]
    assert test["credible"].nunique() == 2
    assert len(fixed) == len(table) and set(fixed["split"]) == {"val", "test"}
    # whole topic groups move: no group on both sides
    g = fixed[fixed["dataset"] == "mediaeval"].groupby("topic_group")["split"].nunique()
    assert (g == 1).all()
    # datasets with two classes in test are untouched
    other = ~me
    assert (fixed.loc[other, "split"] == table.loc[other, "split"]).all()


def test_model_fingerprint_changes_when_a_model_file_changes(tmp_path):
    import os

    from src.pipeline import MODEL_VERSION, model_fingerprint

    (tmp_path / "meta_lgbm.txt").write_text("tree v1")
    v1 = model_fingerprint(tmp_path)
    assert v1.startswith(MODEL_VERSION + "+") and v1 == model_fingerprint(tmp_path)
    (tmp_path / "meta_lgbm.txt").write_text("tree version 2")
    os.utime(tmp_path / "meta_lgbm.txt", ns=(1, 2))
    assert model_fingerprint(tmp_path) != v1


def test_fit_without_dropout_does_not_hit_read_only_array():
    """Regression: model_matrix wrote into a read-only view when no rows were appended."""
    grid = T.PARAM_GRID
    T.PARAM_GRID = [{"num_leaves": 7, "learning_rate": 0.1}]
    try:
        model, report = T.fit_meta(synthetic_table(n_per=60), dropout=False)
    finally:
        T.PARAM_GRID = grid
    assert report["augmented_rows"] == 0 and model.info["image_dropout"] is False


def test_unused_features_have_no_effect(fitted):
    table, model, _ = fitted
    X = table[F.FEATURE_COLUMNS].to_numpy(float)[:100]
    X2 = X.copy()
    for c in T.UNUSED_FEATURES:
        X2[:, F.FEATURE_COLUMNS.index(c)] = np.random.default_rng(1).random(len(X2))
    assert np.allclose(model.predict_raw(X), model.predict_raw(X2))


def test_gated_claim_score():
    assert F.gated_claim_score(1.0, 1.0) == 1.0
    assert math.isnan(F.gated_claim_score(1.0, 0.001))  # verdict on unrelated evidence: no information
    assert F.gated_claim_score(0.0, 0.5) == 0.25
    assert math.isnan(F.gated_claim_score(math.nan, 0.9))
    df = F.add_derived(pd.DataFrame({"claim_score_mean": [1.0, np.nan], "evidence_relevance": [0.2, 0.9]}))
    assert df["claim_evidence_score"].iloc[0] == pytest.approx(0.6) and np.isnan(df["claim_evidence_score"].iloc[1])


def test_dropout_covers_source_too():
    tr = synthetic_table(n_per=20)
    tr = tr[tr["split"] == "val"]
    aug = T.image_dropout(tr)
    n_img = int(tr[T.IMAGE_FEATURES].notna().any(axis=1).sum())
    n_src = int(tr[F.CHANNEL_FEATURES["source"]].notna().any(axis=1).sum())
    assert aug["augmented"].sum() == n_img + n_src
    assert aug[aug["augmented"]][F.CHANNEL_FEATURES["source"]].isna().all(axis=1).sum() >= n_src


def test_pipeline_off_topic_evidence_is_not_counted():
    class OffTopicVerifier:
        def verify(self, claim, hits, **kw):
            v = Verdict("Supported", {"Supported": 0.96}, 0.96, 0.96, [])
            v.relevance = 0.0001  # evidence about some other claim
            return v

    rep = make_pipeline(verifier=OffTopicVerifier()).analyze(NewsItem(
        text="The unemployment rate fell to 6.1% in March 2021, the ministry announced."))
    c = rep["claims"][0]
    assert c["verdict"] == "Not Enough Evidence" and c["evidence_off_topic"] and c["credibility"] is None
    assert rep["features"]["claim_evidence_score"] is None  # the meta-learner sees "not run"
    assert "did not change the score" in c["justification"]
