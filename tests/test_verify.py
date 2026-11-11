import json
import math

import numpy as np
import pandas as pd
import pytest

from src.retrieval.search import Hit, Passage
from src.verify import model as vm
from src.verify.model import (
    LABELS,
    ClaimVerifier,
    aggregate,
    build_pairs,
    class_weights,
    credibility_from_probs,
    passage_weights,
    reliability_of,
)

QUESTIONS = json.dumps(
    [
        {"question": "What was the rate?", "answers": [{"answer": "6.1%", "answer_type": "Extractive"}]},
        {"question": "Did it fall?", "answers": [{"answer": "Yes", "answer_type": "Boolean", "boolean_explanation": "By 0.2."}]},
    ]
)


def test_build_pairs_modes():
    df = pd.DataFrame(
        {"text": ["Rate fell to 6.1%"], "raw_label": ["Supported"], "split": ["train"], "questions_json": [QUESTIONS]}
    )
    both = build_pairs(df)
    assert both["kind"].tolist() == ["concat", "passage", "passage"]
    assert both.loc[0, "evidence"] == "Q: What was the rate? A: 6.1% Q: Did it fall? A: Yes. By 0.2."
    assert set(both["label"]) == {0}
    assert len(build_pairs(df, "concat")) == 1
    with pytest.raises(ValueError):
        build_pairs(df, "bogus")


def test_class_weights_upweight_rare_classes():
    w = class_weights([1] * 60 + [0] * 25 + [2] * 8 + [3] * 7)
    assert w[3] > w[2] > w[0] > w[1] and math.isclose(w.mean(), 1.0)


def test_reliability_lookup():
    table = {"cnn.com": 0.8, "theonion.com": 0.1}
    assert reliability_of("edition.cnn.com", table) == 0.8
    assert reliability_of("www.theonion.com", table) == 0.1
    assert reliability_of("unknown.org", table) == vm.DEFAULT_RELIABILITY
    assert reliability_of(None, table) == vm.DEFAULT_RELIABILITY


def test_passage_weights_combine_relevance_and_reliability():
    w = passage_weights([2.0, 2.0], [0.9, 0.1])
    assert math.isclose(w.sum(), 1.0) and math.isclose(w[0], 0.9)
    w = passage_weights([5.0, -5.0], [0.5, 0.5])
    assert w[0] > 0.99


def test_aggregate_and_credibility():
    probs = np.array([[0.9, 0.05, 0.03, 0.02], [0.1, 0.8, 0.05, 0.05]])
    p, w = aggregate(probs, [3.0, 0.0], [0.9, 0.9])
    assert np.argmax(p) == 0 and math.isclose(p.sum(), 1.0) and w[0] > w[1]
    score, conf = credibility_from_probs(np.array([0.0, 0.0, 0.0, 1.0]))  # all Conflicting
    assert score == 0.5 and conf == 1.0
    score, conf = credibility_from_probs(np.array([0.0, 0.0, 1.0, 0.0]))  # all NEE
    assert math.isnan(score) and conf == 0.0


def _hit(pid, score, domain):
    return Hit(Passage(pid, f"text {pid}", domain=domain), score, None, None)


def test_verifier_with_injected_model():
    fake = lambda claim, passages: np.array(  # noqa: E731
        [[0.8, 0.1, 0.05, 0.05] if "a" in p else [0.1, 0.8, 0.05, 0.05] for p in passages]
    )
    v = ClaimVerifier(predict_fn=fake, reliability={"good.org": 0.95, "bad.net": 0.05})
    verdict = v.verify("claim", [_hit("a", 1.0, "good.org"), _hit("b", 1.0, "bad.net")])
    assert verdict.label == "Supported" and 0.5 < verdict.score <= 1
    assert verdict.evidence[0].passage_id == "a" and math.isclose(sum(e.weight for e in verdict.evidence), 1.0)
    # relevance = sigmoid of the best re-ranker score
    assert verdict.relevance == pytest.approx(1 / (1 + math.exp(-1.0)))


def test_no_evidence_gives_nan_not_zero():
    verdict = ClaimVerifier(predict_fn=lambda c, p: None, reliability={}).verify("claim", [])
    assert verdict.label == LABELS[2] and math.isnan(verdict.score) and verdict.confidence == 0.0
    assert math.isnan(verdict.relevance)


@pytest.mark.skipif(not (vm.DEFAULT_MODEL_DIR / "config.json").exists(), reason="verifier not trained yet")
def test_trained_verifier_outputs_distribution():
    p = ClaimVerifier().predict("Unemployment fell to 6.1%.", ["The rate fell to 6.1 percent in March."])
    assert p.shape == (1, 4) and math.isclose(float(p.sum()), 1.0, rel_tol=1e-4)
