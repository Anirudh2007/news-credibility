"""India-context data (src/ingest/load_india.py) and fact-check evidence passages."""

import math

import pandas as pd
import pytest

from src.ingest.load_india import clean_ifnd, fakecovid_india, multiclaim_en, verdict_score
from src.retrieval.search import passages_from_factchecks


@pytest.mark.parametrize("label, score", [
    ("False", 0.0), ("FALSE", 0.0), ("Mostly false", 0.0), ("Misleading", 0.0), ("Altered photo", 0.0),
    ("False information.", 0.0), ("Partly false information", 0.5), ("half true", 0.5), ("Missing context", 0.5),
    ("True", 1.0), ("Explanatory", None), ("No evidence", None), ("na", None),
])
def test_verdict_words_map_to_the_credibility_scale(label, score):
    got = verdict_score(label)
    assert (math.isnan(got) if score is None else got == score)


def test_clean_ifnd_drops_generated_fakes_debunk_headlines_and_duplicates():
    raw = pd.DataFrame({
        "claim": ["Govt announces new rail line in Kerala", "Govt announces new rail line in Kerala",
                  "Rail line announced Kerala govt new",  # an AUGMENT fake
                  "Fact Check: viral video of flood in Mumbai is fake",  # a fact-checker's headline
                  "Muslims spitting on food in Delhi restaurant", "Opposition claims the bill is unfair"],
        "label": ["true", "true", "fake", "fake", "fake", "true"],
        "web": ["TRIBUNEINDIA", "TRIBUNEINDIA", "AUGMENT", "BOOMLIVE", "NEWSMOBILE", "THEPRINT"],
        "subject": ["GOVERNMENT"] * 6, "image": ["x"] * 6, "split": ["test"] * 6,
    })
    out, steps = clean_ifnd(raw)
    assert steps["after_dropping_generated_fakes"] == 5
    assert steps["debunk_wording_dropped"] == {"true": 1, "fake": 1}  # same filter for both labels
    assert steps["duplicates_dropped"] == 1
    assert sorted(out["text"]) == ["Govt announces new rail line in Kerala", "Muslims spitting on food in Delhi restaurant"]
    assert set(out["credible"]) == {0.0, 1.0}


def _shared_schema(**cols):
    n = len(next(iter(cols.values())))
    base = {"claim": ["c"] * n, "label": ["False"] * n, "split": ["train"] * n, "fact_check_url": ["https://www.factly.in/x"] * n,
            "date": ["2020/03/23"] * n, "article_title": ["na"] * n, "country": ["India"] * n, "language": ["en"] * n,
            "claim_en": ["c"] * n}
    base.update(cols)
    return pd.DataFrame(base)


def test_fakecovid_keeps_indian_english_fact_checks_with_iso_dates():
    raw = _shared_schema(country=["India", "Spain", "India"], language=["en", "es", "hi"], claim=["a", "b", "c"])
    out = fakecovid_india(raw)
    assert list(out["text"]) == ["a"] and out.loc[0, "date"] == "2020-03-23" and out.loc[0, "fact_checker"] == "factly.in"


def test_multiclaim_needs_an_english_claim_and_a_verdict():
    raw = _shared_schema(claim_en=["English claim", "na", "Another"], label=["False information", "False", "Explanatory"])
    out = multiclaim_en(raw)
    assert list(out["text"]) == ["English claim"]


def test_fact_check_passages_use_the_verifier_format_and_keep_dates():
    df = pd.DataFrame({"item_id": ["fc1", "fc2"], "text": ["5G towers spread COVID-19 in Mumbai", "No verdict here"],
                       "credible": [0.0, float("nan")], "label_raw": ["False", "Explanatory"],
                       "fact_checker": ["boomlive.in", "factly.in"], "fact_check_url": ["https://boomlive.in/a", None],
                       "date": ["2020-04-02", None], "title": ["No, 5G does not spread the virus", None]})
    ps = passages_from_factchecks(df)
    assert len(ps) == 1
    p = ps[0]
    assert p.text.startswith("Q: Is it true that 5G towers spread COVID-19 in Mumbai? A: No. boomlive.in rated this claim False.")
    assert p.date == "2020-04-02" and p.domain == "boomlive.in" and p.source_url == "https://boomlive.in/a"
