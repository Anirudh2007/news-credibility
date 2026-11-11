from datetime import date

import pandas as pd
import pytest

from src.claims import extract
from src.claims.extract import (
    HeuristicCheckWorthiness,
    _regex_sentences,
    build_checkworthiness_dataset,
    extract_claims,
    resolve_relative_dates,
    split_sentences,
)

TEXT = (
    "Dr. Rao said unemployment fell to 6.1% in March. I think this is a sad day for democracy. "
    "Was the bridge really built in 2018? The U.S. Army announced 200 new recruits yesterday."
)


def test_regex_splitter_handles_abbreviations_and_decimals():
    sents = [s.text for s in _regex_sentences(TEXT)]
    assert sents == [
        "Dr. Rao said unemployment fell to 6.1% in March.",
        "I think this is a sad day for democracy.",
        "Was the bridge really built in 2018?",
        "The U.S. Army announced 200 new recruits yesterday.",
    ]


def test_split_sentences_offsets_point_into_text():
    for s in split_sentences(TEXT):
        assert TEXT[s.start : s.end].strip() == s.text
    assert split_sentences("") == []


def test_heuristic_ranks_facts_above_opinions_and_questions():
    fact, opinion, question, _ = HeuristicCheckWorthiness().score([s.text for s in _regex_sentences(TEXT)])
    assert fact > 0.5 and opinion < 0.5 and question < 0.5


def test_resolve_relative_dates():
    out, dates = resolve_relative_dates("Protest held yesterday near India Gate.", date(2024, 3, 10))
    assert out == "Protest held on 2024-03-09 near India Gate." and dates == ["2024-03-09"]
    assert resolve_relative_dates("Held yesterday.", None) == ("Held yesterday.", [])


def test_extract_claims_keeps_checkworthy_only():
    claims = extract_claims(TEXT, publish_date=date(2024, 3, 10), scorer=HeuristicCheckWorthiness())
    texts = [c.text for c in claims]
    assert "I think this is a sad day for democracy." not in texts
    army = next(c for c in claims if "Army" in c.text)
    assert "2024-03-09" in army.dates and "on 2024-03-09" in army.text
    assert all(0.5 <= c.checkworthiness <= 1 for c in claims)
    assert extract_claims("   ") == []


def test_build_checkworthiness_dataset():
    df = build_checkworthiness_dataset(["A rose 5%.", "B won.", "A rose 5%."], ["I feel sad.", ""])
    assert sorted(df["label"].tolist()) == [0, 1, 1] and set(df.columns) == {"text", "label"}


@pytest.mark.skipif(
    not (extract.DEFAULT_CHECKWORTHY_DIR / "config.json").exists(), reason="fine-tuned DeBERTa not trained yet"
)
def test_deberta_scorer_outputs_probabilities():
    scores = extract.DebertaCheckWorthiness().score(["Unemployment fell to 6.1% in March.", "What a day!"])
    assert len(scores) == 2 and all(0 <= s <= 1 for s in scores)


def test_checkthat_split_names_and_table():
    from pathlib import Path

    from src.claims.prepare_data import _split_of, build_table

    assert _split_of(Path("CT24_checkworthy_english_dev-test.tsv")) == "test"
    assert _split_of(Path("CT24_checkworthy_english_dev.tsv")) == "val"
    assert _split_of(Path("CT24_checkworthy_english_test_gold.tsv")) == "test"
    ct = pd.DataFrame(
        {
            "text": ["Taxes rose 5%.", "Taxes rose 5%!", "Thank you all.", "Crime fell 3%.", ""],
            "label": [1, 1, 0, 1, 0],
            "source": "checkthat",
            "split": ["train", "train", "train", "test", "train"],
        }
    )
    av = pd.DataFrame({"text": ["Crime fell 3%"], "label": [1], "source": "averitec", "split": ["train"]})
    out = build_table(ct, av)
    assert len(out[out["split"] == "train"]) == 2  # dup merged, empty dropped, test overlap removed
    assert out.loc[out["split"] == "test", "text"].tolist() == ["Crime fell 3%."]
