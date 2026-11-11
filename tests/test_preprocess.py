import math

import pandas as pd
import pytest

from src.preprocess.clean_text import clean_text, extract_dates, is_english, parse_date
from src.preprocess.dedup import drop_exact_duplicates, drop_near_duplicates
from src.preprocess.split import group_split
from src.preprocess.unify_labels import map_averitec, map_fakeddit, map_fakenewsnet, map_mediaeval, unify_labels


# --- clean_text ---------------------------------------------------------------

def test_clean_text_strips_html_urls_emojis_whitespace():
    raw = "<p>Breaking&amp;  news 🔥🔥</p>\n<script>x()</script> see https://t.co/abc   www.x.com/y now"
    assert clean_text(raw) == "Breaking& news see now"


def test_clean_text_removes_boilerplate_lines():
    assert clean_text("Real sentence.\nRead more at our site\nAdvertisement") == "Real sentence."


def test_clean_text_non_string():
    assert clean_text(None) == "" and clean_text(float("nan")) == ""


def test_is_english():
    assert is_english("The government announced a new policy on public transport today.")
    assert is_english("Senate passes bipartisan infrastructure bill")  # langdetect says 'fr'
    assert not is_english(
        "El gobierno anunció hoy una nueva política de transporte público para todas las ciudades del país "
        "durante los próximos cinco años."
    )
    assert not is_english("सरकार ने नई नीति की घोषणा की")  # short, non-Latin script


# --- dates --------------------------------------------------------------------

@pytest.mark.parametrize(
    "value,expected",
    [
        ("25-8-2020", "2020-08-25"),  # AVeriTeC, day first
        ("05-08-2020", "2020-08-05"),
        ("2020-08-05", "2020-08-05"),  # ISO not flipped
        ("March 12, 2020", "2020-03-12"),
        (1_500_000_000, "2017-07-14"),  # Fakeddit created_utc
        ("1500000000.0", "2017-07-14"),
        ("not a date", None),
        (None, None),
        (float("nan"), None),
    ],
)
def test_parse_date(value, expected):
    assert parse_date(value) == expected


def test_extract_dates():
    text = "Shared on 12 March 2020, but the photo is from 2018-06-01 and 3/4/2017."
    assert extract_dates(text) == ["2020-03-12", "2018-06-01", "2017-04-03"]


# --- labels -------------------------------------------------------------------

def test_averitec_mapping():
    assert map_averitec("Supported").credible == 1.0
    assert map_averitec("Refuted").credible == 0.0
    assert map_averitec("Conflicting Evidence/Cherrypicking").credible == 0.5
    nee = map_averitec("Not Enough Evidence")
    assert math.isnan(nee.credible) and not nee.meta_eligible


def test_fakeddit_mapping_codes_and_names():
    assert map_fakeddit(0).credible == 1.0
    assert map_fakeddit("true").credible == 1.0
    r = map_fakeddit("false connection")
    assert r.credible == 0.0 and r.reason_tag == "false_connection"
    assert map_fakeddit("5").reason_tag == "manipulated_content"


def test_binary_mappings():
    assert map_fakenewsnet("real").credible == 1.0 and map_fakenewsnet("Fake").credible == 0.0
    assert map_mediaeval("fake").credible == 0.0 and map_mediaeval("humor").reason_tag == "humor"


def test_unknown_label_raises():
    with pytest.raises(ValueError):
        map_averitec("Mostly True")


def test_unify_labels_frame():
    df = pd.DataFrame({"text": ["a", "b"], "label": ["real", "fake"]})
    out = unify_labels(df, "fakenewsnet")
    assert list(out["credible"]) == [1.0, 0.0] and "label" not in out


# --- dedup and split ----------------------------------------------------------

def test_dedup():
    df = pd.DataFrame(
        {
            "text": [
                "Senate passes the infrastructure bill today",
                "senate passes the infrastructure bill today!",  # exact after normalising
                "Senate passes the infrastructure bill today in Washington",  # near
                "Celebrity spotted at coffee shop",
                "",
                "",  # empty texts never merged
            ]
        }
    )
    exact = drop_exact_duplicates(df)
    assert len(exact) == 5
    near = drop_near_duplicates(exact, threshold=0.7)
    assert near["text"].tolist() == ["Senate passes the infrastructure bill today", "Celebrity spotted at coffee shop", "", ""]


def test_group_split_keeps_groups_whole_and_hits_ratios():
    groups = pd.Series([f"g{i // 10}" for i in range(1000)])  # 100 groups of 10
    split = group_split(groups)
    assert (pd.DataFrame({"g": groups, "s": split}).groupby("g")["s"].nunique() == 1).all()
    frac = split.value_counts(normalize=True)
    assert abs(frac["train"] - 0.70) < 0.02 and abs(frac["val"] - 0.15) < 0.02 and abs(frac["test"] - 0.15) < 0.02


# --- loaders (no raw data touched) ---------------------------------------------

def test_mediaeval_event_from_media_id():
    from src.ingest.load_raw import _event_from_media_id

    assert _event_from_media_id("sandyA_fake_35,sandyA_fake_36") == "sandy"
    assert _event_from_media_id("boston_real_01") == "boston"
    assert _event_from_media_id(None) == "unknown"


def test_fakeddit_balanced_sample():
    from src.ingest.load_raw import _balanced_sample

    df = pd.DataFrame({"2_way_label": ["0"] * 900 + ["1"] * 100, "x": range(1000)})
    out = _balanced_sample(df, 200, seed=1)
    assert out["2_way_label"].value_counts().to_dict() == {"0": 100, "1": 100}
    assert _balanced_sample(df, 5000, seed=1) is df
