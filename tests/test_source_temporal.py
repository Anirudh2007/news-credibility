import math
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.ingest.download_mediaeval import collect_urls, parse_url_rows
from src.media import reuse
from src.source.credibility import (
    FEATURE_NAMES,
    SourceChecker,
    SourceRating,
    domain_features,
    item_features,
    parse_domain,
    social_features,
    style_features,
    train_source_classifier,
)
from src.temporal.consistency import check, claimed_date_from_text, gap_days

RATINGS = {"reuters.com": SourceRating(0.95, "center"), "theonion.com": SourceRating(0.1, "satire")}


# --- MediaEval download + index from images/ -------------------------------------

def test_parse_url_rows_detects_columns_by_content():
    assert parse_url_rows("sandyA_fake_29\thttp://i.imgur.com/x.jpg\tfake\tsandy\n") == (
        "sandyA_fake_29", "http://i.imgur.com/x.jpg")
    assert parse_url_rows("123456\tboston_real_01\thttps://pbs.twimg.com/a.png") == (
        "boston_real_01", "https://pbs.twimg.com/a.png")
    assert parse_url_rows("image_id\timage_url\tlabel") is None  # header: no URL
    assert parse_url_rows("http://x.com/a.jpg\tfake") is None  # no id-like field


def test_collect_urls_skips_images_dir(tmp_path, monkeypatch):
    (tmp_path / "set_images.txt").write_text("a_fake_1\thttp://x/a.jpg\nb_real_2\thttp://x/b.jpg\na_fake_1\thttp://y/a.jpg\n")
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "notes.txt").write_text("c_fake_3\thttp://x/c.jpg\n")
    import src.ingest.download_mediaeval as dm

    monkeypatch.setattr(dm, "IMAGE_DIR", tmp_path / "images")
    assert collect_urls(tmp_path) == {"a_fake_1": "http://x/a.jpg", "b_real_2": "http://x/b.jpg"}


def test_build_mediaeval_index_reads_images_dir(tmp_path):
    rng = np.random.default_rng(0)
    for name in ("sandyA_fake_01", "boston_real_02"):
        Image.fromarray(rng.integers(0, 255, (6, 8, 3), dtype=np.uint8)).resize((160, 120)).save(tmp_path / f"{name}.jpg")
    df = pd.DataFrame({
        "media_id": ["sandyA_fake_01", "boston_real_02,missing_x_9"],
        "date": ["2012-10-30", "2013-04-16"],
        "topic_group": ["mediaeval:event:sandy", "mediaeval:event:boston"],
        "text": ["t1", "t2"],
    })
    index, missing = reuse.build_mediaeval_index(df, encoder=None, image_dir=tmp_path)
    assert {r.media_id for r in index.records} == {"sandyA_fake_01", "boston_real_02"} and missing == ["missing_x_9"]
    reuse.save_index(index, tmp_path / "idx")
    again = reuse.load_index(tmp_path / "idx")
    assert [r.phash for r in again.records] == [r.phash for r in index.records]


# --- 3F domain, style, social ------------------------------------------------------

def test_parse_domain():
    assert parse_domain("https://edition.cnn.com/x") == ("cnn.com", "com", "edition", "https")
    assert parse_domain("pib.gov.in")[0:2] == ("pib.gov.in", "gov.in")
    assert parse_domain(None) == (None, None, None, None)


def test_domain_features():
    f = domain_features("https://www.reuters.com/world", RATINGS)
    assert f["reliability"] == 0.95 and f["https"] == 1.0 and f["is_rated"] == 1.0 and f["subdomain_depth"] == 0
    assert domain_features("theonion.com", RATINGS)["is_satire"] == 1.0
    unknown = domain_features("http://news-24x7.example.com", RATINGS)
    assert math.isnan(unknown["reliability"]) and unknown["https"] == 0.0 and unknown["domain_has_digit"] == 0.0
    assert domain_features("https://pib.gov.in", RATINGS)["official_suffix"] == 1.0
    assert all(math.isnan(v) for v in domain_features(None, RATINGS).values())


def test_style_features():
    bait = style_features("SHOCKING: You won't believe what happens next!!!")
    calm = style_features("Parliament passes data protection bill")
    assert bait["clickbait_hits"] >= 2 and bait["exclamations"] == 3 and bait["second_person"] == 1
    assert bait["capital_ratio"] > calm["capital_ratio"] and calm["clickbait_hits"] == 0
    assert style_features("10 things doctors hide")["starts_with_number"] == 1.0
    assert math.isnan(style_features("")["n_words"])


def test_social_features_optional():
    assert all(math.isnan(v) for v in social_features().values())
    assert social_features(tweet_ids="1\t2\t3")["n_tweets"] == 3
    s = social_features(edges=[("r", "a"), ("r", "b"), ("a", "c"), ("c", "d")], root="r")
    assert s["cascade_size"] == 5 and s["cascade_depth"] == 3 and s["cascade_breadth"] == 2
    t0 = datetime(2024, 1, 1)
    s = social_features(timestamps={str(i): t0 + timedelta(minutes=6 * i) for i in range(120)})
    assert math.isclose(s["hours_to_100_shares"], 9.9)


def test_item_features_length_and_checker(tmp_path):
    assert len(item_features("reuters.com", "Calm headline", RATINGS)) == len(FEATURE_NAMES)
    # rule fallback: point at a missing model file so a trained models/source_clf.joblib is not picked up
    checker = SourceChecker(model=None, ratings=RATINGS, model_path=tmp_path / "none.joblib")
    good = checker.score("https://www.reuters.com/x", "Parliament passes bill")
    bad = checker.score("theonion.com", "You won't believe this shocking secret")
    assert good.score > bad.score and any("satire" in r for r in bad.reasons)
    assert math.isnan(checker.score(None, None).score)


def test_train_source_classifier_with_nan():
    rng = np.random.default_rng(0)
    n = 400
    y = rng.integers(0, 2, n)
    X = rng.standard_normal((n, len(FEATURE_NAMES)))
    X[:, FEATURE_NAMES.index("clickbait_hits")] = (1 - y) * 2 + rng.random(n)
    X[rng.random(n) < 0.5, FEATURE_NAMES.index("reliability")] = np.nan
    model, f1 = train_source_classifier(X[:300], y[:300], X[300:], y[300:])
    assert f1 > 0.9


# --- 3G temporal ------------------------------------------------------------------

def test_gap_days():
    assert gap_days(date(2024, 7, 3), "2024-07-01") == 2
    assert math.isnan(gap_days(None, "2024-07-01")) and math.isnan(gap_days(date(2024, 7, 3), None))


def test_claimed_date_from_text():
    assert claimed_date_from_text("Floods hit Mumbai on 3 July 2024") == date(2024, 7, 3)
    assert claimed_date_from_text("Protest held yesterday", publish_date="2024-03-10") == date(2024, 3, 9)
    assert claimed_date_from_text("Protest held", publish_date="2024-03-10") is None


def test_check_flags_old_media():
    r = check(claimed_date="2024-07-03", earliest_copy_date="2015-12-02", publish_date="2024-07-03")
    assert "media_predates_claim" in r.flags and r.score == pytest.approx(0.4) and r.reasons
    assert r.gaps["earliest_copy"] > 3000 and r.features()["temporal_consistent"] == 0.0


def test_check_consistent_and_tolerance():
    r = check(claimed_date="2024-07-03", earliest_copy_date="2024-07-02", exif_date="2024-07-03", publish_date="2024-07-04")
    assert r.flags == [] and r.score == 1.0 and r.confidence == 0.75


def test_check_published_before_event_and_old_evidence():
    r = check(claimed_date="2023-05-10", publish_date="2023-05-01", evidence_dates=["2019-01-01", "2020-02-02"])
    assert set(r.flags) == {"published_before_event", "evidence_much_older"} and r.score == pytest.approx(0.5)


def test_check_nan_without_claimed_date():
    r = check(publish_date="2024-01-01", earliest_copy_date="2015-01-01")
    assert math.isnan(r.score) and r.confidence == 0.0 and math.isnan(r.features()["temporal_consistent"])
