"""URL inputs (2026-11-05): source = outlet rating, platforms not rated, article pages read for their headline."""

import math

import pytest

from src.ingest import fetch_article as FA
from src.source.credibility import SourceChecker, SourceRating, is_platform
from test_api import client  # noqa: F401  (the API fixture with a fake pipeline: no models, no network)

RATINGS = {"thehindu.com": SourceRating(0.9, "news"), "infowars.com": SourceRating(0.1, "fake"),
           "theonion.com": SourceRating(0.1, "satire")}


def test_rating_mode_uses_the_rating_and_ignores_headline_style():
    c = SourceChecker(ratings=RATINGS, mode="rating")
    calm, shouty = "Government announces new policy on rural roads", "SHOCKING!!! You won't believe what happened"
    assert c.score("https://www.thehindu.com/a/b.ece").score == pytest.approx(0.9)
    # before: clickbait raised the classifier's score (thehindu 0.62 -> 0.94, infowars 0.49 -> 0.66)
    assert c.score("thehindu.com", calm).score == c.score("thehindu.com", shouty).score
    assert c.score("infowars.com", shouty).score < c.score("thehindu.com", shouty).score
    onion = c.score("theonion.com")
    assert onion.score == pytest.approx(0.1) and "satire" in onion.reasons[0]
    # headline features are still computed (the meta-learner reads clickbait_hits separately)
    assert c.score("thehindu.com", shouty).features["clickbait_hits"] > 0


def test_platforms_and_unrated_domains_get_no_source_score():
    c = SourceChecker(ratings=RATINGS, mode="rating")
    for url in ("https://twitter.com/a/status/1", "x.com/a", "https://www.youtube.com/watch?v=1", "bit.ly/x",
                "m.facebook.com/post/1", "someblog.blogspot.com"):
        r = c.score(url)
        assert math.isnan(r.score) and "platform" in r.reasons[0], url
    r = c.score("breaking-news-india247.com")
    assert math.isnan(r.score) and "not in the ratings list" in r.reasons[0]
    assert is_platform("m.facebook.com") and not is_platform("thehindu.com")


PAGE = """<html><head>
<title>Old title tag - The Hindu</title>
<meta property="og:site_name" content="The Hindu">
<meta property="og:title" content="ISRO launches Chandrayaan-3 from Sriharikota - The Hindu">
<meta name="description" content="The lunar mission lifted off at 2.35 p.m. on Friday.">
<meta property="article:published_time" content="2023-07-14T15:05:00+05:30">
</head><body><p>body</p></body></html>"""


def test_parse_html_reads_headline_summary_and_date():
    a = FA.parse_html(PAGE, "https://www.thehindu.com/sci-tech/x.ece")
    assert a.title == "ISRO launches Chandrayaan-3 from Sriharikota"  # site suffix dropped
    assert a.published == "2023-07-14"
    assert a.text() == "ISRO launches Chandrayaan-3 from Sriharikota. The lunar mission lifted off at 2.35 p.m. on Friday."


def test_parse_html_json_ld_date_and_title_fallback():
    page = ('<html><head><title>Rs 2000 note to carry a GPS chip | Fact Check</title>'
            '<script type="application/ld+json">{"@graph":[{"@type":"NewsArticle","datePublished":"2016-11-09"}]}'
            '</script></head></html>')
    a = FA.parse_html(page, "https://example.org/a")
    assert a.title == "Rs 2000 note to carry a GPS chip" and a.published == "2016-11-09" and a.text() == a.title


def test_article_url_rules():
    assert FA.is_article_url("https://www.thehindu.com/news/x.ece")
    assert not FA.is_article_url("thehindu.com") and not FA.is_article_url("https://www.thehindu.com/")
    assert not FA.is_article_url("ftp://x.org/a")


def test_fetch_refuses_private_addresses_and_platforms():
    for url in ("http://127.0.0.1/admin", "http://localhost/x", "http://192.168.1.1/a", "http://169.254.169.254/latest"):
        a = FA.fetch_article(url)
        assert a.error and "not allowed" in a.error and a.title is None, url
    assert "social-media" in FA.fetch_article("https://twitter.com/a/status/1").error


# --- API: a URL without text is read from the page ------------------------------------------------

def test_api_reads_the_article_when_only_a_url_is_given(client, monkeypatch):
    monkeypatch.setattr(FA, "fetch_article", lambda url, timeout=FA.TIMEOUT_S: FA.parse_html(PAGE, url))
    r = client.post("/api/v1/analyze", data={"url": "https://www.thehindu.com/sci-tech/x.ece"})
    assert r.status_code == 200
    item = client.fake.items[-1]
    assert item.text.startswith("ISRO launches Chandrayaan-3") and item.publish_date == "2023-07-14"
    body = r.json()
    assert body["fetched_article"]["title"].startswith("ISRO") and body["fetched_article"]["error"] is None


def test_api_falls_back_to_the_url_alone_when_the_page_cannot_be_read(client, monkeypatch):
    monkeypatch.setattr(FA, "fetch_article", lambda url, timeout=FA.TIMEOUT_S: FA.Article(url=url, error="page returned HTTP 403"))
    r = client.post("/api/v1/analyze", data={"url": "https://www.thehindu.com/blocked.ece"})
    assert r.status_code == 200 and client.fake.items[-1].text is None
    assert "was not checked" in " ".join(r.json()["reasons"])


def test_api_does_not_fetch_when_text_is_given_or_for_a_bare_domain(client, monkeypatch):
    calls = []
    monkeypatch.setattr(FA, "fetch_article", lambda url, timeout=FA.TIMEOUT_S: calls.append(url) or FA.Article(url=url))
    client.post("/api/v1/analyze", data={"text": "ISRO launched Chandrayaan-3 in July 2023.",
                                         "url": "https://www.thehindu.com/x.ece"})
    client.post("/api/v1/analyze", data={"url": "thehindu.com"})
    assert calls == []


def test_api_does_not_check_a_fact_checkers_headline_as_a_claim(client, monkeypatch):
    """BOOM's 'Deepfakes Of Jaishankar, Lavrov Fuel False Claims ...' scored 35 when read as a claim."""
    import api.main as M

    monkeypatch.setattr(M, "_ratings", lambda: {"boomlive.in": SourceRating(0.9, "fact-check")})
    monkeypatch.setattr(FA, "fetch_article", lambda url, timeout=FA.TIMEOUT_S: FA.parse_html(PAGE, url))
    r = client.post("/api/v1/analyze", data={"url": "https://www.boomlive.in/fact-check/some-rumour-123"})
    assert r.status_code == 200 and client.fake.items[-1].text is None
    assert "fact-check article" in r.json()["fetched_article"]["error"]
