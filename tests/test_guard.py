"""Input guard (src/ingest/guard.py): refuse unreadable text, never refuse real headlines.
A small English text stands in for the evidence passages, so no models/ file is read."""

import pytest

from src.ingest.guard import CharModel, InputGuard, _strip_social

ENGLISH = (
    "The government announced on Monday that the new rules will start in March. The minister said the plan "
    "would help farmers and small businesses across the country. Officials confirmed that the report was "
    "published by the health department after a review of the data. Police arrested two people in the city "
    "after the protest. The court said the decision was final and the election will be held next year. "
    "Prices of food and fuel rose again this month, according to the statistics office. "
) * 3


@pytest.fixture(scope="module")
def guard():
    return InputGuard(corpus=[ENGLISH], words=[])


def test_character_model_finds_english_more_likely_than_letter_soup():
    m = CharModel([ENGLISH])
    assert m.nll("The minister said the report was published") < m.nll("qxzv jkwp fghq zzxv wqkj")


@pytest.mark.parametrize("text, reason", [
    ("😂😂😂🔥🔥", "not_text"),
    ("12345 67890 !!!", "not_text"),
    ("http://t.co/abc @someone", "not_text"),
    ("BREAKING BREAKING BREAKING BREAKING", "repetitive"),
    ("aaaaaaaaaaaaaaaaaa", "repetitive"),
    ("ok", "too_short"),
    ("मोदी ने नया नियम बनाया", "not_english"),
])
def test_rules_need_no_corpus(text, reason):
    r = InputGuard(corpus=[]).check(text)
    assert not r.ok and r.reason == reason and r.message


def test_letter_soup_is_refused_and_real_sentences_pass(guard):
    soup = guard.check("asdfghjkl qwertyuiop zxcvbnm lkjhg")
    assert not soup.ok and soup.reason == "gibberish" and soup.details["english_word_share"] == 0
    assert guard.check("The minister said the new rules will help farmers").ok


def test_rare_names_alone_are_not_gibberish(guard):
    """Letter-level surprise alone would refuse Indian names; known words must be missing too."""
    r = guard.check("The minister said Chandrayaan launched from Sriharikota")
    assert r.ok and r.details["char_nll"] > 0


def test_caption_with_image_may_be_short_or_playful(guard):
    assert not guard.check("Hedgehog").ok  # alone: nothing to check
    assert guard.check("Hedgehog", caption=True).ok
    assert guard.check("Ribbit ribbit", caption=True).ok
    assert not guard.check("मोदी ने नया नियम बनाया", caption=True).ok  # another language is still refused


def test_tweets_are_judged_without_links_handles_and_hash_signs():
    assert _strip_social("RT @user Look #PrayForBoston http://t.co/x") == "Look Pray For Boston"


def test_empty_text_is_left_to_the_api():
    assert InputGuard(corpus=[]).check("   ").ok  # image-only or URL-only requests have no text
