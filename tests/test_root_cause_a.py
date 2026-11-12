"""Root cause A (reports/known_issues.md): a fact-check of a related rumour counts against a true claim.

Measured on 2026-11-12 (src/fusion/india_tune.py `ROOT_CAUSE_A_DECISION`, reports/root_cause_a_eval.json): an
entailment gate between the two claims fixes the known cases but fails the decision rule on the India dev set,
so it is not switched on. These tests reproduce KI-10, KI-11 and FB-04 offline as they are served today, and pin
the gate that was measured. Similarities and entailment values are the ones recorded for the real evidence.
"""

import math

import numpy as np

from src.retrieval.search import Hit, Passage
from src.verify import model as vm
from src.verify.model import ClaimEntailment, ClaimMatcher, ClaimVerifier, entailment_allows, verdict_from

REFUTE = [0.02, 0.9, 0.05, 0.03]
SUPPORT = [0.9, 0.02, 0.05, 0.03]

# our claim -> (claim the fact-check judged, cosine, P(ours entails it), P(it entails ours)), recorded 2026-11-12
KNOWN = {
    "The Eiffel Tower is located in Paris, France.":  # KI-10 (AFP Fact Check: False)
        ("Video shows the Eiffel Tower collapsing after being set ablaze", 0.60, 0.00, 0.01),
    "Ratan Tata died in October 2024.":  # KI-11 (BOOM: False)
        ("Two viral videos circulating online show the last public appearances of Ratan Tata before his death",
         0.73, 0.00, 0.01),
    "Barack Obama was born in Hawaii.":  # FB-04 (USA Today: False)
        ("Image of Barack Obama's birth certificate shows he was born in Kenya", 0.72, 0.00, 0.00),
}
SAME = ("The earth is flat.", "The Earth is flat.", 1.00, 0.99, 0.99)  # the same claim: must keep refuting


def fact_check(judged: str) -> Hit:
    text = f"Q: Is it true that {judged}? A: No. A fact-checker rated this claim False."
    return Hit(Passage("gfc:https://example.org/fc", text, "https://example.org/fc", "factcheck.afp.com"), 1.0, None, None)


def matcher_with(cosines: dict[str, float]) -> ClaimMatcher:
    """Embeddings in the plane: our claim on the x axis, each judged claim at the recorded angle."""
    def enc(texts):
        return np.array([[cosines[t], math.sqrt(1 - cosines[t] ** 2)] if t in cosines else [1.0, 0.0] for t in texts])

    return ClaimMatcher(encoder=enc, model=None)


def verifier_for(claim: str, judged: str, cos: float, fwd: float, bwd: float, **kw) -> ClaimVerifier:
    table = {(claim, judged): fwd, (judged, claim): bwd}
    return ClaimVerifier(predict_fn=lambda c, ps: np.array([REFUTE for _ in ps]), reliability={},
                         matcher=matcher_with({judged: cos}),
                         entailer=ClaimEntailment(predict_fn=lambda pairs: [table[p] for p in pairs]), **kw)


def test_known_cases_are_still_refuted_as_served():
    """KI-10, KI-11, FB-04: open. Similar (cosine >= 0.55) is taken for the same claim."""
    assert vm.ENTAIL_MODE == "off" and vm.MATCH_MIN == 0.55
    for claim, (judged, cos, fwd, bwd) in KNOWN.items():
        v = verifier_for(claim, judged, cos, fwd, bwd).verify(claim, [fact_check(judged)])
        assert v.label == "Refuted" and v.score < 0.1 and v.evidence[0].counted, claim


def test_entailment_gate_leaves_the_related_fact_check_out():
    for mode in ("directional", "both"):
        for claim, (judged, cos, fwd, bwd) in KNOWN.items():
            v = verifier_for(claim, judged, cos, fwd, bwd, entail_mode=mode, entail_min=0.5).verify(
                claim, [fact_check(judged)])
            assert v.label == "Not Enough Evidence" and math.isnan(v.score) and not v.evidence[0].counted, (mode, claim)


def test_entailment_gate_keeps_a_fact_check_of_the_same_claim():
    claim, judged, cos, fwd, bwd = SAME
    for mode in ("off", "directional", "both"):
        v = verifier_for(claim, judged, cos, fwd, bwd, entail_mode=mode, entail_min=0.5).verify(claim, [fact_check(judged)])
        assert v.label == "Refuted" and v.evidence[0].counted, mode


def test_direction_follows_what_the_passage_says():
    # "B is false" refutes A only if A implies B; "B is true" supports A only if B implies A.
    a_implies_b, b_implies_a = [0.9, 0.1], [0.1, 0.9]
    assert entailment_allows(REFUTE, a_implies_b, "directional", 0.5)
    assert not entailment_allows(REFUTE, b_implies_a, "directional", 0.5)  # a more specific rumour was false
    assert entailment_allows(SUPPORT, b_implies_a, "directional", 0.5)
    assert not entailment_allows(SUPPORT, a_implies_b, "directional", 0.5)
    assert not entailment_allows(REFUTE, a_implies_b, "both", 0.5) and entailment_allows(REFUTE, [0.9, 0.9], "both", 0.5)


def test_gate_is_off_unless_asked_for():
    hits = [fact_check("Video shows the Eiffel Tower collapsing after being set ablaze")]
    plain = verdict_from(hits, np.array([REFUTE]), [0.6], [0.5])
    with_values = verdict_from(hits, np.array([REFUTE]), [0.6], [0.5], entail=np.array([[0.0, 0.01]]))
    assert plain.label == with_values.label == "Refuted" and plain.score == with_values.score
    assert ClaimVerifier(predict_fn=lambda c, ps: np.array([REFUTE for _ in ps]), reliability={}).entailer is None


def test_news_and_wikipedia_passages_are_not_tested_for_entailment():
    news = Hit(Passage("news:1", "voanews.com reported (2024-10-10): Ratan Tata, Indian industry legend, dies at 86",
                       None, "voanews.com"), 2.0, None, None)
    v = verdict_from([news], np.array([SUPPORT]), [np.nan], [0.9], entail=np.array([[np.nan, np.nan]]),
                     entail_mode="directional")
    assert v.label == "Supported" and v.evidence[0].counted


# --- Round 2 (2026-11-12): mode "same", the two claims say the same thing -----------------------------------------
# (our claim, the claim the fact-check judged, its answer to its own claim, cosine, P(ours entails it),
#  P(ours contradicts it), P(it entails ours), P(it contradicts ours), may it count?). Values recorded 2026-11-12.
SAME_MODE = [
    ("The Eiffel Tower is located in Paris, France.",  # KI-10
     "Video shows the Eiffel Tower collapsing after being set ablaze", "No", 0.60, 0.00, 0.01, 0.01, 0.00, False),
    ("Ratan Tata died in October 2024.",  # KI-11
     "Two viral videos circulating online show the last public appearances of Ratan Tata before his death",
     "No", 0.73, 0.00, 0.00, 0.01, 0.02, False),
    ("Barack Obama was born in Hawaii.",  # FB-04: the two claims contradict each other; "No" to one is not "No" to the other
     "Image of Barack Obama's birth certificate shows he was born in Kenya", "No", 0.72, 0.00, 1.00, 0.00, 1.00, False),
    ("ISRO successfully launches Chandrayaan-3 lunar mission.",  # a fact-check of footage, not of the event
     "This video shows ISRO's Chandrayaan-3 being launched from Shriharikota's Satish Dhawan Space Centre",
     "No", 0.73, 0.00, 0.00, 0.02, 0.00, False),
    ("Pope Francis died in April 2025.",  # implies ours, but the wording is not close: a false detail on a true fact
     'The animated TV show "The Simpsons" accurately predicted the death of Pope Francis on April 21, 2025',
     "No", 0.62, 0.00, 0.00, 0.97, 0.00, False),
    ("Pope Francis died in April 2025.",  # a verdict that has expired: "False" was right before he died
     "Pope Francis had passed away", "No", 0.81, 0.95, 0.02, 0.01, 0.01, False),
    ("The earth is flat.", "The Earth is flat.", "No", 1.00, 0.99, 0.00, 0.99, 0.00, True),
    ("Barack Obama was born in Kenya.",  # the checked claim says everything ours says, in close wording
     "Image of Barack Obama's birth certificate shows he was born in Kenya", "No", 0.89, 0.04, 0.00, 0.99, 0.00, True),
    ("5G spreads the coronavirus.",
     "5G WiFi networks could be responsible for the rapid spread of the new coronavirus", "No", 0.84, 0.01, 0.00, 0.89, 0.01, True),
    ("Bill Gates is using COVID-19 vaccines to implant microchips in people.",  # ours says everything it says, very close wording
     "Bill Gates is using vaccines to implant a microchip in humans", "No", 0.92, 0.99, 0.00, 0.00, 0.00, True),
    ("Vaccines cause autism.",  # a fact-check that CONFIRMED the opposite claim
     "Vaccines do not cause autism", "Yes", 0.90, 0.00, 1.00, 0.00, 1.00, True),
]


def rated_fact_check(judged: str, answer: str) -> Hit:
    rating = {"No": "False", "Yes": "True", "Partly": "Half true"}[answer]
    text = f"Q: Is it true that {judged}? A: {answer}. A fact-checker rated this claim {rating}."
    return Hit(Passage("gfc:https://example.org/fc", text, "https://example.org/fc", "factcheck.afp.com"), 1.0, None, None)


def test_same_claim_rules():
    from src.verify.model import same_claim_allows

    for claim, judged, answer, cos, ef, cf, eb, cb, counts in SAME_MODE:
        assert same_claim_allows(REFUTE, [ef, cf, eb, cb], cos, answer) is counts, (claim, judged)
    mutual, contains, forward = [0.9, 0.0, 0.9, 0.0], [0.1, 0.0, 0.9, 0.0], [0.9, 0.0, 0.1, 0.0]
    assert same_claim_allows(REFUTE, mutual, 0.56, "No")  # each implies the other: wording does not matter
    assert same_claim_allows(REFUTE, contains, 0.70, "No") and not same_claim_allows(REFUTE, contains, 0.69, "No")
    assert same_claim_allows(REFUTE, forward, 0.90, "No") and not same_claim_allows(REFUTE, forward, 0.89, "No")
    # A confirmed claim that the verifier reads as speaking against ours counts only if the two contradict.
    assert not same_claim_allows(REFUTE, mutual, 0.99, "Yes") and same_claim_allows(REFUTE, [0.0, 0.9, 0.0, 0.9], 0.6, "Yes")
    assert same_claim_allows(SUPPORT, mutual, 0.99, "Yes")  # ... and supports ours when it is the same claim


def test_same_mode_through_the_verifier_marks_related_fact_checks():
    for claim, judged, answer, cos, ef, cf, eb, cb, counts in SAME_MODE:
        hit = rated_fact_check(judged, answer)
        table = {(claim, judged): (ef, cf), (judged, claim): (eb, cb)}
        v = ClaimVerifier(predict_fn=lambda c, ps: np.array([REFUTE for _ in ps]), reliability={},
                          matcher=ClaimMatcher(encoder=lambda texts, cos=cos, judged=judged: np.array(
                              [[cos, math.sqrt(1 - cos ** 2)] if t == judged else [1.0, 0.0] for t in texts]), model=None),
                          entailer=ClaimEntailment(predict_fn=lambda pairs, table=table: [table[p] for p in pairs]),
                          entail_mode="same").verify(claim, [hit])
        e = v.evidence[0]
        if counts:
            assert v.label == "Refuted" and e.counted and not e.related, claim
        else:  # a related claim: left out of the vote and marked, so it can be shown as context
            assert v.label == "Not Enough Evidence" and math.isnan(v.score) and not e.counted and e.related, claim


def test_same_mode_is_off_unless_asked_for_and_keeps_other_passages():
    hits = [rated_fact_check("Video shows the Eiffel Tower collapsing after being set ablaze", "No")]
    off = verdict_from(hits, np.array([REFUTE]), [0.6], [0.5], entail=np.array([[0.0, 0.01, 0.01, 0.0]]))
    assert off.label == "Refuted" and not off.evidence[0].related
    news = Hit(Passage("news:1", "voanews.com reported (2024-10-10): Ratan Tata, Indian industry legend, dies at 86",
                       None, "voanews.com"), 2.0, None, None)
    v = verdict_from([news], np.array([SUPPORT]), [np.nan], [0.9], entail=np.full((1, 4), np.nan), entail_mode="same")
    assert v.label == "Supported" and v.evidence[0].counted and not v.evidence[0].related
