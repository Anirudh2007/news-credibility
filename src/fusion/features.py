"""Phase 4 feature vector: one row per news item, built from the channel outputs.

The same functions run when building the training table and when the API
scores a live item, so the meta-learner never sees features computed two
different ways (training/serving skew).

Missing-value rule: a check that does not apply is
NaN, never 0. "No image" must not look like "image failed the check".
One exception by design: `n_claims` is 0 (not NaN) when there is text but no
check-worthy sentence, because "nothing checkable" is itself information.
"""

from __future__ import annotations

import math
from typing import Sequence

NAN = math.nan

CHANNEL_FEATURES: dict[str, list[str]] = {
    "claims": [  # 3A extraction + 3B retrieval + 3C verification
        "n_claims", "claim_support_ratio", "claim_refute_ratio", "claim_score_mean",
        "claim_score_min", "verifier_confidence_mean", "checkworthiness_max", "evidence_relevance",
        "claim_evidence_score",
    ],
    "text_image": ["text_image_score", "text_image_confidence", "clip_similarity"],  # 3D
    "reuse": ["reuse_score", "reuse_n_matches", "reuse_date_gap_days", "location_similarity", "ela"],  # 3E
    "source": ["source_score", "source_reliability", "clickbait_hits", "capital_ratio"],  # 3F
    "temporal": ["temporal_score", "gap_earliest_copy_days", "gap_publish_days"],  # 3G
    "modality": ["has_text", "has_image", "has_video"],
}
FEATURE_COLUMNS: list[str] = [c for cols in CHANNEL_FEATURES.values() for c in cols]
CHANNELS = [c for c in CHANNEL_FEATURES if c != "modality"]

# One headline score per channel: used by the equal-weight baseline / fallback.
CHANNEL_SCORE = {
    "claims": "claim_score_mean",
    "text_image": "text_image_score",
    "reuse": "reuse_score",
    "source": "source_score",
    "temporal": "temporal_score",
}
# Scores that carry evidence about *this* item (the claim score only after the relevance gate).
# If none is present the meta-learner can only return its prior for the input type (60 for any
# text: gibberish included), so the pipeline gives no score instead.
EVIDENCE_SCORES = ["claim_evidence_score", "text_image_score", "reuse_score", "source_score", "temporal_score"]
# The date check counts only when it compared the claimed date with a date of this item: its publish
# date or the earliest copy of its image. With nothing to compare it says "consistent" by default, and
# evidence-passage dates alone do not count (those passages may be about other claims; when they are
# about this claim, the claim check counts already). Before this rule, "On 5 March 2024 a crowd gathered
# at India Gate" + a claimed date got a score although nothing about it had been checked. Features are unchanged.
TEMPORAL_REFERENCES = ["gap_publish_days", "gap_earliest_copy_days"]


def _present(row: dict, col: str) -> bool:
    return row.get(col) is not None and not math.isnan(row[col])


def has_evidence(row: dict) -> bool:
    if any(_present(row, c) for c in EVIDENCE_SCORES if c != "temporal_score"):
        return True
    return _present(row, "temporal_score") and any(_present(row, c) for c in TEMPORAL_REFERENCES)


import os

# Minimum evidence relevance (sigmoid of the best cross-encoder score) for a verdict to count.
# Relevance is bimodal: AVeriTeC claims median 0.11 (75th pct 0.83), while for Reddit titles,
# news headlines and tweets 90% are below 0.007 (the index holds nothing about them).
RELEVANCE_MIN = float(os.environ.get("NEWSCRED_RELEVANCE_MIN", "0.01"))


def gated_claim_score(score: float, relevance: float, threshold: float | None = None) -> float:
    """Verdict credibility weighted by evidence relevance, 0.5 + (score - 0.5) * relevance,
    if relevance >= threshold; else NaN. A verdict built on evidence about some *other*
    claim is no information, so the claim check counts as not run instead of moving the
    score (without the gate, trees split at 0.4999 / 0.5001 and a 0.0001-relevance verdict
    still moved a live score by about 10 points)."""
    t = RELEVANCE_MIN if threshold is None else threshold
    if score is None or relevance is None or math.isnan(score) or math.isnan(relevance) or relevance < t:
        return NAN
    return 0.5 + (float(score) - 0.5) * float(relevance)


def add_derived(df):
    """Derived columns for a meta table built before they existed (same rule as serving)."""
    df["claim_evidence_score"] = [gated_claim_score(s, r) for s, r in zip(df["claim_score_mean"], df["evidence_relevance"])]
    return df


def _f(x) -> float:
    """float, with None -> NaN."""
    return NAN if x is None else float(x)


def empty(channel: str) -> dict[str, float]:
    return {c: NAN for c in CHANNEL_FEATURES[channel]}


def claim_features(claims: Sequence | None, verdicts: Sequence | None) -> dict[str, float]:
    """claims: `Claim`s from 3A (None = no text). verdicts: `Verdict`s from 3C, one
    per claim, or None when the verifier/index is not available."""
    out = empty("claims")
    if claims is None:
        return out
    out["n_claims"] = float(len(claims))
    if not claims:
        return out
    out["checkworthiness_max"] = max(c.checkworthiness for c in claims)
    if not verdicts:
        return out
    scored = [v for v in verdicts if not math.isnan(v.score)]  # NaN = no evidence for that claim
    if scored:
        out["claim_support_ratio"] = sum(v.label == "Supported" for v in scored) / len(scored)
        out["claim_refute_ratio"] = sum(v.label == "Refuted" for v in scored) / len(scored)
        out["claim_score_mean"] = sum(v.score for v in scored) / len(scored)
        out["claim_score_min"] = min(v.score for v in scored)
    out["verifier_confidence_mean"] = sum(v.confidence for v in verdicts) / len(verdicts)
    # How well retrieval matched each claim (best re-ranker score, 0-1), averaged. Lets the
    # meta-learner trust a verdict only when the evidence is about the claim: for most
    # Reddit titles the index holds nothing related and the verdict is noise.
    rel = [v.relevance for v in verdicts if not math.isnan(getattr(v, "relevance", NAN))]
    if rel:
        out["evidence_relevance"] = sum(rel) / len(rel)
    out["claim_evidence_score"] = gated_claim_score(out["claim_score_mean"], out["evidence_relevance"])
    return out


def text_image_features(res) -> dict[str, float]:
    if res is None or math.isnan(res.score):
        return empty("text_image")
    return {"text_image_score": res.score, "text_image_confidence": res.confidence, "clip_similarity": _f(res.clip_similarity)}


def reuse_features(res) -> dict[str, float]:
    if res is None or res.signals is None or math.isnan(res.score):
        return empty("reuse")
    s = res.signals
    return {
        "reuse_score": res.score,
        "reuse_n_matches": float(s.n_matches),
        "reuse_date_gap_days": _f(s.date_gap_days),
        "location_similarity": _f(s.location_similarity),
        "ela": _f(s.ela),
    }


def source_features(res) -> dict[str, float]:
    if res is None or math.isnan(res.score):
        return empty("source")
    f = res.features
    return {
        "source_score": res.score,
        "source_reliability": _f(f.get("reliability")),
        "clickbait_hits": _f(f.get("clickbait_hits")),
        "capital_ratio": _f(f.get("capital_ratio")),
    }


def temporal_features(res) -> dict[str, float]:
    if res is None or math.isnan(res.score):
        return empty("temporal")
    return {
        "temporal_score": res.score,
        "gap_earliest_copy_days": _f(res.gaps.get("earliest_copy")),
        "gap_publish_days": _f(res.gaps.get("publish")),
    }


def modality_features(has_text: bool, has_image: bool, has_video: bool) -> dict[str, float]:
    return {"has_text": float(has_text), "has_image": float(has_image), "has_video": float(has_video)}


def assemble(*parts: dict[str, float]) -> dict[str, float]:
    """Merge channel dicts into one full row; any column not supplied is NaN."""
    row = {c: NAN for c in FEATURE_COLUMNS}
    for p in parts:
        row.update({k: v for k, v in p.items() if k in row})
    return row


def to_vector(row: dict[str, float]) -> list[float]:
    return [_f(row.get(c)) for c in FEATURE_COLUMNS]
