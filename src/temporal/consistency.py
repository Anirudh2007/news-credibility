"""Channel 3G: temporal consistency, rule-based (Phase 3G).

Compares the date a news item claims ("protest today", "flood on 3 July 2024")
with independent dates:

    metadata     article publish date, image EXIF capture date
    earliest     earliest known copy of the media (Channel 3E)
    evidence     dates of retrieved evidence passages (Channel 3B)

gap_days = claimed - reference. Positive = the reference is older than the claim.

Rules (each failure adds a flag and a plain-language reason):
- earliest copy or EXIF capture > 2 days before the claimed date
      -> media existed before the event it is said to show (strong)
- publish date > 2 days before the claimed date
      -> the article predates what it reports (medium: scheduled events such
         as elections are legitimately announced in advance)
- every dated evidence passage is > 365 days older than the claim
      -> the story may be old news resurfacing (weak)

score = 1 - sum of failed-rule weights (min 0); confidence grows with the number
of reference dates that could be checked. No claimed date -> score NaN: the
check does not apply, which is different from "inconsistent".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Sequence

TOLERANCE_DAYS = 2  # time zones and next-day reporting
OLD_NEWS_DAYS = 365
WEIGHTS = {"media_predates_claim": 0.6, "exif_predates_claim": 0.4, "published_before_event": 0.3, "evidence_much_older": 0.2}


def to_date(value) -> date | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, date):
        return value
    from src.preprocess.clean_text import parse_date

    iso = parse_date(value)
    return date.fromisoformat(iso) if iso else None


def gap_days(claimed: date | None, reference) -> float:
    ref = to_date(reference)
    return float((claimed - ref).days) if claimed and ref else math.nan


def claimed_date_from_text(text: str, publish_date=None) -> date | None:
    """First explicit date in the claim, else a relative one ('yesterday')
    resolved against the publish date (reuses 3A rules), else None."""
    from src.claims.extract import resolve_relative_dates
    from src.preprocess.clean_text import extract_dates

    found = extract_dates(text or "")
    if found:
        return date.fromisoformat(found[0])
    pub = to_date(publish_date)
    _, resolved = resolve_relative_dates(text or "", pub)
    return date.fromisoformat(resolved[0]) if resolved else None


@dataclass
class TemporalResult:
    score: float  # 1 consistent ... 0 inconsistent; NaN if no claimed date
    confidence: float
    claimed_date: str | None
    gaps: dict[str, float] = field(default_factory=dict)  # days, NaN when that reference is missing
    flags: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    def features(self) -> dict[str, float]:
        """For the meta-learner: gaps in days + consistency flag (NaN stays NaN)."""
        consistent = math.nan if math.isnan(self.score) else float(not self.flags)
        return {**{f"gap_{k}_days": v for k, v in self.gaps.items()}, "temporal_consistent": consistent}


def check(
    claimed_date=None,
    publish_date=None,
    exif_date=None,
    earliest_copy_date=None,
    evidence_dates: Sequence = (),
    claim_text: str | None = None,
) -> TemporalResult:
    claimed = to_date(claimed_date) or (claimed_date_from_text(claim_text, publish_date) if claim_text else None)
    ev = [d for d in (to_date(x) for x in evidence_dates) if d]
    gaps = {
        "publish": gap_days(claimed, publish_date),
        "exif": gap_days(claimed, exif_date),
        "earliest_copy": gap_days(claimed, earliest_copy_date),
        "newest_evidence": gap_days(claimed, max(ev)) if ev else math.nan,
    }
    if claimed is None:
        return TemporalResult(math.nan, 0.0, None, gaps)

    flags, reasons = [], []
    if gaps["earliest_copy"] > TOLERANCE_DAYS:
        flags.append("media_predates_claim")
        reasons.append(f"The image/video appeared online {int(gaps['earliest_copy'])} days before the claimed date.")
    if gaps["exif"] > TOLERANCE_DAYS:
        flags.append("exif_predates_claim")
        reasons.append(f"Camera metadata says it was taken {int(gaps['exif'])} days before the claimed date.")
    if gaps["publish"] > TOLERANCE_DAYS:
        flags.append("published_before_event")
        reasons.append(
            f"The article was published {int(gaps['publish'])} days before the date it reports "
            "(fine only for scheduled events)."
        )
    if gaps["newest_evidence"] > OLD_NEWS_DAYS:
        flags.append("evidence_much_older")
        reasons.append(f"All dated evidence is over {OLD_NEWS_DAYS} days older than the claim; possibly old news resurfacing.")

    score = max(0.0, 1.0 - sum(WEIGHTS[f] for f in flags))
    checked = sum(not math.isnan(v) for v in gaps.values())
    confidence = min(1.0, 0.25 * checked + (0.25 if flags else 0.0))
    return TemporalResult(score, confidence, claimed.isoformat(), gaps, flags, reasons)
