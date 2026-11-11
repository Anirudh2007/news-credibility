"""Pydantic response models (Phase 6). Requests are multipart forms,
validated in `api/main.py`, because an image upload cannot travel inside JSON."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class Evidence(BaseModel):
    source: Optional[str] = Field(None, description="Domain of the evidence passage")
    url: Optional[str] = None
    snippet: str
    weight: float = Field(..., description="Share of the claim verdict this passage carried (re-ranker x reliability)")
    date: Optional[str] = Field(None, description="Date of the passage (ISO), if known")


class FactCheckBadge(BaseModel):
    publisher: str = Field(..., description="Who published the fact-check, e.g. BOOM, Alt News, PolitiFact")
    rating: str = Field(..., description="The fact-checker's own verdict wording, e.g. False, Misleading")
    claim_reviewed: str
    url: Optional[str] = None
    date: Optional[str] = None
    relevance: float = Field(..., description="Re-ranker relevance to this claim (0-1); badges need >= 0.5")


class EvidenceBoxItem(BaseModel):
    claim: str
    verdict: str = Field(..., description="Supported or Refuted (only decisive, confident, on-topic verdicts)")
    quote: str = Field(..., description="The strongest evidence passage, quoted, never rewritten")
    source: Optional[str] = None
    url: Optional[str] = None
    date: Optional[str] = None
    fact_checked_by: Optional[str] = None


class ClaimReport(BaseModel):
    text: str
    checkworthiness: float
    verdict: Optional[str] = Field(None, description="Supported / Refuted / Not Enough Evidence / Conflicting...; null if verifier unavailable")
    credibility: Optional[float] = None
    confidence: Optional[float] = None
    evidence_relevance: Optional[float] = Field(None, description="How closely the best evidence matches this claim (0-1, re-ranker); below 0.5 it is about a different claim")
    evidence_off_topic: bool = Field(False, description="Evidence found is about other claims (relevance below the threshold); verdict not counted")
    evidence: list[Evidence] = []
    fact_checks: list[FactCheckBadge] = Field([], description="Published fact-checks of this claim found in the evidence")
    entities: list[list[str]] = []
    dates: list[str] = []
    justification: Optional[str] = Field(None, description="Templated one-sentence justification (Phase 5)")
    context: Optional[str] = Field(None, description="'missing' if the sentence does not say who or what it is about; such claims are not verified")
    coverage: Optional[dict[str, Any]] = Field(None, description="Would trusted outlets report this claim if true (expected, why), and what the news search found (searched, search, confirmed)")


class Contribution(BaseModel):
    check: str = Field(..., description="claim_verification / text_image_match / media_reuse / source / temporal; plus calibration (isotonic step) and missing_coverage (no trusted report of a claim that would be reported). Checks that did not run have no line")
    points: float = Field(..., description="Points added (+) or removed (-) relative to baseline_score")
    applicable: bool = Field(..., description="False only for the bucket of checks that did not run")


class CitedEvidence(BaseModel):
    id: str
    source: Optional[str] = None
    snippet: str


class Explanation(BaseModel):
    summary: str
    explanation: str
    cited_evidence: list[str] = []
    evidence: list[CitedEvidence] = []
    source: str = Field(..., description="'llm' if the LLM text passed validation, else 'template'")
    model: Optional[str] = None
    prompt_version: Optional[str] = None
    validation_errors: list[str] = Field([], description="Why LLM output was rejected (template used instead)")
    llm_error: Optional[str] = Field(None, description="Why the LLM could not be called")


class AnalysisReport(BaseModel):
    analysis_id: str
    input_hash: Optional[str] = None
    score: Optional[int] = Field(None, ge=0, le=100, description="100 x calibrated probability credible; null if no check found evidence about the item")
    probability: Optional[float] = None
    band: str = Field(..., description="Low (0-30) / Questionable (31-60) / Mostly credible (61-80) / High (81-100) / Unverified (no score: no evidence) / Unknown")
    abstained: bool = Field(False, description="True if no check found evidence about this item, so no score is given (reason in `reasons`)")
    claims: list[ClaimReport] = []
    evidence_box: list[EvidenceBoxItem] = Field([], description="'What the evidence says': quoted passages for confident, on-topic verdicts")
    media: dict[str, Any] = {}
    channels: dict[str, dict[str, Any]] = {}
    features: dict[str, Optional[float]] = {}
    baseline_score: Optional[float] = Field(None, description="Score of an average item with the same inputs (checks that did not run are folded in); baseline_score + sum(points) = score")
    contributions: list[Contribution] = Field([], description="Points per check (SHAP for LightGBM)")
    explanation: Optional[Explanation] = Field(None, description="Readable summary (LLM, validated; template fallback)")
    skipped_checks: list[str] = []
    reasons: list[str] = []
    coverage: Optional[dict[str, Any]] = Field(None, description="Set when silence was used as evidence: a claim trusted outlets would report if true, a news search that ran, no report confirming it (likelihood ratio, search window, reason)")
    meta_model: str
    model_version: str
    disclaimer: str = "This score is an aid, not a final judgement. Read the evidence."
    created_at: Optional[str] = None
    cached: bool = Field(False, description="True if this identical input was already analysed by this model version")
    fetched_article: Optional[dict[str, Optional[str]]] = Field(
        None, description="URL given without text: the headline, summary and publish date read from the page "
                          "(url, title, description, published, error); the headline + summary were checked")
    feedback: list["FeedbackItem"] = Field([], description="User feedback on this analysis (GET /api/v1/analyses/{id})")


CorrectLabel = Literal["credible", "not_credible", "mixed", "unsure"]


class FeedbackIn(BaseModel):
    analysis_id: str = Field(..., max_length=32)
    agrees: bool = Field(..., description="Does the user agree with the score/band?")
    correct_label: Optional[CorrectLabel] = Field(None, description="What the user thinks is right, if they disagree")
    comment: Optional[str] = Field(None, max_length=1000)


class FeedbackItem(BaseModel):
    id: int
    analysis_id: str
    agrees: bool
    correct_label: Optional[str] = None
    comment: Optional[str] = None
    created_at: str


class HistoryItem(BaseModel):
    analysis_id: str
    created_at: str
    input_type: str
    preview: str
    score: Optional[int] = None
    band: str
    feedback_count: int


class HistoryPage(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[HistoryItem]


class FeedbackSummary(BaseModel):
    feedback: int
    agree: int
    agree_rate: Optional[float] = None


class Health(BaseModel):
    status: str
    model_version: str
    components: dict[str, str]
    explainer: Optional[str] = None
    database: Optional[str] = None
    input_guard: Optional[str] = None


class SourceInfo(BaseModel):
    domain: str
    reliability: float
    bias: Optional[str] = None


AnalysisReport.model_rebuild()  # resolve the forward reference to FeedbackItem
