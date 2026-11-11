"""FastAPI app (Phase 6). Wraps `src/pipeline.py`.

Run from `news-credibility/`:
    uvicorn api.main:app --reload            # http://localhost:8000/docs
    NEWSCRED_WARMUP=1 uvicorn api.main:app   # load all models at startup instead of on first request

Endpoints
    POST   /api/v1/analyze            multipart: text, url, image, claimed_date, claimed_location,
                                      publish_date, explain, refresh
    POST   /verify                    same as /api/v1/analyze (short alias)
    GET    /api/v1/history            past analyses (limit, offset), newest first
    GET    /api/v1/analyses/{id}      one stored report, with its feedback
    DELETE /api/v1/analyses/{id}      remove an analysis and its feedback
    POST   /api/v1/feedback           JSON: analysis_id, agrees, correct_label, comment
    GET    /api/v1/feedback/summary   agreement rate so far
    GET    /api/v1/sources/{domain}   stored reliability rating
    GET    /health                    service, components, explainer, database

Storage (Phase 6): every analysis is saved in SQLite (`api/db.py`; NEWSCRED_DB_URL,
default data/app.db). Identical input + same model version returns the stored
report (`cached: true`) unless `refresh=true`. "Model version" includes a fingerprint of
the trained files, so retraining any model invalidates the cache. Images are never
stored, only a hash.

Robustness (Phase 8): text is checked first (`src/ingest/guard.py`); gibberish, symbol strings,
repetition and non-English text get a 422 with the reason, never a score. A report whose checks
found no evidence about the item has `score: null`, band "Unverified" and `abstained: true`.

Explanations (Phase 5): every report carries points per check (`contributions`) and,
unless `explain=false`, an `explanation` written by the local LLM router
(EXPLAIN_BASE_URL, default http://localhost:3001/v1), validated against the report's
facts, with a templated fallback. The router sends prompts to outside providers.

URLs: a bare domain is used only for the source-credibility check. An article URL sent
without text is fetched (`src/ingest/fetch_article.py`) and its headline, summary and
publish date are checked like pasted text (off with NEWSCRED_FETCH_URL=0).

Not yet: video analysis (future work).
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Response, UploadFile
from sqlalchemy.orm import Session

from api import db
from api.schemas import (
    AnalysisReport,
    FeedbackIn,
    FeedbackItem,
    FeedbackSummary,
    Health,
    HistoryPage,
    SourceInfo,
)
from src.explain.generator import Explainer
from src.ingest.guard import InputGuard, default_guard
from src.pipeline import MODEL_VERSION, NewsItem, Pipeline

MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_TEXT_CHARS = 20_000
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif", "image/bmp"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.configure(os.environ.get("NEWSCRED_DB_URL"))
    app.state.pipeline = Pipeline()  # components load lazily, once, then stay in memory
    app.state.explainer = Explainer()  # no network call until the first explanation
    if os.environ.get("NEWSCRED_WARMUP") == "1":
        app.state.pipeline.load_all()
        default_guard()
    yield


app = FastAPI(
    title="News Credibility and Claim Verification API",
    version=MODEL_VERSION,
    description="Claim-level, multimodal credibility scoring with reasons. Scores are an aid, not a verdict.",
    lifespan=lifespan,
)


def get_pipeline() -> Pipeline:
    """Dependency: the app-wide pipeline (tests override this with a fake)."""
    return app.state.pipeline


def get_explainer() -> Explainer:
    """Dependency: the app-wide explainer (tests override this with a fake LLM)."""
    return app.state.explainer


def get_guard() -> InputGuard:
    """Dependency: the input guard (built on first use from the evidence passages)."""
    return default_guard()


def model_version(pipeline) -> str:
    """Version of the loaded models (code version + fingerprint of the trained files)."""
    return getattr(pipeline, "model_version", MODEL_VERSION)


def _iso_or_422(value: Optional[str], field: str) -> Optional[str]:
    if not value:
        return None
    from src.preprocess.clean_text import parse_date

    iso = parse_date(value)
    if iso is None:
        raise HTTPException(422, f"{field}: cannot parse date {value!r} (use YYYY-MM-DD)")
    return iso


def _read_image(image: Optional[UploadFile]):
    if image is None or not image.filename:
        return None
    if image.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(415, f"image type {image.content_type!r} not supported; use JPEG, PNG, WebP, GIF or BMP")
    data = image.file.read(MAX_IMAGE_BYTES + 1)
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(413, f"image larger than {MAX_IMAGE_BYTES // (1024 * 1024)} MB")
    from src.media.text_image import load_image

    img = load_image(data)
    if img is None:
        raise HTTPException(422, "image is corrupt or smaller than 32 px")
    return img


@lru_cache(maxsize=1)
def _ratings() -> dict:
    from src.source.credibility import load_ratings

    return load_ratings()


def _is_fact_checker(url: Optional[str]) -> bool:
    from src.source.credibility import parse_domain

    rating = _ratings().get(parse_domain(url)[0] or "")
    return bool(rating and rating.bias == "fact-check")


def _fetch_article_text(item: NewsItem, guard: InputGuard) -> Optional[dict]:
    """URL and no text: read the article's headline + summary (+ publish date) into the item, so the
    claims are checked instead of only the outlet. Returns what was read (shown to the user), or None
    when there was nothing to fetch (bare domain, fetching off)."""
    from src.ingest import fetch_article as FA

    if not (FA.enabled() and FA.is_article_url(item.source_url)):
        return None
    art = FA.fetch_article(item.source_url)
    out = {"url": item.source_url, "title": art.title, "description": art.description,
           "published": art.published, "error": art.error}
    if not art.error and _is_fact_checker(item.source_url):
        # A fact-check's headline names the rumour it debunks ("Deepfakes ... fuel false claims ..."):
        # checked as a claim it reads as the rumour (BOOM's article scored 35 as if it were the rumour).
        out["error"] = ("this is a fact-check article: its headline describes the rumour it checks. Paste the "
                        "rumour's text instead; published fact-checks like this one are found as evidence")
        return out
    text = art.text() if not art.error else None
    if text:
        checked = guard.check(text)
        if not checked.ok:  # e.g. a Hindi page: same rule as typed text, never scored
            out["error"] = f"the page's headline was not usable: {checked.message}"
            return out
        item.text = text[:MAX_TEXT_CHARS]
        if not item.publish_date and art.published:
            item.publish_date = art.published
    return out


def _analyze(
    pipeline: Pipeline,
    explainer: Explainer,
    session: Session,
    guard: InputGuard,
    explain: bool,
    refresh: bool,
    text: Optional[str],
    url: Optional[str],
    image: Optional[UploadFile],
    claimed_date: Optional[str],
    claimed_location: Optional[str],
    publish_date: Optional[str],
) -> dict:
    text = (text or "").strip() or None
    url = (url or "").strip() or None
    if text and len(text) > MAX_TEXT_CHARS:
        raise HTTPException(413, f"text longer than {MAX_TEXT_CHARS} characters")
    img = _read_image(image)
    if text:  # gibberish, symbols, repetition, other languages: refuse, never score
        checked = guard.check(text, caption=img is not None)
        if not checked.ok:
            raise HTTPException(422, f"{checked.message} Nothing was analysed.")
    if not (text or url or img is not None):
        raise HTTPException(422, "provide at least one of: text, url, image")
    item = NewsItem(
        text=text,
        image=img,
        source_url=url,
        claimed_date=_iso_or_422(claimed_date, "claimed_date"),
        claimed_place=(claimed_location or "").strip() or None,
        publish_date=_iso_or_422(publish_date, "publish_date"),
    )
    input_hash = item.input_hash()  # of what the user sent: a cached result skips the page fetch
    version = model_version(pipeline)

    if not refresh:
        row = db.find_cached(session, input_hash, version)
        if row is not None:
            report = row.report()
            if explain and not report.get("explanation"):
                report["explanation"] = explainer.explain(report, text)
                db.update_report(session, row, report)
            if not explain:
                report["explanation"] = None
            return {**report, "cached": True}

    fetched = _fetch_article_text(item, guard) if not text else None
    report = pipeline.analyze(item)
    if fetched is not None:
        report["fetched_article"] = fetched
        if fetched.get("error"):
            report.setdefault("reasons", []).append(
                f"The article itself was not checked: {fetched['error']}. Only the source was used.")
    text = item.text
    if explain:
        # Never fails: router errors and rejected LLM text fall back to the template.
        report["explanation"] = explainer.explain(report, text)
    report["input_hash"] = input_hash
    report["model_version"] = version  # the cache key: same input + same trained models
    row = db.save_analysis(session, report, input_hash=input_hash, text=text, url=url,
                           claimed_date=item.claimed_date, claimed_location=item.claimed_place,
                           has_image=img is not None)
    return {**row.report(), "cached": False}


# Sync `def` endpoints: FastAPI runs them in a worker thread, so model inference
# and the LLM call do not block the event loop.
@app.post("/api/v1/analyze", response_model=AnalysisReport, tags=["analysis"])
def analyze(
    text: Optional[str] = Form(None, description="Claim, post or article text"),
    url: Optional[str] = Form(None, description="Source URL or domain (used for source credibility)"),
    image: Optional[UploadFile] = File(None, description="Optional image (JPEG/PNG/WebP/GIF/BMP, max 10 MB)"),
    claimed_date: Optional[str] = Form(None, description="Date the item claims, e.g. 2024-07-03"),
    claimed_location: Optional[str] = Form(None, description="Place the item claims, e.g. India Gate"),
    publish_date: Optional[str] = Form(None, description="When the item was published"),
    explain: bool = Form(True, description="Add the readable explanation (LLM with template fallback)"),
    refresh: bool = Form(False, description="Ignore a stored result for identical input and analyse again"),
    pipeline: Pipeline = Depends(get_pipeline),
    explainer: Explainer = Depends(get_explainer),
    session: Session = Depends(db.get_session),
    guard: InputGuard = Depends(get_guard),
):
    return _analyze(pipeline, explainer, session, guard, explain, refresh,
                    text, url, image, claimed_date, claimed_location, publish_date)


@app.post("/verify", response_model=AnalysisReport, tags=["analysis"])
def verify(
    text: Optional[str] = Form(None),
    url: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    claimed_date: Optional[str] = Form(None),
    claimed_location: Optional[str] = Form(None),
    publish_date: Optional[str] = Form(None),
    explain: bool = Form(True),
    refresh: bool = Form(False),
    pipeline: Pipeline = Depends(get_pipeline),
    explainer: Explainer = Depends(get_explainer),
    session: Session = Depends(db.get_session),
    guard: InputGuard = Depends(get_guard),
):
    """Alias of POST /api/v1/analyze."""
    return _analyze(pipeline, explainer, session, guard, explain, refresh,
                    text, url, image, claimed_date, claimed_location, publish_date)


# --- History and feedback (Phase 6) ----------------------------------------------------

def _feedback_item(fb: db.Feedback) -> dict:
    return {"id": fb.id, "analysis_id": fb.analysis_id, "agrees": fb.agrees, "correct_label": fb.correct_label,
            "comment": fb.comment, "created_at": db.iso(fb.created_at)}


@app.get("/api/v1/history", response_model=HistoryPage, tags=["history"])
def history(
    limit: int = Query(20, ge=1, le=200),
    offset: int = Query(0, ge=0),
    session: Session = Depends(db.get_session),
):
    return {"total": db.count(session), "limit": limit, "offset": offset,
            "items": db.history(session, limit, offset)}


@app.get("/api/v1/analyses/{analysis_id}", response_model=AnalysisReport, tags=["history"])
def get_analysis(analysis_id: str, session: Session = Depends(db.get_session)):
    row = session.get(db.Analysis, analysis_id)
    if row is None:
        raise HTTPException(404, f"no analysis {analysis_id}")
    return {**row.report(), "cached": True, "feedback": [_feedback_item(f) for f in row.feedback]}


@app.delete("/api/v1/analyses/{analysis_id}", status_code=204, tags=["history"])
def delete_analysis(analysis_id: str, session: Session = Depends(db.get_session)):
    if not db.delete_analysis(session, analysis_id):
        raise HTTPException(404, f"no analysis {analysis_id}")
    return Response(status_code=204)


@app.post("/api/v1/feedback", response_model=FeedbackItem, status_code=201, tags=["feedback"])
def submit_feedback(body: FeedbackIn, session: Session = Depends(db.get_session)):
    comment = (body.comment or "").strip() or None
    fb = db.add_feedback(session, body.analysis_id, body.agrees, body.correct_label, comment)
    if fb is None:
        raise HTTPException(404, f"no analysis {body.analysis_id}")
    return _feedback_item(fb)


@app.get("/api/v1/feedback/summary", response_model=FeedbackSummary, tags=["feedback"])
def feedback_summary(session: Session = Depends(db.get_session)):
    return db.feedback_summary(session)


# --- Service --------------------------------------------------------------------------------

@app.get("/health", response_model=Health, tags=["service"])
def health(
    pipeline: Pipeline = Depends(get_pipeline),
    explainer: Explainer = Depends(get_explainer),
    session: Session = Depends(db.get_session),
    guard: InputGuard = Depends(get_guard),
):
    """Component status as loaded so far (lazy loading: untouched components are absent)."""
    return {"status": "ok", "model_version": model_version(pipeline), "components": dict(pipeline.status),
            "explainer": explainer.status, "database": f"ok ({db.count(session)} analyses stored)",
            "input_guard": guard.status}


@app.get("/api/v1/sources/{domain}", response_model=SourceInfo, tags=["sources"])
def source_rating(domain: str):
    from src.source.credibility import load_ratings, parse_domain

    reg = parse_domain(domain)[0] or domain.lower()
    rating = load_ratings().get(reg)
    if rating is None:
        raise HTTPException(404, f"no rating stored for {reg}")
    return {"domain": reg, "reliability": rating.reliability, "bias": rating.bias}
