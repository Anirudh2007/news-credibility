"""SQLite storage for analyses and user feedback (Phase 6).

Tables
    analyses   one row per analysed input: input hash, the submitted text/URL/claimed
               date/place, whether an image was sent (the image itself is NOT stored,
               only its hash), score, band, model version, full JSON report, time.
    feedback   user reactions to an analysis: agrees (yes/no), optional corrected
               label, optional comment. Many per analysis; deleted with it.

Caching: an identical input analysed by the same
model version returns the stored report instead of running the pipeline again.

Database URL: env NEWSCRED_DB_URL, default sqlite:///data/app.db (git-ignored,
like all of data/). Tests use "sqlite://" (in memory).
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text, create_engine, event, func, literal_column, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker
from sqlalchemy.pool import StaticPool

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_URL = f"sqlite:///{(ROOT / 'data' / 'app.db').as_posix()}"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime) -> str:
    """ISO time, always with the UTC offset. SQLite stores no timezone, so rows read back
    from disk are naive while freshly created ones are aware; both are UTC."""
    return (dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)).isoformat()


# --- Models ------------------------------------------------------------------------

class Base(DeclarativeBase):
    pass


class Analysis(Base):
    __tablename__ = "analyses"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=lambda: uuid.uuid4().hex[:12])
    input_hash: Mapped[str] = mapped_column(String(64), index=True)
    input_type: Mapped[str] = mapped_column(String(32))  # e.g. "text", "text+image", "url"
    text: Mapped[Optional[str]] = mapped_column(Text)
    url: Mapped[Optional[str]] = mapped_column(String(2048))
    claimed_date: Mapped[Optional[str]] = mapped_column(String(10))
    claimed_location: Mapped[Optional[str]] = mapped_column(String(200))
    has_image: Mapped[bool] = mapped_column(Boolean, default=False)
    score: Mapped[Optional[int]] = mapped_column(Integer)
    band: Mapped[str] = mapped_column(String(32))
    meta_model: Mapped[str] = mapped_column(String(100))
    model_version: Mapped[str] = mapped_column(String(20), index=True)
    report_json: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)

    feedback: Mapped[list["Feedback"]] = relationship(back_populates="analysis", cascade="all, delete-orphan")

    def report(self) -> dict:
        rep = json.loads(self.report_json)
        rep["analysis_id"] = self.id
        rep["created_at"] = iso(self.created_at)
        return rep


class Feedback(Base):
    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    analysis_id: Mapped[str] = mapped_column(ForeignKey("analyses.id", ondelete="CASCADE"), index=True)
    agrees: Mapped[bool] = mapped_column(Boolean)
    correct_label: Mapped[Optional[str]] = mapped_column(String(20))  # credible / not_credible / mixed / unsure
    comment: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    analysis: Mapped[Analysis] = relationship(back_populates="feedback")


# --- Session --------------------------------------------------------------------------

_engine = None
_Session: sessionmaker | None = None


def configure(url: str | None = None):
    """Create the engine and tables. Called once at API startup; tests pass "sqlite://"."""
    global _engine, _Session
    url = url or DEFAULT_DB_URL
    kwargs: dict = {}
    if url.startswith("sqlite"):
        # FastAPI runs sync endpoints in worker threads; one connection per thread is fine.
        kwargs["connect_args"] = {"check_same_thread": False}
        if url in ("sqlite://", "sqlite:///:memory:"):
            kwargs["poolclass"] = StaticPool  # one shared in-memory database
        else:
            Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    _engine = create_engine(url, **kwargs)
    if url.startswith("sqlite"):
        @event.listens_for(_engine, "connect")
        def _pragmas(conn, _record):
            cur = conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")  # needed for ON DELETE CASCADE in SQLite
            cur.execute("PRAGMA journal_mode=WAL")  # readers do not block the writer
            cur.close()
    Base.metadata.create_all(_engine)
    _Session = sessionmaker(_engine, expire_on_commit=False)
    return _engine


def get_session() -> Iterator[Session]:
    """FastAPI dependency: one session per request, always closed."""
    if _Session is None:
        configure()
    with _Session() as session:
        yield session


# --- Queries ----------------------------------------------------------------------------

def input_type(text: Optional[str], url: Optional[str], has_image: bool) -> str:
    parts = [p for p, present in (("text", bool(text)), ("url", bool(url)), ("image", has_image)) if present]
    return "+".join(parts) or "empty"


def find_cached(session: Session, input_hash: str, model_version: str) -> Optional[Analysis]:
    stmt = (select(Analysis).where(Analysis.input_hash == input_hash, Analysis.model_version == model_version)
            .order_by(Analysis.created_at.desc()).limit(1))
    return session.scalars(stmt).first()


def save_analysis(session: Session, report: dict, *, input_hash: str, text, url, claimed_date, claimed_location,
                  has_image: bool) -> Analysis:
    row = Analysis(
        input_hash=input_hash, input_type=input_type(text, url, has_image), text=text, url=url,
        claimed_date=claimed_date, claimed_location=claimed_location, has_image=has_image,
        score=report.get("score"), band=report.get("band", "Unknown"), meta_model=report.get("meta_model", ""),
        model_version=report.get("model_version", ""), report_json=json.dumps(report, allow_nan=False),
    )
    session.add(row)
    session.commit()
    return row


def update_report(session: Session, row: Analysis, report: dict) -> None:
    row.report_json = json.dumps({k: v for k, v in report.items() if k not in ("analysis_id", "created_at")},
                                 allow_nan=False)
    session.commit()


def history(session: Session, limit: int = 20, offset: int = 0) -> list[dict]:
    n_fb = (select(Feedback.analysis_id, func.count(Feedback.id).label("n")).group_by(Feedback.analysis_id).subquery())
    stmt = (select(Analysis, func.coalesce(n_fb.c.n, 0)).outerjoin(n_fb, n_fb.c.analysis_id == Analysis.id)
            # ties (same Windows clock tick, ~15 ms): newest insert first via SQLite's rowid, not the hash id
            .order_by(Analysis.created_at.desc(), literal_column("analyses.rowid").desc()).limit(limit).offset(offset))
    out = []
    for a, n in session.execute(stmt):
        preview = (a.text or a.url or ("[image]" if a.has_image else ""))[:140]
        out.append({"analysis_id": a.id, "created_at": iso(a.created_at), "input_type": a.input_type,
                    "preview": preview, "score": a.score, "band": a.band, "feedback_count": int(n)})
    return out


def count(session: Session) -> int:
    return session.scalar(select(func.count(Analysis.id))) or 0


def add_feedback(session: Session, analysis_id: str, agrees: bool, correct_label: Optional[str],
                 comment: Optional[str]) -> Optional[Feedback]:
    if session.get(Analysis, analysis_id) is None:
        return None
    fb = Feedback(analysis_id=analysis_id, agrees=agrees, correct_label=correct_label, comment=comment)
    session.add(fb)
    session.commit()
    return fb


def delete_analysis(session: Session, analysis_id: str) -> bool:
    row = session.get(Analysis, analysis_id)
    if row is None:
        return False
    session.delete(row)
    session.commit()
    return True


def feedback_summary(session: Session) -> dict:
    """Agreement rate so far: a cheap real-world quality signal for the report."""
    total = session.scalar(select(func.count(Feedback.id))) or 0
    agree = session.scalar(select(func.count(Feedback.id)).where(Feedback.agrees.is_(True))) or 0
    return {"feedback": total, "agree": agree, "agree_rate": (agree / total) if total else None}
