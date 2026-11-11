"""Headless Streamlit tests (streamlit.testing.AppTest) with the API client mocked."""

import sys
from pathlib import Path

import pytest

FRONTEND = Path(__file__).resolve().parents[1] / "frontend"
sys.path.insert(0, str(FRONTEND))

from streamlit.testing.v1 import AppTest  # noqa: E402

from components import api_client as api  # noqa: E402  (same module object the app imports)
from components.charts import band_style, contributions_bar  # noqa: E402
from components.theme import meter_html, score_html  # noqa: E402

REPORT = {
    "analysis_id": "abc123def456", "created_at": "2026-10-27T10:00:00+00:00", "cached": False,
    "score": 28, "probability": 0.28, "band": "Low", "baseline_score": 52.0, "meta_model": "lightgbm+isotonic",
    "model_version": "0.4.0", "features": {"has_image": 0.0},
    "contributions": [{"check": "claim_verification", "points": -20.0, "applicable": True},
                      {"check": "source", "points": -4.0, "applicable": True},
                      {"check": "not_applicable", "points": 0.3, "applicable": False}],
    "explanation": {"summary": "The claim is contradicted by the evidence.", "explanation": "Refuted [E1].",
                    "evidence": [{"id": "E1", "source": "pib.gov.in", "snippet": "No new Rs 1000 notes."}],
                    "cited_evidence": ["E1"], "source": "llm"},
    "claims": [{"text": "RBI announces new Rs 1000 notes.", "checkworthiness": 0.9, "verdict": "Refuted",
                "confidence": 0.91, "justification": "The evidence found contradicts this claim.",
                "evidence": [{"source": "pib.gov.in", "url": "https://pib.gov.in/x", "snippet": "No new notes.",
                              "weight": 1.0}]}],
    "media": {}, "reasons": ["Source domain is not in the ratings list."],
    "skipped_checks": ["text_image_match: no image"], "disclaimer": "Aid, not a verdict.",
}


@pytest.fixture()
def fake_api(monkeypatch):
    calls = {"analyze": [], "feedback": []}
    monkeypatch.setattr(api, "health", lambda: {"model_version": "0.4.0", "components": {"meta": "ready"},
                                                "database": "ok (1 analyses stored)"})
    monkeypatch.setattr(api, "analyze", lambda **kw: calls["analyze"].append(kw) or REPORT)
    monkeypatch.setattr(api, "send_feedback", lambda *a: calls["feedback"].append(a) or {"id": 1})
    monkeypatch.setattr(api, "history", lambda limit=20, offset=0: {"total": 1, "limit": limit, "offset": offset, "items": [
        {"analysis_id": "abc123def456", "created_at": "2026-10-27T10:00:00+00:00", "input_type": "text",
         "preview": "RBI announces new Rs 1000 notes.", "score": 28, "band": "Low", "feedback_count": 0}]})
    monkeypatch.setattr(api, "feedback_summary", lambda: {"feedback": 0, "agree": 0, "agree_rate": None})
    monkeypatch.setattr(api, "get_analysis", lambda aid: {**REPORT, "cached": True, "feedback": []})
    return calls


def app() -> AppTest:
    at = AppTest.from_file(str(FRONTEND / "app.py"), default_timeout=30)
    return at


def test_analyze_page_submits_and_renders_report(fake_api):
    at = app().run()
    assert not at.exception
    at.text_area(key="text_input").set_value("RBI announces new Rs 1000 notes.")
    at.button[0].click().run()  # the form's Analyze button
    assert not at.exception
    assert fake_api["analyze"][0]["text"] == "RBI announces new Rs 1000 notes."
    assert fake_api["analyze"][0]["explain"] is True and fake_api["analyze"][0]["image"] is None
    md = " ".join(m.value for m in at.markdown)
    assert "Why this score" in md and "What affected this score" in md and "Claims found (1)" in md
    assert any("contradicted by the evidence" in s.value for s in at.subheader)


def test_no_llm_switch_on_a_server_without_llm(fake_api, monkeypatch):
    """Azure VM: EXPLAIN_ENABLED=0, so no LLM switch; the explanation is still requested (templates)."""
    import streamlit as st

    monkeypatch.setattr(api, "health", lambda: {"model_version": "0.9.1", "components": {},
                                                "explainer": "templates only (EXPLAIN_ENABLED=0)",
                                                "database": "ok (1 analyses stored)"})
    st.cache_data.clear()  # the app caches /health for 15 s
    at = app().run()
    assert not any(t.label.startswith("Write an explanation") for t in at.toggle)
    at.text_area(key="text_input").set_value("Claim")
    at.button[0].click().run()
    assert not at.exception and fake_api["analyze"][0]["explain"] is True
    assert any("Why this score" in m.value for m in at.markdown)
    st.cache_data.clear()


def test_example_fills_text_box_once(fake_api):
    at = app().run()
    at.selectbox(key="example").set_value("Chandrayaan-3 launch").run()
    assert "Chandrayaan-3" in at.text_area(key="text_input").value


def test_feedback_submission(fake_api):
    at = app().run()
    at.text_area(key="text_input").set_value("Claim")
    at.button[0].click().run()
    next(r for r in at.radio if r.label == "Do you agree with this result?").set_value("No")
    next(s for s in at.selectbox if s.label.startswith("If not")).set_value("not_credible")
    at.text_area[1].set_value("Old notes only")
    feedback_button = next(b for b in at.button if b.label == "Send feedback")
    feedback_button.click().run()
    assert fake_api["feedback"] == [("abc123def456", False, "not_credible", "Old notes only")]
    assert any("feedback was saved" in s.value for s in at.success)


def test_history_page_opens_analysis(fake_api):
    at = app().run()
    at.sidebar.radio[0].set_value("History").run()
    assert not at.exception and len(at.dataframe) == 1
    at.selectbox[0].set_value("abc123def456").run()
    assert any("Why this score" in m.value for m in at.markdown)


def test_history_page_that_emptied_jumps_to_last_page(fake_api, monkeypatch):
    """Deleting the only item on the last page must not show 'No analyses yet'."""
    item = {"analysis_id": "abc123def456", "created_at": "2026-10-27T10:00:00+00:00", "input_type": "text",
            "preview": "p", "score": 28, "band": "Low", "feedback_count": 0}
    monkeypatch.setattr(api, "history", lambda limit=20, offset=0: {
        "total": 20, "limit": limit, "offset": offset, "items": [item] * 20 if offset == 0 else []})
    at = app()
    at.session_state["history_page"] = 1
    at.run()
    at.sidebar.radio[0].set_value("History").run()
    assert not at.exception and at.session_state["history_page"] == 0 and len(at.dataframe) == 1
    assert not any("No analyses yet" in i.value for i in at.info)


def test_weak_evidence_caption_on_claim_card(fake_api, monkeypatch):
    weak = {**REPORT, "claims": [{**REPORT["claims"][0], "evidence_relevance": 0.2}]}
    monkeypatch.setattr(api, "analyze", lambda **kw: weak)
    at = app().run()
    at.text_area[0].set_value("RBI announces new Rs 1000 notes.")
    at.button[0].click().run()
    assert any("loosely related" in c.value for c in at.caption)


def test_silence_box_when_no_trusted_outlet_reported_it(fake_api, monkeypatch):
    cov = {"expected": True, "event": "a death", "searched": True, "confirmed": False,
           "search": {"trusted_reports": 0, "outlets": 300, "from": None, "to": "2026-11-06"}}
    silent = {**REPORT, "score": 20, "band": "Low", "baseline_score": 50.0,
              "contributions": [{"check": "missing_coverage", "points": -30.0, "applicable": True}],
              "claims": [{**REPORT["claims"][0], "verdict": "Not Enough Evidence", "coverage": cov}],
              "coverage": {"applied": True, "claim": "Shah Rukh Khan died in a car accident this morning.",
                           "why": "a death involving Shah Rukh Khan would be reported by trusted outlets if true",
                           "search": cov["search"], "likelihood_ratio": 0.25}}
    monkeypatch.setattr(api, "analyze", lambda **kw: silent)
    at = app().run()
    at.text_area[0].set_value("Shah Rukh Khan died in a car accident this morning.")
    at.button[0].click().run()
    assert not at.exception
    assert "What the news says" in " ".join(m.value for m in at.markdown)
    assert any("Would be in the news if true (a death)" in c.value for c in at.caption)


def test_refused_input_shows_reason_without_status_code(fake_api, monkeypatch):
    def refuse(**kw):
        raise api.ApiError("The text does not look like real words (it reads like random letters).", 422)

    monkeypatch.setattr(api, "analyze", refuse)
    at = app().run()
    at.text_area(key="text_input").set_value("asdfgh qwerty")
    at.button[0].click().run()
    assert not at.exception and not at.error
    assert any(w.value.startswith("The text does not look like real words") for w in at.warning)


def test_unscored_item_says_why_no_score(fake_api, monkeypatch):
    unscored = {**REPORT, "score": None, "band": "Unverified", "abstained": True, "contributions": [],
                "explanation": {"summary": "Not enough evidence to score this item.", "explanation": "Not scored.",
                                "evidence": [], "source": "template"}}
    monkeypatch.setattr(api, "analyze", lambda **kw: unscored)
    at = app().run()
    at.text_area(key="text_input").set_value("Reserve Bank of India announces new Rs 1000 currency notes.")
    at.button[0].click().run()
    md = " ".join(m.value for m in at.markdown)
    assert not at.exception and "Why no score" in md and "Why this score" not in md


def test_api_down_shows_error(monkeypatch):
    def down(*a, **k):
        raise api.ApiError("Cannot reach the API at http://localhost:8000.")

    for name in ("health", "history", "analyze"):
        monkeypatch.setattr(api, name, down)
    import streamlit as st

    st.cache_data.clear()  # the app caches /health for 15 s
    at = app().run()
    assert not at.exception and any("Cannot reach the API" in e.value for e in at.sidebar.error)


def test_charts():
    fig = contributions_bar(REPORT["contributions"])
    bar = fig.data[0]
    assert list(bar.y) == ["Checks that did not run", "Source rating", "Claim verification"]  # largest on top
    assert bar.marker.color[-1] == "#C4480E" and bar.marker.color[0] == "#B8B2A6"  # rust lowered, grey not a check
    assert band_style("High")[0] == "#1F5FA6"
    assert "left:28%" in meter_html(28) and ">28<" in score_html(28, "Low")
