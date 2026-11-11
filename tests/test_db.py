import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api import db
from api.main import app, get_explainer, get_guard, get_pipeline
from src.ingest.guard import GuardResult


class CountingPipeline:
    status = {}

    def __init__(self):
        self.calls = 0

    def analyze(self, item):
        self.calls += 1
        return {"analysis_id": "pipeline-hash", "score": 64, "probability": 0.64, "band": "Mostly credible",
                "claims": [], "media": {}, "channels": {}, "features": {}, "baseline_score": 50.0,
                "contributions": [{"check": "source", "points": 14.0, "applicable": True}],
                "skipped_checks": [], "reasons": [], "meta_model": "fake", "model_version": "0.4.0"}


class PassGuard:
    """Storage tests use placeholder texts ("Claim A"); input checking is tested in test_api/test_guard."""
    status = "pass-through"

    def check(self, text, caption=False):
        return GuardResult(True)


class CountingExplainer:
    status = "fake"

    def __init__(self):
        self.calls = 0

    def explain(self, report, item_text=None):
        self.calls += 1
        return {"summary": "s", "explanation": "e", "cited_evidence": [], "source": "template"}


@pytest.fixture()
def client():
    pipe, expl = CountingPipeline(), CountingExplainer()
    app.dependency_overrides[get_pipeline] = lambda: pipe
    app.dependency_overrides[get_explainer] = lambda: expl
    app.dependency_overrides[get_guard] = lambda: PassGuard()
    with TestClient(app) as c:
        c.pipe, c.expl = pipe, expl
        yield c
    app.dependency_overrides.clear()


def png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (0, 90, 200)).save(buf, "PNG")
    return buf.getvalue()


def test_analysis_is_stored_and_identical_input_is_cached(client):
    a = client.post("/api/v1/analyze", data={"text": "Claim A"}).json()
    assert a["cached"] is False and len(a["analysis_id"]) == 12 and a["created_at"]
    b = client.post("/api/v1/analyze", data={"text": "Claim A"}).json()
    assert b["cached"] is True and b["analysis_id"] == a["analysis_id"] and client.pipe.calls == 1
    c = client.post("/api/v1/analyze", data={"text": "Claim A", "refresh": "true"}).json()
    assert c["cached"] is False and c["analysis_id"] != a["analysis_id"] and client.pipe.calls == 2


def test_retrained_models_invalidate_the_cache(client):
    client.pipe.model_version = "0.5.0+aaaaaaaa"
    a = client.post("/api/v1/analyze", data={"text": "Claim B"}).json()
    assert a["model_version"] == "0.5.0+aaaaaaaa"
    client.pipe.model_version = "0.5.0+bbbbbbbb"  # e.g. meta-learner refit
    b = client.post("/api/v1/analyze", data={"text": "Claim B"}).json()
    assert b["cached"] is False and client.pipe.calls == 2
    assert client.get("/health").json()["model_version"] == "0.5.0+bbbbbbbb"


def test_timestamps_always_carry_utc_offset(client):
    a = client.post("/api/v1/analyze", data={"text": "Claim C"}).json()
    again = client.get(f"/api/v1/analyses/{a['analysis_id']}").json()
    listed = client.get("/api/v1/history").json()["items"][0]
    assert a["created_at"].endswith("+00:00") and again["created_at"].endswith("+00:00")
    assert listed["created_at"].endswith("+00:00")


def test_cache_adds_explanation_later_and_explain_false_hides_it(client):
    first = client.post("/verify", data={"text": "Claim B", "explain": "false"}).json()
    assert first["explanation"] is None and client.expl.calls == 0
    second = client.post("/verify", data={"text": "Claim B"}).json()
    assert second["cached"] and second["explanation"]["summary"] == "s" and client.expl.calls == 1
    third = client.post("/verify", data={"text": "Claim B"}).json()
    assert third["explanation"]["summary"] == "s" and client.expl.calls == 1  # stored now
    assert client.post("/verify", data={"text": "Claim B", "explain": "false"}).json()["explanation"] is None


def test_history_lists_newest_first_with_preview_and_feedback_count(client):
    ids = [client.post("/verify", data={"text": f"Claim {i}"}).json()["analysis_id"] for i in range(3)]
    client.post("/verify", data={"text": "With picture"}, files={"image": ("p.png", png(), "image/png")})
    client.post("/api/v1/feedback", json={"analysis_id": ids[0], "agrees": True})
    page = client.get("/api/v1/history", params={"limit": 2}).json()
    assert page["total"] == 4 and len(page["items"]) == 2
    assert page["items"][0]["preview"] == "With picture" and page["items"][0]["input_type"] == "text+image"
    everything = client.get("/api/v1/history", params={"limit": 10}).json()["items"]
    assert {i["analysis_id"]: i["feedback_count"] for i in everything}[ids[0]] == 1
    assert client.get("/api/v1/history", params={"limit": 0}).status_code == 422


def test_image_bytes_are_not_stored(client):
    rid = client.post("/verify", files={"image": ("p.png", png(), "image/png")}).json()["analysis_id"]
    gen = db.get_session()
    session = next(gen)
    row = session.get(db.Analysis, rid)
    assert row.has_image and row.input_type == "image" and row.text is None and b"PNG" not in row.report_json.encode()
    gen.close()


def test_feedback_flow_and_validation(client):
    rid = client.post("/verify", data={"text": "Claim C"}).json()["analysis_id"]
    r = client.post("/api/v1/feedback", json={"analysis_id": rid, "agrees": False,
                                              "correct_label": "not_credible", "comment": "  Old photo  "})
    assert r.status_code == 201 and r.json()["comment"] == "Old photo" and r.json()["correct_label"] == "not_credible"
    assert client.post("/api/v1/feedback", json={"analysis_id": "nope", "agrees": True}).status_code == 404
    assert client.post("/api/v1/feedback", json={"analysis_id": rid, "agrees": True, "correct_label": "fake"}).status_code == 422
    assert client.post("/api/v1/feedback", json={"analysis_id": rid, "agrees": True, "comment": "x" * 1001}).status_code == 422
    client.post("/api/v1/feedback", json={"analysis_id": rid, "agrees": True})
    detail = client.get(f"/api/v1/analyses/{rid}").json()
    assert [f["agrees"] for f in detail["feedback"]] == [False, True] and detail["score"] == 64
    assert client.get("/api/v1/feedback/summary").json() == {"feedback": 2, "agree": 1, "agree_rate": 0.5}


def test_delete_cascades_feedback(client):
    rid = client.post("/verify", data={"text": "Claim D"}).json()["analysis_id"]
    client.post("/api/v1/feedback", json={"analysis_id": rid, "agrees": True})
    assert client.delete(f"/api/v1/analyses/{rid}").status_code == 204
    assert client.get(f"/api/v1/analyses/{rid}").status_code == 404
    assert client.delete(f"/api/v1/analyses/{rid}").status_code == 404
    assert client.get("/api/v1/feedback/summary").json()["feedback"] == 0


def test_health_reports_database(client):
    client.post("/verify", data={"text": "Claim E"})
    assert client.get("/health").json()["database"] == "ok (1 analyses stored)"
