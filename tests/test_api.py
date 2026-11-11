import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.main import app, get_explainer, get_guard, get_pipeline
from src.ingest.guard import InputGuard

RULES_GUARD = InputGuard(corpus=[])  # corpus-free rules only: tests never read models/


class FakePipeline:
    status = {"meta": "unavailable: meta-learner not trained; using equal-weight average"}

    def __init__(self):
        self.items = []

    def analyze(self, item):
        self.items.append(item)
        return {
            "analysis_id": "abc123", "score": 72, "probability": 0.72, "band": "Mostly credible",
            "claims": [{"text": item.text or "", "checkworthiness": 0.8, "verdict": "Supported", "credibility": 0.9,
                        "confidence": 0.8, "evidence": [{"source": "x.gov", "url": "https://x.gov", "snippet": "s", "weight": 1.0}],
                        "entities": [["India", "GPE"]], "dates": []}] if item.text else [],
            "media": {}, "channels": {"source": {"score": 0.7}}, "features": {"source_score": 0.7, "text_image_score": None},
            "contributions": [], "skipped_checks": [] if item.image is not None else ["media_reuse: no image"],
            "reasons": [], "meta_model": "equal-weight average", "model_version": "0.4.0",
        }


class FakeExplainer:
    status = "fake"

    def explain(self, report, item_text=None):
        return {"summary": "s", "explanation": "e", "cited_evidence": [], "source": "template"}


@pytest.fixture()
def client():
    fake = FakePipeline()
    app.dependency_overrides[get_pipeline] = lambda: fake
    app.dependency_overrides[get_explainer] = lambda: FakeExplainer()  # never call the real LLM router in tests
    app.dependency_overrides[get_guard] = lambda: RULES_GUARD
    with TestClient(app) as c:
        c.fake = fake
        yield c
    app.dependency_overrides.clear()


def png_bytes(size=(64, 64)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (10, 120, 200)).save(buf, "PNG")
    return buf.getvalue()


def test_analyze_text(client):
    r = client.post("/api/v1/analyze", data={"text": "ISRO launched Chandrayaan-3.", "claimed_date": "14 July 2023"})
    assert r.status_code == 200
    body = r.json()
    assert body["score"] == 72 and body["band"] == "Mostly credible" and body["claims"][0]["verdict"] == "Supported"
    assert body["features"]["text_image_score"] is None and "disclaimer" in body
    assert client.fake.items[-1].claimed_date == "2023-07-14"  # parsed to ISO before the pipeline


def test_verify_alias_with_image(client):
    r = client.post("/verify", data={"text": "Flood in Mumbai today", "claimed_location": "Mumbai"},
                    files={"image": ("flood.png", png_bytes(), "image/png")})
    assert r.status_code == 200
    item = client.fake.items[-1]
    assert item.image is not None and item.image.size == (64, 64) and item.claimed_place == "Mumbai"


def test_image_only_and_url_only_are_accepted(client):
    assert client.post("/verify", files={"image": ("a.png", png_bytes(), "image/png")}).status_code == 200
    assert client.post("/verify", data={"url": "https://www.reuters.com/x"}).status_code == 200


def test_empty_request_rejected(client):
    r = client.post("/api/v1/analyze", data={"text": "   "})
    assert r.status_code == 422 and "at least one" in r.json()["detail"]


def test_unreadable_text_is_refused_before_analysis(client):
    for text in ["😂😂😂🔥", "%%%% $$$$ ####", "BREAKING BREAKING BREAKING BREAKING", "ok", "मोदी ने नया नियम बनाया"]:
        r = client.post("/api/v1/analyze", data={"text": text})
        assert r.status_code == 422 and "Nothing was analysed" in r.json()["detail"], text
    assert client.fake.items == []  # the pipeline never ran, so no score was made up


def test_short_caption_with_image_is_accepted(client):
    r = client.post("/verify", data={"text": "2 cats"}, files={"image": ("c.png", png_bytes(), "image/png")})
    assert r.status_code == 200  # alone "2 cats" is too short; with its image it is a caption


def test_bad_inputs(client):
    assert client.post("/verify", data={"text": "Flood in Mumbai today", "claimed_date": "not a date"}).status_code == 422
    assert client.post("/verify", files={"image": ("a.txt", b"hello", "text/plain")}).status_code == 415
    assert client.post("/verify", files={"image": ("a.png", b"not really a png", "image/png")}).status_code == 422
    assert client.post("/verify", files={"image": ("a.png", png_bytes((16, 16)), "image/png")}).status_code == 422  # tiny
    assert client.post("/verify", data={"text": "x" * 20_001}).status_code == 413


def test_image_too_large(client, monkeypatch):
    import api.main as m

    monkeypatch.setattr(m, "MAX_IMAGE_BYTES", 100)
    assert client.post("/verify", files={"image": ("a.png", png_bytes(), "image/png")}).status_code == 413


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok" and "meta" in r.json()["components"]


def test_sources_endpoint(client, monkeypatch):
    import src.source.credibility as sc

    monkeypatch.setattr(sc, "load_ratings", lambda: {"reuters.com": sc.SourceRating(0.95, "center")})
    r = client.get("/api/v1/sources/www.reuters.com")
    assert r.status_code == 200 and r.json() == {"domain": "reuters.com", "reliability": 0.95, "bias": "center"}
    assert client.get("/api/v1/sources/unknown-site.example").status_code == 404


def test_openapi_lists_endpoints(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/api/v1/analyze", "/verify", "/health", "/api/v1/sources/{domain}"} <= set(paths)
