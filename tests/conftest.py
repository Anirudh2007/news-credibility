import pytest

from api import db


@pytest.fixture(autouse=True)
def in_memory_db(monkeypatch):
    """Every test gets a fresh in-memory SQLite database; nothing touches data/app.db.
    The API lifespan reads NEWSCRED_DB_URL, so TestClient(app) uses it too."""
    monkeypatch.setenv("NEWSCRED_DB_URL", "sqlite://")
    db.configure("sqlite://")
    yield


@pytest.fixture(autouse=True)
def no_live_evidence(monkeypatch):
    """No test calls the Google Fact Check API: live lookups are off unless a test injects a fake client."""
    monkeypatch.setenv("NEWSCRED_LIVE_FACTCHECK", "0")
