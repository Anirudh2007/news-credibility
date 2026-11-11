"""Thin HTTP client for the FastAPI backend. The UI never imports the ML code:
everything goes through the API, exactly as a deployed frontend would."""

from __future__ import annotations

import os
from typing import Any, Optional

import requests

API_URL = os.environ.get("NEWSCRED_API", "http://localhost:8000").rstrip("/")
ANALYZE_TIMEOUT = 300  # first request loads every model; later ones take seconds


class ApiError(Exception):
    """Readable error for the UI (connection problems and HTTP errors). status: the HTTP status, if any."""

    def __init__(self, message: str, status: Optional[int] = None):
        super().__init__(message)
        self.status = status


def _detail(resp: requests.Response) -> str:
    try:
        detail = resp.json().get("detail", resp.text)
    except ValueError:
        return resp.text[:300]
    if isinstance(detail, list):  # Pydantic validation errors
        return "; ".join(f"{'.'.join(map(str, d.get('loc', [])[1:]))}: {d.get('msg')}" for d in detail)
    return str(detail)


def _request(method: str, path: str, timeout: float = 30, **kwargs) -> Any:
    try:
        resp = requests.request(method, f"{API_URL}{path}", timeout=timeout, **kwargs)
    except requests.ConnectionError:
        raise ApiError(f"Cannot reach the API at {API_URL}. Start it with: uvicorn api.main:app") from None
    except requests.Timeout:
        raise ApiError(f"The API did not answer within {timeout} s.") from None
    if resp.status_code >= 400:
        detail = _detail(resp)
        if resp.status_code in (400, 404, 413, 422):  # input refused or not found: the API's reason is the message
            raise ApiError(detail, resp.status_code)
        raise ApiError(f"The API returned an error ({resp.status_code}): {detail}", resp.status_code)
    return resp.json() if resp.content else None


def health() -> dict:
    return _request("GET", "/health", timeout=5)


def analyze(
    text: Optional[str] = None,
    url: Optional[str] = None,
    image: Optional[tuple[str, bytes, str]] = None,  # (filename, bytes, mime type)
    claimed_date: Optional[str] = None,
    claimed_location: Optional[str] = None,
    publish_date: Optional[str] = None,
    explain: bool = True,
    refresh: bool = False,
) -> dict:
    data = {k: v for k, v in {
        "text": text, "url": url, "claimed_date": claimed_date, "claimed_location": claimed_location,
        "publish_date": publish_date,
    }.items() if v}
    data["explain"] = str(explain).lower()
    data["refresh"] = str(refresh).lower()
    files = {"image": image} if image else None
    return _request("POST", "/api/v1/analyze", timeout=ANALYZE_TIMEOUT, data=data, files=files)


def history(limit: int = 20, offset: int = 0) -> dict:
    return _request("GET", "/api/v1/history", params={"limit": limit, "offset": offset})


def get_analysis(analysis_id: str) -> dict:
    return _request("GET", f"/api/v1/analyses/{analysis_id}")


def delete_analysis(analysis_id: str) -> None:
    _request("DELETE", f"/api/v1/analyses/{analysis_id}")


def send_feedback(analysis_id: str, agrees: bool, correct_label: Optional[str] = None,
                  comment: Optional[str] = None) -> dict:
    body = {"analysis_id": analysis_id, "agrees": agrees, "correct_label": correct_label, "comment": comment or None}
    return _request("POST", "/api/v1/feedback", json=body)


def feedback_summary() -> dict:
    return _request("GET", "/api/v1/feedback/summary")
