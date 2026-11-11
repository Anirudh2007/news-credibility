"""Phase 5: explanation layer.

1. Contributions: how many points each check added or removed (SHAP).
   LightGBM computes exact TreeSHAP values itself (`pred_contrib=True`), in
   log-odds. They are converted to points on the 0-100 scale so that
       baseline_score + sum(points) = final score
   ("the average item scores 52; the media check removed 31 points ...").
   Checks that did not run move no points: their SHAP share is folded into the
   baseline, which therefore means "an average item with the same inputs"
   (e.g. text only). SHAP gives missing features a share only because the
   training average had images and URLs; showing it as a bar made skipped
   checks look like evidence. The isotonic calibration step is its own line.
2. Templates: deterministic sentences per claim and for the whole report.
   Always available; used whenever the LLM is off, down, or fails validation.
3. LLM explanation: a local OpenAI-compatible router (freellmapi, default
   http://localhost:3001/v1) rewrites the curated FACTS into readable prose.
   Output is validated in code: wrong shape, unknown evidence ids, numbers that
   are not in FACTS, or talk about skipped checks all reject it (template used).

Configuration (environment variables):
    EXPLAIN_BASE_URL   default http://localhost:3001/v1
    EXPLAIN_MODEL      model id; default = first id the router lists
    EXPLAIN_API_KEY    default "not-needed"
    EXPLAIN_ENABLED    "0" disables the LLM (templates only)

Privacy: the router forwards prompts to third-party model providers, so the
submitted text and evidence snippets leave this machine when the LLM is on.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
import time
from typing import Any

import numpy as np

from src.explain.prompt import PROMPT_VERSION, messages
from src.fusion.features import CHANNEL_FEATURES, CHANNEL_SCORE, FEATURE_COLUMNS

DEFAULT_BASE_URL = "http://localhost:3001/v1"
CONNECT_TIMEOUT = 3.0  # seconds to reach the router; generation itself may take up to `timeout`
ROUTER_RETRY_SECONDS = 60  # after a connection failure, use templates for this long
CHECK_NAMES = {  # channel group -> name shown to users
    "claims": "claim_verification",
    "text_image": "text_image_match",
    "reuse": "media_reuse",
    "source": "source",
    "temporal": "temporal",
    "modality": "input_type",
}
NOT_APPLICABLE = "not_applicable"
CALIBRATION = "calibration"  # isotonic adjustment; a model step, not a check, so never shown to the LLM
NOT_CHECKS = {NOT_APPLICABLE, CALIBRATION}
MIN_POINTS_FOR_FACTS = 0.5  # smaller contributions are noise; keep them out of the LLM prompt
# Display threshold for "the evidence is about this claim": sigmoid(re-ranker logit) 0.5 = logit 0,
# the cross-encoder's own relevant/not-relevant boundary. Used for wording only, never for the score
# (the meta-learner reads the raw relevance as a feature).
LOW_RELEVANCE = 0.5


# --- 1. Contributions ------------------------------------------------------------------

def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def _ran(group: str, row: dict[str, Any]) -> bool:
    if group == "modality":
        return True
    return any(row.get(c) is not None and not (isinstance(row.get(c), float) and math.isnan(row[c]))
               for c in CHANNEL_FEATURES[group])


def _package(points: dict[str, float], row: dict[str, Any], score: float) -> tuple[float, list[dict]]:
    """Checks that ran get bars; the points of checks that did not run are folded into the
    baseline ("average item with the same inputs"). baseline + bars = score, exactly."""
    applicable = {g: p for g, p in points.items() if _ran(g, row) and (g != "modality" or abs(p) >= 0.05)}
    out = [{"check": CHECK_NAMES[g], "points": round(p, 1), "applicable": True}
           for g, p in sorted(applicable.items(), key=lambda kv: -abs(kv[1]))]
    baseline = score - sum(applicable.values())  # additivity holds by construction
    return round(baseline, 1), out


def meta_contributions(meta, vec: np.ndarray, row: dict[str, Any]) -> tuple[float, list[dict]]:
    """Points per check for a LightGBM + isotonic meta-model.

    TreeSHAP gives log-odds contributions phi_i with base + sum(phi) = logit(p_raw).
    They are mapped to points in the model's own (uncalibrated) probability space,
    which always reflects what the checks did:
        100*sigmoid(base) + sum(check points) = 100*p_raw
    The isotonic calibrator is a step function and can map very different raw
    probabilities to the same value (flat steps), which would turn every check's
    points into 0. So calibration is reported as its own line instead:
        baseline + sum(check points) + calibration = final score   (exact)
    """
    X = np.atleast_2d(np.asarray(vec, dtype=float))
    contrib = np.asarray(meta.booster.predict(X, pred_contrib=True))[0]
    phi, base = contrib[:-1], float(contrib[-1])
    total = float(phi.sum())
    p0, p1 = _sigmoid(base), _sigmoid(base + total)
    # one factor maps the whole log-odds path base -> final onto probability points
    scale = 100.0 * (p1 - p0) / total if abs(total) > 1e-9 else 100.0 * p1 * (1 - p1)
    groups = {g: [FEATURE_COLUMNS.index(c) for c in cols] for g, cols in CHANNEL_FEATURES.items()}
    points = {g: scale * float(phi[idx].sum()) for g, idx in groups.items()}
    final = 100.0 * float(np.clip(meta.calibrator.predict([p1])[0], 0, 1)) if meta.calibrator is not None else 100.0 * p1
    baseline, out = _package(points, row, 100.0 * p1)
    calibration = final - 100.0 * p1
    if abs(calibration) >= 0.05:
        out.append({"check": CALIBRATION, "points": round(calibration, 1), "applicable": True})
    return baseline, out


def average_contributions(row: dict[str, Any]) -> tuple[float, list[dict]]:
    """Points for the equal-weight fallback: score = mean of n channel scores, so
    channel c contributes 100 * (s_c - 0.5) / n on top of a 50-point baseline. Exact."""
    scores = {g: row.get(col) for g, col in CHANNEL_SCORE.items()}
    scores = {g: s for g, s in scores.items() if s is not None and not (isinstance(s, float) and math.isnan(s))}
    if not scores:
        return math.nan, []
    n = len(scores)
    points = {g: 100.0 * (s - 0.5) / n for g, s in scores.items()}
    return _package(points, row, 50.0 + sum(points.values()))


def contributions(meta, vec: np.ndarray, row: dict[str, Any]) -> tuple[float, list[dict]]:
    return meta_contributions(meta, vec, row) if meta is not None else average_contributions(row)


# --- 2. Templates ----------------------------------------------------------------------

def weak_evidence(claim: dict) -> bool:
    rel = claim.get("evidence_relevance")
    return rel is not None and not (isinstance(rel, float) and math.isnan(rel)) and rel < LOW_RELEVANCE


def claim_justification(claim: dict) -> str:
    """One templated sentence per claim."""
    verdict = claim.get("verdict")
    if claim.get("context") == "missing":
        return ("This sentence does not say who or what it is about, so it cannot be checked against evidence."
                " Add names, places or dates.")
    if verdict is None:
        return "This claim was extracted but not verified against evidence."
    text = _verdict_sentence(claim)
    if weak_evidence(claim) and verdict != "Not Enough Evidence":
        text += (" The closest evidence in the index is only loosely related to this claim"
                 " (it was written about a different claim), so treat this verdict with caution.")
    return text


# "What the evidence says": quoted evidence only, never text written by a model. A claim gets an
# entry only when its verdict is decisive (Supported / Refuted), the verifier is confident, and the
# evidence is about this claim. Thresholds: see `python -m src.fusion.evaluate evidence-box`.
BOX_MIN_CONFIDENCE = 0.6  # 1 - P(Not Enough Evidence)
BOX_MIN_RELEVANCE = 0.5  # sigmoid of the best re-ranker score


def evidence_box(claims: list[dict]) -> list[dict]:
    """One entry per qualifying claim: the verdict and the strongest passage, quoted with its source,
    link and date. Empty list = nothing clear enough to quote (the UI then shows no box)."""
    out = []
    for c in claims:
        ev = (c.get("evidence") or [None])[0]
        if (c.get("verdict") in ("Supported", "Refuted") and ev and not c.get("evidence_off_topic")
                and (c.get("confidence") or 0) >= BOX_MIN_CONFIDENCE
                and (c.get("evidence_relevance") or 0) >= BOX_MIN_RELEVANCE):
            fc = (c.get("fact_checks") or [None])[0]
            out.append({"claim": c["text"], "verdict": c["verdict"], "quote": ev["snippet"],
                        "source": ev.get("source"), "url": ev.get("url"), "date": ev.get("date"),
                        "fact_checked_by": fc["publisher"] if fc else None})
    return out


def _verdict_sentence(claim: dict) -> str:
    verdict, conf = claim.get("verdict"), claim.get("confidence")
    ev = (claim.get("evidence") or [None])[0]
    cite = f' Strongest evidence ({ev.get("source") or "unknown source"}): "{ev["snippet"][:160].strip()}"' if ev else ""
    conf_txt = f" ({round(100 * conf)}% confidence)" if conf is not None else ""
    if verdict == "Supported":
        return f"The evidence found supports this claim{conf_txt}.{cite}"
    if verdict == "Refuted":
        return f"The evidence found contradicts this claim{conf_txt}.{cite}"
    if verdict == "Not Enough Evidence":
        if claim.get("evidence_off_topic"):
            return ("No evidence about this claim was found: the closest passages in the evidence index are about"
                    " other claims, so this claim did not change the score.")
        return "Not enough evidence was found to support or refute this claim."
    return f"The evidence found is mixed or only partly supports this claim{conf_txt}.{cite}"


_CHECK_PHRASES = {
    "claim_verification": "claim verification",
    "text_image_match": "the caption-image match",
    "media_reuse": "the media reuse check",
    "source": "the source rating",
    "temporal": "the date consistency check",
    "input_type": "the type of input",
    "missing_coverage": "no trusted news report of a claim that would be widely reported if true",
}


def _headline(report: dict, applicable: list[dict]) -> str | None:
    """One sentence on what decided the score, from the report's own fields (the big number is shown next to
    it, so it is not repeated). A confident, on-topic verdict first, in the direction of the score; otherwise
    the check that moved the score most. None: nothing to say beyond the score."""
    claims = report.get("claims") or []
    decisive = [c for c in claims if c.get("verdict") in ("Supported", "Refuted")
                and not c.get("evidence_off_topic") and not weak_evidence(c)]
    order = [("Refuted", "contradicts"), ("Supported", "supports")]
    if (report.get("score") or 0) > 60:
        order.reverse()
    for verdict, verb in order:
        hits = [c for c in decisive if c["verdict"] == verdict]
        if hits:
            which = ("the claim" if len(claims) == 1 else
                     f"{len(hits)} of the {len(claims)} claims" if len(hits) > 1 else f"one of the {len(claims)} claims")
            best = max(hits, key=lambda c: c.get("confidence") or 0)
            # Cite a fact-check only if its rating agrees with the verdict: a matched "Mostly false" check must not
            # appear as "The evidence supports the claim (fact-checked by ...: Mostly false)".
            from src.retrieval.factcheck_api import rating_to_credible

            agrees = 0.0 if verdict == "Refuted" else 1.0
            fc = next((f for f in best.get("fact_checks") or [] if f.get("publisher") and f.get("rating")
                       and rating_to_credible(f["rating"]) == agrees), None)
            by = f" (fact-checked by {fc['publisher']}: {fc['rating']})" if fc else ""
            return f"The evidence {verb} {which}{by}."
    top = max(applicable, key=lambda c: abs(c["points"]), default=None)
    if top is None or abs(top["points"]) < MIN_POINTS_FOR_FACTS:
        return None
    unsettled = ("No evidence settled the claim. " if len(claims) == 1 else
                 "No evidence settled the claims. " if claims else "")
    return f"{unsettled}Mostly {'lowered' if top['points'] < 0 else 'raised'} by {_CHECK_PHRASES.get(top['check'], top['check'])}."


def template_explanation(report: dict) -> dict:
    """Deterministic explanation from the report's own fields."""
    score, band = report.get("score"), report.get("band", "Unknown")
    if score is None and report.get("abstained"):
        why = next((r for r in report.get("reasons", []) if r.startswith("Not scored")), "Not scored.")
        return {"summary": "Not enough evidence to score this item.",
                "explanation": f"{why} Add a source URL or an image, or search fact-checkers such as PIB Fact Check,"
                               " BOOM, Alt News or Factly.", "cited_evidence": []}
    if score is None:
        return {"summary": "No check could be run on this input, so no score was produced.",
                "explanation": "Provide text, an image or a source URL to run the checks.", "cited_evidence": []}
    applicable = [c for c in report.get("contributions", []) if c.get("applicable") and c["check"] not in NOT_CHECKS]
    ups = [c for c in applicable if c["points"] >= MIN_POINTS_FOR_FACTS]
    downs = [c for c in applicable if c["points"] <= -MIN_POINTS_FOR_FACTS]
    headline = _headline(report, applicable)
    parts = [headline] if headline else []
    parts.append(f"Scored {score}/100 ({band})." if headline else f"The item scored {score}/100 ({band}).")
    if downs:
        parts.append("Lowered by " + ", ".join(f"{_CHECK_PHRASES.get(c['check'], c['check'])} ({c['points']:+.0f})" for c in downs[:3]) + ".")
    if ups:
        parts.append("Raised by " + ", ".join(f"{_CHECK_PHRASES.get(c['check'], c['check'])} ({c['points']:+.0f})" for c in ups[:3]) + ".")
    verdicts = [c for c in report.get("claims", []) if c.get("verdict")]
    if verdicts:
        counts = {}
        for c in verdicts:
            counts[c["verdict"]] = counts.get(c["verdict"], 0) + 1
        parts.append("Claims: " + ", ".join(f"{n} {v}" for v, n in counts.items()) + ".")
    parts += report.get("reasons", [])[:3]
    return {"summary": parts[0], "explanation": " ".join(parts[1:]) or parts[0], "cited_evidence": []}


# --- 3. LLM explanation ------------------------------------------------------------------

def build_facts(report: dict) -> dict:
    """Curated input for the LLM: only results of checks that ran. Skipped checks,
    raw features and internal names are left out on purpose."""
    facts: dict[str, Any] = {"score": report.get("score"), "band": report.get("band")}
    if report.get("baseline_score") is not None and not math.isnan(report["baseline_score"]):
        facts["average_item_score"] = round(report["baseline_score"])
    checks = [{"check": _CHECK_PHRASES.get(c["check"], c["check"]), "points": int(round(c["points"]))}
              for c in report.get("contributions", [])
              if c.get("applicable") and c["check"] not in NOT_CHECKS and abs(c["points"]) >= MIN_POINTS_FOR_FACTS]
    if checks:
        facts["check_points"] = checks
    claims, n = [], 0
    for c in report.get("claims", []):
        if not c.get("verdict"):
            continue
        ev = []
        for e in c.get("evidence", []):
            n += 1
            ev.append({"id": f"E{n}", "source": e.get("source"), "snippet": (e.get("snippet") or "")[:300]})
        claims.append({"claim": c["text"], "verdict": c["verdict"],
                       "confidence": None if c.get("confidence") is None else round(c["confidence"], 2),
                       **({"evidence_match": "weak, written about a different claim"} if weak_evidence(c) else {}),
                       "evidence": ev})
    if claims:
        facts["claims"] = claims
    media = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in (report.get("media") or {}).items() if v is not None}
    if media:
        facts["media_findings"] = media
    if report.get("reasons"):
        facts["findings"] = list(report["reasons"])
    return facts


# Thousands separators first ("4,000" is 4000, not 4.0), then plain decimals.
_NUM_RE = re.compile(r"(?<![A-Za-z\d])[-−+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
_ID_RE = re.compile(r"\[(E\d+)\]")
_SKIP_RE = re.compile(r"\b(skipped|not applicable|unavailable|not available|not (?:been )?checked|could not be (?:checked|verified))\b", re.I)
_LOADED_RE = re.compile(r"\b(fake news|hoax|propaganda|liars?|lies)\b", re.I)


def _numbers(text: str) -> set[float]:
    out = set()
    for m in _NUM_RE.findall(text or ""):
        try:
            out.add(abs(float(m.replace("−", "-").replace(",", ""))))
        except ValueError:
            pass
    return out


def allowed_numbers(facts: dict, item_text: str | None) -> set[float]:
    """Numbers the explanation may use: everything in FACTS and the item text,
    fractions in FACTS also as percentages (0.87 -> 87), and small counts 0-10."""
    blob = json.dumps(facts, ensure_ascii=False)
    nums = _numbers(blob) | _numbers(item_text or "") | {float(i) for i in range(11)} | {100.0}
    nums |= {float(round(x * 100)) for x in list(nums) if 0 < x < 1}
    return nums


def _strip_fences(content: str) -> str:
    s = (content or "").strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    start, end = s.find("{"), s.rfind("}")
    return s[start:end + 1] if start != -1 and end > start else s


def validate(content: str, facts: dict, item_text: str | None = None) -> tuple[dict | None, list[str]]:
    """Parse and check the LLM output. Returns (parsed, errors); errors -> reject."""
    errors: list[str] = []
    try:
        out = json.loads(_strip_fences(content))
    except (json.JSONDecodeError, TypeError):
        return None, ["not valid JSON"]
    if not isinstance(out, dict):
        return None, ["JSON is not an object"]
    summary, expl = out.get("summary"), out.get("explanation")
    cited = out.get("cited_evidence", [])
    if not isinstance(summary, str) or not summary.strip():
        errors.append("missing summary")
    if not isinstance(expl, str) or not expl.strip():
        errors.append("missing explanation")
    if not isinstance(cited, list) or not all(isinstance(c, str) for c in cited):
        errors.append("cited_evidence is not a list of ids")
        cited = []
    if errors:
        return None, errors
    text = f"{summary} {expl}"
    if len(summary.split()) > 40 or len(expl.split()) > 160:
        errors.append("too long")
    known_ids = {e["id"] for c in facts.get("claims", []) for e in c["evidence"]}
    bad_ids = (set(cited) | set(_ID_RE.findall(text))) - known_ids
    if bad_ids:
        errors.append(f"unknown evidence ids: {sorted(bad_ids)}")
    if re.search(r"https?://", text):
        errors.append("contains a URL")
    unknown = sorted(n for n in _numbers(text) if n not in allowed_numbers(facts, item_text))
    if unknown:
        errors.append(f"numbers not in facts: {unknown[:5]}")
    if _SKIP_RE.search(text):
        errors.append("mentions skipped or unavailable checks")
    facts_blob = json.dumps(facts).lower() + " " + (item_text or "").lower()
    loaded = [w for w in _LOADED_RE.findall(text) if w.lower() not in facts_blob]
    if loaded:
        errors.append(f"loaded wording not in facts: {sorted(set(w.lower() for w in loaded))}")
    return ({"summary": summary.strip(), "explanation": expl.strip(), "cited_evidence": list(dict.fromkeys(cited))},
            errors)


class Explainer:
    """explain(report, item_text) -> explanation dict, LLM if possible, else template."""

    def __init__(self, base_url: str | None = None, model: str | None = None, api_key: str | None = None,
                 timeout: float = 20.0, client=None, enabled: bool | None = None, cache_size: int = 256):
        self.base_url = base_url or os.environ.get("EXPLAIN_BASE_URL", DEFAULT_BASE_URL)
        self.model = model or os.environ.get("EXPLAIN_MODEL") or None
        self.api_key = api_key or os.environ.get("EXPLAIN_API_KEY", "not-needed")
        self.timeout = timeout
        self.enabled = enabled if enabled is not None else os.environ.get("EXPLAIN_ENABLED", "1") != "0"
        self._client = client
        self._cache: dict[str, dict] = {}
        self._cache_size = cache_size
        self._lock = threading.Lock()
        self._down_until = 0.0  # circuit breaker: skip the LLM until this time after a connection failure

    @property
    def status(self) -> str:
        if not self.enabled:
            return "templates only (EXPLAIN_ENABLED=0)"
        down = self._down_until - time.monotonic()
        state = f"router unreachable, retry in {int(down)} s" if down > 0 else "template fallback"
        return f"llm via {self.base_url} (model: {self.model or 'auto'}); {state}"

    def client(self):
        if self._client is None:
            import httpx
            from openai import OpenAI

            # No retries and a short connect timeout: a dead router must cost ~seconds once,
            # not on every request (measured: 9 s per request with default retries on Windows).
            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key, max_retries=0,
                                  timeout=httpx.Timeout(self.timeout, connect=CONNECT_TIMEOUT))
        return self._client

    def resolve_model(self) -> str:
        if not self.model:
            models = list(self.client().models.list())
            if not models:
                raise RuntimeError("router lists no models; set EXPLAIN_MODEL")
            self.model = models[0].id
        return self.model

    def explain(self, report: dict, item_text: str | None = None) -> dict:
        fallback = {**template_explanation(report), "source": "template", "model": None,
                    "prompt_version": PROMPT_VERSION, "validation_errors": [], "llm_error": None}
        if not self.enabled or report.get("score") is None:
            return fallback
        facts = build_facts(report)
        key = hashlib.sha256(json.dumps([facts, item_text], sort_keys=True, default=str).encode()).hexdigest()
        with self._lock:
            if key in self._cache:
                return self._cache[key]
        if time.monotonic() < self._down_until:
            return {**fallback, "llm_error": f"router unreachable; retrying after {ROUTER_RETRY_SECONDS} s"}
        try:
            model = self.resolve_model()
            resp = self.client().chat.completions.create(
                model=model, messages=messages(facts, item_text), temperature=0.2, max_tokens=400,
            )
            content = resp.choices[0].message.content
        except Exception as e:  # router down, timeout, provider error: never break the report
            if type(e).__name__ in {"APIConnectionError", "APITimeoutError", "ConnectError", "ConnectionError"}:
                self._down_until = time.monotonic() + ROUTER_RETRY_SECONDS
            return {**fallback, "llm_error": f"{type(e).__name__}: {str(e)[:160]}"}
        parsed, errors = validate(content, facts, item_text)
        if parsed is None or errors:
            result = {**fallback, "model": model, "validation_errors": errors}
        else:
            evidence = {e["id"]: e for c in facts.get("claims", []) for e in c["evidence"]}
            result = {**parsed, "evidence": [evidence[i] for i in parsed["cited_evidence"] if i in evidence],
                      "source": "llm", "model": model, "prompt_version": PROMPT_VERSION,
                      "validation_errors": [], "llm_error": None}
        with self._lock:
            if len(self._cache) >= self._cache_size:
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = result
        return result
