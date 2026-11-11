"""Map every dataset's raw label onto one target, `credible` in [0, 1].

Mapping table (Phase 2):

| Dataset     | Raw label                                       | credible |
| ----------- | ----------------------------------------------- | -------- |
| AVeriTeC    | Supported                                       | 1.0      |
| AVeriTeC    | Refuted                                         | 0.0      |
| AVeriTeC    | Conflicting Evidence/Cherrypicking              | 0.5      |
| AVeriTeC    | Not Enough Evidence                             | NaN (verifier only, not meta-learner) |
| Fakeddit    | True                                            | 1.0      |
| Fakeddit    | Satire, Misleading, Imposter, False connection, Manipulated | 0.0 (+ reason tag) |
| FakeNewsNet | Real / Fake                                     | 1.0 / 0.0 |
| MediaEval   | Real / Fake (Humor treated as Fake)             | 1.0 / 0.0 |

Each mapper returns a `LabelResult`. Unknown labels raise `ValueError` so a
typo or a new label in a dataset release fails loudly instead of becoming a
silent wrong target.
"""

from __future__ import annotations

import math
import re
from typing import NamedTuple

import pandas as pd


class LabelResult(NamedTuple):
    credible: float  # 1.0 credible, 0.0 not credible, 0.5 mixed, NaN no verdict
    raw_label: str  # canonical name of the original label
    reason_tag: str | None  # misinformation sub-type, kept for explanations
    meta_eligible: bool  # may this row train the meta-learner (Phase 4)?


def _norm(label: object) -> str:
    """Lower-case, unify separators: 'False_Connection' -> 'false connection'."""
    s = str(label).strip().lower()
    s = re.sub(r"[_\-/]+", " ", s)
    return re.sub(r"\s+", " ", s)


# --- AVeriTeC -----------------------------------------------------------------

_AVERITEC = {
    "supported": LabelResult(1.0, "Supported", None, True),
    "refuted": LabelResult(0.0, "Refuted", None, True),
    "conflicting evidence cherrypicking": LabelResult(0.5, "Conflicting Evidence/Cherrypicking", "conflicting", True),
    "conflicting evidence cherry picking": LabelResult(0.5, "Conflicting Evidence/Cherrypicking", "conflicting", True),
    "not enough evidence": LabelResult(math.nan, "Not Enough Evidence", None, False),
}


def map_averitec(label: object) -> LabelResult:
    key = _norm(label)
    if key not in _AVERITEC:
        raise ValueError(f"Unknown AVeriTeC label: {label!r}")
    return _AVERITEC[key]


# --- Fakeddit -----------------------------------------------------------------

# Fakeddit 6-way codes as released (README of entitize/Fakeddit).
FAKEDDIT_6WAY_CODES = {
    0: "true",
    1: "satire",
    2: "misleading content",
    3: "imposter content",
    4: "false connection",
    5: "manipulated content",
}
_FAKEDDIT_ALIASES = {
    "true": "true",
    "satire": "satire",
    "satire parody": "satire",
    "parody": "satire",
    "misleading": "misleading content",
    "misleading content": "misleading content",
    "imposter": "imposter content",
    "imposter content": "imposter content",
    "false connection": "false connection",
    "manipulated": "manipulated content",
    "manipulated content": "manipulated content",
}


def map_fakeddit(label: object) -> LabelResult:
    """Accepts the 6-way integer code (0-5) or its name."""
    if isinstance(label, (int, float)) and not pd.isna(label) and int(label) in FAKEDDIT_6WAY_CODES:
        name = FAKEDDIT_6WAY_CODES[int(label)]
    else:
        key = _norm(label)
        if key.isdigit() and int(key) in FAKEDDIT_6WAY_CODES:
            name = FAKEDDIT_6WAY_CODES[int(key)]
        elif key in _FAKEDDIT_ALIASES:
            name = _FAKEDDIT_ALIASES[key]
        else:
            raise ValueError(f"Unknown Fakeddit 6-way label: {label!r}")
    if name == "true":
        return LabelResult(1.0, "True", None, True)
    tag = name.replace(" ", "_")  # e.g. "false_connection"
    return LabelResult(0.0, name.title(), tag, True)


# --- FakeNewsNet and MediaEval (binary) ---------------------------------------

_REAL = {"real", "true", "1", "1.0"}
_FAKE = {"fake", "false", "0", "0.0"}


def map_fakenewsnet(label: object) -> LabelResult:
    key = _norm(label)
    if key in _REAL:
        return LabelResult(1.0, "Real", None, True)
    if key in _FAKE:
        return LabelResult(0.0, "Fake", None, True)
    raise ValueError(f"Unknown FakeNewsNet label: {label!r}")


def map_mediaeval(label: object) -> LabelResult:
    key = _norm(label)
    if key in _REAL:
        return LabelResult(1.0, "Real", None, True)
    if key in _FAKE:
        return LabelResult(0.0, "Fake", None, True)
    if key == "humor":
        # MediaEval 2016 adds 'humor' (jokes using real images out of context);
        # the task itself scores it as fake.
        return LabelResult(0.0, "Humor", "humor", True)
    raise ValueError(f"Unknown MediaEval label: {label!r}")


MAPPERS = {
    "averitec": map_averitec,
    "fakeddit": map_fakeddit,
    "fakenewsnet": map_fakenewsnet,
    "mediaeval": map_mediaeval,
}


def unify_labels(df: pd.DataFrame, dataset: str, label_col: str = "label") -> pd.DataFrame:
    """Return a copy of `df` with `credible`, `raw_label`, `reason_tag`, `meta_eligible`.

    The original `label_col` is dropped; `raw_label` keeps its canonical name.
    """
    mapper = MAPPERS[dataset]
    results = [mapper(v) for v in df[label_col]]
    out = df.drop(columns=[label_col]).copy()
    out["credible"] = [r.credible for r in results]
    out["raw_label"] = [r.raw_label for r in results]
    out["reason_tag"] = [r.reason_tag for r in results]
    out["meta_eligible"] = [r.meta_eligible for r in results]
    return out
