"""Channel 3F: source and social credibility (Phase 3F).

Three feature groups, then one classifier trained on FakeNewsNet:

1. Domain: reliability and bias from data/reference/sources.csv (registered
   domain via tldextract, so 'edition.cnn.com' -> 'cnn.com'), HTTPS, known
   satire, official/academic suffix (.gov, .edu, .gov.in), hyphens/digits in
   the name, subdomain depth. Domain age (WHOIS) is a hook left NaN.
2. Style (headline): clickbait phrases, sensational words, capital-letter and
   ALL-CAPS-word ratios, '!' and '?', second-person "you", length.
3. Social (optional): tweet count, and if a retweet graph is supplied, cascade
   size, depth, breadth and spread speed (NetworkX). The X/Twitter API is paid,
   so these are usually NaN; the model handles NaN natively.

The raw domain string is never a feature. In FakeNewsNet whole sites are fake
or real, so a domain identity feature would memorise sites instead of learning
what makes a source credible, and fail on any site not seen in training.

Output: score = P(real); confidence = |2p - 1|; NaN when there is neither a
domain nor a headline.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Sequence
from urllib.parse import urlparse

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCES_CSV = ROOT / "data" / "reference" / "sources.csv"
MODEL_PATH = ROOT / "models" / "source_clf.joblib"

_OFFICIAL_SUFFIXES = ("gov", "edu", "mil", "int", "gov.in", "nic.in", "ac.in", "gov.uk", "ac.uk", "edu.au")
_CLICKBAIT = [
    r"you won'?t believe", r"what happens next", r"this is why", r"here'?s why", r"will blow your mind",
    r"shocking", r"jaw[- ]dropping", r"goes viral", r"breaks the internet", r"the reason why",
    r"can'?t stop", r"you need to", r"must see", r"\bomg\b", r"\bwow\b", r"^\d+\s+(things|reasons|ways|times)\b",
]
_SENSATIONAL = {
    "shocking", "unbelievable", "outrage", "outrageous", "slams", "destroys", "exposed", "bombshell", "scandal",
    "horrifying", "terrifying", "miracle", "secret", "banned", "leaked", "explosive", "disaster", "urgent",
    "breaking", "insane", "epic", "hoax", "conspiracy", "cover-up", "furious", "stunning",
}
_CLICKBAIT_RE = re.compile("|".join(_CLICKBAIT), re.IGNORECASE)
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")


# --- Domain -------------------------------------------------------------------

@lru_cache(maxsize=1)
def _extractor():
    import tldextract

    return tldextract.TLDExtract(suffix_list_urls=())  # bundled public-suffix snapshot, works offline


def parse_domain(url_or_domain: str | None) -> tuple[str | None, str | None, str | None, str | None]:
    """-> (registered_domain, suffix, subdomain, scheme). 'https://edition.cnn.com/x'
    -> ('cnn.com', 'com', 'edition', 'https')."""
    if not url_or_domain or not isinstance(url_or_domain, str):
        return None, None, None, None
    s = url_or_domain.strip()
    scheme = urlparse(s).scheme.lower() if "://" in s else None
    ext = _extractor()(s)
    # tldextract >= 5.3 renamed registered_domain; read the old name only on old versions
    reg = ext.top_domain_under_public_suffix if hasattr(ext, "top_domain_under_public_suffix") else ext.registered_domain
    return (reg.lower() or None) if reg else None, ext.suffix.lower() or None, ext.subdomain.lower() or None, scheme


@dataclass(frozen=True)
class SourceRating:
    reliability: float
    bias: str | None


def load_ratings(path: str | Path = SOURCES_CSV) -> dict[str, SourceRating]:
    """registered domain -> rating, from sources.csv (domain, reliability, bias, ...)."""
    import pandas as pd

    path = Path(path)
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    out = {}
    for d, r, b in zip(df["domain"], df["reliability"], df.get("bias", [None] * len(df))):
        reg = parse_domain(str(d))[0] or str(d).lower()
        out[reg] = SourceRating(float(r), None if b != b else str(b).lower())
    return out


DOMAIN_FEATURES = [
    "reliability", "is_satire", "is_rated", "https", "official_suffix", "domain_has_digit",
    "domain_has_hyphen", "subdomain_depth", "domain_age_days",
]


def domain_features(url_or_domain: str | None, ratings: dict[str, SourceRating]) -> dict[str, float]:
    reg, suffix, sub, scheme = parse_domain(url_or_domain)
    nan = math.nan
    if reg is None:
        return {k: nan for k in DOMAIN_FEATURES}
    rating = ratings.get(reg)
    name = reg.split(".")[0]
    return {
        "reliability": rating.reliability if rating else nan,  # unknown stays NaN, not 0.5
        "is_satire": float(bool(rating and rating.bias == "satire")),
        "is_rated": float(rating is not None),
        "https": nan if scheme is None else float(scheme == "https"),
        "official_suffix": float(bool(suffix) and any(suffix == s or suffix.endswith("." + s) for s in _OFFICIAL_SUFFIXES)),
        "domain_has_digit": float(any(c.isdigit() for c in name)),
        "domain_has_hyphen": float("-" in name),
        "subdomain_depth": float(len(sub.split(".")) if sub and sub != "www" else 0),
        "domain_age_days": nan,  # hook: WHOIS creation date (python-whois) if added later
    }


# --- Style --------------------------------------------------------------------

STYLE_FEATURES = [
    "n_words", "clickbait_hits", "sensational_ratio", "capital_ratio", "allcaps_word_ratio",
    "exclamations", "questions", "second_person", "starts_with_number",
]


def style_features(headline: str | None) -> dict[str, float]:
    if not headline or not headline.strip():
        return {k: math.nan for k in STYLE_FEATURES}
    words = _WORD_RE.findall(headline)
    letters = [c for c in headline if c.isalpha()]
    n = max(len(words), 1)
    return {
        "n_words": float(len(words)),
        "clickbait_hits": float(len(_CLICKBAIT_RE.findall(headline))),
        "sensational_ratio": sum(w.lower() in _SENSATIONAL for w in words) / n,
        "capital_ratio": sum(c.isupper() for c in letters) / max(len(letters), 1),
        "allcaps_word_ratio": sum(len(w) > 1 and w.isupper() for w in words) / n,
        "exclamations": float(headline.count("!")),
        "questions": float(headline.count("?")),
        "second_person": float(sum(w.lower() in {"you", "your", "you're", "yours"} for w in words)),
        "starts_with_number": float(bool(re.match(r"^\s*\d", headline))),
    }


# --- Social (optional) ----------------------------------------------------------

SOCIAL_FEATURES = ["n_tweets", "cascade_size", "cascade_depth", "cascade_breadth", "hours_to_100_shares"]


def social_features(
    tweet_ids: str | Sequence[str] | None = None,
    edges: Sequence[tuple[str, str]] | None = None,
    root: str | None = None,
    timestamps: dict[str, datetime] | None = None,
) -> dict[str, float]:
    """All NaN when nothing is supplied (the normal case without the paid X API).

    tweet_ids: FakeNewsNet's tab-separated id list (released) -> n_tweets.
    edges: (parent, child) retweet pairs from `root` -> cascade size/depth/breadth.
    timestamps: id -> time; hours until the 100th share (fast spread is a
        known marker of fake stories).
    """
    out = {k: math.nan for k in SOCIAL_FEATURES}
    if tweet_ids is not None and not (isinstance(tweet_ids, float) and math.isnan(tweet_ids)):
        ids = tweet_ids.split() if isinstance(tweet_ids, str) else list(tweet_ids)
        out["n_tweets"] = float(len(ids))
    if edges:
        import networkx as nx

        g = nx.DiGraph(list(edges))
        r = root if root in g else next((n for n in g if g.in_degree(n) == 0), None)
        if r is not None:
            depth = nx.single_source_shortest_path_length(g, r)
            levels = np.bincount(list(depth.values()))
            out.update(cascade_size=float(g.number_of_nodes()), cascade_depth=float(max(depth.values())),
                       cascade_breadth=float(levels.max()))
    if timestamps and len(timestamps) >= 100:
        times = sorted(timestamps.values())
        out["hours_to_100_shares"] = (times[99] - times[0]).total_seconds() / 3600
    return out


FEATURE_NAMES = DOMAIN_FEATURES + STYLE_FEATURES + SOCIAL_FEATURES


def item_features(url_or_domain, headline, ratings, tweet_ids=None, edges=None, root=None, timestamps=None) -> list[float]:
    f = {**domain_features(url_or_domain, ratings), **style_features(headline), **social_features(tweet_ids, edges, root, timestamps)}
    return [f[k] for k in FEATURE_NAMES]


# --- Model ----------------------------------------------------------------------

def train_source_classifier(X: np.ndarray, y: np.ndarray, X_val=None, y_val=None, seed: int = 42):
    """LightGBM (handles NaN natively), class-balanced. Returns (model, val macro-F1 or NaN)."""
    from lightgbm import LGBMClassifier
    from sklearn.metrics import f1_score

    model = LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=20,
        class_weight="balanced", random_state=seed, verbose=-1,
    )
    model.fit(X, y)
    f1 = math.nan
    if X_val is not None and len(X_val) and len(set(y_val)) > 1:
        f1 = float(f1_score(y_val, model.predict(X_val), average="macro"))
    return model, f1


@dataclass
class SourceResult:
    score: float
    confidence: float
    features: dict[str, float]
    reasons: list[str]


# Platforms and link shorteners host anyone's posts: the domain says nothing about who wrote the item, so
# there is no source to rate (a tweet is not "from twitter.com" the way an article is from thehindu.com).
PLATFORM_DOMAINS = frozenset({
    "twitter.com", "x.com", "t.co", "facebook.com", "fb.com", "fb.watch", "instagram.com", "threads.net",
    "youtube.com", "youtu.be", "whatsapp.com", "wa.me", "t.me", "telegram.org", "telegram.me", "reddit.com",
    "redd.it", "imgur.com", "tiktok.com", "snapchat.com", "linkedin.com", "pinterest.com", "tumblr.com",
    "quora.com", "medium.com", "substack.com", "blogspot.com", "blogger.com", "wordpress.com", "wix.com",
    "weebly.com", "sharechat.com", "kooapp.com", "mojapp.in", "dailyhunt.in", "news.google.com",
    "bit.ly", "tinyurl.com", "goo.gl", "ow.ly", "buff.ly", "rebrand.ly", "cutt.ly", "shorturl.at", "is.gd",
    "lnkd.in", "dlvr.it", "archive.org", "archive.ph", "archive.today", "google.com", "drive.google.com",
    "docs.google.com", "dropbox.com", "scribd.com", "change.org",
})

# "rating" (default since 2026-11-05): the score of a rated outlet is its rating from public lists; an
# unrated domain or a platform gets no source score (not counted). "classifier": the LightGBM trained on
# FakeNewsNet (domain + headline style), kept only for the comparison in `evaluate source-compare`: it read
# FakeNewsNet's headline typography (capital_ratio was its top feature), so a clickbait headline *raised*
# the score (thehindu.com 0.62 -> 0.94, infowars.com 0.49 -> 0.66) and the rating barely mattered.
SOURCE_MODE = "rating"


def is_platform(reg: str | None) -> bool:
    return bool(reg) and (reg in PLATFORM_DOMAINS or any(reg.endswith("." + p) for p in PLATFORM_DOMAINS))


class SourceChecker:
    def __init__(self, model=None, ratings: dict[str, SourceRating] | None = None, model_path: str | Path = MODEL_PATH,
                 mode: str | None = None):
        self.mode = mode or SOURCE_MODE
        if model is None and self.mode == "classifier" and Path(model_path).exists():
            import joblib

            model = joblib.load(model_path)
        self.model = model
        self.ratings = load_ratings() if ratings is None else ratings

    def score(self, url_or_domain=None, headline=None, **social) -> SourceResult:
        values = item_features(url_or_domain, headline, self.ratings, **social)
        feats = dict(zip(FEATURE_NAMES, values))
        reg = parse_domain(url_or_domain)[0]
        if reg is None and not (headline and headline.strip()):
            return SourceResult(math.nan, 0.0, feats, [])
        if self.mode != "classifier":
            return self._rating_score(reg, feats)
        if self.model is not None:
            p = float(self.model.predict_proba(np.array([values]))[0, 1])
        else:  # rule fallback until trained: rating if known, nudged by style
            p = feats["reliability"] if not math.isnan(feats["reliability"]) else 0.5
            p -= 0.1 * min(feats["clickbait_hits"], 2) if not math.isnan(feats["clickbait_hits"]) else 0.0
            p = min(max(p, 0.0), 1.0)
        reasons = []
        if feats["is_satire"] == 1.0:
            reasons.append("Source is a known satire site.")
        if not math.isnan(feats["reliability"]):
            reasons.append(f"Source reliability rating {feats['reliability']:.2f}.")
        elif parse_domain(url_or_domain)[0]:
            reasons.append("Source domain is not in the ratings list.")
        if feats.get("clickbait_hits", 0) and feats["clickbait_hits"] > 0:
            reasons.append("Headline uses clickbait phrasing.")
        return SourceResult(p, abs(2 * p - 1), feats, reasons)

    def _rating_score(self, reg: str | None, feats: dict) -> SourceResult:
        """Rated outlet -> its rating; platform or unrated domain -> no score (the check does not count)."""
        if reg is None:
            return SourceResult(math.nan, 0.0, feats, [])
        if is_platform(reg):
            return SourceResult(math.nan, 0.0, feats, [
                f"{reg} is a platform or link shortener, not the author: the source is not rated."])
        rating = self.ratings.get(reg)
        if rating is None:
            return SourceResult(math.nan, 0.0, feats, [f"{reg} is not in the ratings list: the source is not rated."])
        p = rating.reliability
        reasons = [f"Source reliability rating {p:.2f} ({reg})."]
        if rating.bias == "satire":
            reasons.insert(0, "Source is a known satire site.")
        return SourceResult(p, abs(2 * p - 1), feats, reasons)


def main() -> None:
    """python -m src.source.credibility [--social] : features for FakeNewsNet, train LightGBM, save, report.

    Social features are left out by default. The API never receives tweet ids, and a
    LightGBM model that never saw a missing tweet count reads a missing one as 0
    ("nobody shared this"), which skewed every live score. --social keeps them (for
    the comparison in the report only)."""
    import argparse
    import json

    import joblib
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--social", action="store_true", help="also train on tweet counts (not available in the app)")
    args = ap.parse_args()

    df = pd.read_parquet(ROOT / "data" / "processed" / "fakenewsnet.parquet")
    df = df[df["credible"].notna()]
    ratings = load_ratings()
    tweets = df["tweet_ids"] if args.social else [None] * len(df)
    X = np.array([item_features(d, t, ratings, tweet_ids=tw) for d, t, tw in zip(df["source_domain"], df["text"], tweets)])
    y = (df["credible"].to_numpy() >= 0.5).astype(int)
    split = df["split"].to_numpy()
    model, val_f1 = train_source_classifier(X[split == "train"], y[split == "train"], X[split == "val"], y[split == "val"])
    joblib.dump(model, MODEL_PATH)
    from sklearn.metrics import f1_score

    test = split == "test"
    report = {
        "rows": int(len(y)),
        "rated_share": float(np.mean(~np.isnan(X[:, FEATURE_NAMES.index("reliability")]))),
        "social_features": bool(args.social),
        "val_macro_f1": val_f1,
        "test_macro_f1": float(f1_score(y[test], model.predict(X[test]), average="macro")) if test.any() else None,
        "feature_importance": dict(zip(FEATURE_NAMES, map(int, model.feature_importances_))),
    }
    (ROOT / "reports" / "source_metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
