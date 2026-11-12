"""Channel 3E: media reuse and authenticity (Phase 3E).

Out-of-context reuse is the commonest image misinformation: a real photo from
an old event is posted as new ("2015 Chennai flood shared as Mumbai today").
The pixels are genuine, so a fake-image detector misses it; finding an earlier
copy catches it.

    image -> pHash (64-bit fingerprint) + CLIP vector
          -> near-duplicates in an index of known media with first-seen date + event
          -> signals: match count, earliest copy, date gap vs claimed date,
             different-event flag, location similarity, EXIF date/GPS, ELA tamper score
          -> classifier (trained on MediaEval) -> score = P(real); NaN if no media

Two matchers because they fail differently:
- pHash survives resizing, recompression and colour tweaks, but not crops or
  overlays. Hamming distance <= 10 of 64 bits = same picture.
- CLIP cosine >= 0.92 survives crops, borders and captions burned into the image,
  but can also match two different photos of the same scene.

Video is out of scope (future work): Indian news videos are mostly talk,
music and overlays, and choosing which frames and words to check is not reliable yet.

Location runs in the background: the claimed place (or the first place named in the text that has
reference photos) is compared with photos in data/reference/landmarks/; only a mismatch is reported.
"""

from __future__ import annotations

import io
import math
import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LANDMARK_DIR = ROOT / "data" / "reference" / "landmarks"  # landmarks/<place_slug>/*.jpg
MEDIAEVAL_IMAGE_DIR = ROOT / "data" / "raw" / "mediaeval" / "images"  # filled by src.ingest.download_mediaeval
INDEX_DIR = ROOT / "models" / "media_index"
MODEL_PATH = ROOT / "models" / "media_reuse_clf.joblib"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}

PHASH_MAX_HAMMING = 10
CLIP_MIN_COSINE = 0.92
DATE_TOLERANCE_DAYS = 2  # a copy 1-2 days earlier is normal spread, not reuse
# Below this CLIP cosine the photo "does not resemble" the place. Calibrated on Wikimedia Commons
# landmark photos (`python -m src.ingest.download_landmarks --calibrate`, reports/location_metrics.json);
# 0.25 was the placeholder before reference photos existed. Calibrated 2026-11-05: 0.49 (5th percentile of
# same-place similarity; on held-out places 6.5% of true photos wrongly flagged, 19% of wrong places caught).
LOCATION_MIN_SIMILARITY = float(os.environ.get("NEWSCRED_LOCATION_MIN", "0.49"))

ImageEncoder = Callable[[Sequence], np.ndarray]  # PIL images -> (n, d) L2-normalised


# --- Fingerprints -------------------------------------------------------------

def phash(image) -> int:
    """64-bit perceptual hash as int (DCT of a 32x32 greyscale thumbnail)."""
    import imagehash

    return int(str(imagehash.phash(image)), 16)


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


# --- Forensics and metadata ---------------------------------------------------

def ela_score(image, quality: int = 90) -> float:
    """Error Level Analysis: re-save as JPEG, mean absolute pixel change (0-255).

    A region pasted in from another photo has a different compression history,
    so it changes more on re-save. One number per image is a weak hint, never
    proof: it also rises with heavy edits, screenshots and text overlays.
    """
    from PIL import Image, ImageChops

    rgb = image.convert("RGB")
    buf = io.BytesIO()
    rgb.save(buf, "JPEG", quality=quality)
    resaved = Image.open(io.BytesIO(buf.getvalue()))
    diff = np.asarray(ImageChops.difference(rgb, resaved), dtype=np.float32)
    return float(diff.mean())


def _gps_to_deg(values, ref) -> float | None:
    try:
        d, m, s = (float(x) for x in values)
    except (TypeError, ValueError):
        return None
    deg = d + m / 60 + s / 3600
    return -deg if ref in ("S", "W") else deg


def read_exif(image) -> dict:
    """{'capture_date': ISO or None, 'gps': (lat, lon) or None}. Social platforms
    strip EXIF, so None is the normal case and is not suspicious by itself."""
    out = {"capture_date": None, "gps": None}
    try:
        exif = image.getexif()
    except Exception:
        return out
    raw = exif.get(36867) or exif.get(306)  # DateTimeOriginal, else DateTime
    if not raw:
        raw = exif.get_ifd(0x8769).get(36867) if hasattr(exif, "get_ifd") else None
    if isinstance(raw, str) and len(raw) >= 10:
        out["capture_date"] = raw[:10].replace(":", "-")
    gps = exif.get_ifd(0x8825) if hasattr(exif, "get_ifd") else {}
    if gps and 2 in gps and 4 in gps:
        lat, lon = _gps_to_deg(gps[2], gps.get(1)), _gps_to_deg(gps[4], gps.get(3))
        if lat is not None and lon is not None:
            out["gps"] = (lat, lon)
    return out


# --- Index of known media -----------------------------------------------------

@dataclass
class MediaRecord:
    media_id: str
    phash: int
    first_seen: str | None  # ISO date of earliest known post
    event: str | None
    context: str | None = None  # caption / tweet text of the earliest post


@dataclass
class Match:
    record: MediaRecord
    hamming: int
    clip_cosine: float


class MediaIndex:
    """pHash + CLIP index. CLIP vectors are optional (pHash-only index works)."""

    def __init__(self, records: Sequence[MediaRecord], clip_vectors: np.ndarray | None = None):
        self.records = list(records)
        self.hashes = np.array([r.phash for r in self.records], dtype=np.uint64)
        self.clip = None if clip_vectors is None else np.asarray(clip_vectors, dtype=np.float32)

    def query(
        self, image_hash: int, clip_vector: np.ndarray | None = None, exclude_ids: set[str] | None = None
    ) -> list[Match]:
        """Near-duplicates by pHash OR CLIP. `exclude_ids` drops the query's own
        record when evaluating on indexed data (otherwise every image matches itself)."""
        if not self.records:
            return []
        x = np.bitwise_xor(self.hashes, np.uint64(image_hash))
        # popcount of each 64-bit XOR, vectorised (a Python loop per record was the slow part)
        ham = np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1).astype(int)
        cos = (
            self.clip @ np.asarray(clip_vector, dtype=np.float32)
            if self.clip is not None and clip_vector is not None
            else np.full(len(self.records), -1.0)
        )
        hit = (ham <= PHASH_MAX_HAMMING) | (cos >= CLIP_MIN_COSINE)
        out = [
            Match(self.records[i], int(ham[i]), float(cos[i]))
            for i in np.flatnonzero(hit)
            if not exclude_ids or self.records[i].media_id not in exclude_ids
        ]
        return sorted(out, key=lambda m: (m.hamming, -m.clip_cosine))


def image_files(image_dir: Path = MEDIAEVAL_IMAGE_DIR) -> dict[str, Path]:
    """media_id -> file in data/raw/mediaeval/images/ (file stem = image id)."""
    if not image_dir.exists():
        return {}
    return {p.stem: p for p in sorted(image_dir.iterdir()) if p.suffix.lower() in IMAGE_EXTS}


def build_mediaeval_index(
    df, encoder: ImageEncoder | None = None, image_dir: Path = MEDIAEVAL_IMAGE_DIR, batch_size: int = 32
) -> tuple[MediaIndex, list[str]]:
    """Index every downloaded MediaEval image that the processed tweet table refers to.

    Returns (index, missing_ids): ids referenced by tweets but not on disk (dead URLs).
    GIFs use their first frame.
    """
    from PIL import Image

    files = image_files(image_dir)
    meta = records_from_mediaeval(df)
    present = [m for m in meta if m[0] in files]
    missing = [m[0] for m in meta if m[0] not in files]
    records, vectors = [], []
    for start in range(0, len(present), batch_size):
        chunk = present[start : start + batch_size]
        images = []
        for media_id, first_seen, event, context in chunk:
            try:
                with Image.open(files[media_id]) as im:
                    img = im.convert("RGB")
            except OSError:
                missing.append(media_id)
                continue
            images.append(img)
            records.append(MediaRecord(media_id, phash(img), first_seen, event, context))
        if encoder is not None and images:
            vectors.append(encoder(images))
    clip = np.concatenate(vectors) if vectors else None
    return MediaIndex(records, clip), missing


def save_index(index: MediaIndex, directory: Path = INDEX_DIR) -> None:
    import pandas as pd

    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [(r.media_id, str(r.phash), r.first_seen, r.event, r.context) for r in index.records],
        columns=["media_id", "phash", "first_seen", "event", "context"],
    ).to_parquet(directory / "records.parquet", index=False)
    if index.clip is not None:
        np.save(directory / "clip.npy", index.clip)


def load_index(directory: Path = INDEX_DIR) -> MediaIndex:
    import pandas as pd

    t = pd.read_parquet(directory / "records.parquet")
    t = t.astype(object).where(t.notna(), None)
    records = [MediaRecord(m, int(h), f, e, c) for m, h, f, e, c in t.itertuples(index=False, name=None)]
    clip_path = directory / "clip.npy"
    return MediaIndex(records, np.load(clip_path) if clip_path.exists() else None)


def records_from_mediaeval(df) -> list[tuple[str, str | None, str | None, str | None]]:
    """(media_id, first_seen, event, context) per distinct image in the processed
    MediaEval table: first_seen = earliest tweet date that posted it. A tweet may
    list several ids ('a,b'); each id is its own record."""
    import pandas as pd

    rows = []
    for mids, d, ev, txt in zip(df["media_id"], df["date"], df["topic_group"], df["text"]):
        for m in str(mids or "").split(","):
            if m.strip():
                rows.append((m.strip(), d, ev, txt))
    t = pd.DataFrame(rows, columns=["media_id", "date", "event", "text"]).sort_values("date", na_position="last")
    first = t.groupby("media_id", sort=False).first().reset_index()
    first = first.astype(object).where(first.notna(), None)  # NaN is truthy; temporal logic needs None
    return list(first[["media_id", "date", "event", "text"]].itertuples(index=False, name=None))


# --- Signals ------------------------------------------------------------------

def _days_between(earlier: str | None, later: str | date | None) -> float:
    if not earlier or not later:
        return math.nan
    later = later if isinstance(later, date) else date.fromisoformat(str(later)[:10])
    return float((later - date.fromisoformat(earlier[:10])).days)


@dataclass
class ReuseSignals:
    n_matches: int
    min_hamming: float  # NaN if no match
    max_clip_cosine: float
    earliest_match: str | None
    date_gap_days: float  # claimed date - earliest copy; >0 = image is older than claimed
    predates_claim: bool  # earliest copy more than DATE_TOLERANCE_DAYS before the claim
    different_event: bool  # earliest copy belongs to another event
    location_similarity: float  # NaN if no claimed place or no reference images
    location_mismatch: bool
    ela: float
    exif_date_gap_days: float  # claimed date - EXIF capture date; NaN if no EXIF date
    reasons: list[str] = field(default_factory=list)

    def features(self) -> list[float]:
        """Fixed-order feature vector for the classifier. NaN stays NaN."""
        return [
            float(self.n_matches), self.min_hamming, self.max_clip_cosine, self.date_gap_days,
            float(self.predates_claim), float(self.different_event), self.location_similarity,
            self.ela, self.exif_date_gap_days,
        ]


FEATURE_NAMES = [
    "n_matches", "min_hamming", "max_clip_cosine", "date_gap_days", "predates_claim",
    "different_event", "location_similarity", "ela", "exif_date_gap_days",
]


def temporal_signals(matches: Sequence[Match], claimed_date: str | date | None, claimed_event: str | None):
    """(earliest, gap_days, predates, different_event) from index matches.
    No claimed date -> gap NaN and predates False: the check does not apply."""
    dated = [m for m in matches if m.record.first_seen]
    if not dated:
        return None, math.nan, False, False
    earliest = min(dated, key=lambda m: m.record.first_seen)
    gap = _days_between(earliest.record.first_seen, claimed_date)
    predates = not math.isnan(gap) and gap > DATE_TOLERANCE_DAYS
    different = bool(claimed_event and earliest.record.event and earliest.record.event != claimed_event)
    return earliest.record.first_seen, gap, predates, different


class LocationReference:
    """CLIP vectors of reference photos per place (data/reference/landmarks/<place>/).

    similarity = best cosine between the query image and any reference photo of the
    claimed place. Max, not mean: one matching viewpoint is enough.
    """

    def __init__(self, encoder: ImageEncoder, root: Path = LANDMARK_DIR, refs: dict[str, np.ndarray] | None = None):
        self.encoder = encoder
        self.refs: dict[str, np.ndarray] = dict(refs or {})
        if refs is None and root.exists():
            from PIL import Image

            for place_dir in sorted(p for p in root.iterdir() if p.is_dir()):
                files = sorted(f for f in place_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
                if files:
                    self.refs[place_dir.name] = encoder([Image.open(f).convert("RGB") for f in files])

    @staticmethod
    def slug(place: str) -> str:
        return "_".join(place.lower().replace(",", " ").split())

    def find(self, places) -> str | None:
        """First place with reference photos among names found in the text ("Jantar Mantar",
        "India Gate, Delhi" -> "india_gate"). None when no named place has references."""
        for place in places or []:
            slug = self.slug(place)
            if slug in self.refs:
                return place
            for known in self.refs:  # "India Gate, New Delhi" contains the known place "india_gate"
                if f"_{known}_" in f"_{slug}_":
                    return known.replace("_", " ")
        return None

    def similarity(self, image_vector: np.ndarray, place: str | None) -> float:
        refs = self.refs.get(self.slug(place)) if place else None
        if refs is None or len(refs) == 0:
            return math.nan
        return float(np.max(refs @ np.asarray(image_vector, dtype=np.float32)))


# --- Checker ------------------------------------------------------------------

@dataclass
class ReuseResult:
    score: float  # P(authentic, used in context); NaN without media
    confidence: float
    signals: ReuseSignals | None
    matches: list[Match]


class MediaReuseChecker:
    def __init__(
        self,
        index: MediaIndex,
        encoder: ImageEncoder | None = None,
        locations: LocationReference | None = None,
        model=None,
    ):
        self.index = index
        self.encoder = encoder
        self.locations = locations
        self.model = model

    def signals(
        self,
        image,
        claimed_date: str | date | None = None,
        claimed_event: str | None = None,
        claimed_place: str | None = None,
        exclude_ids: set[str] | None = None,
    ) -> tuple[ReuseSignals, list[Match]]:
        vec = self.encoder([image])[0] if self.encoder is not None else None
        matches = self.index.query(phash(image), vec, exclude_ids)
        earliest, gap, predates, different = temporal_signals(matches, claimed_date, claimed_event)
        loc = self.locations.similarity(vec, claimed_place) if self.locations is not None and vec is not None else math.nan
        exif = read_exif(image)
        exif_gap = _days_between(exif["capture_date"], claimed_date)
        reasons = []
        if predates:
            reasons.append(f"Earliest copy found {int(gap)} days before the claimed date ({earliest}).")
        if different:
            reasons.append("Earliest copy belongs to a different event.")
        loc_mismatch = not math.isnan(loc) and loc < LOCATION_MIN_SIMILARITY
        if loc_mismatch:
            reasons.append(f"Image does not resemble reference photos of {claimed_place}.")
        if not math.isnan(exif_gap) and exif_gap > DATE_TOLERANCE_DAYS:
            reasons.append(f"Camera metadata date is {int(exif_gap)} days before the claimed date.")
        sig = ReuseSignals(
            n_matches=len(matches),
            min_hamming=float(min((m.hamming for m in matches), default=math.nan)),
            max_clip_cosine=float(max((m.clip_cosine for m in matches), default=math.nan)),
            earliest_match=earliest,
            date_gap_days=gap,
            predates_claim=predates,
            different_event=different,
            location_similarity=loc,
            location_mismatch=loc_mismatch,
            ela=ela_score(image),
            exif_date_gap_days=exif_gap,
            reasons=reasons,
        )
        return sig, matches

    def score(
        self, image, claimed_date=None, claimed_event=None, claimed_place=None, exclude_ids: set[str] | None = None
    ) -> ReuseResult:
        if image is None:
            return ReuseResult(math.nan, 0.0, None, [])
        sig, matches = self.signals(image, claimed_date, claimed_event, claimed_place, exclude_ids)
        if self.model is not None:
            p = float(self.model.predict_proba(np.array([sig.features()]))[0, 1])
        else:  # rule fallback until the classifier is trained
            p = 0.5
            p -= 0.35 if sig.predates_claim else 0.0
            p -= 0.15 if sig.different_event else 0.0
            p -= 0.2 if sig.location_mismatch else 0.0
            p += 0.1 if sig.n_matches == 0 else 0.0
            p = min(max(p, 0.0), 1.0)
        return ReuseResult(p, abs(p - 0.5) * 2, sig, matches)


class DropUninformativeColumns:
    """Drop columns that are all-NaN or constant, learned on the training fold only.

    HistGradientBoosting crashes on an all-NaN column ("window shape cannot be
    larger than input array shape": its binner gets zero distinct values). With
    MediaEval this happens for `location_similarity` (no landmark photos yet) and
    `exif_date_gap_days` (Twitter strips EXIF). Constant columns carry no signal
    either, so they go too. Living inside the pipeline means each CV fold and the
    saved model drop the same columns at fit and at predict time.
    """

    def __init__(self, feature_names: Sequence[str] | None = None):
        self.feature_names = feature_names

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        keep = []
        for j in range(X.shape[1]):
            col = X[:, j][~np.isnan(X[:, j])]
            keep.append(col.size > 0 and np.unique(col).size > 1)
        self.keep_ = np.array(keep, dtype=bool)
        if not self.keep_.any():
            raise ValueError("every feature is all-NaN or constant; nothing to train on")
        names = list(self.feature_names) if self.feature_names is not None else [f"x{j}" for j in range(X.shape[1])]
        self.dropped_ = [n for n, k in zip(names, self.keep_) if not k]
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)[:, self.keep_]

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def get_params(self, deep=True):
        return {"feature_names": self.feature_names}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self


def train_reuse_classifier(X: np.ndarray, y: np.ndarray, groups: Sequence[str], seed: int = 42):
    """Drop uninformative columns, then HistGradientBoosting on ReuseSignals
    features (handles partly-missing columns natively, like LightGBM).
    y = 1 real, 0 fake. Grouped CV by event, because MediaEval images repeat
    within an event. Returns (fitted pipeline, mean CV macro-F1).

    The fitted pipeline's `named_steps["drop"].dropped_` lists the dropped features.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold, cross_val_score
    from sklearn.pipeline import Pipeline

    model = Pipeline([
        ("drop", DropUninformativeColumns(FEATURE_NAMES if np.shape(X)[1] == len(FEATURE_NAMES) else None)),
        ("hgb", HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, class_weight="balanced", random_state=seed)),
    ])
    n_groups = len(set(groups))
    cv = float("nan")
    if n_groups >= 3:
        cv = float(np.nanmean(cross_val_score(
            model, X, y, groups=groups, cv=GroupKFold(n_splits=min(5, n_groups)), scoring="f1_macro", error_score="raise"
        )))
    return model.fit(X, y), cv


def build_training_table(df, index: MediaIndex, encoder: ImageEncoder | None = None, image_dir: Path = MEDIAEVAL_IMAGE_DIR):
    """One feature row per tweet whose first image is on disk.

    Claimed date = tweet date, claimed event = tweet's event group, and the
    tweet's own image id is excluded from matching (otherwise every image
    matches itself). Returns (X, y, groups, split).
    """
    from PIL import Image

    files = image_files(image_dir)
    checker = MediaReuseChecker(index, encoder=encoder)
    X, y, groups, splits = [], [], [], []
    for media, d, ev, cred, split in zip(df["media_id"], df["date"], df["topic_group"], df["credible"], df["split"]):
        first = str(media or "").split(",")[0].strip()
        if first not in files or cred != cred:  # no image on disk, or NaN label
            continue
        with Image.open(files[first]) as im:
            img = im.convert("RGB")
        sig, _ = checker.signals(img, d, ev, exclude_ids={first})
        X.append(sig.features())
        y.append(int(cred >= 0.5))
        groups.append(ev)
        splits.append(split)
    return np.array(X, dtype=float), np.array(y), groups, splits


def main() -> None:
    """python -m src.media.reuse : build the MediaEval pHash + CLIP index from
    data/raw/mediaeval/images/, train the reuse classifier, save both."""
    import json

    import joblib
    import pandas as pd
    from sklearn.metrics import f1_score

    from src.media.text_image import ClipEncoder

    df = pd.read_parquet(ROOT / "data" / "processed" / "mediaeval.parquet")
    clip = ClipEncoder()
    index, missing = build_mediaeval_index(df, clip.encode_images)
    save_index(index)
    print(f"indexed {len(index.records)} images; {len(missing)} referenced but missing")

    X, y, groups, splits = build_training_table(df, index, clip.encode_images)
    train = np.array([s != "test" for s in splits])
    model, cv = train_reuse_classifier(X[train], y[train], [g for g, t in zip(groups, train) if t])
    joblib.dump(model, MODEL_PATH)
    report = {
        "rows": int(len(y)),
        "train_rows": int(train.sum()),
        "class_counts": {"fake": int((y == 0).sum()), "real": int((y == 1).sum())},
        "events": len(set(groups)),
        "dropped_features": model.named_steps["drop"].dropped_,
        "cv_macro_f1_by_event": cv,
    }
    y_te = y[~train]
    report["test_rows"] = int(y_te.size)
    report["test_class_counts"] = {"fake": int((y_te == 0).sum()), "real": int((y_te == 1).sum())}
    if len(set(y_te)) > 1:
        report["test_macro_f1"] = float(f1_score(y_te, model.predict(X[~train]), average="macro"))
    else:  # macro-F1 is undefined with one class; say so instead of omitting silently
        report["test_macro_f1"] = None
        report["test_note"] = "test split has fewer than 2 classes (event split); macro-F1 undefined"
    (ROOT / "reports").mkdir(parents=True, exist_ok=True)
    (ROOT / "reports" / "media_reuse_metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
