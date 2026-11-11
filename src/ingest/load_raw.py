"""Read each raw dataset in `data/raw/` into one common column layout.

Every loader returns a DataFrame with at least:
    id, text, label, date_raw, has_image, has_video, source_domain, event, orig_split
Dataset-specific columns (AVeriTeC evidence, Fakeddit image URL, ...) are kept
as extra columns for that dataset's channel model.

Loaders read the full releases (nested folders under data/raw/) and fall back to
the Phase 1 sample files only when the full files are absent.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd


def domain_of(url: object) -> str | None:
    """'https://www.bbc.co.uk/news/x' -> 'bbc.co.uk'."""
    if not isinstance(url, str) or not url.strip():
        return None
    netloc = urlparse(url if "://" in url else f"http://{url}").netloc.lower()
    netloc = netloc.split("@")[-1].split(":")[0]
    return netloc[4:] if netloc.startswith("www.") else (netloc or None)


def _first_col(df: pd.DataFrame, *names: str):
    for n in names:
        if n in df.columns:
            return df[n]
    return pd.Series([None] * len(df), index=df.index, dtype=object)


# --- AVeriTeC -----------------------------------------------------------------

def load_averitec(raw_dir: Path) -> pd.DataFrame:
    rows = []
    for split_file in ("train", "dev"):
        path = raw_dir / "averitec" / f"{split_file}.json"
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            items = json.load(f)
        for i, it in enumerate(items):
            rows.append(
                {
                    "id": f"averitec_{split_file}_{i}",
                    "text": it.get("claim"),
                    "label": it.get("label"),
                    "date_raw": it.get("claim_date"),
                    "has_image": False,
                    "has_video": False,
                    "source_domain": domain_of(it.get("original_claim_url")),
                    "event": None,
                    "orig_split": split_file,
                    # extras for the verifier channel (3B/3C)
                    "justification": it.get("justification"),
                    "speaker": it.get("speaker"),
                    "reporting_source": it.get("reporting_source"),
                    "location_iso": it.get("location_ISO_code"),
                    "claim_types": "|".join(it.get("claim_types") or []),
                    "fact_checking_article": it.get("fact_checking_article"),
                    "questions_json": json.dumps(it.get("questions") or [], ensure_ascii=False),
                }
            )
    return pd.DataFrame(rows)


# --- Fakeddit -----------------------------------------------------------------

FAKEDDIT_MAX_ROWS = 60_000  # Phase 1: balanced 50k-100k sample
FAKEDDIT_SEED = 42


def _fakeddit_files(fakeddit_dir: Path) -> list[Path]:
    """Real files: data/raw/fakeddit/**/multimodal_{train,validate,test_public}.tsv
    (any *.tsv outside the Phase 1 sample). Falls back to the sample TSV."""
    full = [p for p in sorted(fakeddit_dir.rglob("*.tsv")) if "sample" not in p.name.lower()]
    return full or sorted(fakeddit_dir.glob("*.tsv"))


def _balanced_sample(df: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    """Half true / half fake (the unified target); fake keeps its natural sub-type mix.

    Seeded, so reruns give the same rows. It also keeps the near-duplicate step
    tractable, because that step compares rows pairwise.
    """
    if len(df) <= max_rows:
        return df
    if "2_way_label" not in df.columns:
        return df.sample(n=max_rows, random_state=seed)
    half = max_rows // 2
    parts = [g.sample(n=min(len(g), half), random_state=seed) for _, g in df.groupby("2_way_label")]
    return pd.concat(parts).sort_index()


def load_fakeddit(raw_dir: Path, max_rows: int = FAKEDDIT_MAX_ROWS) -> pd.DataFrame:
    frames = []
    for path in _fakeddit_files(raw_dir / "fakeddit"):
        df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_values=[""], on_bad_lines="skip")
        m = re.search(r"(train|validate|val|test)", path.stem)
        df["orig_split"] = {"validate": "val"}.get(m.group(1), m.group(1)) if m else None
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    label_col = "6_way_label" if "6_way_label" in df.columns else "label"
    df = df[df[label_col].notna()]
    df = _balanced_sample(df, max_rows, FAKEDDIT_SEED).reset_index(drop=True)
    image_url = _first_col(df, "image_url")
    has_image = _first_col(df, "hasImage")
    has_image = has_image.map(lambda v: str(v).lower() in {"true", "1"}) if has_image.notna().any() else image_url.notna()
    return pd.DataFrame(
        {
            "id": "fakeddit_" + _first_col(df, "id").astype(str),
            "text": _first_col(df, "title", "clean_title"),
            "label": df[label_col],
            "date_raw": _first_col(df, "created_utc"),
            "has_image": has_image.astype(bool),
            "has_video": False,
            "source_domain": _first_col(df, "domain").map(domain_of),
            "event": None,
            "orig_split": df["orig_split"],
            "image_url": image_url,
            "subreddit": _first_col(df, "subreddit"),
        }
    )


# --- FakeNewsNet --------------------------------------------------------------

_FNN_FILE_RE = re.compile(r"(politifact|gossipcop)_(fake|real)", re.IGNORECASE)


def load_fakenewsnet(raw_dir: Path) -> pd.DataFrame:
    """Real files: data/raw/fakenewsnet/dataset/{politifact,gossipcop}_{fake,real}.csv.

    Columns: id, news_url, title, tweet_ids. The label is in the file name. The
    release has no article body or publish date (the scraper in `code/` fetches
    them), so `text` is the headline. Falls back to the Phase 1 sample CSV.
    """
    fnn_dir = raw_dir / "fakenewsnet"
    csvs = [p for p in sorted(fnn_dir.rglob("*.csv")) if _FNN_FILE_RE.search(p.stem)]
    csvs = csvs or sorted(fnn_dir.glob("*.csv"))
    frames = []
    for path in csvs:
        df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
        m = _FNN_FILE_RE.search(path.stem)
        if m:
            df["source"] = m.group(1).lower()
            df["real_fake"] = m.group(2).lower()
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    title = _first_col(df, "title").fillna("")
    body = _first_col(df, "text", "content").fillna("")
    text = title.where(body == "", title + ". " + body)
    return pd.DataFrame(
        {
            "id": "fakenewsnet_" + _first_col(df, "news_id", "id").astype(str),
            "text": text,
            "label": _first_col(df, "real_fake", "label"),
            "date_raw": _first_col(df, "publish_date", "date"),
            "has_image": _first_col(df, "top_img").notna(),
            "has_video": False,
            "source_domain": _first_col(df, "news_url", "url").map(domain_of),
            "event": None,
            "orig_split": None,
            "fnn_source": _first_col(df, "source"),
            "tweet_ids": _first_col(df, "tweet_ids"),
        }
    )


# --- MediaEval ----------------------------------------------------------------

_VIDEO_RE = re.compile(r"\.(mp4|avi|mov|mkv|webm)$", re.IGNORECASE)

# (path under data/raw/mediaeval, split role). Posts without a label (e.g. test
# files released without ground truth) are dropped.
_MEDIAEVAL_FILES = [
    ("mediaeval2015/devset/tweets.txt", "dev"),
    ("mediaeval2015/testset/tweets.txt", "test"),
    ("mediaeval2016/devset/posts.txt", "dev"),
    ("mediaeval2016/testset/posts_groundtruth.txt", "test"),
]


def _event_from_media_id(media_id: object) -> str:
    """'sandyA_fake_35,sandyA_fake_36' -> 'sandy'.

    Released image ids start with the event name plus an optional capital letter
    for the sub-event (sandyA, sandyB). The event is the group a split must keep
    together.
    """
    if not isinstance(media_id, str) or not media_id.strip():
        return "unknown"
    head = media_id.split(",")[0].strip().split("_")[0]
    return re.sub(r"[A-Z]$", "", head).lower() or "unknown"


def _read_mediaeval(path: Path) -> pd.DataFrame:
    # Tab-separated; tweet text contains quote marks, so quote handling is off (3).
    return pd.read_csv(
        path, sep="\t", dtype=str, keep_default_na=False, na_values=[""],
        quoting=3, on_bad_lines="skip", encoding="utf-8", encoding_errors="replace",
    )


def load_mediaeval(raw_dir: Path) -> pd.DataFrame:
    """Real files: mediaeval2015/{devset,testset}/tweets.txt and
    mediaeval2016/{devset/posts.txt, testset/posts_groundtruth.txt}.

    Columns: tweetId|post_id, tweetText|post_text, imageId(s)|image_id(s),
    timestamp, label. Falls back to the Phase 1 sample CSV.
    """
    me_dir = raw_dir / "mediaeval"
    frames = []
    for rel, role in _MEDIAEVAL_FILES:
        path = me_dir / rel
        if path.exists():
            df = _read_mediaeval(path)
            df["orig_split"] = role
            df["year"] = "2015" if "2015" in rel else "2016"
            frames.append(df)
    if not frames:
        for path in sorted(me_dir.glob("*sample*.csv")):
            df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
            df["orig_split"] = None
            df["year"] = ""
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = df[df["label"].notna()].reset_index(drop=True)
    media = _first_col(df, "imageId(s)", "image_id(s)", "image_name", "imageId").fillna("")
    is_video = media.map(lambda m: any(_VIDEO_RE.search(part.strip()) for part in m.split(",")))
    event = _first_col(df, "event_name", "event")
    if event.isna().all():
        event = media.map(_event_from_media_id)
    year = df["year"].fillna("")
    return pd.DataFrame(
        {
            "id": "mediaeval_" + year + "_" + _first_col(df, "tweetId", "post_id", "tweet_id").astype(str),
            "text": _first_col(df, "tweetText", "post_text", "text").fillna(""),
            "label": df["label"],
            "date_raw": _first_col(df, "timestamp"),
            "has_image": (media != "") & ~is_video,
            "has_video": is_video,
            "source_domain": None,
            "event": event,
            "orig_split": df["orig_split"],
            "media_id": media,
        }
    )


LOADERS = {
    "averitec": load_averitec,
    "fakeddit": load_fakeddit,
    "fakenewsnet": load_fakenewsnet,
    "mediaeval": load_mediaeval,
}
