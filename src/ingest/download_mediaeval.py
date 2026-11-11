"""Download MediaEval / image-verification-corpus images from the URL lists.

The cloned repository ships text files (tab-separated) that map image ids to
source URLs, not the images themselves. This script:

1. scans every *.txt file under data/raw/mediaeval/ (skipping images/),
2. finds rows that contain an http(s) URL pointing at an image, and the image id
   on the same row (the first non-URL field that looks like an id, e.g.
   `sandyA_fake_29`),
3. downloads each image once, verifies it opens, drops files < 32 px, and saves
   it unchanged (no resize: pHash and EXIF need the original) as
   data/raw/mediaeval/images/<image_id>.<ext>,
4. writes data/raw/mediaeval/images/_download_log.csv (id, url, status).

Many URLs are from 2012-2016 Twitter/Flickr/news posts and are dead; failures
are logged, not fatal. Columns are detected from content, so the script does
not depend on exact header names.

Run from `news-credibility/`:
    python -m src.ingest.download_mediaeval --dry-run    # counts only
    python -m src.ingest.download_mediaeval              # download
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MEDIAEVAL_DIR = ROOT / "data" / "raw" / "mediaeval"
IMAGE_DIR = MEDIAEVAL_DIR / "images"
LOG_NAME = "_download_log.csv"
MIN_SIDE = 32
_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+$")  # sandyA_fake_29, boston_real_01
_EXT_FROM_TYPE = {"image/jpeg": ".jpg", "image/png": ".png", "image/gif": ".gif", "image/webp": ".webp", "image/bmp": ".bmp"}


def parse_url_rows(line: str) -> tuple[str, str] | None:
    """(image_id, url) from one tab-separated line, or None if the line has no
    URL or no id-like field."""
    fields = [f.strip() for f in line.rstrip("\n").split("\t")]
    url = next((m.group(0) for f in fields for m in [_URL_RE.match(f)] if m), None)
    if url is None:
        return None
    image_id = next((f for f in fields if not _URL_RE.match(f) and _ID_RE.match(f)), None)
    return (image_id, url) if image_id else None


def collect_urls(root: Path = MEDIAEVAL_DIR) -> dict[str, str]:
    """image_id -> URL across all URL list files. First URL seen for an id wins."""
    out: dict[str, str] = {}
    for path in sorted(root.rglob("*.txt")):
        if IMAGE_DIR in path.parents:
            continue
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                row = parse_url_rows(line)
                if row and row[0] not in out:
                    out[row[0]] = row[1]
    return out


def _existing(image_id: str, image_dir: Path) -> Path | None:
    hits = [p for p in image_dir.glob(f"{image_id}.*") if p.name != LOG_NAME]
    return hits[0] if hits else None


def fetch_one(image_id: str, url: str, image_dir: Path = IMAGE_DIR, timeout: float = 15.0) -> str:
    """Download and verify one image. Returns a status string for the log."""
    from PIL import Image, UnidentifiedImageError

    if _existing(image_id, image_dir):
        return "exists"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "news-credibility-research/0.1"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = r.read()
            ctype = (r.headers.get_content_type() or "").lower()
    except Exception as e:  # dead link, timeout, HTTP error
        return f"error:{type(e).__name__}"
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.verify()
        with Image.open(io.BytesIO(data)) as im:
            if min(im.size) < MIN_SIDE:
                return "too_small"
            fmt = (im.format or "").lower()
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
        return "not_an_image"
    ext = _EXT_FROM_TYPE.get(ctype) or {"jpeg": ".jpg", "png": ".png", "gif": ".gif", "webp": ".webp"}.get(fmt, ".jpg")
    (image_dir / f"{image_id}{ext}").write_bytes(data)
    return "ok"


def download_all(urls: dict[str, str], image_dir: Path = IMAGE_DIR, workers: int = 16) -> dict[str, int]:
    image_dir.mkdir(parents=True, exist_ok=True)
    items = sorted(urls.items())
    with ThreadPoolExecutor(max_workers=workers) as pool:
        statuses = list(pool.map(lambda kv: fetch_one(kv[0], kv[1], image_dir), items))
    with open(image_dir / LOG_NAME, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "url", "status"])
        w.writerows((i, u, s) for (i, u), s in zip(items, statuses))
    counts: dict[str, int] = {}
    for s in statuses:
        key = s.split(":")[0]
        counts[key] = counts.get(key, 0) + 1
    return counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="count URLs only, download nothing")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    urls = collect_urls()
    print(f"{len(urls)} image URLs found")
    if not args.dry_run:
        print(download_all(urls, workers=args.workers))


if __name__ == "__main__":
    main()
