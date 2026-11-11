"""Reference photos of Indian landmarks for the location check (3E), from Wikimedia Commons.

Why Commons and not Google: Google Maps / Street View / Places photos may not be downloaded, stored
or used to build a dataset (Google Maps Platform terms: no caching beyond short limits, no creating
content from Maps content). Wikimedia Commons photos are openly licensed (CC BY / CC BY-SA / public
domain), sorted into one category per landmark, and the API gives the licence and author of each
file, so the attribution is kept next to every photo (`sources.csv` in each folder).

    python -m src.ingest.download_landmarks              # ~20 photos per landmark -> data/reference/landmarks/<slug>/
    python -m src.ingest.download_landmarks --calibrate  # choose LOCATION_MIN_SIMILARITY -> reports/location_metrics.json

Add a landmark: one line in LANDMARKS (folder name -> Commons category names). The folder name is
what the pipeline matches against places named in the text ("Jantar Mantar" -> jantar_mantar).
Other open sources: Mapillary street-level photos (CC BY-SA, free API token), the Google Landmarks
Dataset v2 (built from Commons, 5M images). Both are optional; Commons is enough for a demo.
"""

from __future__ import annotations

import argparse
import csv
import html
import io
import json
import re
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "reference" / "landmarks"
REPORT = ROOT / "reports" / "location_metrics.json"
API = "https://commons.wikimedia.org/w/api.php"
# Wikimedia asks every client to identify itself (no personal data needed).
HEADERS = {"User-Agent": "NewsCredibilityCollegeProject/1.0 (academic; python-requests)"}
PER_PLACE = 20
THUMB_WIDTH = 640  # CLIP reads 224 px; 640 keeps detail for a human check
MIN_SIDE = 400
SKIP = re.compile(r"\b(map|plan|logo|stamp|drawing|diagram|coin|postcard|sketch|painting|svg|interior|inside|"
                  r"ceiling|inscription|plaque|illustrat\w*|aquatint|engraving|lithograph|poster|watercolou?r|"
                  r"replica|model|miniature)\b", re.I)

# Places that often appear in Indian news photos (protests, ceremonies, disasters, "viral" claims).
LANDMARKS: dict[str, list[str]] = {
    "india_gate": ["India Gate"],
    "jantar_mantar": ["Jantar Mantar (Delhi)", "Demonstrations and protests at Jantar Mantar (Delhi)"],
    "red_fort": ["Red Fort"],
    "parliament_house": ["Old Parliament House, New Delhi", "New Parliament House, New Delhi"],
    "rashtrapati_bhavan": ["Rashtrapati Bhavan"],
    "qutb_minar": ["Qutb Minar"],
    "gateway_of_india": ["Gateway of India"],
    "chhatrapati_shivaji_terminus": ["Chhatrapati Shivaji Maharaj Terminus", "Chhatrapati Shivaji Terminus"],
    "taj_mahal": ["Taj Mahal"],
    "charminar": ["Charminar"],
    "howrah_bridge": ["Howrah Bridge"],
    "victoria_memorial": ["Victoria Memorial, Kolkata"],
    "golden_temple": ["Golden Temple (Harmandir Sahib)", "Harmandir Sahib"],
}


def _get(url: str, params: dict | None = None, tries: int = 4):
    """GET with backoff: Commons answers 429 (too many requests) to fast clients."""
    import requests

    for attempt in range(tries):
        r = requests.get(url, params=params, headers=HEADERS, timeout=30)
        if r.status_code != 429:
            r.raise_for_status()
            return r
        time.sleep(5 * (attempt + 1))
    r.raise_for_status()
    return r


def _api(params: dict) -> dict:
    return _get(API, {"format": "json", **params}).json()


def category_files(category: str, limit: int = 200) -> list[dict]:
    """Files directly in a Commons category, with thumbnail URL, size, licence and author."""
    out, cont = [], {}
    while len(out) < limit:
        d = _api({"action": "query", "generator": "categorymembers", "gcmtitle": f"Category:{category}",
                  "gcmtype": "file", "gcmlimit": 50, "prop": "imageinfo",
                  "iiprop": "url|size|mime|extmetadata", "iiurlwidth": THUMB_WIDTH, **cont})
        for p in d.get("query", {}).get("pages", {}).values():
            ii = (p.get("imageinfo") or [{}])[0]
            meta = ii.get("extmetadata", {})
            out.append({
                "pageid": p["pageid"], "title": p["title"], "mime": ii.get("mime"),
                "width": ii.get("width", 0), "height": ii.get("height", 0),
                "thumb": ii.get("thumburl") or ii.get("url"), "page": ii.get("descriptionurl"),
                "license": meta.get("LicenseShortName", {}).get("value"),
                "artist": html.unescape(re.sub(r"<[^>]+>", "", meta.get("Artist", {}).get("value", ""))).strip(),
            })
        if "continue" not in d:
            break
        cont = d["continue"]
    return out


def usable(f: dict) -> bool:
    """A photo of the outside of the landmark: JPEG/PNG, large enough, openly licensed, not a map."""
    return (f["mime"] in ("image/jpeg", "image/png") and min(f["width"], f["height"]) >= MIN_SIDE
            and bool(f["license"]) and not SKIP.search(f["title"].replace("_", " ")))


def download(slugs: list[str] | None = None, per_place: int = PER_PLACE, seed: int = 42) -> dict:
    import random

    from PIL import Image

    rng = random.Random(seed)
    summary = {}
    for slug, categories in LANDMARKS.items():
        if slugs and slug not in slugs:
            continue
        files = [f for c in categories for f in category_files(c) if usable(f)]
        rng.shuffle(files)  # not the alphabetical first 20: spread over photographers and years
        folder = OUT_DIR / slug
        folder.mkdir(parents=True, exist_ok=True)
        rows, saved = [], 0
        for f in files:
            if saved >= per_place:
                break
            path = folder / f"{f['pageid']}.jpg"
            if not path.exists():
                try:
                    r = _get(f["thumb"])
                    Image.open(io.BytesIO(r.content)).convert("RGB").save(path, "JPEG", quality=90)
                except Exception as e:  # dead thumbnail, not an image
                    print(f"  skip {f['title']}: {type(e).__name__}")
                    continue
                time.sleep(1.0)  # be polite to Commons
            rows.append({k: f[k] for k in ("pageid", "title", "page", "license", "artist")} | {"file": path.name})
            saved += 1
        with open(folder / "sources.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["file", "pageid", "title", "page", "license", "artist"])
            w.writeheader()
            w.writerows(rows)
        summary[slug] = {"categories": categories, "candidates": len(files), "saved": saved}
        print(f"{slug:30s} {saved:3d} photos ({len(files)} usable in {categories})")
    return summary


# --- Calibration --------------------------------------------------------------------------------

def similarity_samples(vectors: dict[str, np.ndarray]) -> tuple[list[float], list[float]]:
    """Leave-one-out. Positive: a photo of P against the other photos of P (max cosine, the
    location check's own rule). Negative: the same photo against every other place's photos
    ("a photo of P posted as Q")."""
    pos, neg = [], []
    for p, vp in vectors.items():
        for i in range(len(vp)):
            others = np.delete(vp, i, axis=0)
            if len(others):
                pos.append(float(np.max(others @ vp[i])))
            for q, vq in vectors.items():
                if q != p:
                    neg.append(float(np.max(vq @ vp[i])))
    return pos, neg


def calibrate(max_false_mismatch: float = 0.05) -> dict:
    """Threshold = the cosine below which only `max_false_mismatch` of true photos fall, chosen on
    half of the places and reported on the other half (places never used to choose it)."""
    from PIL import Image
    from sklearn.metrics import roc_auc_score

    from src.media.text_image import ClipEncoder

    enc = ClipEncoder()
    vectors = {}
    for folder in sorted(p for p in OUT_DIR.iterdir() if p.is_dir()):
        files = sorted(folder.glob("*.jpg"))
        if len(files) >= 3:
            vectors[folder.name] = enc.encode_images([Image.open(f).convert("RGB") for f in files])
    places = sorted(vectors)
    choose, report = places[::2], places[1::2]
    pos_c, _ = similarity_samples({p: vectors[p] for p in choose})
    t = float(np.quantile(pos_c, max_false_mismatch))
    pos_r, neg_r = similarity_samples({p: vectors[p] for p in report})
    pos_all, neg_all = similarity_samples(vectors)
    pos_r, neg_r = np.array(pos_r), np.array(neg_r)
    out = {
        "threshold": round(t, 3), "rule": f"{int(100 * max_false_mismatch)}th percentile of same-place similarity "
                                          "(places used to choose it)",
        "places_choose": choose, "places_report": report,
        "photos": {p: int(len(v)) for p, v in vectors.items()},
        "report_half": {"true_photos_wrongly_flagged": float(np.mean(pos_r < t)),
                        "wrong_place_flagged": float(np.mean(neg_r < t)),
                        "roc_auc": float(roc_auc_score(np.r_[np.ones(len(pos_r)), np.zeros(len(neg_r))],
                                                       np.r_[pos_r, neg_r]))},
        "all_places": {"same_place_median": float(np.median(pos_all)), "other_place_median": float(np.median(neg_all))},
        "old_placeholder": 0.25,
        "note": "Commons photos are clear, well-framed views. News photos (night, crowds, close-ups) match less "
                "well, so a real mismatch rate on news photos would be higher than wrong_place_flagged suggests "
                "and false alarms more frequent than true_photos_wrongly_flagged.",
    }
    REPORT.write_text(json.dumps(out, indent=2))
    return out


def main() -> None:
    import sys

    sys.stdout.reconfigure(errors="replace")  # file titles in Devanagari on a cp1252 console
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="landmark folder names (default: all)")
    ap.add_argument("--per-place", type=int, default=PER_PLACE)
    ap.add_argument("--calibrate", action="store_true", help="choose the location threshold from the photos on disk")
    args = ap.parse_args()
    if args.calibrate:
        print(json.dumps(calibrate(), indent=2))
    else:
        download(args.only, args.per_place)


if __name__ == "__main__":
    main()
