"""Download CLEF CheckThat! 2024 Task 1 (check-worthiness, English) into data/raw/checkthat/.

Used by Channel 3A: sentences labelled Yes (check-worthy) / No (not check-worthy)
from political debates and speeches. The "No" sentences are the negative
examples the scorer needs; "Yes" sentences add positives beside AVeriTeC.

Source: https://gitlab.com/checkthat_lab/clef2024-checkthat-lab (task1/data).
Licence: research use, cite the CheckThat! 2024 overview paper (see DATASETS.md).

Run from `news-credibility/`:
    python -m src.ingest.download_checkthat
"""

from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "data" / "raw" / "checkthat"
BASE_URL = "https://gitlab.com/checkthat_lab/clef2024-checkthat-lab/-/raw/main/task1/data/"
FILES = ["CT24_checkworthy_english.zip", "CT24_checkworthy_test_gold.zip"]


def download(dest: Path = DEST, force: bool = False) -> list[Path]:
    """Download and unzip; skips files already present unless `force`. Returns extracted TSV paths."""
    dest.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        zpath = dest / name
        if force or not zpath.exists():
            print(f"Downloading {name} ...")
            urllib.request.urlretrieve(BASE_URL + name, zpath)
        with zipfile.ZipFile(zpath) as zf:
            for member in zf.namelist():
                target = (dest / member).resolve()
                if not str(target).startswith(str(dest.resolve())):  # zip-slip guard
                    raise ValueError(f"Unsafe path in {name}: {member}")
            zf.extractall(dest)
    tsvs = sorted(p for p in dest.rglob("*.tsv") if "english" in p.name.lower() and "__MACOSX" not in p.parts)
    print(f"{len(tsvs)} English TSV files in {dest}")
    return tsvs


if __name__ == "__main__":
    download()
