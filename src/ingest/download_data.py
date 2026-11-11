"""Phase 1: download AVeriTeC and write small fallback files.

    python -m src.ingest.download_data

- AVeriTeC train/dev JSON from GitHub -> data/raw/averitec/
- tiny Fakeddit / FakeNewsNet / MediaEval sample files, used by `load_raw` only when the full
  releases are absent (so the pipeline runs end to end before the big downloads)
- data/reference/india_test_set.csv: 30 hand-written placeholder rows, not a benchmark (DATASETS.md §5)

Existing files are never overwritten. The outlet ratings (data/reference/sources.csv) come from
`python -m src.ingest.source_ratings`.
"""

import json
import os
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
REFERENCE = ROOT / "data" / "reference"


def _write_once(df: pd.DataFrame, path: Path, **to_csv_kwargs) -> bool:
    """Write `df` to `path` unless the file already exists (never overwrite real data)."""
    if path.exists():
        print(f"Kept existing {path.relative_to(ROOT)}")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, **to_csv_kwargs)
    return True


def download_averitec():
    print("Downloading AVeriTeC dataset from GitHub...")
    os.makedirs(RAW / "averitec", exist_ok=True)

    base_url = "https://raw.githubusercontent.com/MichSchli/AVeriTeC/main/data/"
    files = ["train.json", "dev.json"]

    for filename in files:
        url = base_url + filename
        dest = RAW / "averitec" / filename
        if dest.exists():
            print(f"Kept existing {dest.relative_to(ROOT)}")
            continue
        print(f"Downloading {filename}...")
        try:
            urllib.request.urlretrieve(url, dest)
            with open(dest, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"Loaded {filename}: {len(data)} items")
        except Exception as e:
            print(f"Error downloading {filename}: {e}")

def create_fakeddit_sample():
    print("Creating Fakeddit raw sample structure and metadata...")

    # Create sample TSV data mimicking Fakeddit structure for testing / MVP pipeline
    sample_data = {
        "id": ["sample_01", "sample_02", "sample_03", "sample_04"],
        "title": [
            "Local mayor announces new park development downtown",
            "SHOCKING: Scientists discover aliens living in local water supply!",
            "Old photo of 2018 protests shared as happening today",
            "New economic report shows steady growth across sectors"
        ],
        "image_url": [
            "https://example.com/park.jpg",
            "https://example.com/alien.jpg",
            "https://example.com/protest.jpg",
            "https://example.com/economy.jpg"
        ],
        "label": [1, 0, 0, 1],
        "6_way_label": ["true", "false connection", "manipulated", "true"],
        "domain": ["localnews.com", "conspiracy.net", "socialrepost.org", "reuters.com"]
    }
    df = pd.DataFrame(sample_data)
    tsv_path = RAW / "fakeddit" / "fakeddit_sample.tsv"
    if _write_once(df, tsv_path, sep="\t"):
        print(f"Created Fakeddit sample TSV at {tsv_path.relative_to(ROOT)} with {len(df)} rows.")

def create_fakenewsnet_sample():
    print("Creating FakeNewsNet sample structure...")
    sample_data = {
        "news_id": ["politifact123", "gossipcop456"],
        "title": ["Senate passes bipartisan infrastructure bill", "Celebrity spotted at local coffee shop"],
        "real_fake": ["real", "fake"],
        "source": ["politifact", "gossipcop"]
    }
    df = pd.DataFrame(sample_data)
    if _write_once(df, RAW / "fakenewsnet" / "fakenewsnet_sample.csv"):
        print("Created FakeNewsNet sample CSV.")

def create_mediaeval_sample():
    print("Creating MediaEval Verifying Multimedia Use sample structure...")
    sample_data = {
        "tweet_id": ["m1", "m2"],
        "image_name": ["img1.jpg", "img2.jpg"],
        "event_name": ["protest_2024", "flood_2025"],
        "label": ["real", "fake"]
    }
    df = pd.DataFrame(sample_data)
    if _write_once(df, RAW / "mediaeval" / "mediaeval_sample.csv"):
        print("Created MediaEval sample CSV.")

def create_india_test_set():
    """SUPERSEDED (found 2026-10-29): these 30 rows were written by hand, not taken from published fact-checks,
    so no result is reported on them. Real India evaluation: the 2023-2025 test set, dev set and matched set
    (src/ingest/live_testset.py) and cleaned IFND + FakeCovid India (src/ingest/load_india.py); DATASETS.md §7.
    Kept only because older notes refer to the file; it is written once and never overwritten."""
    print("Creating the superseded placeholder India set (30 hand-written rows; not used for any result)...", flush=True)
    items = [
        {"id": "in_01", "text": "PIB Fact Check: Viral video claiming EVMs can be hacked is completely fake.", "verdict": "Refuted", "modality": "video", "source": "PIB Fact Check"},
        {"id": "in_02", "text": "Alt News: Old 2022 video of crowd in Madhya Pradesh shared as recent Delhi protest.", "verdict": "Refuted", "modality": "video", "source": "Alt News"},
        {"id": "in_03", "text": "ISRO successfully launches Chandrayaan-3 lunar mission.", "verdict": "Supported", "modality": "text", "source": "ISRO / Press Release"},
        {"id": "in_04", "text": "BOOM Live: Image of PM Modi bowing to foreign leader is digitally edited.", "verdict": "Refuted", "modality": "image", "source": "BOOM"},
        {"id": "in_05", "text": "Factly: Government is NOT distributing free smartphones to all students via WhatsApp link.", "verdict": "Refuted", "modality": "text", "source": "Factly"},
        {"id": "in_06", "text": "Reserve Bank of India announces new Rs 1000 currency notes.", "verdict": "Refuted", "modality": "text", "source": "RBI / PIB"},
        {"id": "in_07", "text": "Supreme Court directs immediate release of verdict on landmark privacy case.", "verdict": "Supported", "modality": "text", "source": "Supreme Court"},
        {"id": "in_08", "text": "Viral photo showing Taj Mahal submerged in flood water is morphed.", "verdict": "Refuted", "modality": "image", "source": "Alt News"},
        {"id": "in_09", "text": "WHO confirms outbreak of new mystery virus in southern India.", "verdict": "Refuted", "modality": "text", "source": "WHO / PIB"},
        {"id": "in_10", "text": "Indian Railways introduces bullet train service between Mumbai and Ahmedabad by 2026.", "verdict": "Supported", "modality": "text", "source": "Ministry of Railways"},
        {"id": "in_11", "text": "Viral audio clip claims government is banning cash transactions above Rs 5000.", "verdict": "Refuted", "modality": "audio", "source": "PIB Fact Check"},
        {"id": "in_12", "text": "Election Commission announces dates for General Elections.", "verdict": "Supported", "modality": "text", "source": "ECI"},
        {"id": "in_13", "text": "Photo of bridge collapse in Gujarat is actually from a 2018 incident in South America.", "verdict": "Refuted", "modality": "image", "source": "BOOM"},
        {"id": "in_14", "text": "Government launches universal basic income scheme crediting Rs 10,000 monthly.", "verdict": "Refuted", "modality": "text", "source": "PIB Fact Check"},
        {"id": "in_15", "text": "NASA satellite captures clear image of Diwali lights across India from space.", "verdict": "Supported", "modality": "image", "source": "NASA"},
        {"id": "in_16", "text": "Viral message claims salt water gargling cures viral fever instantly.", "verdict": "Refuted", "modality": "text", "source": "Factly"},
        {"id": "in_17", "text": "India signs major bilateral trade agreement with European Union.", "verdict": "Supported", "modality": "text", "source": "MEA"},
        {"id": "in_18", "text": "Old temple discovered beneath demolished commercial complex in Ayodhya.", "verdict": "Supported", "modality": "text", "source": "ASI"},
        {"id": "in_19", "text": "Viral video of police lathi-charge is from a football riot, not a political protest.", "verdict": "Refuted", "modality": "video", "source": "Alt News"},
        {"id": "in_20", "text": "Finance Ministry announces 50% tax exemption for electric vehicles.", "verdict": "Supported", "modality": "text", "source": "Ministry of Finance"},
        {"id": "in_21", "text": "WhatsApp to start charging users subscription fee starting next month.", "verdict": "Refuted", "modality": "text", "source": "PIB Fact Check"},
        {"id": "in_22", "text": "Viral photo of glowing Eiffel Tower is digital artwork, not real illumination.", "verdict": "Refuted", "modality": "image", "source": "BOOM"},
        {"id": "in_23", "text": "India's GDP growth rate exceeds 7.5% in latest financial quarter.", "verdict": "Supported", "modality": "text", "source": "MOSPI"},
        {"id": "in_24", "text": "Free laptop distribution scheme for women announced by central government.", "verdict": "Refuted", "modality": "text", "source": "PIB Fact Check"},
        {"id": "in_25", "text": "Video showing bridge swinging violently is caused by structural resonance during storm.", "verdict": "Supported", "modality": "video", "source": "Factly"},
        {"id": "in_26", "text": "World Bank praises India's digital public infrastructure framework.", "verdict": "Supported", "modality": "text", "source": "World Bank"},
        {"id": "in_27", "text": "Viral medical advice claims lemon and baking soda cures cancer.", "verdict": "Refuted", "modality": "text", "source": "Alt News"},
        {"id": "in_28", "text": "Indian Navy commissions new stealth destroyer warship.", "verdict": "Supported", "modality": "text", "source": "Indian Navy"},
        {"id": "in_29", "text": "Photo of snow-covered tropical beach in Kerala is AI-generated.", "verdict": "Refuted", "modality": "image", "source": "BOOM"},
        {"id": "in_30", "text": "Parliament passes landmark legislation on data privacy and digital rights.", "verdict": "Supported", "modality": "text", "source": "Lok Sabha / Rajya Sabha"}
    ]
    df = pd.DataFrame(items)
    if _write_once(df, REFERENCE / "india_test_set.csv"):
        print(f"Created India-context test set with {len(df)} items at data/reference/india_test_set.csv")

if __name__ == "__main__":
    download_averitec()
    create_fakeddit_sample()
    create_fakenewsnet_sample()
    create_mediaeval_sample()
    create_india_test_set()
    print("Done. Next: python -m src.ingest.source_ratings (outlet ratings), then the dataset-specific downloads.")
