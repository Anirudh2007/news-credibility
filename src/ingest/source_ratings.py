"""Build data/reference/sources.csv (domain reliability ratings) from public, citable lists.

Before (Phase 1) the file held 6 hand-typed domains, and only 0.4% of FakeNewsNet items had a rated
domain, so the source check learned almost nothing from reliability. Now every rating comes from a
named public source, with a fixed number per category, and the `basis` column says which:

| basis | reliability | bias column |
| --- | --- | --- |
| Wikipedia RSP "generally reliable" | 0.90 | news |
| Wikipedia RSP "no consensus" | 0.60 | news |
| Wikipedia RSP "generally unreliable" | 0.30 | unreliable |
| Wikipedia RSP "deprecated" | 0.10 | unreliable |
| Wikipedia "List of fake news websites" | 0.10 | fake-news |
| Wikipedia "List of satirical news websites" (+3 Indian satire sites, manual) | 0.10 | satire |
| documented Indian fake-news site missing from the list (manual: postcard.news) | 0.10 | fake-news |
| IFCN-verified fact-checker (manual list below) | 0.90 | fact-check |
| official / intergovernmental body (manual) | 0.85 | official |
| established Indian national outlet without an RSP entry (manual) | 0.75 | news |

RSP = Wikipedia:Reliable sources/Perennial sources: editors' consensus on 490+ sources, each
discussed on the noticeboard (CC BY-SA). When sources disagree, the lowest rating is kept (cautious).
The manual tiers are short, listed in this file, and must be reviewed by the team. Political lean
is never rated; satire and documented fake-news sites are.

    python -m src.ingest.source_ratings          # fetch (cached in data/raw/source_ratings/), write sources.csv

Retrain the source check afterwards (`python -m src.source.credibility`): reliability is one of its
features, and the verifier also weights evidence passages by it.
"""

from __future__ import annotations

import json
import re
import time
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "source_ratings"
OUT = ROOT / "data" / "reference" / "sources.csv"
SUMMARY = ROOT / "reports" / "source_ratings_summary.json"
HEADERS = {"User-Agent": "NewsCredibilityCollegeProject/1.0 (academic; python-requests)"}
WIKI_API = "https://en.wikipedia.org/w/api.php"

RSP_PAGES = [f"Wikipedia:Reliable sources/Perennial sources/{i}" for i in range(1, 9)]
FAKE_LIST = "List of fake news websites"
SATIRE_LIST = "List of satirical news websites"

RSP_STATUS = {"gr": (0.90, "news"), "nc": (0.60, "news"), "gu": (0.30, "unreliable"), "d": (0.10, "unreliable")}

# IFCN signatories (ifcncodeofprinciples.poynter.org): India first, then the international ones that
# appear in Google Fact Check results.
FACT_CHECKERS = [
    "boomlive.in", "altnews.in", "factly.in", "newschecker.in", "vishvasnews.com", "newsmeter.in",
    "factcrescendo.com", "digiteye.in", "newsmobile.in", "youturn.in", "factchecker.in",
    "politifact.com", "factcheck.org", "snopes.com", "fullfact.org", "leadstories.com", "checkyourfact.com",
    "africacheck.org", "misbar.com", "logicallyfacts.com", "aap.com.au", "healthfeedback.org", "sciencefeedback.co",
]
OFFICIAL = [
    "pib.gov.in", "mygov.in", "india.gov.in", "rbi.org.in", "isro.gov.in", "mohfw.gov.in", "icmr.gov.in",
    "eci.gov.in", "mea.gov.in", "sci.gov.in", "who.int", "un.org", "cdc.gov", "nih.gov", "nasa.gov",
]
# Indian sites missing from the two Wikipedia lists, each described as such in its own Wikipedia article.
INDIA_SATIRE = ["fakingnews.com", "thefauxy.com", "theunrealtimes.com"]
INDIA_FAKE = ["postcard.news"]
# Major Indian national dailies, broadcasters and agencies with no RSP entry (RSP covers few Indian
# outlets). A middle rating: established newsroom, but no community review to cite.
INDIA_NATIONAL = [
    "ndtv.com", "hindustantimes.com", "livemint.com", "business-standard.com", "deccanherald.com",
    "telegraphindia.com", "tribuneindia.com", "newindianexpress.com", "thestatesman.com", "indiatoday.in",
    "thequint.com", "scroll.in", "theprint.in", "news18.com", "firstpost.com", "moneycontrol.com",
    "outlookindia.com", "deccanchronicle.com", "ptinews.com", "ianslive.in", "newsonair.gov.in", "ddnews.gov.in",
]


SPOT_CHECKS = ["reuters.com", "apnews.com", "bbc.co.uk", "nytimes.com", "cnn.com", "foxnews.com", "dailymail.co.uk",
               "tmz.com", "people.com", "infowars.com", "thehindu.com", "indianexpress.com", "indiatimes.com",
               "ndtv.com", "opindia.com", "republicworld.com", "postcard.news", "theonion.com", "fakingnews.com",
               "boomlive.in", "altnews.in", "pib.gov.in"]


def fetch_wikitext(page: str, refresh: bool = False) -> tuple[str, int | None]:
    """(wikitext, revision id), cached in data/raw/source_ratings/ so the build is reproducible."""
    import requests

    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / (re.sub(r"[^A-Za-z0-9]+", "_", page).strip("_") + ".json")
    if path.exists() and not refresh:
        d = json.loads(path.read_text(encoding="utf-8"))
        return d["wikitext"], d.get("revid")
    r = requests.get(WIKI_API, params={"action": "parse", "page": page, "prop": "wikitext|revid", "format": "json",
                                       "formatversion": 2}, headers=HEADERS, timeout=60)
    r.raise_for_status()
    d = r.json()["parse"]
    path.write_text(json.dumps({"page": page, "revid": d.get("revid"), "fetched": date.today().isoformat(),
                                "wikitext": d["wikitext"]}), encoding="utf-8")
    time.sleep(1.0)
    return d["wikitext"], d.get("revid")


def _clean_domain(cell: str) -> list[str]:
    cell = re.sub(r"<ref.*?(</ref>|/>)", "", cell, flags=re.S)
    cell = re.sub(r"\{\{[^}]*\}\}|\[\[|\]\]|'''?|<[^>]+>", " ", cell)
    out = []
    for tok in re.split(r"[\s,;/]+", cell.lower()):
        tok = tok.strip(".()").removeprefix("www.")
        if re.fullmatch(r"[a-z0-9-]+(\.[a-z0-9-]+)+", tok):
            out.append(tok)
    return out


def parse_rsp(wikitext: str) -> list[tuple[str, str, str]]:
    """(domain, status code, source name) for every RSP row with a status and listed domains."""
    out = []
    for row in re.split(r"\n\|-[^\n]*", wikitext):
        st = re.search(r"\{\{WP:RSPSTATUS\|([a-z]+)", row)
        name = re.search(r"\n\|\s*(?:data-sort-value=\"[^\"]*\"\s*\|)?\s*(.+)", row)
        if not st or st.group(1) not in RSP_STATUS:
            continue
        for uses in re.findall(r"\{\{WP:RSPUSES\|([^}]*)\}\}", row):
            for d in uses.split("|"):
                d = d.strip().lower().removeprefix("www.")
                if "." in d and "=" not in d:
                    label = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]|''|\{\{.*?\}\}", r"\1", name.group(1)).strip() if name else d
                    out.append((d, st.group(1), label[:80]))
    return out


def parse_table_domains(wikitext: str) -> list[tuple[str, str]]:
    """(domain, name) from wikitables whose second column is the domain (both Wikipedia lists)."""
    out = []
    for table in re.findall(r"\{\|.*?\n\|\}", wikitext, flags=re.S):
        for row in table.split("\n|-")[1:]:
            cells = [c.strip() for c in re.split(r"\n\|(?!-)", "\n" + row.strip()) if c.strip()]
            if len(cells) >= 2 and not cells[0].startswith("!"):
                name = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]|''|\{\{[^}]*\}\}", r"\1", cells[0]).strip()
                for d in _clean_domain(cells[1]):
                    out.append((d, name[:80]))
    return out


def build(refresh: bool = False) -> pd.DataFrame:
    from src.source.credibility import parse_domain

    rows, revs = [], {}
    for page in RSP_PAGES:
        text, revs[page] = fetch_wikitext(page, refresh)
        rows += [(d, *RSP_STATUS[s], f"Wikipedia RSP: {s}", n) for d, s, n in parse_rsp(text)]
    text, revs[FAKE_LIST] = fetch_wikitext(FAKE_LIST, refresh)
    rows += [(d, 0.10, "fake-news", "Wikipedia: List of fake news websites", n) for d, n in parse_table_domains(text)]
    text, revs[SATIRE_LIST] = fetch_wikitext(SATIRE_LIST, refresh)
    rows += [(d, 0.10, "satire", "Wikipedia: List of satirical news websites", n) for d, n in parse_table_domains(text)]
    rows += [(d, 0.10, "satire", "Indian satire site (manual list; Wikipedia article)", d) for d in INDIA_SATIRE]
    rows += [(d, 0.10, "fake-news", "documented fake-news site (manual list; Wikipedia article)", d) for d in INDIA_FAKE]
    rows += [(d, 0.90, "fact-check", "IFCN signatory (manual list)", d) for d in FACT_CHECKERS]
    rows += [(d, 0.85, "official", "official / intergovernmental (manual list)", d) for d in OFFICIAL]

    df = pd.DataFrame(rows, columns=["domain", "reliability", "bias", "basis", "name"])
    # Registered domain, as the source check looks it up ('edition.cnn.com' -> 'cnn.com'). Platforms that
    # host anyone's content (blog hosts, social media) are not rated as a whole.
    df["domain"] = [parse_domain(d)[0] or d for d in df["domain"]]
    hosts = {"blogspot.com", "wordpress.com", "medium.com", "substack.com", "facebook.com", "twitter.com", "x.com",
             "youtube.com", "instagram.com", "tiktok.com", "reddit.com", "google.com", "amazon.com", "github.io",
             "tumblr.com", "wix.com", "weebly.com", "quora.com", "linkedin.com", "telegram.org", "whatsapp.com"}
    df = df[~df["domain"].isin(hosts)]
    # Satire wins the label (the source check reads bias == "satire"); otherwise lowest rating wins.
    df["satire"] = df["bias"].eq("satire")
    df = df.sort_values(["domain", "satire", "reliability"], ascending=[True, False, True])
    df = df.drop_duplicates("domain").drop(columns="satire")
    rated = set(df["domain"])
    extra = [(d, 0.75, "news", "established Indian national outlet, no RSP entry (manual list)", d)
             for d in INDIA_NATIONAL if (parse_domain(d)[0] or d) not in rated]
    df = pd.concat([df, pd.DataFrame(extra, columns=df.columns)], ignore_index=True).sort_values("domain")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps({
        "built": date.today().isoformat(), "domains": int(len(df)),
        "by_basis": df["basis"].value_counts().to_dict(), "by_bias": df["bias"].value_counts().to_dict(),
        "wikipedia_revisions": revs, "previous": "6 hand-typed domains (Phase 1)",
        "spot_checks": {d: (df.loc[df["domain"] == d, ["reliability", "bias", "basis"]].iloc[0].tolist()
                            if (df["domain"] == d).any() else None) for d in SPOT_CHECKS},
    }, indent=2))
    return df


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="fetch the Wikipedia pages again instead of the cache")
    df = build(ap.parse_args().refresh)
    print(f"{len(df)} domains -> {OUT.relative_to(ROOT)}")
    print(df["basis"].value_counts().to_string())


if __name__ == "__main__":
    main()
