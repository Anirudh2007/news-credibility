"""India 2023-2025 test set: real, dated, sourced claims from after every training dataset.

Replaces the hand-written test lines (KNOWN_CLAIMS, the 30 synthetic `india_test_set.csv` rows) as the
real-world check. Two public sources, each item with a date and a link:

- **False / misleading claims:** reviews published by Indian fact-checkers (BOOM, Alt News, Factly,
  Vishvas News, The Quint, India Today), from the Google Fact Check Tools API (ClaimReview data the
  fact-checkers publish themselves). Text = the claim as the fact-checker quotes it.
- **True claims:** India-related events from Wikipedia's "Current events" portal: one-sentence summaries
  of reported events, each citing a news outlet (CC BY-SA). Text = the summary sentence.

Why these two: both are 2023-2025, so no model or evidence index has seen them (training data ends
in 2021-2022; the date filter keeps later evidence out), and both carry dates and URLs that can be
checked by hand. Indian fact-checkers almost never publish "True" ratings (6 of 600 reviews sampled
were not False/Misleading), which is why true claims need a second source.

Known bias, reported with the results: the two halves are written by different people (fact-checkers
paraphrase viral posts, "Viral video shows ..."; Wikipedia editors write neutral summaries). A model
that reads writing style can separate them without knowing anything about truth. The evaluation
therefore also reports the per-class rates, and a matched set whose two halves are both written by
fact-checkers (--matched).

    python -m src.ingest.live_testset        # -> data/processed/india/live_testset.parquet, reports/live_testset_summary.json
    python -m src.ingest.live_testset --dev  # -> live_devset.parquet: same recipe, disjoint from test (for tuning)
    python -m src.ingest.live_testset --hoax # -> hoax_set.parquet: invented-event hoaxes (the absence rule's case)
"""

from __future__ import annotations

import json
import re
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "processed" / "india" / "live_testset.parquet"
SUMMARY = ROOT / "reports" / "live_testset_summary.json"
WIKI_API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "NewsCredibilityCollegeProject/1.0 (academic; python-requests)"}
SINCE = "2023-01-01"
UNTIL = "2025-12-31"
SEED = 42

PUBLISHERS = ["boomlive.in", "altnews.in", "factly.in", "vishvasnews.com", "thequint.com", "indiatoday.in"]
PAGES_PER_PUBLISHER = 6  # x 100 reviews; most recent first
PER_PUBLISHER = 60  # sampled per publisher, so no single fact-checker's wording dominates

_INDIA = re.compile(
    r"\b(India|Indian|Delhi|Mumbai|Kolkata|Chennai|Bengaluru|Bangalore|Hyderabad|Kerala|Tamil Nadu|Karnataka|"
    r"Maharashtra|Uttar Pradesh|Bihar|Gujarat|Rajasthan|Punjab|Haryana|Assam|Manipur|Kashmir|Jammu|Odisha|"
    r"West Bengal|Telangana|Andhra Pradesh|Madhya Pradesh|Chhattisgarh|Jharkhand|Uttarakhand|Himachal|Goa|"
    r"Sikkim|Ladakh|Lok Sabha|Rajya Sabha|ISRO|Narendra Modi|Bharatiya Janata|Indian National Congress|"
    r"Mizoram|Nagaland|Meghalaya|Tripura|Arunachal|Lucknow|Patna|Ahmedabad|Pune|Jaipur|Ayodhya)\b")


# --- False claims: Indian fact-checks (Google Fact Check Tools API) -----------------------------

def fetch_factchecks(client=None, publishers=PUBLISHERS, pages: int = PAGES_PER_PUBLISHER,
                     max_age_days: int | None = 1400, query: str | None = None) -> pd.DataFrame:
    from src.retrieval.factcheck_api import default_client, parse_response

    client = client or default_client()
    if client is None:
        raise SystemExit("GOOGLE_FACTCHECK_API_KEY is not set (.env)")
    rows = []
    for site in publishers:
        token = None
        for _ in range(pages):
            params = {"reviewPublisherSiteFilter": site, "languageCode": "en", "pageSize": 100}
            if query:
                params["query"] = query
            if max_age_days:
                params["maxAgeDays"] = max_age_days
            if token:
                params["pageToken"] = token
            body = client.query(params)
            for f in parse_response(body):
                if f.site and site in f.site:
                    rows.append({"text": f.claim, "claimant": f.claimant, "claim_date": f.claim_date,
                                 "review_date": f.review_date, "publisher": f.publisher, "site": site,
                                 "url": f.url, "rating": f.rating, "credible": f.credible})
            token = body.get("nextPageToken")
            if not token:
                break
    return pd.DataFrame(rows)


# --- True claims: Wikipedia "Current events" ------------------------------------------------------

_LINK = re.compile(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]")
_EXT = re.compile(r"\[(https?://\S+)(?:\s+[^\]]*)?\]")


def clean_wikitext(line: str) -> tuple[str, list[str]]:
    """'**At least 33 [[India]]n staff die ... [https://x (Al Jazeera)]' -> (plain sentence, [urls])."""
    urls = _EXT.findall(line)
    t = _EXT.sub("", line.lstrip("*: "))
    t = re.sub(r"<ref.*?(</ref>|/>)", "", t, flags=re.S)
    t = re.sub(r"\{\{[^{}]*\}\}", "", t)
    t = _LINK.sub(r"\1", t)
    t = re.sub(r"'{2,}|<[^>]+>", "", t)
    return " ".join(t.replace("( )", "").split()).strip(" ()"), urls


def india_events(wikitext: str) -> list[tuple[str, list[str]]]:
    """Event sentences (deepest bullets with a source link) about India: the sentence names an Indian
    place/body, or its parent topic bullet does ("*[[2024 Indian general election]]")."""
    out, parents = [], {}
    for line in wikitext.splitlines():
        m = re.match(r"^(\*+)", line)
        if not m:
            continue
        depth = len(m.group(1))
        parents = {d: t for d, t in parents.items() if d < depth}
        text, urls = clean_wikitext(line)
        parents[depth] = text
        context = " ".join(parents.get(d, "") for d in range(1, depth))
        if urls and len(text.split()) >= 8 and (_INDIA.search(text) or _INDIA.search(context)):
            out.append((text, urls))
    return out


RAW_EVENTS = ROOT / "data" / "raw" / "current_events"


def _month_pages(titles: list[str], cache: Path) -> list[dict]:
    """One month of portal pages (one API call for up to 31 titles), cached on disk; backs off on 429."""
    import requests

    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    for attempt in range(6):
        r = requests.get(WIKI_API, params={"action": "query", "prop": "revisions", "rvprop": "content",
                                           "rvslots": "main", "titles": "|".join(titles), "format": "json",
                                           "formatversion": 2}, headers=HEADERS, timeout=60)
        if r.status_code != 429:
            break
        time.sleep(10 * (attempt + 1))
    r.raise_for_status()
    pages = r.json()["query"]["pages"]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(pages), encoding="utf-8")
    time.sleep(2.0)
    return pages


def fetch_current_events(since: str = SINCE, until: str = UNTIL) -> pd.DataFrame:
    rows = []
    for month_start in pd.date_range(since, until, freq="MS"):
        days = pd.date_range(month_start, month_start + pd.offsets.MonthEnd(0))
        titles = [f"Portal:Current events/{d.year} {d.strftime('%B')} {d.day}" for d in days]
        for page in _month_pages(titles, RAW_EVENTS / f"{month_start:%Y-%m}.json"):
            if "revisions" not in page:
                continue
            day = pd.to_datetime(page["title"].split("/", 1)[1], format="%Y %B %d").date().isoformat()
            for text, urls in india_events(page["revisions"][0]["slots"]["main"]["content"]):
                rows.append({"text": text, "claim_date": day, "review_date": None, "publisher": "Wikipedia Current events",
                             "site": "en.wikipedia.org", "url": urls[0], "rating": "reported event", "credible": 1.0,
                             "claimant": None, "wiki_page": f"https://en.wikipedia.org/wiki/{page['title'].replace(' ', '_')}"})
    return pd.DataFrame(rows)


# --- Assembly -------------------------------------------------------------------------------------

def near_duplicates(texts: list[str], threshold: float = 0.8) -> np.ndarray:
    """Mask of rows that repeat an earlier row (TF-IDF cosine >= threshold): the same viral claim
    reviewed by two fact-checkers counts once (same rule as Phase 2 dedup, slightly looser)."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    X = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit_transform(texts)
    sims = (X @ X.T).tocsr()
    dup = np.zeros(len(texts), bool)
    for i in range(len(texts)):
        if not dup[i]:
            row = sims.getrow(i)
            later = [j for j, v in zip(row.indices, row.data) if j > i and v >= threshold]
            dup[later] = True
    return dup


def build(seed: int = SEED) -> pd.DataFrame:
    fc = fetch_factchecks()
    fc["date"] = fc["claim_date"].fillna(fc["review_date"])
    fc = fc[(fc["date"] >= SINCE) & (fc["date"] <= UNTIL) & (fc["credible"] == 0.0)]
    fc = fc[~near_duplicates(fc["text"].tolist())]
    fc = pd.concat([g.sample(min(len(g), PER_PUBLISHER), random_state=seed) for _, g in fc.groupby("site")])
    wiki = fetch_current_events()
    wiki["date"] = wiki["claim_date"]
    wiki = wiki[~near_duplicates(wiki["text"].tolist())]
    wiki = wiki.sample(min(len(wiki), len(fc)), random_state=seed)  # balanced classes
    df = pd.concat([fc.assign(origin="indian_factcheck"), wiki.assign(origin="wikipedia_current_events")],
                   ignore_index=True)
    df["item_id"] = [f"live_{i:04d}" for i in range(len(df))]
    df["split"] = "test"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    SUMMARY.write_text(json.dumps({
        "built": date.today().isoformat(), "rows": int(len(df)), "window": [SINCE, UNTIL],
        "by_origin": df["origin"].value_counts().to_dict(),
        "false_by_publisher": fc["publisher"].value_counts().to_dict(),
        "false_ratings": fc["rating"].str.lower().value_counts().head(10).to_dict(),
        "dates_by_year": df["date"].str[:4].value_counts().sort_index().to_dict(),
        "median_words": {o: float(g["text"].str.split().str.len().median()) for o, g in df.groupby("origin")},
        "true_cited_outlets_top": wiki["url"].str.extract(r"https?://(?:www\.)?([^/]+)")[0].value_counts().head(10).to_dict(),
    }, indent=2))
    return df


DEV_OUT = ROOT / "data" / "processed" / "india" / "live_devset.parquet"
DEV_SUMMARY = ROOT / "reports" / "live_devset_summary.json"


def _not_in_test(texts: list[str], test_texts: list[str], threshold: float = 0.8) -> np.ndarray:
    """Mask of candidate texts that are not a (near-)copy of any test text (same TF-IDF rule as dedup)."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(list(texts) + list(test_texts))
    sims = vec.transform(texts) @ vec.transform(test_texts).T
    return np.asarray(sims.max(axis=1).todense()).ravel() < threshold


def build_dev(seed: int = SEED + 1) -> pd.DataFrame:
    """Development set for tuning thresholds (2026-11-03): same sources, window and recipe as the test
    set, but no item of the test set and no near-copy of one. Thresholds are chosen on this set; the
    test set is scored once, afterwards, so its numbers are not tuned to it."""
    test = pd.read_parquet(OUT)
    fc = fetch_factchecks()
    fc["date"] = fc["claim_date"].fillna(fc["review_date"])
    fc = fc[(fc["date"] >= SINCE) & (fc["date"] <= UNTIL) & (fc["credible"] == 0.0)]
    fc = fc[~near_duplicates(fc["text"].tolist())]
    fc = fc[~fc["url"].isin(set(test["url"].dropna()))]
    fc = fc[_not_in_test(fc["text"].tolist(), test["text"].tolist())]
    fc = pd.concat([g.sample(min(len(g), PER_PUBLISHER), random_state=seed) for _, g in fc.groupby("site")])
    wiki = fetch_current_events()
    wiki["date"] = wiki["claim_date"]
    wiki = wiki[~near_duplicates(wiki["text"].tolist())]
    wiki = wiki[_not_in_test(wiki["text"].tolist(), test["text"].tolist())]
    wiki = wiki.sample(min(len(wiki), len(fc)), random_state=seed)
    df = pd.concat([fc.assign(origin="indian_factcheck"), wiki.assign(origin="wikipedia_current_events")],
                   ignore_index=True)
    df["item_id"] = [f"livedev_{i:04d}" for i in range(len(df))]
    df["split"] = "dev"
    df.to_parquet(DEV_OUT, index=False)
    DEV_SUMMARY.write_text(json.dumps({
        "built": date.today().isoformat(), "rows": int(len(df)), "window": [SINCE, UNTIL],
        "by_origin": df["origin"].value_counts().to_dict(),
        "false_by_publisher": fc["publisher"].value_counts().to_dict(),
        "dates_by_year": df["date"].str[:4].value_counts().sort_index().to_dict(),
        "overlap_with_test": "none (url and TF-IDF cosine >= 0.8 removed)",
    }, indent=2))
    return df


MATCHED_OUT = ROOT / "data" / "processed" / "india" / "matched_testset.parquet"
MATCHED_SUMMARY = ROOT / "reports" / "matched_testset_summary.json"
MATCHED_PUBLISHERS = ["factly.in", "altnews.in", "boomlive.in", "youturn.in", "digiteye.in", "indiatoday.in",
                      "factchecker.in", "newsmobile.in"]
STRICT_TRUE = {"true", "mostly true", "correct", "accurate"}


def build_matched(seed: int = SEED) -> pd.DataFrame:
    """Second India test set (2026-11-03): BOTH halves written by Indian fact-checkers. True = claims they
    rated exactly "True"/"Mostly true"; False = for each true claim, one claim the same fact-checker rated
    False in the same year. Publisher, year and writing style therefore cannot separate the labels (the
    known bias of the 2023-2025 set, whose true half is Wikipedia text). Test only, never tuned on."""
    fc = fetch_factchecks(publishers=MATCHED_PUBLISHERS, pages=200, max_age_days=None)
    fc["date"] = fc["claim_date"].fillna(fc["review_date"])
    fc = fc[fc["date"].notna() & fc["text"].str.split().str.len().ge(5)]
    fc = fc[~near_duplicates(fc["text"].tolist())]
    fc["year"] = fc["date"].str[:4]
    true = fc[fc["rating"].str.strip().str.lower().isin(STRICT_TRUE)]
    false = fc[fc["credible"] == 0.0]
    rng = np.random.default_rng(seed)
    picked = []
    for (site, year), g in true.groupby(["site", "year"]):
        pool = false[(false["site"] == site) & (false["year"] == year)]
        n = min(len(g), len(pool))
        if n:
            picked += [g.sample(n, random_state=seed), pool.iloc[rng.choice(len(pool), n, replace=False)]]
    df = pd.concat(picked, ignore_index=True)
    df["credible"] = (df["rating"].str.strip().str.lower().isin(STRICT_TRUE)).astype(float)
    df["origin"] = np.where(df["credible"] == 1.0, "factcheck_true", "factcheck_false")
    df["item_id"] = [f"matched_{i:04d}" for i in range(len(df))]
    df["split"] = "test"
    df.to_parquet(MATCHED_OUT, index=False)
    MATCHED_SUMMARY.write_text(json.dumps({
        "built": date.today().isoformat(), "rows": int(len(df)),
        "by_label": df["credible"].value_counts().to_dict(),
        "by_site": df.groupby(["site", "credible"]).size().unstack(fill_value=0).to_dict(),
        "by_year": df.groupby("year").size().to_dict(),
        "median_words": {str(k): float(g["text"].str.split().str.len().median()) for k, g in df.groupby("credible")},
    }, indent=2))
    return df


HOAX_OUT = ROOT / "data" / "processed" / "india" / "hoax_set.parquet"
HOAX_SUMMARY = ROOT / "reports" / "hoax_set_summary.json"
# Words that name the events the news always covers (src/retrieval/coverage.py EVENTS), fixed before any search.
HOAX_QUERIES = ["died", "death", "passed away", "dead", "resigned", "arrested", "banned", "ban", "will give", "free",
                "every citizen", "scheme", "notice", "announced", "holiday", "lockdown", "withdrawn", "cancelled"]
HOAX_N = 60  # items, at most HOAX_PER_PUBLISHER from one fact-checker (Tavily: about one search each)
HOAX_PER_PUBLISHER = 12


def build_hoaxes(seed: int = SEED + 2) -> pd.DataFrame:
    """Invented-event hoaxes (2026-11-07), the case where silence should be evidence: "X died", "Rs 5,000 for
    every citizen", "RBI bans ...". The 2023-2025 India sets hold only 4-7 such fakes each (most fact-checked
    rumours twist a real event, so the topic is in the news), too few to measure the absence rule.

    Recipe, fixed before any news search: the same six Indian fact-checkers and window as the India sets; claims
    found by the event words in HOAX_QUERIES and rated False; no item (url) or near-copy (TF-IDF >= 0.8) of the dev
    or test set; kept only if the app's own rule says the news would cover it if true (`expected_coverage`, with
    the Wikipedia notability check); at most HOAX_PER_PUBLISHER per fact-checker, HOAX_N in all. Split in half at
    random: "dev" (for the decision) and "test" (scored once). Fakes only: the true side is the India sets'."""
    from src.claims.extract import tag_entities
    from src.retrieval.coverage import expected_coverage
    from src.retrieval.wiki_search import WikiSearchClient

    frames = []
    for q in HOAX_QUERIES:
        f = fetch_factchecks(pages=1, query=q)
        frames.append(f.assign(query=q))
        print(f"  {q!r}: {len(f)} reviews", flush=True)
    fc = pd.concat(frames, ignore_index=True)
    found = len(fc)
    fc["date"] = fc["claim_date"].fillna(fc["review_date"])
    fc = fc[(fc["date"] >= SINCE) & (fc["date"] <= UNTIL) & (fc["credible"] == 0.0)]
    fc = fc.drop_duplicates("url").reset_index(drop=True)
    fc = fc[~near_duplicates(fc["text"].tolist())]
    seen = pd.concat([pd.read_parquet(OUT), pd.read_parquet(DEV_OUT)], ignore_index=True)
    fc = fc[~fc["url"].isin(set(seen["url"].dropna()))]
    fc = fc[_not_in_test(fc["text"].tolist(), seen["text"].tolist())].reset_index(drop=True)
    candidates = len(fc)
    wiki = WikiSearchClient(retries=6)
    cov = [expected_coverage(t, tag_entities(t), wiki.is_well_known) for t in fc["text"]]
    fc["event"] = [c.event for c in cov]
    fc = fc[[c.expected for c in cov]].reset_index(drop=True)
    expected = len(fc)
    fc = pd.concat([g.sample(min(len(g), HOAX_PER_PUBLISHER), random_state=seed) for _, g in fc.groupby("site")])
    fc = fc.sample(min(len(fc), HOAX_N), random_state=seed).reset_index(drop=True)
    fc["split"] = np.where(np.random.default_rng(seed).permutation(len(fc)) % 2 == 0, "dev", "test")
    fc["origin"] = "invented_event_hoax"
    fc["item_id"] = [f"hoax_{i:03d}" for i in range(len(fc))]
    HOAX_OUT.parent.mkdir(parents=True, exist_ok=True)
    fc.to_parquet(HOAX_OUT, index=False)
    HOAX_SUMMARY.write_text(json.dumps({
        "built": date.today().isoformat(), "queries": HOAX_QUERIES, "reviews_found": found,
        "false_in_window_new_not_near_dev_or_test": candidates, "expected_coverage": expected,
        "rows": int(len(fc)), "by_split": fc["split"].value_counts().to_dict(),
        "by_publisher": fc["publisher"].value_counts().to_dict(),
        "by_event": fc["event"].value_counts().to_dict(),
        "dates_by_year": fc["date"].str[:4].value_counts().sort_index().to_dict(),
    }, indent=2))
    return fc


def main() -> None:
    import sys

    SUMMARY.parent.mkdir(parents=True, exist_ok=True)  # all four summaries go to reports/, not in git
    if "--hoax" in sys.argv:
        build_hoaxes()
        print(json.dumps(json.loads(HOAX_SUMMARY.read_text()), indent=1))
        return
    if "--matched" in sys.argv:
        build_matched()
        print(json.dumps(json.loads(MATCHED_SUMMARY.read_text()), indent=1))
        return
    if "--dev" in sys.argv:
        build_dev()
        print(json.dumps(json.loads(DEV_SUMMARY.read_text()), indent=1))
        return
    build()
    print(json.dumps(json.loads(SUMMARY.read_text()), indent=1))


if __name__ == "__main__":
    main()
