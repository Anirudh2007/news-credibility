# Datasets Documentation (DATASETS.md)

This document details the datasets collected and prepared for the News Credibility and Claim Verification System (Phase 1).

## 1. AVeriTeC
- **Source:** [MichSchli/AVeriTeC](https://github.com/MichSchli/AVeriTeC) / HuggingFace `pminervini/averitec`
- **Licence:** CC BY 4.0
- **Size:** Train (~5,600 claims) & Dev (~1,700 claims)
- **Download Date:** 2026-10-26
- **Purpose:** Claim extraction, evidence retrieval, and 4-way claim verification (Supported, Refuted, Conflicting, Not Enough Evidence).

## 2. Fakeddit
- **Source:** r/Fakeddit Benchmark Dataset ([entitize/Fakeddit](https://github.com/entitize/Fakeddit))
- **Licence:** Academic Research License
- **Size:** Balanced sample of TSV metadata & image URL references (~50k-100k target)
- **Download Date:** 2026-10-26
- **Purpose:** Text-image consistency and multimodal fake news classification.

## 3. FakeNewsNet
- **Source:** PolitiFact & GossipCop repositories
- **Licence:** MIT / Academic
- **Size:** Article texts and metadata (social features optional / re-hydration ready)
- **Download Date:** 2026-10-26
- **Purpose:** Source and social credibility feature modeling.

## 4. MediaEval Verifying Multimedia Use
- **Source:** MediaEval benchmarking initiative
- **Licence:** Academic Research Use
- **Size:** Image verification corpus with event tags and real/fake labels
- **Download Date:** 2026-10-26
- **Purpose:** Media reuse detection and out-of-context image checking (video was dropped 2026-11-02: future work).

## 5. Reference Datasets & Local Test Sets
- **`sources.csv`** (rebuilt 2026-11-02, `python -m src.ingest.source_ratings`): **1,082 rated domains** (was 6
  hand-typed rows). Every rating comes from a named public list, with one fixed number per category:
  Wikipedia "Reliable sources/Perennial sources" (RSP; editors' consensus per outlet: generally reliable 0.90,
  no consensus 0.60, generally unreliable 0.30, deprecated 0.10), Wikipedia "List of fake news websites" (0.10)
  and "List of satirical news websites" (0.10, satire), plus three short manual lists in the code: IFCN fact-checkers
  (0.90), official bodies (0.85) and 22 established Indian national outlets that RSP does not cover (0.75). Lowest
  rating wins when lists disagree; political lean is never rated. Wikipedia revision ids are in
  `reports/source_ratings_summary.json`; raw pages cached in `data/raw/source_ratings/`. Licence: CC BY-SA (cite).
  Effect: FakeNewsNet items with a rated domain 0.4% → 44%; source check test macro-F1 0.612 → 0.635.
  Limitation: RSP rates some outlets per topic (e.g. Fox News politics vs other news); the lowest rating is used.
- **`landmarks/<place>/`** (2026-11-02, `python -m src.ingest.download_landmarks`): ~20 reference photos each of 13
  Indian landmarks (India Gate, Jantar Mantar, Red Fort, Parliament House, Rashtrapati Bhavan, Qutb Minar, Gateway
  of India, CSMT, Taj Mahal, Charminar, Howrah Bridge, Victoria Memorial, Golden Temple) from **Wikimedia Commons**,
  each with licence and author in `sources.csv` in its folder (CC BY / CC BY-SA / public domain). Used only for the
  location check, never shown or redistributed. Why not Google: Google Maps / Street View / Places photos may not be
  stored or used to build a dataset (Maps Platform terms). Threshold calibrated on these photos:
  `--calibrate` → `reports/location_metrics.json`.
- **`india_test_set.csv`:** 30 Indian-context sentences (columns `id, text, verdict, modality, source`). **Placeholder, not
  a benchmark** (checked 2026-10-29): the rows are written by hand in `src/ingest/download_data.py`, not collected from
  fact-check articles. There are no article URLs or dates, the "image"/"video"/"audio" rows have no media file, several
  rows are fact-check headlines rather than the claim as it circulated, and some labels are wrong (the NASA "Diwali
  lights from space" photo is a known hoax but is labelled Supported). Do not report agreement numbers on it. A real
  set needs, per item: the claim as circulated, the fact-checker's verdict and article URL, the date, and the media file.

## 6. CLEF CheckThat! 2024 Task 1 (check-worthiness, English)
- **Source:** [gitlab.com/checkthat_lab/clef2024-checkthat-lab](https://gitlab.com/checkthat_lab/clef2024-checkthat-lab) `task1/data/CT24_checkworthy_english.zip` + `CT24_checkworthy_test_gold.zip`
- **Licence:** Research use; cite the CheckThat! 2024 overview paper
- **Size:** train 22,501 (5,413 Yes) · dev 1,032 · dev-test 318 · test gold 341 sentences
- **Download Date:** 2026-10-26 (`python -m src.ingest.download_checkthat`)
- **Purpose:** Channel 3A check-worthiness classifier; "No" sentences are the negative examples.

## 7. India-context data (Phase 8): test sets and evidence, not training data

All three via the ComplexDataLab collection on Hugging Face,
[`ComplexDataLab/Misinfo_Datasets`](https://huggingface.co/datasets/ComplexDataLab/Misinfo_Datasets) (Thibault et al.,
"A Guide to Misinformation Detection Data and Evaluation", arXiv 2411.05060), which re-hosts published datasets in one
schema with train/validation/test splits. The collection is tagged Apache-2.0 but lists no per-dataset licence: treat
each as **academic research use**, cite the original paper, and do not redistribute the files or deploy publicly with
them without checking the original terms. Download date 2026-10-29. Code: `python -m src.ingest.load_india`
(`src/ingest/load_india.py`) → `data/processed/india/*.parquet`, counts in `reports/india_data_summary.json`.

### 7a. IFND, Indian Fake News Dataset (test set)
- **Paper:** Sharma & Garg, "IFND: a benchmark dataset for fake news detection", *Complex & Intelligent Systems* 2021,
  doi:10.1007/s40747-021-00552-1. Original release on Kaggle.
- **Raw:** 56,698 English headlines about India, 2013–2021: 37,785 true (TribuneIndia, The Statesman, ThePrint, Indian
  Express …) and 18,913 fake. Fields used: headline, label, category, source site, image URL. **No dates.**
- **Problems found (2026-10-29), and the cleaning:**
  - 11,321 fakes (60%) are *generated* by the authors (source field `AUGMENT`: true headlines with words swapped) → dropped.
  - Of the remaining fakes, about half are fact-checkers' headlines ("Fact Check: viral video of X is fake"): a true
    statement about a debunk, not the claim → rows with debunk wording dropped from **both** labels (5,481 fake, 568 true).
  - 262 exact duplicates dropped.
  - Shortcuts that remain, so never train a model on it naively: category ("TERROR" is always true), source site
    (Tribune/Statesman/Indian Express always true), headline length (fakes longer).
- **Clean:** 37,101 true, 1,965 fake (train 26,070 / 1,395 · val 3,654 / 183 · test 7,377 / 387).
- **Use:** India test set: all val + test fakes (570) and as many sampled true headlines.

### 7b. FakeCovid, Indian English part (evidence + fake-only test)
- **Paper:** Shahi & Nandini, "FakeCovid – A Multilingual Cross-domain Fact Check News Dataset for COVID-19", ICWSM
  workshops 2020 (arXiv 2006.11343). Fact-checks collected from the Poynter CoronaVirusFacts database. This copy matches
  the July 2020 release (7,623 articles; Zenodo 3965871). Licence: CC BY-NC 4.0 (research, non-commercial).
- **India English:** 748 fact-checks, Jan–Jun 2020, by Factly (189), BOOM (184), The Quint (96), India Today (80),
  Factcrescendo (57), Vishvas News (52), AFP (50), Digiteye (25) … Claim as it circulated, verdict, date, fact-check URL,
  article title and text. 742 False/Misleading, 6 partly false: fake-only.
- **Use:** train + val (602) → evidence passages; test (146) → fake-only India test (dates filter later evidence).

### 7c. MultiClaim (evidence)
- **Paper:** Pikuliak et al., "Multilingual Previously Fact-Checked Claim Retrieval", EMNLP 2023.
- **Used:** 22,802 fact-checked social-media claims with an English translation and a verdict (mostly AFP, many
  countries; 1,162 mention India). False 16,045 / partly false or missing context 6,757. No dates, no country field.
- **Use:** train + val (18,285) → evidence passages (`models/retrieval_index_india`, `python -m src.retrieval.search --india`).

### 7d. India 2023–2025 test set (real, dated, sourced; built 2026-11-02)
- **Code:** `python -m src.ingest.live_testset` → `data/processed/india/live_testset.parquet`, summary
  `reports/live_testset_summary.json`. Test only; nothing is trained on it.
- **False (300):** claims reviewed by Indian fact-checkers, 60 each from BOOM, Factly, India Today, The Quint, Vishvas
  News (Alt News had no reviews in the window), via the Google Fact Check Tools API (the fact-checkers' own
  ClaimReview markup). Ratings False 260, Misleading 22, Mostly false 18. Near-duplicates (same viral claim checked
  twice) removed. Text = the claim as the fact-checker quotes it.
- **True (300):** India-related events from Wikipedia's "Current events" portal, 2023–2025: one-sentence summaries of
  reported events, each citing a news outlet (AP 45, Reuters 33, Al Jazeera 30, The Hindu 19, DW 19, BBC 15 …).
- **Dates:** 2023 199 · 2024 187 · 2025 214, i.e. after every training set (2013–2022). Each item keeps its date, so
  evidence written later is filtered out; the live lookup uses only fact-checks published before the claim.
- **Known bias:** the halves are written differently (fact-checkers paraphrase posts, median 14 words; Wikipedia
  summaries, median 24 words, neutral register). A model that reads wording could separate them without knowing
  the facts, so per-class rates are reported next to ROC-AUC, and the matched set (7d-3) removes the difference.

### 7d-2. India development set (built 2026-11-03), for tuning only
- **Code:** `python -m src.ingest.live_testset --dev` → `data/processed/india/live_devset.parquet`, summary
  `reports/live_devset_summary.json`.
- Same sources, window (2023–2025) and recipe as 7d, with every test item and every near-copy of one (TF-IDF cosine
  ≥ 0.8, or same URL) removed: 300 false (60 per fact-checker) + 227 true (the Wikipedia pool left after the test set).
- **Why:** thresholds (claim matching) are chosen on this set; the test set is scored once afterwards, so its numbers
  are not tuned to it.

### 7d-3. Matched India test set (built 2026-11-03): both halves written by fact-checkers
- **Code:** `python -m src.ingest.live_testset --matched` → `data/processed/india/matched_testset.parquet`, summary
  `reports/matched_testset_summary.json`.
- **True (169):** claims Indian fact-checkers rated exactly "True" / "Mostly true" (Google Fact Check API, English,
  full history). **False (169):** for each true claim, one claim the *same* fact-checker rated False in the *same
  year*. Factly 92 pairs, Alt News 33, BOOM 27, YouTurn 9, India Today 6, FactChecker 2; mostly 2016–2019.
- **Why:** removes the style bias of 7d (its true half is Wikipedia prose). Median length is 13 words in both halves.
- **Limits:** small; skewed to Factly's checks of government statistics; older than the training data's end.
- Count behind it (2026-11-03): Indian fact-checkers published 467 "True" vs 51,239 "False/misleading" verdicts
  (277 vs 32,338 in English); "True" is rare because fact-checkers pick suspicious claims.

### 7e. Getting the remaining Indian datasets, step by step
All of these need a person (an account, a form, a promise about use). Put files in
`data/raw/<name>/` and write a loader for them (same pattern as `load_india.py`).

1. **FactDrill** (Singhal et al., IIIT-Delhi PreCog; 22,435 fact-checks from 13 Indian languages incl. English, with
   article text, verdict, date, images). The best Indian evidence and training source for this project.
   **Update 2026-11-03: the record is open access** (no account, no form). 32 `.xlsx` files, one per fact-checker
   and language; download the 11 `*_english.xlsx` files from `https://zenodo.org/records/5854856/files/<name>.xlsx?download=1`
   into `data/raw/factdrill/` (needs `pip install openpyxl`). **Fake-only**: the authors removed articles whose claim
   was found true (paper, "Data Annotation", Task 1). Use: evidence passages + a dated fake-only test split; no loader yet.
2. **FakeNewsIndia** (Dhawan et al., *Computer Communications* 2022; 4,803 fake-news incidents in India, 2016–2021,
   with fact-check links): request through the Google Form linked in the paper / its GitHub page. Same purpose text.
3. **IndicClaimBuster** (check-worthy claims in English, Hindi, Bengali and code-mixed text): request form on the
   project page. Useful only if check-worthiness on Indian posts is to be improved.
4. **CONSTRAINT 2021 COVID fake news (English)**: register on CodaLab for the CONSTRAINT@AAAI2021 shared task (or
   use the organisers' public GitHub copy if it is still online). Not India-only; low priority.
5. **Google Fact Check Tools API**: done (key in `.env`). It supplies live evidence and the 2023–2025 test set.
6. **Hindi / regional languages**: future work (translate-then-check with IndicTrans2; Hindi evidence from BOOM Hindi
   and Vishvas News). Nothing to request now.

### Not used, and why
- **FactDrill** (Singhal et al., ICWSM 2022; 22,435 Indian fact-checks, 13 languages): open on Zenodo since at least
  2026-11-03 but **fake-only** (true claims were removed by the authors), so not a training set; planned as evidence.
  **FakeNewsIndia** (Dhawan et al. 2022; 4,803 incidents, fake-only) and **IndicClaimBuster** (2025): request form only.
- **Indian real + fake datasets checked 2026-11-03 and not used for training:** FakeRealIndian (real side from Times
  of India), BharatGather (LLM-generated items), BharatFakeNewsKosh (real side from newspapers), CONSTRAINT 2021
  (COVID-only; real side = official tweets). In each, the true and fake halves come from different sources, so a
  model learns source style, not truth.
- **CONSTRAINT 2021 COVID** (Patwa et al.): official files need CodaLab registration; COVID-only and not India-specific.
- **X-FACT:** Indian-language claims; the system is English-only.
