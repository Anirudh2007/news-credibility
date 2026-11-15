# News Credibility and Claim Verification System

Breaks a news item (text, URL, image) into claims, checks each claim against evidence (an offline index plus live
published fact-checks), and returns a 0–100 credibility score with plain-language reasons and quoted evidence.
The score comes only from checks that look at evidence (claims, image, source, dates): how a text is worded is never
scored. English only; video and Indian languages are future work.

Status: phases 0–7 built (channels, meta-learner, explanations, API, React web app). Phase 8: evaluation, Docker and
India test sets done; deployment and the final report remain.

| Phase | Stage | Code |
| --- | --- | --- |
| 1 | Data collection | `src/ingest/` |
| 2 | Cleaning, label unification, de-duplication, topic-based splits | `src/preprocess/` |
| 3A | Claim extraction (check-worthiness, DeBERTa) | `src/claims/` |
| 3B | Evidence retrieval (BM25 + dense vectors, cross-encoder re-ranking) | `src/retrieval/` |
| 3C | Claim verification (DeBERTa, AVeriTeC) | `src/verify/` |
| 3D | Caption-image consistency (CLIP) | `src/media/text_image.py` |
| 3E | Media reuse and location check (perceptual hash, CLIP) | `src/media/reuse.py` |
| 3F | Source rating | `src/source/credibility.py` |
| 3G | Date consistency (rule-based) | `src/temporal/` |
| 4 | Meta-learner (LightGBM + isotonic calibration) | `src/fusion/` |
| 5 | Explanations (SHAP points, templates, optional LLM wording) | `src/explain/` |
| 6 | FastAPI backend, SQLite history and feedback | `api/` |
| 7 | Web app (React, Vite, Tailwind) | `frontend/` |
| 8 | Evaluation, robustness, Docker | `src/fusion/evaluate.py`, `Dockerfile` |

## Setup

Python 3.11 (3.13+ is not supported by the pinned packages).

```bash
py -3.11 -m venv .venv          # Windows; use python3.11 on Linux/macOS
.venv\Scripts\activate
pip install -r requirements.txt -c requirements_locked.txt   # the versions the tests pass with
python -m spacy download en_core_web_sm
copy .env.example .env          # then fill in the keys you have (all optional)
```

### Trained models and reference data

`data/` and `models/` are git-ignored (several GB). To run the app without retraining, copy them from the team's
shared copy into this folder:

```
models/           meta_lgbm.txt, meta_calibrator.joblib, source_clf.joblib, text_image_clf.joblib,
                  checkworthy-deberta/, verifier-deberta/, retrieval_index/, media_index/
data/reference/   sources.csv (outlet ratings), landmarks/ (reference photos for the location check)
```

A missing model is skipped, not fatal: `http://localhost:8000/health` lists what loaded. To build everything from
scratch instead, create the folders and follow the rebuild order below:

```bash
mkdir data\raw data\interim data\processed data\reference models
```

### API keys (`.env`)

- `GOOGLE_FACTCHECK_API_KEY`: live lookup of published fact-checks (free: Google Cloud console, enable the
  "Fact Check Tools API", create an API key). Without it most of the built-in examples come back "Unverified",
  because the offline evidence index holds few Indian claims.
- `NEWSCRED_CONTACT`: a contact URL or email; turns on Wikipedia evidence (Wikimedia's rate limits need it).
- `TAVILY_API_KEY`: news search of trusted outlets (free plan: 1,000 searches a month, no card,
  https://app.tavily.com). It confirms true claims from news reports, and it is what lets silence count as evidence
  (next section). Without it, big claims that nothing else covers stay "Unverified" and the report says why.

The app reads these three from `.env` (this folder or the one above). Every other setting in `.env.example` is an
environment variable: Docker Compose reads it from `.env`, but for a local run set it in the shell first, e.g.
`$env:NEWSCRED_LIVE_FACTCHECK="0"` (PowerShell) to switch live lookups off, or `$env:EXPLAIN_ENABLED="0"` to skip
the optional LLM wording of explanations (templates are used whenever no LLM endpoint answers). Never commit `.env`.

## When silence is evidence (expected news coverage)

"Absence of evidence is not evidence of absence", unless evidence would be expected. If "RBI bans Rs 500 notes
from tomorrow" or "Shah Rukh Khan died this morning" were true, every trusted outlet would report it within hours,
so finding no report is strong evidence against it; for "a man in a Bihar village rescued a dog" it means nothing.
`src/retrieval/coverage.py` decides when the system may use silence:

1. **Expected coverage** (transparent rules): an event the news always covers (death, resignation, arrest or court
   ruling, ban, payment for everyone, official announcement, war or disaster, election result) about a well-known
   subject (a national institution, or a name whose Wikipedia article had at least 3,000 views in 30 days), or at
   national scale ("every citizen"), or an extraordinary claim ("aliens", "cures cancer"); never a local story.
2. **We listened**: the news search of ~240 trusted outlets ran and answered (a timeout or a used-up quota is not
   silence), in the claim's window (7 days up to a dated claim; all reports so far for an undated one). A claim
   that heard nothing is searched once more with its keywords before its silence is believed.
3. **Nothing else spoke**: no relevant trusted report among everything the search returned (undated reports count;
   fact-check / debunk headlines do not, since they exist because a claim is false), and no other claim evidence
   counted (a fact-check that refutes or a passage that confirms already decides the claim).

Then Bayes' rule: posterior odds = prior odds x likelihood ratio, LR = P(silence | true) / P(silence | false). With
nothing else known the prior is even odds. **Measured (7 Nov, `python -m src.fusion.india_tune absence-hoax`, the
India sets plus 60 invented-event hoaxes, decision rule written first):** silence is evidence (LR 0.27 on dev,
0.42 on test, both 95% intervals below 1). The search also misses about 1 in 10 real big events, so the rule has a
price: in the deployed setting true claims rated Low would rise 4.4% → 5.7% on dev (target ≤ 5%) and 8.7% → 12.0%
on test (target < 10%), for invented-event hoaxes rated Low 47% → 70% (test). Both targets fail, so the rule is
**off**: the report shows "No trusted outlet reported this" without scoring it (`python -m src.fusion.india_tune absence-hoax`).
`NEWSCRED_ABSENCE_LR=0.27` switches it on for a demonstration. An earlier measurement (6 Nov, LR 1.26) had three flaws, explained in
`src/retrieval/coverage.py`.

Two related rules: Wikipedia text can confirm a dated claim only if it names the claim's year (the 2016
demonetisation article used to "confirm" "Rs 500 notes banned from tomorrow": 67/100), and every passage a trusted
live source returns is kept in an **evidence memory** (`src/retrieval/evidence_memory.py`) and searched again for
later claims, so the system improves with use without storing anything a user typed (no poisoning through feedback).

## Which evidence a claim is verified on

Each claim gets two separate lists, each ranked by the cross-encoder among its own kind: the 5 best passages that
can **refute** (the offline index and published fact-checks) and the 5 best that may only **confirm** (Wikipedia as
of the claim date, trusted news reports). `evidence_budget` in `src/verify/model.py` makes the cut. Until v0.10.1 both
kinds shared one top 5, and for well-known myths the Wikipedia articles *about* the myth ranked highest: they could not
refute, so they did not count, and they pushed the refutations out ("The earth is flat." came back Unverified).
The two lists always contain the old shared top 5, so the rule only adds passages.

## Run the app

```bash
python -m uvicorn api.main:app              # terminal 1, API docs at http://localhost:8000/docs
cd frontend                                 # terminal 2 (Node 22 or newer)
npm install                                 # first time only
npm run dev                                 # http://localhost:5173, forwards /api to the API
```

The first analysis loads every model, so it is slow; later ones take seconds.

The web app shows a verdict in words, a two-sentence bottom line, three reasoning cards and the claims with
their sources. `frontend/src/translate/` turns an API report into that wording; the score, the points per check
and the other model numbers appear only under "View technical verification details". `npm test` runs the
frontend tests, `npm run build` type-checks and builds `frontend/dist/`.

## Rebuild order (after changing data or a channel model)

Each step reads the outputs of the ones above it.

```bash
python -m src.ingest.download_data          # Phase 1: AVeriTeC + small fallback samples (never overwrites)
python -m src.ingest.download_checkthat     # Phase 1: CheckThat! 2024 check-worthiness sentences
python -m src.ingest.download_mediaeval     # Phase 1: MediaEval images (clone the corpus into data/raw/mediaeval/ first)
                                            # Fakeddit and FakeNewsNet full releases: see DATASETS.md
python -m src.preprocess.build_dataset      # Phase 2: data/processed/*.parquet
python -m src.claims.prepare_data --train   # 3A check-worthiness (GPU)
python -m src.retrieval.search              # 3B evidence index
python -m src.verify.model                  # 3C verifier (GPU)
python -m src.media.text_image              # 3D text-image classifier
python -m src.ingest.source_ratings         # 3F data/reference/sources.csv from public lists (1,082 domains)
python -m src.source.credibility            # 3F source classifier
python -m src.ingest.download_landmarks     # 3E landmark reference photos (Wikimedia Commons); --calibrate
python -m src.fusion.train build            # Phase 4 feature table v3 (~80 min, resumable; live lookups off)
python -m src.fusion.train fit              # Phase 4 meta-learner + reports/meta_metrics.json
python -m pytest -q                         # tests
```

The API caches results per input and per *model fingerprint* (sizes and times of the trained files),
so after retraining, old cached reports are not reused.

## Evaluation (Phase 8)

Run after the rebuild; each writes JSON to `reports/`. All system-level numbers use the meta-learner's test rows.

```bash
python -m src.claims.prepare_data --eval        # 3A P/R/F1 -> checkworthiness_metrics.json
python -m src.retrieval.search --eval           # 3B Recall@5, MRR on val claims -> retrieval_metrics.json
python -m src.verify.model --eval               # 3C macro-F1 + confusion matrix -> verifier_metrics.json
python -m src.fusion.evaluate ablate            # refit without each channel -> meta_ablation/retrain_without_channel.json
python -m src.fusion.evaluate text-baseline --bert   # text-only baselines (BERT fine-tune: GPU, ~15 min)
python -m src.fusion.evaluate errors            # worst mistakes per dataset -> error_analysis.csv
python -m src.fusion.evaluate explain-sample    # 20 explanations to rate -> explanation_rating_sheet.csv
python -m src.fusion.evaluate figures           # reports/figures/*.png from the JSON files
python -m src.fusion.evaluate robustness --tag after  # gibberish, nonsense, injection through the API
python -m src.fusion.evaluate guard             # input guard: false refusals on real text, noise caught
python -m src.fusion.evaluate real-fake         # well-known true/false claims + real fact-checks via the API
python -m src.fusion.evaluate evidence-box      # evidence box: shown how often, right how often -> evidence_box_metrics.json
```

India-context data (test sets + Indian fact-checks as evidence; `DATASETS.md` §7):

```bash
python -m src.ingest.load_india                 # download + clean IFND, FakeCovid India, MultiClaim
python -m src.retrieval.search --india          # evidence index with Indian/other fact-checks
python -m src.fusion.evaluate india             # India test sets, both indexes, text baselines
python -m src.ingest.live_testset               # 600 real 2023-2025 Indian claims (needs the Google key)
python -m src.fusion.evaluate india-live        # all India sets: evidence indexes x live lookup
```

How to request the Indian datasets that need a form (FactDrill, FakeNewsIndia, IndicClaimBuster): `DATASETS.md` §7e.

To run the app with the Indian evidence index: `$env:NEWSCRED_RETRIEVAL_DIR="models/retrieval_index_india"`
(PowerShell; `set NEWSCRED_RETRIEVAL_DIR=...` in cmd) before starting the API. Default: `models/retrieval_index`.

Input that is not readable English (keyboard mashing, symbols, other languages) is refused with a reason
(`src/ingest/guard.py`); input with nothing checkable gets no score ("Unverified"). Measured with `python -m src.fusion.evaluate robustness`.

## Docker

Two images: the API, and the web app (the built React files served by Caddy, which also forwards the API routes
the app uses). `models/` and `data/reference/` are mounted from the host, so fill them first (rebuild order).
Needs Docker Desktop (Windows: WSL 2 backend) running. API keys are read from `.env` at start, never copied into the image.

```bash
docker compose up --build                       # API http://localhost:8000/docs, web app http://localhost:8501
docker compose run --rm api python -m pytest -q # test suite inside the image
```

The image uses CPU-only PyTorch and the exact versions in `requirements_locked.txt` (regenerate it with `pip freeze`
after changing packages). First start downloads ~1 GB of base models into a cache volume.

### Share a public demo link

With the containers running, a Cloudflare quick tunnel (free, no account; install `cloudflared`) gives the UI a public
https address while this machine stays on. Models and data never leave the machine.

```bash
docker compose up -d                              # wait until `docker compose ps` shows the api as healthy
cloudflared tunnel --url http://localhost:8501    # prints https://<random-words>.trycloudflare.com
```

The address changes on every run and has no uptime guarantee. Anyone with it can use the app. Through the web app
only four API routes are reachable (run a check, open or delete one check by its id, send feedback, health); the
history list and `/docs` are not.

### Host on an Azure VM (fixed https address)

`configs/azure/`: `create_vm.ps1` makes an Ubuntu VM (Standard_B2as_v2, 2 vCPU, 8 GiB; Docker installed on first
boot; SSH only from your address), `deploy.ps1` copies the code, `models/`, `data/reference/` and `.env` and starts
the same compose stack with HTTPS on the web app container (`docker-compose.azure.yml`). Needs the Azure CLI (`az login`) and an SSH
key at `~/.ssh/newscred_azure`.

```powershell
powershell -ExecutionPolicy Bypass -File configs\azure\create_vm.ps1   # prints https name, uses Azure credit
powershell -ExecutionPolicy Bypass -File configs\azure\deploy.ps1      # re-run after code changes
az vm deallocate -g newscred-rg -n newscred-vm                        # stop paying for compute; az vm start to resume
``` Cloud hosting runs the same image (Hugging Face Docker Spaces need a paid plan; free tiers with ~1 GB RAM are
too small for the models).

## Layout

| Path | Purpose |
| --- | --- |
| `data/` | raw, interim, processed datasets and reference files (not in git) |
| `notebooks/` | EDA and experiments (not in git) |
| `src/` | pipeline code, one folder per stage |
| `models/` | saved checkpoints (not in git) |
| `api/` | FastAPI backend |
| `frontend/` | web app: React + Vite + Tailwind (`src/`), its tests, its Dockerfile and Caddyfile |
| `tests/` | pytest tests |
| `configs/azure/` | Azure VM scripts: create the VM, deploy, HTTPS (settings are environment variables, `.env.example`) |
| `reports/` | output of the evaluation scripts: JSON results and figures (not in git) |
