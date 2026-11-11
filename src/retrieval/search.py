"""Channel 3B: hybrid evidence retrieval (Phase 3B).

Two searchers over the same passage collection, merged:

- Dense (sentence-transformers + FAISS): each passage becomes a vector; a claim
  finds passages whose vectors point the same way (cosine similarity). Catches
  paraphrases: "jobless rate" matches "unemployment".
- Sparse (BM25): classic keyword ranking; rewards rare shared words. Catches
  exact names, numbers and IDs that embeddings blur ("Section 144", "Rs 2,000").

Merge: Reciprocal Rank Fusion (RRF), score = sum over searchers of 1 / (60 + rank).
Ranks, not raw scores, are combined because cosine (0..1) and BM25 (0..~30) are
on different scales. A passage ranked high by both wins.

Then an optional cross-encoder re-ranks the merged candidates by reading claim
and passage together (slower, more accurate) and the top 5 are returned.

FAISS is optional at import time: if it cannot load, an exact NumPy inner-product
search gives identical results (FAISS `IndexFlatIP` is the same computation, faster).
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from rank_bm25 import BM25Okapi

try:
    import faiss

    HAVE_FAISS = True
except Exception:  # ImportError, or OS policy blocking its DLL
    faiss = None
    HAVE_FAISS = False

DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim, fast on CPU
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RRF_K = 60  # standard constant from Cormack et al. (2009); damps the effect of top ranks
TOP_K = 5
CANDIDATES = 50  # per searcher, before fusion

Encoder = Callable[[Sequence[str]], np.ndarray]  # texts -> (n, d) float32


@dataclass
class Passage:
    id: str
    text: str
    source_url: str | None = None
    domain: str | None = None
    date: str | None = None  # ISO; used to drop evidence published after the claim


@dataclass
class Hit:
    passage: Passage
    score: float  # fused (or re-ranker) score; higher is better
    dense_rank: int | None  # 1-based; None if not in that searcher's candidates
    bm25_rank: int | None


# --- Building blocks ----------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "a an the and or of to in on at for by with from is are was were be been it its this that as".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP]


def sentence_transformer_encoder(model_name: str = DENSE_MODEL, batch_size: int = 64) -> Encoder:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)

    def encode(texts: Sequence[str]) -> np.ndarray:
        return model.encode(list(texts), batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True).astype(
            np.float32
        )

    return encode


def _normalise(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


class _DenseIndex:
    """Inner-product index over L2-normalised vectors (inner product = cosine)."""

    def __init__(self, vectors: np.ndarray):
        self.vectors = _normalise(vectors)
        self.index = None
        if HAVE_FAISS:
            self.index = faiss.IndexFlatIP(self.vectors.shape[1])
            self.index.add(self.vectors)

    def search(self, q: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        q = _normalise(q.reshape(1, -1))
        k = min(k, len(self.vectors))
        if self.index is not None:
            scores, ids = self.index.search(q, k)
            return scores[0], ids[0]
        sims = self.vectors @ q[0]
        ids = np.argsort(-sims, kind="stable")[:k]
        return sims[ids], ids


def rrf_fuse(rankings: Sequence[Sequence[int]], k: int = RRF_K) -> dict[int, float]:
    """Reciprocal Rank Fusion. Each ranking is a list of doc ids, best first."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            fused[doc] = fused.get(doc, 0.0) + 1.0 / (k + rank)
    return fused


# --- Hybrid searcher ----------------------------------------------------------

class HybridSearcher:
    def __init__(self, passages: Sequence[Passage], encoder: Encoder | None = None, vectors: np.ndarray | None = None):
        if not passages:
            raise ValueError("HybridSearcher needs at least one passage")
        self.passages = list(passages)
        self.encoder = encoder or sentence_transformer_encoder()
        if vectors is None:
            vectors = self.encoder([p.text for p in self.passages])
        self.dense = _DenseIndex(vectors)
        self.bm25 = BM25Okapi([tokenize(p.text) or ["_"] for p in self.passages])
        self._qmemo: dict[str, np.ndarray] = {}  # query -> vector; filled by prefetch_queries()

    def prefetch_queries(self, queries: Sequence[str]) -> None:
        """Embed many queries in one encoder call; search() then reads the cache."""
        misses = [q for q in dict.fromkeys(queries) if q not in self._qmemo]
        if misses:
            self._qmemo.update(zip(misses, self.encoder(misses)))

    def clear_memo(self) -> None:
        self._qmemo.clear()

    def candidates(self, query: str, candidates: int = CANDIDATES, before: str | None = None):
        """Fused dense + BM25 candidates, before re-ranking.
        Returns (order, fused scores, dense ranking, bm25 ranking)."""
        qv = self._qmemo[query] if query in self._qmemo else self.encoder([query])[0]
        _, dense_ids = self.dense.search(qv, candidates)
        dense_ranking = [int(i) for i in dense_ids if i >= 0]

        bm25_scores = self.bm25.get_scores(tokenize(query))
        bm25_order = np.argsort(-bm25_scores, kind="stable")[:candidates]
        bm25_ranking = [int(i) for i in bm25_order if bm25_scores[i] > 0]

        fused = rrf_fuse([dense_ranking, bm25_ranking])
        if before:
            fused = {i: s for i, s in fused.items() if not self.passages[i].date or self.passages[i].date <= before}
        order = sorted(fused, key=lambda i: (-fused[i], i))
        return order, fused, dense_ranking, bm25_ranking

    def search(
        self,
        query: str,
        k: int = TOP_K,
        candidates: int = CANDIDATES,
        before: str | None = None,
        reranker: Callable[[str, list[str]], Sequence[float]] | None = None,
    ) -> list[Hit]:
        """Top-`k` passages for `query`.

        before: ISO date; passages dated after it are dropped. Stops evidence
            leakage, e.g. a fact-check written after the claim answering it.
            Undated passages are kept.
        reranker: (query, texts) -> scores, e.g. `cross_encoder_reranker()`.
        """
        order, fused, dense_ranking, bm25_ranking = self.candidates(query, candidates, before)
        dense_pos = {d: r for r, d in enumerate(dense_ranking, start=1)}
        bm25_pos = {d: r for r, d in enumerate(bm25_ranking, start=1)}
        scores = fused
        if reranker is not None and order:
            order = order[:candidates]
            rr = reranker(query, [self.passages[i].text for i in order])
            scores = dict(zip(order, map(float, rr)))
            order = sorted(order, key=lambda i: (-scores[i], i))

        return [
            Hit(self.passages[i], float(scores[i]), dense_pos.get(i), bm25_pos.get(i)) for i in order[:k]
        ]

    # --- persistence ---
    def save(self, directory: str | Path) -> None:
        """Saves passages (JSONL) and vectors (.npy). BM25 is rebuilt on load
        (fast), which avoids pickling."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "vectors.npy", self.dense.vectors)
        with open(d / "passages.jsonl", "w", encoding="utf-8") as f:
            for p in self.passages:
                f.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, directory: str | Path, encoder: Encoder | None = None) -> "HybridSearcher":
        d = Path(directory)
        with open(d / "passages.jsonl", encoding="utf-8") as f:
            passages = [Passage(**json.loads(line)) for line in f]
        return cls(passages, encoder=encoder, vectors=np.load(d / "vectors.npy"))


class CrossEncoderReranker:
    """Callable (query, texts) -> scores, with a (query, text) cache that
    prefetch() fills in large batches across many queries."""

    def __init__(self, model_name: str = RERANK_MODEL, batch_size: int = 64, model=None):
        if model is None:
            from sentence_transformers import CrossEncoder

            model = CrossEncoder(model_name)
        self.model = model
        self.batch_size = batch_size
        self._memo: dict[tuple[str, str], float] = {}

    def _predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        return [float(s) for s in self.model.predict(pairs, batch_size=self.batch_size)] if pairs else []

    def __call__(self, query: str, texts: Sequence[str]) -> list[float]:
        pairs = [(query, t) for t in texts]
        misses = [p for p in dict.fromkeys(pairs) if p not in self._memo]
        fresh = dict(zip(misses, self._predict(misses)))
        return [self._memo[p] if p in self._memo else fresh[p] for p in pairs]

    def prefetch(self, pairs: Sequence[tuple[str, str]]) -> None:
        misses = [p for p in dict.fromkeys(pairs) if p not in self._memo]
        self._memo.update(zip(misses, self._predict(misses)))

    def clear_memo(self) -> None:
        self._memo.clear()


def cross_encoder_reranker(model_name: str = RERANK_MODEL) -> CrossEncoderReranker:
    return CrossEncoderReranker(model_name)


# --- Passage sources ----------------------------------------------------------

def passages_from_averitec(df) -> list[Passage]:
    """Evidence passages from AVeriTeC question-answer pairs (`questions_json`
    column of `data/processed/averitec.parquet`). One passage per answer, with
    its source URL and domain. Use only the train split to build an index you
    evaluate on val/test; otherwise each test claim finds its own gold answers.

    Text format "Q: ... A: ..." is the verifier's training format (`qa_text`).
    Skipped: unanswerable answers, and yes/no answers without an explanation.
    A bare "Did RBI make this announcement? No." was written about one specific
    claim; retrieved for a different claim about the RBI it reads as a refutation
    of that claim too (the "RBI Rs 1000 notes" error).
    """
    from src.ingest.load_raw import domain_of
    from src.verify.model import qa_text

    out: list[Passage] = []
    for item_id, qjson in zip(df["item_id"], df["questions_json"]):
        for qi, q in enumerate(json.loads(qjson or "[]")):
            for ai, a in enumerate(q.get("answers") or []):
                if a.get("answer_type") == "Unanswerable":
                    continue
                if a.get("answer_type") == "Boolean" and not (a.get("boolean_explanation") or "").strip():
                    continue
                text = qa_text(q.get("question"), a)
                if not text:
                    continue
                url = a.get("source_url")
                out.append(Passage(id=f"{item_id}_q{qi}_a{ai}", text=text, source_url=url, domain=domain_of(url)))
    return out


_VERDICT_WORD = {0.0: "No", 0.5: "Partly", 1.0: "Yes"}


def passages_from_factchecks(df, max_claim_chars: int = 300) -> list[Passage]:
    """One passage per published fact-check (data/processed/india/*.parquet), in the verifier's
    "Q: ... A: ..." format, e.g.
        Q: Is it true that <claim as it circulated>? A: No. factly.in rated this claim False. <title>
    The answer opens with Yes/No/Partly like AVeriTeC's yes/no answers, and names who checked it.
    Rows without a verdict are skipped. Dated passages are dropped by the pipeline's `before=`
    filter for claims made earlier, so a later fact-check never decides an earlier claim."""
    out: list[Passage] = []
    for row in df.itertuples(index=False):
        if row.credible not in _VERDICT_WORD:
            continue
        claim = " ".join(str(row.text).split())[:max_claim_chars].rstrip(" ?.")
        who = row.fact_checker or "A fact-checker"
        text = f"Q: Is it true that {claim}? A: {_VERDICT_WORD[row.credible]}. {who} rated this claim {str(row.label_raw).strip()}."
        title = getattr(row, "title", None)
        if isinstance(title, str) and title.strip():
            text += f" {title.strip()[:200]}"
        out.append(Passage(id=row.item_id, text=text, source_url=row.fact_check_url, domain=row.fact_checker,
                           date=row.date if isinstance(row.date, str) else None))
    return out


# --- Evaluation (Recall@5, MRR) -----------------------------------------------

def rank_metrics(rankings: Sequence[Sequence[str]], gold: Sequence[set[str]], k: int = TOP_K) -> dict:
    """Ranking quality over queries, given each query's ranked passage ids and its gold ids.

    hit@k     share of queries with at least one gold passage in the top k
    recall@k  share of each query's gold passages found in the top k, averaged
    mrr       mean of 1 / rank of the first gold passage (0 if none was retrieved)
    """
    hits, recalls, rr = [], [], []
    for ranked, g in zip(rankings, gold):
        top = list(ranked[:k])
        found = [p for p in top if p in g]
        hits.append(bool(found))
        recalls.append(len(found) / len(g))
        first = next((r for r, p in enumerate(ranked, start=1) if p in g), None)
        rr.append(1.0 / first if first else 0.0)
    return {"queries": len(rr), f"hit@{k}": float(np.mean(hits)), f"recall@{k}": float(np.mean(recalls)),
            "mrr": float(np.mean(rr))}


def evaluate_retrieval(df, encoder: Encoder | None = None, reranker=None, k: int = TOP_K,
                       split: str = "val") -> dict:
    """Recall@k and MRR on AVeriTeC `split` claims.

    The index holds train + `split` passages, so each evaluated claim's own QA answers
    (its gold evidence) are findable among thousands of others. The pipeline's index
    holds train passages only: this measures the retriever, not the deployed index.
    Gold = the claim's own passages; another claim's passage on the same story counts
    as a miss, so the numbers are a lower bound. Compares each stage: dense only,
    BM25 only, hybrid (RRF) and hybrid + cross-encoder re-rank (what the pipeline uses).
    """
    pool = df[df["split"].isin(["train", split])]
    passages = passages_from_averitec(pool)
    searcher = HybridSearcher(passages, encoder=encoder)
    ids = [p.id for p in searcher.passages]
    queries = df[df["split"] == split]
    gold_of = {}
    for p in passages:
        gold_of.setdefault(p.id.rsplit("_q", 1)[0], set()).add(p.id)
    rows = [(t, gold_of[str(i)]) for i, t in zip(queries["item_id"], queries["text"]) if str(i) in gold_of]
    texts, gold = [t for t, _ in rows], [g for _, g in rows]
    searcher.prefetch_queries(texts)

    stages: dict[str, list[list[str]]] = {"dense": [], "bm25": [], "hybrid_rrf": []}
    fused_orders = []
    for q in texts:
        order, _, dense_ranking, bm25_ranking = searcher.candidates(q, CANDIDATES)
        stages["dense"].append([ids[i] for i in dense_ranking])
        stages["bm25"].append([ids[i] for i in bm25_ranking])
        stages["hybrid_rrf"].append([ids[i] for i in order])
        fused_orders.append(order[:CANDIDATES])
    if reranker is not None:
        if hasattr(reranker, "prefetch"):
            reranker.prefetch([(q, searcher.passages[i].text) for q, order in zip(texts, fused_orders) for i in order])
        stages["hybrid_rerank"] = [[h.passage.id for h in searcher.search(q, k=CANDIDATES, reranker=reranker)]
                                   for q in texts]
    return {
        "split": split, "index_passages": len(passages), "queries_without_gold": int(len(queries) - len(rows)),
        "gold_per_query_mean": float(np.mean([len(g) for g in gold])) if gold else math.nan,
        "candidates_per_searcher": CANDIDATES,
        "stages": {name: rank_metrics(r, gold, k) for name, r in stages.items()},
    }


def main() -> None:
    """python -m src.retrieval.search : build the evidence index used by the pipeline
    from AVeriTeC *train* QA evidence only (val/test claims must not find their own
    gold answers). Saved to models/retrieval_index/.
    --eval  : Recall@5 / MRR on val claims instead -> reports/retrieval_metrics.json.
    --india : also add published fact-checks (FakeCovid India + MultiClaim, train + val splits only;
              python -m src.ingest.load_india first) -> models/retrieval_index_india/. The app uses it
              when NEWSCRED_RETRIEVAL_DIR points there."""
    import argparse

    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="store_true", help="measure Recall@5 / MRR on val claims; no index is saved")
    ap.add_argument("--india", action="store_true", help="add Indian and other published fact-checks to the index")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    df = pd.read_parquet(root / "data" / "processed" / "averitec.parquet")
    if args.eval:
        report = evaluate_retrieval(df, reranker=cross_encoder_reranker())
        out = root / "reports" / "retrieval_metrics.json"
        out.write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        return
    passages = passages_from_averitec(df[df["split"] == "train"])
    print(f"{len(passages)} passages from {int((df['split'] == 'train').sum())} train claims")
    target = root / "models" / "retrieval_index"
    if args.india:
        india = root / "data" / "processed" / "india"
        for name in ("fakecovid_india", "multiclaim_en"):
            fc = pd.read_parquet(india / f"{name}.parquet")
            extra = passages_from_factchecks(fc[fc["split"].isin(["train", "val"])])  # test claims must not find themselves
            print(f"+ {len(extra)} fact-check passages from {name}")
            passages += extra
        target = root / "models" / "retrieval_index_india"
    searcher = HybridSearcher(passages)
    searcher.save(target)
    print(f"saved {target.relative_to(root)} ({len(passages)} passages; FAISS: {HAVE_FAISS})")


if __name__ == "__main__":
    main()
