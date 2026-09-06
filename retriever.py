"""
retriever.py — Hybrid retrieval with Reciprocal Rank Fusion (Phase 1 Baseline)

Public API:
    dense_search(query, k=10)  -> List[(chunk_id, rank)]
    sparse_search(query, k=10) -> List[(chunk_id, rank)]
    reciprocal_rank_fusion(dense_results, sparse_results, k=60)
                               -> List[(chunk_id, rrf_score)]
    retrieve(query, top_n=5)   -> List[Dict]  (full chunk dicts, RRF-ranked)

Indices must be built by index_builder.py before calling any search function.

Phase 2 improvement candidate (BM25 tokenization):
    Current tokenizer uses lowercase + whitespace split.  This under-performs
    on legal text because punctuation is adjacent to terms — e.g.
    "sub-section(1)" is a single token instead of ["sub-section", "1"].
    Consider stripping punctuation before splitting, or adopting a dedicated
    legal/NLP tokenizer (e.g. spaCy with a legal model) in Phase 2.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chromadb
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Shared constants (keep in sync with index_builder.py)
# ---------------------------------------------------------------------------

CHROMA_PATH = "./chroma_db"
CHROMA_COLLECTION = "pocso_dense"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
BM25_INDEX_PATH = "./indices/bm25_index.pkl"
CHUNK_MAP_PATH = "./indices/chunk_map.json"

# ---------------------------------------------------------------------------
# Lazy-loaded singletons (initialised on first call to avoid startup cost)
# ---------------------------------------------------------------------------

_embed_model: Optional[SentenceTransformer] = None
_chroma_collection = None
_bm25_payload: Optional[Dict] = None   # {"bm25": BM25Okapi, "chunk_ids": [...]}
_chunk_map: Optional[Dict[str, str]] = None


def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embed_model


def _get_chroma_collection():
    global _chroma_collection
    if _chroma_collection is None:
        if not Path(CHROMA_PATH).exists():
            raise FileNotFoundError(
                f"Chroma DB not found at '{CHROMA_PATH}'. "
                "Run `python index_builder.py` first."
            )
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        _chroma_collection = client.get_collection(CHROMA_COLLECTION)
    return _chroma_collection


def _get_bm25_payload() -> Dict:
    global _bm25_payload
    if _bm25_payload is None:
        if not Path(BM25_INDEX_PATH).exists():
            raise FileNotFoundError(
                f"BM25 index not found at '{BM25_INDEX_PATH}'. "
                "Run `python index_builder.py` first."
            )
        with open(BM25_INDEX_PATH, "rb") as f:
            _bm25_payload = pickle.load(f)
    return _bm25_payload


def _get_chunk_map() -> Dict[str, str]:
    global _chunk_map
    if _chunk_map is None:
        with open(CHUNK_MAP_PATH, "r", encoding="utf-8") as f:
            _chunk_map = json.load(f)
    return _chunk_map


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    """
    Lowercase + whitespace tokenizer.

    Phase 2 improvement candidate: strip punctuation before tokenizing so that
    legal patterns like "sub-section(1)" are split into meaningful sub-tokens,
    improving BM25 recall on cross-referenced provisions.
    """
    return text.lower().split()


# ---------------------------------------------------------------------------
# Individual search functions
# ---------------------------------------------------------------------------

def dense_search(query: str, k: int = 10) -> List[Tuple[str, int]]:
    """
    Embed `query` and return the top-k closest chunks from Chroma.

    Returns
    -------
    List of (chunk_id, rank) tuples, rank 1 = most similar.
    """
    model = _get_embed_model()
    collection = _get_chroma_collection()

    query_emb = model.encode([query], normalize_embeddings=True).tolist()[0]
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=min(k, collection.count()),
        include=["distances", "metadatas"],
    )
    chunk_ids: List[str] = results["ids"][0]
    return [(cid, rank + 1) for rank, cid in enumerate(chunk_ids)]


def sparse_search(query: str, k: int = 10) -> List[Tuple[str, int]]:
    """
    Score all chunks with BM25 and return the top-k results.

    Returns
    -------
    List of (chunk_id, rank) tuples, rank 1 = highest BM25 score.
    """
    payload = _get_bm25_payload()
    bm25 = payload["bm25"]
    chunk_ids: List[str] = payload["chunk_ids"]

    tokenized_query = _tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    ranked = sorted(
        zip(chunk_ids, scores),
        key=lambda x: x[1],
        reverse=True,
    )[:k]

    return [(cid, rank + 1) for rank, (cid, _) in enumerate(ranked)]


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    dense_results: List[Tuple[str, int]],
    sparse_results: List[Tuple[str, int]],
    k: int = 60,
) -> List[Tuple[str, float]]:
    """
    Merge two ranked lists using Reciprocal Rank Fusion (RRF).

    Formula: score(d) = Σ  1 / (k + rank(d))
             summed over all lists in which d appears.

    Parameters
    ----------
    dense_results  : output of dense_search()
    sparse_results : output of sparse_search()
    k              : RRF constant (default 60, standard in literature)

    Returns
    -------
    List of (chunk_id, rrf_score) sorted descending by score.
    """
    scores: Dict[str, float] = {}

    for cid, rank in dense_results:
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)

    for cid, rank in sparse_results:
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------------------
# Full hybrid retrieval
# ---------------------------------------------------------------------------

def retrieve(query: str, top_n: int = 5) -> List[Dict]:
    """
    Full hybrid retrieval pipeline.

    Runs dense_search + sparse_search, fuses with RRF, then fetches
    full document text and metadata for the top-n results from Chroma.

    Parameters
    ----------
    query  : natural language question
    top_n  : number of chunks to return (default 5)

    Returns
    -------
    List of dicts, each with keys:
      chunk_id  — unique chunk identifier
      text      — full chunk text (header + body)
      metadata  — Chroma metadata dict (section_number, section_title, …)
      rrf_score — fused RRF score (higher = more relevant)
    Ordered best-first.
    """
    dense_res = dense_search(query, k=10)
    sparse_res = sparse_search(query, k=10)
    fused = reciprocal_rank_fusion(dense_res, sparse_res)

    top_ids = [cid for cid, _ in fused[:top_n]]
    rrf_score_map = {cid: score for cid, score in fused}

    if not top_ids:
        return []

    # Fetch text + metadata from Chroma (.get() doesn't guarantee order)
    collection = _get_chroma_collection()
    raw = collection.get(ids=top_ids, include=["documents", "metadatas"])

    id_to_doc = dict(zip(raw["ids"], raw["documents"]))
    id_to_meta = dict(zip(raw["ids"], raw["metadatas"]))

    # Re-order to match RRF rank
    retrieved: List[Dict] = []
    for cid in top_ids:
        if cid in id_to_doc:
            retrieved.append(
                {
                    "chunk_id": cid,
                    "text": id_to_doc[cid],
                    "metadata": id_to_meta[cid],
                    "rrf_score": rrf_score_map[cid],
                }
            )

    return retrieved


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    query = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "What is the punishment for aggravated penetrative sexual assault?"
    )
    print(f"[retriever] Query: {query}\n")

    results = retrieve(query, top_n=5)
    if not results:
        print("[retriever] No results returned — is the index built?")
        sys.exit(1)

    for i, r in enumerate(results, start=1):
        meta = r["metadata"]
        print(
            f"  [{i}] Section {meta.get('section_number')} — "
            f"{meta.get('section_title')} "
            f"(rrf={r['rrf_score']:.4f}, id={r['chunk_id']})"
        )
        print(f"       {r['text'][:120].replace(chr(10), ' ')}…\n")
