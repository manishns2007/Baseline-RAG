"""
index_builder.py — Build dense (Chroma) and sparse (BM25) indices (Phase 1 Baseline)

Entrypoint:
    build_indices(chunks)   — builds both indices from a loaded chunk list
    python index_builder.py — CLI: loads chunks.json, builds both indices

Dense index:
    Model      : BAAI/bge-base-en-v1.5  (via sentence-transformers)
    Store      : ChromaDB persistent client at ./chroma_db/
    Collection : "pocso_dense"

Sparse index:
    Model      : BM25Okapi  (rank_bm25)
    Tokenizer  : lowercase + whitespace split  (Phase 2 candidate: strip punctuation)
    Persistence: ./indices/bm25_index.pkl   — pickled BM25Okapi + ordered chunk_ids
                 ./indices/chunk_map.json   — chunk_id → chunk_text lookup
"""

import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

CHROMA_PATH = "./chroma_db"
CHROMA_COLLECTION = "pocso_dense"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

INDICES_DIR = "./indices"
BM25_INDEX_PATH = f"{INDICES_DIR}/bm25_index.pkl"
CHUNK_MAP_PATH = f"{INDICES_DIR}/chunk_map.json"

EMBED_BATCH_SIZE = 32  # safe for 8 GB VRAM / CPU; increase if you have more memory


# ---------------------------------------------------------------------------
# Tokenizer (shared with retriever.py)
# ---------------------------------------------------------------------------

def tokenize(text: str) -> List[str]:
    """
    Lowercase + whitespace tokenizer for BM25.

    Phase 2 improvement candidate: strip punctuation before splitting so that
    tokens like "sub-section(1)" become ["sub-section", "1"] or similar,
    improving recall on legal cross-references.
    """
    return text.lower().split()


# ---------------------------------------------------------------------------
# Dense index (ChromaDB + bge-base-en-v1.5)
# ---------------------------------------------------------------------------

def build_dense_index(chunks: List[Dict]) -> None:
    """Embed all chunks and upsert into a persistent Chroma collection."""
    print(f"[index_builder] Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Drop and recreate collection to allow clean re-indexing
    try:
        client.delete_collection(CHROMA_COLLECTION)
        print(f"[index_builder] Dropped existing collection '{CHROMA_COLLECTION}'.")
    except Exception:
        pass  # Collection didn't exist yet — fine

    collection = client.create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    texts = [c["text"] for c in chunks]
    ids = [c["chunk_id"] for c in chunks]

    # Chroma metadata values must be str / int / float — cast bool to str
    metadatas = [
        {
            "section_number": c["section_number"],
            "section_title": c["section_title"],
            "act_name": c["act_name"],
            "has_proviso": str(c["has_proviso"]),
            "chunk_index": int(c["chunk_index"]),
        }
        for c in chunks
    ]

    # Embed in batches to avoid OOM errors on large corpora
    print(
        f"[index_builder] Embedding {len(texts)} chunks "
        f"(batch_size={EMBED_BATCH_SIZE}) …"
    )
    all_embeddings: List[List[float]] = []
    for i in tqdm(range(0, len(texts), EMBED_BATCH_SIZE), desc="  Embedding"):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        embs = model.encode(batch, normalize_embeddings=True).tolist()
        all_embeddings.extend(embs)

    collection.add(
        ids=ids,
        embeddings=all_embeddings,
        documents=texts,
        metadatas=metadatas,
    )
    print(
        f"[index_builder] Dense index ready — "
        f"{collection.count()} vectors in '{CHROMA_COLLECTION}'."
    )


# ---------------------------------------------------------------------------
# Sparse index (BM25Okapi)
# ---------------------------------------------------------------------------

def build_sparse_index(chunks: List[Dict]) -> None:
    """Build a BM25Okapi index and persist it along with the chunk-text map."""
    Path(INDICES_DIR).mkdir(parents=True, exist_ok=True)

    print(f"[index_builder] Tokenizing {len(chunks)} chunks for BM25 …")
    chunk_ids = [c["chunk_id"] for c in chunks]
    tokenized = [tokenize(c["text"]) for c in chunks]

    bm25 = BM25Okapi(tokenized)

    payload = {"bm25": bm25, "chunk_ids": chunk_ids}
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(payload, f)
    print(f"[index_builder] BM25 index saved → {BM25_INDEX_PATH}")

    chunk_map: Dict[str, str] = {c["chunk_id"]: c["text"] for c in chunks}
    with open(CHUNK_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(chunk_map, f, ensure_ascii=False, indent=2)
    print(f"[index_builder] Chunk map saved  → {CHUNK_MAP_PATH} ({len(chunk_map)} entries)")


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

def build_indices(chunks: List[Dict]) -> None:
    """
    Build both the dense (Chroma) and sparse (BM25) indices from `chunks`.

    Parameters
    ----------
    chunks : List[Dict]
        Output of chunker.parse_chunks() or chunker.load_chunks().
    """
    if not chunks:
        raise ValueError(
            "[index_builder] Chunk list is empty. "
            "Make sure chunker.py ran successfully and data/chunks.json exists."
        )
    build_dense_index(chunks)
    build_sparse_index(chunks)
    print("[index_builder] ✓ All indices built successfully.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Allow optional path override: python index_builder.py [chunks.json]
    chunks_path = sys.argv[1] if len(sys.argv) > 1 else "./data/chunks.json"

    if not Path(chunks_path).exists():
        print(f"[index_builder] ERROR: '{chunks_path}' not found.")
        print("[index_builder] Run `python chunker.py` first to generate chunks.json.")
        sys.exit(1)

    # Local import to avoid circular dependency when index_builder is imported elsewhere
    from chunker import load_chunks

    chunks = load_chunks(chunks_path)
    print(f"[index_builder] Loaded {len(chunks)} chunks from '{chunks_path}'.")
    build_indices(chunks)
