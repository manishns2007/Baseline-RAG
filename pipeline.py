"""
pipeline.py — Top-level RAG pipeline + CLI entrypoint (Phase 1 Baseline)

Wires retriever.retrieve() → generator.generate_answer() into a single
answer_question() function, extracts cited sections from the LLM output,
and records end-to-end latency.

Usage (CLI):
    python pipeline.py "What is the punishment for aggravated penetrative
                        sexual assault under POCSO?"

Programmatic:
    from pipeline import answer_question
    result = answer_question("What are the duties of a Special Court?")
    print(result["answer"])
"""

import re
import sys
import time
from typing import Dict, List

from generator import REFUSAL_PHRASE, generate_answer
from retriever import retrieve

# ---------------------------------------------------------------------------
# Citation extraction
# ---------------------------------------------------------------------------

# Matches [Section 5], [Section 10A], [section 2], etc. (case-insensitive)
_CITATION_RE = re.compile(r"\[Section\s+(\d+[A-Z]?)\]", re.IGNORECASE)


def _extract_cited_sections(answer: str) -> List[str]:
    """
    Extract all unique section numbers cited in [Section X] format from `answer`.
    Order of first appearance is preserved.
    """
    matches = _CITATION_RE.findall(answer)
    seen: set = set()
    unique: List[str] = []
    for m in matches:
        if m not in seen:
            seen.add(m)
            unique.append(m)
    return unique


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def answer_question(query: str, top_n: int = 5) -> Dict:
    """
    End-to-end POCSO RAG pipeline.

    Steps
    -----
    1. Hybrid retrieval (dense + sparse → RRF fusion) for top_n chunks.
    2. LLM generation grounded in retrieved chunks.
    3. Citation extraction from the answer text.

    Parameters
    ----------
    query  : natural language legal question
    top_n  : number of chunks to retrieve (default 5)

    Returns
    -------
    {
        "answer"               : str    — LLM-generated answer with [Section X] citations
        "cited_sections"       : List[str] — section numbers found in the answer
        "retrieved_chunk_ids"  : List[str] — chunk IDs passed to the LLM
        "latency_seconds"      : float  — wall-clock time for the full pipeline
        "is_refusal"           : bool   — True if LLM emitted the refusal phrase
    }
    """
    start = time.perf_counter()

    # Step 1 — Retrieve
    retrieved_chunks = retrieve(query, top_n=top_n)
    retrieved_chunk_ids = [c["chunk_id"] for c in retrieved_chunks]

    # Step 2 — Generate
    answer = generate_answer(query, retrieved_chunks)

    latency = time.perf_counter() - start

    # Step 3 — Extract citations
    cited_sections = _extract_cited_sections(answer)

    # TODO Phase 2: validate cited_sections against retrieved_chunk_ids — currently
    # trusting LLM-formatted citations with no cross-check against actual retrieved chunks

    is_refusal = answer.strip().startswith(REFUSAL_PHRASE)

    return {
        "answer": answer,
        "cited_sections": cited_sections,
        "retrieved_chunk_ids": retrieved_chunk_ids,
        "latency_seconds": round(latency, 3),
        "is_refusal": is_refusal,
    }


# ---------------------------------------------------------------------------
# CLI display
# ---------------------------------------------------------------------------

def _pretty_print(query: str, result: Dict) -> None:
    hr = "─" * 70
    print(f"\n{'═' * 70}")
    print(f"  QUERY")
    print(f"{'═' * 70}")
    print(f"  {query}")
    print(f"\n{'═' * 70}")
    print(f"  ANSWER")
    print(f"{'═' * 70}")
    print(result["answer"])
    print(f"\n{hr}")
    cited = ", ".join(f"Section {s}" for s in result["cited_sections"]) or "none"
    print(f"  Cited sections   : {cited}")
    print(f"  Retrieved chunks : {', '.join(result['retrieved_chunk_ids'])}")
    print(f"  Latency          : {result['latency_seconds']}s")
    if result["is_refusal"]:
        print("  [OUT-OF-SCOPE] Model indicated insufficient coverage.")
    print(f"{'═' * 70}\n")


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage:   python pipeline.py \"<question about POCSO>\"\n"
            "Example: python pipeline.py "
            "\"What is the punishment for aggravated penetrative sexual assault "
            "under POCSO?\""
        )
        sys.exit(1)

    query = sys.argv[1]
    print(f"\n[pipeline] Running query …")

    try:
        result = answer_question(query)
    except FileNotFoundError as e:
        print(f"\n[pipeline] ERROR: Index not found.\n  {e}")
        print(
            "\n  Make sure you have run:\n"
            "    1. python chunker.py          # generate data/chunks.json\n"
            "    2. python index_builder.py    # build Chroma + BM25 indices\n"
        )
        sys.exit(1)
    except RuntimeError as e:
        print(f"\n[pipeline] ERROR: {e}")
        sys.exit(1)

    _pretty_print(query, result)
