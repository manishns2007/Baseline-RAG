"""
generator.py — LLM answer generation via local Ollama (Phase 1 Baseline)

Public API:
    generate_answer(query, retrieved_chunks) -> str

Calls the local Ollama API (llama3.1:8b) with a structured prompt that:
  1. Instructs the model to answer ONLY from the provided section excerpts.
  2. Requires inline [Section X] citations after every claim.
  3. Instructs the model to emit a fixed refusal string when coverage is
     insufficient (placeholder for the Phase 2 confidence gate).

Client selection:
  - Uses the `ollama` Python package (ollama.chat) if installed.
  - Falls back to raw HTTP (requests.post → /api/chat) otherwise.
"""

from typing import Dict, List

# ---------------------------------------------------------------------------
# Ollama client selection
# ---------------------------------------------------------------------------

try:
    import ollama as _ollama_pkg
    _HAVE_OLLAMA_PKG = True
except ImportError:
    _HAVE_OLLAMA_PKG = False
    import requests as _requests  # type: ignore[import]

OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_TIMEOUT_SECONDS = 180

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a precise legal assistant specialising in Indian child protection law.
You will be given numbered excerpts from the Protection of Children from Sexual \
Offences (POCSO) Act, 2012.

STRICT RULES — follow every one without exception:

1. Answer ONLY using information that is explicitly present in the provided section \
excerpts below.  Do NOT use any external knowledge, prior legal training, or \
inferences beyond what the text states.

2. After every factual claim you make, append an inline citation in the exact format \
[Section X], where X is the section number from the excerpt (e.g. [Section 5], \
[Section 10A]).  Every sentence that asserts a legal fact must end with at least \
one such citation.

3. If the provided excerpts do not contain sufficient information to answer the \
question — for example, if the question concerns a different law, an aspect not \
covered by the retrieved sections, or a topic the excerpts are silent on — respond \
with EXACTLY the following phrase and nothing else:
   "I don't have enough information in the retrieved sections to answer this."

4. Do not speculate, extrapolate, add disclaimers, or expand on the law beyond what \
the retrieved text explicitly states.
"""

_USER_TEMPLATE = """\
RETRIEVED EXCERPTS FROM THE POCSO ACT, 2012:
{context}

---
QUESTION: {question}

ANSWER (cite every legal claim with [Section X]):"""

# Refusal phrase the model is instructed to emit — kept here for downstream
# parsing in Phase 2 (confidence gate).
REFUSAL_PHRASE = (
    "I don't have enough information in the retrieved sections to answer this."
)


# ---------------------------------------------------------------------------
# Context formatting
# ---------------------------------------------------------------------------

def _build_context(retrieved_chunks: List[Dict]) -> str:
    """
    Format retrieved chunks into a numbered, LLM-readable context block.

    Each block is labelled with the section number and title so the model can
    construct well-formed [Section X] citations without guessing.
    """
    parts: List[str] = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        meta = chunk.get("metadata", {})
        sec_num = meta.get("section_number", "?")
        sec_title = meta.get("section_title", "")
        header = f"[Excerpt {i}] Section {sec_num}: {sec_title}"
        parts.append(f"{header}\n{chunk['text']}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Ollama call wrappers
# ---------------------------------------------------------------------------

def _call_via_package(messages: List[Dict]) -> str:
    """Use the `ollama` Python package to call the local model."""
    response = _ollama_pkg.chat(
        model=OLLAMA_MODEL,
        messages=messages,
    )
    # ollama >= 0.2 returns a ChatResponse object; attribute and dict access both work
    try:
        return response.message.content  # attribute access (preferred)
    except AttributeError:
        return response["message"]["content"]  # dict fallback


def _call_via_http(messages: List[Dict]) -> str:
    """Fall back to raw HTTP POST when the `ollama` package is not installed."""
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
    }
    resp = _requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json=payload,
        timeout=OLLAMA_TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_answer(query: str, retrieved_chunks: List[Dict]) -> str:
    """
    Generate a cited answer to `query` grounded in `retrieved_chunks`.

    Parameters
    ----------
    query            : the user's legal question
    retrieved_chunks : list of chunk dicts from retriever.retrieve()

    Returns
    -------
    str — the model's answer, ideally containing [Section X] inline citations,
          or REFUSAL_PHRASE if the retrieved context is insufficient.

    Raises
    ------
    ConnectionError  if Ollama is not reachable at OLLAMA_BASE_URL.
    RuntimeError     for unexpected API errors.
    """
    if not retrieved_chunks:
        # Nothing to ground on — return refusal immediately without calling LLM
        return REFUSAL_PHRASE

    context = _build_context(retrieved_chunks)
    user_message = _USER_TEMPLATE.format(context=context, question=query)

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    try:
        if _HAVE_OLLAMA_PKG:
            return _call_via_package(messages)
        else:
            return _call_via_http(messages)
    except Exception as exc:
        # Surface a clear error rather than a silent empty string
        raise RuntimeError(
            f"[generator] Ollama call failed. "
            f"Is '{OLLAMA_MODEL}' running at {OLLAMA_BASE_URL}?\n"
            f"Original error: {exc}"
        ) from exc
