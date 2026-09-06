"""
chunker.py — Section-aware chunker for the POCSO Act (Phase 1 Baseline)

Parses a plain-text copy of the Act and produces section-aligned chunks.
Each chunk carries metadata:  section_number, section_title, act_name,
chunk_id, chunk_index, has_proviso.

Section boundary detection strategy
-------------------------------------
The POCSO Act (and most Indian central acts) uses a consistent format:

    <blank line>
    <section-number>. <Section Title>.-
    <body text>
    ...
    <blank line>

Two constraints are applied together to identify a genuine section start,
rather than relying on the digit-dot pattern alone (which would also match
cross-references like "under sub-section (1) of section 4." in body text):

  1. The line must match the regex:  ^(\\d{1,3}[A-Z]?)\\.\\s+([A-Z].*)$
       - 1–3 digit number optionally followed by ONE uppercase letter (e.g. 10A)
       - literal period
       - one or more spaces
       - section title must begin with an uppercase letter
         (cross-references inside body text typically continue in lowercase)

  2. The *previous* non-empty line must be blank (empty string after strip).
       This eliminates most in-sentence digit-dot occurrences because real
       sections are always preceded by a blank line in well-formatted acts.

If the source text has non-standard formatting (e.g. no blank lines between
sections), run `debug_chunk_boundaries()` first and adjust the regex or
blank-line constraint here.

Long sections (>500 words) are split at sub-section boundaries (lines
starting with optional whitespace + "(N)") while retaining the parent
section number/title in every resulting chunk's metadata.
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ACT_NAME = "Protection of Children from Sexual Offences Act, 2012"
DATA_PATH = "./data/pocso_act.txt"
CHUNKS_OUTPUT_PATH = "./data/chunks.json"

# Max word-count proxy for "token" threshold before splitting at sub-sections
_MAX_WORDS: int = 500

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Genuine section header:
#   Group 1 — section number  (e.g. "1", "10", "10A")
#   Group 2 — section title   (must start with uppercase letter)
#
# Note: the \\s* allows for minor leading whitespace in case the source text
# has shallow indentation on section numbers (some printed act PDFs do this).
# Adjust to `^` only if your source is strictly left-aligned.
_SECTION_RE = re.compile(
    r"^(\d{1,3}[A-Z]?)\.\s+([A-Z][^\n]*)$"
)

# Sub-section opener: optional whitespace + "(N)" at line start
# Catches both "    (1) The..." and "(1) The..." forms
_SUBSECTION_RE = re.compile(r"^\s*\(\d+\)\s+\S")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_section_start(line: str, prev_line_blank: bool) -> Optional[re.Match]:
    """
    Return a regex Match if `line` is a genuine section header, else None.

    Constraint 1: must match _SECTION_RE (digit-dot + uppercase title start).
    Constraint 2: the previous non-empty line must have been blank
                  (section starts are always preceded by a blank line).
    """
    if not prev_line_blank:
        return None
    return _SECTION_RE.match(line.strip())


def _word_count(text: str) -> int:
    return len(text.split())


def _normalise_title(raw: str) -> str:
    """Strip trailing punctuation that appears in act formatting (e.g. '.-')."""
    return raw.rstrip(".-— ").strip()


def _split_at_subsections(body_text: str) -> List[str]:
    """
    Split `body_text` at sub-section boundaries ((1), (2), …).
    Each part is returned as a string; the first part may be a preamble
    before sub-section (1) and will be kept if non-empty.
    """
    lines = body_text.splitlines(keepends=True)
    parts: List[str] = []
    current: List[str] = []

    for line in lines:
        if _SUBSECTION_RE.match(line) and current:
            chunk = "".join(current).strip()
            if chunk:
                parts.append(chunk)
            current = [line]
        else:
            current.append(line)

    # Final accumulated block
    if current:
        chunk = "".join(current).strip()
        if chunk:
            parts.append(chunk)

    return parts


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_chunks(text: str) -> List[Dict]:
    """
    Parse the full POCSO Act text into section-aligned chunks.

    Parameters
    ----------
    text : str
        Full plain-text content of the Act (UTF-8).

    Returns
    -------
    List[Dict]
        Each dict has keys:
          chunk_id        — unique identifier e.g. "sec_5_chunk_0"
          section_number  — string e.g. "5", "10A"
          section_title   — cleaned title string
          act_name        — ACT_NAME constant
          has_proviso     — bool, True if "Provided that" appears in chunk
          chunk_index     — int, 0 for single-chunk sections, 0/1/2… for split ones
          text            — full chunk text (header + body)
    """
    lines = text.splitlines()
    # Collect raw sections as (number, title, body_lines)
    raw_sections: List[Tuple[str, str, List[str]]] = []

    current_num: Optional[str] = None
    current_title: Optional[str] = None
    current_body: List[str] = []
    prev_line_blank: bool = True  # Treat document start as if preceded by blank

    for line in lines:
        m = _is_section_start(line, prev_line_blank)
        if m:
            # Flush previous section
            if current_num is not None:
                raw_sections.append((current_num, current_title, current_body))
            current_num = m.group(1)
            current_title = _normalise_title(m.group(2))
            current_body = []
        else:
            if current_num is not None:
                current_body.append(line)

        prev_line_blank = (line.strip() == "")

    # Flush the final section
    if current_num is not None:
        raw_sections.append((current_num, current_title, current_body))

    # Build chunk list
    chunks: List[Dict] = []

    for sec_num, sec_title, body_lines in raw_sections:
        body_text = "\n".join(body_lines).strip()
        header = f"Section {sec_num}. {sec_title}\n\n"
        full_text = (header + body_text).strip()

        if _word_count(full_text) <= _MAX_WORDS:
            # ── Single chunk ──────────────────────────────────────────────
            chunks.append(_make_chunk(sec_num, sec_title, full_text, 0))
        else:
            # ── Split at sub-section boundaries ───────────────────────────
            sub_parts = _split_at_subsections(body_text)

            if len(sub_parts) <= 1:
                # Cannot split (no sub-sections found); keep as single chunk
                # despite exceeding word limit — flag would appear during eval.
                chunks.append(_make_chunk(sec_num, sec_title, full_text, 0))
            else:
                for idx, part in enumerate(sub_parts):
                    chunk_text = (header + part).strip()
                    chunks.append(_make_chunk(sec_num, sec_title, chunk_text, idx))

    return chunks


def _make_chunk(
    sec_num: str, sec_title: str, text: str, chunk_index: int
) -> Dict:
    return {
        "chunk_id": f"sec_{sec_num}_chunk_{chunk_index}",
        "section_number": sec_num,
        "section_title": sec_title,
        "act_name": ACT_NAME,
        "has_proviso": "Provided that" in text,
        "chunk_index": chunk_index,
        "text": text,
    }


def debug_chunk_boundaries(text: str) -> None:
    """
    Print every line that the section-start detector classifies as a new
    section header.

    Run this BEFORE a full indexing pass to manually verify the regex is
    not picking up cross-references or page numbers.  Compare the printed
    list against the first ~10 sections of your actual source file.

    Usage:
        from chunker import debug_chunk_boundaries
        text = open("data/pocso_act.txt", encoding="utf-8").read()
        debug_chunk_boundaries(text)
    """
    lines = text.splitlines()
    prev_line_blank: bool = True

    hits: List[Tuple[int, str, str]] = []  # (lineno, sec_num, title_preview)

    for i, line in enumerate(lines, start=1):
        m = _is_section_start(line, prev_line_blank)
        if m:
            hits.append((i, m.group(1), m.group(2)[:70]))
        prev_line_blank = (line.strip() == "")

    width = 70
    print("=" * width)
    print("DEBUG — Lines detected as section boundaries")
    print("=" * width)
    if not hits:
        print("  *** NO SECTION BOUNDARIES FOUND ***")
        print("  Check that your source file has blank lines before each section")
        print("  and that section numbers appear at the start of the line.")
    else:
        print(f"  {'Line':>6}  {'Sec':>5}  Title preview")
        print("  " + "-" * (width - 2))
        for lineno, sec_num, title_preview in hits:
            print(f"  {lineno:>6}  {sec_num:>5}  {title_preview}")
    print("-" * width)
    print(f"  Total boundaries found: {len(hits)}")
    print("=" * width)


def save_chunks(chunks: List[Dict], path: str = CHUNKS_OUTPUT_PATH) -> None:
    """Serialize the chunk list to JSON at `path`."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"[chunker] Saved {len(chunks)} chunks → {path}")


def load_chunks(path: str = CHUNKS_OUTPUT_PATH) -> List[Dict]:
    """Load a previously saved chunk list from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    source = sys.argv[1] if len(sys.argv) > 1 else DATA_PATH

    if not Path(source).exists():
        print(f"[chunker] ERROR: Source file not found at '{source}'")
        print("[chunker] Please place the POCSO Act plain text at that path.")
        print(f"[chunker] Usage:  python chunker.py [path/to/pocso_act.txt]")
        sys.exit(1)

    text = Path(source).read_text(encoding="utf-8")
    print(f"[chunker] Loaded {len(text):,} characters from '{source}'")
    print()

    # ── Step 1: sanity-check section boundary detection ──────────────────
    debug_chunk_boundaries(text)
    print()

    # ── Step 2: parse and report ──────────────────────────────────────────
    chunks = parse_chunks(text)
    print(f"[chunker] Parsed {len(chunks)} chunks total.")

    if chunks:
        split_sections = {c["section_number"] for c in chunks if c["chunk_index"] > 0}
        proviso_count = sum(1 for c in chunks if c["has_proviso"])
        print(f"[chunker] Sections split into sub-chunks: {sorted(split_sections) or 'none'}")
        print(f"[chunker] Chunks containing a proviso   : {proviso_count}")

        print("\n[chunker] Sample — first chunk:")
        s = chunks[0]
        for k, v in s.items():
            if k == "text":
                print(f"  {'text':<18}: {v[:200]}…")
            else:
                print(f"  {k:<18}: {v}")

    save_chunks(chunks)
