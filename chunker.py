"""
chunker.py — Section-aware chunker for the POCSO Act (Phase 1 Baseline)

Parses a plain-text copy of the Act and produces section-aligned chunks.
Each chunk carries metadata:  section_number, section_title, act_name,
chunk_id, chunk_index, has_proviso.

Section boundary detection strategy
-------------------------------------
PDF-extracted POCSO text has a specific format that differs from idealised
plain-text acts:

  • Section headers appear inline with body text — the section number, title,
    em-dash (—), and first sub-section are all on ONE line:
      "2. Definitions.—(1) In this Act, unless the context otherwise requires"

  • There is NO guaranteed blank line before each section start.

  • The PDF extractor injects footnote lines into the body text:
      "1. The words ..."  (footnote text starting with a digit-dot)
      bare numbers like "4", "1", "2" on their own line
      "IndiaCode" page watermarks

Detection strategy used here
------------------------------
  1. PRIMARY regex: ^(\d{1,3}[A-Z]?)\.\s+([A-Z][^.\n]{2,})
       - 1–3 digits + optional uppercase letter, period, spaces
       - Title MUST start with uppercase letter and be ≥3 chars before any
         punctuation — this rules out footnote lines like "1. The words…" whose
         first word is lowercase after the period, and cross-references whose
         title fragments are short.

  2. NEGATIVE filter: rejects lines that look like footnotes:
       - Line starts with a digit followed by a period then a LOWERCASE letter
         OR a digit followed by a star  ("1***")
       - Line is a bare number, a page artefact ("IndiaCode"), or a bracket-
         prefixed amendment note ("[(da)…")

  3. TITLE extraction: the portion of the line up to the em-dash (—) or the
     first opening parenthesis is used as the section title.  This handles
     the merged header+body format.

Run debug_chunk_boundaries() after any format change to re-verify.

Long sections (>500 words) are split at sub-section boundaries (lines
starting with optional whitespace + "(N)") while retaining parent metadata.
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

# PRIMARY section-header pattern for PDF-extracted POCSO text.
# Matches lines like:
#   "1. Short title, extent and commencement.—(1) This Act…"
#   "2. Definitions.—(1) In this Act…"
#   "10A. Aggravated penetrative sexual assault.—…"
#
# Group 1 — section number (e.g. "1", "2", "10A")
# Group 2 — everything after the number+period up to end-of-line
#           (title + body run together; we extract title separately below)
#
# Constraint: title portion must start with an uppercase letter and be at
# least 3 characters before any punctuation to reject short footnote lines.
_SECTION_RE = re.compile(
    r"^(\d{1,3}[A-Z]?)\.\s+([A-Z][A-Za-z ]{2,})"
)

# NEGATIVE filter — lines that look like footnotes or PDF artefacts:
#   "1. The words…"  (footnote: digit-dot-space-lowercase)
#   "2. 14th November…"  (footnote: digit-dot-space-digit)
#   bare numbers on their own line, "IndiaCode" watermark
_FOOTNOTE_RE = re.compile(
    r"^\d{1,3}\.\s+[a-z0-9]"   # footnote: starts with digit-dot then lowercase/digit
    r"|^\d+$"                   # bare standalone number (page artefact)
    r"|^IndiaCode\s*$"          # PDF watermark
    r"|^\[\("                   # amendment note like "[(da)"
)

# Em-dash and related separators used between section title and body
_EMDASH_RE = re.compile(r"[\u2014\u2013\-]{1,2}")

# Sub-section opener: optional whitespace + "(N)" at line start
# Catches both "    (1) The..." and "(1) The..." forms
_SUBSECTION_RE = re.compile(r"^\s*\(\d+\)\s+\S")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_section_start(line: str) -> Optional[re.Match]:
    """
    Return a regex Match if `line` looks like a genuine POCSO section header.

    Strategy (for PDF-extracted text with no guaranteed blank lines):
      1. Must match _SECTION_RE  (digit-dot + uppercase title ≥3 chars).
      2. Must NOT match _FOOTNOTE_RE  (footnotes, bare numbers, watermarks).

    The blank-line constraint is intentionally removed here because the
    PDF-extracted source does not reliably produce blank lines before each
    section.  If you have a clean plain-text source with blank lines, you
    can re-add: `if not prev_line_blank: return None` before the FOOTNOTE
    check for additional precision.
    """
    stripped = line.strip()
    if _FOOTNOTE_RE.match(stripped):
        return None
    return _SECTION_RE.match(stripped)


def _extract_title(header_tail: str) -> str:
    """
    Extract the section title from the tail portion of a section header line.

    In PDF-extracted POCSO text the line looks like:
        "Short title, extent and commencement.—(1) This Act…"
    We want only: "Short title, extent and commencement"

    Strategy: take everything up to the first em-dash (—) or the first
    opening parenthesis "(", whichever comes first.
    """
    # Split on em-dash / en-dash
    if '\u2014' in header_tail:
        title = header_tail.split('\u2014')[0]
    elif '\u2013' in header_tail:
        title = header_tail.split('\u2013')[0]
    elif '.—' in header_tail:
        title = header_tail.split('.—')[0]
    else:
        # Fallback: take up to first "(" which starts a sub-section inline
        paren_idx = header_tail.find('(')
        title = header_tail[:paren_idx] if paren_idx != -1 else header_tail
    return _normalise_title(title)


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

def _preprocess(text: str) -> str:
    """
    Remove known PDF artefact lines before parsing.

    Strips:
      - Bare numbers on their own line (page numbers injected by PDF extractor)
      - "IndiaCode" watermark lines
      - Footnote markers like "1***" alone on a line

    Does NOT strip footnote text lines ("1. The words…") because those are
    already handled by _FOOTNOTE_RE in _is_section_start().
    """
    cleaned: List[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        # Drop: bare number, "IndiaCode", or isolated "N***" footnote markers
        if re.match(r'^\d+$', stripped) or re.match(r'^IndiaCode\s*$', stripped):
            continue
        if re.match(r'^\d+\*+$', stripped):  # e.g. "1***"
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


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
    text = _preprocess(text)
    lines = text.splitlines()
    # Collect raw sections as (number, title, body_lines)
    raw_sections: List[Tuple[str, str, List[str]]] = []

    current_num: Optional[str] = None
    current_title: Optional[str] = None
    current_body: List[str] = []

    for line in lines:
        m = _is_section_start(line)
        if m:
            # Flush previous section
            if current_num is not None:
                raw_sections.append((current_num, current_title, current_body))
            current_num = m.group(1)
            # Extract clean title from the matched tail
            current_title = _extract_title(m.group(2))
            # The remainder of the header line (after the title/dash) becomes
            # the first line of the body
            full_line = line.strip()
            # Find where the body starts (after the em-dash or first "(")
            emdash_pos = full_line.find('\u2014')
            if emdash_pos != -1:
                body_start = full_line[emdash_pos + 1:].strip()
            else:
                paren_pos = full_line.find('(')
                body_start = full_line[paren_pos:].strip() if paren_pos != -1 else ""
            current_body = [body_start] if body_start else []
        else:
            if current_num is not None:
                current_body.append(line)

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
    text = _preprocess(text)  # apply same pre-processing as parse_chunks
    lines = text.splitlines()

    hits: List[Tuple[int, str, str]] = []  # (lineno, sec_num, title_preview)

    for i, line in enumerate(lines, start=1):
        m = _is_section_start(line)
        if m:
            title = _extract_title(m.group(2))
            hits.append((i, m.group(1), title[:70]))

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
