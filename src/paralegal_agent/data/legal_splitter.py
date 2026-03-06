

from __future__ import annotations

import re
from typing import Any

from langchain_core.documents import Document

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHUNK_SIZE = 1800      # characters
CHUNK_OVERLAP = 100    # characters

# Legal structure separators (ordered by priority, coarsest → finest)
_SEPARATORS = [
    r"\n(?=Khoản \d+[\.\:])",          # Khoản
    r"\n(?=Điểm [a-zđ]\))",            # Điểm
    r"\n(?=\d+[\.\)]\s)",              # numbered list item
    r"\n\n",                            # blank line / paragraph
    r"\n",                              # newline
    r"(?<=\.) ",                        # sentence boundary
]

_SEP_PATTERN = re.compile("|".join(_SEPARATORS))

# Detect a markdown table line: starts/ends with |
_TABLE_LINE_RE = re.compile(r"^\s*\|.*\|\s*$")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def split_law(law: dict[str, Any], merged_units: list[dict[str, str]]) -> list[Document]:
    """
    Split a normalized law dict + its merged units into ready-to-embed Documents.

    Args:
        law:           Output of document_loader._normalize_law()
        merged_units:  Output of unit_merger.merge_units()

    Returns:
        List of LangChain Documents with rich metadata.
    """
    base_meta = _build_base_meta(law)
    doc_header = _build_doc_header(law)

    documents: list[Document] = []
    for unit in merged_units:
        unit_docs = _split_unit(unit, base_meta, doc_header)
        documents.extend(unit_docs)

    return documents


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_doc_header(law: dict) -> str:
    doc_number = law.get("doc_number", "")
    doc_title = law.get("doc_title", "")
    legal_field = law.get("legal_field", "")
    header = f"Văn bản: {doc_number} - {doc_title}\nLĩnh vực: {legal_field}"
    return header


def _build_base_meta(law: dict) -> dict[str, Any]:
    return {
        "doc_number":        law.get("doc_number", ""),
        "doc_title":         law.get("doc_title", ""),
        "doc_type":          law.get("doc_type", ""),
        "legal_field":       law.get("legal_field", ""),         # comma-joined string (display)
        "legal_fields":      law.get("legal_fields", []),        # list (for filtering)
        "issuing_authority": law.get("issuing_authority", ""),
        "signer":            law.get("signer", ""),
        "issue_date":        law.get("issue_date", ""),
        "source_url":        law.get("source_url", ""),
    }


def _split_unit(
    unit: dict[str, str],
    base_meta: dict[str, Any],
    doc_header: str,
) -> list[Document]:
    """Split one merged unit into ≤ CHUNK_SIZE chunks."""
    unit_text = unit["unit_text"]
    unit_id = unit["unit_id"]
    unit_title = unit["unit_title"]

    chunks = _split_text_preserve_tables(unit_text, CHUNK_SIZE, CHUNK_OVERLAP)

    docs = []
    for idx, chunk in enumerate(chunks):
        page_content = f"{doc_header}\n\n{chunk}"
        meta = {
            **base_meta,
            "unit_id": unit_id,
            "unit_title": unit_title,
            "unit_parent": unit_id,   # flat structure; extendable to nested later
            "chunk_index": idx,
            "total_chunks": len(chunks),
        }
        docs.append(Document(page_content=page_content, metadata=meta))

    return docs


def _split_text_preserve_tables(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Split text into chunks of ≤ chunk_size characters.

    Tables (consecutive | lines) are treated as atomic blocks —
    they will NOT be split mid-table. If a table block alone exceeds
    chunk_size, it is kept in one chunk (with a warning).
    """
    # 1. Identify atomic blocks (table blocks vs text blocks)
    atomic_blocks = _extract_atomic_blocks(text)

    # 2. Greedily pack blocks into chunks
    chunks: list[str] = []
    current_parts: list[str] = []
    current_len = 0

    for block in atomic_blocks:
        block_len = len(block)

        if current_len + block_len <= chunk_size:
            current_parts.append(block)
            current_len += block_len + 2  # +2 for "\n\n"
        else:
            # Flush current chunk
            if current_parts:
                chunks.append("\n\n".join(current_parts))

            # Large text block: further split by separators
            if not _is_table_block(block) and block_len > chunk_size:
                sub_chunks = _split_by_separators(block, chunk_size, overlap)
                # overlap: carry tail of previous chunk into first sub_chunk
                if chunks and overlap > 0:
                    tail = chunks[-1][-overlap:]
                    sub_chunks[0] = tail + "\n\n" + sub_chunks[0]
                chunks.extend(sub_chunks)
                current_parts = []
                current_len = 0
            else:
                # Start new chunk with this block (table or small block)
                current_parts = [block]
                current_len = block_len + 2

    if current_parts:
        chunks.append("\n\n".join(current_parts))

    # Apply overlap: prepend tail of previous chunk to each subsequent chunk
    if overlap > 0:
        chunks = _apply_overlap(chunks, overlap)

    return chunks if chunks else [text]


def _extract_atomic_blocks(text: str) -> list[str]:
    """Split text into list of atomic blocks: table blocks or text segments."""
    lines = text.split("\n")
    blocks: list[str] = []
    current_text_lines: list[str] = []
    in_table = False
    table_lines: list[str] = []

    for line in lines:
        if _TABLE_LINE_RE.match(line):
            if not in_table:
                # Flush pending text
                if current_text_lines:
                    blocks.append("\n".join(current_text_lines).strip())
                    current_text_lines = []
                in_table = True
            table_lines.append(line)
        else:
            if in_table:
                # Flush table block
                blocks.append("\n".join(table_lines))
                table_lines = []
                in_table = False
            current_text_lines.append(line)

    # Flush remaining
    if in_table and table_lines:
        blocks.append("\n".join(table_lines))
    elif current_text_lines:
        blocks.append("\n".join(current_text_lines).strip())

    return [b for b in blocks if b.strip()]


def _is_table_block(block: str) -> bool:
    first_line = block.split("\n")[0]
    return bool(_TABLE_LINE_RE.match(first_line))


def _split_by_separators(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split a non-table text block by legal separators."""
    segments = _SEP_PATTERN.split(text)
    chunks: list[str] = []
    current_parts: list[str] = []
    current_len = 0

    for seg in segments:
        seg_len = len(seg)
        if current_len + seg_len <= chunk_size:
            current_parts.append(seg)
            current_len += seg_len
        else:
            if current_parts:
                chunks.append(" ".join(current_parts))
            current_parts = [seg]
            current_len = seg_len

    if current_parts:
        chunks.append(" ".join(current_parts))

    return chunks if chunks else [text]


def _apply_overlap(chunks: list[str], overlap: int) -> list[str]:
    """Prepend tail overlap from previous chunk to each subsequent chunk."""
    if len(chunks) <= 1:
        return chunks
    result = [chunks[0]]
    for i in range(1, len(chunks)):
        tail = _safe_tail(chunks[i - 1], overlap)
        result.append(tail + "\n\n" + chunks[i])
    return result
def _safe_tail(text: str, overlap: int) -> str:
    tail = text[-overlap:]
    # tìm khoảng trắng đầu tiên
    first_space = tail.find(" ")
    if first_space != -1:
        tail = tail[first_space + 1 :]
    return tail