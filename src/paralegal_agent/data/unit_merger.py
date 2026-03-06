
from __future__ import annotations

import re
from typing import Any

from bs4 import BeautifulSoup


def merge_units(content: list[dict]) -> list[dict[str, str]]:
    """
    Convert a law's content list into merged unit dicts.

    Each output dict has:
        unit_id    : e.g. "dieu_1"
        unit_title : e.g. "Điều 1. Phạm vi điều chỉnh..."
        unit_text  : merged text ready for splitting
    """
    merged = []
    for unit in content:
        unit_id = unit.get("unit_id", "")
        unit_title = (unit.get("unit_title") or "").strip()
        unit_content = unit.get("unit_content") or []

        body = _merge_content_items(unit_content)
        unit_text = f"{unit_title}\n\n{body}".strip()

        merged.append({
            "unit_id": unit_id,
            "unit_title": unit_title,
            "unit_text": unit_text,
        })

    return merged


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _merge_content_items(items: list[dict]) -> str:
    """Merge a list of {type, data} content items into one string."""
    parts: list[str] = []
    for item in items:
        item_type = item.get("type", "")
        data = item.get("data", "") or ""

        if item_type == "text":
            text = data.strip()
            if text:
                parts.append(text)

        elif item_type == "table_html":
            md_table = _html_table_to_markdown(data)
            if md_table:
                parts.append(md_table)

    return "\n\n".join(parts)


def _html_table_to_markdown(html: str) -> str:
    """Convert an HTML table string to a Markdown table string."""
    try:
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table")
        if not table:
            # Might be raw text wrapped in HTML
            return soup.get_text(separator=" ").strip()

        rows: list[list[str]] = []
        for tr in table.find_all("tr"):
            cells = [_cell_text(td) for td in tr.find_all(["th", "td"])]
            if any(c.strip() for c in cells):
                rows.append(cells)

        if not rows:
            return ""

        return _rows_to_markdown(rows)

    except Exception:  # noqa: BLE001
        # Fallback: strip tags
        return BeautifulSoup(html, "html.parser").get_text(separator=" ").strip()


def _cell_text(tag) -> str:
    """Extract clean text from a table cell, collapse whitespace."""
    text = tag.get_text(separator=" ")
    return re.sub(r"\s+", " ", text).strip()


def _rows_to_markdown(rows: list[list[str]]) -> str:
    """Convert list of row-lists into a Markdown table string."""
    if not rows:
        return ""

    # Normalize column count
    max_cols = max(len(r) for r in rows)
    padded = [r + [""] * (max_cols - len(r)) for r in rows]

    # Column widths
    col_widths = [
        max(len(padded[ri][ci]) for ri in range(len(padded)))
        for ci in range(max_cols)
    ]
    col_widths = [max(w, 3) for w in col_widths]

    def fmt_row(row: list[str]) -> str:
        cells = [row[ci].ljust(col_widths[ci]) for ci in range(max_cols)]
        return "| " + " | ".join(cells) + " |"

    def sep_row() -> str:
        return "| " + " | ".join("-" * w for w in col_widths) + " |"

    lines = [fmt_row(padded[0]), sep_row()]
    for row in padded[1:]:
        lines.append(fmt_row(row))

    return "\n".join(lines)
