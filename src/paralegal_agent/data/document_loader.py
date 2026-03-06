"""
document_loader.py
------------------
Reads corpus_final.json and normalizes each law into a structured dict
with all fields needed for embedding and metadata filtering.

Notes on legal_field:
- Source: diagram_info["field"]
- May contain multiple fields separated by commas:
    e.g. "Tiền tệ - Ngân hàng, Tài chính nhà nước"
- Some crawled values are missing Vietnamese diacritics → normalized via map.
- Result stored as List[str] (one item per field) for easy filtering.
"""

import json
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Field normalization map
# Covers all known corrupted/non-accented variants found in corpus_final.json
# ---------------------------------------------------------------------------
FIELD_NORMALIZATION_MAP: dict[str, str] = {
    # Missing diacritics (confirmed from corpus scan)
    "Bao Hiem":          "Bảo hiểm",
    "Tai Chinh Nha Nuoc": "Tài chính nhà nước",
    "Tien Te Ngan Hang": "Tiền tệ - Ngân hàng",
    # Canonical forms (identity mapping for explicit completeness)
    "Bảo hiểm":              "Bảo hiểm",
    "Bất động sản":          "Bất động sản",
    "Bộ máy hành chính":     "Bộ máy hành chính",
    "Chứng khoán":           "Chứng khoán",
    "Công nghệ thông tin":   "Công nghệ thông tin",
    "Doanh nghiệp":          "Doanh nghiệp",
    "Dịch vụ pháp lý":      "Dịch vụ pháp lý",
    "Giao thông - Vận tải":  "Giao thông - Vận tải",
    "Giáo dục":              "Giáo dục",
    "Kế toán - Kiểm toán":  "Kế toán - Kiểm toán",
    "Lao động - Tiền lương": "Lao động - Tiền lương",
    "Lĩnh vực khác":         "Lĩnh vực khác",
    "Quyền dân sự":          "Quyền dân sự",
    "Sở hữu trí tuệ":        "Sở hữu trí tuệ",
    "Thuế - Phí - Lệ Phí":  "Thuế - Phí - Lệ Phí",
    "Thương mại":            "Thương mại",
    "Thể thao - Y tế":       "Thể thao - Y tế",
    "Thủ tục Tố tụng":       "Thủ tục Tố tụng",
    "Tiền tệ - Ngân hàng":   "Tiền tệ - Ngân hàng",
    "Trách nhiệm hình sự":   "Trách nhiệm hình sự",
    "Tài chính nhà nước":    "Tài chính nhà nước",
    "Tài nguyên - Môi trường": "Tài nguyên - Môi trường",
    "Vi phạm hành chính":    "Vi phạm hành chính",
    "Văn hóa - Xã hội":      "Văn hóa - Xã hội",
    "Xuất nhập khẩu":        "Xuất nhập khẩu",
    "Xây dựng - Đô thị":     "Xây dựng - Đô thị",
    "Đầu tư":                "Đầu tư",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_corpus(file_path: str) -> list[dict[str, Any]]:
    """Load and normalize corpus_final.json into a list of structured law dicts."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Corpus file not found: {file_path}")

    with open(path, encoding="utf-8") as f:
        raw_data = json.load(f)

    return [_normalize_law(law) for law in raw_data]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalize_law(law: dict) -> dict[str, Any]:
    """Extract and normalize all fields from a raw law entry."""
    meta = law.get("metadata") or {}
    diagram = law.get("diagram_info") or {}

    raw_field = (diagram.get("field") or "").strip()
    legal_fields = _normalize_legal_fields(raw_field)

    return {
        # --- Document identity ---
        "doc_number":        meta.get("doc_number", "").strip(),
        "doc_title":         (law.get("title") or "").strip(),
        "doc_type":          meta.get("doc_type", "").strip(),
        # --- Legal fields (normalized list, e.g. ["Tiền tệ - Ngân hàng", "Lao động - Tiền lương"]) ---
        "legal_fields":      legal_fields,
        # Comma-joined string for page_content display
        "legal_field":       ", ".join(legal_fields),
        # --- Issuance info ---
        "issuing_authority": meta.get("issuing_authority", "").strip(),
        "signer":            meta.get("signer", "").strip(),
        "issue_date":        meta.get("issue_date", "").strip(),
        "effective_date":    meta.get("effective_date", "").strip(),
        "gazette_date":      meta.get("gazette_date", "").strip(),
        "gazette_number":    meta.get("gazette_number", "").strip(),
        "status":            meta.get("status", "").strip(),
        "source_url":        (law.get("source_url") or "").strip(),
        # --- Content units (kept as-is for further processing) ---
        "content":           law.get("content") or [],
    }


def _normalize_legal_fields(raw: str) -> list[str]:
    """
    Split a comma-separated field string and normalize each item.

    - Splits on comma
    - Strips whitespace
    - Applies FIELD_NORMALIZATION_MAP to fix missing diacritics
    - Drops empty values
    - Preserves original order, deduplicates while keeping order

    Example:
        "Tien Te Ngan Hang, Tài chính nhà nước"
        → ["Tiền tệ - Ngân hàng", "Tài chính nhà nước"]
    """
    if not raw:
        return []

    seen: set[str] = set()
    result: list[str] = []
    for part in raw.split(","):
        item = part.strip()
        if not item:
            continue
        # Normalize: map known variants, fallback to original if not in map
        normalized = FIELD_NORMALIZATION_MAP.get(item, item)
        if normalized not in seen:
            seen.add(normalized)
            result.append(normalized)

    return result
