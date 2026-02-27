import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger
from config.settings import settings


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ChunkDocument:
    """A single indexable chunk with rich metadata."""

    # Content
    text: str

    # Document-level metadata
    source_url: str
    doc_title: str
    doc_number: str
    doc_type: str
    issuing_authority: str
    signer: str
    issue_date: str
    effective_date: str
    gazette_date: str
    gazette_number: str
    doc_status: str
    legal_field: str          # from diagram_info.field
    summary: str              # document summary

    # Unit-level metadata
    unit_id: str              # e.g. "dieu_1"
    unit_title: str           # e.g. "Điều 1. Sửa đổi..."

    # Chunk-level metadata
    has_table: bool = False
    chunk_index: int = 0      # sequential index within the whole corpus
    unit_chunk_index: int = 0 # index of this chunk within its unit (if split)

    def to_payload(self) -> dict:
        """Serialise to Qdrant payload dict."""
        return asdict(self)


# ---------------------------------------------------------------------------
# CorpusLoader
# ---------------------------------------------------------------------------

class CorpusLoader:
    """
    Load corpus.json and convert every legal unit (Điều/Khoản)
    into one or more ChunkDocuments ready for indexing.

    Pipeline
    --------
    load() → List[raw dicts]
    to_chunks() → List[ChunkDocument]
        For each document:
            For each unit:
                merge_and_split_unit() → List[str]  (1..N chunks)
                wrap each text with full metadata → ChunkDocument
    """

    def __init__(self, corpus_path: str = None):
        self.corpus_path = corpus_path or settings.docs_path
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> list:
        """Return raw list of document dicts from the JSON file."""
        with open(self.corpus_path, encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} documents from {self.corpus_path}")
        return data

    def to_chunks(self, documents: list) -> List[ChunkDocument]:
        """Convert raw document list to flat list of ChunkDocuments."""
        chunks: List[ChunkDocument] = []
        global_index = 0

        for doc in documents:
            doc_meta = self._extract_doc_meta(doc)
            for unit in doc.get("content", []):
                unit_texts, has_table = self._merge_and_split_unit(unit)
                for unit_chunk_idx, text in enumerate(unit_texts):
                    chunks.append(
                        ChunkDocument(
                            text=text,
                            has_table=has_table,
                            chunk_index=global_index,
                            unit_chunk_index=unit_chunk_idx,
                            unit_id=unit.get("unit_id", ""),
                            unit_title=unit.get("unit_title", ""),
                            **doc_meta,
                        )
                    )
                    global_index += 1

        logger.info(
            f"Created {len(chunks)} chunks "
            f"({sum(1 for c in chunks if c.has_table)} with tables)"
        )
        return chunks

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_doc_meta(self, doc: dict) -> dict:
        """Pull document-level fields into a flat dict for ChunkDocument."""
        meta = doc.get("metadata", {})
        diagram = doc.get("diagram_info", {})
        return {
            "source_url": doc.get("source_url", ""),
            "doc_title": doc.get("title", ""),
            "doc_number": meta.get("doc_number", ""),
            "doc_type": meta.get("doc_type", ""),
            "issuing_authority": meta.get("issuing_authority", ""),
            "signer": meta.get("signer", ""),
            "issue_date": meta.get("issue_date", ""),
            "effective_date": meta.get("effective_date", ""),
            "gazette_date": meta.get("gazette_date", ""),
            "gazette_number": meta.get("gazette_number", ""),
            "doc_status": meta.get("status", ""),
            "legal_field": diagram.get("field", ""),
            "summary": doc.get("summary", ""),
        }

    def _merge_and_split_unit(self, unit: dict) -> tuple[List[str], bool]:
        """
        Merge all content blocks of a unit into one text string,
        then split with RecursiveCharacterTextSplitter if needed.

        Returns
        -------
        (list_of_text_chunks, has_table)
        """
        parts: List[str] = []
        has_table = False

        unit_title = unit.get("unit_title", "")
        if unit_title:
            parts.append(unit_title)

        for block in unit.get("unit_content", []):
            block_type = block.get("type", "text")
            data = block.get("data", "").strip()

            if not data:
                continue

            if block_type == "text":
                parts.append(data)
            elif block_type == "table_html":
                has_table = True
                markdown_table = self._html_table_to_markdown(data)
                if markdown_table:
                    parts.append(markdown_table)

        merged = "\n\n".join(parts)

        # Only split if the merged text exceeds chunk_size
        if len(merged) <= settings.chunk_size:
            return [merged], has_table

        split_texts = self._splitter.split_text(merged)
        return split_texts, has_table

    def _html_table_to_markdown(self, html: str) -> str:
        """
        Convert an HTML table string into a GitHub-Flavored Markdown table.
        Strips all HTML styling; retains only cell text content.
        """
        try:
            soup = BeautifulSoup(html, "html.parser")
            rows = soup.find_all("tr")
            if not rows:
                return soup.get_text(separator=" ", strip=True)

            md_rows: List[str] = []
            for i, row in enumerate(rows):
                cells = [
                    cell.get_text(separator=" ", strip=True)
                    for cell in row.find_all(["td", "th"])
                ]
                md_rows.append("| " + " | ".join(cells) + " |")
                if i == 0:
                    md_rows.append("|" + "|".join(["---"] * len(cells)) + "|")

            return "\n".join(md_rows)

        except Exception as e:
            logger.warning(f"Failed to parse HTML table: {e}")
            # Fallback: strip all tags
            return BeautifulSoup(html, "html.parser").get_text(separator=" ", strip=True)
