from src.paralegal_agent.data.document_loader import load_corpus
from src.paralegal_agent.data.unit_merger import merge_units
from src.paralegal_agent.data.legal_splitter import split_law

def build_documents(corpus_path: str, limit: int | None = None):

    print(f"Loading corpus: {corpus_path}")
    laws = load_corpus(corpus_path)
    if limit:
        laws = laws[:limit]
    print(f" {len(laws)} laws loaded")

    all_documents = []
    for law in laws:
        merged = merge_units(law["content"])
        docs = split_law(law, merged)
        all_documents.extend(docs)

    print(f"   {len(all_documents)} chunks total")
    return all_documents
    
import json
from langchain_core.documents import Document

def save_documents_jsonl(docs: list[Document], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for doc in docs:
            record = {
                "page_content": doc.page_content,
                "metadata": doc.metadata,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
import pandas as pd


if __name__ == "__main__":
    docs = build_documents("data/corpus_final.json")

    print(docs[0])
    save_documents_jsonl(docs, "data/documents.jsonl")
