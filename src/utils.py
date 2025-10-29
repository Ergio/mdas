"""Utility functions for the Multi-Document Analysis System."""

from typing import List
from pathlib import Path
from langchain_core.documents import Document


def filter_by_document(docs: List[Document], document_name: str, limit: int = 5) -> List[Document]:
    """
    Filter documents by document name and limit results.

    Args:
        docs: List of documents to filter
        document_name: Document name to filter by
        limit: Maximum number of documents to return

    Returns:
        Filtered list of documents
    """
    filtered = [
        doc for doc in docs
        if document_name.lower() in doc.metadata.get('source', '').lower()
    ]
    return filtered[:limit]


def get_document_name(source_path: str) -> str:
    """
    Extract document name from source path (cross-platform).

    Args:
        source_path: Full path to document

    Returns:
        Document filename
    """
    return Path(source_path).name


def serialize_documents(docs: List[Document]) -> str:
    """
    Serialize documents with metadata for display.

    Args:
        docs: List of documents to serialize

    Returns:
        Formatted string with document content and metadata
    """
    return "\n\n".join(
        f"[Document: {get_document_name(doc.metadata.get('source', 'unknown'))} | "
        f"Page: {doc.metadata.get('page', '?')}]\n"
        f"{doc.page_content}"
        for doc in docs
    )
