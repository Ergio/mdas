"""PDF processing & chunking module."""

from typing import List
from pathlib import Path
from langchain_community.document_loaders import FileSystemBlobLoader
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import PyPDFParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from config import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP


def load_pdfs(pdf_directory: str = None) -> List[Document]:
    """
    Load PDF documents from specified directory.

    Args:
        pdf_directory: Path to directory containing PDF files (defaults to config)

    Returns:
        List of Document objects containing loaded PDF content

    Raises:
        FileNotFoundError: If directory does not exist
    """
    pdf_directory = pdf_directory or str(DATA_DIR)
    pdf_path = Path(pdf_directory)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF directory not found: {pdf_directory}")

    loader = GenericLoader(
        blob_loader=FileSystemBlobLoader(
            path=str(pdf_path),
            glob="*.pdf",
        ),
        blob_parser=PyPDFParser(),
    )

    docs = loader.load()

    if not docs:
        raise FileNotFoundError(f"No PDF files found in: {pdf_directory}")

    return docs


def chunk_documents(
    docs: List[Document],
    chunk_size: int = None,
    chunk_overlap: int = None
) -> List[Document]:
    """
    Split documents into chunks for better processing.

    Improved chunking strategy:
    - Larger chunks to preserve financial tables and context
    - Increased overlap for better context preservation
    - Better separators hierarchy
    - Track index in original document

    Args:
        docs: List of documents to split
        chunk_size: Size of each chunk (defaults to config)
        chunk_overlap: Overlap between chunks (defaults to config)

    Returns:
        List of chunked Document objects
    """
    chunk_size = chunk_size or CHUNK_SIZE
    chunk_overlap = chunk_overlap or CHUNK_OVERLAP

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n\n", "\n\n", "\n", " ", ""],
        add_start_index=True,
    )

    all_splits = text_splitter.split_documents(docs)
    return all_splits


def process_documents(pdf_directory: str = None) -> List[Document]:
    """
    Main function to load and process PDF documents.

    Args:
        pdf_directory: Path to directory containing PDF files (defaults to config)

    Returns:
        List of processed and chunked Document objects
    """
    docs = load_pdfs(pdf_directory)
    chunks = chunk_documents(docs)
    return chunks
