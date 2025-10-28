"""PDF processing & chunking module."""

from typing import List
from langchain_community.document_loaders import FileSystemBlobLoader
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import PyPDFParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def load_pdfs(pdf_directory: str = "./data/sample_pdfs/") -> List[Document]:
    """
    Load PDF documents from specified directory.

    Args:
        pdf_directory: Path to directory containing PDF files

    Returns:
        List of Document objects containing loaded PDF content
    """
    loader = GenericLoader(
        blob_loader=FileSystemBlobLoader(
            path=pdf_directory,
            glob="*.pdf",
        ),
        blob_parser=PyPDFParser(),
    )

    docs = loader.load()
    return docs


def chunk_documents(
    docs: List[Document],
    chunk_size: int = 2000,
    chunk_overlap: int = 400
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
        chunk_size: Size of each chunk (default 2000)
        chunk_overlap: Overlap between chunks (default 400)

    Returns:
        List of chunked Document objects
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n\n", "\n\n", "\n", " ", ""],
        add_start_index=True,
    )

    all_splits = text_splitter.split_documents(docs)
    return all_splits


def process_documents(pdf_directory: str = "./data/sample_pdfs/") -> List[Document]:
    """
    Main function to load and process PDF documents.

    Args:
        pdf_directory: Path to directory containing PDF files

    Returns:
        List of processed and chunked Document objects
    """
    docs = load_pdfs(pdf_directory)
    chunks = chunk_documents(docs)
    print(f"Processed {len(docs)} documents into {len(chunks)} chunks")
    return chunks
