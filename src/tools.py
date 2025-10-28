"""Agent tools/functions module."""

from typing import Tuple, List
from langchain.tools import tool
from langchain_core.documents import Document
from src.retrieval import VectorStoreRetriever


# Global retriever instance (will be initialized by agent)
_retriever: VectorStoreRetriever = None


def set_retriever(retriever: VectorStoreRetriever):
    """
    Set the global retriever instance.

    Args:
        retriever: VectorStoreRetriever instance to use
    """
    global _retriever
    _retriever = retriever


@tool(response_format="content_and_artifact")
def retrieve_context(query: str, document_name: str = None) -> Tuple[str, List[Document]]:
    """
    Retrieve information to help answer a query.

    Improved retrieval strategy:
    - Uses MMR (Maximal Marginal Relevance) for diversity
    - Filters by document name when specified
    - Enhanced serialization with metadata

    Args:
        query: The search query
        document_name: Optional document name to filter results (e.g., 'Accenture.pdf', 'Siemens.pdf', 'Infineon.pdf')

    Returns:
        Tuple of (serialized context string, list of retrieved documents)
    """
    if _retriever is None:
        raise RuntimeError("Retriever not initialized. Call set_retriever() first.")

    # Increase k significantly to get better coverage
    k = 20 if document_name else 10

    # Use MMR (Maximal Marginal Relevance) for diversity
    retrieved_docs = _retriever.mmr_search(
        query=query,
        k=k,
        fetch_k=50,
        lambda_mult=0.5,
        document_name=document_name
    )

    # Enhanced serialization with metadata
    serialized = "\n\n".join(
        (f"[Document: {doc.metadata.get('source', 'unknown').split('\\\\')[-1]} | "
         f"Page: {doc.metadata.get('page', '?')}]\n"
         f"{doc.page_content}")
        for doc in retrieved_docs
    )

    return serialized, retrieved_docs


def get_available_tools():
    """
    Get list of available tools for the agent.

    Returns:
        List of tool functions
    """
    return [retrieve_context]
