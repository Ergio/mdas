"""RAG/vector store logic module."""

from typing import List
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from src.utils import filter_by_document
from config import EMBEDDING_MODEL, FILTERED_TOP_K


class VectorStoreRetriever:
    """
    Vector store retriever for semantic search over document chunks.
    """

    def __init__(self, embedding_model: str = None):
        """
        Initialize the vector store retriever.

        Args:
            embedding_model: Name of the OpenAI embedding model to use (defaults to config)
        """
        embedding_model = embedding_model or EMBEDDING_MODEL
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.vector_store = InMemoryVectorStore(self.embeddings)

    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add documents to the vector store.

        Args:
            documents: List of Document objects to add

        Returns:
            List of document IDs
        """
        document_ids = self.vector_store.add_documents(documents=documents)
        return document_ids

    def mmr_search(
        self,
        query: str,
        k: int = 10,
        fetch_k: int = 50,
        lambda_mult: float = 0.5,
        document_name: str = None
    ) -> List[Document]:
        """
        Search using Maximal Marginal Relevance for diversity.

        Args:
            query: The search query
            k: Number of documents to retrieve
            fetch_k: Number of candidates to fetch before MMR selection
            lambda_mult: Balance between relevance and diversity (0-1)
            document_name: Optional document name to filter results

        Returns:
            List of relevant Document objects
        """
        try:
            retrieved_docs = self.vector_store.max_marginal_relevance_search(
                query,
                k=k,
                fetch_k=fetch_k,
                lambda_mult=lambda_mult
            )
        except (AttributeError, NotImplementedError):
            # Fallback to similarity search if MMR not supported
            retrieved_docs = self.vector_store.similarity_search(query, k=k)

        # Filter by document name if provided
        if document_name:
            retrieved_docs = filter_by_document(retrieved_docs, document_name, FILTERED_TOP_K)

        return retrieved_docs
