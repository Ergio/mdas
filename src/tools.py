"""Agent tools/functions module."""

from typing import Tuple, List
from langchain.tools import tool
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.retrieval import VectorStoreRetriever
from src.document_processor import load_pdfs
import os


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


@tool
def summarize_document(document_name: str, notes: str = "") -> str:
    """
    Summarize a document using hierarchical summarization.

    This tool loads a document, splits it into chunks, summarizes each chunk,
    then combines and summarizes the chunk summaries to create a final summary.

    Args:
        document_name: Name of the document to summarize (e.g., 'Accenture.pdf')
        notes: Optional notes or instructions for summarization (e.g., 'Focus on financial metrics', 'Extract key findings')

    Returns:
        Final summary of the document
    """

    llm = ChatOpenAI(
        model="gpt-5-mini-2025-08-07",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY")
    )


    pdf_directory = "./data/sample_pdfs/"
    all_docs = load_pdfs(pdf_directory)

    target_docs = [doc for doc in all_docs if document_name in doc.metadata.get('source', '')]

    if not target_docs:
        return f"Document '{document_name}' not found in {pdf_directory}"

    full_text = "\n\n".join([doc.page_content for doc in target_docs])

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=12000,  # ~3000 tokens
        chunk_overlap=500,
        separators=["\n\n\n", "\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text(full_text)

    if len(chunks) == 0:
        return "Document is empty or could not be processed"

    notes_instruction = f"\n\nAdditional instructions: {notes}" if notes else ""

    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        prompt = f"""Summarize the following text concisely, preserving key information, facts, and numbers:{notes_instruction}

Text:
{chunk}

Summary:"""

        response = llm.invoke(prompt)
        chunk_summaries.append(response.content)

    # Step 3: If we have multiple chunk summaries, combine and summarize again
    if len(chunk_summaries) == 1:
        final_summary = chunk_summaries[0]
    else:
        # Combine all chunk summaries
        combined_summaries = "\n\n---\n\n".join(chunk_summaries)

        # If combined summaries are too long, split and summarize again (recursive)
        if len(combined_summaries) > 48000:  # ~12000 tokens
            summary_chunks = text_splitter.split_text(combined_summaries)
            intermediate_summaries = []

            for summary_chunk in summary_chunks:
                prompt = f"""Combine and summarize the following summaries into a coherent summary:{notes_instruction}

Summaries:
{summary_chunk}

Combined Summary:"""

                response = llm.invoke(prompt)
                intermediate_summaries.append(response.content)

            combined_summaries = "\n\n---\n\n".join(intermediate_summaries)

        # Final summarization
        final_prompt = f"""Create a comprehensive final summary by combining the following summaries.
Preserve important details, key findings, and numerical data:{notes_instruction}

Summaries:
{combined_summaries}

Final Summary:"""

        final_response = llm.invoke(final_prompt)
        final_summary = final_response.content

    return f"Summary of {document_name}:\n\n{final_summary}"


def get_available_tools():
    """
    Get list of available tools for the agent.

    Returns:
        List of tool functions
    """
    return [retrieve_context, summarize_document]
