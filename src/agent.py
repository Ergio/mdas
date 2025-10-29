"""Main agent implementation module."""

from typing import Dict, Any
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langgraph.errors import GraphRecursionError
from src.document_processor import process_documents
from src.retrieval import VectorStoreRetriever
from src.tools import get_available_tools, set_retriever
from config import LLM_MODEL, DATA_DIR


# Core system prompt (domain-agnostic)
BASE_SYSTEM_PROMPT = """You are a precise document analysis assistant.

Core Guidelines:
- Extract exact figures and facts from provided documents
- Use the retrieval tool with document_name parameter for targeted queries
- Use the summarize_document tool to get comprehensive document summaries
- Provide concise, direct answers without elaboration
- Format: State facts directly, avoid explanatory prose
- Source attribution: Clearly cite which documents were used
- When comparing multiple sources, query each separately

Tools Available:
1. retrieve_context - Search for specific information in documents
2. summarize_document - Get a comprehensive summary of an entire document

Few-Shot Examples:

Q: What was Company A's revenue in Q3?
A: $5.2B [CompanyA.pdf, p.2]

Q: Compare margins for Company A vs Company B.
A: Company A: 15.5% [CompanyA.pdf, p.3], Company B: 12.3% [CompanyB.pdf, p.1]. Company A higher.

Q: Summarize the key points from Accenture.pdf
A: [Uses summarize_document tool to generate comprehensive summary]"""

# Domain-specific instructions (financial earnings analysis)
DOMAIN_INSTRUCTIONS = """
Domain Context: Q3 FY2025 Financial Earnings Analysis

Key Requirements:
1. Distinguish actual results from forecasts/guidance
2. Extract metrics: revenue growth (%), operating margins (%), free cash flow, net income
3. Include currency symbols when available
4. Use date format: Month DD, YYYY
5. Keep answers brief - state numbers directly without lengthy explanations
6. Comparison format: "Company A: [metric], Company B: [metric]. [Conclusion]"

Tool Selection Guidelines:
- Use retrieve_context for specific queries (e.g., "What was Q3 revenue?")
- Use summarize_document for overview requests (e.g., "Summarize key findings", "What are the main points?")
- Pass additional instructions in the 'notes' parameter when summarizing (e.g., notes="Focus on financial metrics")

Response Style Examples:
- Good: "Accenture +8% USD, Siemens +3% actual. Accenture higher."
- Bad: "Accenture's Q3 FY2025 revenue growth was 8% in U.S. dollars... (lengthy explanation)"
"""


def build_system_prompt(documents: list = None) -> str:
    """
    Build system prompt with dynamically generated document list.

    Args:
        documents: List of processed documents

    Returns:
        Complete system prompt string
    """
    prompt = f"{BASE_SYSTEM_PROMPT}\n{DOMAIN_INSTRUCTIONS}"

    if documents:
        # Extract unique document names
        doc_names = sorted(set(doc.metadata.get('source', 'Unknown') for doc in documents))
        doc_list = "\n".join(f"- {name}" for name in doc_names)
        prompt += f"\nAvailable Documents:\n{doc_list}"

    return prompt


class MultiDocumentAgent:
    """
    AI Agent for multi-document analysis and question answering.
    """

    def __init__(
        self,
        model_name: str = None,
        pdf_directory: str = None
    ):
        """
        Initialize the agent.

        Args:
            model_name: Name of the LLM model to use (defaults to config)
            pdf_directory: Path to directory containing PDF files (defaults to config)
        """
        model_name = model_name or LLM_MODEL
        pdf_directory = pdf_directory or str(DATA_DIR)

        self.model = init_chat_model(model_name)
        self.documents = process_documents(pdf_directory)
        self.retriever = VectorStoreRetriever()
        self.retriever.add_documents(self.documents)
        set_retriever(self.retriever)
        self.tools = get_available_tools()
        system_prompt = build_system_prompt(self.documents)
        self.agent = create_agent(
            self.model,
            self.tools,
            system_prompt=system_prompt
        )

    def query(self, question: str, stream: bool = True) -> Dict[str, Any]:
        """
        Ask the agent a question.

        Args:
            question: The question to ask
            stream: Whether to stream the response

        Returns:
            Dictionary containing the agent's response
        """
        config = {"recursion_limit": 50}

        try:
            if stream:

                for event in self.agent.stream(
                    {"messages": [{"role": "user", "content": question}]},
                    stream_mode="values",
                    config=config,
                ):
                    event["messages"][-1].pretty_print()

                return event
            else:
                response = self.agent.invoke(
                    {"messages": [{"role": "user", "content": question}]},
                    config=config,
                )
                return response
        except GraphRecursionError as e:
            error_message = (
                "The agent encountered too many processing steps while answering your question. "
                "This usually happens when the query is too complex or the agent gets stuck in a loop. "
                "Please try:\n"
                "- Simplifying your question\n"
                "- Breaking it into smaller, more specific questions\n"
                "- Being more explicit about which documents to search"
            )

            # Return error in the same format as normal responses
            return {
                "messages": [{
                    "type": "ai",
                    "content": error_message
                }],
                "error": str(e)
            }
