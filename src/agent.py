"""Main agent implementation module."""

import os
from typing import Dict, Any
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from src.document_processor import process_documents
from src.retrieval import VectorStoreRetriever
from src.tools import get_available_tools, set_retriever


# System prompt for the agent
SYSTEM_PROMPT = """You are a precise financial analyst assistant analyzing Q3 FY2025 earnings reports.

🎯 CRITICAL INSTRUCTIONS:

1. **ACTUAL vs FORECAST**: When asked about Q3 FY2025 results:
   - Look for ACTUAL quarterly results (tables with "Q3 FY 2025", "Third Quarter Fiscal 2025")
   - IGNORE full-year outlook/guidance sections (these are forecasts, not Q3 actuals)
   - Search for sections titled: "Q3 FY25 Financial Review", "Third Quarter Results", "Consolidated Statement"

2. **Extract EXACT numbers**:
   - Revenue growth: Look for "revenue increased X%" or "revenue growth of X%"
   - Operating/Segment margins: Look for "operating margin X%", "Segment Result Margin X%"
   - Free cash flow: Look for "Free Cash Flow €X billion" or "$X billion"
   - Net income: Look for "Net income €X" or "Profit for the period"

3. **Currency specification**:
   - Accenture.pdf: Uses USD ($)
   - Siemens.pdf: Uses EUR (€)
   - Infineon.pdf: Uses EUR (€)
   - ALWAYS include currency symbol in your answer

4. **Comparison format**: For multi-company questions, use this structure:
   "Company A: [exact metric], Company B: [exact metric]. [Conclusion]"
   Example: "Accenture: +8% USD, Siemens: +3% actual. Accenture had higher growth."

5. **Date format**: When reporting dates, use format "Month DD, YYYY"
   Example: "June 20, 2025"

6. **Use retrieval tool smartly**:
   - For company-specific questions, ALWAYS use document_name parameter
   - Example: retrieve_context(query="Q3 revenue growth", document_name="Accenture.pdf")
   - For comparisons, make separate calls for each company

Available documents:
- Accenture.pdf (Currency: USD $)
- Siemens.pdf (Currency: EUR €)
- Infineon.pdf (Currency: EUR €)

Provide factual, precise answers with exact figures. Be concise and direct."""


class MultiDocumentAgent:
    """
    AI Agent for multi-document analysis and question answering.
    """

    def __init__(
        self,
        model_name: str = "openai:gpt-4.1",
        pdf_directory: str = "./data/sample_pdfs/"
    ):
        """
        Initialize the agent.

        Args:
            model_name: Name of the LLM model to use
            pdf_directory: Path to directory containing PDF files
        """
        # Initialize LLM
        self.model = init_chat_model(model_name)

        # Process documents
        print("Processing documents...")
        self.documents = process_documents(pdf_directory)

        # Initialize retriever
        print("Initializing vector store...")
        self.retriever = VectorStoreRetriever()
        self.retriever.add_documents(self.documents)

        # Set global retriever for tools
        set_retriever(self.retriever)

        # Get tools
        self.tools = get_available_tools()

        # Create agent
        print("Creating agent...")
        self.agent = create_agent(
            self.model,
            self.tools,
            system_prompt=SYSTEM_PROMPT
        )

        print("Agent initialized successfully!")

    def query(self, question: str, stream: bool = True) -> Dict[str, Any]:
        """
        Ask the agent a question.

        Args:
            question: The question to ask
            stream: Whether to stream the response

        Returns:
            Dictionary containing the agent's response
        """
        if stream:
            print(f"\n{'='*80}")
            print(f"Question: {question}")
            print(f"{'='*80}\n")

            for event in self.agent.stream(
                {"messages": [{"role": "user", "content": question}]},
                stream_mode="values",
            ):
                event["messages"][-1].pretty_print()

            return event
        else:
            response = self.agent.invoke(
                {"messages": [{"role": "user", "content": question}]}
            )
            return response


def main():
    """
    Main function to run the agent interactively.
    """
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()

    # Verify API key is set
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    # Initialize agent
    agent = MultiDocumentAgent()

    # Example queries
    example_queries = [
        "Compare the revenue growth in Q3 FY2025 from Accenture.pdf and Siemens.pdf. Which company had higher growth?",
        "What are the operating margin figures for Accenture.pdf, Siemens.pdf, and Infineon.pdf in Q3 FY2025?",
    ]

    # Run example queries
    for query in example_queries:
        agent.query(query, stream=True)
        print("\n")


if __name__ == "__main__":
    main()
