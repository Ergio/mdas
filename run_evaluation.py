"""
Run RAGAS evaluation on the Multi-Document Analysis System.
"""

import os
from dotenv import load_dotenv
from src.agent import MultiDocumentAgent
from src.evaluation import evaluate_agent, save_results


def main():
    """Run evaluation."""
    # Load environment variables
    load_dotenv()

    # Verify API key is set
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    print("Initializing agent...")
    agent = MultiDocumentAgent()

    print("\nStarting evaluation...")
    results = evaluate_agent(agent)

    # Save results to JSON
    filepath = save_results(results)
    print(f"\n💾 Results saved to: {filepath}")


if __name__ == "__main__":
    main()
