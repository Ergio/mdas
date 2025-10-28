"""
Main entry point for the Multi-Document Analysis System.
Run with: python main.py
"""

import os
import sys
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from src.agent import MultiDocumentAgent


def main():
    """Main function to run the agent system."""
    # Load environment variables
    load_dotenv()

    # Verify API key is set
    if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") == "your_openai_api_key_here":
        print("ERROR: OPENAI_API_KEY not found or not set properly!")
        print("Please set your OpenAI API key in the .env file")
        print("\nSteps to fix:")
        print("1. Open the .env file")
        print("2. Replace 'your_openai_api_key_here' with your actual OpenAI API key")
        print("3. Get your API key from: https://platform.openai.com/api-keys")
        sys.exit(1)

    print("="*80)
    print("Multi-Document Analysis System")
    print("="*80)
    print()

    try:
        # Initialize agent
        agent = MultiDocumentAgent()

        print("\n" + "="*80)
        print("Running example queries...")
        print("="*80)

        # Example queries
        example_queries = [
            "Compare the revenue growth in Q3 FY2025 from Accenture.pdf and Siemens.pdf. Which company had higher growth?",
            "What are the operating margin figures for Accenture.pdf, Siemens.pdf, and Infineon.pdf in Q3 FY2025?",
        ]

        # Run example queries
        for i, query in enumerate(example_queries, 1):
            print(f"\n{'='*80}")
            print(f"Example Query {i}/{len(example_queries)}")
            print(f"{'='*80}")
            agent.query(query, stream=True)

        print("\n" + "="*80)
        print("All example queries completed!")
        print("="*80)

    except Exception as e:
        print(f"\nERROR: {str(e)}")
        print("\nPlease check:")
        print("1. Your .env file is configured correctly")
        print("2. PDF files exist in ./data/sample_pdfs/")
        print("3. All dependencies are installed: pip install -r requirements.txt")
        sys.exit(1)


if __name__ == "__main__":
    main()
