"""Centralized configuration for Multi-Document Analysis System."""

import os
from pathlib import Path

# Directories
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data" / "sample_pdfs"
RESULTS_DIR = PROJECT_ROOT / "results"

# Model Configuration
LLM_MODEL = "openai:gpt-5-2025-08-07"
LLM_MODEL_MINI = "gpt-5-mini-2025-08-07"
EMBEDDING_MODEL = "text-embedding-3-large"

# Document Processing
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 400
SUMMARY_CHUNK_SIZE = 12000
SUMMARY_CHUNK_OVERLAP = 500

# Retrieval Configuration
DEFAULT_K = 10
FILTERED_K = 20
MMR_FETCH_K = 50
MMR_LAMBDA = 0.5
FILTERED_TOP_K = 5

# API Configuration
def get_api_key() -> str:
    """Get OpenAI API key from environment."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your_openai_api_key_here":
        raise ValueError("OPENAI_API_KEY not set or invalid")
    return api_key
