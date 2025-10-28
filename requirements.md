# AI Agent Coding Assessment: Multi-Document Analysis System

## Overview

Build an AI agent system that analyzes multiple PDF documents and answers complex questions requiring information synthesis across documents.

**Submission:** GitHub repository with code, documentation, and evaluation results

---

## Task Description

Create an intelligent document analysis agent that can:

1. Ingest and process multiple PDF documents
2. Answer questions that require information from multiple sources
3. Provide citations/references to source documents
4. Self-evaluate response quality

---

## Requirements

### 1. Document Processing & RAG Pipeline

- Accept 3-5 PDF documents as input
- Extract and chunk text appropriately
- Implement a vector store for semantic search
- Create embeddings for document chunks
- Implement retrieval mechanism

### 2. Agent Implementation

Your agent should demonstrate:

- **Multi-step reasoning**: Break down complex queries into sub-tasks
- **Tool usage**: Use at least 2 tools/functions (e.g., retrieval, summarization, calculation)
- **Source attribution**: Clearly cite which documents were used
- **Structured outputs**: Return answers in a consistent format

### 4. Free Tier Requirements

Use only free-tier services:

- **LLM**: OpenAI (trial credits), Anthropic (trial), Google Gemini (free tier), or any open model (create .env example, to run with own credentials).
- **Vector DB**: ChromaDB, FAISS, or in-memory solutions
- **Embeddings**: OpenAI embeddings (free trial) or sentence-transformers (local/free)

---

## Deliverables

### 1. Code (Required)

`project/
├── src/
│   ├── agent.py              # Main agent implementation
│   ├── document_processor.py # PDF processing & chunking
│   ├── retrieval.py          # RAG/vector store logic
│   └── tools.py              # Agent tools/functions
├── data/
│   └── sample_pdfs/          # 3-5 sample PDFs you tested with
├── results/
│   └── evaluation_results.json # Output from your evaluation
├── requirements.txt
└── README.md`

### 2. Documentation (README.md should include)

- Setup instructions
- Architecture overview with diagram/description
- Design decisions and trade-offs
- How to run the system
- How to run tests and evaluation
- Sample queries and outputs
- Limitations and future improvements

### 3. Evaluation Report

Include in your repo or README:

- Excpectation outputs summary
- Performance metrics (latency, accuracy, etc.)
- Example outputs showing agent reasoning trace
- Analysis of failure cases (if any)

---

## Evaluation Criteria

### Code Quality (40%)

- Clean, readable, well-structured code
- Proper error handling
- Type hints and documentation
- Follows Python best practices

### Agent Design (35%)

- Effective multi-step reasoning
- Appropriate tool usage
- Good prompt engineering
- Handles edge cases

### RAG Implementation (25%)

- Effective chunking strategy
- Quality of retrieval
- Proper citation mechanism
- Handles multi-document synthesis

---

## Sample Challenge Questions

Your agent should handle questions like:

1. "Compare the key findings from Document A and Document B on [topic]"
2. "What are the contradictions between the three documents regarding [specific claim]?"
3. "Based on all documents, what is the timeline of events?"
4. "Which document provides the most detailed explanation of [concept]?"
5. "Summarize the consensus view across all documents on [topic]"

---

## Getting Started Tips

1. **Start simple**: Get basic RAG working first, then add agent capabilities
2. **Use existing frameworks**: LangChain, LlamaIndex, or build from scratch (your choice)
3. **Focus on evaluation**: This is critical for production systems
4. **Document decisions**: Explain your reasoning in code comments and README

---

## Submission

1. Push code to a public GitHub repository
2. Include clear README with setup instructions
3. Ensure we can run your code with: `pip install -r requirements.txt && python main.py`
4. Include sample PDFs or instructions for obtaining them