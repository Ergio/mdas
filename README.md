# Multi-Document Analysis System
An AI-powered system for analyzing and querying multiple PDF documents using RAG (Retrieval Augmented Generation) and LangChain agents.

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env`:
```
OPENAI_API_KEY=your_openai_api_key_here
```

## Run

```bash
streamlit run main.py
```

## Evaluation

Run RAGAS evaluation:

```bash
python run_evaluation.py
```

Results saved to `results/evaluation_results_YYYYMMDD_HHMMSS.json`

### Metrics

- **Correctness**: Answer accuracy vs ground truth
- **Faithfulness**: Answer grounded in context
- **Precision**: Context relevance
- **Relevancy**: Answer relevance to query




## Architecture Overview

### System Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                          User Interface                             │
│                     (Streamlit Web App)                             │
│                          main.py                                    │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             │ User Query
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      MultiDocumentAgent                             │
│                         (agent.py)                                  │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  System Prompt                                                │  │
│  │  - Base instructions for document analysis                    │  │
│  │  - Domain-specific financial analysis guidelines              │  │
│  │  - Tool selection logic                                       │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                             │                                        │
│              ┌──────────────┴──────────────┐                        │
│              │                             │                        │
│              ▼                             ▼                        │
│  ┌──────────────────────┐      ┌─────────────────────┐             │
│  │   retrieve_context   │      │ summarize_document  │             │
│  │     (tools.py)       │      │     (tools.py)      │             │
│  └──────────┬───────────┘      └──────────┬──────────┘             │
└─────────────┼──────────────────────────────┼─────────────────────────┘
              │                              │
              │                              │
              ▼                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   Retrieval & Processing Layer                      │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │           VectorStoreRetriever (retrieval.py)                │  │
│  │  ┌────────────────────────────────────────────────────────┐  │  │
│  │  │  InMemoryVectorStore + OpenAI Embeddings              │  │  │
│  │  │  - MMR search for diverse results                      │  │  │
│  │  │  - Document filtering by name                          │  │  │
│  │  └────────────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │      Document Processor (document_processor.py)              │  │
│  │  ┌────────────────────────────────────────────────────────┐  │  │
│  │  │  PDF Loading → Chunking → Metadata Enrichment         │  │  │
│  │  │  - PyPDFParser for PDF extraction                      │  │  │
│  │  │  - RecursiveCharacterTextSplitter (1500/200)          │  │  │
│  │  │  - Source tracking and page numbers                    │  │  │
│  │  └────────────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                             ▲
                             │
                             │ PDF Files
                             │
┌─────────────────────────────────────────────────────────────────────┐
│                        Data Directory                               │
│                    (Accenture.pdf, Siemens.pdf, Infineon.pdf)       │
└─────────────────────────────────────────────────────────────────────┘
```

### Components

#### 1. **User Interface Layer** ([main.py](main.py))
   - **Streamlit web application** providing chat interface
   - Displays user messages, assistant responses, tool calls, and tool outputs
   - Manages chat history and session state
   - Shows available documents in sidebar

#### 2. **Agent Layer** ([src/agent.py](src/agent.py))
   - **MultiDocumentAgent**: Orchestrates the entire analysis workflow
   - **LLM Integration**: Uses GPT-5 (2025-08-07) with custom system prompts
   - **Dual prompting strategy**:
     - Base prompt: Domain-agnostic document analysis guidelines
     - Domain prompt: Financial metrics extraction and formatting rules
   - **Tool orchestration**: Decides which tools to call based on query type
   - **Error handling**: Manages recursion limits and graceful failures

#### 3. **Tool Layer** ([src/tools.py](src/tools.py))
   - **retrieve_context**: Semantic search for specific information
     - Uses MMR (Maximal Marginal Relevance) for diverse results
     - Supports document-specific filtering
     - Returns serialized context with metadata
   - **summarize_document**: Hierarchical document summarization
     - Multi-stage summarization for large documents
     - Customizable focus via notes parameter
     - Preserves key facts and numerical data

#### 4. **Retrieval Layer** ([src/retrieval.py](src/retrieval.py))
   - **VectorStoreRetriever**: Manages semantic search
   - **InMemoryVectorStore**: Stores document embeddings
   - **OpenAI Embeddings**: text-embedding-3-large model
   - **MMR Search**: Balances relevance and diversity (λ=0.5)
   - **Document filtering**: Filters results by source document

#### 5. **Document Processing Layer** ([src/document_processor.py](src/document_processor.py))
   - **PDF Loading**: PyPDFParser for text extraction
   - **Text Chunking**: RecursiveCharacterTextSplitter
     - Chunk size: 2000 characters
     - Overlap: 400 characters
     - Smart separators: paragraphs → lines → spaces
   - **Metadata**: Tracks source file and page numbers

#### 6. **Configuration** ([config.py](config.py))
   - Centralized settings for models, directories, and parameters
   - Environment variable management
   - Hyperparameter tuning (chunk sizes, retrieval k values, etc.)

### Data Flow

1. **Initialization**:
   - Load PDFs from `data/` directory
   - Split into chunks with metadata
   - Generate embeddings and store in vector database
   - Initialize agent with tools and system prompt

2. **Query Processing**:
   - User submits question via Streamlit interface
   - Agent analyzes query and selects appropriate tool(s)
   - Tools retrieve context or generate summaries
   - Agent synthesizes final response
   - Response displayed with source attribution

3. **Retrieval Strategy**:
   - Query → Embedding → MMR Search → Filter by document → Top-k chunks
   - Serialization with source metadata for citation

4. **Summarization Strategy**:
   - Load full document → Chunk → Summarize each chunk → Combine → Final summary
   - Recursive summarization for very large documents


## Design Decisions and Trade-offs

### 1. **Agent Architecture over Simple RAG**
   - **Decision**: Use LangChain agents with tool calling instead of simple query-retrieval-generation
   - **Rationale**: Enables dynamic tool selection, multi-step reasoning, and complex queries (e.g., "compare X across documents")
   - **Trade-off**: Higher latency and cost vs. simpler RAG, but significantly better for complex analytical queries

### 2. **MMR (Maximal Marginal Relevance) Search**
   - **Decision**: Use MMR with λ=0.5 instead of pure similarity search
   - **Rationale**: Reduces redundancy in retrieved chunks, provides diverse context
   - **Trade-off**: Slightly more computation, but improves answer quality by avoiding repetitive information

### 3. **In-Memory Vector Store**
   - **Decision**: Use `InMemoryVectorStore` instead of persistent databases (Pinecone, Chroma, etc.)
   - **Rationale**: Simplicity, zero infrastructure, fast prototyping, suitable for small document sets
   - **Trade-off**: Not scalable beyond ~100 documents, requires reprocessing on restart, no distributed access
   - **Future consideration**: Migrate to persistent store for production use

### 4. **Hierarchical Summarization**
   - **Decision**: Chunk → Summarize → Combine → Final summary (map-reduce pattern)
   - **Rationale**: Handles documents exceeding context window, preserves details better than truncation
   - **Trade-off**: Multiple LLM calls increase cost and latency, but necessary for long documents

### 5. **Dual System Prompts (Base + Domain)**
   - **Decision**: Separate base instructions from domain-specific guidelines
   - **Rationale**: Modularity allows easy adaptation to different domains (legal, medical, etc.)
   - **Trade-off**: More prompt engineering required, but improves maintainability and reusability

### 6. **Chunk Size: 2000 characters with 400 overlap**
   - **Decision**: Larger chunks (2000) vs. common smaller sizes (512-1000)
   - **Rationale**: Preserves financial tables, maintains context for multi-line data
   - **Trade-off**: Fewer chunks retrieved per query (less diversity), but better context preservation

### 7. **GPT-5 for Agent, GPT-5-mini for Summarization**
   - **Decision**: Use GPT-5 (2025-08-07) for main agent reasoning, GPT-5-mini for hierarchical summarization
   - **Rationale**: Balance quality and cost - advanced reasoning for query analysis, efficient model for summarization workloads
   - **Trade-off**: Higher cost than using mini for everything, but better agent performance for complex queries

### 8. **Streamlit for UI**
   - **Decision**: Streamlit instead of React/FastAPI/Gradio
   - **Rationale**: Rapid prototyping, Python-native, minimal frontend code
   - **Trade-off**: Limited customization and scalability vs. full web frameworks, but perfect for demos

### 9. **Document-Specific Filtering**
   - **Decision**: Allow queries to target specific documents via `document_name` parameter
   - **Rationale**: Enables precise comparisons (e.g., "Accenture revenue" vs. "Siemens revenue")
   - **Trade-off**: Requires agent to know document names, but dramatically improves multi-document analysis

### 10. **Error Handling with Recursion Limits**
   - **Decision**: Set recursion limit to 50 and provide user-friendly error messages
   - **Rationale**: Prevents infinite loops while allowing complex multi-tool workflows
   - **Trade-off**: May truncate very complex queries, but protects against runaway costs
