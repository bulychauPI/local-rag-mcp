# Local RAG/MCP Knowledge Base Assistant

# 📋 The Problem

- **Growing Documentation**: Knowledge scattered across files
- **Information Retrieval**: Hard to find answers without keywords
- **Privacy Concerns**: Cloud solutions may not comply with policies

```
Users → Search → Answer = 😫
```

# ✨ The Solution

A **local, intelligent Q&A system** using:

- **RAG**: Semantic search over documentation
- **MCP**: Dynamic document access
- **Local LLM**: Privacy-preserving answers (Ollama)

# ✨ Key Benefits

- ✅ Privacy-first (runs locally)
- ✅ No API costs
- ✅ Fast semantic search
- ✅ Intelligent document access
- ✅ Complete data control

# 🏗️ Architecture - Top Level

```
┌──────────────────────┐
│   User Interface     │ (CLI)
└──────────┬───────────┘
           │
     ┌─────┴─────┐
     ▼           ▼
  [RAG]       [MCP]
   Query      Tools
     │           │
     └─────┬─────┘
           ▼
    [Ollama LLM]
```

# 🏗️ Architecture - Storage

```
┌────────────────┐
│  FAISS Index   │ Vector Database
│  + MCP Tools   │
└────────┬───────┘
         │
    ┌────▼─────┐
    │   docs/  │
    │directory │
    └──────────┘
```

# 🔍 RAG Pipeline

1. Document Loading → Read .md, .txt, .pdf, .docx
2. Chunking → Split into 700-char chunks
3. Embedding → Use SentenceTransformers
4. Indexing → Build FAISS vector index
5. Query Expansion → LLM extracts keywords from the question
6. Hybrid Retrieval → Vector search + BM25 in parallel, fused with RRF
7. Prompt Building → Create context-aware prompt
8. LLM Generation → Get answer from model

# 🔀 Hybrid Search (Vector + Full-Text)

Vector similarity alone loses rare terms, abbreviations, identifiers and file
names — the embedding of "INC-4473" is nearly identical to that of "INC-4471".
Retrieval therefore runs two retrievers over the *same* chunk list:

```
          User question
                │
    ┌───────────┴──────────────┐
    │  LLM keyword generation  │  temperature 0, comma-separated output,
    └───────────┬──────────────┘  falls back to the raw question on any failure
                │
        ┌───────┴────────┐        both branches run concurrently
        ▼                ▼        in a ThreadPoolExecutor
  Vector search      BM25 / FTS
  (FAISS, question)  (question + keywords)
        └───────┬────────┘
                ▼
   Reciprocal Rank Fusion (k = 60)
                ▼
          Top-K chunks → LLM / MCP
```

- **Same id space.** Both retrievers return positions in the global `chunks`
  list, so RRF recognises a chunk found by both and rewards it.
- **Candidate depth.** Each retriever contributes `CANDIDATE_K = 20`
  candidates; fusion then trims to `TOP_K = 5`.
- **Fallback everywhere.** Bad keyword formats, a `<think>` block, a dead
  Ollama or a failing branch all degrade to a working search instead of an
  exception.

`RRF_Score(d) = Σ_m weight_m / (k + rank_m(d))`, with `k = 60`.

# 🔍 Why FAISS?

- Fast vector similarity search
- Lightweight and memory-efficient
- No external dependencies
- Perfect for local deployments
- Millions of vectors supported

# 🔧 MCP - Model Context Protocol

MCP provides **standardized interface** for LLM tool access:

```python
read_document(file_path)
list_documents()
search_documents(query)
```

# 🔧 MCP Benefits

- Tool Use by LLM
- Real-time document access
- Standardized interface
- Easy to extend
- Local tool execution

# 💻 Tech Stack

```
Language:      Python 3.10+
Vector DB:     FAISS
Embeddings:    SentenceTransformers
LLM:           Ollama (local)
MCP:           FastMCP
```

# 📁 Project Structure

```
src/
├── config.py           Configuration
├── main.py             CLI entry point
├── assistant.py        Main orchestrator
├── benchmark.py        Vector-only vs hybrid comparison
├── rag/
│   ├── ingest.py      Load documents
│   ├── chunk.py       Split text
│   ├── embed.py       Generate embeddings
│   ├── build_index.py Build FAISS index
│   ├── query_expansion.py  LLM keyword generation + fallbacks
│   ├── fts.py         BM25 full-text search
│   ├── fusion.py      Reciprocal Rank Fusion
│   └── query.py       Hybrid retrieval & generation
├── mcp/
│   ├── server.py      MCP tool definitions
│   └── client.py      MCP client wrapper
├── tests/
│   └── test_hybrid.py Tests for expansion, FTS and RRF
└── docs/              Documentation
```

# 🚀 Index Building (Setup)

```
$ python main.py build-index

1. Load documents
  ↓
2. Split into chunks
  ↓
3. Generate embeddings
  ↓
4. Build FAISS index
  ↓
5. Save files
```

# 🚀 Query Processing (Runtime)

```
User Question
  ↓
LLM extracts keywords (temperature 0)
  ↓
Vector search ‖ BM25 search (parallel, 20 candidates each)
  ↓
Reciprocal Rank Fusion → Top 5 chunks
  ↓
LLM decides: Use MCP tools?
  ↓
Build prompt + context
  ↓
Call Ollama
  ↓
Return answer + sources
```

# ✨ Core Features

- **Hybrid Search**: Vector similarity + BM25, fused with RRF
- **Query Expansion**: LLM-generated keywords sharpen the full-text branch
- **Semantic Search**: Find by meaning, not keywords
- **Multi-format**: .md, .txt, .pdf, .docx files
- **Source Attribution**: Shows document sources
- **MCP Tools**: LLM can read full documents
- **No External APIs**: Runs locally only
- **Fast Retrieval**: Sub-second search

# ⚙️ Configuration Options

```python
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "qwen3:1.7b"
TOP_K = 5

# Hybrid search
CANDIDATE_K = 20            # candidates per retriever before fusion
RRF_K = 60                  # RRF damping constant
VECTOR_WEIGHT = 1.0         # weight of the vector branch in RRF
FTS_WEIGHT = 1.0            # weight of the BM25 branch in RRF
ENABLE_QUERY_EXPANSION = True
KEYWORD_TEMPERATURE = 0.0   # deterministic keywords for small models
```

# 🎬 Live Demo - Starting

```bash
$ python main.py
```

Output:
```
🤖 Company Knowledge Base
Ask questions about documentation
Type 'exit' to stop
```

# 🎬 Demo - Query 1

```
❓ What are company values?

🤖 Innovation, integrity, collaboration

📚 Sources:
  • Loan Rangers Team.md
  • Info Security.md
```

# 🎬 Demo - Query 2

```
❓ What documents do we have?

🤖 [Uses MCP list_documents]
  • Loan Rangers Team.md
  • Information Security.md
  • Services.md
```

# 🎬 Demo - Query 3

```
❓ Full security policy?

🤖 [Uses MCP read_document]
[Full document content...]
```

# 🔐 Security - Local vs Cloud

**Cloud**: Data → Internet → Server
- ⚠️ Network transmission
- ⚠️ External storage
- ⚠️ Subscription costs

**Local**: Data → Local System
- ✅ No transmission
- ✅ Local storage only
- ✅ No costs

# 🔐 Implementation Safeguards

- **MCP Sandbox**: Prevents path traversal
- **Local Storage**: Documents stay on device
- **No Telemetry**: No tracking
- **Offline Ready**: Works without internet

# ⚡ Performance Benchmarks

```
Index Building:   ~30s (one-time)
Query Embedding:  ~50ms
FAISS Search:     ~5ms
LLM Generation:   2-5s
Total Cycle:      2-6s
```

# ⚡ Tuning for Speed

```python
# Faster (smaller model):
OLLAMA_MODEL = "qwen3:0.6b"

# Faster retrieval:
TOP_K = 3
CHUNK_SIZE = 500
```

# 🚢 Deployment - Single Machine

```
1. Install Ollama & Python deps
2. Copy docs/ to server
3. Build index
4. Run with nohup

$ nohup python main.py > log &
```

# 🚢 Scaling - Option 1: FastAPI

```
[HTTP Clients]			[HTTP Clients + Webllm]
       ↓        						 ↓
   [FastAPI]     				 [FastAPI]
       ↓         					 ↓
[Ollama + FAISS]      			  [FAISS]
```

# 🚢 Scaling - Option 2: Distributed

```
[Clients] → [Load Balancer]
             ↓
      [Multiple Retrievers]
```

# 🚢 Storage Scaling

```
Docs     Index      Build
10 MB    ~2 MB      ~5s
100 MB   ~20 MB     ~30s
1 GB     ~200 MB    ~5min
```

# 🔮 Phase 2: Enhanced Features

- ☐ Web UI (Streamlit)
- ☐ API endpoints
- ☐ Multi-language support
- ☐ Document versioning
- ☐ Fine-tuned embeddings

# 🔮 Phase 3: Advanced

- ☐ Conversation memory
- ☐ Multi-hop reasoning
- ☐ Metadata filtering
- ☐ Feedback loop
- ☐ Analytics dashboard

# 🔮 Phase 4: Enterprise

- ☐ User authentication
- ☐ Audit logging
- ☐ Role-based access
- ☐ LLM fine-tuning
- ☐ Cost analysis

# 📊 Why This Works

| Aspect | Traditional | Our RAG |
|--------|---|---|
| **Understanding** | Keywords | Semantic |
| **Answers** | Documents | Direct |
| **Privacy** | Cloud | Local |
| **Cost** | Subscription | One-time |
| **Speed** | Slow | Sub-second |

# ✅ What You Have Now

- Local privacy-first knowledge base
- Fast semantic search (FAISS)
- Intelligent tool use (MCP)
- Maintainable Python code
- Foundation for enterprise features

# 🙋 Quick Reference

```bash
# Run the tests
python tests/test_hybrid.py

# Compare vector-only vs hybrid retrieval
python benchmark.py

# Build index
python main.py build-index

# Run interactively
python main.py

# Check config
cat config.py
```

# 📚 Resources

- **Code**: MobilaName/local-rag-mcp
- **FAISS**: facebook/faiss
- **Ollama**: ollama.ai
- **FastMCP**: github.com/jlowin/fastmcp
- **Transformers**: huggingface.co

**Thank You!**
