# Configuration for Company Knowledge Base Assistant

# Document directory - update this to point to your company documentation
DOCUMENTS_DIR = "./docs"

# Chunking configuration
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100

# Embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# FAISS index paths (relative to src directory)
FAISS_INDEX_PATH = "index.faiss"
CHUNKS_PATH = "chunks.pkl"

# Ollama configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen3:1.7b"

# RAG retrieval configuration
TOP_K = 5

# --- Hybrid search (Vector + FTS/BM25) configuration ---

# Number of candidates each retriever (vector, BM25) fetches before fusion.
# Must be >= TOP_K: fusing two short lists barely improves recall.
CANDIDATE_K = 20

# Reciprocal Rank Fusion constant: RRF(d) = sum_m 1 / (RRF_K + rank_m(d))
RRF_K = 60

# Relative weight of each retriever inside RRF (1.0 = equal contribution)
VECTOR_WEIGHT = 1.0
FTS_WEIGHT = 1.0

# --- Query expansion (LLM keyword generation) ---

# Set to False to skip the extra LLM round-trip and search with the raw query only
ENABLE_QUERY_EXPANSION = True

# Deterministic decoding keeps a 0.6B model from inventing keywords
KEYWORD_TEMPERATURE = 0.0

# Max keywords/phrases kept from the LLM output
MAX_KEYWORDS = 8

# Seconds to wait for the keyword-generation call before falling back to the raw query
KEYWORD_TIMEOUT = 30
