import faiss
import pickle
import requests
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    FAISS_INDEX_PATH,
    CHUNKS_PATH,
    EMBEDDING_MODEL,
    OLLAMA_URL,
    OLLAMA_MODEL,
    TOP_K,
    CANDIDATE_K,
    RRF_K,
    VECTOR_WEIGHT,
    FTS_WEIGHT,
)
from rag.fts import fts_search, reset as reset_fts
from rag.fusion import reciprocal_rank_fusion
from rag.query_expansion import build_fts_query

model = SentenceTransformer(EMBEDDING_MODEL)

# Global variables for index and chunks
index = None
chunks = []


def _ensure_index_exists():
    """Ensure FAISS index exists, build it if it doesn't."""
    global index, chunks
    
    # Resolve paths relative to src directory
    src_dir = Path(__file__).parent.parent
    index_path = src_dir / FAISS_INDEX_PATH
    chunks_path = src_dir / CHUNKS_PATH
    
    # Check if index exists
    if index_path.exists() and chunks_path.exists():
        try:
            index = faiss.read_index(str(index_path))
            with open(chunks_path, "rb") as f:
                chunks = pickle.load(f)
            reset_fts()
            return True
        except Exception as e:
            print(f"⚠️  Warning: Error loading existing index: {e}")
            print("Rebuilding index...")
    
    # Index doesn't exist or failed to load, build it
    print("📦 Index not found. Building index from documents...")
    try:
        from rag.build_index import build_index
        build_index()
        
        # Load the newly created index
        if index_path.exists() and chunks_path.exists():
            index = faiss.read_index(str(index_path))
            with open(chunks_path, "rb") as f:
                chunks = pickle.load(f)
            reset_fts()
            print("✅ Index built and loaded successfully")
            return True
        else:
            print("❌ Failed to build index. No documents found or error occurred.")
            from config import DOCUMENTS_DIR
            docs_path = src_dir / DOCUMENTS_DIR
            print(f"   Check that documents exist in: {docs_path}")
            return False
    except Exception as e:
        print(f"❌ Error building index: {e}")
        import traceback
        traceback.print_exc()
        return False


# Initialize index on module load
_ensure_index_exists()


# Two workers: the vector branch and the FTS branch run concurrently. Both
# release the GIL for most of their work (FAISS C++ / numpy), so threads are
# enough and every caller stays synchronous. Created once, not per query.
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="hybrid-search")


def _index_ready():
    """Load the index on demand; return False if there is nothing to search."""
    if index is None or len(chunks) == 0:
        if not _ensure_index_exists():
            return False
    return index is not None and len(chunks) > 0


def vector_search(query: str, top_k: int = CANDIDATE_K):
    """Rank chunk indices by embedding similarity. Returns indices, best first."""
    q_emb = model.encode([query])
    faiss.normalize_L2(q_emb)

    scores, ids = index.search(q_emb, min(top_k, index.ntotal))
    # FAISS pads with -1 when fewer than top_k vectors exist; chunks[-1] would
    # silently return the last chunk.
    return [int(i) for i in ids[0] if 0 <= i < len(chunks)]


def hybrid_retrieve(query: str, keywords=None, top_k: int = TOP_K,
                    candidate_k: int = CANDIDATE_K):
    """Retrieve chunks with vector search and BM25 in parallel, fused by RRF.

    Args:
        query: the original user question (used verbatim for vector search).
        keywords: optional keywords from query expansion, appended to the FTS
            query only - embeddings work better on the natural question.
        top_k: how many chunks to return after fusion.
        candidate_k: how many candidates each retriever contributes to fusion.

    Returns:
        (contexts, stats) where contexts is a list of chunk dicts and stats
        carries per-branch timings and hit counts for verbose output.
    """
    stats = {
        "vector_hits": 0, "fts_hits": 0, "fused_hits": 0,
        "vector_ms": 0.0, "fts_ms": 0.0, "total_ms": 0.0,
        "keywords": list(keywords or []),
    }

    if not _index_ready():
        return [], stats

    snapshot = chunks  # bind once: both branches must see the same list
    fts_query = build_fts_query(query, keywords)

    def run_vector():
        started = time.perf_counter()
        try:
            return vector_search(query, candidate_k), (time.perf_counter() - started) * 1000
        except Exception as e:
            print(f"⚠️  Vector search failed: {e}")
            return [], (time.perf_counter() - started) * 1000

    def run_fts():
        started = time.perf_counter()
        try:
            return fts_search(fts_query, snapshot, candidate_k), (time.perf_counter() - started) * 1000
        except Exception as e:
            print(f"⚠️  Full-text search failed: {e}")
            return [], (time.perf_counter() - started) * 1000

    started = time.perf_counter()
    vector_future = _executor.submit(run_vector)
    fts_future = _executor.submit(run_fts)
    vector_ids, stats["vector_ms"] = vector_future.result()
    fts_ids, stats["fts_ms"] = fts_future.result()
    stats["total_ms"] = (time.perf_counter() - started) * 1000

    stats["vector_hits"] = len(vector_ids)
    stats["fts_hits"] = len(fts_ids)

    # Both branches speak the same id space (positions in `chunks`), so RRF can
    # recognise a chunk found by both and reward it.
    fused = reciprocal_rank_fusion(
        [vector_ids, fts_ids],
        k=RRF_K,
        weights=[VECTOR_WEIGHT, FTS_WEIGHT],
        top_k=top_k,
    )
    stats["fused_hits"] = len(fused)
    return [snapshot[i] for i, _ in fused], stats


def retrieve(query: str, keywords=None, top_k: int = TOP_K):
    """Retrieve relevant chunks for a query (hybrid vector + BM25)."""
    contexts, _ = hybrid_retrieve(query, keywords=keywords, top_k=top_k)
    return contexts


def build_prompt(query, contexts):
    """Build prompt with retrieved context."""
    if not contexts:
        return f"""
<role>You are a helpful assistant that answers questions about company information.</role>
<instructions>Answer the question based on your general knowledge. If you don't know, say so.</instructions>

<query>
{query}
</query>

<assistant>
"""

    context_text = "\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}"
        for c in contexts
    )

    return f"""
<role>You are a helpful assistant that answers questions about company information.</role>
<instructions>Answer the question ONLY based on the context provided below. If the answer is not in the context, say "I don't have that information in the knowledge base."</instructions>

<context>
{context_text}
</context>

<query>
{query}
</query>

<assistant>
"""


def ask_llm(prompt, options=None):
    """Query Ollama LLM.

    `options` is passed through to Ollama (temperature, num_predict, ...). The
    non-Ollama key "timeout" is consumed here as the HTTP timeout.
    """
    options = dict(options or {})
    timeout = options.pop("timeout", None)

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        # qwen3 emits <think> blocks by default; they only cost tokens here.
        "think": False,
    }
    if options:
        payload["options"] = options

    response = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()["response"]


def ask(query: str):
    """Answer a question using the full hybrid RAG pipeline."""
    from rag.query_expansion import generate_keywords

    keywords = generate_keywords(query)
    contexts = retrieve(query, keywords=keywords)
    prompt = build_prompt(query, contexts)
    return ask_llm(prompt), contexts


if __name__ == "__main__":
    while True:
        q = input("\n❓ Question: ")
        if q.lower() in {"exit", "quit"}:
            break
        print("\n🤖 Answer:\n")
        answer, sources = ask(q)
        print(answer)
        if sources:
            print("\n📚 Sources:")
            for src in sources:
                print(f"  - {src['source']}")
