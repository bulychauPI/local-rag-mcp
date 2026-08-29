"""Full-text search (BM25) over the same chunk list used by the vector index.

The BM25 index is built lazily from the in-memory chunks and cached, so the
first query after an index (re)build pays for it and later queries do not.
"""

import re
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from rank_bm25 import BM25Okapi
except ImportError:  # pragma: no cover - dependency is declared in requirements.txt
    BM25Okapi = None


# Words plus compound tokens such as "vacation-policy.md" or "on_call".
_TOKEN_RE = re.compile(r"[^\W_]+(?:[._\-/][^\W_]+)*", re.UNICODE)

_lock = threading.Lock()
_bm25 = None
_indexed_texts = None
_token_sets = None
_doc_freq = None


def tokenize(text: str):
    """Lowercase tokenizer that keeps compound tokens and also their parts.

    Emitting both "vacation-policy.md" and "vacation", "policy", "md" lets BM25
    match an exact file name as well as a loose mention of it.
    """
    tokens = []
    for match in _TOKEN_RE.findall(text.lower()):
        tokens.append(match)
        parts = re.split(r"[._\-/]", match)
        if len(parts) > 1:
            tokens.extend(p for p in parts if p)
    return tokens


def reset():
    """Drop the cached BM25 index (call after the chunk list is rebuilt)."""
    global _bm25, _indexed_texts, _token_sets, _doc_freq
    with _lock:
        _bm25 = None
        _indexed_texts = None
        _token_sets = None
        _doc_freq = None


def _get_bm25(chunks):
    """Return a BM25 index for `chunks`, rebuilding it only when they changed."""
    global _bm25, _indexed_texts, _token_sets, _doc_freq
    if BM25Okapi is None:
        return None, None, None

    texts = [c["text"] for c in chunks]
    with _lock:
        if _bm25 is not None and _indexed_texts == texts:
            return _bm25, _token_sets, _doc_freq
        if not texts:
            return None, None, None
        # The source path carries signal too ("what does onboarding.md say?").
        corpus = [
            tokenize(c["text"]) + tokenize(str(c.get("source", "")))
            for c in chunks
        ]
        _bm25 = BM25Okapi(corpus)
        _token_sets = [set(doc) for doc in corpus]
        _doc_freq = {}
        for token_set in _token_sets:
            for token in token_set:
                _doc_freq[token] = _doc_freq.get(token, 0) + 1
        _indexed_texts = texts
        return _bm25, _token_sets, _doc_freq


def fts_search(query: str, chunks, top_k: int):
    """Rank chunk indices by BM25 relevance to `query`.

    Returns a list of indices into `chunks`, best first. Chunks sharing no
    *discriminative* term with the query are dropped rather than ranked: BM25
    still assigns them a score, and feeding those into fusion would only add
    noise. Two details matter here:

    - The overlap is checked on tokens, not on `score > 0`: BM25 idf goes
      negative for terms present in (almost) every document.
    - Terms occurring in more than half of the corpus are ignored for the
      overlap check. The production query is the whole question, so without
      this "of" or "the" alone would match every chunk and the filter would
      never drop anything.
    """
    bm25, token_sets, doc_freq = _get_bm25(chunks)
    if bm25 is None:
        return []

    tokens = tokenize(query)
    if not tokens:
        return []

    query_tokens = set(tokens)
    cutoff = 0.5 * len(token_sets)
    discriminative = {t for t in query_tokens if doc_freq.get(t, 0) <= cutoff}
    if not discriminative:  # every query term is a corpus-wide stopword
        discriminative = query_tokens

    scores = bm25.get_scores(tokens)
    matched = [i for i in range(len(scores)) if discriminative & token_sets[i]]
    matched.sort(key=lambda i: -scores[i])
    return matched[:top_k]


if __name__ == "__main__":
    demo = [
        {"text": "Employees accrue 20 days of paid leave per year.", "source": "docs/pto.md"},
        {"text": "To reset your VPN token run `vpn-reset --force`.", "source": "docs/it.md"},
        {"text": "The office is closed on public holidays.", "source": "docs/office.md"},
    ]
    for i in fts_search("vpn-reset", demo, top_k=3):
        print(i, demo[i]["source"])
