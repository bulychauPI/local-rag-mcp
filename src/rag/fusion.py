"""Reciprocal Rank Fusion for combining several ranked result lists.

Kept free of faiss / sentence-transformers imports so it can be unit-tested
(and reasoned about) without the heavy retrieval stack.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RRF_K


def reciprocal_rank_fusion(ranked_lists, k=RRF_K, weights=None, top_k=None):
    """Fuse ranked lists of document ids with RRF.

    RRF_Score(d) = sum over lists m of  weight_m / (k + rank_m(d))
    where rank_m(d) is the 1-based position of d in list m.

    Documents missing from a list simply contribute nothing for that list, so
    the scores of the individual retrievers never have to be comparable.

    Args:
        ranked_lists: sequence of id sequences, each ordered best-first.
        k: RRF damping constant (60 is the value from the original paper).
        weights: optional per-list weights, same length as ranked_lists.
        top_k: keep only the best top_k ids (None keeps all).

    Returns:
        List of (doc_id, score) ordered by descending score.
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError("weights must have the same length as ranked_lists")

    scores = {}
    # Best rank seen per document, used only to break score ties deterministically.
    best_rank = {}

    for ranked, weight in zip(ranked_lists, weights):
        seen = set()
        for rank, doc_id in enumerate(ranked, start=1):
            if doc_id in seen:  # a retriever must not score the same doc twice
                continue
            seen.add(doc_id)
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (k + rank)
            if doc_id not in best_rank or rank < best_rank[doc_id]:
                best_rank[doc_id] = rank

    fused = sorted(scores.items(), key=lambda item: (-item[1], best_rank[item[0]]))
    if top_k is not None:
        fused = fused[:top_k]
    return fused


if __name__ == "__main__":
    vector = ["a", "b", "c"]
    fts = ["c", "a", "d"]
    for doc_id, score in reciprocal_rank_fusion([vector, fts]):
        print(f"{doc_id}: {score:.6f}")
