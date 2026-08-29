#!/usr/bin/env python3
"""Tests for the hybrid search building blocks.

Run with:  python tests/test_hybrid.py   (or: pytest tests/test_hybrid.py)

Only the light modules are imported at the top level, so the RRF and
query-expansion tests run without faiss / sentence-transformers installed.
"""

import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.fusion import reciprocal_rank_fusion
from rag.fts import fts_search, tokenize, reset as reset_fts
from rag.query_expansion import build_fts_query, generate_keywords, parse_keywords


def test_rrf_matches_hand_computed_scores():
    vector = ["a", "b", "c"]
    fts = ["c", "a", "d"]
    fused = dict(reciprocal_rank_fusion([vector, fts], k=60))

    assert abs(fused["a"] - (1 / 61 + 1 / 62)) < 1e-12
    assert abs(fused["c"] - (1 / 63 + 1 / 61)) < 1e-12
    assert abs(fused["b"] - 1 / 62) < 1e-12
    assert abs(fused["d"] - 1 / 63) < 1e-12


def test_rrf_rewards_documents_found_by_both_retrievers():
    # "c" is only 3rd for the vector search but 1st for FTS, so it must
    # outrank "b", which a single retriever ranked 2nd.
    order = [doc for doc, _ in reciprocal_rank_fusion([["a", "b", "c"], ["c", "a", "d"]])]
    assert order[:2] == ["a", "c"]
    assert order.index("c") < order.index("b")


def test_rrf_fuses_duplicates_into_one_entry():
    fused = reciprocal_rank_fusion([[1, 2], [2, 1]])
    assert len(fused) == 2
    assert {doc for doc, _ in fused} == {1, 2}


def test_rrf_handles_empty_and_weighted_lists():
    assert reciprocal_rank_fusion([[], []]) == []

    only_vector = dict(reciprocal_rank_fusion([["a"], []], weights=[2.0, 1.0]))
    assert abs(only_vector["a"] - 2 / 61) < 1e-12

    # A list that contributes nothing must not change the other list's order.
    assert [d for d, _ in reciprocal_rank_fusion([["a", "b"], []])] == ["a", "b"]


def test_rrf_respects_top_k():
    assert len(reciprocal_rank_fusion([[1, 2, 3, 4], [4, 3]], top_k=2)) == 2


def test_tokenizer_keeps_compound_tokens_and_their_parts():
    tokens = tokenize("Run vpn-reset --force, see onboarding.md")
    assert "vpn-reset" in tokens and "vpn" in tokens and "reset" in tokens
    assert "onboarding.md" in tokens and "onboarding" in tokens


def test_fts_finds_exact_rare_terms():
    chunks = [
        {"text": "Incident INC-4471 was caused by an expired certificate.", "source": "docs/incidents.md"},
        {"text": "Incident INC-4472 was caused by a full disk.", "source": "docs/incidents.md"},
        {"text": "The office closes on public holidays.", "source": "docs/office.md"},
    ]
    reset_fts()
    assert fts_search("INC-4472", chunks, top_k=3)[0] == 1

    # The production path passes the whole question, stopwords included.
    reset_fts()
    assert fts_search("What was the root cause of INC-4472?", chunks, top_k=3)[0] == 1

    # No overlapping term at all -> no results rather than noise.
    reset_fts()
    assert fts_search("kubernetes autoscaling", chunks, top_k=3) == []


def test_fts_ignores_corpus_wide_stopwords():
    """A question that only shares filler words must not match everything."""
    chunks = [
        {"text": "The office is open on the weekdays of every week.", "source": "a.md"},
        {"text": "The laptops of new hires are issued on the first day.", "source": "b.md"},
        {"text": "The rooms of the upper floor are open to every team.", "source": "c.md"},
        {"text": "Incident INC-4471 was caused by an expired certificate.", "source": "d.md"},
    ]
    reset_fts()
    assert fts_search("What was the cause of INC-4471?", chunks, top_k=4) == [3]


def test_fts_index_is_rebuilt_after_reset():
    first = [{"text": "alpha term", "source": "a.md"}]
    second = [{"text": "beta term", "source": "b.md"}, {"text": "alpha term", "source": "c.md"}]
    reset_fts()
    assert fts_search("alpha", first, top_k=1) == [0]
    reset_fts()
    assert fts_search("alpha", second, top_k=1) == [1]


def test_keyword_parsing_handles_the_requested_format():
    assert parse_keywords("VPN token, reset VPN, vpn-reset") == ["VPN token", "reset VPN", "vpn-reset"]


def test_keyword_parsing_strips_qwen3_think_blocks():
    assert parse_keywords('<think>Let me think.</think>\n["VPN token", "reset"]') == ["VPN token", "reset"]
    # An unterminated think block leaves nothing usable.
    assert parse_keywords("<think>still reasoning") == []


def test_keyword_parsing_survives_malformed_output():
    assert parse_keywords("") == []
    assert parse_keywords(None) == []
    assert parse_keywords("I am not sure what you mean.") == []
    assert parse_keywords("Keywords: 1. vacation policy 2. paid leave") == ["vacation policy", "paid leave"]
    assert parse_keywords('{"keywords": ["onboarding.md", "new hire"]}') == ["onboarding.md", "new hire"]


def test_keyword_parsing_drops_echo_of_the_question():
    assert parse_keywords("What is the SOC?, SOC", "What is the SOC?") == ["SOC"]


def test_generate_keywords_falls_back_when_the_llm_fails():
    def broken_llm(prompt, options=None):
        raise ConnectionError("ollama is down")

    assert generate_keywords("What is the SOC?", llm=broken_llm) == []
    # ... and the FTS query then degrades to the original question.
    assert build_fts_query("What is the SOC?", []) == "What is the SOC?"


def test_generate_keywords_uses_deterministic_decoding():
    captured = {}

    def recording_llm(prompt, options=None):
        captured["options"] = options
        captured["prompt"] = prompt
        return "SOC, security operations center"

    keywords = generate_keywords("What is the SOC?", llm=recording_llm)
    assert keywords == ["SOC", "security operations center"]
    assert captured["options"]["temperature"] == 0.0
    assert "What is the SOC?" in captured["prompt"]


def test_build_fts_query_appends_keywords_to_the_question():
    combined = build_fts_query("Where is the key?", ["hardware key", "IT helpdesk"])
    assert combined.startswith("Where is the key?")
    assert "hardware key" in combined and "IT helpdesk" in combined


def test_branches_run_on_separate_threads():
    """The vector and FTS branches must overlap, not run one after the other."""
    try:
        import faiss  # noqa: F401
        import rag.query as rq
    except ImportError:
        print("  (skipped test_branches_run_on_separate_threads: faiss not installed)")
        return

    barrier = threading.Barrier(2, timeout=5)
    threads = {}

    def slow_vector(query, top_k):
        threads["vector"] = threading.current_thread().name
        barrier.wait()  # deadlocks unless the FTS branch runs concurrently
        return [0]

    def slow_fts(query, chunks, top_k):
        threads["fts"] = threading.current_thread().name
        barrier.wait()
        return [1]

    original_vector, original_fts = rq.vector_search, rq.fts_search
    original_index, original_chunks = rq.index, rq.chunks
    rq.vector_search, rq.fts_search = slow_vector, slow_fts
    rq.index = object()  # only needs to be non-None for the readiness check
    rq.chunks = [{"text": "a", "source": "a.md"}, {"text": "b", "source": "b.md"}]
    try:
        contexts, stats = rq.hybrid_retrieve("q", keywords=["k"], top_k=2)
    finally:
        rq.vector_search, rq.fts_search = original_vector, original_fts
        rq.index, rq.chunks = original_index, original_chunks

    assert threads["vector"] != threads["fts"], "branches ran on the same thread"
    assert stats["vector_hits"] == 1 and stats["fts_hits"] == 1
    assert {c["source"] for c in contexts} == {"a.md", "b.md"}


def test_a_failing_branch_does_not_break_retrieval():
    try:
        import faiss  # noqa: F401
        import rag.query as rq
    except ImportError:
        print("  (skipped test_a_failing_branch_does_not_break_retrieval: faiss not installed)")
        return

    def broken_vector(query, top_k):
        raise RuntimeError("faiss exploded")

    original_vector, original_index, original_chunks = rq.vector_search, rq.index, rq.chunks
    rq.vector_search = broken_vector
    rq.index = object()
    rq.chunks = [{"text": "alpha term", "source": "a.md"}]
    try:
        contexts, stats = rq.hybrid_retrieve("alpha", keywords=[], top_k=2)
    finally:
        rq.vector_search, rq.index, rq.chunks = original_vector, original_index, original_chunks

    assert stats["vector_hits"] == 0
    assert [c["source"] for c in contexts] == ["a.md"]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"✅ {test.__name__}")
        except Exception as e:
            failures += 1
            print(f"❌ {test.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
