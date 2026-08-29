#!/usr/bin/env python3
"""Before/after benchmark: vector-only retrieval vs hybrid (vector + BM25 + RRF).

Self-contained: it builds a small FAISS index over an inline corpus, so it does
not touch your real docs/ index. Ollama is used for keyword generation when it
is reachable; otherwise the run continues without query expansion and says so.

Usage:  python benchmark.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import faiss
import numpy as np

import rag.query as rq
from rag import fts
from rag.query_expansion import generate_keywords
from config import TOP_K


# A corpus with the failure mode vector search is known for: rare terms, exact
# abbreviations, command names and file names.
CORPUS = [
    # 0
    ("docs/pto-policy.md", "Full-time employees accrue 20 working days of paid time off per calendar year. Unused days roll over until March 31 of the following year."),
    # 1
    ("docs/pto-policy.md", "Time off requests must be submitted at least two weeks in advance and approved by your direct manager."),
    # 2
    ("docs/onboarding.md", "New hires complete the orientation session during their first week and are paired with an onboarding buddy."),
    # 3
    ("docs/onboarding.md", "Laptops are issued by the IT team on day one. Accounts are provisioned automatically once the contract is signed."),
    # 4
    ("docs/it-vpn.md", "To reset an expired certificate run `vpn-reset --force` from the corporate terminal and re-authenticate with your hardware key."),
    # 5
    ("docs/it-vpn.md", "Remote access requires the company VPN. Connection issues are usually solved by reconnecting to the nearest gateway."),
    # 6
    ("docs/security.md", "The Security Operations Center, abbreviated SOC, monitors alerts around the clock and triages incidents by severity."),
    # 7
    ("docs/security.md", "Report any suspected phishing message to the security team immediately. Do not click links in unexpected emails."),
    # 8
    ("docs/finance.md", "Expense reports are filed monthly. Receipts above 50 EUR must be attached as a scanned copy."),
    # 9
    ("docs/finance.md", "The QBR takes place in the last week of every quarter and is chaired by the finance lead."),
    # 10
    ("docs/office.md", "The office is open from 8:00 to 20:00 on weekdays and closed on public holidays."),
    # 11
    ("docs/office.md", "Meeting rooms are booked through the internal calendar. Room Aurora seats twelve people."),

    # --- distractors: same vocabulary, wrong answer ---
    # 12
    ("docs/handbook.md", "This handbook describes how the company works: working hours, days off, equipment, security and internal reviews."),
    # 13
    ("docs/handbook.md", "Every team runs a regular review meeting to discuss progress, blockers and results of the quarter."),
    # 14
    ("docs/handbook.md", "Employees can take unpaid leave in exceptional circumstances after discussing it with HR."),
    # 15
    ("docs/it-general.md", "If you cannot connect to an internal service, restart the client, check your credentials and try again later."),
    # 16
    ("docs/it-general.md", "Passwords must be reset every 90 days. Use the self-service portal to set a new password."),
    # 17
    ("docs/it-general.md", "Hardware keys are distributed by the IT helpdesk and must be returned when you leave the company."),
    # 18
    ("docs/glossary.md", "This glossary explains internal terms and abbreviations used across company documentation."),
    # 19
    ("docs/glossary.md", "Documentation files are named after their topic, for example the file that describes the first days of a new employee."),
    # 20
    ("docs/facilities.md", "Rooms can be reserved for team events. Larger rooms are available on the upper floor."),
    # 21
    ("docs/facilities.md", "Please leave meeting rooms tidy and cancel the reservation if the meeting does not happen."),
    # 22
    ("docs/travel.md", "Business trips must be approved in advance. Submit the request together with the estimated costs."),
    # 23
    ("docs/travel.md", "Travel costs are reimbursed after the trip. Keep every document proving the amount you paid."),

    # --- near-identical chunks that differ only by an identifier ---
    # 24
    ("docs/incidents.md", "Incident INC-4471: the internal wiki was unavailable for two hours. Root cause was an expired TLS certificate."),
    # 25
    ("docs/incidents.md", "Incident INC-4472: the build pipeline failed for all teams. Root cause was a full disk on the runner host."),
    # 26
    ("docs/incidents.md", "Incident INC-4473: the payroll export was delayed by one day. Root cause was a misconfigured cron schedule."),
    # 27
    ("docs/incidents.md", "Incident INC-4474: email delivery was degraded in the morning. Root cause was an upstream provider outage."),
    # 28
    ("docs/policies.md", "Policy HR-118-B: employees working from another country for more than 30 days need written approval from HR and finance."),
    # 29
    ("docs/policies.md", "Policy HR-119-A: employees may swap a public holiday for another day off within the same quarter."),
]


# question -> index of the chunk that actually answers it
QUERIES = [
    ("How many vacation days do I get?", 0),
    ("What does `vpn-reset --force` do?", 4),
    ("What is the SOC responsible for?", 6),
    ("When is the QBR held?", 9),
    ("What is written in onboarding.md?", 2),
    ("How do I book Room Aurora?", 11),
    ("What is the deadline for submitting a time off request?", 1),
    ("Rules for expense receipts", 8),
    ("Who gets a hardware key?", 17),
    ("How often must I change my password?", 16),
    ("What was the root cause of INC-4473?", 26),
    ("Tell me about INC-4471", 24),
    ("What does policy HR-118-B require?", 28),
    ("What is HR-119-A about?", 29),
]



# Filler chunks added on top of CORPUS. BM25 over 30 chunks costs 0.3 ms, which
# makes the parallelism measurement meaningless; a realistic index is far
# larger, so the benchmark pads the corpus with generated noise. The labelled
# chunks keep their positions because the padding is appended after them.
PADDING_CHUNKS = 2000


def build_test_index():
    """Point rag.query's globals at the inline corpus (plus padding)."""
    corpus = list(CORPUS)
    topics = ["deployment", "reporting", "training", "procurement", "support"]
    for i in range(PADDING_CHUNKS):
        topic = topics[i % len(topics)]
        corpus.append((
            f"docs/archive-{topic}.md",
            f"Archived note {i} about {topic}: the team reviewed the {topic} process "
            f"for period {2000 + i} and recorded the outcome in ticket ARCH-{9000 + i}.",
        ))

    chunks = [
        {"text": text, "source": source, "chunk_id": i}
        for i, (source, text) in enumerate(corpus)
    ]
    embeddings = np.array(rq.model.encode([c["text"] for c in chunks]))
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    rq.index = index
    rq.chunks = chunks
    fts.reset()
    return chunks


def rank_of_expected(contexts, expected_text):
    for rank, ctx in enumerate(contexts, start=1):
        if ctx["text"] == expected_text:
            return rank
    return None


def score(results):
    """Recall@K and Mean Reciprocal Rank over the query set."""
    hits = sum(1 for r in results if r is not None)
    mrr = sum(1.0 / r for r in results if r is not None) / len(results)
    return hits / len(results), mrr


def main():
    chunks = build_test_index()
    print(
        f"Indexed {len(chunks)} chunks from "
        f"{len(set(c['source'] for c in chunks))} documents "
        f"({len(CORPUS)} labelled + {PADDING_CHUNKS} padding)\n"
    )

    # Warm up both retrievers so the first timed query does not pay for lazy
    # init of the embedding model and the BM25 index.
    for question, _ in QUERIES:
        rq.vector_search(question, TOP_K)
        rq.hybrid_retrieve(question, keywords=["warmup"])

    keywords_available = generate_keywords("How many vacation days do I get?") != []
    if keywords_available:
        print("Query expansion: ON (Ollama reachable)\n")
    else:
        print("Query expansion: OFF (no keywords returned - fallback to raw query)\n")

    baseline_ranks, hybrid_ranks = [], []
    baseline_ms, hybrid_ms = 0.0, 0.0
    branch_ms = {"vector": 0.0, "fts": 0.0, "parallel": 0.0}

    header = f"{'Question':<52} {'vector-only':>12} {'hybrid':>8}"
    print(header)
    print("-" * len(header))

    for question, expected_idx in QUERIES:
        expected = chunks[expected_idx]["text"]
        started = time.perf_counter()
        vector_ids = rq.vector_search(question, TOP_K)
        baseline_ms += (time.perf_counter() - started) * 1000
        baseline = [chunks[i] for i in vector_ids]

        keywords = generate_keywords(question)
        started = time.perf_counter()
        hybrid, stats = rq.hybrid_retrieve(question, keywords=keywords)
        hybrid_ms += (time.perf_counter() - started) * 1000

        branch_ms["vector"] += stats["vector_ms"]
        branch_ms["fts"] += stats["fts_ms"]
        branch_ms["parallel"] += stats["total_ms"]

        b_rank = rank_of_expected(baseline, expected)
        h_rank = rank_of_expected(hybrid, expected)
        baseline_ranks.append(b_rank)
        hybrid_ranks.append(h_rank)

        fmt = lambda r: f"#{r}" if r else "miss"
        print(f"{question[:52]:<52} {fmt(b_rank):>12} {fmt(h_rank):>8}")

    b_recall, b_mrr = score(baseline_ranks)
    h_recall, h_mrr = score(hybrid_ranks)

    print()
    print(f"{'metric':<20} {'vector-only':>12} {'hybrid':>10}")
    print("-" * 44)
    print(f"{'Recall@' + str(TOP_K):<20} {b_recall:>12.2f} {h_recall:>10.2f}")
    print(f"{'MRR':<20} {b_mrr:>12.2f} {h_mrr:>10.2f}")
    print(f"{'avg retrieval ms':<20} {baseline_ms / len(QUERIES):>12.1f} {hybrid_ms / len(QUERIES):>10.1f}")
    n = len(QUERIES)
    print()
    print("Parallelism (average per query, inside hybrid_retrieve)")
    print("-" * 44)
    print(f"{'vector branch':<20} {branch_ms['vector'] / n:>12.1f} ms")
    print(f"{'fts branch':<20} {branch_ms['fts'] / n:>12.1f} ms")
    print(f"{'if run one by one':<20} {(branch_ms['vector'] + branch_ms['fts']) / n:>12.1f} ms")
    print(f"{'measured wall clock':<20} {branch_ms['parallel'] / n:>12.1f} ms")
    print("\n(retrieval ms excludes keyword generation, which is a separate LLM call)")


if __name__ == "__main__":
    main()
