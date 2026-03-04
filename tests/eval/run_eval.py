"""Evaluation runner — runs RAG eval against a live or mocked retrieval pipeline.

Usage:
    python -m tests.eval.run_eval [--live] [--k 5]

    --live  Run against the live Retrieval Service (requires services running)
    --k     Top-K parameter for retrieval metrics (default: 5)
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from .dataset import EVAL_DATASET
from .metrics import EvalSample, RetrievalResult, evaluate


async def run_mock_eval(k: int = 5):
    """Run evaluation with mock retrieval results for testing the framework."""
    samples = []

    for sample in EVAL_DATASET:
        # Simulate retrieval: some hits, some misses
        mock_results = []
        for i, doc_id in enumerate(sample.relevant_doc_ids[:k]):
            mock_results.append(
                RetrievalResult(
                    chunk_id=f"chunk_{doc_id}_{i}",
                    text=sample.expected_answer,
                    score=0.95 - (i * 0.1),
                    metadata={"doc_id": doc_id},
                )
            )
        # Add some irrelevant results
        for i in range(len(mock_results), k):
            mock_results.append(
                RetrievalResult(
                    chunk_id=f"noise_{i}",
                    text="Unrelated content about something else entirely.",
                    score=0.3 - (i * 0.05),
                    metadata={"doc_id": f"noise_doc_{i}"},
                )
            )

        eval_sample = EvalSample(
            question=sample.question,
            expected_answer=sample.expected_answer,
            relevant_doc_ids=sample.relevant_doc_ids,
            retrieved_results=mock_results,
            generated_answer=sample.expected_answer,  # Perfect answer for mock
            tags=sample.tags,
        )
        samples.append(eval_sample)

    return evaluate(samples, k=k)


async def run_live_eval(k: int = 5):
    """Run evaluation against the live retrieval and agent services."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from services.gateway.src.grpc_clients.retrieval import RetrievalClient
    except ImportError:
        print("ERROR: Cannot import RetrievalClient. Ensure the project is in PYTHONPATH.")
        sys.exit(1)

    client = RetrievalClient(host="localhost:50051")
    await client.connect()

    samples = []
    for sample in EVAL_DATASET:
        result = await client.search(
            query=sample.question,
            collection="team_platform",
            top_k=k,
        )

        retrieved = [
            RetrievalResult(
                chunk_id=r["chunk_id"],
                text=r["text"],
                score=r["score"],
                metadata=r["metadata"],
            )
            for r in result.get("results", [])
        ]

        eval_sample = EvalSample(
            question=sample.question,
            expected_answer=sample.expected_answer,
            relevant_doc_ids=sample.relevant_doc_ids,
            retrieved_results=retrieved,
            generated_answer=retrieved[0].text if retrieved else "",
            tags=sample.tags,
        )
        samples.append(eval_sample)

    await client.close()
    return evaluate(samples, k=k)


def print_report(report):
    """Pretty-print an evaluation report."""
    print("\n" + "=" * 60)
    print("RAG EVALUATION REPORT")
    print("=" * 60)
    print(f"  Total samples:        {report.total_samples}")
    print(f"  Avg Precision@5:      {report.avg_precision_at_5}")
    print(f"  Avg Recall@5:         {report.avg_recall_at_5}")
    print(f"  Avg Faithfulness:     {report.avg_faithfulness}")
    print(f"  Avg Relevance:        {report.avg_relevance}")
    print("-" * 60)

    for i, s in enumerate(report.per_sample, 1):
        print(f"\n  [{i}] {s['question'][:60]}...")
        print(f"      P@5={s['precision_at_k']}  R@5={s['recall_at_k']}  "
              f"Faith={s['faithfulness']}  Rel={s['relevance']}")

    print("\n" + "=" * 60)

    # Also write JSON report
    report_path = Path(__file__).parent / "report.json"
    with open(report_path, "w") as f:
        json.dump({
            "total_samples": report.total_samples,
            "avg_precision_at_5": report.avg_precision_at_5,
            "avg_recall_at_5": report.avg_recall_at_5,
            "avg_faithfulness": report.avg_faithfulness,
            "avg_relevance": report.avg_relevance,
            "per_sample": report.per_sample,
        }, f, indent=2)
    print(f"\nReport saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Run RAG evaluation")
    parser.add_argument("--live", action="store_true", help="Run against live services")
    parser.add_argument("--k", type=int, default=5, help="Top-K for retrieval metrics")
    args = parser.parse_args()

    if args.live:
        report = asyncio.run(run_live_eval(k=args.k))
    else:
        report = asyncio.run(run_mock_eval(k=args.k))

    print_report(report)


if __name__ == "__main__":
    main()
