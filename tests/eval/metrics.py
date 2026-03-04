"""RAG evaluation metrics.

Measures retrieval precision, answer faithfulness, and answer relevance.
"""

from dataclasses import dataclass, field


@dataclass
class RetrievalResult:
    """A single retrieval result from the search pipeline."""

    chunk_id: str
    text: str
    score: float
    metadata: dict = field(default_factory=dict)


@dataclass
class EvalSample:
    """A single evaluation sample with question, expected answer, and context."""

    question: str
    expected_answer: str
    relevant_doc_ids: list[str]  # Ground truth doc IDs
    retrieved_results: list[RetrievalResult] = field(default_factory=list)
    generated_answer: str = ""
    tags: list[str] = field(default_factory=list)


def retrieval_precision_at_k(sample: EvalSample, k: int = 5) -> float:
    """Precision@K: fraction of top-K retrieved docs that are relevant.

    Args:
        sample: An evaluation sample with ground truth doc IDs and retrieved results.
        k: Number of top results to consider.

    Returns:
        Float between 0.0 and 1.0.
    """
    top_k = sample.retrieved_results[:k]
    if not top_k:
        return 0.0

    relevant_set = set(sample.relevant_doc_ids)
    hits = sum(
        1
        for r in top_k
        if r.chunk_id in relevant_set or r.metadata.get("doc_id") in relevant_set
    )
    return hits / len(top_k)


def retrieval_recall_at_k(sample: EvalSample, k: int = 5) -> float:
    """Recall@K: fraction of relevant docs found in top-K.

    Args:
        sample: An evaluation sample with ground truth doc IDs and retrieved results.
        k: Number of top results to consider.

    Returns:
        Float between 0.0 and 1.0.
    """
    if not sample.relevant_doc_ids:
        return 1.0

    top_k = sample.retrieved_results[:k]
    relevant_set = set(sample.relevant_doc_ids)
    found = set()
    for r in top_k:
        if r.chunk_id in relevant_set:
            found.add(r.chunk_id)
        if r.metadata.get("doc_id") in relevant_set:
            found.add(r.metadata["doc_id"])

    return len(found) / len(relevant_set)


def answer_faithfulness(
    generated_answer: str,
    retrieved_contexts: list[str],
) -> float:
    """Faithfulness: fraction of answer sentences grounded in retrieved context.

    Uses simple sentence-level overlap checking. For production, replace with
    an LLM-based NLI check.

    Args:
        generated_answer: The generated answer text.
        retrieved_contexts: List of retrieved context strings.

    Returns:
        Float between 0.0 and 1.0.
    """
    if not generated_answer.strip():
        return 0.0

    sentences = [s.strip() for s in generated_answer.split(".") if s.strip()]
    if not sentences:
        return 0.0

    combined_context = " ".join(retrieved_contexts).lower()
    grounded = 0

    for sentence in sentences:
        # Check if key words from the sentence appear in context
        words = sentence.lower().split()
        content_words = [w for w in words if len(w) > 3]  # Skip short words
        if not content_words:
            grounded += 1  # Skip trivial sentences
            continue

        overlap = sum(1 for w in content_words if w in combined_context)
        if overlap / len(content_words) >= 0.5:
            grounded += 1

    return grounded / len(sentences)


def answer_relevance(
    question: str,
    generated_answer: str,
) -> float:
    """Relevance: does the answer address the question?

    Uses keyword overlap as a simple heuristic. For production, replace with
    an LLM-based relevance scorer.

    Args:
        question: The original question.
        generated_answer: The generated answer.

    Returns:
        Float between 0.0 and 1.0.
    """
    if not generated_answer.strip():
        return 0.0

    q_words = set(question.lower().split())
    a_words = set(generated_answer.lower().split())

    # Remove common stop words
    stop_words = {"the", "a", "an", "is", "are", "was", "were", "do", "does", "how",
                  "what", "when", "where", "why", "who", "which", "that", "this",
                  "to", "for", "of", "in", "on", "at", "by", "and", "or", "not",
                  "can", "could", "should", "would", "will", "i", "you", "it", "my"}
    q_content = q_words - stop_words
    a_content = a_words - stop_words

    if not q_content:
        return 1.0

    overlap = len(q_content & a_content)
    return min(overlap / len(q_content), 1.0)


@dataclass
class EvalReport:
    """Aggregated evaluation report."""

    total_samples: int = 0
    avg_precision_at_5: float = 0.0
    avg_recall_at_5: float = 0.0
    avg_faithfulness: float = 0.0
    avg_relevance: float = 0.0
    per_sample: list[dict] = field(default_factory=list)


def evaluate(samples: list[EvalSample], k: int = 5) -> EvalReport:
    """Run all metrics on a set of eval samples.

    Args:
        samples: List of EvalSample objects.
        k: Top-K parameter for retrieval metrics.

    Returns:
        EvalReport with aggregated and per-sample metrics.
    """
    report = EvalReport(total_samples=len(samples))
    total_p, total_r, total_f, total_rel = 0.0, 0.0, 0.0, 0.0

    for sample in samples:
        p = retrieval_precision_at_k(sample, k)
        r = retrieval_recall_at_k(sample, k)
        contexts = [res.text for res in sample.retrieved_results[:k]]
        f = answer_faithfulness(sample.generated_answer, contexts)
        rel = answer_relevance(sample.question, sample.generated_answer)

        total_p += p
        total_r += r
        total_f += f
        total_rel += rel

        report.per_sample.append({
            "question": sample.question,
            "precision_at_k": round(p, 3),
            "recall_at_k": round(r, 3),
            "faithfulness": round(f, 3),
            "relevance": round(rel, 3),
        })

    n = len(samples) or 1
    report.avg_precision_at_5 = round(total_p / n, 3)
    report.avg_recall_at_5 = round(total_r / n, 3)
    report.avg_faithfulness = round(total_f / n, 3)
    report.avg_relevance = round(total_rel / n, 3)

    return report
