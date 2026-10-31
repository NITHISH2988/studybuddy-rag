"""Recall@k and reciprocal-rank evaluation for labeled retrieval queries."""

from collections.abc import Callable


def evaluate_retrieval(
    examples: list[dict],
    search: Callable[[str, int], list[dict]],
    *,
    k: int = 5,
) -> dict[str, float | int]:
    if not examples:
        raise ValueError("evaluation set cannot be empty")
    if k < 1:
        raise ValueError("k must be positive")
    recalls = []
    reciprocal_ranks = 0.0
    for example in examples:
        relevant = set(example["relevant_chunk_ids"])
        if not relevant:
            raise ValueError("each query must have at least one relevant chunk id")
        results = search(example["query"], k)
        result_ids = [item["chunk_id"] for item in results]
        matches = relevant.intersection(result_ids)
        recalls.append(len(matches) / len(relevant))
        reciprocal_ranks += next(
            (
                1.0 / rank
                for rank, chunk_id in enumerate(result_ids, start=1)
                if chunk_id in relevant
            ),
            0.0,
        )
    return {
        "queries": len(examples),
        f"recall@{k}": sum(recalls) / len(recalls),
        f"mrr@{k}": reciprocal_ranks / len(examples),
    }
