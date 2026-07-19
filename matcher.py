"""
Stage 4 — match extracted claims/limitations to citation contexts by embedding
cosine similarity. Used identically by both tracks; only the items being matched
differ.
"""
import numpy as np


def cosine_similarity(a: list[float], b: list[float]) -> float:
    vec_a, vec_b = np.array(a, dtype=float), np.array(b, dtype=float)
    denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    if denom == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


def match_items_to_contexts(
    items: list[dict],
    item_embeddings: dict[str, list[float]],
    contexts: list[dict],
    context_embeddings: list[list[float]],
    threshold: float = 0.55,
) -> dict[str, list[int]]:
    """For each item (a claim or a limitation, identified by its "id" field),
    return the indices into `contexts` whose embedding similarity to that item
    exceeds `threshold`.

    Contexts that match no item at all are simply absent from every list — that's
    intentional. Forcing a weak match onto the nearest item would inject noise
    into classification that's worse than leaving it unmatched; see the plan
    doc's note on this.

    threshold=0.55 is a starting point, not a validated value — the hand-check
    step in the README is where you'd tune this against real output.
    """
    matches: dict[str, list[int]] = {item["id"]: [] for item in items}
    for item in items:
        item_vec = item_embeddings[item["id"]]
        for idx, ctx_vec in enumerate(context_embeddings):
            if cosine_similarity(item_vec, ctx_vec) >= threshold:
                matches[item["id"]].append(idx)
    return matches
