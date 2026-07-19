"""
Stage 6 — transpose per-claim/per-limitation verdicts into a per-citing-paper
view: for each citing paper that was actually matched to at least one claim or
limitation, list every relationship it has to the source paper.

This organizes the report around "which papers cited this, and what did each
one do" rather than "which claims held up" -- the former is what's actually
checkable and useful; the latter buries the citing papers inside a tally.

The exact citation-context quote is pulled here from the server's own cached
citation data by index, not re-transcribed by whichever model produced the
verdicts -- so the quote shown is always the authoritative original text, not
a possibly-paraphrased restatement.
"""


def _external_link(external_ids: dict) -> str | None:
    """Build a link for a citing paper from Semantic Scholar's externalIds,
    preferring arXiv (open, directly checkable) over a DOI resolver link.
    Returns None if neither is available -- not every citing paper has one.
    """
    if not external_ids:
        return None
    arxiv_id = external_ids.get("ArXiv")
    if arxiv_id:
        return f"https://arxiv.org/abs/{arxiv_id}"
    doi = external_ids.get("DOI")
    if doi:
        return f"https://doi.org/{doi}"
    return None


def aggregate_by_citation(
    citations: list[dict],
    claims: list[dict],
    verdicts_by_claim: dict,
    limitations: list[dict],
    verdicts_by_limitation: dict,
) -> list[dict]:
    """Returns one entry per citing paper that matched at least one claim or
    limitation, each with a link (if available), the exact quote, and every
    relationship (claim support/dispute/extend, or limitation resolution) it
    has to the source paper -- sorted influential-and-most-relationships
    first.
    """
    claims_by_id = {c["id"]: c for c in claims}
    limitations_by_id = {l["id"]: l for l in limitations}

    by_citation: dict[int, list[dict]] = {}

    for claim_id, entries in verdicts_by_claim.items():
        claim = claims_by_id.get(claim_id)
        if claim is None:
            continue
        for ctx_idx, verdict in entries:
            by_citation.setdefault(ctx_idx, []).append({
                "relationship_type": "claim",
                "item_id": claim_id,
                "item_text": claim["text"],
                "verdict": verdict["verdict"],
                "confidence": verdict.get("confidence", "n/a"),
                "reasoning": verdict["reasoning"],
            })

    for lim_id, entries in verdicts_by_limitation.items():
        lim = limitations_by_id.get(lim_id)
        if lim is None:
            continue
        for ctx_idx, verdict in entries:
            by_citation.setdefault(ctx_idx, []).append({
                "relationship_type": "limitation",
                "item_id": lim_id,
                "item_text": lim["text"],
                "verdict": verdict["verdict"],
                "confidence": verdict.get("confidence", "n/a"),
                "reasoning": verdict["reasoning"],
            })

    result = []
    for ctx_idx, relationships in by_citation.items():
        citation = citations[ctx_idx]
        result.append({
            "citation_index": ctx_idx,
            "title": citation["title"],
            "year": citation["year"],
            "is_influential": citation["is_influential"],
            "link": _external_link(citation.get("external_ids", {})),
            "quote": citation["context_text"],
            "relationships": relationships,
        })

    result.sort(key=lambda r: (not r["is_influential"], -len(r["relationships"]), -(r["year"] or 0)))
    return result


def unmatched_items(
    claims: list[dict], verdicts_by_claim: dict,
    limitations: list[dict], verdicts_by_limitation: dict,
) -> dict:
    """Claims/limitations with zero matched citations, surfaced explicitly so
    the reader knows what wasn't found evidence for rather than it silently
    disappearing from the report.
    """
    return {
        "claims": [c["text"] for c in claims if not verdicts_by_claim.get(c["id"])],
        "limitations": [l["text"] for l in limitations if not verdicts_by_limitation.get(l["id"])],
    }
