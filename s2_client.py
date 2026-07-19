"""
Stage 1b — fetch the citation graph from Semantic Scholar.

Requires SEMANTIC_SCHOLAR_API_KEY to be set for a stable per-key rate limit;
falls back to the shared unauthenticated pool if unset (works, just less
predictable under load — fine for single-paper runs).
"""
import os
import random
import time

import requests

S2_BASE = "https://api.semanticscholar.org/graph/v1"
_CITATION_FIELDS = "contexts,intents,isInfluential,year,title,externalIds,abstract"

# Exponential backoff with jitter on 429s: 2^retry seconds (capped at
# _MAX_BACKOFF_SECONDS) plus up to 1s of jitter, giving up after _MAX_RETRIES.
# Kept short deliberately: a single MCP tool call that hangs too long risks
# timing out on the calling side (ChatGPT, Claude, etc.) before the retry even
# finishes, which looks like a hard failure regardless of whether the request
# would eventually have succeeded. Better to fail fast and clearly than to
# retry patiently and time out silently.
_MAX_RETRIES = 4
_MAX_BACKOFF_SECONDS = 12

# Small proactive gap between paginated requests, separate from the reactive
# backoff above -- spreads out a burst of 2-5 sequential page fetches instead
# of firing them back-to-back, which is what actually trips the rate limit on
# a heavily-cited paper in the first place.
_INTER_PAGE_DELAY_SECONDS = 0.75


def _headers() -> dict:
    key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    return {"x-api-key": key} if key else {}


def _get_with_backoff(url: str, params: dict) -> requests.Response:
    """GET with exponential backoff + jitter on 429, raising after _MAX_RETRIES."""
    for attempt in range(_MAX_RETRIES + 1):
        resp = requests.get(url, headers=_headers(), params=params, timeout=30)
        if resp.status_code != 429:
            resp.raise_for_status()
            return resp
        if attempt == _MAX_RETRIES:
            raise RuntimeError(
                f"Semantic Scholar rate limit exceeded after {_MAX_RETRIES} retries on {url}"
            )
        backoff = min(_MAX_BACKOFF_SECONDS, 2 ** attempt) + random.uniform(0, 1)
        time.sleep(backoff)
    raise AssertionError("unreachable")  # loop always returns or raises above


def get_paper_id_from_arxiv(arxiv_id: str) -> str:
    """Cross-walk an arXiv id to a Semantic Scholar paperId.

    S2 keys papers by the version-less arXiv id, so v2/v3 suffixes are stripped
    before the lookup.
    """
    bare_id = arxiv_id.split("v")[0]
    url = f"{S2_BASE}/paper/arXiv:{bare_id}"
    resp = _get_with_backoff(url, {"fields": "paperId,title"})
    data = resp.json()
    if "paperId" not in data:
        raise ValueError(f"Semantic Scholar has no record for arXiv:{bare_id}")
    return data["paperId"]


def fetch_citations(paper_id: str, max_raw: int = 150) -> list[dict]:
    """Fetch up to max_raw citing papers, keeping only fields needed downstream.

    max_raw default lowered from an earlier version that always fetched up to
    500 regardless of the actual requested cap — that meant up to 5 sequential
    paginated requests even when the caller only wanted 30 final citations,
    which is exactly what tripped Semantic Scholar's rate limit on
    heavily-cited papers. Callers should pass max_raw scaled to their actual
    cap (see select_citation_sample) rather than relying on this default alone.

    Filters out entries with no citation-context text at fetch time — they carry
    no signal for stance/resolution classification and would just add noise to
    the sampling step below.
    """
    citations: list[dict] = []
    offset = 0
    limit = 100

    while len(citations) < max_raw:
        url = f"{S2_BASE}/paper/{paper_id}/citations"
        params = {"fields": _CITATION_FIELDS, "offset": offset, "limit": limit}
        resp = _get_with_backoff(url, params)
        data = resp.json()
        batch = data.get("data", [])
        if not batch:
            break

        for item in batch:
            if not item.get("contexts"):
                continue
            citing = item.get("citingPaper", {}) or {}
            citations.append({
                "context_text": " ".join(item["contexts"]),
                "intents": item.get("intents", []),
                "is_influential": bool(item.get("isInfluential", False)),
                "year": citing.get("year"),
                "title": citing.get("title") or "(untitled)",
                "external_ids": citing.get("externalIds", {}) or {},
            })

        offset += limit
        if "next" not in data:
            break
        time.sleep(_INTER_PAGE_DELAY_SECONDS)

    return citations


def select_citation_sample(citations: list[dict], cap: int) -> list[dict]:
    """Blend influence and recency so the sample isn't skewed toward old,
    already-established citations — a pure top-influence cap tends to miss
    recent disputes/resolutions, since influence accrues over time.

    Roughly 70% by influence (ties broken by recency), 30% by pure recency
    among what's left, deduplicated by title+year.
    """
    if len(citations) <= cap:
        return citations

    n_by_influence = int(cap * 0.7)
    n_by_recency = cap - n_by_influence

    ranked_by_influence = sorted(
        citations, key=lambda c: (not c["is_influential"], -(c["year"] or 0))
    )
    top_influence = ranked_by_influence[:n_by_influence]

    seen = {(c["title"], c["year"]) for c in top_influence}
    remaining = [c for c in citations if (c["title"], c["year"]) not in seen]
    ranked_by_recency = sorted(remaining, key=lambda c: -(c["year"] or 0))
    top_recent = ranked_by_recency[:n_by_recency]

    return top_influence + top_recent
