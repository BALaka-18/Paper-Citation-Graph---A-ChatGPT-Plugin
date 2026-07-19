#!/usr/bin/env python3
"""
Paper accountability checker — MVP entrypoint.

Usage:
    python main.py https://arxiv.org/abs/2401.12345 [--cap 30] [--out report.md]

Pipeline (see plan doc / README.md for the full design):
    1. Fetch paper metadata (arXiv) and citation graph (Semantic Scholar)
    2. Extract claims (track A) and limitations (track B) from the abstract via Gemini
    3. Embed claims/limitations and citation contexts, match by cosine similarity
    4. Classify each matched pair: stance (track A) or resolution (track B) via Gemini
    5. Aggregate and render a two-section markdown report

MVP scope: abstract-only extraction, a single arXiv version, a citation cap
blended 70% by influence / 30% by recency. See README.md for what's deliberately
deferred to v2, and for the honesty caveats both tracks' output needs.
"""
import argparse
import os
import sys

from dotenv import load_dotenv

load_dotenv()

from arxiv_client import normalize_arxiv_id, fetch_metadata
from s2_client import get_paper_id_from_arxiv, fetch_citations, select_citation_sample
from llm_client import extract_claims, extract_limitations, embed_text, classify_stance, classify_resolution
from matcher import match_items_to_contexts
from aggregate import aggregate_by_citation, unmatched_items
from report import render_report
from cache import Cache


def run(arxiv_url_or_id: str, cap: int = 30, cache_path: str = "cache.db") -> str:
    if not os.environ.get("GEMINI_API_KEY"):
        sys.exit("GEMINI_API_KEY not set. Copy .env.example to .env and add your key.")

    arxiv_id = normalize_arxiv_id(arxiv_url_or_id)
    cache = Cache(cache_path)

    print(f"[1/6] Fetching paper metadata for {arxiv_id}...")
    meta = cache.get(arxiv_id, "meta")
    if meta is None:
        meta = fetch_metadata(arxiv_id)
        cache.set(arxiv_id, "meta", meta)

    print("[2/6] Fetching citation graph from Semantic Scholar...")
    citations = cache.get(arxiv_id, "citations")
    if citations is None:
        s2_id = get_paper_id_from_arxiv(arxiv_id)
        raw = fetch_citations(s2_id, max_raw=max(150, cap * 15))
        citations = select_citation_sample(raw, cap)
        cache.set(arxiv_id, "citations", citations)
    print(f"    -> {len(citations)} citations selected (cap={cap})")

    print("[3/6] Extracting claims and limitations from the abstract...")
    claims_data = cache.get(arxiv_id, "claims")
    if claims_data is None:
        claims_data = extract_claims(meta["title"], meta["abstract"]).model_dump()["claims"]
        cache.set(arxiv_id, "claims", claims_data)
    limitations_data = cache.get(arxiv_id, "limitations")
    if limitations_data is None:
        limitations_data = extract_limitations(meta["title"], meta["abstract"]).model_dump()["limitations"]
        cache.set(arxiv_id, "limitations", limitations_data)
    print(f"    -> {len(claims_data)} claims, {len(limitations_data)} limitations extracted")

    print("[4/6] Embedding and matching claims/limitations to citation contexts...")
    context_embeddings = [embed_text(c["context_text"]) for c in citations]
    claim_embeddings = {c["id"]: embed_text(c["text"]) for c in claims_data}
    limitation_embeddings = {l["id"]: embed_text(l["text"]) for l in limitations_data}

    claim_matches = match_items_to_contexts(claims_data, claim_embeddings, citations, context_embeddings)
    limitation_matches = match_items_to_contexts(
        limitations_data, limitation_embeddings, citations, context_embeddings
    )

    print("[5/6] Classifying stance and resolution for each matched pair...")
    print("      (free-tier rate limits mean this may pause and retry — see messages below if so)")
    verdicts_by_claim = {}
    for claim in claims_data:
        entries = []
        matched_idxs = claim_matches[claim["id"]]
        for i, ctx_idx in enumerate(matched_idxs):
            print(f"    stance: {claim['id']} <- citation {i + 1}/{len(matched_idxs)}")
            ctx = citations[ctx_idx]
            verdict = classify_stance(claim["text"], ctx["context_text"], ctx["intents"]).model_dump()
            entries.append((ctx_idx, verdict))
        verdicts_by_claim[claim["id"]] = entries

    verdicts_by_limitation = {}
    for lim in limitations_data:
        entries = []
        matched_idxs = limitation_matches[lim["id"]]
        for i, ctx_idx in enumerate(matched_idxs):
            print(f"    resolution: {lim['id']} <- citation {i + 1}/{len(matched_idxs)}")
            ctx = citations[ctx_idx]
            verdict = classify_resolution(lim["text"], ctx["context_text"], ctx["intents"]).model_dump()
            entries.append((ctx_idx, verdict))
        verdicts_by_limitation[lim["id"]] = entries

    print("[6/6] Aggregating and rendering report...")
    track_by_citation = aggregate_by_citation(citations, claims_data, verdicts_by_claim,
                                               limitations_data, verdicts_by_limitation)
    unmatched = unmatched_items(claims_data, verdicts_by_claim, limitations_data, verdicts_by_limitation)
    report_md = render_report(meta, track_by_citation, unmatched)

    cache.close()
    return report_md


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Paper accountability checker")
    parser.add_argument("arxiv_url", help="arXiv URL or id, e.g. https://arxiv.org/abs/2401.12345")
    parser.add_argument("--cap", type=int, default=30, help="max citations to analyze (default: 30)")
    parser.add_argument("--out", default="report.md", help="output markdown file path")
    args = parser.parse_args()

    report_markdown = run(args.arxiv_url, cap=args.cap)
    with open(args.out, "w") as f:
        f.write(report_markdown)
    print(f"\nReport written to {args.out}")
