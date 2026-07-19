"""
MCP server for the paper accountability checker, built to run as a ChatGPT App
via Developer Mode.

Architecture shift from the CLI version (main.py): there are no Gemini calls
and no embedding-based matching here. ChatGPT's own model does the claim and
limitation extraction, and the stance/resolution classification, using the
caller's normal ChatGPT usage instead of a separate metered API. This server's
job is just data plumbing:

    1. fetch_paper_and_citations — fetch the paper + its citation graph, hand
       it to the model along with instructions on how to reason over it.
    2. submit_report — the model calls this once, with its extracted claims/
       limitations and their evidence-backed verdicts, typed against a strict
       schema. This validates, aggregates, and renders the final report.

Reuses arxiv_client.py, s2_client.py, aggregate.py, report.py, and cache.py
unchanged from the CLI version. llm_client.py and matcher.py are not used at
all — their job is now done by ChatGPT's own reasoning.

Run:
    python mcp_server.py
Then tunnel it (see README's "Running as a ChatGPT App" section) and point
ChatGPT Developer Mode at the resulting HTTPS /mcp URL.
"""
from typing import Literal

from dotenv import load_dotenv

load_dotenv()

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import BaseModel, Field

from arxiv_client import normalize_arxiv_id, fetch_metadata
from s2_client import get_paper_id_from_arxiv, fetch_citations, select_citation_sample
from aggregate import aggregate_by_citation, unmatched_items
from report import render_report
from cache import Cache

SERVER_VERSION = "2026-07-18-dotenv-fix-v4"
DEFAULT_CAP = 30

mcp = FastMCP(
    "paper-accountability-checker",
    instructions=(
        "Checks whether a paper's central claims have been supported or disputed "
        "by later citing papers, and whether its self-acknowledged limitations "
        "have since been addressed. Call fetch_paper_and_citations with an arXiv "
        "URL or id first, reason over the result as instructed in its response, "
        "then call submit_report exactly once with your findings. Present "
        "submit_report's returned report_markdown as your entire answer — do not "
        "write a separate freeform summary instead of, or in addition to, it."
    ),
    # The MCP SDK's DNS-rebinding protection checks incoming Origin/Host headers
    # against an allowlist that defaults to empty when active -- which silently
    # rejects every real request (including ChatGPT's) with a 403, independent
    # of CORS. Disabling it here is a real security trade-off, but a reasonable
    # one for this setup specifically: this server is only ever reachable via a
    # random, unguessable, ephemeral tunnel URL you start yourself and never
    # publish anywhere -- not a permanently public deployment. If this ever
    # becomes a long-running public service, revisit this and set
    # allowed_origins/allowed_hosts explicitly instead of disabling protection.
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
)
cache = Cache("mcp_cache.db")

_REASONING_INSTRUCTIONS = """
You have been given a paper's title, abstract, and a list of citation contexts
from later papers that cite it (each with an index, the citing sentence(s),
an intent tag, an influence flag, and a year).

Do the following, then call submit_report with your results:

1. Extract 1-5 atomic, falsifiable claims from the abstract — specific,
   checkable assertions about the method or results, not motivation or
   background. If there are none, use an empty list rather than inventing one.

2. Extract any self-acknowledged limitations or open problems the paper admits
   to (e.g. "we leave X to future work", "does not handle Y"). Abstracts often
   don't state these explicitly — if none are stated or clearly implied, use
   an empty list. Do not infer limitations the paper doesn't mention.

3. For EACH claim, go through the citations list and, for any citation whose
   context actually relates to that claim, classify it as exactly one of:
   supports, disputes, extends, insufficient_info. Base this only on the
   context text given, not outside knowledge of the topic. Skip citations that
   don't relate to the claim at all rather than forcing a weak match.

4. For EACH limitation, go through the citations list the same way and
   classify related citations as exactly one of: addresses, partially_addresses,
   does_not_address. "Addresses" means the later paper CLAIMS to solve it — you
   are detecting that claim, not independently verifying it succeeded. Keep
   that distinction in your reasoning text.

5. Treat each citation's intent tag as a weak auxiliary signal only
   ("background" = general context, don't over-read it as agreement or
   disagreement) — never treat the intent tag itself as the verdict.

6. Do NOT invent a page number or section name for the citing paper — that
   information was not retrieved and is not available to you. The server will
   attach the exact citation-context quote to your verdict automatically; you
   do not need to (and should not try to) reproduce or paraphrase it yourself.

7. Assign each claim/limitation a short id (c1, c2, ... / l1, l2, ...) and
   call submit_report exactly once with the full arxiv_id, your claims list,
   and your limitations list. Reference citations by their index from the
   list above — an index outside that range will be dropped from the report.

8. After submit_report returns, present its report_markdown field as your
   entire response, unmodified. Do not additionally summarize, rephrase, or
   replace it with your own prose report — the structured tool output IS the
   answer.
"""


class ClaimEvidence(BaseModel):
    citation_index: int = Field(description="Index into the citations list this verdict is based on")
    verdict: Literal["supports", "disputes", "extends", "insufficient_info"]
    confidence: Literal["low", "medium", "high"]
    reasoning: str = Field(description="One or two sentences grounding the verdict in the citation's context text")


class ClaimResult(BaseModel):
    id: str = Field(description="Short id you assigned, e.g. c1")
    text: str = Field(description="The claim text as you extracted it")
    evidence: list[ClaimEvidence] = Field(default_factory=list)


class LimitationEvidence(BaseModel):
    citation_index: int = Field(description="Index into the citations list this verdict is based on")
    verdict: Literal["addresses", "partially_addresses", "does_not_address"]
    confidence: Literal["low", "medium", "high"]
    reasoning: str = Field(description="One or two sentences grounding the verdict in the citation's context text")


class LimitationResult(BaseModel):
    id: str = Field(description="Short id you assigned, e.g. l1")
    text: str = Field(description="The limitation text as you extracted it")
    evidence: list[LimitationEvidence] = Field(default_factory=list)


@mcp.tool()
def fetch_paper_and_citations(arxiv_url: str, cap: int = DEFAULT_CAP) -> dict:
    """Fetch a paper's title/abstract from arXiv and its citation graph from
    Semantic Scholar, sampled and capped (70% by influence, 30% by recency, to
    avoid skewing toward old, already-established citations). Returns the
    paper info, an indexed citations list, instructions for how to reason over
    them before calling submit_report, and an honest note if fewer citations
    were available than requested.
    """
    arxiv_id = normalize_arxiv_id(arxiv_url)

    meta = cache.get(arxiv_id, "meta")
    if meta is None:
        meta = fetch_metadata(arxiv_id)
        cache.set(arxiv_id, "meta", meta)

    citations = cache.get(arxiv_id, "citations")
    if citations is None:
        s2_id = get_paper_id_from_arxiv(arxiv_id)
        # max_raw scaled generously (not just cap*4) because the real limiter
        # for heavily-cited papers isn't request volume, it's how many of a
        # paper's citations have Semantic Scholar-extracted context text at
        # all -- that extraction depends on S2 having full-text access to the
        # citing paper (open-access corpus), so a landmark paper cited by many
        # closed-access venues can have far fewer *usable* citations than its
        # total citation count suggests. Searching deeper trades off against
        # a longer-running tool call, not against rate-limit risk (the
        # inter-page delay and backoff in s2_client.py handle that part).
        raw = fetch_citations(s2_id, max_raw=max(150, cap * 15))
        citations = select_citation_sample(raw, cap)
        cache.set(arxiv_id, "citations", citations)

    indexed_citations = [
        {
            "index": i,
            "context_text": c["context_text"],
            "intents": c["intents"],
            "is_influential": c["is_influential"],
            "year": c["year"],
            "title": c["title"],
        }
        for i, c in enumerate(citations)
    ]

    result = {
        "server_version": SERVER_VERSION,
        "arxiv_id": arxiv_id,
        "paper": meta,
        "citations": indexed_citations,
        "instructions": _REASONING_INSTRUCTIONS,
    }

    # Server-computed, factual note on count -- not left for the model to
    # narrate its own guess about why a number came out lower than requested.
    actual = len(citations)
    if actual < cap:
        result["citation_count_note"] = (
            f"Requested up to {cap} citations; found {actual} usable citation "
            f"contexts (citations with retrievable snippet text) for this paper. "
            f"This is very likely a genuine data-coverage limit, not an error: "
            f"Semantic Scholar can only extract a citation-context sentence from "
            f"a citing paper it has full-text access to (its open-access corpus), "
            f"so heavily-cited papers with many closed-access citers often have "
            f"far fewer usable citations than their total citation count suggests. "
            f"State this actual number and the likely reason plainly if it comes "
            f"up — do not invent a different explanation such as rate-limiting "
            f"unless you actually encountered an error message."
        )

    return result


@mcp.tool()
def submit_report(arxiv_id: str, claims: list[ClaimResult], limitations: list[LimitationResult]) -> dict:
    """Call this exactly once, after reasoning over fetch_paper_and_citations'
    output, with your extracted claims/limitations and their evidence-backed
    verdicts. Renders and returns the final two-track report.
    """
    meta = cache.get(arxiv_id, "meta")
    citations = cache.get(arxiv_id, "citations")
    if meta is None or citations is None:
        return {"error": f"No cached paper/citations found for arxiv_id={arxiv_id!r} — "
                          f"call fetch_paper_and_citations first."}

    num_citations = len(citations)
    dropped = []

    def _valid_evidence(evidence_list, item_id):
        kept = []
        for e in evidence_list:
            if 0 <= e.citation_index < num_citations:
                kept.append(e)
            else:
                dropped.append(f"{item_id}: citation_index {e.citation_index} out of range (0-{num_citations - 1})")
        return kept

    claims_data = [{"id": c.id, "text": c.text, "source": "abstract"} for c in claims]
    verdicts_by_claim = {}
    for c in claims:
        valid = _valid_evidence(c.evidence, c.id)
        verdicts_by_claim[c.id] = [
            (e.citation_index, {"verdict": e.verdict, "confidence": e.confidence, "reasoning": e.reasoning})
            for e in valid
        ]

    limitations_data = [{"id": l.id, "text": l.text, "source": "abstract"} for l in limitations]
    verdicts_by_limitation = {}
    for l in limitations:
        valid = _valid_evidence(l.evidence, l.id)
        verdicts_by_limitation[l.id] = [
            (e.citation_index, {"verdict": e.verdict, "confidence": e.confidence, "reasoning": e.reasoning})
            for e in valid
        ]

    track_by_citation = aggregate_by_citation(citations, claims_data, verdicts_by_claim,
                                               limitations_data, verdicts_by_limitation)
    unmatched = unmatched_items(claims_data, verdicts_by_claim, limitations_data, verdicts_by_limitation)
    report_md = render_report(meta, track_by_citation, unmatched)

    result = {"report_markdown": report_md}
    if dropped:
        result["warnings"] = dropped
    return result


if __name__ == "__main__":
    import uvicorn
    from starlette.middleware.cors import CORSMiddleware

    # FastMCP's default streamable-http server sends no CORS headers at all.
    # If ChatGPT's connector-creation step probes this server from a browser
    # context (rather than server-side), the browser blocks the response
    # before ChatGPT ever sees it -- indistinguishable from the server being
    # broken. Wrapping with permissive CORS fixes this; safe here since this
    # only ever runs behind an ephemeral tunnel URL you control, not deployed
    # as a public service.
    app = CORSMiddleware(
        mcp.streamable_http_app(),
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    uvicorn.run(app, host=mcp.settings.host, port=mcp.settings.port)
