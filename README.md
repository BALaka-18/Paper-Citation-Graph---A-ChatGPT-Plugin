# Paper Citation Graph - A ChatGPT Plugin

Give it an arXiv link. It finds papers that later cited it, and for each one tells you whether it **supports**, **disputes**, or **extends** a specific claim the original paper made — or whether it **addresses** a limitation the original paper admitted to. Every verdict comes with pointwise reasoning and the exact citation-context quote it's based on.

Not a citation-count graph (Connected Papers, ResearchRabbit, and Litmaps already do that well, and are out of scope here on purpose) — the point of this tool is the claim/limitation-level judgment underneath the citation graph, which those tools don't attempt.

**New here?** Start with `SETUP.md` for step-by-step local setup. This file covers what the project is, how it's organized, what it currently gets wrong on purpose (i.e. known scope limits, not bugs), and how to read the issues that tend to come up.

## What a report looks like

One evidence table, one row per (citing paper, claim-or-limitation) relationship — a paper matched to multiple claims gets one row per claim, with its name/year/status/quote shown once and left blank on the rows below it (the standard convention for "this belongs to the group above," since markdown tables have no real merged cells). Followed by a compact claim-level rollup (how many citing papers landed in each verdict bucket, per claim), and a list of any claims/limitations that no citation matched closely enough to evaluate.

## Architecture

```
arxiv_client.py    Fetch paper title/abstract from arXiv
s2_client.py       Fetch the citation graph from Semantic Scholar; sample it
                    (70% by influence, 30% by recency) so it isn't skewed
                    toward old, already-established citations
models.py          Shared Pydantic schemas for claims/limitations/verdicts
aggregate.py       Transpose per-claim/per-limitation verdicts into a
                    per-citing-paper view (one entry per citing paper, with
                    every relationship it has to the source paper)
report.py          Render that view as the markdown evidence table + rollup
cache.py           SQLite cache so re-running the same paper doesn't re-fetch
                    or re-classify from scratch

matcher.py         Embedding-similarity matching between
                    claims/limitations and citation contexts
main.py            

mcp_server.py       Entry point — exposes fetch_paper_and_citations and
                    submit_report as MCP tools; ChatGPT's own model does the
                    extraction/classification, this file just does data
                    plumbing and validates the model's structured output
                    (Mode B does not use llm_client.py or matcher.py at all)
```

## Current scope

1. **Abstract-only extraction.** Claims and limitations come from the title + abstract, not the full paper. This particularly weakens limitation-finding — abstracts rarely state limitations explicitly, so an empty "no limitations found" result is expected often, not a bug. Full-text parsing is the natural next step, not yet built.
2. **Both classifiers are unvalidated.** There's no existing benchmark for either task — SciFact is the closest analog for the claim-support/dispute judgment and still isn't the same task; the limitation-resolution judgment has no benchmark at all. Before trusting output on a paper you don't already know, run it on 2-3 papers where you know the field's consensus and check whether the verdicts look right.
3. **"Addresses" means claims to address, not verified to.** A citing paper "addressing" a limitation reflects that paper's own framing of its contribution — the tool detects the claim, it doesn't independently verify the paper actually succeeded. The report says this explicitly; don't summarize it away.
4. **Citation context is short** — a 1-3 sentence excerpt from Semantic Scholar, not a page or section number. That excerpt (shown as the report's quote) is the closest available pointer to "where to check" in the citing paper; true page/section references aren't available from this data source.
5. **Citation coverage isn't the same as citation count.** Semantic Scholar can only extract a context sentence from a citing paper it has full-text access to (its open-access corpus) — a landmark, heavily-cited paper can still return far fewer *usable* citations than its total citation count implies, because most of its citing papers are behind closed access. A low count on a famous paper is often this, not a bug.

## Known issues and how to read them

- **`Error creating connector` in ChatGPT** — usually one of: the tunnel/server isn't actually running, the URL is missing `/mcp`, or (less common) a platform-side timeout unrelated to your setup. Restart both terminals, get a fresh tunnel URL, retry.
- **`403 Forbidden: Invalid Origin header` / `421 Misdirected Request`** — the origin/host allowlist rejected the request. For local + tunnel use, set `ENABLE_HOST_PROTECTION=false`. For a real deployment, set `ALLOWED_HOSTS` to your actual domain.
- **The model writes its own summary instead of using the report format** — nothing forces ChatGPT to call `submit_report`; a vague prompt like "evaluate this paper" leaves it free to improvise instead. Use the specific prompt in `SETUP.md`'s Mode B section, which names the tool calls explicitly and tells the model to return `report_markdown` verbatim.
- **Output looks like an old version (wrong format, missing fields)** — `mcp_server.py`'s `fetch_paper_and_citations` returns a `server_version` field on every call. If it's missing, or doesn't match the version string at the top of `mcp_server.py`, the running server is stale — stop both terminals, confirm you're running the current file, and restart both (server first, then the tunnel).
- **Fewer citations than the `cap` you asked for** — check the `citation_count_note` field in the tool output first; it states the actual count and the most likely reason (usually genuine data-coverage limits on a heavily-cited paper, not an error) rather than leaving the model to guess.
