"""
Stage 7 — render the citing-paper-centric report as tables: one evidence
table with a row per (citing paper, claim/limitation) relationship, where a
paper's identity/year/status/quote are only shown on its first row -- blank
on subsequent rows for the same paper, the standard markdown convention for
"this belongs to the group above" since markdown tables have no real rowspan.
Followed by a compact claim/limitation-level rollup and the unmatched list.
"""

_VERDICT_LABELS = {
    "supports": "Supports",
    "disputes": "Disputes",
    "extends": "Extends",
    "insufficient_info": "Insufficient info",
    "addresses": "Addresses",
    "partially_addresses": "Partially addresses",
    "does_not_address": "Does not address",
}


def _escape_cell(text: str) -> str:
    """Markdown table cells break on literal pipe characters and newlines."""
    return text.replace("|", "/").replace("\n", " ").strip()


def _evidence_table(citation_entries: list[dict]) -> str:
    lines = [
        "| Citing paper | Year | Status | Claim / limitation evaluated | Verdict | Confidence | Why | Quote |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for entry in citation_entries:
        title_display = _escape_cell(
            f"[{entry['title']}]({entry['link']})" if entry["link"] else entry["title"]
        )
        status = "Influential" if entry["is_influential"] else "Non-influential"
        year_display = str(entry["year"]) if entry["year"] else "—"
        quote_display = _escape_cell(entry["quote"])

        for i, rel in enumerate(entry["relationships"]):
            label = "Claim" if rel["relationship_type"] == "claim" else "Limitation"
            item_display = _escape_cell(f"{label}: \"{rel['item_text']}\"")
            verdict_display = _VERDICT_LABELS.get(rel["verdict"], rel["verdict"])
            confidence_display = rel["confidence"].capitalize() if isinstance(rel["confidence"], str) else rel["confidence"]
            reasoning_display = _escape_cell(rel["reasoning"])

            if i == 0:
                lines.append(
                    f"| {title_display} | {year_display} | {status} | {item_display} | "
                    f"{verdict_display} | {confidence_display} | {reasoning_display} | {quote_display} |"
                )
            else:
                # Same paper as the row above -- identity/year/status/quote
                # are identical (one citation context per citing paper), so
                # blank them rather than repeat, per the standard markdown
                # convention for grouped rows.
                lines.append(
                    f"|  |  |  | {item_display} | {verdict_display} | "
                    f"{confidence_display} | {reasoning_display} |  |"
                )
    return "\n".join(lines) + "\n"


def _claim_level_summary(citation_entries: list[dict]) -> str:
    """Compact rollup: for each distinct claim/limitation, how many citing
    papers landed in each verdict bucket."""
    tallies: dict[str, dict[str, int]] = {}
    for entry in citation_entries:
        for rel in entry["relationships"]:
            label = "Claim" if rel["relationship_type"] == "claim" else "Limitation"
            key = f"{label}: {rel['item_text']}"
            tallies.setdefault(key, {})
            tallies[key][rel["verdict"]] = tallies[key].get(rel["verdict"], 0) + 1

    if not tallies:
        return ""

    lines = [
        "\n---\n### Claim / limitation summary\n",
        "| Claim or limitation | Supports | Disputes | Extends | Addresses | Partially addresses | Does not address |",
        "|---|---|---|---|---|---|---|",
    ]
    for item_text, tally in tallies.items():
        lines.append(
            f"| {_escape_cell(item_text)} | {tally.get('supports', 0)} | {tally.get('disputes', 0)} | "
            f"{tally.get('extends', 0)} | {tally.get('addresses', 0)} | "
            f"{tally.get('partially_addresses', 0)} | {tally.get('does_not_address', 0)} |"
        )
    return "\n".join(lines) + "\n"


def _unmatched_section(unmatched: dict) -> str:
    if not (unmatched["claims"] or unmatched["limitations"]):
        return ""
    parts = ["\n---\n### Not covered by any matched citation\n"]
    if unmatched["claims"]:
        parts.append("**Claims with no matching citation found:**")
        parts.extend(f"- {c}" for c in unmatched["claims"])
        parts.append("")
    if unmatched["limitations"]:
        parts.append("**Limitations with no matching citation found:**")
        parts.extend(f"- {l}" for l in unmatched["limitations"])
        parts.append("")
    return "\n".join(parts)


def _caveat_note() -> str:
    return (
        "\n---\n_Citation context is limited to the 1-3 sentence excerpt Semantic "
        "Scholar retrieved around each citation — not a page or section number. "
        "Full-text parsing to pin down exact sections isn't implemented yet, so "
        "the quote above is the closest available pointer to \"where to check\" "
        "in the citing paper._\n"
    )


def render_report(paper_meta: dict, citation_entries: list[dict], unmatched: dict) -> str:
    parts = [
        f"# Paper accountability report: {paper_meta['title']}",
        f"\narXiv: {paper_meta['arxiv_id']} — {paper_meta['id_url']}\n",
    ]

    if not citation_entries:
        parts.append(
            "\n_No citing paper's context matched closely enough to any extracted "
            "claim or limitation to report._\n"
        )
    else:
        parts.append(
            f"\n**{len(citation_entries)} citing paper(s)** with at least one matched "
            f"relationship, most relevant (influential, most relationships) first.\n"
        )
        parts.append(_evidence_table(citation_entries))
        parts.append(_claim_level_summary(citation_entries))

    parts.append(_unmatched_section(unmatched))
    parts.append(_caveat_note())
    return "\n".join(parts)
