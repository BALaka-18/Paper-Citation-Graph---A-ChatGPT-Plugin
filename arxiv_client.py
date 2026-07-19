"""
Stage 1a — fetch paper metadata from arXiv.

MVP scope: title + abstract only. Full-text (PDF/LaTeX) parsing for richer claim
and limitation extraction is deferred to v2 — see README.md for why, and what that
costs the limitation-extraction track specifically.
"""
import re
import time
import xml.etree.ElementTree as ET

import requests

ARXIV_API = "http://export.arxiv.org/api/query"
ATOM_NS = "{http://www.w3.org/2005/Atom}"

# arXiv's terms of use ask for no more than 1 request per 3 seconds, single
# connection. At MVP scale (one paper per run) this costs a few seconds total.
_ARXIV_MIN_INTERVAL_SECONDS = 3


def normalize_arxiv_id(url_or_id: str) -> str:
    """Extract a bare arXiv id (with version suffix if present) from a URL or raw id.

    Accepts forms like:
      https://arxiv.org/abs/2401.12345
      https://arxiv.org/abs/2401.12345v2
      https://arxiv.org/pdf/2401.12345
      2401.12345
    """
    match = re.search(r"(\d{4}\.\d{4,5}(v\d+)?)", url_or_id)
    if not match:
        raise ValueError(f"Could not find an arXiv id in: {url_or_id!r}")
    return match.group(1)


def fetch_metadata(arxiv_id: str) -> dict:
    """Fetch title, abstract, and canonical URL for a single arXiv paper.

    Returns a dict with keys: arxiv_id, title, abstract, id_url.
    Raises ValueError if arXiv returns no matching entry (e.g. bad id, withdrawn paper).
    """
    time.sleep(_ARXIV_MIN_INTERVAL_SECONDS)
    resp = requests.get(ARXIV_API, params={"id_list": arxiv_id}, timeout=30)
    resp.raise_for_status()

    root = ET.fromstring(resp.text)
    entry = root.find(f"{ATOM_NS}entry")
    if entry is None:
        raise ValueError(f"arXiv returned no entry for id {arxiv_id!r} — check the id/URL is correct")

    title_el = entry.find(f"{ATOM_NS}title")
    summary_el = entry.find(f"{ATOM_NS}summary")
    id_el = entry.find(f"{ATOM_NS}id")

    if title_el is None or summary_el is None:
        raise ValueError(f"arXiv entry for {arxiv_id!r} is missing title or abstract")

    return {
        "arxiv_id": arxiv_id,
        "title": " ".join(title_el.text.split()),
        "abstract": " ".join(summary_el.text.split()),
        "id_url": id_el.text.strip() if id_el is not None else f"https://arxiv.org/abs/{arxiv_id}",
    }
