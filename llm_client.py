"""
Stages 2A/2B (extraction), 4 (embeddings), 5A/5B (classification) — every call to
Gemini goes through this module.

Model names are configurable via env vars because free-tier model access has been
shifting (see README.md) — gemini-2.5-flash was deprecated for new API keys in
2026, which is why the default below is gemini-3.5-flash. If that also 404s on
your key by the time you read this, override GEMINI_MODEL in .env rather than
editing this file.
"""
import hashlib
import itertools
import os
import re
import time

from google import genai
from google.genai import types
from google.genai import errors

from models import Claim, ClaimList, Limitation, LimitationList, StanceVerdict, ResolutionVerdict

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")
GEMINI_EMBED_MODEL = os.environ.get("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")

# Set MOCK_LLM=1 in .env to run the entire pipeline against canned responses
# instead of real Gemini calls. This exists because debugging the pipeline's
# wiring (does stage 4 correctly feed stage 5? does the report render right?)
# doesn't need real model output, but was silently spending real free-tier
# quota every time regardless — exactly the quota that then wasn't available
# for the real analysis run. Use this while iterating on code; turn it off
# for an actual paper analysis.
_MOCK_MODE = os.environ.get("MOCK_LLM", "0") == "1"
if _MOCK_MODE:
    print("    [MOCK_LLM=1] Using canned responses — no real Gemini calls will be made.")

_mock_stance_cycle = itertools.cycle(["supports", "disputes", "extends", "insufficient_info"])
_mock_resolution_cycle = itertools.cycle(["addresses", "partially_addresses", "does_not_address"])

# The free tier's requests-per-minute limit is low enough (observed as low as 5
# RPM on gemini-3.5-flash) that stage 3's back-to-back extraction calls, and
# especially stage 5's per-citation classification loop, hit it on almost any
# real run -- this isn't a sign of unusually heavy usage, it's just how tight
# the free tier currently is. Retrying with the server's own suggested delay
# (rather than a fixed guess) means this keeps working if the number changes
# again without needing a code update.
_MAX_RETRIES = 8
_DEFAULT_RETRY_DELAY = 15.0

_client = None


def get_client() -> genai.Client:
    """Lazily construct a single shared client (reads GEMINI_API_KEY from the
    environment automatically — no need to pass it explicitly)."""
    global _client
    if _client is None:
        _client = genai.Client()
    return _client


def _extract_retry_delay(error: Exception) -> float:
    """Parse the server-suggested retry delay out of a rate-limit error's
    message (e.g. "'retryDelay': '22s'"), falling back to a default if the
    format isn't found. Parsed from the string representation rather than a
    specific attribute, since the exact error object shape has already proven
    fragile once in this codebase (see embed_text's note below) and a string
    search degrades more gracefully than an AttributeError would.
    """
    match = re.search(r"'retryDelay':\s*'(\d+(?:\.\d+)?)s'", str(error))
    return float(match.group(1)) if match else _DEFAULT_RETRY_DELAY


def _is_daily_quota_error(error: Exception) -> bool:
    """Distinguish a daily (RPD) quota exhaustion from a per-minute (RPM) one.
    Google's quotaId strings include 'PerDay' for daily limits — retrying a
    daily exhaustion with a short sleep is pointless (it won't clear for
    hours), so this should fail fast with a clear message instead of spinning.
    """
    return "PerDay" in str(error)


def _call_with_rate_limit_retry(fn):
    """Call fn() (a zero-arg callable), retrying on free-tier RESOURCE_EXHAUSTED
    errors with the server's suggested delay, up to _MAX_RETRIES times.

    Daily-quota exhaustion is treated differently: it fails immediately with a
    clear explanation rather than retrying uselessly for a couple of minutes
    and then failing anyway with a confusing generic error.
    """
    for attempt in range(_MAX_RETRIES + 1):
        try:
            return fn()
        except errors.ClientError as e:
            if "RESOURCE_EXHAUSTED" not in str(e):
                raise
            if _is_daily_quota_error(e):
                raise RuntimeError(
                    "Gemini's free-tier DAILY request quota is exhausted for this "
                    "model/project. This will not clear with a retry — daily quotas "
                    "reset at midnight Pacific Time. Options: wait for reset, use "
                    "MOCK_LLM=1 to keep developing/testing without hitting the API "
                    "(see README.md), or switch to a different free provider "
                    "(Groq/OpenRouter) for now."
                ) from e
            if attempt == _MAX_RETRIES:
                raise
            delay = _extract_retry_delay(e)
            print(f"    (Gemini free-tier per-minute rate limit hit, waiting {delay:.0f}s — "
                  f"retry {attempt + 1}/{_MAX_RETRIES})")
            time.sleep(delay)


def _structured_call(prompt: str, schema):
    """Run a Gemini call constrained to a Pydantic schema and parse the result.

    Uses response.text + model_validate_json rather than any SDK-provided
    .parsed convenience attribute, since that attribute's availability has
    varied across google-genai SDK versions — this path is stable regardless.
    """
    def _do_call():
        client = get_client()
        resp = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema,
                temperature=0.1,
            ),
        )
        return schema.model_validate_json(resp.text)

    return _call_with_rate_limit_retry(_do_call)


# ---------------------------------------------------------------------------
# Stage 2A — extract claims
# ---------------------------------------------------------------------------

_CLAIM_PROMPT = """You are extracting atomic, falsifiable claims from a research paper's title and abstract.

A claim is a specific, checkable assertion the paper makes about its method, results, or the world
(e.g. "our method improves accuracy by 12% over baseline X"), not a generic statement of motivation
or background (e.g. "accuracy matters in this domain").

If the abstract contains no clearly checkable claims, return an empty list rather than inventing one.

Title: {title}
Abstract: {abstract}

Return each claim with a short id (c1, c2, ...), the claim text, and source="abstract".
"""


def extract_claims(title: str, abstract: str) -> ClaimList:
    if _MOCK_MODE:
        return ClaimList(claims=[
            Claim(id="c1", text=f"[MOCK] '{title[:50]}' improves over its stated baseline", source="abstract"),
            Claim(id="c2", text="[MOCK] The proposed method generalizes beyond the training setting", source="abstract"),
        ])
    prompt = _CLAIM_PROMPT.format(title=title, abstract=abstract)
    return _structured_call(prompt, ClaimList)


# ---------------------------------------------------------------------------
# Stage 2B — extract limitations
# ---------------------------------------------------------------------------

_LIMITATION_PROMPT = """You are extracting self-acknowledged limitations or open problems from a
research paper's title and abstract.

Only extract limitations the paper itself states or clearly implies it hasn't solved
(e.g. "we leave X to future work", "our approach does not handle Y", "requires Z which may not
always be available"). Abstracts often do not state limitations explicitly — if none are stated
or clearly implied, return an empty list. Do not infer limitations the paper doesn't mention.

Title: {title}
Abstract: {abstract}

Return each limitation with a short id (l1, l2, ...), the limitation text, and source="abstract".
"""


def extract_limitations(title: str, abstract: str) -> LimitationList:
    if _MOCK_MODE:
        return LimitationList(limitations=[
            Limitation(id="l1", text=f"[MOCK] '{title[:50]}' does not handle the noisy-label case", source="abstract"),
        ])
    prompt = _LIMITATION_PROMPT.format(title=title, abstract=abstract)
    return _structured_call(prompt, LimitationList)


# ---------------------------------------------------------------------------
# Stage 4 — embeddings
# ---------------------------------------------------------------------------

def embed_text(text: str) -> list[float]:
    """Return a single embedding vector for the given text.

    NOTE: the exact response shape (result.embeddings[0].values vs a bare list)
    has shifted across google-genai SDK releases. If this raises an
    AttributeError on your installed version, print(result) once to see the
    actual shape and adjust the line below — it's a one-line fix.
    """
    if _MOCK_MODE:
        # Deterministic, text-dependent pseudo-embedding (SHA-256 bytes -> floats)
        # so cosine similarity in matcher.py still produces varied, reproducible
        # results for wiring tests — not semantically meaningful, just stable.
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [b / 255.0 for b in digest[:16]]

    def _do_call():
        client = get_client()
        result = client.models.embed_content(model=GEMINI_EMBED_MODEL, contents=text)
        embedding = result.embeddings[0]
        return embedding.values if hasattr(embedding, "values") else embedding

    return _call_with_rate_limit_retry(_do_call)


# ---------------------------------------------------------------------------
# Stage 5A — classify stance (track A: claim durability)
# ---------------------------------------------------------------------------

_STANCE_PROMPT = """You are checking whether a later paper's citation of an earlier paper supports,
disputes, or extends a specific claim from the earlier paper.

Claim from the earlier paper: "{claim_text}"

Citation context from the later paper (the sentence(s) where it cites the earlier paper):
"{context_text}"

Semantic Scholar's automatic citation-intent tag for this citation: {intent}
("background" = general context only; "method" = reuses the approach; "result" = compares results.
Treat "background" as weak evidence either way — don't over-read a background mention as agreement.)

Classify the relationship as exactly one of:
- "supports": the context provides evidence consistent with the claim
- "disputes": the context provides evidence against the claim
- "extends": the context builds on, qualifies, or narrows the scope of the claim without directly
  agreeing or disagreeing
- "insufficient_info": the context doesn't contain enough information to judge

Base your answer only on the text given, not on outside knowledge of the topic.
"""


def classify_stance(claim_text: str, context_text: str, intents: list[str]) -> StanceVerdict:
    if _MOCK_MODE:
        return StanceVerdict(
            verdict=next(_mock_stance_cycle),
            confidence="low",
            reasoning="[MOCK] Canned verdict for pipeline testing — not a real classification.",
        )
    intent = ", ".join(intents) if intents else "none"
    prompt = _STANCE_PROMPT.format(claim_text=claim_text, context_text=context_text, intent=intent)
    return _structured_call(prompt, StanceVerdict)


# ---------------------------------------------------------------------------
# Stage 5B — classify resolution (track B: gap resolution)
# ---------------------------------------------------------------------------

_RESOLUTION_PROMPT = """You are checking whether a later paper's citation of an earlier paper claims to
address a specific limitation the earlier paper acknowledged.

Limitation acknowledged by the earlier paper: "{limitation_text}"

Citation context from the later paper (the sentence(s) where it cites the earlier paper):
"{context_text}"

Semantic Scholar's automatic citation-intent tag for this citation: {intent}

Classify the relationship as exactly one of:
- "addresses": the later paper explicitly claims to solve or overcome this specific limitation
- "partially_addresses": the later paper claims to improve on this limitation without fully resolving it
- "does_not_address": the context gives no indication this limitation is being tackled

Important: you are detecting what the later paper CLAIMS about its own contribution, not verifying
that it actually succeeded — those are different things and the report language downstream depends
on you keeping that distinction. Base your answer only on the text given.
"""


def classify_resolution(limitation_text: str, context_text: str, intents: list[str]) -> ResolutionVerdict:
    if _MOCK_MODE:
        return ResolutionVerdict(
            verdict=next(_mock_resolution_cycle),
            confidence="low",
            reasoning="[MOCK] Canned verdict for pipeline testing — not a real classification.",
        )
    intent = ", ".join(intents) if intents else "none"
    prompt = _RESOLUTION_PROMPT.format(
        limitation_text=limitation_text, context_text=context_text, intent=intent
    )
    return _structured_call(prompt, ResolutionVerdict)
