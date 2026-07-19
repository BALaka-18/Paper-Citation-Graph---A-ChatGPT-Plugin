"""
Pydantic schemas used as structured-output contracts for Gemini calls, and as the
shared data shapes passed between pipeline stages.

Keeping these in one place means the extraction prompts (llm_client.py) and the
aggregation/report code (aggregate.py, report.py) can't silently drift apart on
field names.
"""
from typing import Literal

from pydantic import BaseModel, Field


class Claim(BaseModel):
    id: str = Field(description="Short id, e.g. c1, c2")
    text: str = Field(description="The atomic, falsifiable claim")
    source: str = Field(description="Where in the paper this came from, e.g. 'abstract'")


class ClaimList(BaseModel):
    claims: list[Claim]


class Limitation(BaseModel):
    id: str = Field(description="Short id, e.g. l1, l2")
    text: str = Field(description="The self-acknowledged limitation or open problem")
    source: str = Field(description="Where in the paper this came from, e.g. 'abstract'")


class LimitationList(BaseModel):
    limitations: list[Limitation]


class StanceVerdict(BaseModel):
    verdict: Literal["supports", "disputes", "extends", "insufficient_info"]
    confidence: Literal["low", "medium", "high"]
    reasoning: str = Field(description="One or two sentences grounding the verdict in the context text")


class ResolutionVerdict(BaseModel):
    verdict: Literal["addresses", "partially_addresses", "does_not_address"]
    confidence: Literal["low", "medium", "high"]
    reasoning: str = Field(description="One or two sentences grounding the verdict in the context text")
