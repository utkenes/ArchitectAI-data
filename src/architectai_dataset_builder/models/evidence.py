"""
Evidence & Provenance Tracking Models
"""

from enum import Enum

from pydantic import BaseModel, Field


class EvidenceType(str, Enum):
    EXPLICIT = "explicit"
    DERIVED = "derived"
    INFERRED = "inferred"
    UNKNOWN = "unknown"


class EvidenceItem(BaseModel):
    value: str = Field(..., description="The textual value or claim")
    evidence_type: EvidenceType = Field(
        default=EvidenceType.UNKNOWN, description="Source grounding type"
    )
    source_span: str | None = Field(
        default=None, description="Exact substring span from raw source"
    )
    confidence: float | None = Field(
        default=None, description="Confidence score if derived/inferred"
    )
    inference_rule: str | None = Field(
        default=None, description="Identifier of normalizer rule applied"
    )
