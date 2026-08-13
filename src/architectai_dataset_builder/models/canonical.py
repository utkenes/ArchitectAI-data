"""
Canonical Schema Data Models for ArchitectAI Architecture Training Corpus
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field
from architectai_dataset_builder.models.evidence import EvidenceItem, EvidenceType


class TaskType(str, Enum):
    ADR_REASONING = "adr_reasoning"
    TRADEOFF_ANALYSIS = "tradeoff_analysis"
    QUALITY_ATTRIBUTE_REASONING = "quality_attribute_reasoning"
    ARCHITECTURE_GENERATION = "architecture_generation"
    ARCHITECTURE_EXPLANATION = "architecture_explanation"
    TECHNOLOGY_SELECTION = "technology_selection"
    SCALING_REASONING = "scaling_reasoning"
    ARCHITECTURE_REVIEW = "architecture_review"
    ARCHITECTURE_RECOMMENDATION = "architecture_recommendation"
    ANTI_PATTERN_DETECTION = "anti_pattern_detection"


class ReviewStatus(str, Enum):
    UNREVIEWED = "unreviewed"
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    FLAGGED = "flagged"


class ReviewInfo(BaseModel):
    status: ReviewStatus = ReviewStatus.UNREVIEWED
    reviewer: Optional[str] = None
    reviewed_at: Optional[str] = None
    notes: Optional[str] = None


class SourceMetadata(BaseModel):
    source_id: str
    source_name: str
    source_url: Optional[str] = None
    source_version: Optional[str] = None
    source_commit_sha: Optional[str] = None
    source_file_path: str
    source_record_id: str
    project_id: Optional[str] = None
    group_id: Optional[str] = None
    provenance_type: str = "real_world"
    license_id: str
    license_verified: bool = False
    raw_sha256: str
    normalized_sha256: str
    split: Optional[str] = None
    kep_status: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class Alternative(BaseModel):
    option: str
    pros: list[str] = Field(default_factory=list)
    cons: list[str] = Field(default_factory=list)
    verdict: Optional[str] = None


class RecommendedArchitecture(BaseModel):
    summary: str
    components: list[str] = Field(default_factory=list)
    diagram_plantuml: Optional[str] = None
    quality_attributes: list[str] = Field(default_factory=list)


class ArchitectAISample(BaseModel):
    id: str
    source: SourceMetadata
    scenario: str
    task_type: TaskType = TaskType.ADR_REASONING
    facts: list[EvidenceItem] = Field(default_factory=list)
    architecture_drivers: list[EvidenceItem] = Field(default_factory=list)
    recommended_architecture: Optional[RecommendedArchitecture] = None
    alternatives: list[Alternative] = Field(default_factory=list)
    decisions: list[EvidenceItem] = Field(default_factory=list)
    tradeoffs: list[EvidenceItem] = Field(default_factory=list)
    final_answer: Optional[str] = None
    review: ReviewInfo = Field(default_factory=ReviewInfo)
