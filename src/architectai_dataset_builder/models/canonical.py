"""
Canonical Schema Data Models for ArchitectAI Architecture Training Corpus
"""

from datetime import UTC, datetime
from enum import Enum

from pydantic import BaseModel, Field

from architectai_dataset_builder.models.evidence import EvidenceItem


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
    reviewer: str | None = None
    reviewed_at: str | None = None
    notes: str | None = None


class SourceMetadata(BaseModel):
    source_id: str
    source_name: str
    source_url: str | None = None
    source_version: str | None = None
    source_commit_sha: str | None = None
    source_file_path: str
    source_record_id: str
    project_id: str | None = None
    group_id: str | None = None
    provenance_type: str = "real_world"
    license_id: str
    license_verified: bool = False
    raw_sha256: str
    normalized_sha256: str
    split: str | None = None
    kep_status: str | None = None
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class Alternative(BaseModel):
    option: str
    pros: list[str] = Field(default_factory=list)
    cons: list[str] = Field(default_factory=list)
    verdict: str | None = None


class RecommendedArchitecture(BaseModel):
    summary: str
    components: list[str] = Field(default_factory=list)
    diagram_plantuml: str | None = None
    quality_attributes: list[str] = Field(default_factory=list)


class ArchitectAISample(BaseModel):
    id: str
    source: SourceMetadata
    scenario: str
    task_type: TaskType = TaskType.ADR_REASONING
    facts: list[EvidenceItem] = Field(default_factory=list)
    architecture_drivers: list[EvidenceItem] = Field(default_factory=list)
    recommended_architecture: RecommendedArchitecture | None = None
    alternatives: list[Alternative] = Field(default_factory=list)
    decisions: list[EvidenceItem] = Field(default_factory=list)
    tradeoffs: list[EvidenceItem] = Field(default_factory=list)
    final_answer: str | None = None
    review: ReviewInfo = Field(default_factory=ReviewInfo)
