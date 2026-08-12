"""
Canonical ArchitectAI Training Sample Models
"""

from enum import Enum

from pydantic import BaseModel, Field

from architectai_dataset_builder.models.evidence import EvidenceItem


class TaskType(str, Enum):
    ARCHITECTURE_RECOMMENDATION = "architecture_recommendation"
    ARCHITECTURE_EXPLANATION = "architecture_explanation"
    ADR_REASONING = "adr_reasoning"
    TRADEOFF_ANALYSIS = "tradeoff_analysis"
    QUALITY_ATTRIBUTE_REASONING = "quality_attribute_reasoning"
    ARCHITECTURE_GENERATION = "architecture_generation"
    ARCHITECTURE_REVIEW = "architecture_review"
    ANTI_PATTERN_DETECTION = "anti_pattern_detection"
    TECHNOLOGY_SELECTION = "technology_selection"
    SCALING_REASONING = "scaling_reasoning"


class ReviewStatus(str, Enum):
    UNREVIEWED = "unreviewed"
    SILVER = "silver"
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_REVISION = "needs_revision"


class ReviewInfo(BaseModel):
    status: ReviewStatus = ReviewStatus.UNREVIEWED
    reviewer: str | None = None
    notes: list[str] = Field(default_factory=list)
    quality_score: float | None = None


class SourceMetadata(BaseModel):
    source_id: str
    source_name: str
    source_url: str | None = None
    source_version: str | None = None
    source_commit_sha: str | None = None
    source_file_path: str
    source_record_id: str
    project_id: str | None = None
    provenance_type: str = "real_world"
    license_id: str
    license_verified: bool = False
    raw_sha256: str
    normalized_sha256: str
    parser_version: str = "0.1.0"
    normalizer_version: str = "0.1.0"
    split: str | None = None
    split_reason: str | None = None
    created_at: str


class Alternative(BaseModel):
    option: str
    advantages: list[EvidenceItem] = Field(default_factory=list)
    disadvantages: list[EvidenceItem] = Field(default_factory=list)
    why_rejected: EvidenceItem | None = None


class RecommendedArchitecture(BaseModel):
    summary: str
    style: str | None = None
    components: list[str] = Field(default_factory=list)


class FailureMode(BaseModel):
    failure: str
    impact: str
    mitigation: str
    recovery: str | None = None


class ArchitectAISample(BaseModel):
    id: str = Field(..., description="Stable deterministic sample ID: arch_<sha256_prefix>")
    source: SourceMetadata
    scenario: str
    task_type: TaskType = TaskType.ADR_REASONING
    
    facts: list[EvidenceItem] = Field(default_factory=list)
    assumptions: list[EvidenceItem] = Field(default_factory=list)
    
    architecture_drivers: list[EvidenceItem] = Field(default_factory=list)
    constraints: list[EvidenceItem] = Field(default_factory=list)
    quality_attributes: list[EvidenceItem] = Field(default_factory=list)
    invariants: list[EvidenceItem] = Field(default_factory=list)
    
    recommended_architecture: RecommendedArchitecture | None = None
    alternatives: list[Alternative] = Field(default_factory=list)
    
    decisions: list[EvidenceItem] = Field(default_factory=list)
    tradeoffs: list[EvidenceItem] = Field(default_factory=list)
    failure_modes: list[FailureMode] = Field(default_factory=list)
    
    consistency_semantics: list[EvidenceItem] = Field(default_factory=list)
    data_ownership: list[EvidenceItem] = Field(default_factory=list)
    source_of_truth: list[EvidenceItem] = Field(default_factory=list)
    
    security_considerations: list[EvidenceItem] = Field(default_factory=list)
    multi_tenancy: list[EvidenceItem] = Field(default_factory=list)
    scaling_considerations: list[EvidenceItem] = Field(default_factory=list)
    operational_considerations: list[EvidenceItem] = Field(default_factory=list)
    
    metrics: list[EvidenceItem] = Field(default_factory=list)
    evolution_triggers: list[EvidenceItem] = Field(default_factory=list)
    
    recommended_now: list[EvidenceItem] = Field(default_factory=list)
    add_when_needed: list[EvidenceItem] = Field(default_factory=list)
    avoid_for_now: list[EvidenceItem] = Field(default_factory=list)
    
    confidence: float | None = None
    final_answer: str | None = None
    
    review: ReviewInfo = Field(default_factory=ReviewInfo)
