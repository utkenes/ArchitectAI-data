"""
Canonical ArchitectAI Training Sample Models
"""

from enum import Enum
from typing import List, Optional
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
    reviewer: Optional[str] = None
    notes: List[str] = Field(default_factory=list)
    quality_score: Optional[float] = None


class SourceMetadata(BaseModel):
    source_id: str
    source_name: str
    source_url: Optional[str] = None
    source_version: Optional[str] = None
    source_commit_sha: Optional[str] = None
    source_file_path: str
    source_record_id: str
    project_id: Optional[str] = None
    provenance_type: str = "real_world"
    license_id: str
    license_verified: bool = False
    raw_sha256: str
    normalized_sha256: str
    parser_version: str = "0.1.0"
    normalizer_version: str = "0.1.0"
    split: Optional[str] = None
    split_reason: Optional[str] = None
    created_at: str


class Alternative(BaseModel):
    option: str
    advantages: List[EvidenceItem] = Field(default_factory=list)
    disadvantages: List[EvidenceItem] = Field(default_factory=list)
    why_rejected: Optional[EvidenceItem] = None


class RecommendedArchitecture(BaseModel):
    summary: str
    style: Optional[str] = None
    components: List[str] = Field(default_factory=list)


class FailureMode(BaseModel):
    failure: str
    impact: str
    mitigation: str
    recovery: Optional[str] = None


class ArchitectAISample(BaseModel):
    id: str = Field(..., description="Stable deterministic sample ID: arch_<sha256_prefix>")
    source: SourceMetadata
    scenario: str
    task_type: TaskType = TaskType.ADR_REASONING
    
    facts: List[EvidenceItem] = Field(default_factory=list)
    assumptions: List[EvidenceItem] = Field(default_factory=list)
    
    architecture_drivers: List[EvidenceItem] = Field(default_factory=list)
    constraints: List[EvidenceItem] = Field(default_factory=list)
    quality_attributes: List[EvidenceItem] = Field(default_factory=list)
    invariants: List[EvidenceItem] = Field(default_factory=list)
    
    recommended_architecture: Optional[RecommendedArchitecture] = None
    alternatives: List[Alternative] = Field(default_factory=list)
    
    decisions: List[EvidenceItem] = Field(default_factory=list)
    tradeoffs: List[EvidenceItem] = Field(default_factory=list)
    failure_modes: List[FailureMode] = Field(default_factory=list)
    
    consistency_semantics: List[EvidenceItem] = Field(default_factory=list)
    data_ownership: List[EvidenceItem] = Field(default_factory=list)
    source_of_truth: List[EvidenceItem] = Field(default_factory=list)
    
    security_considerations: List[EvidenceItem] = Field(default_factory=list)
    multi_tenancy: List[EvidenceItem] = Field(default_factory=list)
    scaling_considerations: List[EvidenceItem] = Field(default_factory=list)
    operational_considerations: List[EvidenceItem] = Field(default_factory=list)
    
    metrics: List[EvidenceItem] = Field(default_factory=list)
    evolution_triggers: List[EvidenceItem] = Field(default_factory=list)
    
    recommended_now: List[EvidenceItem] = Field(default_factory=list)
    add_when_needed: List[EvidenceItem] = Field(default_factory=list)
    avoid_for_now: List[EvidenceItem] = Field(default_factory=list)
    
    confidence: Optional[float] = None
    final_answer: Optional[str] = None
    
    review: ReviewInfo = Field(default_factory=ReviewInfo)
