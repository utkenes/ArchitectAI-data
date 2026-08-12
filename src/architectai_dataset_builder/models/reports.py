"""
Dataset Quality, Contamination, and Statistics Report Models
"""

from pydantic import BaseModel, Field


class LeakageDetail(BaseModel):
    training_sample_id: str
    evaluation_sample_id: str
    match_type: str
    similarity_score: float
    source_a: str
    source_b: str


class ContaminationReport(BaseModel):
    has_leakage: bool
    total_leaks_detected: int
    leakage_details: list[LeakageDetail] = Field(default_factory=list)


class DatasetStatsReport(BaseModel):
    total_samples: int
    sample_count_by_split: dict[str, int]
    sample_count_by_source: dict[str, int]
    sample_count_by_task_type: dict[str, int]
    sample_count_by_quality_class: dict[str, int]
    sample_count_by_license: dict[str, int]
    duplicate_count: int
    near_duplicate_count: int
    quarantine_count: int
    review_status_distribution: dict[str, int]
