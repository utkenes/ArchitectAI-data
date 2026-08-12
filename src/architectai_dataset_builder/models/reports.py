"""
Dataset Quality, Contamination, and Statistics Report Models
"""

from typing import List, Dict, Any, Optional
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
    leakage_details: List[LeakageDetail] = Field(default_factory=list)


class DatasetStatsReport(BaseModel):
    total_samples: int
    sample_count_by_split: Dict[str, int]
    sample_count_by_source: Dict[str, int]
    sample_count_by_task_type: Dict[str, int]
    sample_count_by_quality_class: Dict[str, int]
    sample_count_by_license: Dict[str, int]
    duplicate_count: int
    near_duplicate_count: int
    quarantine_count: int
    review_status_distribution: Dict[str, int]
