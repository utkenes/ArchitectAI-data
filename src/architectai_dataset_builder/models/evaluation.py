"""
Evaluation-Specific Schemas Isolated from Canonical Training Schema
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


class EvalSourceMetadata(BaseModel):
    benchmark_id: str
    sample_id: str
    license_id: str
    raw_sha256: str
    normalized_sha256: str
    evaluation_only: bool = True
    split: str = "held_out_eval"


class MultipleChoiceEvalSample(BaseModel):
    id: str
    source: EvalSourceMetadata
    question: str
    options: Dict[str, str]
    correct_answer: str
    explanation: Optional[str] = None
    taxonomy_category: Optional[str] = None


class FreeResponseEvalSample(BaseModel):
    id: str
    source: EvalSourceMetadata
    prompt: str
    reference_answer: str
    grading_rubric: List[str] = Field(default_factory=list)
    context: Optional[str] = None


class ArchitectureGenerationEvalSample(BaseModel):
    id: str
    source: EvalSourceMetadata
    requirements: str
    constraints: List[str] = Field(default_factory=list)
    expected_components: List[str] = Field(default_factory=list)
    reference_solution: str


class DiagramEvalSample(BaseModel):
    id: str
    source: EvalSourceMetadata
    project_id: str
    requirements_text: str
    reference_plantuml: str
    diagram_type: str = "component"
