"""
Evaluation-Specific Schemas Isolated from Canonical Training Schema
"""

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
    options: dict[str, str]
    correct_answer: str
    explanation: str | None = None
    taxonomy_category: str | None = None


class FreeResponseEvalSample(BaseModel):
    id: str
    source: EvalSourceMetadata
    prompt: str
    reference_answer: str
    grading_rubric: list[str] = Field(default_factory=list)
    context: str | None = None


class ArchitectureGenerationEvalSample(BaseModel):
    id: str
    source: EvalSourceMetadata
    requirements: str
    constraints: list[str] = Field(default_factory=list)
    expected_components: list[str] = Field(default_factory=list)
    reference_solution: str


class DiagramEvalSample(BaseModel):
    id: str
    source: EvalSourceMetadata
    project_id: str
    requirements_text: str
    reference_plantuml: str
    diagram_type: str = "component"
