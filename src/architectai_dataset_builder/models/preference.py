"""
Preference and Anti-Pattern Models (Reserved for Roadmap V1.1 / V2)
"""

from typing import Any

from pydantic import BaseModel, Field


class PreferenceSample(BaseModel):
    id: str
    scenario_id: str
    prompt: str
    chosen: str
    rejected: str
    rejection_reasons: list[str] = Field(default_factory=list)
    generator_metadata: dict[str, Any] | None = None


class NegativeSample(BaseModel):
    id: str
    scenario_id: str
    anti_pattern_category: str
    description: str
    flawed_solution: str
    why_flawed: str
    corrected_solution: str | None = None
