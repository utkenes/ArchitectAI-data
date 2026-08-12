"""
Evidence-Grounded SFT Formatter (System / User / Assistant Message Converter)
"""

from typing import Dict, Any, List
from architectai_dataset_builder.models.canonical import ArchitectAISample, TaskType


class SFTFormatter:
    def __init__(self, system_prompt: str | None = None):
        self.system_prompt = system_prompt or (
            "You are ArchitectAI, a software architecture decision engine. "
            "Analyze project requirements, constraints, and trade-offs to provide grounded architectural decisions."
        )

    def format_sample(self, sample: ArchitectAISample) -> Dict[str, Any]:
        """Formats a canonical ArchitectAISample into OpenAI/HuggingFace SFT format."""
        
        # Grounded user prompt construction based on evidence-backed task_type
        if sample.task_type == TaskType.ADR_REASONING:
            user_text = f"Given this project context:\n\n{sample.scenario}\n\nWhat architectural decision was made and why?"
        elif sample.task_type == TaskType.TRADEOFF_ANALYSIS:
            user_text = f"Analyze the trade-offs and options for this architectural scenario:\n\n{sample.scenario}"
        elif sample.task_type == TaskType.ARCHITECTURE_GENERATION:
            user_text = f"Design the system architecture for the following requirements:\n\n{sample.scenario}"
        else:
            user_text = f"Architectural Scenario:\n\n{sample.scenario}\n\nProvide architectural analysis and recommendations."

        assistant_text = sample.final_answer or "Architectural rationale recorded in source metadata."

        return {
            "sample_id": sample.id,
            "task_type": sample.task_type.value,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": assistant_text},
            ],
        }
