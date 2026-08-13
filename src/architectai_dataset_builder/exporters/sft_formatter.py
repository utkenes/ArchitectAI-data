"""
Grounded SFT Multi-Turn Conversation Formatter with Composite Group Provenance
"""

from typing import Any

from architectai_dataset_builder.models.canonical import ArchitectAISample, TaskType


class SFTFormatter:
    """Formats canonical ArchitectAISample into Instruction-Response SFT conversations."""

    SYSTEM_PROMPT = (
        "You are ArchitectAI, an expert software architecture AI assistant. "
        "Provide evidence-grounded architectural recommendations, tradeoff analyses, and design decisions."
    )

    def format_sample(self, sample: ArchitectAISample) -> dict[str, Any] | None:
        user_prompt = f"### Architectural Scenario:\n{sample.scenario}\n\n"

        if sample.facts:
            user_prompt += "### Key Facts & Context:\n"
            for f in sample.facts:
                user_prompt += f"- {f.value}\n"
            user_prompt += "\n"

        if sample.architecture_drivers:
            user_prompt += "### Architectural Drivers:\n"
            for d in sample.architecture_drivers:
                user_prompt += f"- {d.value}\n"
            user_prompt += "\n"

        if sample.alternatives:
            user_prompt += "### Considered Alternatives:\n"
            for alt in sample.alternatives:
                user_prompt += f"- {alt.option}\n"
            user_prompt += "\n"

        if sample.task_type == TaskType.ADR_REASONING:
            user_prompt += "### Instruction:\nWhat architectural decision was made based on the scenario?"
        else:
            user_prompt += f"### Instruction:\nAnalyze the scenario and provide the architectural design recommendation for task: {sample.task_type.value}."

        assistant_response = ""
        if sample.final_answer and "not explicitly stated" not in sample.final_answer.lower():
            assistant_response = sample.final_answer
        else:
            valid_decisions = [d for d in sample.decisions if "not explicitly stated" not in d.value.lower()]
            if valid_decisions:
                assistant_response += "### Decision:\n" + "\n".join([f"- {d.value}" for d in valid_decisions]) + "\n\n"

            if sample.tradeoffs:
                assistant_response += "### Trade-off & Consequence Analysis:\n" + "\n".join([f"- {t.value}" for t in sample.tradeoffs]) + "\n\n"

            if sample.recommended_architecture and "not explicitly stated" not in sample.recommended_architecture.summary.lower():
                assistant_response += f"### Recommended Architecture:\n{sample.recommended_architecture.summary}\n"

        response_clean = assistant_response.strip()

        # Reject empty or Not explicitly stated responses
        if not response_clean or "not explicitly stated" in response_clean.lower():
            return None

        group_id = sample.source.group_id or f"group_{sample.source.source_id}_{sample.source.project_id or 'default'}_{sample.source.source_record_id}"

        return {
            "id": sample.id,
            "sample_id": sample.id,
            "group_id": group_id,
            "task_type": sample.task_type.value,
            "source_id": sample.source.source_id,
            "source_record_id": sample.source.source_record_id,
            "project_id": sample.source.project_id,
            "kep_status": sample.source.kep_status,
            "split": sample.source.split,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": response_clean},
            ],
            "provenance": {
                "license_id": sample.source.license_id,
                "raw_sha256": sample.source.raw_sha256,
                "normalized_sha256": sample.source.normalized_sha256,
            },
        }
