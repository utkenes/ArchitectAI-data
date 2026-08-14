"""
Grounded SFT Multi-Turn Conversation Formatter with Composite Group Provenance V2
"""

from typing import Any

from architectai_dataset_builder.models.canonical import ArchitectAISample, TaskType
from architectai_dataset_builder.utils.markdown import (
    has_template_placeholders,
    is_lifecycle_status_only,
    sanitize_markdown,
)


def _trim_text(text: str, max_chars: int = 800) -> str:
    """Trim long text at sentence/paragraph boundaries up to max_chars."""
    clean = text.strip()
    if len(clean) <= max_chars:
        return clean
    sub = clean[:max_chars]
    last_break = max(sub.rfind("."), sub.rfind("\n"), sub.rfind(" "))
    if last_break > max_chars // 2:
        return sub[:last_break].strip() + "..."
    return sub.strip() + "..."


class SFTFormatter:
    """Formats canonical ArchitectAISample into Task-Aligned Instruction-Response SFT conversations."""

    SYSTEM_PROMPT = (
        "You are ArchitectAI, an expert software architecture AI assistant. "
        "Provide evidence-grounded architectural recommendations, tradeoff analyses, and design decisions."
    )

    def compose_grounded_response(self, sample: ArchitectAISample) -> str:
        """Compose structured, grounded assistant reasoning answer based on task type."""
        task = sample.task_type

        # Extract decision string
        dec_val = ""
        if sample.decisions:
            valid_decs = [d.value for d in sample.decisions if "not explicitly stated" not in d.value.lower()]
            if valid_decs:
                dec_val = "\n".join([f"- {d}" for d in valid_decs])
        elif sample.recommended_architecture and sample.recommended_architecture.summary:
            if "not explicitly stated" not in sample.recommended_architecture.summary.lower():
                dec_val = sample.recommended_architecture.summary
        elif sample.final_answer and "not explicitly stated" not in sample.final_answer.lower():
            dec_val = sample.final_answer

        dec_clean = _trim_text(dec_val, max_chars=900)

        alternatives = [a.option for a in sample.alternatives if a.option and a.option.strip()]
        tradeoffs = [t.value for t in sample.tradeoffs if t.value and t.value.strip()]
        drivers = [d.value for d in sample.architecture_drivers if d.value and d.value.strip()]
        facts = [f.value for f in sample.facts if f.value and f.value.strip()]

        response = ""

        # 1. ADR reasoning
        if task == TaskType.ADR_REASONING:
            sections = []
            if dec_clean:
                sections.append(f"Decision:\n{dec_clean}")
            if drivers:
                sections.append("Rationale:\n" + "\n".join([f"- {d}" for d in drivers[:3]]))
            elif facts:
                sections.append("Rationale:\n" + "\n".join([f"- {f}" for f in facts[:3]]))
            if tradeoffs:
                pos_t = [t for t in tradeoffs if "Advantage" in t or "pro" in t.lower()]
                neg_t = [t for t in tradeoffs if "Disadvantage" in t or "con" in t.lower() or "Risk" in t]
                if pos_t:
                    sections.append("Key Trade-offs:\n" + "\n".join([f"- {t}" for t in pos_t[:3]]))
                if neg_t:
                    sections.append("Risks / Limitations:\n" + "\n".join([f"- {t}" for t in neg_t[:3]]))
                elif tradeoffs and not pos_t:
                    sections.append("Key Trade-offs:\n" + "\n".join([f"- {t}" for t in tradeoffs[:3]]))

            response = "\n\n".join(sections)

        # 2. Tradeoff analysis
        elif task == TaskType.TRADEOFF_ANALYSIS:
            sections = []
            if dec_clean:
                sections.append(f"Decision / Preferred Direction:\n{dec_clean}")
            if alternatives:
                sections.append("Alternatives Considered:\n" + "\n".join([f"- {a}" for a in alternatives[:4]]))
            if drivers or facts:
                sections.append("Why:\n" + "\n".join([f"- {r}" for r in (drivers or facts)[:3]]))
            if tradeoffs:
                sections.append("Trade-offs:\n" + "\n".join([f"- {t}" for t in tradeoffs[:4]]))

            response = "\n\n".join(sections)

        # 3. Technology selection
        elif task == TaskType.TECHNOLOGY_SELECTION:
            sections = []
            if dec_clean:
                sections.append(f"Selected Approach:\n{dec_clean}")
            if alternatives:
                sections.append("Compared Against:\n" + "\n".join([f"- {a}" for a in alternatives[:4]]))
            if drivers or facts:
                sections.append("Selection Rationale:\n" + "\n".join([f"- {r}" for r in (drivers or facts)[:3]]))
            if tradeoffs:
                sections.append("Trade-offs:\n" + "\n".join([f"- {t}" for t in tradeoffs[:3]]))

            response = "\n\n".join(sections)

        # 4. Scaling reasoning
        elif task == TaskType.SCALING_REASONING:
            sections = []
            if drivers:
                sections.append("Scaling Driver:\n" + "\n".join([f"- {d}" for d in drivers[:3]]))
            else:
                sections.append("Scaling Driver:\nWorkload growth, high throughput, or capacity requirement.")
            if dec_clean:
                sections.append(f"Architectural Response:\n{dec_clean}")
            if facts:
                sections.append("Why:\n" + "\n".join([f"- {f}" for f in facts[:3]]))
            if tradeoffs:
                sections.append("Trade-offs / Risks:\n" + "\n".join([f"- {t}" for t in tradeoffs[:3]]))

            response = "\n\n".join(sections)

        # 5. Quality attribute reasoning
        elif task == TaskType.QUALITY_ATTRIBUTE_REASONING:
            sections = []
            if drivers or facts:
                sections.append("Quality Attribute:\n" + "\n".join([f"- {q}" for q in (drivers or facts)[:3]]))
            if dec_clean:
                sections.append(f"Architectural Decision:\n{dec_clean}")
            if tradeoffs:
                sections.append("Trade-offs:\n" + "\n".join([f"- {t}" for t in tradeoffs[:3]]))

            response = "\n\n".join(sections)

        # 6. Architecture Generation (PlantUML)
        elif (
            task == TaskType.ARCHITECTURE_GENERATION
            and sample.recommended_architecture
            and sample.recommended_architecture.diagram_plantuml
        ):
            response = f"Architecture Design (PlantUML):\n```plantuml\n{sample.recommended_architecture.diagram_plantuml.strip()}\n```"

        # Default fallback
        else:
            sections = []
            if dec_clean:
                sections.append(f"Decision:\n{dec_clean}")
            if tradeoffs:
                sections.append("Trade-offs:\n" + "\n".join([f"- {t}" for t in tradeoffs[:3]]))
            response = "\n\n".join(sections) if sections else dec_clean

        return sanitize_markdown(response)

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

        assistant_response = self.compose_grounded_response(sample)
        response_clean = assistant_response.strip()

        # Reject empty, Not explicitly stated, status/lifecycle metadata only, or unresolved template placeholders
        if (
            not response_clean
            or "not explicitly stated" in response_clean.lower()
            or is_lifecycle_status_only(response_clean)
            or has_template_placeholders(response_clean)
            or "<!--" in response_clean
        ):
            return None

        group_id = (
            sample.source.group_id
            or f"group_{sample.source.source_id}_{sample.source.project_id or 'default'}_{sample.source.source_record_id}"
        )

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
