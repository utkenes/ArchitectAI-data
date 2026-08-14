"""
Semantic Training Quality Validator for ArchitectAI Model Fine-Tuning Gate
"""

from dataclasses import dataclass, field
from typing import Any

from architectai_dataset_builder.models.canonical import ArchitectAISample, TaskType
from architectai_dataset_builder.normalizers.task_taxonomy import TaskTaxonomyClassifier
from architectai_dataset_builder.utils.markdown import (
    has_template_placeholders,
    is_lifecycle_status_only,
)


@dataclass
class SemanticQualityResult:
    passed: bool
    checks: dict[str, bool]
    reasons: list[str] = field(default_factory=list)
    quarantine_category: str | None = None


class SemanticQualityValidator:
    """
    Evaluates whether a sample satisfies semantic quality requirements for SFT training.
    Failed samples are quarantined into explicit categories.
    """

    def __init__(self) -> None:
        self.taxonomy_classifier = TaskTaxonomyClassifier()

    def validate_sample(
        self, sample: ArchitectAISample, sft_formatted: dict[str, Any] | None = None
    ) -> SemanticQualityResult:
        checks = {
            "identity_valid": True,
            "template_clean": True,
            "task_semantics_valid": True,
            "context_present": True,
            "decision_present": True,
            "answer_grounded": True,
            "answer_task_aligned": True,
            "alternatives_valid": True,
            "evidence_sufficient": True,
            "output_structurally_valid": True,
        }
        reasons: list[str] = []
        quarantine_category: str | None = None

        # 1. Identity Valid
        if not sample.id or not sample.source.source_record_id:
            checks["identity_valid"] = False
            reasons.append("Sample missing valid sample ID or source record ID")
            quarantine_category = "identity_collision"

        # 2. Template Clean (No HTML comments or unresolved placeholders)
        full_text = f"{sample.scenario} {sample.final_answer or ''}"
        if "<!--" in full_text or "-->" in full_text:
            checks["template_clean"] = False
            reasons.append("HTML comments leaked into scenario or final answer")
            quarantine_category = "template_leakage"
        elif has_template_placeholders(full_text):
            checks["template_clean"] = False
            reasons.append("Unresolved template placeholders detected in sample text")
            quarantine_category = "template_leakage"

        # 3. Context Present
        if not sample.scenario or len(sample.scenario.strip()) < 20:
            checks["context_present"] = False
            reasons.append("Insufficient or missing scenario/context description")
            quarantine_category = quarantine_category or "insufficient_evidence"

        # 4. Decision Present (where required by task type)
        if sample.task_type in [
            TaskType.ADR_REASONING,
            TaskType.TECHNOLOGY_SELECTION,
            TaskType.SCALING_REASONING,
            TaskType.QUALITY_ATTRIBUTE_REASONING,
        ]:
            has_dec = bool(sample.decisions) or (
                sample.recommended_architecture is not None
                and bool(sample.recommended_architecture.summary)
            )
            if not has_dec:
                checks["decision_present"] = False
                reasons.append(f"Task '{sample.task_type.value}' requires explicit decision outcome")
                quarantine_category = quarantine_category or "missing_decision"

        # 5. Alternatives Valid (where required)
        if sample.task_type in [TaskType.TRADEOFF_ANALYSIS, TaskType.TECHNOLOGY_SELECTION]:
            if len(sample.alternatives) < 1 and len(sample.tradeoffs) < 1:
                checks["alternatives_valid"] = False
                reasons.append(f"Task '{sample.task_type.value}' requires explicit alternatives or tradeoffs")
                quarantine_category = quarantine_category or "invalid_alternatives"

        # 6. Task Semantics Valid
        record_dict = {
            "raw_text": full_text,
            "options": [a.option for a in sample.alternatives],
            "plantuml_text": (
                sample.recommended_architecture.diagram_plantuml
                if sample.recommended_architecture
                else None
            ),
            "summary": sample.scenario,
            "decision": sample.decisions[0].value if sample.decisions else None,
        }
        classified_res = self.taxonomy_classifier.classify_with_evidence(record_dict)
        if (
            sample.task_type == TaskType.SCALING_REASONING
            and classified_res.task_type != TaskType.SCALING_REASONING
        ):
            checks["task_semantics_valid"] = False
            reasons.append(
                "Sample classified as scaling reasoning lacks scaling driver / response evidence contract"
            )
            quarantine_category = quarantine_category or "task_semantics_mismatch"
        elif (
            sample.task_type == TaskType.TECHNOLOGY_SELECTION
            and classified_res.task_type != TaskType.TECHNOLOGY_SELECTION
        ):
            checks["task_semantics_valid"] = False
            reasons.append(
                "Sample classified as technology selection lacks comparative tech evaluation contract"
            )
            quarantine_category = quarantine_category or "task_semantics_mismatch"

        # 7. Answer Grounded & Task Aligned (if SFT formatted)
        if sft_formatted:
            messages = sft_formatted.get("messages", [])
            assistant_content = ""
            for m in messages:
                if m.get("role") == "assistant":
                    assistant_content = m.get("content", "")

            if (
                not assistant_content
                or "not explicitly stated" in assistant_content.lower()
                or is_lifecycle_status_only(assistant_content)
            ):
                checks["answer_grounded"] = False
                checks["answer_task_aligned"] = False
                reasons.append("Assistant answer is empty, ungrounded, or lifecycle metadata only")
                quarantine_category = quarantine_category or "answer_not_task_aligned"

            if (
                "<!--" in assistant_content
                or "-->" in assistant_content
                or has_template_placeholders(assistant_content)
            ):
                checks["output_structurally_valid"] = False
                checks["template_clean"] = False
                reasons.append(
                    "Assistant response contains raw HTML comments or template placeholders"
                )
                quarantine_category = quarantine_category or "template_leakage"

        passed = all(checks.values())
        return SemanticQualityResult(
            passed=passed,
            checks=checks,
            reasons=reasons,
            quarantine_category=quarantine_category,
        )
