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
        reasons: list[str] = []
        quarantine_category: str | None = None

        # 1. Identity Valid
        identity_valid = bool(sample.id and sample.source.source_record_id)
        if not identity_valid:
            reasons.append("Sample missing valid sample ID or source record ID")
            quarantine_category = "identity_collision"

        # 2. Template Clean
        full_text = f"{sample.scenario} {sample.final_answer or ''}"
        template_clean = not ("<!--" in full_text or "-->" in full_text or has_template_placeholders(full_text))
        if not template_clean:
            reasons.append("HTML comments or template placeholders detected in sample text")
            quarantine_category = quarantine_category or "template_leakage"

        # 3. Context Present
        context_present = bool(sample.scenario and len(sample.scenario.strip()) >= 20)
        if not context_present:
            reasons.append("Insufficient or missing scenario/context description")
            quarantine_category = quarantine_category or "insufficient_evidence"

        # 4. Decision Present
        has_dec = bool(sample.decisions) or (
            sample.recommended_architecture is not None
            and bool(sample.recommended_architecture.summary)
        )
        decision_present = has_dec or (sample.task_type not in [
            TaskType.ADR_REASONING,
            TaskType.TECHNOLOGY_SELECTION,
            TaskType.SCALING_REASONING,
            TaskType.QUALITY_ATTRIBUTE_REASONING,
        ])
        if not decision_present:
            reasons.append(f"Task '{sample.task_type.value}' requires explicit decision outcome")
            quarantine_category = quarantine_category or "missing_decision"

        # 5. Alternatives / Tradeoffs Valid
        if sample.task_type in [TaskType.TRADEOFF_ANALYSIS, TaskType.TECHNOLOGY_SELECTION]:
            alternatives_valid = len(sample.alternatives) >= 1 or len(sample.tradeoffs) >= 1
            if not alternatives_valid:
                reasons.append(f"Task '{sample.task_type.value}' requires explicit alternatives or tradeoffs")
                quarantine_category = quarantine_category or "invalid_alternatives"
        else:
            alternatives_valid = True

        # 6. Task Semantics & Evidence Contract Validation
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
            "context": sample.scenario,
            "positive_consequences": [t.value for t in sample.tradeoffs if "Advantage" in t.value or "pro" in t.value.lower()],
            "negative_consequences": [t.value for t in sample.tradeoffs if "Disadvantage" in t.value or "con" in t.value.lower()],
        }

        classified_res = self.taxonomy_classifier.classify_with_evidence(record_dict)

        task_semantics_valid = True
        evidence_sufficient = True

        # Task-specific contract validation helpers
        if sample.task_type == TaskType.SCALING_REASONING:
            if classified_res.task_type != TaskType.SCALING_REASONING:
                task_semantics_valid = False
                evidence_sufficient = False
                reasons.append("Sample classified as scaling reasoning lacks scaling driver / response evidence contract")
                quarantine_category = quarantine_category or "task_semantics_mismatch"

        elif sample.task_type == TaskType.TECHNOLOGY_SELECTION:
            if classified_res.task_type != TaskType.TECHNOLOGY_SELECTION:
                task_semantics_valid = False
                evidence_sufficient = False
                reasons.append("Sample classified as technology selection lacks comparative tech evaluation contract")
                quarantine_category = quarantine_category or "task_semantics_mismatch"

        elif sample.task_type == TaskType.QUALITY_ATTRIBUTE_REASONING:
            if classified_res.task_type != TaskType.QUALITY_ATTRIBUTE_REASONING:
                task_semantics_valid = False
                evidence_sufficient = False
                reasons.append("Sample classified as quality attribute reasoning lacks quality attribute / NFR contract")
                quarantine_category = quarantine_category or "task_semantics_mismatch"

        elif sample.task_type == TaskType.TRADEOFF_ANALYSIS:
            if classified_res.task_type != TaskType.TRADEOFF_ANALYSIS:
                task_semantics_valid = False
                evidence_sufficient = False
                reasons.append("Sample classified as tradeoff analysis lacks explicit tradeoff evidence contract")
                quarantine_category = quarantine_category or "task_semantics_mismatch"

        elif sample.task_type == TaskType.ADR_REASONING and not (context_present and decision_present):
            task_semantics_valid = False
            evidence_sufficient = False
            reasons.append("ADR reasoning sample lacks required context or decision contract")
            quarantine_category = quarantine_category or "insufficient_evidence"

        # 7. Answer Grounded & Task Aligned (if SFT formatted)
        answer_grounded = True
        answer_task_aligned = True
        output_structurally_valid = True

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
                answer_grounded = False
                answer_task_aligned = False
                reasons.append("Assistant answer is empty, ungrounded, or lifecycle metadata only")
                quarantine_category = quarantine_category or "answer_not_task_aligned"

            if (
                "<!--" in assistant_content
                or "-->" in assistant_content
                or has_template_placeholders(assistant_content)
            ):
                output_structurally_valid = False
                template_clean = False
                reasons.append("Assistant response contains raw HTML comments or template placeholders")
                quarantine_category = quarantine_category or "template_leakage"

        checks = {
            "identity_valid": identity_valid,
            "template_clean": template_clean,
            "task_semantics_valid": task_semantics_valid,
            "context_present": context_present,
            "decision_present": decision_present,
            "answer_grounded": answer_grounded,
            "answer_task_aligned": answer_task_aligned,
            "alternatives_valid": alternatives_valid,
            "evidence_sufficient": evidence_sufficient,
            "output_structurally_valid": output_structurally_valid,
        }

        passed = all(checks.values())
        return SemanticQualityResult(
            passed=passed,
            checks=checks,
            reasons=reasons,
            quarantine_category=quarantine_category,
        )
