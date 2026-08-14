"""
Automated Training Readiness Reporter V2.2 for ArchitectAI Model Fine-Tuning Gate
"""

from pathlib import Path
from typing import Any

from architectai_dataset_builder.models.canonical import ArchitectAISample
from architectai_dataset_builder.utils.io import write_jsonl
from architectai_dataset_builder.validators.sft_export_validator import ExportValidationResult


class ReadinessReporter:
    """
    Computes measurable training readiness criteria based on actual dataset metrics, export validation,
    and task coverage.
    Statuses: BUILD_INVALID, BUILD_VALID_REVIEW_REQUIRED, TRAINING_READY
    """

    def generate_report(
        self,
        train_samples: list[ArchitectAISample],
        val_samples: list[ArchitectAISample],
        eval_benchmark_counts: dict[str, int],
        quarantine_count: int = 0,
        failed_parse_count: int = 0,
        exact_dups: int = 0,
        near_dups: int = 0,
        has_contamination: bool = False,
        gold_reviewed_count: int = 0,
        manual_review_completed: bool = False,
        export_validation_result: ExportValidationResult | dict[str, Any] | None = None,
        readiness_policy: dict[str, Any] | None = None,
        training_profile_concentration: float | None = None,
        output_file: Path | None = None,
    ) -> dict[str, Any]:
        policy = readiness_policy or {}
        min_gold = policy.get("min_gold_samples", 30)
        min_eval = policy.get("min_eval_samples", 30)
        max_source_ratio = policy.get("max_single_source_ratio", 0.80)
        require_manual_review = policy.get("require_manual_review", True)
        min_task_coverage = policy.get("minimum_task_coverage", {})

        max_duplicate_ids = policy.get("max_duplicate_ids", 0)
        max_conflicting_ids = policy.get("max_conflicting_duplicate_ids", 0)
        max_template_leakage = policy.get("max_template_leakage", 0)
        max_placeholders = policy.get("max_unresolved_placeholders", 0)
        max_empty_answers = policy.get("max_empty_assistant_answers", 0)
        max_semantic_failures = policy.get("max_semantic_failures_exported", 0)

        # Process export validation metrics
        if isinstance(export_validation_result, ExportValidationResult):
            val_res = export_validation_result.model_dump()
        elif isinstance(export_validation_result, dict):
            val_res = export_validation_result
        else:
            val_res = {}

        duplicate_ids = val_res.get("duplicate_sample_ids", 0)
        conflicting_ids = val_res.get("conflicting_duplicate_sample_ids", 0)
        template_leakage_count = val_res.get("template_leakage_count", 0)
        unresolved_placeholders = val_res.get("unresolved_placeholder_count", 0)
        empty_assistant_answers = val_res.get("empty_assistant_answers", 0)
        semantic_failures_exported = val_res.get("semantic_failures_exported", 0)
        group_overlap_count = val_res.get("group_overlap_count", 0)

        all_samples = train_samples + val_samples
        total_silver = len(all_samples)

        source_dist: dict[str, int] = {}
        task_dist: dict[str, int] = {}
        license_dist: dict[str, int] = {}

        unsupported_task_counts = 0
        missing_evidence_counts = 0

        for s in all_samples:
            src = s.source.source_id
            task = s.task_type.value if hasattr(s.task_type, "value") else str(s.task_type)
            lic = s.source.license_id

            source_dist[src] = source_dist.get(src, 0) + 1
            task_dist[task] = task_dist.get(task, 0) + 1
            license_dist[lic] = license_dist.get(lic, 0) + 1

            if not s.decisions and not (s.recommended_architecture and s.recommended_architecture.summary):
                missing_evidence_counts += 1

        # Group Split Integrity
        train_groups = {s.source.group_id for s in train_samples if s.source.group_id}
        val_groups = {s.source.group_id for s in val_samples if s.source.group_id}
        group_overlap = len(train_groups.intersection(val_groups)) + group_overlap_count

        # Source Concentration Ratios
        max_source_count = max(source_dist.values()) if source_dist else 0
        corpus_source_concentration = round(max_source_count / max(total_silver, 1), 4)

        total_processed = total_silver + quarantine_count + failed_parse_count
        quarantine_rate = round(quarantine_count / max(total_processed, 1), 4)
        parse_failure_rate = round(failed_parse_count / max(total_processed, 1), 4)
        exact_dup_rate = round(exact_dups / max(total_silver, 1), 4)
        near_dup_rate = round(near_dups / max(total_silver, 1), 4)

        total_eval_samples = sum(eval_benchmark_counts.values())

        blocking_reasons: list[str] = []
        warnings: list[str] = []

        # 1. Hard Structural & Export Gate Failures
        if has_contamination:
            blocking_reasons.append("Cross-split contamination detected between train and eval sets!")

        if conflicting_ids > max_conflicting_ids:
            blocking_reasons.append(f"Conflicting duplicate sample IDs detected ({conflicting_ids} > {max_conflicting_ids})!")

        if duplicate_ids > max_duplicate_ids:
            blocking_reasons.append(f"Duplicate sample IDs detected ({duplicate_ids} > {max_duplicate_ids})!")

        if template_leakage_count > max_template_leakage:
            blocking_reasons.append(f"Template leakage detected in exported SFT data ({template_leakage_count} > {max_template_leakage})!")

        if unresolved_placeholders > max_placeholders:
            blocking_reasons.append(f"Unresolved placeholders in exported SFT data ({unresolved_placeholders} > {max_placeholders})!")

        if empty_assistant_answers > max_empty_answers:
            blocking_reasons.append(f"Empty assistant answers in exported SFT data ({empty_assistant_answers} > {max_empty_answers})!")

        if semantic_failures_exported > max_semantic_failures:
            blocking_reasons.append(f"Semantic failures in exported SFT data ({semantic_failures_exported} > {max_semantic_failures})!")

        if group_overlap > 0:
            blocking_reasons.append(f"Train/validation group leakage detected ({group_overlap} overlapping groups)!")

        if total_silver == 0:
            blocking_reasons.append("No valid silver training samples produced.")

        # 2. Priority Task Coverage Gate
        if min_task_coverage:
            for task_name, min_cnt in min_task_coverage.items():
                actual_cnt = task_dist.get(task_name, 0)
                if actual_cnt < min_cnt:
                    warnings.append(
                        f"Priority task '{task_name}' count is below minimum threshold ({actual_cnt} < {min_cnt})."
                    )

        # 3. Readiness Status Evaluation
        if blocking_reasons:
            status = "BUILD_INVALID"
        else:
            if gold_reviewed_count < min_gold:
                warnings.append(
                    f"Reviewed gold sample count is below threshold ({gold_reviewed_count} < {min_gold})."
                )

            if total_eval_samples < min_eval:
                warnings.append(
                    f"Evaluation coverage is below recommended threshold ({total_eval_samples} < {min_eval})."
                )

            if corpus_source_concentration > max_source_ratio:
                warnings.append(
                    f"High corpus source concentration: single source represents {corpus_source_concentration * 100:.1f}% (> {max_source_ratio * 100:.1f}%)."
                )

            if parse_failure_rate > 0.05:
                warnings.append(f"High parse failure rate: {parse_failure_rate * 100:.1f}%")

            if require_manual_review and not manual_review_completed:
                warnings.append(
                    "Manual Quality Gate: Review quality_review_samples.jsonl and complete signoff before GPU model fine-tuning."
                )

            if warnings:
                status = "BUILD_VALID_REVIEW_REQUIRED"
            else:
                status = "TRAINING_READY"

        eval_summary = {**eval_benchmark_counts, "total_eval_samples": total_eval_samples}

        report: dict[str, Any] = {
            "readiness_status": status,
            "total_silver_samples": total_silver,
            "train_samples": len(train_samples),
            "validation_samples": len(val_samples),
            "gold_samples": gold_reviewed_count,
            "manual_review_completed": manual_review_completed,
            "evaluation_counts": eval_summary,
            "source_distribution": source_dist,
            "corpus_source_concentration": corpus_source_concentration,
            "training_profile_source_concentration": training_profile_concentration or corpus_source_concentration,
            "source_concentration_ratio": corpus_source_concentration,
            "task_distribution": task_dist,
            "license_coverage": license_dist,
            "quarantine_rate": quarantine_rate,
            "parse_failure_rate": parse_failure_rate,
            "exact_duplicate_count": exact_dups,
            "near_duplicate_count": near_dups,
            "exact_duplicate_rate": exact_dup_rate,
            "near_duplicate_rate": near_dup_rate,
            "duplicate_sample_ids": duplicate_ids,
            "conflicting_duplicate_sample_ids": conflicting_ids,
            "template_leakage_count": template_leakage_count,
            "unresolved_placeholder_count": unresolved_placeholders,
            "empty_assistant_answers": empty_assistant_answers,
            "semantic_failures_exported": semantic_failures_exported,
            "unsupported_task_counts": unsupported_task_counts,
            "missing_evidence_counts": missing_evidence_counts,
            "group_split_integrity": "FAILED" if group_overlap > 0 else "PASSED",
            "train_eval_contamination": "FAILED" if has_contamination else "PASSED",
            "blocking_reasons": blocking_reasons,
            "warnings": warnings,
        }

        if output_file:
            write_jsonl([report], output_file)

        return report
