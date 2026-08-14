"""
Automated Training Readiness Reporter V2 for ArchitectAI Model Fine-Tuning Gate
"""

from pathlib import Path
from typing import Any

from architectai_dataset_builder.models.canonical import ArchitectAISample
from architectai_dataset_builder.utils.io import write_jsonl


class ReadinessReporter:
    """
    Computes measurable training readiness criteria based on actual dataset metrics and policies.
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
        duplicate_sample_ids: int = 0,
        template_leakage_count: int = 0,
        semantic_gate_failures: int = 0,
        gold_reviewed_count: int = 0,
        readiness_policy: dict[str, Any] | None = None,
        output_file: Path | None = None,
    ) -> dict[str, Any]:
        policy = readiness_policy or {}
        min_gold = policy.get("min_gold_samples", 30)
        min_eval = policy.get("min_eval_samples", 30)
        max_source_ratio = policy.get("max_single_source_ratio", 0.80)
        require_manual_review = policy.get("require_manual_review", True)

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

        # Calculate Group Split Integrity (overlap check)
        train_groups = {s.source.group_id for s in train_samples if s.source.group_id}
        val_groups = {s.source.group_id for s in val_samples if s.source.group_id}
        group_overlap = len(train_groups.intersection(val_groups))

        # Calculate Source Concentration Ratio
        max_source_count = max(source_dist.values()) if source_dist else 0
        source_concentration_ratio = round(max_source_count / max(total_silver, 1), 4)

        total_processed = total_silver + quarantine_count + failed_parse_count
        quarantine_rate = round(quarantine_count / max(total_processed, 1), 4)
        parse_failure_rate = round(failed_parse_count / max(total_processed, 1), 4)
        exact_dup_rate = round(exact_dups / max(total_silver, 1), 4)
        near_dup_rate = round(near_dups / max(total_silver, 1), 4)

        total_eval_samples = sum(eval_benchmark_counts.values())

        blocking_reasons: list[str] = []
        warnings: list[str] = []

        # Structural & Hard Validation Failures
        if has_contamination:
            blocking_reasons.append("Cross-split contamination detected between train and eval sets!")

        if duplicate_sample_ids > 0:
            blocking_reasons.append(f"Duplicate sample IDs detected ({duplicate_sample_ids})!")

        if template_leakage_count > 0:
            blocking_reasons.append(f"Template leakage detected in exported SFT data ({template_leakage_count})!")

        if group_overlap > 0:
            blocking_reasons.append(f"Train/validation group leakage detected ({group_overlap} overlapping groups)!")

        if total_silver == 0:
            blocking_reasons.append("No valid silver training samples produced.")

        # Determine Readiness Status
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

            if source_concentration_ratio > max_source_ratio:
                warnings.append(
                    f"High source concentration: single source represents {source_concentration_ratio * 100:.1f}% (> {max_source_ratio * 100:.1f}%)."
                )

            if parse_failure_rate > 0.05:
                warnings.append(f"High parse failure rate: {parse_failure_rate * 100:.1f}%")

            if require_manual_review:
                warnings.append(
                    "Manual Quality Gate: Review quality_review_samples.jsonl before starting GPU model fine-tuning."
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
            "evaluation_counts": eval_summary,
            "source_distribution": source_dist,
            "source_concentration_ratio": source_concentration_ratio,
            "task_distribution": task_dist,
            "license_coverage": license_dist,
            "quarantine_rate": quarantine_rate,
            "parse_failure_rate": parse_failure_rate,
            "exact_duplicate_rate": exact_dup_rate,
            "near_duplicate_rate": near_dup_rate,
            "duplicate_sample_ids": duplicate_sample_ids,
            "template_leakage_count": template_leakage_count,
            "semantic_gate_failures": semantic_gate_failures,
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
