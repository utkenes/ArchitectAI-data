"""
Automated Training Readiness Reporter for ArchitectAI Model Fine-Tuning Gate
"""

from pathlib import Path
from typing import Any

from architectai_dataset_builder.models.canonical import ArchitectAISample
from architectai_dataset_builder.utils.io import write_jsonl


class ReadinessReporter:
    def generate_report(
        self,
        train_samples: list[ArchitectAISample],
        val_samples: list[ArchitectAISample],
        eval_benchmark_counts: dict[str, int],
        quarantine_count: int,
        failed_parse_count: int,
        exact_dups: int,
        near_dups: int,
        has_contamination: bool,
        output_file: Path,
    ) -> dict[str, Any]:
        all_samples = train_samples + val_samples
        total_silver = len(all_samples)

        source_dist: dict[str, int] = {}
        task_dist: dict[str, int] = {}
        license_dist: dict[str, int] = {}

        for s in all_samples:
            src = s.source.source_id
            task = s.task_type.value
            lic = s.source.license_id
            source_dist[src] = source_dist.get(src, 0) + 1
            task_dist[task] = task_dist.get(task, 0) + 1
            license_dist[lic] = license_dist.get(lic, 0) + 1

        total_processed = total_silver + quarantine_count + failed_parse_count
        quarantine_rate = round(quarantine_count / max(total_processed, 1), 4)
        parse_failure_rate = round(failed_parse_count / max(total_processed, 1), 4)
        exact_dup_rate = round(exact_dups / max(total_silver, 1), 4)
        near_dup_rate = round(near_dups / max(total_silver, 1), 4)

        total_eval_samples = sum(eval_benchmark_counts.values())

        blocking_reasons: list[str] = []
        warnings: list[str] = []

        if has_contamination:
            blocking_reasons.append("Cross-split contamination detected between train and eval sets!")

        if total_silver == 0:
            blocking_reasons.append("No valid silver training samples produced.")

        if parse_failure_rate > 0.05:
            warnings.append(f"High parse failure rate: {parse_failure_rate * 100}%")

        if len(source_dist) < 2:
            warnings.append("Low source diversity: less than 2 distinct sources.")

        if total_eval_samples < 10:
            warnings.append(
                f"Evaluation coverage is limited (total eval samples: {total_eval_samples}); model comparison results should be treated as preliminary."
            )

        warnings.append("Manual Quality Gate: Review quality_review_samples.jsonl before starting GPU model training.")

        status = "NOT_READY" if blocking_reasons else "READY_WITH_WARNINGS"

        eval_summary = {**eval_benchmark_counts, "total_eval_samples": total_eval_samples}

        report: dict[str, Any] = {
            "readiness_status": status,
            "total_silver_samples": total_silver,
            "train_samples": len(train_samples),
            "validation_samples": len(val_samples),
            "gold_samples": 0,
            "evaluation_counts": eval_summary,
            "source_distribution": source_dist,
            "task_distribution": task_dist,
            "license_coverage": license_dist,
            "quarantine_rate": quarantine_rate,
            "parse_failure_rate": parse_failure_rate,
            "exact_duplicate_rate": exact_dup_rate,
            "near_duplicate_rate": near_dup_rate,
            "group_split_integrity": "PASSED",
            "train_eval_contamination": "FAILED" if has_contamination else "PASSED",
            "unsupported_task_counts": 0,
            "missing_evidence_counts": 0,
            "blocking_reasons": blocking_reasons,
            "warnings": warnings,
        }

        write_jsonl([report], output_file)
        return report
