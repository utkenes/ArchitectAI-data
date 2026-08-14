"""
Stage-Wise Pipeline Attrition & Corpus Coverage Reporter
"""

from collections import Counter
from pathlib import Path
from typing import Any

from architectai_dataset_builder.models.canonical import ArchitectAISample, TaskType
from architectai_dataset_builder.utils.io import write_jsonl


class CorpusCoverageReporter:
    """
    Tracks stage-wise pipeline attrition per source and computes source x task distribution matrices.
    """

    def generate_report(
        self,
        raw_counts: dict[str, int],
        parsed_counts: dict[str, int],
        quarantine_breakdowns: dict[str, Counter[str]],
        valid_samples: list[ArchitectAISample],
        train_samples: list[ArchitectAISample],
        val_samples: list[ArchitectAISample],
        output_file: Path | None = None,
    ) -> dict[str, Any]:
        all_samples = train_samples + val_samples

        source_ids = sorted(
            set(raw_counts.keys())
            | set(parsed_counts.keys())
            | {s.source.source_id for s in all_samples}
        )

        attrition_by_source: dict[str, dict[str, Any]] = {}
        for sid in source_ids:
            s_raw = raw_counts.get(sid, 0)
            s_parsed = parsed_counts.get(sid, 0)
            s_train = sum(1 for s in train_samples if s.source.source_id == sid)
            s_val = sum(1 for s in val_samples if s.source.source_id == sid)
            s_quarantine = dict(quarantine_breakdowns.get(sid, Counter()))

            attrition_by_source[sid] = {
                "raw_discovered": s_raw,
                "parsed_records": s_parsed,
                "final_train": s_train,
                "final_validation": s_val,
                "final_silver_total": s_train + s_val,
                "quarantine_breakdown": s_quarantine,
            }

        # Source x Task distribution matrix
        source_task_matrix: dict[str, dict[str, int]] = {}
        for sid in source_ids:
            source_task_matrix[sid] = {t.value: 0 for t in TaskType}

        for s in all_samples:
            sid = s.source.source_id
            tval = s.task_type.value
            if sid not in source_task_matrix:
                source_task_matrix[sid] = {t.value: 0 for t in TaskType}
            source_task_matrix[sid][tval] += 1

        # Global task distribution
        task_counts: Counter[str] = Counter()
        for s in all_samples:
            task_counts[s.task_type.value] += 1

        priority_tasks = [
            "adr_reasoning",
            "tradeoff_analysis",
            "technology_selection",
            "scaling_reasoning",
            "quality_attribute_reasoning",
        ]

        missing_priority_tasks = [t for t in priority_tasks if task_counts[t] == 0]
        low_volume_tasks = [t for t in priority_tasks if 0 < task_counts[t] < 10]

        report = {
            "attrition_by_source": attrition_by_source,
            "source_task_matrix": source_task_matrix,
            "global_task_distribution": dict(task_counts),
            "priority_task_coverage": {
                "priority_tasks": priority_tasks,
                "missing_priority_tasks": missing_priority_tasks,
                "low_volume_tasks": low_volume_tasks,
            },
            "summary": {
                "total_raw_records": sum(raw_counts.values()),
                "total_parsed_records": sum(parsed_counts.values()),
                "total_silver_samples": len(all_samples),
                "total_train_samples": len(train_samples),
                "total_validation_samples": len(val_samples),
            },
        }

        if output_file:
            write_jsonl([report], Path(output_file))

        return report
