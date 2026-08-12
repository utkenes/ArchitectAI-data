"""
Dataset Statistics Generator
"""

from collections import Counter
from pathlib import Path

from architectai_dataset_builder.models.canonical import ArchitectAISample
from architectai_dataset_builder.models.reports import DatasetStatsReport
from architectai_dataset_builder.utils.io import write_jsonl


class StatsGenerator:
    def generate_stats(
        self,
        train_samples: list[ArchitectAISample],
        val_samples: list[ArchitectAISample],
        exact_dups: int,
        near_dups: int,
        quarantine_count: int,
        quarantine_reasons: dict[str, int],
        failed_parse_count: int,
        source_mode: str,
        build_status: str,
        output_file: Path,
    ) -> DatasetStatsReport:
        all_samples = train_samples + val_samples

        split_counts = Counter([s.source.split or "unknown" for s in all_samples])
        source_counts = Counter([s.source.source_id for s in all_samples])
        task_counts = Counter([s.task_type.value for s in all_samples])
        quality_counts = Counter([s.review.status.value for s in all_samples])
        license_counts = Counter([s.source.license_id for s in all_samples])
        review_counts = Counter([s.review.status.value for s in all_samples])

        report = DatasetStatsReport(
            source_mode=source_mode,
            build_status=build_status,
            total_samples=len(all_samples),
            sample_count_by_split=dict(split_counts),
            sample_count_by_source=dict(source_counts),
            sample_count_by_task_type=dict(task_counts),
            sample_count_by_quality_class=dict(quality_counts),
            sample_count_by_license=dict(license_counts),
            duplicate_count=exact_dups,
            near_duplicate_count=near_dups,
            quarantine_count=quarantine_count,
            quarantine_reasons=quarantine_reasons,
            failed_parse_count=failed_parse_count,
            review_status_distribution=dict(review_counts),
        )

        write_jsonl([report.model_dump()], output_file)
        return report
