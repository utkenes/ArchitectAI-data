from pathlib import Path

from architectai_dataset_builder.models.canonical import (
    ArchitectAISample,
    ReviewInfo,
    ReviewStatus,
    SourceMetadata,
    TaskType,
)
from architectai_dataset_builder.reports.readiness_reporter import ReadinessReporter


def test_readiness_report_build_valid_review_required(tmp_path: Path):
    reporter = ReadinessReporter()
    sample = ArchitectAISample(
        id="s1",
        source=SourceMetadata(
            source_id="opendatahub_adr",
            source_name="ODH",
            source_file_path="1.md",
            source_record_id="1",
            license_id="Apache-2.0",
            license_verified=True,
            raw_sha256="abc",
            normalized_sha256="def",
        ),
        scenario="Scenario description",
        task_type=TaskType.ADR_REASONING,
        review=ReviewInfo(status=ReviewStatus.UNREVIEWED),
    )

    out_file = tmp_path / "training_readiness_report.json"
    rep = reporter.generate_report(
        train_samples=[sample],
        val_samples=[],
        eval_benchmark_counts={"sake": 10, "cake": 10, "archbench": 10, "r2abench": 10},
        quarantine_count=2,
        failed_parse_count=0,
        exact_dups=0,
        near_dups=0,
        has_contamination=False,
        gold_reviewed_count=0,  # Below threshold
        output_file=out_file,
    )

    assert rep["readiness_status"] == "BUILD_VALID_REVIEW_REQUIRED"
    assert rep["total_silver_samples"] == 1
    assert rep["evaluation_counts"]["total_eval_samples"] == 40
    assert out_file.exists()


def test_readiness_report_build_invalid_on_contamination(tmp_path: Path):
    reporter = ReadinessReporter()
    sample = ArchitectAISample(
        id="s1",
        source=SourceMetadata(
            source_id="opendatahub_adr",
            source_name="ODH",
            source_file_path="1.md",
            source_record_id="1",
            license_id="Apache-2.0",
            license_verified=True,
            raw_sha256="abc",
            normalized_sha256="def",
        ),
        scenario="Scenario description",
        task_type=TaskType.ADR_REASONING,
        review=ReviewInfo(status=ReviewStatus.UNREVIEWED),
    )

    rep = reporter.generate_report(
        train_samples=[sample],
        val_samples=[],
        eval_benchmark_counts={"sake": 10, "cake": 10, "archbench": 10, "r2abench": 10},
        has_contamination=True,
        gold_reviewed_count=50,
    )

    assert rep["readiness_status"] == "BUILD_INVALID"
    assert "Cross-split contamination detected between train and eval sets!" in rep["blocking_reasons"]


def test_readiness_report_training_ready(tmp_path: Path):
    reporter = ReadinessReporter()
    sample1 = ArchitectAISample(
        id="s1",
        source=SourceMetadata(
            source_id="opendatahub_adr",
            source_name="ODH",
            source_file_path="1.md",
            source_record_id="1",
            license_id="Apache-2.0",
            license_verified=True,
            raw_sha256="abc",
            normalized_sha256="def",
        ),
        scenario="Scenario 1 description",
        task_type=TaskType.ADR_REASONING,
        review=ReviewInfo(status=ReviewStatus.APPROVED),
    )
    sample2 = ArchitectAISample(
        id="s2",
        source=SourceMetadata(
            source_id="madr",
            source_name="MADR",
            source_file_path="2.md",
            source_record_id="2",
            license_id="MIT",
            license_verified=True,
            raw_sha256="xyz",
            normalized_sha256="uvw",
        ),
        scenario="Scenario 2 description",
        task_type=TaskType.ADR_REASONING,
        review=ReviewInfo(status=ReviewStatus.APPROVED),
    )

    rep = reporter.generate_report(
        train_samples=[sample1],
        val_samples=[sample2],
        eval_benchmark_counts={"sake": 10, "cake": 10, "archbench": 10, "r2abench": 10},
        has_contamination=False,
        gold_reviewed_count=35,
        readiness_policy={
            "min_gold_samples": 30,
            "min_eval_samples": 30,
            "max_single_source_ratio": 0.80,
            "require_manual_review": False,
        },
    )

    assert rep["readiness_status"] == "TRAINING_READY"
