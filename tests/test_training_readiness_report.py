from pathlib import Path
from architectai_dataset_builder.reports.readiness_reporter import ReadinessReporter
from architectai_dataset_builder.models.canonical import ArchitectAISample, SourceMetadata, TaskType, ReviewInfo, ReviewStatus


def test_readiness_report_generation(tmp_path: Path):
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
        scenario="Scenario",
        task_type=TaskType.ADR_REASONING,
        review=ReviewInfo(status=ReviewStatus.UNREVIEWED),
    )

    out_file = tmp_path / "training_readiness_report.json"
    rep = reporter.generate_report(
        train_samples=[sample],
        val_samples=[],
        eval_samples_count=5,
        quarantine_count=2,
        failed_parse_count=0,
        exact_dups=0,
        near_dups=0,
        has_contamination=False,
        output_file=out_file,
    )

    assert rep["readiness_status"] in ["READY", "READY_WITH_WARNINGS"]
    assert rep["total_silver_samples"] == 1
    assert out_file.exists()
