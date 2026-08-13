from pathlib import Path
from architectai_dataset_builder.exporters.quality_sampler import QualitySampler
from architectai_dataset_builder.models.canonical import ArchitectAISample, SourceMetadata, TaskType, ReviewInfo, ReviewStatus


def test_quality_sampler_export(tmp_path: Path):
    sampler = QualitySampler(tmp_path)
    sample = ArchitectAISample(
        id="quality_sample_1",
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
        final_answer="Decision answer",
        review=ReviewInfo(status=ReviewStatus.UNREVIEWED),
    )

    out = sampler.export_quality_samples([sample])
    assert out.exists()
