from pathlib import Path

from architectai_dataset_builder.exporters.gold_seed_exporter import GoldSeedExporter
from architectai_dataset_builder.models.canonical import (
    ArchitectAISample,
    ReviewInfo,
    ReviewStatus,
    SourceMetadata,
    TaskType,
)


def test_gold_seed_export(tmp_path: Path):
    exporter = GoldSeedExporter(tmp_path)
    sample = ArchitectAISample(
        id="gold_candidate_1",
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

    out = exporter.export_review_candidates([sample])
    assert out.exists()
