from pathlib import Path

from architectai_dataset_builder.exporters.jsonl_exporter import JSONLExporter
from architectai_dataset_builder.models.canonical import (
    ArchitectAISample,
    ReviewStatus,
    SourceMetadata,
)
from architectai_dataset_builder.utils.io import read_jsonl


def test_gold_silver_export(tmp_path: Path):
    exporter = JSONLExporter(tmp_path)

    sample1 = ArchitectAISample(
        id="arch_s001",
        source=SourceMetadata(
            source_id="madr",
            source_name="MADR",
            source_file_path="0001.md",
            source_record_id="r1",
            license_id="MIT",
            license_verified=True,
            raw_sha256="hash1",
            normalized_sha256="nhash1",
            created_at="2026-08-12T12:00:00Z",
        ),
        scenario="Silver candidate scenario",
    )

    sample2 = ArchitectAISample(
        id="arch_g001",
        source=SourceMetadata(
            source_id="madr",
            source_name="MADR",
            source_file_path="0002.md",
            source_record_id="r2",
            license_id="MIT",
            license_verified=True,
            raw_sha256="hash2",
            normalized_sha256="nhash2",
            created_at="2026-08-12T12:00:00Z",
        ),
        scenario="Gold approved scenario",
    )

    approved_ids = ["arch_g001"]

    paths = exporter.export_training_datasets(
        train_samples=[sample1, sample2],
        val_samples=[],
        approved_sample_ids=approved_ids,
    )

    silver_data = read_jsonl(paths["silver"])
    gold_data = read_jsonl(paths["gold"])

    # Silver contains all non-approved samples
    assert len(silver_data) == 1
    assert silver_data[0]["id"] == "arch_s001"

    # Gold contains only manually approved samples
    assert len(gold_data) == 1
    assert gold_data[0]["id"] == "arch_g001"
    assert gold_data[0]["review"]["status"] == ReviewStatus.APPROVED.value
