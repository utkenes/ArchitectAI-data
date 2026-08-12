from pathlib import Path

from architectai_dataset_builder.exporters.build_manifest_exporter import BuildManifestExporter
from architectai_dataset_builder.utils.hashing import compute_sha256_file
from architectai_dataset_builder.utils.io import read_jsonl


def test_build_manifest_generation(tmp_path: Path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    policy_file = config_dir / "dataset_policy.yaml"
    policy_file.write_text("min_relevance_score: 0.60\n", encoding="utf-8")

    split_file = tmp_path / "splits.yaml"
    split_file.write_text("train: 0.8\n", encoding="utf-8")

    dummy_export = tmp_path / "train_sft.jsonl"
    dummy_export.write_text('{"test": 1}\n', encoding="utf-8")

    exporter = BuildManifestExporter()
    output_manifest = tmp_path / "build_manifest.json"

    manifest = exporter.export_build_manifest(
        build_id="build_test_123",
        config_dir=config_dir,
        split_manifest_path=split_file,
        sources_summary={"madr": {"commit_sha": "abc"}},
        sample_counts={"train": 10},
        export_paths={"train_sft": dummy_export},
        output_file=output_manifest,
    )

    assert manifest.build_id == "build_test_123"
    assert manifest.export_hashes["train_sft_sha256"] == compute_sha256_file(dummy_export)

    read_data = read_jsonl(output_manifest)[0]
    assert read_data["build_id"] == "build_test_123"
