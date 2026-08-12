import pytest
from pathlib import Path
from click.testing import CliRunner
from architectai_dataset_builder.cli import build_dataset
from architectai_dataset_builder.config import Config
from architectai_dataset_builder.utils.io import read_jsonl


def test_full_pipeline_integration():
    cfg = Config()
    runner = CliRunner()

    result = runner.invoke(build_dataset, ["--build-id", "test_integration_run"])
    assert result.exit_code == 0, f"Pipeline failed with output:\n{result.output}"

    export_dir = cfg.data_dir / "exports"
    assert (export_dir / "silver.jsonl").exists()
    assert (export_dir / "gold.jsonl").exists()
    assert (export_dir / "train_sft.jsonl").exists()
    assert (export_dir / "validation_sft.jsonl").exists()
    assert (export_dir / "eval_manifest.json").exists()
    assert (export_dir / "dataset_stats.json").exists()
    assert (export_dir / "contamination_report.json").exists()
    assert (export_dir / "build_manifest.json").exists()

    # Assert contamination report is clean
    contam_data = read_jsonl(export_dir / "contamination_report.json")[0]
    assert contam_data["has_leakage"] is False
    assert contam_data["total_leaks_detected"] == 0

    # Assert build manifest records export hashes
    build_manifest = read_jsonl(export_dir / "build_manifest.json")[0]
    assert build_manifest["build_id"] == "test_integration_run"
    assert "train_sft_sha256" in build_manifest["export_hashes"]
