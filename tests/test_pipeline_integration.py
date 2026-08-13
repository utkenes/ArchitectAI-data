from click.testing import CliRunner

from architectai_dataset_builder.cli import build_dataset
from architectai_dataset_builder.config import Config


def test_full_pipeline_integration():
    cfg = Config()
    runner = CliRunner()

    result = runner.invoke(build_dataset, ["--build-id", "test_integration_run", "--mode", "fixture"])
    assert result.exit_code == 0, f"Pipeline failed with output:\n{result.output}"

    # Verify export files exist
    export_dir = cfg.data_dir / "exports"
    assert (export_dir / "train_sft.jsonl").exists()
    assert (export_dir / "validation_sft.jsonl").exists()
    assert (export_dir / "silver.jsonl").exists()
    assert (export_dir / "gold.jsonl").exists()
    assert (export_dir / "eval" / "sake.jsonl").exists()
    assert (export_dir / "eval_manifest.json").exists()
    assert (export_dir / "contamination_report.json").exists()
    assert (export_dir / "build_manifest.json").exists()
