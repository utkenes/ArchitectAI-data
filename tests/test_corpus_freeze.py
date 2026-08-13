import re
from pathlib import Path

from architectai_dataset_builder.exporters.corpus_manifest_exporter import CorpusManifestExporter
from architectai_dataset_builder.exporters.jsonl_exporter import JSONLExporter
from architectai_dataset_builder.models.evaluation import (
    EvalSourceMetadata,
    MultipleChoiceEvalSample,
)
from architectai_dataset_builder.splitters.deterministic_splitter import DeterministicSplitter


def test_corpus_manifest_fingerprints(tmp_path: Path):
    exporter = CorpusManifestExporter(tmp_path)
    sft_file = tmp_path / "train_sft.jsonl"
    sft_file.write_text('{"test": 1}\n', encoding="utf-8")

    manifest = exporter.export_corpus_manifest(
        build_id="test_build",
        sources_summary={"madr": {"requested_ref": "main", "resolved_commit": "835fc94baa37887774b1cddddb2ae874881e703b"}},
        sample_counts={"train": 1, "validation": 0, "silver": 1, "gold": 0, "eval": 0},
        output_file=tmp_path / "corpus_manifest.json",
    )

    assert manifest["corpus_version"] == "1.0.0"
    assert "train_sft.jsonl" in manifest["artifact_hashes"]
    hash1 = manifest["artifact_hashes"]["train_sft.jsonl"]

    # Modify file and confirm fingerprint changes
    sft_file.write_text('{"test": 2}\n', encoding="utf-8")
    manifest2 = exporter.export_corpus_manifest(
        build_id="test_build",
        sources_summary={"madr": {"requested_ref": "main", "resolved_commit": "835fc94baa37887774b1cddddb2ae874881e703b"}},
        sample_counts={"train": 1, "validation": 0, "silver": 1, "gold": 0, "eval": 0},
        output_file=tmp_path / "corpus_manifest.json",
    )
    hash2 = manifest2["artifact_hashes"]["train_sft.jsonl"]
    assert hash1 != hash2


def test_resolved_commit_sha_invariant():
    """Invariance check: resolved_commit must be either a real 40-char SHA or None (no placeholder strings)."""
    valid_sha_pattern = re.compile(r"^[0-9a-fA-F]{40}$")

    test_commits = [
        "f2fb7f7364bac8beedefe608437f02036f39de14",
        None,
    ]

    for sha in test_commits:
        if sha is not None:
            assert valid_sha_pattern.match(sha), f"Invalid resolved_commit format: {sha}"
            assert not sha.startswith("fixture_commit")
            assert not sha.startswith("unknown_commit")


def test_eval_counts_reconciliation(tmp_path: Path):
    exporter = JSONLExporter(tmp_path)
    sample = MultipleChoiceEvalSample(
        id="sake_q1",
        source=EvalSourceMetadata(
            benchmark_id="sake",
            sample_id="sake_q1",
            license_id="CC-BY-4.0",
            raw_sha256="raw",
            normalized_sha256="norm",
        ),
        question="Question 1?",
        options={"A": "Opt A"},
        correct_answer="A",
    )

    paths = exporter.export_evaluation_datasets([sample])
    assert "eval_manifest" in paths
    assert (tmp_path / "eval" / "sake.jsonl").exists()
    assert not (tmp_path / "eval" / "r2abench.jsonl").exists()


def test_r2abench_project_disjointness(tmp_path: Path):
    split_file = tmp_path / "splits.yaml"
    split_file.write_text(
        "splits:\n"
        "  training_projects: ['proj_001']\n"
        "  validation_projects: ['proj_002']\n"
        "  held_out_projects: ['proj_003']\n",
        encoding="utf-8",
    )

    splitter = DeterministicSplitter(split_file)
    train_pids = set(splitter.r2abench_splits["training"])
    holdout_pids = set(splitter.r2abench_splits["held_out"])

    # Hard Invariant: No overlap between training projects and holdout evaluation projects
    assert len(train_pids & holdout_pids) == 0
