import pytest

from architectai_dataset_builder.models.canonical import ArchitectAISample, SourceMetadata
from architectai_dataset_builder.models.evaluation import (
    EvalSourceMetadata,
    MultipleChoiceEvalSample,
)
from architectai_dataset_builder.splitters.contamination_checker import (
    ContaminationChecker,
    EvaluationLeakageError,
)


def test_eval_contamination_detection():
    checker = ContaminationChecker(protected_sources=["sake", "cake", "archbench"])

    # Normal clean training sample
    train_sample = ArchitectAISample(
        id="arch_train001",
        source=SourceMetadata(
            source_id="madr",
            source_name="MADR",
            source_file_path="0001.md",
            source_record_id="r1",
            license_id="MIT",
            license_verified=True,
            raw_sha256="hash_train_raw",
            normalized_sha256="hash_train_norm",
            created_at="2026-08-12T12:00:00Z",
        ),
        scenario="Unique context for training scenario",
    )

    eval_sample = MultipleChoiceEvalSample(
        id="eval_sake_001",
        source=EvalSourceMetadata(
            benchmark_id="sake",
            sample_id="eval_sake_001",
            license_id="CC-BY-4.0",
            raw_sha256="hash_sake_raw",
            normalized_sha256="hash_sake_norm",
        ),
        question="SAKE question text",
        options={"A": "1", "B": "2"},
        correct_answer="A",
    )

    # 1. Clean check passes
    report = checker.verify_no_leakage([train_sample], [eval_sample])
    assert report.has_leakage is False

    # 2. Contaminated training sample with protected source_id raises EvaluationLeakageError
    leaked_sample = ArchitectAISample(
        id="arch_leaked",
        source=SourceMetadata(
            source_id="sake",  # Protected evaluation benchmark source in training!
            source_name="SAKE Leaked",
            source_file_path="sake.json",
            source_record_id="r1",
            license_id="CC-BY-4.0",
            license_verified=True,
            raw_sha256="hash_sake_raw",
            normalized_sha256="hash_sake_norm",
            created_at="2026-08-12T12:00:00Z",
        ),
        scenario="SAKE question text leaked",
    )

    with pytest.raises(EvaluationLeakageError):
        checker.verify_no_leakage([leaked_sample], [eval_sample])
