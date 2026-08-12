from architectai_dataset_builder.models.canonical import ArchitectAISample
from architectai_dataset_builder.models.evaluation import (
    EvalSourceMetadata,
    MultipleChoiceEvalSample,
)


def test_eval_schema_isolation():
    eval_meta = EvalSourceMetadata(
        benchmark_id="sake",
        sample_id="eval_sake_001",
        license_id="CC-BY-4.0",
        raw_sha256="raw_hash",
        normalized_sha256="norm_hash",
        evaluation_only=True,
    )

    mc_sample = MultipleChoiceEvalSample(
        id="eval_sake_001",
        source=eval_meta,
        question="Which component is pub/sub?",
        options={"A": "Kafka", "B": "MySQL"},
        correct_answer="A",
    )

    assert isinstance(mc_sample, MultipleChoiceEvalSample)
    assert not isinstance(mc_sample, ArchitectAISample)
    assert mc_sample.source.evaluation_only is True
