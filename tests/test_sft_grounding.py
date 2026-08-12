from architectai_dataset_builder.exporters.sft_formatter import SFTFormatter
from architectai_dataset_builder.models.canonical import ArchitectAISample, SourceMetadata, TaskType


def test_sft_formatter_grounding():
    formatter = SFTFormatter()

    sample = ArchitectAISample(
        id="arch_sft001",
        source=SourceMetadata(
            source_id="madr",
            source_name="MADR",
            source_file_path="0001.md",
            source_record_id="rec1",
            license_id="MIT",
            license_verified=True,
            raw_sha256="raw",
            normalized_sha256="norm",
            created_at="2026-08-12T12:00:00Z",
        ),
        scenario="We need decoupled order processing.",
        task_type=TaskType.ADR_REASONING,
        final_answer="Decision: Use Event-Driven Architecture with RabbitMQ.",
    )

    formatted = formatter.format_sample(sample)

    assert formatted["sample_id"] == "arch_sft001"
    assert formatted["task_type"] == TaskType.ADR_REASONING.value
    assert len(formatted["messages"]) == 3
    assert formatted["messages"][0]["role"] == "system"
    assert "What architectural decision was made" in formatted["messages"][1]["content"]
    assert "Decision: Use Event-Driven Architecture" in formatted["messages"][2]["content"]
