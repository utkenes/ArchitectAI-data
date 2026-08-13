from architectai_dataset_builder.exporters.sft_formatter import SFTFormatter
from architectai_dataset_builder.models.canonical import ArchitectAISample, SourceMetadata, TaskType
from architectai_dataset_builder.utils.markdown import (
    has_template_placeholders,
    is_boilerplate_filename,
)


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
        scenario="We need decoupled order processing for microservices messaging.",
        task_type=TaskType.ADR_REASONING,
        final_answer="Decision: Use Event-Driven Architecture with RabbitMQ.",
    )

    formatted = formatter.format_sample(sample)
    assert formatted is not None
    assert formatted["sample_id"] == "arch_sft001"
    assert formatted["task_type"] == TaskType.ADR_REASONING.value
    assert len(formatted["messages"]) == 3
    assert formatted["messages"][0]["role"] == "system"
    assert "What architectural decision was made" in formatted["messages"][1]["content"]
    assert "Decision: Use Event-Driven Architecture" in formatted["messages"][2]["content"]


def test_not_explicitly_stated_rejection():
    formatter = SFTFormatter()

    sample = ArchitectAISample(
        id="arch_sft_invalid",
        source=SourceMetadata(
            source_id="madr",
            source_name="MADR",
            source_file_path="0002.md",
            source_record_id="rec2",
            license_id="MIT",
            license_verified=True,
            raw_sha256="raw",
            normalized_sha256="norm",
            created_at="2026-08-12T12:00:00Z",
        ),
        scenario="Test scenario",
        task_type=TaskType.ADR_REASONING,
        final_answer="Decision / Proposal: Not explicitly stated",
    )

    formatted = formatter.format_sample(sample)
    assert formatted is None


def test_template_placeholder_detection():
    assert has_template_placeholders("Use ${NNNN} for decision records")
    assert has_template_placeholders("Title: {{ title }}")
    assert has_template_placeholders("Option: {title of option 1}")
    assert has_template_placeholders("Problem: [short title of solved problem]")
    assert not has_template_placeholders("We selected PostgreSQL for relational ACID consistency.")


def test_boilerplate_filename_detection():
    assert is_boilerplate_filename("release-notes.md")
    assert is_boilerplate_filename("changelog.md")
    assert is_boilerplate_filename("version_1.0.md")
    assert is_boilerplate_filename("adr000-template.md")
    assert not is_boilerplate_filename("0001-pipeline-orchestration.md")
