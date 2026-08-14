from architectai_dataset_builder.models.canonical import (
    ArchitectAISample,
    ReviewInfo,
    SourceMetadata,
    TaskType,
)
from architectai_dataset_builder.models.evidence import EvidenceItem, EvidenceType
from architectai_dataset_builder.validators.semantic_quality import SemanticQualityValidator


def test_semantic_quality_gate_accepts_good_sample():
    validator = SemanticQualityValidator()
    sample = ArchitectAISample(
        id="arch_test123",
        source=SourceMetadata(
            source_id="madr",
            source_name="MADR",
            source_file_path="0001.md",
            source_record_id="0001",
            license_id="CC-BY-4.0",
            raw_sha256="hash1",
            normalized_sha256="hash2",
        ),
        scenario="We need a persistent relational database for user profile storage.",
        task_type=TaskType.ADR_REASONING,
        decisions=[
            EvidenceItem(value="Use PostgreSQL with multi-AZ replication", evidence_type=EvidenceType.EXPLICIT)
        ],
        final_answer="Decision: Use PostgreSQL with multi-AZ replication",
        review=ReviewInfo(),
    )
    result = validator.validate_sample(sample)
    assert result.passed is True
    assert len(result.reasons) == 0


def test_semantic_quality_gate_rejects_html_comments():
    validator = SemanticQualityValidator()
    sample = ArchitectAISample(
        id="arch_leak123",
        source=SourceMetadata(
            source_id="madr",
            source_name="MADR",
            source_file_path="0001.md",
            source_record_id="0001",
            license_id="CC-BY-4.0",
            raw_sha256="hash1",
            normalized_sha256="hash2",
        ),
        scenario="<!-- Describe problem --> Scenario text",
        task_type=TaskType.ADR_REASONING,
        decisions=[EvidenceItem(value="Use PostgreSQL", evidence_type=EvidenceType.EXPLICIT)],
        review=ReviewInfo(),
    )
    result = validator.validate_sample(sample)
    assert result.passed is False
    assert result.quarantine_category == "template_leakage"
