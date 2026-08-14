"""
Comprehensive V1.1.1 Pre-Training Gate Hardening Regressions
"""

from pathlib import Path

from architectai_dataset_builder.exporters.sft_formatter import SFTFormatter
from architectai_dataset_builder.models.canonical import (
    ArchitectAISample,
    ReviewInfo,
    ReviewStatus,
    SourceMetadata,
    TaskType,
)
from architectai_dataset_builder.models.evidence import EvidenceItem, EvidenceType
from architectai_dataset_builder.normalizers.task_taxonomy import TaskTaxonomyClassifier
from architectai_dataset_builder.reports.readiness_reporter import ReadinessReporter
from architectai_dataset_builder.utils.io import write_jsonl
from architectai_dataset_builder.validators.semantic_quality import SemanticQualityValidator
from architectai_dataset_builder.validators.sft_export_validator import SFTExportValidator


# 1. SFTExportValidator Disk JSONL Tests
def test_export_validator_clean_and_failure_modes(tmp_path: Path):
    validator = SFTExportValidator()

    train_clean = [
        {
            "id": "s1",
            "group_id": "g1",
            "messages": [
                {"role": "system", "content": "Sys"},
                {"role": "user", "content": "User prompt"},
                {"role": "assistant", "content": "Selected Approach:\nUse PostgreSQL cluster."},
            ],
        }
    ]
    val_clean = [
        {
            "id": "s2",
            "group_id": "g2",
            "messages": [
                {"role": "system", "content": "Sys"},
                {"role": "user", "content": "User prompt"},
                {"role": "assistant", "content": "Decision:\nImplement Redis cache."},
            ],
        }
    ]

    write_jsonl(train_clean, tmp_path / "train_sft.jsonl")
    write_jsonl(val_clean, tmp_path / "validation_sft.jsonl")

    res = validator.validate_exports(tmp_path)
    assert res.has_critical_failures is False
    assert res.total_exported_samples == 2
    assert res.template_leakage_count == 0
    assert res.duplicate_sample_ids == 0

    # Leakage & Duplicate ID mode
    train_bad = [
        {
            "id": "s1",
            "group_id": "g1",
            "messages": [
                {"role": "system", "content": "Sys"},
                {"role": "user", "content": "User prompt"},
                {"role": "assistant", "content": "<!-- raw comment --> Selected Approach:\nUse PostgreSQL."},
            ],
        },
        {
            "id": "s1",  # Conflicting duplicate ID
            "group_id": "g1",
            "messages": [
                {"role": "system", "content": "Sys"},
                {"role": "user", "content": "User prompt"},
                {"role": "assistant", "content": "Decision:\nUse MySQL cluster."},
            ],
        },
    ]
    write_jsonl(train_bad, tmp_path / "train_sft.jsonl")
    res_bad = validator.validate_exports(tmp_path)
    assert res_bad.has_critical_failures is True
    assert res_bad.template_leakage_count >= 1
    assert res_bad.conflicting_duplicate_sample_ids >= 1


# 2. Positive Path TRAINING_READY
def test_readiness_positive_path_reaches_training_ready():
    reporter = ReadinessReporter()
    sample1 = ArchitectAISample(
        id="s1",
        source=SourceMetadata(
            source_id="madr",
            source_name="MADR",
            source_file_path="01.md",
            source_record_id="01",
            license_id="CC-BY-4.0",
            raw_sha256="h1",
            normalized_sha256="h2",
        ),
        scenario="Architectural scenario context for system design.",
        task_type=TaskType.ADR_REASONING,
        review=ReviewInfo(status=ReviewStatus.APPROVED),
    )
    sample2 = ArchitectAISample(
        id="s2",
        source=SourceMetadata(
            source_id="opendatahub_adr",
            source_name="ODH",
            source_file_path="02.md",
            source_record_id="02",
            license_id="Apache-2.0",
            raw_sha256="h3",
            normalized_sha256="h4",
        ),
        scenario="Architectural scenario context for system design.",
        task_type=TaskType.ADR_REASONING,
        review=ReviewInfo(status=ReviewStatus.APPROVED),
    )

    rep = reporter.generate_report(
        train_samples=[sample1],
        val_samples=[sample2],
        eval_benchmark_counts={"sake": 10, "cake": 10, "archbench": 10, "r2abench": 10},
        has_contamination=False,
        gold_reviewed_count=35,
        manual_review_completed=True,
        export_validation_result={
            "duplicate_sample_ids": 0,
            "conflicting_duplicate_sample_ids": 0,
            "template_leakage_count": 0,
            "unresolved_placeholder_count": 0,
            "empty_assistant_answers": 0,
            "semantic_failures_exported": 0,
            "group_overlap_count": 0,
        },
        readiness_policy={
            "min_gold_samples": 30,
            "min_eval_samples": 30,
            "max_single_source_ratio": 0.80,
            "require_manual_review": True,
        },
    )

    assert rep["readiness_status"] == "TRAINING_READY"
    assert rep["manual_review_completed"] is True
    assert len(rep["blocking_reasons"]) == 0
    assert len(rep["warnings"]) == 0


# 3. Grounding Rules & No Synthetic Scaling Driver
def test_sft_formatter_no_synthetic_scaling_driver():
    formatter = SFTFormatter()

    sample_no_driver = ArchitectAISample(
        id="s_scale_1",
        source=SourceMetadata(
            source_id="k8s_keps",
            source_name="KEP",
            source_file_path="k1.md",
            source_record_id="k1",
            license_id="Apache-2.0",
            raw_sha256="ha",
            normalized_sha256="hb",
        ),
        scenario="We need to partition data across clusters.",
        task_type=TaskType.SCALING_REASONING,
        decisions=[EvidenceItem(value="Implement database sharding and horizontal pod autoscaler", evidence_type=EvidenceType.EXPLICIT)],
    )

    resp = formatter.compose_grounded_response(sample_no_driver)
    assert "Workload growth, high throughput, or capacity requirement." not in resp
    assert "Architectural Response:" in resp


# 4. Semantic Quality & Taxonomy Tightening
def test_taxonomy_tightening_concrete_vs_generic():
    classifier = TaskTaxonomyClassifier()

    # Generic false positive
    generic_sample = {
        "raw_text": "This framework supports plugins for modularity.",
        "options": [],
        "context": "This framework supports plugins for modularity.",
    }
    assert classifier.classify(generic_sample) != TaskType.TECHNOLOGY_SELECTION

    # Real concrete selection
    concrete_sample = {
        "raw_text": "We benchmarked PostgreSQL vs MongoDB for storage.",
        "options": ["PostgreSQL", "MongoDB"],
        "context": "We benchmarked PostgreSQL vs MongoDB for storage.",
    }
    assert classifier.classify(concrete_sample) == TaskType.TECHNOLOGY_SELECTION


def test_semantic_quality_contracts():
    validator = SemanticQualityValidator()

    # ADR without context fails
    no_ctx_sample = ArchitectAISample(
        id="adr_1",
        source=SourceMetadata(
            source_id="madr",
            source_name="MADR",
            source_file_path="m.md",
            source_record_id="m",
            license_id="MIT",
            raw_sha256="h",
            normalized_sha256="h",
        ),
        scenario="Too short",
        task_type=TaskType.ADR_REASONING,
        decisions=[EvidenceItem(value="Use REST API", evidence_type=EvidenceType.EXPLICIT)],
    )
    res_no_ctx = validator.validate_sample(no_ctx_sample)
    assert res_no_ctx.passed is False
