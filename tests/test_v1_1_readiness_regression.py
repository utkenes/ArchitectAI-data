"""
Comprehensive V1.1 Readiness Regression Suite
"""

import pytest

from architectai_dataset_builder.models.canonical import (
    ArchitectAISample,
    ReviewInfo,
    SourceMetadata,
    TaskType,
)
from architectai_dataset_builder.models.evidence import EvidenceItem, EvidenceType
from architectai_dataset_builder.normalizers.task_taxonomy import TaskTaxonomyClassifier
from architectai_dataset_builder.reports.readiness_reporter import ReadinessReporter
from architectai_dataset_builder.utils.identity import generate_stable_sample_id
from architectai_dataset_builder.utils.markdown import (
    extract_markdown_section,
    extract_structured_items,
    sanitize_markdown,
)
from architectai_dataset_builder.validators.identity_validator import (
    IdentityCollisionError,
    IdentityValidator,
)
from architectai_dataset_builder.validators.semantic_quality import SemanticQualityValidator


# 1. Identity Regressions
def test_identity_determinism_and_path_differentiation():
    id_foo = generate_stable_sample_id("backstage_adrs", "foo/README.md", "README")
    id_bar = generate_stable_sample_id("backstage_adrs", "bar/README.md", "README")
    assert id_foo != id_bar

    # Repeat build yields identical ID
    id_foo_repeat = generate_stable_sample_id("backstage_adrs", "foo/README.md", "README")
    assert id_foo == id_foo_repeat


def test_identity_collision_blocks_build():
    validator = IdentityValidator()
    validator.register_and_validate("id_1", "hash_a", "madr", "01.md", "01")
    with pytest.raises(IdentityCollisionError):
        validator.register_and_validate("id_1", "hash_b", "madr", "02.md", "02")


# 2. Sanitization Regressions
def test_sanitization_removes_html_comments():
    raw = "# Title\n<!-- template guideline -->\nProse description."
    clean = sanitize_markdown(raw)
    assert "<!--" not in clean
    assert "template guideline" not in clean
    assert "Prose description." in clean


# 3. Section Parsing Regressions
def test_heading_level_aware_parsing():
    md = "### Alternatives\nOpt 1 is selected as primary option.\nOpt 2 is secondary backup option.\n### Risks\nRisk 1: High latency."
    sec = extract_markdown_section(md, ["alternatives"])
    assert "Opt 1" in sec
    assert "Risks" not in sec


def test_structured_item_extraction():
    text = "- Option 1\n- Option 2"
    items = extract_structured_items(text)
    assert items == ["Option 1", "Option 2"]


# 4. Taxonomy Regressions
def test_taxonomy_evidence_contracts():
    classifier = TaskTaxonomyClassifier()

    # False scaling keyword
    false_scaling = {
        "context": "APIs allow ecosystems to scale quickly.",
        "decision": "Use REST standard.",
        "raw_text": "APIs allow ecosystems to scale quickly. Decision: Use REST standard.",
    }
    assert classifier.classify(false_scaling) != TaskType.SCALING_REASONING

    # Real scaling
    real_scaling = {
        "context": "High workload growth created partition pressure.",
        "decision": "Use database sharding and horizontal pod autoscaler for high throughput.",
        "raw_text": "High workload growth created partition pressure. Decision: Use database sharding and horizontal pod autoscaler for high throughput.",
    }
    assert classifier.classify(real_scaling) == TaskType.SCALING_REASONING


# 5. Semantic Gate Regressions
def test_semantic_gate_checks():
    validator = SemanticQualityValidator()
    good_sample = ArchitectAISample(
        id="arch_good",
        source=SourceMetadata(
            source_id="madr",
            source_name="MADR",
            source_file_path="01.md",
            source_record_id="01",
            license_id="CC-BY-4.0",
            raw_sha256="h1",
            normalized_sha256="h2",
        ),
        scenario="We need high availability storage.",
        task_type=TaskType.ADR_REASONING,
        decisions=[EvidenceItem(value="Use PostgreSQL cluster", evidence_type=EvidenceType.EXPLICIT)],
        review=ReviewInfo(),
    )
    assert validator.validate_sample(good_sample).passed is True


# 6. Readiness Reporting Regressions
def test_readiness_status_transitions():
    reporter = ReadinessReporter()
    sample = ArchitectAISample(
        id="arch_r1",
        source=SourceMetadata(
            source_id="madr",
            source_name="MADR",
            source_file_path="01.md",
            source_record_id="01",
            license_id="CC-BY-4.0",
            raw_sha256="h1",
            normalized_sha256="h2",
        ),
        scenario="Scenario text",
        task_type=TaskType.ADR_REASONING,
        review=ReviewInfo(),
    )

    # Contamination causes BUILD_INVALID
    rep_invalid = reporter.generate_report(
        train_samples=[sample],
        val_samples=[],
        eval_benchmark_counts={"sake": 10},
        has_contamination=True,
    )
    assert rep_invalid["readiness_status"] == "BUILD_INVALID"

    # Missing manual review / gold threshold causes BUILD_VALID_REVIEW_REQUIRED
    rep_review = reporter.generate_report(
        train_samples=[sample],
        val_samples=[],
        eval_benchmark_counts={"sake": 10},
        has_contamination=False,
        gold_reviewed_count=0,
    )
    assert rep_review["readiness_status"] == "BUILD_VALID_REVIEW_REQUIRED"
