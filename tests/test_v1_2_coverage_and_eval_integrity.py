"""
Comprehensive V1.2 Evaluation Integrity, Coverage & Balance Regressions
"""

from pathlib import Path

import pytest

from architectai_dataset_builder.exporters.gold_seed_exporter import GoldSeedExporter
from architectai_dataset_builder.exporters.training_balancer import TrainingBalancer
from architectai_dataset_builder.models.canonical import (
    ArchitectAISample,
    SourceMetadata,
    TaskType,
)
from architectai_dataset_builder.models.evidence import EvidenceItem, EvidenceType
from architectai_dataset_builder.normalizers.task_taxonomy import TaskTaxonomyClassifier
from architectai_dataset_builder.reports.stats_generator import StatsGenerator
from architectai_dataset_builder.splitters.deterministic_splitter import (
    DeterministicSplitter,
    SplitManifestValidationError,
)
from architectai_dataset_builder.utils.io import save_yaml
from architectai_dataset_builder.utils.r2abench_discovery import (
    discover_r2abench_projects,
    get_r2abench_discovery_diagnostics,
)


# 1. R2ABench Split Manifest Invariants
def test_real_r2abench_split_manifest_invariants():
    manifest_path = Path("manifests/splits/r2abench_v1.yaml")
    assert manifest_path.exists(), "r2abench_v1.yaml manifest must exist!"

    splitter = DeterministicSplitter(manifest_path)
    train_pids = set(splitter.r2abench_splits["training"])
    val_pids = set(splitter.r2abench_splits["validation"])
    holdout_pids = set(splitter.r2abench_splits["held_out"])

    assert len(train_pids) == 5
    assert len(val_pids) == 2
    assert len(holdout_pids) == 3

    assert len(train_pids & val_pids) == 0
    assert len(train_pids & holdout_pids) == 0
    assert len(val_pids & holdout_pids) == 0


def test_split_manifest_validation_failures(tmp_path: Path):
    bad_manifest = tmp_path / "bad_splits.yaml"

    # Empty heldout list
    save_yaml({"train_projects": ["p1"], "validation_projects": ["p2"], "held_out_projects": []}, bad_manifest)
    with pytest.raises(SplitManifestValidationError, match="held-out project list is empty"):
        DeterministicSplitter(bad_manifest)

    # Overlapping project
    save_yaml({"train_projects": ["p1", "p2"], "validation_projects": ["p3"], "held_out_projects": ["p2"]}, bad_manifest)
    with pytest.raises(SplitManifestValidationError, match="duplicate project IDs"):
        DeterministicSplitter(bad_manifest)


def test_held_out_and_unknown_project_isolation(tmp_path: Path):
    manifest_path = tmp_path / "splits.yaml"
    save_yaml(
        {"train_projects": ["proj_tr1"], "validation_projects": ["proj_val1"], "held_out_projects": ["proj_hold1"]},
        manifest_path,
    )

    splitter = DeterministicSplitter(manifest_path)

    sample_heldout = ArchitectAISample(
        id="s_h1",
        source=SourceMetadata(
            source_id="r2abench",
            source_name="R2A",
            source_file_path="f.txt",
            source_record_id="proj_hold1",
            project_id="proj_hold1",
            license_id="CC-BY-4.0",
            raw_sha256="h1",
            normalized_sha256="h2",
        ),
        scenario="Scenario text",
        task_type=TaskType.ARCHITECTURE_GENERATION,
    )

    with pytest.raises(SplitManifestValidationError, match="Held-out R2ABench project 'proj_hold1' attempted"):
        splitter.split_samples([sample_heldout])

    sample_unknown = ArchitectAISample(
        id="s_unk",
        source=SourceMetadata(
            source_id="r2abench",
            source_name="R2A",
            source_file_path="f.txt",
            source_record_id="proj_unk99",
            project_id="proj_unk99",
            license_id="CC-BY-4.0",
            raw_sha256="h1",
            normalized_sha256="h2",
        ),
        scenario="Scenario text",
        task_type=TaskType.ARCHITECTURE_GENERATION,
    )

    with pytest.raises(SplitManifestValidationError, match="Unknown R2ABench project ID 'proj_unk99'"):
        splitter.split_samples([sample_unknown])


# 2. Unified Recursive R2ABench File Discovery
def test_r2abench_nested_discovery(tmp_path: Path):
    proj_dir = tmp_path / "nested_sub" / "proj_100_payments"
    proj_dir.mkdir(parents=True)

    req_file = proj_dir / "proj_100_payments_req.txt"
    arch_file = proj_dir / "proj_100_payments_arch.puml"

    req_file.write_text("Payment gateway requirements.", encoding="utf-8")
    arch_file.write_text("@startuml\nnode Payments\n@enduml", encoding="utf-8")

    discovered = discover_r2abench_projects(tmp_path)
    assert "proj_100_payments" in discovered
    pinfo = discovered["proj_100_payments"]
    assert pinfo.requirements_path == req_file
    assert pinfo.architecture_path == arch_file

    diag = get_r2abench_discovery_diagnostics(tmp_path, ["proj_100_payments", "proj_missing"])
    assert diag["heldout_found"] == ["proj_100_payments"]
    assert diag["heldout_missing"] == ["proj_missing"]


# 3. Tradeoff Recovery & Precedence Safety
def test_tradeoff_recovery_and_precedence_safety():
    classifier = TaskTaxonomyClassifier()

    # Tech selection with pros/cons MUST remain TECHNOLOGY_SELECTION (precedence safe)
    tech_rec = {
        "raw_text": "We benchmarked PostgreSQL vs MongoDB for database storage. PostgreSQL pros: ACID compliant. MongoDB pros: flexible schema.",
        "options": ["PostgreSQL", "MongoDB"],
        "context": "We benchmarked PostgreSQL vs MongoDB for database storage.",
        "positive_consequences": ["ACID compliant", "Flexible schema"],
    }
    assert classifier.classify(tech_rec) == TaskType.TECHNOLOGY_SELECTION

    # Pure tradeoff analysis record
    tradeoff_rec = {
        "raw_text": "Evaluating architectural trade-offs between monolithic deployment and microservices. Consequences and trade-offs:",
        "options": ["Monolith", "Microservices"],
        "tradeoffs": ["High operational complexity", "Independent deployment"],
        "context": "Evaluating architectural trade-offs between monolithic deployment and microservices.",
    }
    assert classifier.classify(tradeoff_rec) == TaskType.TRADEOFF_ANALYSIS


# 4. GoldSeedExporter V2 Stratified Pool
def test_gold_seed_exporter_v2_stratified(tmp_path: Path):
    exporter = GoldSeedExporter(tmp_path)

    sample1 = ArchitectAISample(
        id="g1",
        source=SourceMetadata(
            source_id="madr",
            source_name="MADR",
            source_file_path="1.md",
            source_record_id="1",
            license_id="CC-BY-4.0",
            raw_sha256="h1",
            normalized_sha256="h2",
            group_id="group_g1",
        ),
        scenario="Architectural scenario description text here.",
        task_type=TaskType.ADR_REASONING,
        decisions=[EvidenceItem(value="Use gRPC for communications", evidence_type=EvidenceType.EXPLICIT)],
    )

    out_file = exporter.export_review_candidates([sample1], target_candidate_count=10)
    assert out_file.exists()


# 5. Training Balancer & Non-Blocking Strict Profile
def test_training_balancer_profiles(tmp_path: Path):
    balancer = TrainingBalancer(tmp_path)

    sample1 = ArchitectAISample(
        id="t1",
        source=SourceMetadata(
            source_id="k8s_keps",
            source_name="KEP",
            source_file_path="k1.md",
            source_record_id="k1",
            license_id="Apache-2.0",
            raw_sha256="h1",
            normalized_sha256="h2",
        ),
        scenario="Scenario text",
        task_type=TaskType.ADR_REASONING,
    )
    sample2 = ArchitectAISample(
        id="t2",
        source=SourceMetadata(
            source_id="opendatahub_adr",
            source_name="ODH",
            source_file_path="o1.md",
            source_record_id="o1",
            license_id="Apache-2.0",
            raw_sha256="h3",
            normalized_sha256="h4",
        ),
        scenario="Scenario text",
        task_type=TaskType.ADR_REASONING,
    )

    res = balancer.generate_profiles([sample1, sample2])
    assert "balanced_strict" in res["profiles"]
    assert (tmp_path / "train_sft_balanced.jsonl").exists()


# 6. Single Canonical Names for Duplicate Metrics
def test_single_canonical_duplicate_metric_names(tmp_path: Path):
    stats_gen = StatsGenerator()
    rep = stats_gen.generate_stats(
        train_samples=[],
        val_samples=[],
        exact_dups=5,
        near_dups=2,
        quarantine_count=0,
        quarantine_reasons={},
        failed_parse_count=0,
        source_mode="production",
        build_status="PRODUCTION_RELEASE",
        output_file=tmp_path / "dataset_stats.json",
    )
    rep_dict = rep.model_dump()
    assert rep_dict["exact_duplicate_count"] == 5
    assert rep_dict["near_duplicate_count"] == 2
    assert "duplicate_count" not in rep_dict  # Ensures no schema drift
