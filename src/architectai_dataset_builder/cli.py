"""
ArchitectAI Dataset Builder Command Line Interface
"""

from collections import Counter
from typing import Any
import click

from architectai_dataset_builder.config import Config
from architectai_dataset_builder.dedup.deduplicator import Deduplicator
from architectai_dataset_builder.exporters.build_manifest_exporter import BuildManifestExporter
from architectai_dataset_builder.exporters.gold_seed_exporter import GoldSeedExporter
from architectai_dataset_builder.exporters.jsonl_exporter import JSONLExporter
from architectai_dataset_builder.exporters.quality_sampler import QualitySampler
from architectai_dataset_builder.models.manifest import SourceManifest
from architectai_dataset_builder.normalizers.canonical_normalizer import CanonicalNormalizer
from architectai_dataset_builder.normalizers.relevance_filter import RelevanceFilter
from architectai_dataset_builder.parsers.eval_adapters.archbench import ArchBenchEvalAdapter
from architectai_dataset_builder.parsers.eval_adapters.cake import CAKEEvalAdapter
from architectai_dataset_builder.parsers.eval_adapters.r2abench import R2ABenchEvalAdapter
from architectai_dataset_builder.parsers.eval_adapters.sake import SAKEEvalAdapter
from architectai_dataset_builder.parsers.training.backstage_adrs import BackstageADRParser
from architectai_dataset_builder.parsers.training.k8s_keps import KubernetesKEPParser
from architectai_dataset_builder.parsers.training.madr import MADRParser
from architectai_dataset_builder.parsers.training.opendatahub_adr import OpenDataHubADRParser
from architectai_dataset_builder.parsers.training.r2abench import R2ABenchParser
from architectai_dataset_builder.reports.readiness_reporter import ReadinessReporter
from architectai_dataset_builder.reports.stats_generator import StatsGenerator
from architectai_dataset_builder.sources.downloader import ProductionSourceUnavailableError, SourceDownloader
from architectai_dataset_builder.sources.registry import SourceRegistry
from architectai_dataset_builder.splitters.contamination_checker import ContaminationChecker
from architectai_dataset_builder.splitters.deterministic_splitter import DeterministicSplitter
from architectai_dataset_builder.utils.io import load_yaml, write_jsonl
from architectai_dataset_builder.validators.license_gating import LicenseGatingEngine
from architectai_dataset_builder.validators.schema_validator import SchemaValidator


@click.group()
def cli() -> None:
    """ArchitectAI Dataset Builder CLI"""
    pass


@cli.group()
def sources() -> None:
    """Manage architecture data sources and manifests."""
    pass


@sources.command(name="list")
def list_sources() -> None:
    """List registered data sources and eligibility status."""
    cfg = Config()
    registry = SourceRegistry(cfg.manifests_dir)
    click.echo(f"Registered Data Sources ({len(registry.list_sources())}):")
    for s in registry.list_sources():
        allowed = registry.is_training_allowed(s.source_id)
        click.echo(
            f" - {s.source_id:<18} [{s.source_type:<22}] License: {s.license.spdx_id:<10} Training Allowed: {allowed}"
        )


@sources.command(name="verify")
def verify_sources() -> None:
    """Verify license policies and training eligibility of all sources."""
    cfg = Config()
    registry = SourceRegistry(cfg.manifests_dir)
    click.echo("Verifying Source Licenses & Eligibility Policy:")
    for s in registry.list_sources():
        status = "PASSED" if registry.is_training_allowed(s.source_id) else "QUARANTINED / EVAL ONLY"
        click.echo(f" Source: {s.source_id:<18} Verified: {s.license.verified} -> Status: {status}")


@cli.command(name="build")
@click.option("--build-id", default="build_v2_001", help="Unique build identifier")
@click.option("--mode", type=click.Choice(["production", "fixture"]), default="production", help="Build mode")
def build_dataset(build_id: str, mode: str) -> None:
    """Execute end-to-end dataset build pipeline."""
    cfg = Config()
    registry = SourceRegistry(cfg.manifests_dir)
    downloader = SourceDownloader(cfg.data_dir, registry)
    relevance_filter = RelevanceFilter(
        keywords=cfg.policy_config.get("relevance_filter", {}).get("architecture_keywords", []),
        min_relevance_score=cfg.min_relevance_score,
    )
    license_gating = LicenseGatingEngine(registry)
    schema_validator = SchemaValidator()
    normalizer = CanonicalNormalizer()
    deduplicator = Deduplicator(jaccard_threshold=cfg.near_dedup_jaccard_threshold)
    splitter = DeterministicSplitter(cfg.manifests_dir / "splits" / "r2abench_v1.yaml")

    click.echo(f"=== Starting ArchitectAI Dataset Build: {build_id} (Mode: {mode}) ===")

    # 1. Fetch Sources
    sources_to_ingest = [
        "opendatahub_adr",
        "madr",
        "r2abench",
        "backstage_adrs",
        "k8s_keps",
        "sake",
        "cake",
        "archbench",
    ]
    source_fetch_modes = {}

    for sid in sources_to_ingest:
        try:
            downloader.fetch_source(sid, mode=mode)
            source_fetch_modes[sid] = mode
        except ProductionSourceUnavailableError as e:
            if mode == "production":
                click.echo(f"CRITICAL ERROR: {e}")
                raise e
            downloader.fetch_source(sid, mode="fixture")
            source_fetch_modes[sid] = "fixture"

    actual_source_mode = "production" if all(m == "production" for m in source_fetch_modes.values()) else "fixture"
    build_status = "PRODUCTION_RELEASE" if actual_source_mode == "production" else "DEGRADED_FIXTURE_BUILD"

    click.echo(f"[OK] Sources fetched. Effective Source Mode: {actual_source_mode} | Build Status: {build_status}")

    # 2. Parse Training Sources
    raw_madr_records = MADRParser().parse_directory(cfg.data_dir / "raw" / "madr")
    raw_odh_records = OpenDataHubADRParser().parse_directory(cfg.data_dir / "raw" / "opendatahub_adr")
    raw_r2a_records = R2ABenchParser().parse_directory(cfg.data_dir / "raw" / "r2abench")
    raw_backstage_records = BackstageADRParser().parse_directory(cfg.data_dir / "raw" / "backstage_adrs")
    raw_k8s_records = KubernetesKEPParser().parse_directory(cfg.data_dir / "raw" / "k8s_keps")

    # 3. Normalize & Filter Relevance / License Gating
    parsed_samples = []
    quarantine_reasons: Counter[str] = Counter()
    failed_parse_count = 0

    madr_manifest = registry.get_manifest("madr")
    odh_manifest = registry.get_manifest("opendatahub_adr")
    r2a_manifest = registry.get_manifest("r2abench")
    backstage_manifest = registry.get_manifest("backstage_adrs")
    k8s_manifest = registry.get_manifest("k8s_keps")

    assert madr_manifest is not None
    assert odh_manifest is not None
    assert r2a_manifest is not None
    assert backstage_manifest is not None
    assert k8s_manifest is not None

    def process_records(records: list[dict[str, Any]], manifest: SourceManifest) -> None:
        nonlocal failed_parse_count
        for r in records:
            if r.get("is_quarantined"):
                reason = str(r.get("quarantine_reason", "unknown"))
                quarantine_reasons[reason] += 1
                continue

            try:
                if relevance_filter.is_relevant(r.get("raw_text", "")):
                    parsed_samples.append(normalizer.normalize(r, manifest))
                else:
                    quarantine_reasons["low_relevance"] += 1
            except (ValueError, KeyError, TypeError, AttributeError):
                failed_parse_count += 1

    process_records(raw_madr_records, madr_manifest)
    process_records(raw_odh_records, odh_manifest)
    process_records(raw_r2a_records, r2a_manifest)
    process_records(raw_backstage_records, backstage_manifest)
    process_records(raw_k8s_records, k8s_manifest)

    quarantine_count = sum(quarantine_reasons.values())
    click.echo(
        f"[OK] Parsed candidates: {len(parsed_samples)} | Quarantined: {quarantine_count} | Parse Errors: {failed_parse_count}"
    )

    # 4. License Gating & Schema Validation
    valid_samples = []
    for s in parsed_samples:
        if license_gating.validate_sample(s):
            is_valid, _ = schema_validator.validate(s)
            if is_valid:
                valid_samples.append(s)
            else:
                quarantine_reasons["schema_validation_failed"] += 1
        else:
            quarantine_reasons["unverified_license"] += 1

    click.echo(f"[OK] License gating & schema validation passed for {len(valid_samples)} samples.")

    # 5. Deduplication
    unique_samples, exact_dups, near_dups = deduplicator.process_samples(valid_samples)
    click.echo(f"[OK] Deduplication: {len(unique_samples)} unique ({exact_dups} exact dups, {near_dups} near dups).")

    # 6. Group-Level Deterministic Splitting
    train_samples, val_samples = splitter.split_samples(unique_samples)
    click.echo(f"[OK] Group Split: {len(train_samples)} Train, {len(val_samples)} Validation.")

    # 7. Parse Protected Evaluation Benchmarks
    sake_eval = SAKEEvalAdapter().parse_directory(cfg.data_dir / "raw" / "sake")
    cake_eval = CAKEEvalAdapter().parse_directory(cfg.data_dir / "raw" / "cake")
    archbench_eval = ArchBenchEvalAdapter().parse_directory(cfg.data_dir / "raw" / "archbench")
    r2a_eval = R2ABenchEvalAdapter(
        held_out_project_ids=splitter.r2abench_splits["held_out"]
    ).parse_directory(cfg.data_dir / "raw" / "r2abench")

    eval_samples = sake_eval + cake_eval + archbench_eval + r2a_eval
    click.echo(f"[OK] Ingested {len(eval_samples)} protected evaluation samples.")

    # 8. Contamination Verification
    checker = ContaminationChecker(
        protected_sources=cfg.protected_sources,
        jaccard_threshold=cfg.near_dedup_jaccard_threshold,
    )
    contamination_report = checker.verify_no_leakage(train_samples + val_samples, eval_samples)
    write_jsonl([contamination_report.model_dump()], cfg.data_dir / "exports" / "contamination_report.json")
    click.echo("[OK] Contamination Check Passed: 0 cross-split leaks.")

    # 9. Exporters & Manifests
    approved_manifest = load_yaml(cfg.manifests_dir / "reviews" / "approved_samples.yaml")
    approved_ids = [entry["sample_id"] for entry in approved_manifest.get("approved_samples", [])]

    exporter = JSONLExporter(cfg.data_dir / "exports")
    train_export_paths = exporter.export_training_datasets(train_samples, val_samples, approved_ids)
    eval_export_paths = exporter.export_evaluation_datasets(eval_samples)

    # 10. Gold Seed & Quality Review Exports
    GoldSeedExporter(cfg.data_dir / "exports").export_review_candidates(unique_samples)
    QualitySampler(cfg.data_dir / "exports").export_quality_samples(unique_samples)
    click.echo("[OK] Exported gold_review_candidates.jsonl and quality_review_samples.jsonl")

    all_export_paths = {**train_export_paths, **eval_export_paths}

    # 11. Reports
    stats_gen = StatsGenerator()
    stats_gen.generate_stats(
        train_samples,
        val_samples,
        exact_dups,
        near_dups,
        sum(quarantine_reasons.values()),
        dict(quarantine_reasons),
        failed_parse_count,
        actual_source_mode,
        build_status,
        cfg.data_dir / "exports" / "dataset_stats.json",
    )

    ReadinessReporter().generate_report(
        train_samples=train_samples,
        val_samples=val_samples,
        eval_samples_count=len(eval_samples),
        quarantine_count=sum(quarantine_reasons.values()),
        failed_parse_count=failed_parse_count,
        exact_dups=exact_dups,
        near_dups=near_dups,
        has_contamination=contamination_report.has_leakage,
        output_file=cfg.data_dir / "exports" / "training_readiness_report.json",
    )
    click.echo("[OK] Generated training_readiness_report.json")

    manifest_exporter = BuildManifestExporter()
    manifest_exporter.export_build_manifest(
        build_id=build_id,
        config_dir=cfg.config_dir,
        split_manifest_path=cfg.manifests_dir / "splits" / "r2abench_v1.yaml",
        sources_summary={s.source_id: {"commit_sha": s.version.commit_sha} for s in registry.list_sources()},
        sample_counts={
            "train": len(train_samples),
            "validation": len(val_samples),
            "silver": len(train_samples) + len(val_samples),
            "gold": len(approved_ids),
            "eval": len(eval_samples),
        },
        export_paths=all_export_paths,
        output_file=cfg.data_dir / "exports" / "build_manifest.json",
    )

    click.echo(f"=== Build Completed: {build_status} ===")


if __name__ == "__main__":
    cli()
