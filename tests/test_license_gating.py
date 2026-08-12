from architectai_dataset_builder.config import Config
from architectai_dataset_builder.models.canonical import ArchitectAISample, SourceMetadata
from architectai_dataset_builder.sources.registry import SourceRegistry
from architectai_dataset_builder.validators.license_gating import LicenseGatingEngine


def test_license_gating_verified():
    cfg = Config()
    registry = SourceRegistry(cfg.manifests_dir)
    gating = LicenseGatingEngine(registry)

    sample = ArchitectAISample(
        id="arch_test123",
        source=SourceMetadata(
            source_id="opendatahub_adr",
            source_name="ODH ADR",
            source_file_path="0001.md",
            source_record_id="rec01",
            license_id="Apache-2.0",
            license_verified=True,
            raw_sha256="abc",
            normalized_sha256="def",
            created_at="2026-08-12T12:00:00Z",
        ),
        scenario="Test scenario context for architecture decision",
    )

    assert gating.validate_sample(sample) is True


def test_license_gating_unverified_rejected():
    cfg = Config()
    registry = SourceRegistry(cfg.manifests_dir)
    gating = LicenseGatingEngine(registry)

    sample = ArchitectAISample(
        id="arch_unverified",
        source=SourceMetadata(
            source_id="opendatahub_adr",
            source_name="ODH ADR",
            source_file_path="0001.md",
            source_record_id="rec01",
            license_id="UNKNOWN",
            license_verified=False,  # Unverified!
            raw_sha256="abc",
            normalized_sha256="def",
            created_at="2026-08-12T12:00:00Z",
        ),
        scenario="Unverified license scenario",
    )

    assert gating.validate_sample(sample) is False
