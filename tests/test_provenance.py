from architectai_dataset_builder.models.canonical import ArchitectAISample, SourceMetadata


def test_provenance_metadata_fields():
    meta = SourceMetadata(
        source_id="opendatahub_adr",
        source_name="Open Data Hub ADR",
        source_url="https://github.com/opendatahub-io/architecture-decision-records",
        source_version="main",
        source_commit_sha="9f3b1e84a2c01928374651928374651928374651",
        source_file_path="0001-pipeline-orchestration.md",
        source_record_id="0001-pipeline-orchestration",
        license_id="Apache-2.0",
        license_verified=True,
        raw_sha256="abc123raw",
        normalized_sha256="def456norm",
        created_at="2026-08-12T12:00:00Z",
    )

    sample = ArchitectAISample(
        id="arch_prov001",
        source=meta,
        scenario="Provenance test scenario",
    )

    assert sample.source.source_commit_sha == "9f3b1e84a2c01928374651928374651928374651"
    assert sample.source.raw_sha256 == "abc123raw"
    assert sample.source.normalized_sha256 == "def456norm"
