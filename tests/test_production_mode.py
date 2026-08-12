import pytest
from pathlib import Path
from architectai_dataset_builder.sources.downloader import SourceDownloader, ProductionSourceUnavailableError
from architectai_dataset_builder.sources.registry import SourceRegistry
from architectai_dataset_builder.models.manifest import SourceManifest, SourceOrigin, SourceVersion, LicenseMetadata, SourcePolicy


def test_production_mode_unavailable_error(tmp_path: Path):
    manifest = SourceManifest(
        source_id="invalid_repo",
        name="Invalid Repo",
        source_type="adr_repository",
        origin=SourceOrigin(provider="github", repository_url="https://github.com/nonexistent/invalid-repo-12345"),
        version=SourceVersion(revision="main"),
        license=LicenseMetadata(spdx_id="Apache-2.0", verified=True),
        policy=SourcePolicy(training_allowed=True),
    )

    registry = SourceRegistry(tmp_path)
    registry.source_manifests["invalid_repo"] = manifest

    downloader = SourceDownloader(tmp_path, registry)

    with pytest.raises(ProductionSourceUnavailableError):
        downloader.fetch_source("invalid_repo", mode="production")
