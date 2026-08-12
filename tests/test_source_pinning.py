from architectai_dataset_builder.config import Config
from architectai_dataset_builder.sources.registry import SourceRegistry


def test_source_manifest_pinning():
    cfg = Config()
    registry = SourceRegistry(cfg.manifests_dir)

    manifest = registry.get_manifest("opendatahub_adr")
    assert manifest is not None
    assert manifest.version.commit_sha is not None
    assert manifest.version.revision == "main"
    assert manifest.license.spdx_id == "Apache-2.0"
    assert manifest.license.verified is True
