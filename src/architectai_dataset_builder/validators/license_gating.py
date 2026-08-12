"""
License Gating Engine
"""

from architectai_dataset_builder.models.canonical import ArchitectAISample
from architectai_dataset_builder.sources.registry import SourceRegistry


class LicenseGatingEngine:
    def __init__(self, registry: SourceRegistry):
        self.registry = registry

    def validate_sample(self, sample: ArchitectAISample) -> bool:
        source_id = sample.source.source_id
        manifest = self.registry.get_manifest(source_id)
        if not manifest:
            return False

        # 1. License Must Be Verified
        if not sample.source.license_verified:
            return False

        # 2. Source & License Policy Must Permit Training
        if not manifest.policy.training_allowed:
            return False
        if not manifest.license.policy.training_allowed:
            return False

        return True
