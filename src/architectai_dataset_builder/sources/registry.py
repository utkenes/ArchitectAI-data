"""
Source Manifest Registry & License Verification Engine
"""

from pathlib import Path
from typing import Dict, List, Optional
from architectai_dataset_builder.models.manifest import SourceManifest, LicenseMetadata, LicensePolicy
from architectai_dataset_builder.utils.io import load_yaml


class SourceRegistry:
    def __init__(self, manifests_dir: Path):
        self.manifests_dir = Path(manifests_dir)
        self.sources_dir = self.manifests_dir / "sources"
        self.licenses_file = self.manifests_dir / "licenses" / "license_registry.yaml"
        
        self.license_registry: Dict[str, LicenseMetadata] = self._load_license_registry()
        self.source_manifests: Dict[str, SourceManifest] = self._load_source_manifests()

    def _load_license_registry(self) -> Dict[str, LicenseMetadata]:
        if not self.licenses_file.exists():
            return {}
        data = load_yaml(self.licenses_file)
        licenses = {}
        for spdx_id, raw_lic in data.get("licenses", {}).items():
            licenses[spdx_id] = LicenseMetadata(
                spdx_id=raw_lic.get("spdx_id", spdx_id),
                verified=raw_lic.get("verified", False),
                verification_source=raw_lic.get("verification_source"),
                policy=LicensePolicy(**raw_lic.get("policy", {})),
            )
        return licenses

    def _load_source_manifests(self) -> Dict[str, SourceManifest]:
        manifests = {}
        if not self.sources_dir.exists():
            return manifests
        for path in self.sources_dir.glob("*.yaml"):
            data = load_yaml(path)
            manifest = SourceManifest(**data)
            # Cross-reference with license registry policy if present
            lic_spdx = manifest.license.spdx_id
            if lic_spdx in self.license_registry:
                registry_lic = self.license_registry[lic_spdx]
                manifest.license.verified = registry_lic.verified
                manifest.license.policy = registry_lic.policy
            manifests[manifest.source_id] = manifest
        return manifests

    def get_manifest(self, source_id: str) -> Optional[SourceManifest]:
        return self.source_manifests.get(source_id)

    def list_sources(self) -> List[SourceManifest]:
        return list(self.source_manifests.values())

    def is_training_allowed(self, source_id: str) -> bool:
        manifest = self.get_manifest(source_id)
        if not manifest:
            return False
        if not manifest.policy.training_allowed:
            return False
        if not manifest.license.verified:
            return False
        return manifest.license.policy.training_allowed
