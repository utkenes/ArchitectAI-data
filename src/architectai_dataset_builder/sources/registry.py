"""
Source Manifest Registry & License Verification Engine
"""

from pathlib import Path

from architectai_dataset_builder.models.manifest import (
    LicenseMetadata,
    LicensePolicy,
    SourceManifest,
)
from architectai_dataset_builder.utils.io import load_yaml


class SourceRegistry:
    def __init__(self, manifests_dir: Path) -> None:
        self.manifests_dir = Path(manifests_dir)
        self.sources_dir = self.manifests_dir / "sources"
        self.licenses_file = self.manifests_dir / "licenses" / "license_registry.yaml"

        self.license_registry: dict[str, LicenseMetadata] = self._load_license_registry()
        self.source_manifests: dict[str, SourceManifest] = self._load_source_manifests()

    def _load_license_registry(self) -> dict[str, LicenseMetadata]:
        if not self.licenses_file.exists():
            return {}
        data = load_yaml(self.licenses_file)
        licenses: dict[str, LicenseMetadata] = {}
        for spdx_id, raw_lic in data.get("licenses", {}).items():
            licenses[spdx_id] = LicenseMetadata(
                spdx_id=raw_lic.get("spdx_id", spdx_id),
                verified=raw_lic.get("verified", False),
                verification_source=raw_lic.get("verification_source"),
                policy=LicensePolicy(**raw_lic.get("policy", {})),
            )
        return licenses

    def _load_source_manifests(self) -> dict[str, SourceManifest]:
        manifests: dict[str, SourceManifest] = {}
        if not self.sources_dir.exists():
            return manifests
        for path in self.sources_dir.glob("*.yaml"):
            data = load_yaml(path)
            manifest = SourceManifest(**data)
            lic_spdx = manifest.license.spdx_id
            if lic_spdx in self.license_registry:
                registry_lic = self.license_registry[lic_spdx]
                manifest.license.verified = registry_lic.verified
                manifest.license.policy = registry_lic.policy
            manifests[manifest.source_id] = manifest
        return manifests

    def get_manifest(self, source_id: str) -> SourceManifest | None:
        return self.source_manifests.get(source_id)

    def list_sources(self) -> list[SourceManifest]:
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
