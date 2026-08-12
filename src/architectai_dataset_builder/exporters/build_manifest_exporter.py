"""
Build Manifest Exporter
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any
from architectai_dataset_builder.models.manifest import BuildManifest
from architectai_dataset_builder.utils.hashing import compute_sha256_file, compute_sha256_str
from architectai_dataset_builder.utils.io import save_yaml, write_jsonl


class BuildManifestExporter:
    def export_build_manifest(
        self,
        build_id: str,
        config_dir: Path,
        split_manifest_path: Path,
        sources_summary: Dict[str, Dict[str, Any]],
        sample_counts: Dict[str, int],
        export_paths: Dict[str, Path],
        output_file: Path,
    ) -> BuildManifest:
        config_hash = compute_sha256_file(config_dir / "dataset_policy.yaml")
        split_hash = compute_sha256_file(split_manifest_path) if split_manifest_path.exists() else "none"

        export_hashes = {}
        for key, path in export_paths.items():
            if path.exists():
                export_hashes[f"{key}_sha256"] = compute_sha256_file(path)

        manifest = BuildManifest(
            dataset_version="1.0.0",
            builder_version="0.1.0",
            build_id=build_id,
            build_timestamp=datetime.now(timezone.utc).isoformat(),
            config_hash=config_hash,
            split_manifest_hash=split_hash,
            sources=sources_summary,
            sample_counts=sample_counts,
            export_hashes=export_hashes,
        )

        write_jsonl([manifest.model_dump()], output_file)
        return manifest
