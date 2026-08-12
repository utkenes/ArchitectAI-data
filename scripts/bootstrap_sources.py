"""
Script to bootstrap and verify raw source fixtures in data/raw/
"""

import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from architectai_dataset_builder.config import Config
from architectai_dataset_builder.sources.downloader import SourceDownloader
from architectai_dataset_builder.sources.registry import SourceRegistry


def main() -> None:
    cfg = Config()
    registry = SourceRegistry(cfg.manifests_dir)
    downloader = SourceDownloader(cfg.data_dir, registry)

    print("Bootstrapping raw architecture sources...")
    sources = ["opendatahub_adr", "madr", "r2abench", "sake", "cake", "archbench"]
    for sid in sources:
        dest_dir = downloader.fetch_source(sid)
        print(f" [OK] {sid:<18} -> {dest_dir}")

    print("All raw sources successfully bootstrapped and SHA-256 verified.")


if __name__ == "__main__":
    main()
