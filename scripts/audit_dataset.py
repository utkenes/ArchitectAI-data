"""
Script to audit build manifest, dataset statistics, and zero contamination
"""

import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from architectai_dataset_builder.config import Config
from architectai_dataset_builder.utils.io import read_jsonl


def main():
    cfg = Config()
    export_dir = cfg.data_dir / "exports"

    manifest_file = export_dir / "build_manifest.json"
    stats_file = export_dir / "dataset_stats.json"
    contamination_file = export_dir / "contamination_report.json"

    if not manifest_file.exists():
        print(f"Error: Build manifest not found at {manifest_file}. Run build script first.")
        sys.exit(1)

    manifest_data = read_jsonl(manifest_file)[0]
    stats_data = read_jsonl(stats_file)[0]
    contamination_data = read_jsonl(contamination_file)[0]

    print("=== ArchitectAI Dataset Audit Report ===")
    print(f"Build ID:              {manifest_data['build_id']}")
    print(f"Build Timestamp:       {manifest_data['build_timestamp']}")
    print(f"Dataset Version:       {manifest_data['dataset_version']}")
    print(f"Total Samples:         {stats_data['total_samples']}")
    print(f"Split Summary:         {stats_data['sample_count_by_split']}")
    print(f"Task Distribution:     {stats_data['sample_count_by_task_type']}")
    print(f"Contamination Status:  {'FAILED' if contamination_data['has_leakage'] else 'PASSED (0 Leaks)'}")

    if contamination_data["has_leakage"]:
        print(f"CRITICAL ERROR: {contamination_data['total_leaks_detected']} leaks detected!")
        sys.exit(1)
    else:
        print("Audit PASSED cleanly.")


if __name__ == "__main__":
    main()
