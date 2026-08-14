"""
R2ABench Requirements and Architecture Diagram Parser with Developer Test Fixture Excluder & Unified Discovery
"""

from pathlib import Path
from typing import Any

from architectai_dataset_builder.parsers.base import BaseParser
from architectai_dataset_builder.utils.hashing import compute_sha256_file
from architectai_dataset_builder.utils.identity import generate_stable_sample_id
from architectai_dataset_builder.utils.r2abench_discovery import discover_r2abench_projects

FIXTURE_PROJECT_IDS = {
    "proj_001_ecommerce",
    "proj_001",
}


class R2ABenchParser(BaseParser):
    def __init__(self) -> None:
        super().__init__(source_id="r2abench")

    def parse_directory(self, raw_dir: Path) -> list[dict[str, Any]]:
        records = []
        discovered_projects = discover_r2abench_projects(raw_dir)

        for pid, proj_files in sorted(discovered_projects.items()):
            req_file = proj_files.requirements_path
            arch_file = proj_files.architecture_path

            raw_hash = compute_sha256_file(req_file)

            # Exclude developer test fixtures from training corpus
            if pid in FIXTURE_PROJECT_IDS or req_file.name.startswith("proj_001"):
                records.append(
                    {
                        "sample_id": f"quarantine_{pid}",
                        "source_id": self.source_id,
                        "file_name": req_file.name,
                        "record_id": pid,
                        "project_id": pid,
                        "raw_sha256": raw_hash,
                        "is_quarantined": True,
                        "quarantine_reason": "developer_test_fixture",
                        "raw_text": req_file.read_text(encoding="utf-8", errors="ignore"),
                    }
                )
                continue

            arch_text = arch_file.read_text(encoding="utf-8", errors="ignore") if arch_file and arch_file.exists() else ""
            req_text = req_file.read_text(encoding="utf-8", errors="ignore")

            rel_file_path = (
                req_file.relative_to(raw_dir).as_posix()
                if req_file.is_relative_to(raw_dir)
                else req_file.name
            )

            sample_id = generate_stable_sample_id(
                source_id=self.source_id,
                file_path=rel_file_path,
                record_id=pid,
                project_id=pid,
            )

            records.append(
                {
                    "sample_id": sample_id,
                    "source_id": self.source_id,
                    "file_name": req_file.name,
                    "record_id": pid,
                    "project_id": pid,
                    "raw_sha256": raw_hash,
                    "requirements_text": req_text,
                    "plantuml_text": arch_text,
                    "raw_text": f"Requirements:\n{req_text}\n\nArchitecture:\n{arch_text}",
                    "is_quarantined": False,
                }
            )
        return records
