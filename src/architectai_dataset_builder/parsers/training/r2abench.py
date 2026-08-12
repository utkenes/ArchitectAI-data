"""
R2ABench Requirements and Architecture Diagram Parser (Dynamic Auto-Discovery)
"""

from pathlib import Path
from typing import Any
from architectai_dataset_builder.parsers.base import BaseParser
from architectai_dataset_builder.utils.hashing import compute_sha256_file
from architectai_dataset_builder.utils.identity import generate_stable_sample_id


class R2ABenchParser(BaseParser):
    def __init__(self) -> None:
        super().__init__(source_id="r2abench")

    def parse_directory(self, raw_dir: Path) -> list[dict[str, Any]]:
        records = []
        # Dynamic auto-discovery: discover all txt/req files in raw_dir recursively
        req_files = sorted(raw_dir.rglob("*_req.txt")) + sorted(raw_dir.rglob("*.txt"))
        
        seen_projects = set()

        for req_file in req_files:
            if not req_file.is_file() or req_file.name.startswith("."):
                continue
                
            project_id = req_file.stem.replace("_req", "")
            if project_id in seen_projects:
                continue
            seen_projects.add(project_id)

            # Find matching PlantUML architecture diagram dynamically
            arch_file = raw_dir / f"{project_id}_arch.puml"
            if not arch_file.exists():
                arch_file = raw_dir / f"{project_id}.puml"

            arch_text = arch_file.read_text(encoding="utf-8", errors="ignore") if arch_file.exists() else ""
            req_text = req_file.read_text(encoding="utf-8", errors="ignore")
            raw_hash = compute_sha256_file(req_file)

            sample_id = generate_stable_sample_id(
                source_id=self.source_id,
                file_path=req_file.name,
                record_id=project_id,
                project_id=project_id,
            )

            records.append(
                {
                    "sample_id": sample_id,
                    "source_id": self.source_id,
                    "file_name": req_file.name,
                    "record_id": project_id,
                    "project_id": project_id,
                    "raw_sha256": raw_hash,
                    "requirements_text": req_text,
                    "plantuml_text": arch_text,
                    "raw_text": f"Requirements:\n{req_text}\n\nArchitecture:\n{arch_text}",
                    "is_quarantined": False,
                }
            )
        return records
