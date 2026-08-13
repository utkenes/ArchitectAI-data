"""
Kubernetes Enhancement Proposals (KEPs) Parser with Status Tracking & Template Quarantining
"""

import re
from pathlib import Path
from typing import Any
from architectai_dataset_builder.parsers.base import BaseParser
from architectai_dataset_builder.utils.hashing import compute_sha256_file
from architectai_dataset_builder.utils.identity import generate_stable_sample_id

BOILERPLATE_PATTERNS = [
    "nnnn-kep-template",
    "0000-kep-process",
    "readme.md",
    "template.md",
]


class KubernetesKEPParser(BaseParser):
    def __init__(self) -> None:
        super().__init__(source_id="k8s_keps")

    def parse_directory(self, raw_dir: Path) -> list[dict[str, Any]]:
        records = []
        for file_path in sorted(raw_dir.rglob("*.md")):
            if not file_path.is_file() or file_path.name.startswith("."):
                continue

            rel_path_lower = str(file_path.relative_to(raw_dir)).lower()
            if any(b in rel_path_lower for b in BOILERPLATE_PATTERNS):
                records.append(
                    {
                        "sample_id": f"quarantine_{file_path.stem}",
                        "source_id": self.source_id,
                        "file_name": file_path.name,
                        "record_id": file_path.stem,
                        "raw_sha256": compute_sha256_file(file_path),
                        "is_quarantined": True,
                        "quarantine_reason": "boilerplate_or_template",
                        "raw_text": file_path.read_text(encoding="utf-8", errors="ignore"),
                    }
                )
                continue

            record = self._parse_file(file_path, raw_dir)
            if record:
                records.append(record)
        return records

    def _parse_file(self, file_path: Path, raw_dir: Path) -> dict[str, Any]:
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        raw_hash = compute_sha256_file(file_path)

        # Extract title
        title_match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
        title = title_match.group(1).strip() if title_match else file_path.stem

        # Extract kep_status (YAML header status: implementable / implemented / provisional / rejected)
        status_match = re.search(r"^status:\s*(.+)$", text, re.IGNORECASE | re.MULTILINE)
        if not status_match:
            status_match = re.search(r"## Status\s*\n+([^\n#]+)", text, re.IGNORECASE)
        kep_status = status_match.group(1).strip().lower() if status_match else "provisional"

        # Extract key architectural sections
        summary = self._extract_section(text, r"## Summary", r"##")
        motivation = self._extract_section(text, r"## Motivation", r"##")
        proposal = self._extract_section(text, r"## Proposal", r"##")
        design_details = self._extract_section(text, r"## Design Details", r"##")
        alternatives = self._extract_section(text, r"## Alternatives", r"##")
        risks = self._extract_section(text, r"## Risks and Mitigations", r"##")

        # Determine sig or project_id from directory structure
        rel_parts = file_path.relative_to(raw_dir).parts
        sig_id = rel_parts[1] if len(rel_parts) > 2 else "sig-architecture"

        record_id = file_path.stem
        sample_id = generate_stable_sample_id(
            source_id=self.source_id,
            file_path=file_path.name,
            record_id=record_id,
            project_id=sig_id,
        )

        return {
            "sample_id": sample_id,
            "source_id": self.source_id,
            "file_name": file_path.name,
            "record_id": record_id,
            "project_id": sig_id,
            "raw_sha256": raw_hash,
            "title": title,
            "kep_status": kep_status,
            "summary": summary or title,
            "motivation": motivation,
            "proposal": proposal or design_details,
            "alternatives": [a.strip("- *") for a in alternatives.split("\n") if a.strip("- *")],
            "tradeoffs": [r.strip("- *") for r in risks.split("\n") if r.strip("- *")],
            "raw_text": text,
            "is_quarantined": False,
        }

    def _extract_section(self, text: str, header_regex: str, next_header_regex: str) -> str:
        match = re.search(header_regex, text, re.IGNORECASE)
        if not match:
            return ""
        start = match.end()
        remainder = text[start:]
        next_match = re.search(next_header_regex, remainder)
        if next_match:
            section_text = remainder[: next_match.start()]
        else:
            section_text = remainder
        return section_text.strip()
