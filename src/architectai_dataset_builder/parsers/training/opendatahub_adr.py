"""
OpenDataHub Architecture Decision Record (ADR) Parser
"""

import re
from pathlib import Path
from typing import Any
from architectai_dataset_builder.parsers.base import BaseParser
from architectai_dataset_builder.utils.hashing import compute_sha256_file
from architectai_dataset_builder.utils.identity import generate_stable_sample_id

BOILERPLATE_FILENAMES = {
    "readme.md",
    "skill.md",
    "adr-guide.md",
    "odh-adr-0000-template.md",
    "template.md",
    "contributing.md",
}


class OpenDataHubADRParser(BaseParser):
    def __init__(self) -> None:
        super().__init__(source_id="opendatahub_adr")

    def parse_directory(self, raw_dir: Path) -> list[dict[str, Any]]:
        records = []
        for file_path in sorted(raw_dir.rglob("*.md")):
            if not file_path.is_file() or file_path.name.startswith("."):
                continue

            if file_path.name.lower() in BOILERPLATE_FILENAMES:
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

            record = self._parse_file(file_path)
            if record:
                records.append(record)
        return records

    def _parse_file(self, file_path: Path) -> dict[str, Any]:
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        raw_hash = compute_sha256_file(file_path)

        title_match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
        title = title_match.group(1).strip() if title_match else file_path.stem

        status = self._extract_section(text, r"## Status", r"##")
        context = self._extract_section(text, r"## Context", r"##")
        alternatives_raw = self._extract_section(text, r"## Alternatives Considered", r"##")
        decision = self._extract_section(text, r"## Decision", r"##")
        rationale = self._extract_section(text, r"## Rationale", r"##")

        record_id = file_path.stem
        sample_id = generate_stable_sample_id(
            source_id=self.source_id,
            file_path=file_path.name,
            record_id=record_id,
        )

        return {
            "sample_id": sample_id,
            "source_id": self.source_id,
            "file_name": file_path.name,
            "record_id": record_id,
            "raw_sha256": raw_hash,
            "title": title,
            "status": status or "Accepted",
            "context": context or title,
            "alternatives": [a.strip("- *") for a in alternatives_raw.split("\n") if a.strip("- *")],
            "decision": decision or "Not explicitly stated",
            "rationale": rationale or decision,
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
