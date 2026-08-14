"""
Spotify Backstage Architecture Decision Records Parser with Source-Aware Whitelisting & Synonym Mapping
"""

import re
from pathlib import Path
from typing import Any

from architectai_dataset_builder.parsers.base import BaseParser
from architectai_dataset_builder.utils.hashing import compute_sha256_file
from architectai_dataset_builder.utils.identity import generate_stable_sample_id
from architectai_dataset_builder.utils.markdown import (
    ALTERNATIVE_SYNONYMS,
    CONSEQUENCE_SYNONYMS,
    CONTEXT_SYNONYMS,
    DECISION_SYNONYMS,
    extract_markdown_section,
    has_template_placeholders,
    is_boilerplate_filename,
)


class BackstageADRParser(BaseParser):
    def __init__(self) -> None:
        super().__init__(source_id="backstage_adrs")

    def parse_directory(self, raw_dir: Path) -> list[dict[str, Any]]:
        records = []
        for file_path in sorted(raw_dir.rglob("*.md")):
            if not file_path.is_file() or file_path.name.startswith("."):
                continue

            if is_boilerplate_filename(file_path, self.source_id):
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

        record_id = file_path.stem
        rel_path = file_path.relative_to(raw_dir).as_posix()
        sample_id = generate_stable_sample_id(
            source_id=self.source_id,
            file_path=rel_path,
            record_id=record_id,
        )

        # 1. Quarantine unresolved template placeholders
        if has_template_placeholders(text):
            return {
                "sample_id": sample_id,
                "source_id": self.source_id,
                "file_name": file_path.name,
                "record_id": record_id,
                "raw_sha256": raw_hash,
                "is_quarantined": True,
                "quarantine_reason": "unresolved_template_placeholder",
                "raw_text": text,
            }

        title_match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
        title = title_match.group(1).strip() if title_match else file_path.stem

        context = extract_markdown_section(text, CONTEXT_SYNONYMS)
        decision = extract_markdown_section(text, DECISION_SYNONYMS)
        consequences = extract_markdown_section(text, CONSEQUENCE_SYNONYMS)
        alternatives = extract_markdown_section(text, ALTERNATIVE_SYNONYMS)

        # 2. Strict grounding: Require genuine decision and context section
        is_decision_valid = bool(decision) and decision.lower() != "not explicitly stated" and len(decision.strip()) >= 15
        is_context_valid = bool(context) and len(context.strip()) >= 30

        if not is_decision_valid or not is_context_valid:
            return {
                "sample_id": sample_id,
                "source_id": self.source_id,
                "file_name": file_path.name,
                "record_id": record_id,
                "raw_sha256": raw_hash,
                "is_quarantined": True,
                "quarantine_reason": "missing_decision_section",
                "raw_text": text,
            }

        return {
            "sample_id": sample_id,
            "source_id": self.source_id,
            "file_name": file_path.name,
            "record_id": record_id,
            "raw_sha256": raw_hash,
            "title": title,
            "context": context,
            "decision": decision,
            "decision_outcome": decision,
            "consequences": [c.strip("- *") for c in consequences.split("\n") if c.strip("- *")],
            "alternatives": [a.strip("- *") for a in alternatives.split("\n") if a.strip("- *")],
            "raw_text": text,
            "is_quarantined": False,
        }
