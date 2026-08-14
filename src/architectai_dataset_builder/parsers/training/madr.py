"""
MADR (Markdown Architectural Decision Records) Parser with Section Synonym Mapping
"""

import re
from pathlib import Path
from typing import Any

from architectai_dataset_builder.parsers.base import BaseParser
from architectai_dataset_builder.utils.hashing import compute_sha256_file
from architectai_dataset_builder.utils.identity import generate_stable_sample_id
from architectai_dataset_builder.utils.markdown import (
    ALTERNATIVE_SYNONYMS,
    CONTEXT_SYNONYMS,
    DECISION_SYNONYMS,
    extract_markdown_section,
    extract_structured_items,
    has_template_placeholders,
    is_boilerplate_filename,
    sanitize_markdown,
)


class MADRParser(BaseParser):
    def __init__(self) -> None:
        super().__init__(source_id="madr")

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

        sanitized_text = sanitize_markdown(text)

        record_id = file_path.stem
        rel_path = file_path.relative_to(raw_dir).as_posix()
        sample_id = generate_stable_sample_id(
            source_id=self.source_id,
            file_path=rel_path,
            record_id=record_id,
        )

        # 1. Quarantine unresolved template placeholders post-sanitization
        if has_template_placeholders(sanitized_text):
            return {
                "sample_id": sample_id,
                "source_id": self.source_id,
                "file_name": file_path.name,
                "record_id": record_id,
                "raw_sha256": raw_hash,
                "is_quarantined": True,
                "quarantine_reason": "unresolved_template_placeholder",
                "raw_text": sanitized_text,
            }

        title_match = re.search(r"^#\s+(.+)$", sanitized_text, re.MULTILINE)
        title = title_match.group(1).strip() if title_match else file_path.stem

        context = extract_markdown_section(sanitized_text, CONTEXT_SYNONYMS)
        drivers_raw = extract_markdown_section(sanitized_text, ["decision drivers", "drivers"])
        options_raw = extract_markdown_section(sanitized_text, ALTERNATIVE_SYNONYMS)
        outcome_raw = extract_markdown_section(sanitized_text, DECISION_SYNONYMS)
        pos_consequences_raw = extract_markdown_section(sanitized_text, ["positive consequences", "pros"])
        neg_consequences_raw = extract_markdown_section(sanitized_text, ["negative consequences", "cons"])

        # 2. Strict grounding: Require genuine decision outcome and context section
        is_decision_valid = bool(outcome_raw) and outcome_raw.lower() != "not explicitly stated" and len(outcome_raw.strip()) >= 15
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
                "raw_text": sanitized_text,
            }

        drivers = extract_structured_items(drivers_raw)
        options = extract_structured_items(options_raw)
        pos_consequences = extract_structured_items(pos_consequences_raw)
        neg_consequences = extract_structured_items(neg_consequences_raw)

        return {
            "sample_id": sample_id,
            "source_id": self.source_id,
            "file_name": file_path.name,
            "record_id": record_id,
            "raw_sha256": raw_hash,
            "title": title,
            "context": context,
            "drivers": drivers,
            "options": options,
            "decision_outcome": outcome_raw,
            "decision": outcome_raw,
            "positive_consequences": pos_consequences,
            "negative_consequences": neg_consequences,
            "raw_text": sanitized_text,
            "is_quarantined": False,
        }
