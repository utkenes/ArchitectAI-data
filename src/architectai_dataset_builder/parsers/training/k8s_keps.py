"""
Kubernetes Enhancement Proposals (KEPs) Parser with Strict Grounding & Template Quarantining
"""

import re
from pathlib import Path
from typing import Any

from architectai_dataset_builder.parsers.base import BaseParser
from architectai_dataset_builder.utils.hashing import compute_sha256_file
from architectai_dataset_builder.utils.identity import generate_stable_sample_id
from architectai_dataset_builder.utils.markdown import (
    extract_markdown_section,
    has_template_placeholders,
    is_boilerplate_filename,
)


class KubernetesKEPParser(BaseParser):
    def __init__(self) -> None:
        super().__init__(source_id="k8s_keps")

    def parse_directory(self, raw_dir: Path) -> list[dict[str, Any]]:
        records = []
        for file_path in sorted(raw_dir.rglob("*.md")):
            if not file_path.is_file() or file_path.name.startswith("."):
                continue

            rel_path = str(file_path.relative_to(raw_dir))
            if is_boilerplate_filename(file_path.name, rel_path):
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

        rel_parts = file_path.relative_to(raw_dir).parts
        sig_id = rel_parts[1] if len(rel_parts) > 2 else "sig-architecture"
        record_id = file_path.stem
        sample_id = generate_stable_sample_id(
            source_id=self.source_id,
            file_path=file_path.name,
            record_id=record_id,
            project_id=sig_id,
        )

        # 1. Quarantine unresolved template placeholders
        if has_template_placeholders(text):
            return {
                "sample_id": sample_id,
                "source_id": self.source_id,
                "file_name": file_path.name,
                "record_id": record_id,
                "project_id": sig_id,
                "raw_sha256": raw_hash,
                "is_quarantined": True,
                "quarantine_reason": "unresolved_template_placeholder",
                "raw_text": text,
            }

        title_match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
        title = title_match.group(1).strip() if title_match else file_path.stem

        status_match = re.search(r"^status:\s*(.+)$", text, re.IGNORECASE | re.MULTILINE)
        if not status_match:
            status_match = re.search(r"## Status\s*\n+([^\n#]+)", text, re.IGNORECASE)
        kep_status = status_match.group(1).strip().lower() if status_match else "provisional"

        summary = extract_markdown_section(text, r"##\s+Summary")
        motivation = extract_markdown_section(text, r"##\s+Motivation")
        proposal = extract_markdown_section(text, r"##\s+Proposal")
        design_details = extract_markdown_section(text, r"##\s+Design Details")
        alternatives = extract_markdown_section(text, r"##\s+Alternatives")
        risks = extract_markdown_section(text, r"##\s+Risks and Mitigations")

        decision_text = proposal or design_details
        context_text = summary or motivation

        # 2. Strict grounding: Require genuine decision/proposal and context section
        is_decision_valid = bool(decision_text) and len(decision_text.strip()) >= 15
        is_context_valid = bool(context_text) and len(context_text.strip()) >= 30

        if not is_decision_valid or not is_context_valid:
            return {
                "sample_id": sample_id,
                "source_id": self.source_id,
                "file_name": file_path.name,
                "record_id": record_id,
                "project_id": sig_id,
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
            "project_id": sig_id,
            "raw_sha256": raw_hash,
            "title": title,
            "kep_status": kep_status,
            "summary": context_text,
            "motivation": motivation,
            "proposal": decision_text,
            "decision": decision_text,
            "decision_outcome": decision_text,
            "alternatives": [a.strip("- *") for a in alternatives.split("\n") if a.strip("- *")],
            "tradeoffs": [r.strip("- *") for r in risks.split("\n") if r.strip("- *")],
            "raw_text": text,
            "is_quarantined": False,
        }
