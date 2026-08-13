"""
Kubernetes Enhancement Proposals (KEPs) Parser with Source-Aware Whitelisting & Section Synonym Extraction
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
    is_lifecycle_status_only,
)


class KubernetesKEPParser(BaseParser):
    def __init__(self) -> None:
        super().__init__(source_id="k8s_keps")

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

        rel_parts = file_path.relative_to(raw_dir).parts
        sig_id = rel_parts[1] if len(rel_parts) > 2 else "sig-architecture"
        record_id = file_path.parent.name if file_path.name.lower() == "readme.md" else file_path.stem
        sample_id = generate_stable_sample_id(
            source_id=self.source_id,
            file_path=str(file_path.relative_to(raw_dir)),
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
        title = title_match.group(1).strip() if title_match else record_id

        status_match = re.search(r"^status:\s*(.+)$", text, re.IGNORECASE | re.MULTILINE)
        if not status_match:
            status_match = re.search(r"## Status\s*\n+([^\n#]+)", text, re.IGNORECASE)
        kep_status = status_match.group(1).strip().lower() if status_match else "provisional"

        # 2. Extract sections using robust synonym mapping
        context_text = extract_markdown_section(text, CONTEXT_SYNONYMS)
        decision_text = extract_markdown_section(text, DECISION_SYNONYMS)
        consequences_text = extract_markdown_section(text, CONSEQUENCE_SYNONYMS)
        alternatives_text = extract_markdown_section(text, ALTERNATIVE_SYNONYMS)

        # 3. Validate context and decision (reject lifecycle/status metadata only)
        is_decision_valid = (
            bool(decision_text)
            and len(decision_text.strip()) >= 15
            and not is_lifecycle_status_only(decision_text)
        )
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
            "motivation": context_text,
            "proposal": decision_text,
            "decision": decision_text,
            "decision_outcome": decision_text,
            "alternatives": [a.strip("- *") for a in alternatives_text.split("\n") if a.strip("- *")],
            "tradeoffs": [r.strip("- *") for r in consequences_text.split("\n") if r.strip("- *")],
            "raw_text": text,
            "is_quarantined": False,
        }
