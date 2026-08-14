"""
Final Exported SFT Dataset Validator for ArchitectAI Dataset Builder V1.1.1

Independently validates exported train_sft.jsonl and validation_sft.jsonl files on disk
and provides measured metrics as the shared source of truth for ReadinessReporter and audit_dataset.py.
"""

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from architectai_dataset_builder.utils.io import read_jsonl


@dataclass
class ExportValidationResult:
    total_exported_samples: int = 0
    train_sft_samples: int = 0
    validation_sft_samples: int = 0
    duplicate_sample_ids: int = 0
    conflicting_duplicate_sample_ids: int = 0
    empty_assistant_answers: int = 0
    template_leakage_count: int = 0
    unresolved_placeholder_count: int = 0
    suspicious_structural_output_count: int = 0
    semantic_failures_exported: int = 0
    group_overlap_count: int = 0
    suspicious_todos: list[str] = field(default_factory=list)
    raw_token_leakage_findings: list[str] = field(default_factory=list)
    has_critical_failures: bool = False
    failure_reasons: list[str] = field(default_factory=list)

    def model_dump(self) -> dict[str, Any]:
        return {
            "total_exported_samples": self.total_exported_samples,
            "train_sft_samples": self.train_sft_samples,
            "validation_sft_samples": self.validation_sft_samples,
            "duplicate_sample_ids": self.duplicate_sample_ids,
            "conflicting_duplicate_sample_ids": self.conflicting_duplicate_sample_ids,
            "empty_assistant_answers": self.empty_assistant_answers,
            "template_leakage_count": self.template_leakage_count,
            "unresolved_placeholder_count": self.unresolved_placeholder_count,
            "suspicious_structural_output_count": self.suspicious_structural_output_count,
            "semantic_failures_exported": self.semantic_failures_exported,
            "group_overlap_count": self.group_overlap_count,
            "suspicious_todos": self.suspicious_todos,
            "raw_token_leakage_findings": self.raw_token_leakage_findings,
            "has_critical_failures": self.has_critical_failures,
            "failure_reasons": self.failure_reasons,
        }


class SFTExportValidator:
    """
    Inspects exported JSONL files to calculate empirical readiness metrics.
    """

    LEAKAGE_TOKENS: ClassVar[list[str]] = ["<!--", "-->", "${", "{{", "}}"]
    PLACEHOLDER_SIGNATURES: ClassVar[list[str]] = ["[insert ", "<placeholder", "n/a - template", "[todo:", "[tbd:"]

    def validate_exports(self, export_dir: Path) -> ExportValidationResult:
        train_file = export_dir / "train_sft.jsonl"
        val_file = export_dir / "validation_sft.jsonl"

        result = ExportValidationResult()

        train_records = read_jsonl(train_file) if train_file.exists() else []
        val_records = read_jsonl(val_file) if val_file.exists() else []

        result.train_sft_samples = len(train_records)
        result.validation_sft_samples = len(val_records)
        result.total_exported_samples = len(train_records) + len(val_records)

        seen_sample_ids: dict[str, str] = {}  # sample_id -> content_hash
        train_groups: set[str] = set()
        val_groups: set[str] = set()

        def process_file_records(records: list[dict[str, Any]], is_train: bool) -> None:
            for s in records:
                sid = str(s.get("id") or s.get("sample_id") or "")
                group_id = str(s.get("group_id") or "")

                if group_id:
                    if is_train:
                        train_groups.add(group_id)
                    else:
                        val_groups.add(group_id)

                # Extract assistant response
                messages = s.get("messages", [])
                assistant_content = ""
                for m in messages:
                    if m.get("role") == "assistant":
                        assistant_content = m.get("content", "")

                content_hash = hashlib.sha256(assistant_content.encode("utf-8")).hexdigest()

                # 1. Duplicate ID validation
                if sid in seen_sample_ids:
                    result.duplicate_sample_ids += 1
                    if seen_sample_ids[sid] != content_hash:
                        result.conflicting_duplicate_sample_ids += 1
                        result.failure_reasons.append(f"Conflicting content collision for sample ID '{sid}'")
                else:
                    seen_sample_ids[sid] = content_hash

                # 2. Empty assistant answer validation
                if not assistant_content.strip():
                    result.empty_assistant_answers += 1
                    result.failure_reasons.append(f"Empty assistant response for sample ID '{sid}'")

                # 3. Template leakage scan
                found_leakage = False
                for token in self.LEAKAGE_TOKENS:
                    if token in assistant_content:
                        found_leakage = True
                        finding = f"Sample '{sid}': Raw template token '{token}' in assistant response."
                        result.raw_token_leakage_findings.append(finding)

                if found_leakage:
                    result.template_leakage_count += 1
                    result.failure_reasons.append(f"Template leakage in sample '{sid}'")

                # 4. Unresolved placeholder scan
                content_lower = assistant_content.lower()
                for ph in self.PLACEHOLDER_SIGNATURES:
                    if ph in content_lower:
                        result.unresolved_placeholder_count += 1
                        result.failure_reasons.append(f"Unresolved placeholder signature '{ph}' in sample '{sid}'")
                        break

                # 5. Suspicious TODO/TBD scan (warning level)
                if "todo" in content_lower and "[todo:" not in content_lower:
                    result.suspicious_todos.append(f"Sample '{sid}': Contains TODO in text")
                if "tbd" in content_lower and "[tbd:" not in content_lower:
                    result.suspicious_todos.append(f"Sample '{sid}': Contains TBD in text")

                # 6. Status-only metadata / not explicitly stated check
                if "not explicitly stated" in content_lower and len(content_lower) < 50:
                    result.semantic_failures_exported += 1
                    result.failure_reasons.append(f"Status-only or ungrounded fallback in sample '{sid}'")

        process_file_records(train_records, is_train=True)
        process_file_records(val_records, is_train=False)

        # 7. Group leakage check
        group_overlap = train_groups.intersection(val_groups)
        result.group_overlap_count = len(group_overlap)
        if group_overlap:
            result.failure_reasons.append(
                f"Train/Validation group leakage detected: {len(group_overlap)} overlapping groups"
            )

        if (
            result.conflicting_duplicate_sample_ids > 0
            or result.template_leakage_count > 0
            or result.empty_assistant_answers > 0
            or result.semantic_failures_exported > 0
            or result.group_overlap_count > 0
        ):
            result.has_critical_failures = True

        return result
