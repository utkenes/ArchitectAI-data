"""
Canonical Schema Completeness & Structural Validator
"""

from typing import List, Tuple
from architectai_dataset_builder.models.canonical import ArchitectAISample


class SchemaValidator:
    def validate(self, sample: ArchitectAISample) -> Tuple[bool, List[str]]:
        errors = []

        if not sample.id or not sample.id.startswith("arch_"):
            errors.append("Invalid or missing sample id")

        if not sample.scenario or len(sample.scenario.strip()) < 3:
            errors.append("Empty or invalid scenario text")

        if not sample.source.source_id:
            errors.append("Missing source_id")

        if not sample.source.raw_sha256 or not sample.source.normalized_sha256:
            errors.append("Missing integrity SHA-256 hashes")

        return (len(errors) == 0, errors)
