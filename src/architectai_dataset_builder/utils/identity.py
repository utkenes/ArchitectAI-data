"""
Deterministic Stable Sample Identity Generation
"""

import hashlib


def generate_stable_sample_id(
    source_id: str,
    file_path: str,
    record_id: str,
    project_id: str | None = None,
    prefix: str = "arch_",
) -> str:
    """
    Generates a deterministic sample ID:
    SHA256(source_id + ':' + (project_id or '') + ':' + file_path + ':' + record_id)
    """
    proj = project_id or ""
    raw_key = f"{source_id}:{proj}:{file_path}:{record_id}"
    digest = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}{digest}"
