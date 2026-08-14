"""
Deterministic Stable Sample Identity Generation
"""

import hashlib
from pathlib import Path


def generate_stable_sample_id(
    source_id: str,
    file_path: str | Path,
    record_id: str,
    project_id: str | None = None,
    prefix: str = "arch_",
) -> str:
    """
    Generates a deterministic sample ID:
    SHA256(source_id + ':' + (project_id or '') + ':' + normalized_file_path + ':' + record_id)
    """
    norm_path = Path(file_path).as_posix() if isinstance(file_path, (str, Path)) else str(file_path)
    proj = project_id or ""
    raw_key = f"{source_id}:{proj}:{norm_path}:{record_id}"
    digest = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}{digest}"

