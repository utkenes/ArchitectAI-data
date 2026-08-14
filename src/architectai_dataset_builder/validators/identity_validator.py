"""
Global Identity Uniqueness & Collision Validator
"""

from typing import Any


class IdentityCollisionError(Exception):
    """Raised when a sample ID collision with non-matching content is detected."""


class IdentityValidator:
    """
    Validates global sample ID uniqueness across the build pipeline.

    Rules:
    1. Same ID + same content -> valid duplicate record.
    2. Same ID + different content -> HARD BUILD FAILURE (IdentityCollisionError).
    3. Detect duplicate IDs across source/project/path combinations.
    """

    def __init__(self) -> None:
        # Maps sample_id -> {content_hash, source_id, project_id, file_path, record_id}
        self._seen: dict[str, dict[str, Any]] = {}

    def register_and_validate(
        self,
        sample_id: str,
        content_hash: str,
        source_id: str,
        file_path: str,
        record_id: str,
        project_id: str | None = None,
    ) -> None:
        """
        Registers a sample ID and validates identity integrity.
        Raises IdentityCollisionError if a collision with non-matching content is found.
        """
        entry = {
            "content_hash": content_hash,
            "source_id": source_id,
            "file_path": file_path,
            "record_id": record_id,
            "project_id": project_id or "",
        }

        if sample_id in self._seen:
            existing = self._seen[sample_id]
            if existing["content_hash"] != content_hash:
                raise IdentityCollisionError(
                    f"Sample ID collision detected for ID '{sample_id}'! "
                    f"Existing ({existing['source_id']}:{existing['file_path']}) != "
                    f"New ({source_id}:{file_path}). Non-matching content."
                )
        else:
            self._seen[sample_id] = entry

    @property
    def registered_count(self) -> int:
        return len(self._seen)

    def reset(self) -> None:
        self._seen.clear()
