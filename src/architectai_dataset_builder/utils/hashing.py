"""
SHA-256 Hashing Utilities
"""

import hashlib
from pathlib import Path


def compute_sha256_str(content: str) -> str:
    """Compute SHA-256 hex string of text content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def compute_sha256_bytes(content: bytes) -> str:
    """Compute SHA-256 hex string of bytes."""
    return hashlib.sha256(content).hexdigest()


def compute_sha256_file(file_path: Path) -> str:
    """Compute SHA-256 hex string of a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()
