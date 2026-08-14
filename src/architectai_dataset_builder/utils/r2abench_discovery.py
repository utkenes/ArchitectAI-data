"""
Shared Deterministic R2ABench Project File Discovery
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class R2AProjectFiles:
    project_id: str
    requirements_path: Path
    architecture_path: Path | None
    parent_dir: Path


def discover_r2abench_projects(raw_dir: Path) -> dict[str, R2AProjectFiles]:
    """
    Recursively discovers all R2ABench project files in raw_dir.
    Pairs requirements_path with architecture_path relative to req_file.parent first.
    Deduplicates by resolved requirements file path and project_id.
    """
    projects: dict[str, R2AProjectFiles] = {}
    if not raw_dir.exists():
        return projects

    # Find requirement files recursively
    req_files = sorted(raw_dir.rglob("*_req.txt")) + sorted(raw_dir.rglob("*.txt"))
    seen_paths: set[Path] = set()

    for req_file in req_files:
        resolved_path = req_file.resolve()
        if not req_file.is_file() or req_file.name.startswith(".") or resolved_path in seen_paths:
            continue

        seen_paths.add(resolved_path)
        project_id = req_file.stem.replace("_req", "")

        if project_id in projects:
            continue

        parent_dir = req_file.parent

        # Search PlantUML diagram relative to requirement file location
        arch_file = parent_dir / f"{project_id}_arch.puml"
        if not arch_file.exists():
            arch_file = parent_dir / f"{project_id}.puml"
        if not arch_file.exists():
            arch_file = raw_dir / f"{project_id}_arch.puml"
        if not arch_file.exists():
            arch_file = raw_dir / f"{project_id}.puml"

        final_arch_path = arch_file if arch_file.exists() else None

        projects[project_id] = R2AProjectFiles(
            project_id=project_id,
            requirements_path=req_file,
            architecture_path=final_arch_path,
            parent_dir=parent_dir,
        )

    return projects


def get_r2abench_discovery_diagnostics(
    raw_dir: Path, expected_heldout_pids: list[str]
) -> dict[str, Any]:
    discovered_projects = discover_r2abench_projects(raw_dir)
    discovered_pids = sorted(discovered_projects.keys())

    heldout_expected = sorted(expected_heldout_pids)
    heldout_found = sorted([p for p in heldout_expected if p in discovered_projects])
    heldout_missing = sorted([p for p in heldout_expected if p not in discovered_projects])

    pairs_found = sum(1 for p in discovered_projects.values() if p.architecture_path is not None)
    pairs_missing = sum(1 for p in discovered_projects.values() if p.architecture_path is None)

    return {
        "projects_discovered": discovered_pids,
        "heldout_expected": heldout_expected,
        "heldout_found": heldout_found,
        "heldout_missing": heldout_missing,
        "architecture_pairs_found": pairs_found,
        "architecture_pairs_missing": pairs_missing,
    }
