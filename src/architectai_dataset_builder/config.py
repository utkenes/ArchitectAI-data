"""
Config Loader Module
"""

from pathlib import Path
from typing import Any

import yaml


class Config:
    def __init__(self, root_dir: Path | None = None) -> None:
        if root_dir is None:
            # Default to parent directory of src/
            self.root_dir = Path(__file__).resolve().parents[2]
        else:
            self.root_dir = Path(root_dir)

        self.config_dir = self.root_dir / "config"
        self.manifests_dir = self.root_dir / "manifests"
        self.data_dir = self.root_dir / "data"

        self.sources_config = self._load_yaml(self.config_dir / "sources.yaml")
        self.policy_config = self._load_yaml(self.config_dir / "dataset_policy.yaml")
        self.splits_config = self._load_yaml(self.config_dir / "splits.yaml")

    def _load_yaml(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    @property
    def protected_sources(self) -> list[str]:
        val = self.policy_config.get("protected_evaluation_sources", [])
        return [str(x) for x in val]

    @property
    def min_relevance_score(self) -> float:
        val = self.policy_config.get("relevance_filter", {}).get("min_relevance_score", 0.60)
        return float(val)

    @property
    def near_dedup_jaccard_threshold(self) -> float:
        val = self.policy_config.get("deduplication", {}).get("near_duplicate_jaccard_threshold", 0.85)
        return float(val)

    @property
    def ngram_size(self) -> int:
        val = self.policy_config.get("deduplication", {}).get("ngram_size", 3)
        return int(val)
