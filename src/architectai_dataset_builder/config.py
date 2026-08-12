"""
Config Loader Module
"""

import os
from pathlib import Path
from typing import Dict, Any, List
import yaml


class Config:
    def __init__(self, root_dir: Path | None = None):
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

    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    @property
    def protected_sources(self) -> List[str]:
        return self.policy_config.get("protected_evaluation_sources", [])

    @property
    def min_relevance_score(self) -> float:
        return self.policy_config.get("relevance_filter", {}).get("min_relevance_score", 0.60)

    @property
    def near_dedup_jaccard_threshold(self) -> float:
        return self.policy_config.get("deduplication", {}).get(
            "near_duplicate_jaccard_threshold", 0.85
        )

    @property
    def ngram_size(self) -> int:
        return self.policy_config.get("deduplication", {}).get("ngram_size", 3)
