"""
Composite Group-Level Deterministic Dataset Splitter
"""

from pathlib import Path

from architectai_dataset_builder.models.canonical import ArchitectAISample
from architectai_dataset_builder.utils.hashing import hash_string_to_int
from architectai_dataset_builder.utils.io import load_yaml


class DeterministicSplitter:
    def __init__(
        self,
        split_manifest_path: Path,
        train_ratio: float = 0.80,
        val_ratio: float = 0.20,
    ) -> None:
        self.split_manifest_path = Path(split_manifest_path)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.r2abench_splits = self._load_r2abench_splits()

    def _load_r2abench_splits(self) -> dict[str, list[str]]:
        if not self.split_manifest_path.exists():
            return {"training": [], "validation": [], "held_out": []}
        data = load_yaml(self.split_manifest_path)
        splits = data.get("splits", {})
        return {
            "training": splits.get("training_projects", []),
            "validation": splits.get("validation_projects", []),
            "held_out": splits.get("held_out_projects", []),
        }

    def split_samples(
        self, samples: list[ArchitectAISample]
    ) -> tuple[list[ArchitectAISample], list[ArchitectAISample]]:
        train_samples: list[ArchitectAISample] = []
        val_samples: list[ArchitectAISample] = []

        # Map each group_id deterministically to split
        group_split_map: dict[str, str] = {}

        for sample in samples:
            # 1. Composite group_id isolation
            group_id = (
                sample.source.group_id
                or f"group_{sample.source.source_id}_{sample.source.project_id or 'default'}_{sample.source.source_record_id}"
            )

            # 2. Check R2ABench project-level manifest rules
            if sample.source.source_id == "r2abench" and sample.source.project_id:
                pid = sample.source.project_id
                if pid in self.r2abench_splits["training"]:
                    assigned_split = "train"
                elif pid in self.r2abench_splits["validation"]:
                    assigned_split = "validation"
                else:
                    # Deterministic hash split for unscheduled R2ABench projects
                    val = hash_string_to_int(group_id) % 100
                    assigned_split = "train" if val < (self.train_ratio * 100) else "validation"
            else:
                # Deterministic hash split based on composite group_id
                if group_id not in group_split_map:
                    val = hash_string_to_int(group_id) % 100
                    group_split_map[group_id] = "train" if val < (self.train_ratio * 100) else "validation"
                assigned_split = group_split_map[group_id]

            sample.source.split = assigned_split

            if assigned_split == "train":
                train_samples.append(sample)
            else:
                val_samples.append(sample)

        return train_samples, val_samples
