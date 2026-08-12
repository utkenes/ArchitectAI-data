"""
Deterministic Dataset Splitter and R2ABench Project-Level Split Manager
"""

import random
from pathlib import Path
from typing import List, Tuple, Dict, Set
from architectai_dataset_builder.models.canonical import ArchitectAISample
from architectai_dataset_builder.utils.io import load_yaml


class DeterministicSplitter:
    def __init__(self, splits_manifest_path: Path, seed: int = 42, train_ratio: float = 0.80):
        self.seed = seed
        self.train_ratio = train_ratio
        self.r2abench_splits = self._load_r2abench_manifest(splits_manifest_path)

    def _load_r2abench_manifest(self, path: Path) -> Dict[str, Set[str]]:
        if not path.exists():
            return {"train": set(), "validation": set(), "held_out": set()}
        data = load_yaml(path)
        return {
            "train": set(data.get("train_projects", [])),
            "validation": set(data.get("validation_projects", [])),
            "held_out": set(data.get("held_out_projects", [])),
        }

    def split_samples(
        self, samples: List[ArchitectAISample]
    ) -> Tuple[List[ArchitectAISample], List[ArchitectAISample]]:
        train_samples: List[ArchitectAISample] = []
        val_samples: List[ArchitectAISample] = []

        # Separate R2ABench project-level splits from general random splits
        generic_samples: List[ArchitectAISample] = []

        for sample in samples:
            if sample.source.source_id == "r2abench":
                project_id = sample.source.project_id
                if project_id in self.r2abench_splits["held_out"]:
                    # Project is reserved for held-out evaluation! Skip training assignment
                    continue
                elif project_id in self.r2abench_splits["validation"]:
                    sample.source.split = "validation"
                    sample.source.split_reason = "r2abench_manifest_project_split"
                    val_samples.append(sample)
                else:
                    sample.source.split = "train"
                    sample.source.split_reason = "r2abench_manifest_project_split"
                    train_samples.append(sample)
            else:
                generic_samples.append(sample)

        # Apply seeded deterministic shuffle to generic samples
        rng = random.Random(self.seed)
        shuffled = list(generic_samples)
        rng.shuffle(shuffled)

        cutoff = int(len(shuffled) * self.train_ratio)
        for idx, sample in enumerate(shuffled):
            if idx < cutoff:
                sample.source.split = "train"
                sample.source.split_reason = "deterministic_ratio_split"
                train_samples.append(sample)
            else:
                sample.source.split = "validation"
                sample.source.split_reason = "deterministic_ratio_split"
                val_samples.append(sample)

        return train_samples, val_samples
