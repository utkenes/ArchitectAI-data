"""
Composite Group-Level Deterministic Dataset Splitter with Strict R2ABench Split Manifest Invariants
"""

from pathlib import Path

from architectai_dataset_builder.models.canonical import ArchitectAISample
from architectai_dataset_builder.utils.hashing import hash_string_to_int
from architectai_dataset_builder.utils.io import load_yaml


class SplitManifestValidationError(Exception):
    """Raised when dataset split manifest violates structural or disjointness invariants."""



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
        self._validate_manifest_invariants()

    def _load_r2abench_splits(self) -> dict[str, list[str]]:
        if not self.split_manifest_path.exists():
            return {"training": [], "validation": [], "held_out": []}
        data = load_yaml(self.split_manifest_path)

        # Support both top-level schema and legacy nested splits key
        if "train_projects" in data:
            train_pids = data.get("train_projects", [])
            val_pids = data.get("validation_projects", [])
            holdout_pids = data.get("held_out_projects", [])
        else:
            splits = data.get("splits", {})
            train_pids = splits.get("train_projects", []) or splits.get("training_projects", [])
            val_pids = splits.get("validation_projects", [])
            holdout_pids = splits.get("held_out_projects", [])

        return {
            "training": [str(p) for p in train_pids],
            "validation": [str(p) for p in val_pids],
            "held_out": [str(p) for p in holdout_pids],
        }

    def _validate_manifest_invariants(self) -> None:
        if not self.split_manifest_path.exists():
            return

        data = load_yaml(self.split_manifest_path)
        is_r2abench_manifest = data.get("source_id") == "r2abench" or "train_projects" in data or "split_manifest_id" in data

        if not is_r2abench_manifest:
            return

        train_list = self.r2abench_splits.get("training", [])
        val_list = self.r2abench_splits.get("validation", [])
        holdout_list = self.r2abench_splits.get("held_out", [])

        # 1. Non-empty lists check
        if not train_list:
            raise SplitManifestValidationError("R2ABench training project list is empty in split manifest.")
        if not val_list:
            raise SplitManifestValidationError("R2ABench validation project list is empty in split manifest.")
        if not holdout_list:
            raise SplitManifestValidationError("R2ABench held-out project list is empty in split manifest.")

        # 2. Duplicate project IDs check
        all_pids = train_list + val_list + holdout_list
        if len(all_pids) != len(set(all_pids)):
            duplicates = [p for p in all_pids if all_pids.count(p) > 1]
            raise SplitManifestValidationError(
                f"R2ABench split manifest contains duplicate project IDs: {set(duplicates)}"
            )

        # 3. Pairwise disjointness check
        train_set = set(train_list)
        val_set = set(val_list)
        holdout_set = set(holdout_list)

        t_v_overlap = train_set & val_set
        if t_v_overlap:
            raise SplitManifestValidationError(
                f"R2ABench project appears in both training and validation splits: {t_v_overlap}"
            )

        t_h_overlap = train_set & holdout_set
        if t_h_overlap:
            raise SplitManifestValidationError(
                f"R2ABench project appears in both training and held-out splits: {t_h_overlap}"
            )

        v_h_overlap = val_set & holdout_set
        if v_h_overlap:
            raise SplitManifestValidationError(
                f"R2ABench project appears in both validation and held-out splits: {v_h_overlap}"
            )

    def split_samples(
        self, samples: list[ArchitectAISample]
    ) -> tuple[list[ArchitectAISample], list[ArchitectAISample]]:
        train_samples: list[ArchitectAISample] = []
        val_samples: list[ArchitectAISample] = []

        group_split_map: dict[str, str] = {}

        for sample in samples:
            group_id = (
                sample.source.group_id
                or f"group_{sample.source.source_id}_{sample.source.project_id or 'default'}_{sample.source.source_record_id}"
            )

            if sample.source.source_id == "r2abench" and sample.source.project_id:
                pid = sample.source.project_id
                if pid in self.r2abench_splits["training"]:
                    assigned_split = "train"
                elif pid in self.r2abench_splits["validation"]:
                    assigned_split = "validation"
                elif pid in self.r2abench_splits["held_out"]:
                    raise SplitManifestValidationError(
                        f"Evaluation Integrity Error! Held-out R2ABench project '{pid}' "
                        f"attempted to enter training/validation dataset splitter!"
                    )
                else:
                    raise SplitManifestValidationError(
                        f"Unknown R2ABench project ID '{pid}' is not declared in split manifest {self.split_manifest_path.name}!"
                    )
            else:
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
