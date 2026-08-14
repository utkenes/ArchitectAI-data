"""
Deterministic Training Corpus Balancer & Selection Profiles Generator
"""

from collections import Counter
from pathlib import Path
from typing import Any

from architectai_dataset_builder.exporters.sft_formatter import SFTFormatter
from architectai_dataset_builder.models.canonical import ArchitectAISample
from architectai_dataset_builder.utils.hashing import hash_string_to_int
from architectai_dataset_builder.utils.io import write_jsonl


class TrainingBalancer:
    """
    Generates deterministic training profiles (full_clean, balanced_conservative, balanced_strict)
    derived from valid train samples without mutating or deleting raw silver exports.
    """

    def __init__(self, export_dir: Path) -> None:
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self.sft_formatter = SFTFormatter()

    def generate_profiles(
        self, train_samples: list[ArchitectAISample]
    ) -> dict[str, Any]:
        full_clean_samples = train_samples

        # Profile 1: Full Clean
        full_clean_manifest = self._build_profile_metadata("full_clean", full_clean_samples, len(train_samples))

        # Profile 2: Balanced Conservative (Cap dominant source to <= 80%)
        cons_samples = self._downselect_profile(
            train_samples, max_single_source_ratio=0.80
        )
        cons_manifest = self._build_profile_metadata("balanced_conservative", cons_samples, len(train_samples))

        # Profile 3: Balanced Strict (Analytical export profile, cap dominant source to <= 60%)
        strict_samples = self._downselect_profile(
            train_samples, max_single_source_ratio=0.60
        )
        strict_manifest = self._build_profile_metadata("balanced_strict", strict_samples, len(train_samples))

        profiles_report = {
            "profiles": {
                "full_clean": full_clean_manifest,
                "balanced_conservative": cons_manifest,
                "balanced_strict": strict_manifest,
            },
            "default_recommended_profile": "balanced_conservative",
        }

        # Export training selection manifest
        write_jsonl([profiles_report], self.export_dir / "training_selection_manifest.json")

        # Export default balanced SFT dataset (balanced_conservative)
        balanced_sft_entries = []
        for s in cons_samples:
            entry = self.sft_formatter.format_sample(s)
            if entry:
                balanced_sft_entries.append(entry)

        write_jsonl(balanced_sft_entries, self.export_dir / "train_sft_balanced.jsonl")

        return profiles_report

    def _downselect_profile(
        self, train_samples: list[ArchitectAISample], max_single_source_ratio: float
    ) -> list[ArchitectAISample]:
        source_counts: Counter[str] = Counter(s.source.source_id for s in train_samples)
        other_sources_count = sum(c for sid, c in source_counts.items() if sid != "k8s_keps")

        # If non-KEP sources are small, determine target cap for KEPs
        if other_sources_count > 0:
            max_keps = int((other_sources_count * max_single_source_ratio) / (1.0 - max_single_source_ratio))
        else:
            max_keps = len(train_samples)

        selected_samples: list[ArchitectAISample] = []
        keps_count = 0

        # Sort samples deterministically
        sorted_samples = sorted(
            train_samples,
            key=lambda s: (
                0 if s.source.source_id != "k8s_keps" else 1,
                s.task_type.value,
                hash_string_to_int(s.id),
            ),
        )

        for s in sorted_samples:
            if s.source.source_id == "k8s_keps":
                if keps_count < max_keps:
                    selected_samples.append(s)
                    keps_count += 1
            else:
                # Never drop minority sources!
                selected_samples.append(s)

        return selected_samples

    def _build_profile_metadata(
        self, profile_name: str, samples: list[ArchitectAISample], total_original: int
    ) -> dict[str, Any]:
        source_dist: Counter[str] = Counter(s.source.source_id for s in samples)
        task_dist: Counter[str] = Counter(s.task_type.value for s in samples)
        retained = len(samples)
        dropped = total_original - retained

        max_src = max(source_dist.values()) if source_dist else 0
        conc_ratio = (max_src / retained) if retained > 0 else 0.0

        return {
            "profile_name": profile_name,
            "sample_count": retained,
            "dropped_by_balance_count": dropped,
            "retained_coverage_pct": (retained / total_original * 100.0) if total_original > 0 else 0.0,
            "source_distribution": dict(source_dist),
            "source_concentration_ratio": round(conc_ratio, 4),
            "task_distribution": dict(task_dist),
        }
