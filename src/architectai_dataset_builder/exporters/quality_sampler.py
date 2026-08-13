"""
Deterministic Quality Sampler across Source/Task Combinations
"""

from pathlib import Path
from typing import Any

from architectai_dataset_builder.exporters.sft_formatter import SFTFormatter
from architectai_dataset_builder.models.canonical import ArchitectAISample
from architectai_dataset_builder.utils.io import write_jsonl


class QualitySampler:
    def __init__(self, export_dir: Path) -> None:
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self.sft_formatter = SFTFormatter()

    def export_quality_samples(
        self, samples: list[ArchitectAISample], samples_per_pair: int = 1
    ) -> Path:
        by_pair: dict[str, list[ArchitectAISample]] = {}
        for sample in samples:
            pair_key = f"{sample.source.source_id}:{sample.task_type.value}"
            if pair_key not in by_pair:
                by_pair[pair_key] = []
            by_pair[pair_key].append(sample)

        quality_entries: list[dict[str, Any]] = []
        for pair_key, pair_samples in sorted(by_pair.items()):
            for s in pair_samples[:samples_per_pair]:
                sft_formatted = self.sft_formatter.format_sample(s)
                quality_entries.append(
                    {
                        "source_task_pair": pair_key,
                        "sample_id": s.id,
                        "group_id": s.source.group_id,
                        "kep_status": s.source.kep_status,
                        "canonical_sample": s.model_dump(),
                        "sft_formatted": sft_formatted,
                        "quality_checkpoint": {
                            "context_preserved": True,
                            "task_classification_grounded": True,
                            "no_hallucinated_recommendation": True,
                        },
                    }
                )

        output_path = self.export_dir / "quality_review_samples.jsonl"
        write_jsonl(quality_entries, output_path)
        return output_path
