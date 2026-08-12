"""
JSONL Exporter for Training, Silver, Gold, and Evaluation Datasets
"""

from pathlib import Path
from typing import List, Dict, Any, Union
from architectai_dataset_builder.models.canonical import ArchitectAISample, ReviewStatus
from architectai_dataset_builder.models.evaluation import (
    MultipleChoiceEvalSample,
    FreeResponseEvalSample,
    ArchitectureGenerationEvalSample,
    DiagramEvalSample,
)
from architectai_dataset_builder.exporters.sft_formatter import SFTFormatter
from architectai_dataset_builder.utils.io import write_jsonl, save_yaml
from architectai_dataset_builder.utils.hashing import compute_sha256_file

EvalSampleUnion = Union[
    MultipleChoiceEvalSample,
    FreeResponseEvalSample,
    ArchitectureGenerationEvalSample,
    DiagramEvalSample,
]


class JSONLExporter:
    def __init__(self, export_dir: Path):
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self.eval_export_dir = self.export_dir / "eval"
        self.eval_export_dir.mkdir(parents=True, exist_ok=True)
        self.sft_formatter = SFTFormatter()

    def export_training_datasets(
        self,
        train_samples: List[ArchitectAISample],
        val_samples: List[ArchitectAISample],
        approved_sample_ids: List[str],
    ) -> Dict[str, Path]:
        all_samples = train_samples + val_samples

        # 1. Silver Dataset: Validated unreviewed samples
        silver_samples = [s for s in all_samples if s.review.status != ReviewStatus.APPROVED]
        silver_dicts = [s.model_dump() for s in silver_samples]
        silver_path = self.export_dir / "silver.jsonl"
        write_jsonl(silver_dicts, silver_path)

        # 2. Gold Dataset: Only samples explicitly approved in manifest
        gold_samples = [s for s in all_samples if s.id in approved_sample_ids]
        for g in gold_samples:
            g.review.status = ReviewStatus.APPROVED
        gold_dicts = [g.model_dump() for g in gold_samples]
        gold_path = self.export_dir / "gold.jsonl"
        write_jsonl(gold_dicts, gold_path)

        # 3. SFT Training Dataset
        train_sft_dicts = [self.sft_formatter.format_sample(s) for s in train_samples]
        train_sft_path = self.export_dir / "train_sft.jsonl"
        write_jsonl(train_sft_dicts, train_sft_path)

        # 4. SFT Validation Dataset
        val_sft_dicts = [self.sft_formatter.format_sample(s) for s in val_samples]
        val_sft_path = self.export_dir / "validation_sft.jsonl"
        write_jsonl(val_sft_dicts, val_sft_path)

        return {
            "silver": silver_path,
            "gold": gold_path,
            "train_sft": train_sft_path,
            "validation_sft": val_sft_path,
        }

    def export_evaluation_datasets(
        self, eval_samples: List[EvalSampleUnion]
    ) -> Dict[str, Path]:
        by_benchmark: Dict[str, List[Dict[str, Any]]] = {}

        for sample in eval_samples:
            b_id = sample.source.benchmark_id
            if b_id not in by_benchmark:
                by_benchmark[b_id] = []
            by_benchmark[b_id].append(sample.model_dump())

        paths = {}
        eval_manifest_data = {"benchmarks": {}}

        for b_id, sample_dicts in by_benchmark.items():
            path = self.eval_export_dir / f"{b_id}.jsonl"
            write_jsonl(sample_dicts, path)
            paths[b_id] = path

            eval_manifest_data["benchmarks"][b_id] = {
                "count": len(sample_dicts),
                "path": str(path.relative_to(self.export_dir)),
                "sha256": compute_sha256_file(path),
            }

        eval_manifest_path = self.export_dir / "eval_manifest.json"
        write_jsonl([eval_manifest_data], eval_manifest_path)
        paths["eval_manifest"] = eval_manifest_path

        return paths
