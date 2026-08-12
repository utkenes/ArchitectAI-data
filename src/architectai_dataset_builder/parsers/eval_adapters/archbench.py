"""
ArchBench Evaluation Adapter -> ArchitectureGenerationEvalSample
"""

import json
from pathlib import Path

from architectai_dataset_builder.models.evaluation import (
    ArchitectureGenerationEvalSample,
    EvalSourceMetadata,
)
from architectai_dataset_builder.utils.hashing import compute_sha256_file, compute_sha256_str
from architectai_dataset_builder.utils.identity import generate_stable_sample_id


class ArchBenchEvalAdapter:
    def __init__(self) -> None:
        self.benchmark_id = "archbench"

    def parse_directory(self, raw_dir: Path) -> list[ArchitectureGenerationEvalSample]:
        samples: list[ArchitectureGenerationEvalSample] = []
        json_file = raw_dir / "archbench_tasks.json"
        if not json_file.exists():
            return samples

        raw_hash = compute_sha256_file(json_file)
        data = json.loads(json_file.read_text(encoding="utf-8"))

        for item in data:
            rec_id = item.get("id", "t0")
            sample_id = generate_stable_sample_id(
                source_id=self.benchmark_id,
                file_path=json_file.name,
                record_id=rec_id,
                prefix="eval_archbench_",
            )
            norm_hash = compute_sha256_str(f"{item['requirements']}:{item['reference_solution']}")

            source_meta = EvalSourceMetadata(
                benchmark_id=self.benchmark_id,
                sample_id=sample_id,
                license_id="CC-BY-4.0",
                raw_sha256=raw_hash,
                normalized_sha256=norm_hash,
                evaluation_only=True,
                split="held_out_eval",
            )

            samples.append(
                ArchitectureGenerationEvalSample(
                    id=sample_id,
                    source=source_meta,
                    requirements=item["requirements"],
                    constraints=item.get("constraints", []),
                    expected_components=item.get("expected_components", []),
                    reference_solution=item["reference_solution"],
                )
            )
        return samples
