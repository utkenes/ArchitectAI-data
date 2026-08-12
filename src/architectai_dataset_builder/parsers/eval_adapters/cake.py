"""
CAKE Evaluation Adapter -> FreeResponseEvalSample
"""

import json
from pathlib import Path
from typing import List
from architectai_dataset_builder.models.evaluation import FreeResponseEvalSample, EvalSourceMetadata
from architectai_dataset_builder.utils.hashing import compute_sha256_str, compute_sha256_file
from architectai_dataset_builder.utils.identity import generate_stable_sample_id


class CAKEEvalAdapter:
    def __init__(self):
        self.benchmark_id = "cake"

    def parse_directory(self, raw_dir: Path) -> List[FreeResponseEvalSample]:
        samples = []
        json_file = raw_dir / "cake_eval.json"
        if not json_file.exists():
            return samples

        raw_hash = compute_sha256_file(json_file)
        data = json.loads(json_file.read_text(encoding="utf-8"))

        for item in data:
            rec_id = item.get("id", "q0")
            sample_id = generate_stable_sample_id(
                source_id=self.benchmark_id,
                file_path=json_file.name,
                record_id=rec_id,
                prefix="eval_cake_",
            )
            norm_hash = compute_sha256_str(f"{item['prompt']}:{item['reference_answer']}")

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
                FreeResponseEvalSample(
                    id=sample_id,
                    source=source_meta,
                    prompt=item["prompt"],
                    reference_answer=item["reference_answer"],
                    grading_rubric=item.get("grading_rubric", []),
                    context=item.get("context"),
                )
            )
        return samples
