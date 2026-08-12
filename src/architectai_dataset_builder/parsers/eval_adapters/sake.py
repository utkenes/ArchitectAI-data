"""
SAKE Evaluation Adapter -> MultipleChoiceEvalSample
"""

import json
from pathlib import Path

from architectai_dataset_builder.models.evaluation import (
    EvalSourceMetadata,
    MultipleChoiceEvalSample,
)
from architectai_dataset_builder.utils.hashing import compute_sha256_file, compute_sha256_str
from architectai_dataset_builder.utils.identity import generate_stable_sample_id


class SAKEEvalAdapter:
    def __init__(self) -> None:
        self.benchmark_id = "sake"

    def parse_directory(self, raw_dir: Path) -> list[MultipleChoiceEvalSample]:
        samples: list[MultipleChoiceEvalSample] = []
        json_file = raw_dir / "sake_questions.json"
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
                prefix="eval_sake_",
            )
            norm_hash = compute_sha256_str(f"{item['question']}:{item['correct_answer']}")

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
                MultipleChoiceEvalSample(
                    id=sample_id,
                    source=source_meta,
                    question=item["question"],
                    options=item["options"],
                    correct_answer=item["correct_answer"],
                    explanation=item.get("explanation"),
                )
            )
        return samples
