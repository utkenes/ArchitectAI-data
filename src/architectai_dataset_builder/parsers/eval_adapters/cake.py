"""
CAKE Evaluation Adapter -> FreeResponseEvalSample (Dynamic Ingestion)
"""

import json
from pathlib import Path

from architectai_dataset_builder.models.evaluation import EvalSourceMetadata, FreeResponseEvalSample
from architectai_dataset_builder.utils.hashing import compute_sha256_file, compute_sha256_str
from architectai_dataset_builder.utils.identity import generate_stable_sample_id


class CAKEEvalAdapter:
    def __init__(self) -> None:
        self.benchmark_id = "cake"

    def parse_directory(self, raw_dir: Path) -> list[FreeResponseEvalSample]:
        samples: list[FreeResponseEvalSample] = []
        json_files = sorted(raw_dir.glob("*.json"))

        for json_file in json_files:
            if not json_file.is_file():
                continue

            raw_hash = compute_sha256_file(json_file)
            data = json.loads(json_file.read_text(encoding="utf-8", errors="ignore"))
            if isinstance(data, dict):
                data = [data]

            for idx, item in enumerate(data):
                if not isinstance(item, dict) or "prompt" not in item:
                    continue
                rec_id = item.get("id", f"q_{idx}")
                sample_id = generate_stable_sample_id(
                    source_id=self.benchmark_id,
                    file_path=json_file.name,
                    record_id=str(rec_id),
                    prefix="eval_cake_",
                )
                norm_hash = compute_sha256_str(f"{item['prompt']}:{item.get('reference_answer', '')}")

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
                        reference_answer=item.get("reference_answer", ""),
                        grading_rubric=item.get("grading_rubric", []),
                        context=item.get("context"),
                    )
                )
        return samples
