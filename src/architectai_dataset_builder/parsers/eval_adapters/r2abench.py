"""
R2ABench Evaluation Adapter -> DiagramEvalSample (for Held-Out Projects)
"""

from pathlib import Path
from typing import List, Set
from architectai_dataset_builder.models.evaluation import DiagramEvalSample, EvalSourceMetadata
from architectai_dataset_builder.utils.hashing import compute_sha256_str, compute_sha256_file
from architectai_dataset_builder.utils.identity import generate_stable_sample_id


class R2ABenchEvalAdapter:
    def __init__(self, held_out_project_ids: Set[str]):
        self.benchmark_id = "r2abench_holdout"
        self.held_out_project_ids = held_out_project_ids

    def parse_directory(self, raw_dir: Path) -> List[DiagramEvalSample]:
        samples = []
        req_files = sorted(raw_dir.glob("*_req.txt")) + sorted(raw_dir.glob("*.txt"))
        for req_file in req_files:
            if not req_file.is_file():
                continue
            project_id = req_file.stem.replace("_req", "")
            if project_id not in self.held_out_project_ids:
                continue  # Skip training/validation split projects

            arch_file = raw_dir / f"{project_id}_arch.puml"
            if not arch_file.exists():
                arch_file = raw_dir / f"{project_id}.puml"

            arch_text = arch_file.read_text(encoding="utf-8") if arch_file.exists() else ""
            req_text = req_file.read_text(encoding="utf-8")
            raw_hash = compute_sha256_file(req_file)

            sample_id = generate_stable_sample_id(
                source_id=self.benchmark_id,
                file_path=req_file.name,
                record_id=project_id,
                project_id=project_id,
                prefix="eval_r2a_",
            )

            norm_hash = compute_sha256_str(f"{req_text}:{arch_text}")

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
                DiagramEvalSample(
                    id=sample_id,
                    source=source_meta,
                    project_id=project_id,
                    requirements_text=req_text,
                    reference_plantuml=arch_text,
                    diagram_type="component",
                )
            )
        return samples
