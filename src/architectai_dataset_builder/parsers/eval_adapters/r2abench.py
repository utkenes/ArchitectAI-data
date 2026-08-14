"""
R2ABench Evaluation Adapter -> DiagramEvalSample (Held-Out Evaluation) with Unified Discovery
"""

from pathlib import Path

from architectai_dataset_builder.models.evaluation import DiagramEvalSample, EvalSourceMetadata
from architectai_dataset_builder.utils.hashing import compute_sha256_file, compute_sha256_str
from architectai_dataset_builder.utils.identity import generate_stable_sample_id
from architectai_dataset_builder.utils.r2abench_discovery import discover_r2abench_projects


class R2ABenchEvalAdapter:
    def __init__(self, held_out_project_ids: set[str]):
        self.benchmark_id = "r2abench"
        self.held_out_project_ids = set(held_out_project_ids)

    def parse_directory(self, raw_dir: Path) -> list[DiagramEvalSample]:
        samples: list[DiagramEvalSample] = []
        discovered_projects = discover_r2abench_projects(raw_dir)

        for pid, proj_files in sorted(discovered_projects.items()):
            if pid not in self.held_out_project_ids:
                continue

            req_file = proj_files.requirements_path
            arch_file = proj_files.architecture_path

            arch_text = arch_file.read_text(encoding="utf-8", errors="ignore") if arch_file and arch_file.exists() else ""
            req_text = req_file.read_text(encoding="utf-8", errors="ignore")
            raw_hash = compute_sha256_file(req_file)

            rel_path = (
                req_file.relative_to(raw_dir).as_posix()
                if req_file.is_relative_to(raw_dir)
                else req_file.name
            )

            sample_id = generate_stable_sample_id(
                source_id=self.benchmark_id,
                file_path=rel_path,
                record_id=pid,
                project_id=pid,
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
                    requirements_text=req_text,
                    reference_plantuml=arch_text,
                    diagram_type="component",
                )
            )
        return samples
