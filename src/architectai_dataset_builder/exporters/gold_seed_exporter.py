"""
Gold Seed Candidate Exporter for Human Review Workflow
"""

from pathlib import Path
from typing import Any

from architectai_dataset_builder.models.canonical import ArchitectAISample
from architectai_dataset_builder.utils.io import write_jsonl


class GoldSeedExporter:
    def __init__(self, export_dir: Path) -> None:
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)

    def export_review_candidates(
        self, samples: list[ArchitectAISample], candidates_per_task: int = 2
    ) -> Path:
        by_task: dict[str, list[ArchitectAISample]] = {}
        for sample in samples:
            t = sample.task_type.value
            if t not in by_task:
                by_task[t] = []
            by_task[t].append(sample)

        candidates = []
        for task_samples in by_task.values():
            for s in task_samples[:candidates_per_task]:
                candidate_entry: dict[str, Any] = {
                    "sample_id": s.id,
                    "group_id": s.source.group_id,
                    "source_id": s.source.source_id,
                    "task_type": s.task_type.value,
                    "kep_status": s.source.kep_status,
                    "scenario": s.scenario,
                    "expected_response": s.final_answer or (s.decisions[0].value if s.decisions else s.scenario),
                    "evidence_summary": {
                        "facts_count": len(s.facts),
                        "decisions_count": len(s.decisions),
                        "tradeoffs_count": len(s.tradeoffs),
                    },
                    "review_status": s.review.status.value,
                    "review_notes": "Exported for Gold Seed manual review gate.",
                }
                candidates.append(candidate_entry)

        output_path = self.export_dir / "gold_review_candidates.jsonl"
        write_jsonl(candidates, output_path)
        return output_path
